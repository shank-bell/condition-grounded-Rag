"""Stage 6 against the baselines of the architecture's evaluation plan - plain RAG, abstention on weak retrieval, a Sufficient-Context
autorater - on questions whose right answer to "should this question get a scope warning?" is known.

  python scripts/eval_scope_baselines.py --silver                                  # the 32 store-derived questions of ablate_stage6.py (SILVER truth)
  python scripts/eval_scope_baselines.py --gold eval/labels/questions_gold.jsonl    # the team's job-B questions (GOLD truth)
  python scripts/eval_scope_baselines.py --silver --limit 4                         # a quick check

Each question runs once WITHOUT Stage 6 (plain RAG: its passages and its weak-evidence flag) and once per Stage 6 configuration; no answer
is generated (the decision does not need one). What "warns" means for each system:
  plain_rag                   never: no Stage 6 and no abstention, so every should-warn question is answered as if the evidence covered it
  abstain_on_weak_retrieval   Stage 5's own flag: no retrieved passage scores above the reranker threshold
  sufficient_context          a local LLM (the answer model) judges that the plain-RAG passages are NOT enough to answer the question as asked
  stage6_30sep                Stage 6 as it was on 30 Sep: per-condition coverage, no cards, no profile-guided retrieval, no joint check
  stage6_joint                + retrieval cards + profile-guided retrieval + joint coverage
  stage6_full                 the shipped configuration (also the E2B -> 12B second opinion)
Truth: silver = derived from the profile store and from facts about the benchmarks (not from people); gold = the labellers' job-B answers.
Metrics: precision / recall / F1 of "warns"; extrapolation rate (should-warn questions answered with no warning); false-warning rate.
Needs Ollama; stop the API first (two pipeline processes do not fit in the GPU).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from ablate_stage6 import CONFIGS, build_cases, settings_for       # scripts/ is on sys.path when this file is run

from cgrag.config import get_settings
from cgrag.evaluation.baselines import score_warnings, sufficient_context
from cgrag.evaluation.freeze import provenance
from cgrag.labelling.sampling import load_chunk_index
from cgrag.pipeline.run import Pipeline
from cgrag.stores.profile_store import ProfileStore

STAGE6 = {"stage6_30sep": "baseline", "stage6_joint": "+joint", "stage6_full": "+escalation"}
SYSTEMS = ["plain_rag", "abstain_on_weak_retrieval", "sufficient_context", *STAGE6]


def load_questions(args) -> list[dict]:
    if args.gold:
        gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return [{"id": q["id"], "family": q["planned_type"], "question": q["question"], "expect_warning": q["expect_warning"],
                 "expected_missing": [m.split("=")[0].strip().lower().replace(" ", "_") for m in q["missing"]]} for q in gold]
    cases = build_cases(ProfileStore(), args.seed, args.covered, args.missing, args.language_task)
    return [{"id": f"S{i:02d}", "family": c["family"], "question": c["question"], "expect_warning": c["expect_warning"],
             "expected_missing": [c["expect_missing"]] if c.get("expect_missing") else []} for i, c in enumerate(cases, start=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, help="the gold question file made by score_labels.py B")
    ap.add_argument("--silver", action="store_true", help="the store-derived questions of ablate_stage6.py")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--covered", type=int, default=12)
    ap.add_argument("--missing", type=int, default=12)
    ap.add_argument("--language-task", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if bool(args.gold) == args.silver:
        sys.exit("give exactly one of --gold FILE or --silver")
    sys.stdout.reconfigure(encoding="utf-8")
    truth = "gold" if args.gold else "silver"
    cfg = get_settings()
    out = args.out or cfg.paths.index_dir / f"scope_baselines_{truth}.json"
    questions = load_questions(args)[: args.limit or None]
    print(f"{len(questions)} {truth} questions, {sum(q['expect_warning'] for q in questions)} should get a warning", flush=True)

    plain_settings = cfg.model_copy(deep=True)
    plain_settings.features.applicability = False                       # plain RAG: Stages 1-5 of the shipped system, no Stage 6
    plain = Pipeline(plain_settings)
    plain.warm_up()
    stage6 = {name: Pipeline(settings_for(CONFIGS[config])) for name, config in STAGE6.items()}
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    rows = []
    for n, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        r = plain.run(q["question"], stop_after="applicability")
        plain_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        sufficient, why = sufficient_context(plain.llm, q["question"], [index.texts[s.chunk_id] for s in r.sources])
        systems = {"plain_rag": {"warned": False, "named_missing": None, "seconds": round(plain_s, 2)},
                   "abstain_on_weak_retrieval": {"warned": bool(r.retrieval_weak), "named_missing": None, "seconds": round(plain_s, 2)},
                   "sufficient_context": {"warned": not sufficient, "named_missing": None, "reason": why, "seconds": round(plain_s + time.perf_counter() - t1, 2)}}
        for name, pipe in stage6.items():
            t2 = time.perf_counter()
            app = pipe.run(q["question"], stop_after="applicability").applicability
            systems[name] = {"warned": bool(app and app.warning), "named_missing": [m.split("=")[0] for m in (app.missing if app else [])],
                             "missing": app.missing if app else [], "seconds": round(time.perf_counter() - t2, 2)}
        rows.append({**q, "systems": systems})
        print(f"{n}/{len(questions)} expect={'WARN' if q['expect_warning'] else 'ok  '} " + " ".join(
            f"{name.split('_')[0][:6]}={'W' if systems[name]['warned'] else '.'}" for name in SYSTEMS) + f"  {q['question'][:60]}", flush=True)

    scores = {name: {**score_warnings([{"expect_warning": r["expect_warning"], "warned": r["systems"][name]["warned"],
                                        "expected_missing": r["expected_missing"], "named_missing": r["systems"][name]["named_missing"]} for r in rows]),
                     "seconds_per_question": round(statistics.mean(r["systems"][name]["seconds"] for r in rows), 2)} for name in SYSTEMS}
    out.write_text(json.dumps({**provenance(), "truth": truth, "questions": len(rows), "scores": scores, "rows": rows}, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    def pct(v) -> str:
        return "  n/a" if v is None else f"{100 * v:4.0f}%"
    print(f"\n{truth} truth, {len(rows)} questions ({sum(r['expect_warning'] for r in rows)} should warn)")
    print(f"{'system':<27}{'warns':>6}{'precision':>10}{'recall':>8}{'F1':>7}{'extrapolation':>14}{'false warns':>12}{'cond. named':>12}{'s/q':>7}")
    for name, s in scores.items():
        named = "-" if s["right_condition_named"] is None else f"{s['right_condition_named']}/{s['tp']}"
        print(f"{name:<27}{s['warnings_given']:>6}{pct(s['precision']):>10}{pct(s['recall']):>8}{pct(s['f1']):>7}{pct(s['extrapolation_rate']):>14}"
              f"{pct(s['false_warning_rate']):>12}{named:>12}{s['seconds_per_question']:>7.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
