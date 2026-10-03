"""Answer quality (claim 4 of the evaluation plan, scaled to our own corpus; see src/cgrag/evaluation/answers.py for the departure it is).

  python scripts/eval_answers.py --silver                                    # questions built from the profile store (SILVER answers)
  python scripts/eval_answers.py --verified-from data/labelling/done/A_*_DONE.xlsx    # questions built from rows the labellers judged ALL_OK (GOLD answers)
  python scripts/eval_answers.py --silver --limit 4                          # a quick check

The same questions go through three systems and an answer counts as correct when it states a recorded value (up to rounding):
  full           the shipped pipeline, every stage
  no_stage_c67   the pipeline without stages C, 6 and 7: no Condition Profile is used online (no cards, no recorded results in the prompt, no
                 critic support from profiles, no Stage 1 vocabulary), no applicability check, no conflict check
  llm_only       the answer model alone, from memory, no retrieval (the document's "LLM prompt" baseline)
Also reported: how many numbers an answer states (a longer answer has more chances to hit), the share of claims the critic supported, how many
questions got a scope warning (all questions are answerable, so a warning is a false one), seconds per question, and a paired sign test.
Needs Ollama; stop the API first (two pipeline processes do not fit in the GPU).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cgrag.config import Settings, get_settings
from cgrag.evaluation.answers import answer_numbers, gold_b_cases, is_correct, lookup_cases, paired, score_system, value_recall
from cgrag.evaluation.freeze import provenance
from cgrag.labelling.score import parse_job_a
from cgrag.pipeline.run import Pipeline
from cgrag.stores.profile_store import ProfileStore

LLM_ONLY_SYSTEM = ("You answer questions about published research papers on language models from memory, with no documents. "
                   "If you know the number, state it; if you are not sure, say so briefly. Answer in one or two sentences.")
SYSTEMS = ["full", "no_stage_c67", "llm_only"]


def without_c67(cfg: Settings) -> Settings:
    s = cfg.model_copy(deep=True)
    f = s.features
    f.use_profiles = False
    f.applicability = False
    f.contradiction = False
    f.profile_in_context = False
    f.profile_guided_retrieval = False
    f.joint_coverage = False
    f.escalation = False
    s.retrieval.use_cards = False
    return s


def score_row(case: dict, answer: str, resp, seconds: float) -> dict:
    checks = resp.claim_checks if resp else []
    app = resp.applicability if resp else None
    return {"answer": answer, "correct": is_correct(answer, case), "recall": round(value_recall(answer, case["gold_values"]), 3),
            "numbers": len(answer_numbers(answer)),
            "supported": (sum(c.supported for c in checks) / len(checks)) if checks else None,
            "warned": None if app is None else bool(app.warning), "warning": None if app is None else app.warning,
            "missing": [] if app is None else app.missing, "seconds": round(seconds, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", action="store_true", help="questions built from the profile store")
    ap.add_argument("--verified-from", nargs="+", type=Path, help="filled-in job A workbooks: only rows judged ALL_OK are used (gold answers)")
    ap.add_argument("--gold-b", type=Path, help="the gold question file of job B: the questions the labellers wrote and marked answerable, with their expected facts "
                                                  "(the independent test: nobody built these from the profile store)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if sum([bool(args.verified_from), args.silver, bool(args.gold_b)]) != 1:
        sys.exit("give exactly one of --silver, --verified-from FILES or --gold-b FILE")
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = get_settings()
    verified = None
    if args.verified_from:
        verified = {r["profile_id"] for path in args.verified_from for r in parse_job_a(path)[0] if r["verdict"] == "ALL_OK"}
        if not verified:
            sys.exit("no ALL_OK row in those files")
        print(f"{len(verified)} profiles judged ALL_OK by the labellers")
    truth = "gold_b" if args.gold_b else "gold" if verified is not None else "silver"
    out = args.out or cfg.paths.index_dir / f"answer_quality_{truth}.json"
    if args.gold_b:
        cases = gold_b_cases([json.loads(line) for line in args.gold_b.read_text(encoding="utf-8-sig").splitlines() if line.strip()])[: args.limit or None]
    else:
        cases = lookup_cases(ProfileStore(cfg.paths.profile_db).all(), args.n, args.seed, verified)[: args.limit or None]
    print(f"{len(cases)} {truth} questions, e.g. {cases[0]['question']!r} (gold {cases[0]['gold_values']})" if cases else "no question could be built")
    if not cases:
        return

    full, baseline = Pipeline(cfg), Pipeline(without_c67(cfg))
    full.warm_up()
    rows = []
    for n, case in enumerate(cases, start=1):
        systems = {}
        for name, pipe in (("full", full), ("no_stage_c67", baseline)):
            t0 = time.perf_counter()
            resp = pipe.run(case["question"])
            systems[name] = score_row(case, resp.answer, resp, time.perf_counter() - t0)
        t0 = time.perf_counter()
        text = full.llm.chat([{"role": "system", "content": LLM_ONLY_SYSTEM}, {"role": "user", "content": case["question"]}],
                             temperature=0.0, max_tokens=150).text
        systems["llm_only"] = score_row(case, text, None, time.perf_counter() - t0)
        rows.append({**case, "systems": systems})
        print(f"{n}/{len(cases)} " + " ".join(f"{name}={'Y' if systems[name]['correct'] else '.'}" for name in SYSTEMS) + f"  {case['question']}", flush=True)

    scores = {name: score_system([r["systems"][name] for r in rows]) for name in SYSTEMS}
    flags = {name: [r["systems"][name]["correct"] for r in rows] for name in SYSTEMS}
    comparisons = {"full_vs_no_stage_c67": paired(flags["full"], flags["no_stage_c67"]), "full_vs_llm_only": paired(flags["full"], flags["llm_only"]),
                   "no_stage_c67_vs_llm_only": paired(flags["no_stage_c67"], flags["llm_only"])}
    out.write_text(json.dumps({**provenance(), "truth": truth, "questions": len(rows), "scores": scores, "paired": comparisons, "rows": rows},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{truth} answers, {len(rows)} questions")
    print(f"{'system':<14}{'correct':>8}{'accuracy':>10}{'numbers/answer':>16}{'claims supported':>18}{'false warnings':>16}{'s/q':>7}")
    for name, s in scores.items():
        sup = "-" if s["claims_supported"] is None else f"{100 * s['claims_supported']:.0f}%"
        warned = "-" if s["warned"] is None else str(s["warned"])
        print(f"{name:<14}{s['correct']:>8}{100 * s['accuracy']:>9.0f}%{s['numbers_per_answer']:>16}{sup:>18}{warned:>16}{s['seconds_per_question']:>7.1f}")
    for name, c in comparisons.items():
        a, b = name.split("_vs_")
        print(f"{name}: both right {c['both_right']}, only {a} {c['only_a']}, only {b} {c['only_b']}, both wrong {c['both_wrong']} (sign test p = {c['sign_test_p']})")
    print("wrote", out)


if __name__ == "__main__":
    main()
