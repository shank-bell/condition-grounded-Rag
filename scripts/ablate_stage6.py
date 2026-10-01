"""Does each Stage 4/6 change earn its cost? Runs scope questions through Stages 1-6 under four configurations and compares the
scope-warning decisions and the latency.

  python scripts/ablate_stage6.py --out data/index/ablate_stage6.json

Configurations (each adds one thing to the previous):
  baseline    no retrieval cards, no profile-guided retrieval, no joint check, no escalation   (the pipeline before 2026-10-01)
  +cards      retrieval cards in dense search, keyword index and reranker
  +guided     + Stage 6 asks the profile store for chunks that record a missing condition
  +joint      + Stage 6 requires the named model, dataset and language to be recorded TOGETHER by one result
  +escalation + a "not covered" verdict of the small model gets a second opinion from the bigger one   (the full pipeline)

Question families (truth comes from the profile store and from facts about the benchmarks, NOT from human labels - labelling job B
is the human version of this test):
  covered   model + dataset + language triples that have a recorded result            -> no scope warning expected
  missing   the same models and datasets with a language the benchmark does not have  -> a warning naming the language expected
            (XNLI has 15 languages, XQuAD 11, MLQA 7: none has Kannada, Tamil, Marathi or Malayalam)
  language_task  a language and a task ("How well do models perform on Kannada NLI?"), a result exists   -> no warning expected
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from cgrag.config import get_settings
from cgrag.pipeline.conditions import _LANG_ALIASES, model_family, values_match
from cgrag.pipeline.run import Pipeline
from cgrag.stores.profile_store import ProfileStore

MODELS = {"mBERT": "mbert", "XLM-R": "xlmr", "mT5": "mt5"}
DATASETS = ["XNLI", "XQuAD", "MLQA", "TyDi QA GoldP"]
ABSENT_LANGUAGES = ["Kannada", "Tamil", "Marathi", "Malayalam"]
TASK_NAME = {"natural language inference": "NLI", "question answering": "QA", "named entity recognition": "NER"}
CONFIGS = {
    "baseline": dict(cards=False, guided=False, joint=False, escalation=False),
    "+cards": dict(cards=True, guided=False, joint=False, escalation=False),
    "+guided": dict(cards=True, guided=True, joint=False, escalation=False),
    "+joint": dict(cards=True, guided=True, joint=True, escalation=False),
    "+escalation": dict(cards=True, guided=True, joint=True, escalation=True),
}


def language_name(code_or_name: str) -> str:
    return _LANG_ALIASES.get(code_or_name.lower(), code_or_name).capitalize()


def build_cases(profiles: ProfileStore, seed: int, n_covered: int, n_missing: int, n_lang_task: int) -> list[dict]:
    rnd = random.Random(seed)
    allp = profiles.all()
    triples: dict[tuple[str, str, str], int] = defaultdict(int)
    for p in allp:
        fam = model_family(p.model)
        for name, key in MODELS.items():
            if fam == key and p.language and p.dataset:
                for ds in DATASETS:
                    if values_match("dataset", ds, p.dataset):
                        lang = language_name(p.language)
                        if lang.isalpha() and len(lang) > 3:
                            triples[(name, ds, lang)] += 1
    covered_pool = sorted(triples)
    rnd.shuffle(covered_pool)
    cases = [{"family": "covered", "question": f"What accuracy does {m} get on {d} for {l}?",
              "conditions": {"model": m, "dataset": d, "language": l}, "expect_warning": False} for m, d, l in covered_pool[:n_covered]]

    missing_pool = [(m, d, l) for m in MODELS for d in ("XNLI", "XQuAD", "MLQA") for l in ABSENT_LANGUAGES]
    rnd.shuffle(missing_pool)
    cases += [{"family": "missing", "question": f"What accuracy does {m} get on {d} for {l}?",
               "conditions": {"model": m, "dataset": d, "language": l}, "expect_warning": True, "expect_missing": "language"}
              for m, d, l in missing_pool[:n_missing]]

    pairs: Counter = Counter()
    for p in allp:
        if p.task and p.task.lower() in TASK_NAME and p.language:
            lang = language_name(p.language)
            if lang in {"Kannada", "Tamil", "Telugu", "Hindi", "Marathi", "Bengali", "Malayalam", "Gujarati"}:
                pairs[(lang, p.task.lower())] += 1
    chosen = [k for k, n in pairs.items() if n >= 3]
    rnd.shuffle(chosen)
    if ("Kannada", "natural language inference") in pairs and ("Kannada", "natural language inference") not in chosen[:n_lang_task]:
        chosen.insert(0, ("Kannada", "natural language inference"))                 # the case that started this
    cases += [{"family": "language_task", "question": f"How well do models perform on {l} {TASK_NAME[t]}?",
               "conditions": {"language": l, "task": t}, "expect_warning": False} for l, t in chosen[:n_lang_task]]
    return cases


def settings_for(cfg: dict):
    s = get_settings().model_copy(deep=True)
    s.retrieval.use_cards = cfg["cards"]
    s.features.profile_guided_retrieval = cfg["guided"]
    s.features.joint_coverage = cfg["joint"]
    s.features.escalation = cfg["escalation"]
    return s


def judge(case: dict, applicability) -> dict:
    warned = bool(applicability and applicability.warning)
    named = [m.split("=")[0] for m in (applicability.missing if applicability else [])]
    ok = (warned == case["expect_warning"]) and (not case.get("expect_missing") or case["expect_missing"] in named)
    return {"warned": warned, "missing": applicability.missing if applicability else [], "correct": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--covered", type=int, default=12)
    ap.add_argument("--missing", type=int, default=12)
    ap.add_argument("--language-task", type=int, default=8)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", type=Path, default=get_settings().paths.index_dir / "ablate_stage6.json")
    args = ap.parse_args()

    cases = build_cases(ProfileStore(), args.seed, args.covered, args.missing, args.language_task)
    print(f"{len(cases)} questions: {dict(Counter(c['family'] for c in cases))}", flush=True)
    results: dict[str, list[dict]] = {}
    warmed = False
    for name in args.configs.split(","):
        pipe = Pipeline(settings_for(CONFIGS[name]))
        if not warmed:
            pipe.warm_up()
            warmed = True
        rows = []
        for case in cases:
            r = pipe.run(case["question"], stop_after="applicability")
            a = r.applicability
            rows.append({**case, **judge(case, a), "coverage": a.coverage if a else None, "guided": bool(a and a.profile_guided),
                         "escalated": bool(a and a.escalated), "re_retrieved": bool(a and a.re_retrieved),
                         "conditions_found": r.analysis.conditions.specified() if r.analysis else {},
                         "seconds": round(r.timings_ms.get("total", 0) / 1000, 2),
                         "applicability_s": round(r.timings_ms.get("6_applicability", 0) / 1000, 2)})
        results[name] = rows
        print(f"{name}: done ({sum(r['correct'] for r in rows)}/{len(rows)} correct)", flush=True)
        args.out.write_text(json.dumps({"configs": CONFIGS, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\nconfig        covered: wrongly warned | missing: warned (correctly) | language_task: wrongly warned | all correct | s/question | stage-6 s | escalated | guided")
    for name, rows in results.items():
        def part(fam: str) -> list[dict]:
            return [r for r in rows if r["family"] == fam]
        cov, mis, lt = part("covered"), part("missing"), part("language_task")
        print(f"{name:<12} {sum(r['warned'] for r in cov):>3}/{len(cov):<3}                 {sum(r['correct'] for r in mis):>3}/{len(mis):<3}                      "
              f"{sum(r['warned'] for r in lt):>3}/{len(lt):<3}                   {sum(r['correct'] for r in rows):>3}/{len(rows):<3}      "
              f"{statistics.mean(r['seconds'] for r in rows):6.1f}   {statistics.mean(r['applicability_s'] for r in rows):6.2f}     "
              f"{sum(r['escalated'] for r in rows):>3}      {sum(r['guided'] for r in rows):>3}")


if __name__ == "__main__":
    main()
