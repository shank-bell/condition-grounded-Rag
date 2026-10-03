"""Run the system on the team's labelled questions (the gold file made by `score_labels.py B`) and score it.

  python scripts/eval_questions.py eval/labels/questions_gold.jsonl [--full]

For every question the pipeline runs Stages 1-6 (add --full for the whole pipeline incl. the answer) and is compared with the labels:
  Stage 1  intent and complexity (the trained SciBERT) against the labelled ones: accuracy, macro-F1, the misses
  Stage 1  the conditions the question names: recall per field (labelled condition found?) and precision (anything invented?)
  Stage 6  the scope warning: expected YES/NO against the system's: precision, recall, F1 of "warns"; and whether the missing
           condition the labellers named is the one the system names
  by the planned type (covered / one missing / partly covered) and latency.
`--set features.escalation=false` (any section.key=value of config.toml, repeatable) re-runs the same questions under another
configuration, for the ablation: e.g. escalation / joint_coverage / profile_guided_retrieval off, retrieval.use_cards=false.
Writes <gold>.results.jsonl (one line per question, for error analysis) and <gold>.scores.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from cgrag.labelling.score import prf
from cgrag.pipeline.conditions import values_match
from cgrag.pipeline.intent_classifier import COMPLEXITIES, INTENTS


def macro_f1(gold: list[str], pred: list[str], classes) -> float:
    scores = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        if tp + fn:
            f = prf(tp, fp, fn)["f1"]
            scores.append(f or 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=Path)
    ap.add_argument("--full", action="store_true", help="run the whole pipeline (answer, conflicts, critic), not only Stages 1-6")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value for this run")
    ap.add_argument("--tag", default="", help="suffix for the output files (e.g. no_escalation)")
    args = ap.parse_args()
    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    from cgrag.config import get_settings
    from cgrag.pipeline.run import Pipeline
    settings = get_settings().model_copy(deep=True)
    for item in args.set:
        path, value = item.split("=", 1)
        section, key = path.split(".", 1)
        target = getattr(settings, section)
        current = getattr(target, key)
        setattr(target, key, (value.lower() in ("1", "true", "yes", "on")) if isinstance(current, bool) else type(current)(value))
    pipe = Pipeline(settings)
    pipe.warm_up()
    results = []
    for q in gold:
        r = pipe.run(q["question"], stop_after=None if args.full else "applicability")
        a, app = r.analysis, r.applicability
        found = {f: [v] for f, v in a.conditions.specified().items()} if a else {}
        for f, values in (a.conditions.extras() if a else {}).items():        # the further models / datasets / languages
            found.setdefault(f, []).extend(values)
        warned = bool(app and app.warning)
        names = [m.split("=")[0] for m in (app.missing if app else [])]
        expected = [m.split("=")[0].strip().lower().replace(" ", "_") for m in q["missing"]]
        results.append({**q, "pred_intent": a.intent if a else None, "pred_complexity": a.complexity if a else None, "found_conditions": found,
                        "warned": warned, "system_missing": app.missing if app else [], "coverage": app.coverage if app else None,
                        "joint_covered": app.joint_covered if app else None,
                        "missing_named_ok": (not q["expect_warning"]) or bool(set(expected) & set(names)),
                        "seconds": round(r.timings_ms.get("total", 0) / 1000, 1), "answer": r.answer if args.full else None})
    suffix = f".{args.tag}" if args.tag else ""
    out_jsonl = args.gold.with_suffix(f"{suffix}.results.jsonl")
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    def acc(key_gold: str, key_pred: str) -> float:
        return sum(r[key_gold] == r[key_pred] for r in results) / len(results)

    scores: dict = {"questions": len(results),
                    "intent": {"accuracy": round(acc("intent", "pred_intent"), 4),
                               "macro_f1": round(macro_f1([r["intent"] for r in results], [r["pred_intent"] for r in results], INTENTS), 4)},
                    "complexity": {"accuracy": round(acc("complexity", "pred_complexity"), 4),
                                   "macro_f1": round(macro_f1([r["complexity"] for r in results], [r["pred_complexity"] for r in results], COMPLEXITIES), 4)}}
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

    def as_list(v) -> list[str]:
        return v if isinstance(v, list) else [v]

    for r in results:
        for field, wanted in r["conditions"].items():                       # every labelled value: found by Stage 1, or missed
            found_values = r["found_conditions"].get(field, [])
            for w in as_list(wanted):
                if any(values_match(field, w, f) for f in found_values):
                    tp[field] += 1
                else:
                    fn[field] += 1
        for field, values in r["found_conditions"].items():                 # every value Stage 1 found: labelled, or invented
            labelled = as_list(r["conditions"].get(field, []))
            for f in values:
                if not any(values_match(field, w, f) for w in labelled):
                    fp[field] += 1
    scores["conditions"] = {f: prf(tp[f], fp[f], fn[f]) for f in sorted(set(tp) | set(fp) | set(fn))}
    scores["conditions"]["ALL"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    warn_tp = sum(r["warned"] and r["expect_warning"] for r in results)
    warn_fp = sum(r["warned"] and not r["expect_warning"] for r in results)
    warn_fn = sum((not r["warned"]) and r["expect_warning"] for r in results)
    scores["scope_warning"] = {**prf(warn_tp, warn_fp, warn_fn), "right_condition_named": sum(r["missing_named_ok"] for r in results if r["expect_warning"]),
                               "expected_warnings": sum(r["expect_warning"] for r in results),
                               "accuracy": round(sum(r["warned"] == r["expect_warning"] for r in results) / len(results), 4)}
    scores["by_planned_type"] = {t: {"n": len(rs), "warned": sum(r["warned"] for r in rs), "expected_warning": sum(r["expect_warning"] for r in rs),
                                     "warning_correct": sum(r["warned"] == r["expect_warning"] for r in rs)}
                                 for t, rs in ((t, [r for r in results if r["planned_type"] == t]) for t in sorted({r["planned_type"] for r in results}))}
    scores["seconds_per_question"] = {"median": round(statistics.median(r["seconds"] for r in results), 1),
                                      "max": max(r["seconds"] for r in results)}
    args.gold.with_suffix(f"{suffix}.scores.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in scores.items() if k != "conditions"}, indent=1))
    print("conditions:", {f: (v["precision"], v["recall"]) for f, v in scores["conditions"].items()})
    misses = [r for r in results if r["intent"] != r["pred_intent"] or r["complexity"] != r["pred_complexity"] or r["warned"] != r["expect_warning"]]
    print(f"\n{len(misses)} question(s) with a miss (see {out_jsonl.name}):")
    for r in misses[:12]:
        print(f"  {r['id']} {r['question'][:70]!r}: intent {r['intent']}->{r['pred_intent']}, complexity {r['complexity']}->{r['pred_complexity']}, "
              f"warning expected {r['expect_warning']} got {r['warned']}")


if __name__ == "__main__":
    main()
