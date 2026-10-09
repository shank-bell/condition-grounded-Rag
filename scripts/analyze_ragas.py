"""Analysis of the RAGAS-style faithfulness runs (10 Oct 2026): three answer sets x two judges, plus a judge-free check of the numbers.

  python scripts/analyze_ragas.py

Reads the per-question files that `scripts/eval_ragas.py` leaves in data/labelling/private/ragas/ (git-ignored) and writes the aggregates to
eval/labels/ragas_analysis.json (no answers, no per-question rows). The three answer sets are:
  loop_off    the shipped configuration: no rewrite loop
  loop_same   the rewrite loop on, the answer model (Gemma 4 12B) is its own judge
  loop_llama  the rewrite loop on, an independent judge (Llama 3.1 8B, Meta) is resident
Each set is scored by two judges: Llama 3.1 8B (independent of the writer) and Gemma 4 12B (the writer's own family). A set must never be read
as evidence for itself: the loop optimises the score of ITS judge, so "loop_llama scored by Llama" and "loop_same scored by Gemma" are circular;
"loop_llama scored by Gemma" and "loop_same scored by Llama" are cross-judge checks. The last block counts the correct numbers in the 9 answerable
gold questions with the corrected key (no language model involved), to see that the loop does not remove right numbers.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from cgrag.evaluation.answers import is_correct, paired, sign_test

STORE = Path("data/labelling/private/ragas")
OUT = Path("eval/labels/ragas_analysis.json")
GOLD_NUMBERS = Path("eval/labels/answer_quality_gold_b.final.json")
CORRECTED = {"B-AI-03": ([35.0], "any"), "B-AI-04": ([27.2], "any"), "B-AI-17": ([64.1, 69.5], "any")}       # same corrections as rescore_answers_gold_b.py
TAGS = ["loop_off", "loop_same", "loop_llama"]
JUDGES = {"llama3.1:8b": "Llama 3.1 8B (Meta; not the writer's family)", "gemma4:12b": "Gemma 4 12B (the writer's own family)"}
KIND = {("llama3.1:8b", "loop_llama"): "circular: the loop used this judge", ("gemma4:12b", "loop_same"): "circular: the loop used this judge",
        ("llama3.1:8b", "loop_same"): "cross-judge: the loop used Gemma", ("gemma4:12b", "loop_llama"): "cross-judge: the loop used Llama"}


def _safe(judge: str) -> str:
    return judge.replace(":", "_").replace("/", "_")


def _bootstrap(diffs: list[float], rounds: int = 4000, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(rounds))
    return [round(means[int(0.025 * rounds)], 3), round(means[int(0.975 * rounds)], 3)]


def _rewrites(text: str) -> int:
    """The loop's trace line reads '9 faithfulness 0.83 after 2 rewrite(s) (judge ...)'; no 'after' means no rewrite."""
    if "after" in text and "rewrite" in text:
        try:
            return int(text.split("after")[1].split("rewrite")[0].strip())
        except ValueError:
            return 1
    return 0


def main() -> None:
    rows = {(j, t): {r["id"]: r for r in json.loads((STORE / f"{t}.{_safe(j)}.json").read_text(encoding="utf-8"))["rows"]} for j in JUDGES for t in TAGS}
    out: dict = {"note": "RAGAS-style metrics re-implemented in src/cgrag/pipeline/faithfulness.py (the ragas package is not installed), not comparable with published RAGAS numbers. "
                         "30 gold questions, one run per set. Aggregates only.",
                 "judges": {}, "cost": {}, "gold_numbers": {}}
    for j, name in JUDGES.items():
        block: dict = {"judge": name, "tags": {}}
        base = {i: r["faithfulness"] for i, r in rows[(j, "loop_off")].items() if r["faithfulness"] is not None}
        for t in TAGS:
            rs = rows[(j, t)]
            scored = {i: r for i, r in rs.items() if r["faithfulness"] is not None}
            vals = [r["faithfulness"] for r in scored.values()]
            rel = [r["answer_relevancy"] for r in rs.values() if r["answer_relevancy"] is not None]
            by_type = {}
            for kind in sorted({r["planned_type"] for r in scored.values() if r["planned_type"]}):
                by_type[kind] = round(statistics.mean(r["faithfulness"] for r in scored.values() if r["planned_type"] == kind), 3)
            entry = {"questions": len(rs), "scored": len(vals), "faithfulness_mean": round(statistics.mean(vals), 3), "faithfulness_median": round(statistics.median(vals), 3),
                     "share_at_or_above_0.80": round(sum(v >= 0.8 for v in vals) / len(vals), 3), "answers_at_or_above_0.80": sum(v >= 0.8 for v in vals),
                     "answers_fully_faithful": sum(v == 1.0 for v in vals), "statements_per_answer": round(statistics.mean(r["statements"] for r in scored.values()), 2),
                     "unsupported_per_answer": round(statistics.mean(r["unsupported"] for r in scored.values()), 2), "by_type": by_type,
                     "answer_relevancy_mean": round(statistics.mean(rel), 3), "answer_relevancy_zero_answers": sum(v == 0 for v in rel)}
            if t != "loop_off":
                shared = [i for i in scored if i in base]
                diffs = [scored[i]["faithfulness"] - base[i] for i in shared]
                up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
                entry["against_loop_off"] = {"paired_questions": len(shared), "higher": up, "lower": down, "equal": len(shared) - up - down,
                                             "sign_test_p": round(sign_test(up, down), 4), "mean_difference": round(statistics.mean(diffs), 3),
                                             "bootstrap_ci95": _bootstrap(diffs), "kind": KIND[(j, t)]}
            block["tags"][t] = entry
        out["judges"][j] = block
    for t in TAGS:
        recs = json.loads((STORE / f"{t}.json").read_text(encoding="utf-8"))["records"]
        secs = [r["seconds"] for r in recs]
        rw = [_rewrites(r["rewrites"] or "") for r in recs]
        out["cost"][t] = {"seconds_median": round(statistics.median(secs), 1), "seconds_p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 1),
                          "seconds_mean": round(statistics.mean(secs), 1), "answers_rewritten": sum(k > 0 for k in rw), "rewrites_in_all": sum(rw), "questions": len(recs)}
    gold = json.loads(GOLD_NUMBERS.read_text(encoding="utf-8"))
    cases = {}
    for row in gold["rows"]:
        values = row["gold_values"] if isinstance(row["gold_values"], list) else json.loads(row["gold_values"])
        cases[row["id"]] = ({"gold_values": CORRECTED[row["id"]][0], "mode": CORRECTED[row["id"]][1]} if row["id"] in CORRECTED else {"gold_values": values, "mode": row["mode"]})
    right: dict[str, list[bool]] = {}
    for t in TAGS:
        recs = {r["id"]: r for r in json.loads((STORE / f"{t}.json").read_text(encoding="utf-8"))["records"]}
        right[t] = [is_correct(recs[i]["answer"], cases[i]) for i in cases]
        out["gold_numbers"][t] = {"correct": sum(right[t]), "of": len(cases)}
    for t in TAGS[1:]:
        out["gold_numbers"][t]["against_loop_off"] = paired(right[t], right["loop_off"])
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    for j, block in out["judges"].items():
        print(f"\njudge {block['judge']}")
        for t, e in block["tags"].items():
            line = (f"  {t:11s} n={e['scored']:2d} mean {e['faithfulness_mean']:.3f}  >=0.80 {e['answers_at_or_above_0.80']:2d}  fully {e['answers_fully_faithful']:2d}  "
                    f"stmts {e['statements_per_answer']:.1f} unsupp {e['unsupported_per_answer']:.1f}  relevancy {e['answer_relevancy_mean']:.3f} (zero {e['answer_relevancy_zero_answers']})")
            print(line)
            a = e.get("against_loop_off")
            if a:
                print(f"{'':14s}vs loop_off: +{a['higher']} -{a['lower']} ={a['equal']}  p={a['sign_test_p']}  diff {a['mean_difference']:+.3f} CI {a['bootstrap_ci95']}  [{a['kind']}]")
    print("\ncost", json.dumps(out["cost"]))
    print("gold numbers", json.dumps(out["gold_numbers"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
