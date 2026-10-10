"""Re-score the stored answers of the answer-quality test (eval/labels/answer_quality_gold_b.final.json) with a CORRECTED key. No pipeline run.

  python scripts/rescore_answers_gold_b.py [SRC.json OUT.json]       # default: the 8 Oct final run -> answer_quality_gold_b.corrected_key.json

Why: the scorer takes every decimal number of the key's free-text 'facts' as the expected answer and calls an answer correct when it states at least
half of them. For three questions the facts text lists a whole table row while the question asks for ONE cell, so a right answer scored wrong:
  B-AI-03  "LLaMA 65B ... 5-shot"            facts list the 0-, 1-, 5- and 64-shot values      -> expected: 35.0
  B-AI-04  "GPT-3 ... zero-shot"             facts list zero-, one- and few-shot BLEU          -> expected: 27.2
  B-AI-17  "DistilBERT's F1 on SQuAD 2.0"    facts list EM and F1 for DistilBERT-6L and -4L    -> expected: an F1 value (64.1 or 69.5)
The corrections follow from the question text, not from any answer, and are applied to ALL systems alike. They were made AFTER the first scoring
(error analysis), so the corrected numbers are reported next to the original ones and labelled "key corrected after error analysis, not held-out".
Only 9 questions: the confidence intervals are wide.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from cgrag.evaluation.answers import is_correct, paired

SRC = Path(sys.argv[1]) if len(sys.argv) > 2 else Path("eval/labels/answer_quality_gold_b.final.json")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eval/labels/answer_quality_gold_b.corrected_key.json")
CORRECTED = {"B-AI-03": ([35.0], "any"), "B-AI-04": ([27.2], "any"), "B-AI-17": ([64.1, 69.5], "any")}
SYSTEMS = ("full", "no_stage_c67", "llm_only")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    original = {s: [] for s in SYSTEMS}
    corrected = {s: [] for s in SYSTEMS}
    for row in src["rows"]:
        gold = row["gold_values"] if isinstance(row["gold_values"], list) else json.loads(row["gold_values"])
        case = {"gold_values": gold, "mode": row["mode"]}
        fixed = {"gold_values": CORRECTED[row["id"]][0], "mode": CORRECTED[row["id"]][1]} if row["id"] in CORRECTED else case
        for s in SYSTEMS:
            answer = row["systems"][s].get("answer") or ""
            original[s].append(is_correct(answer, case))
            corrected[s].append(is_correct(answer, fixed))
    n = len(src["rows"])
    out = {"source": str(SRC), "frozen_at_commit": src.get("frozen_at_commit"), "questions": n,
           "note": "key corrected AFTER the first scoring (error analysis): three questions asked for one cell but the key listed the whole row; same correction for all systems; not held-out; n = 9",
           "corrections": {k: {"expected_values": v[0], "mode": v[1]} for k, v in CORRECTED.items()},
           "original": {s: {"correct": sum(v), "of": n} for s, v in original.items()},
           "corrected": {s: {"correct": sum(v), "of": n, "accuracy": round(sum(v) / n, 4), "ci95": wilson(sum(v), n)} for s, v in corrected.items()},
           "paired_corrected": {"full_vs_no_stage_c67": paired(corrected["full"], corrected["no_stage_c67"]),
                                "full_vs_llm_only": paired(corrected["full"], corrected["llm_only"])}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for s in SYSTEMS:
        print(f"{s:14} original {sum(original[s])}/{n} -> corrected {sum(corrected[s])}/{n}  (95% CI {out['corrected'][s]['ci95']})")
    print("full vs LLM alone:", out["paired_corrected"]["full_vs_llm_only"])
    print("full vs without stages C/6/7:", out["paired_corrected"]["full_vs_no_stage_c67"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
