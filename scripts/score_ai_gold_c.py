"""Score Stage 7 and its frozen baselines against an AI-annotated pair key (ONE labeller: no kappa can be computed).

  python scripts/score_ai_gold_c.py data/labelling/done/C_result_pairs_AIannotated_DONE.xlsx

The labels are read with the same reader as scripts/score_labels.py. Because there is no second labeller, the gold is simply the labeller's verdict;
the output says so. The system's frozen key and the frozen baselines (scripts/run_pair_baselines.py) are used exactly as they were frozen.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cgrag.config import ROOT
from cgrag.evaluation.freeze import provenance
from cgrag.labelling import score as S

PRIVATE = ROOT / "data" / "labelling" / "private"
OUT = ROOT / "eval" / "labels" / "job_C_ai_scores.json"


def pct(v) -> str:
    return "  -  " if v is None else f"{100 * v:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--key", type=Path, default=PRIVATE / "job_C_key.json")
    ap.add_argument("--baselines", type=Path, default=PRIVATE / "job_C_baselines.json")
    args = ap.parse_args()

    labels, problems = S.parse_job_c(args.labels)
    for p in problems:
        print("PROBLEM", p)
    gold = {pid: {"verdict": v["verdict"], "differs": v["differs"], "extraction_error": v["extraction_error"], "settled_by_third": False}
            for pid, v in labels.items()}
    key = json.loads(args.key.read_text(encoding="utf-8"))
    base = json.loads(args.baselines.read_text(encoding="utf-8"))
    print(f"\nlabels: {len(gold)} pairs, ONE AI labeller (no kappa) | gold verdicts: {dict(Counter(g['verdict'] for g in gold.values()))}")
    result = {"gold_verdicts": dict(Counter(g["verdict"] for g in gold.values())), "labeller": "AI (single), not a human; no kappa", "problems": problems}
    systems = {"stage7 (the key)": S.score_system_c(gold, key)}
    for name, preds in base["systems"].items():
        systems[name] = S.score_system_c(gold, preds)
    print(f"\n{'system':<22}{'accuracy':>9}{'macro-F1':>10}   confusion rows = gold, columns = system (EXPLAINED / GENUINE / NOT COMPARABLE)")
    for name, sc in systems.items():
        conf = sc["confusion_gold_rows_system_columns"]
        cells = "  ".join(f"{g[:4]}:{'/'.join(str(conf[g][c]) for c in ('EXPLAINED', 'GENUINE', 'NOT COMPARABLE'))}" for g in conf)
        print(f"{name:<22}{pct(sc['accuracy']):>9}{pct(sc['macro_f1']):>10}   {cells}")
        result.setdefault("systems", {})[name] = sc
    s = systems["stage7 (the key)"]
    print("\nStage 7 per class:", {c: {k: v for k, v in d.items() if k in ("precision", "recall", "f1", "support")} for c, d in s["per_class"].items()})
    print("which condition explains it:", s["condition_attribution"])
    sys_counts = Counter(S.SYSTEM_VERDICT[k["system_verdict"]] for k in key.values() if k)
    print("system verdict counts on these 50 pairs:", dict(sys_counts))
    result["freeze_record"] = provenance()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
