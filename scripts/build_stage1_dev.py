"""Turn eval/stage1_dev_raw.txt (intent|complexity|question per line) into eval/stage1_dev.jsonl with a fixed A / B split.

Split A is used for decisions (stopping, error analysis); split B is never looked at until the final report, so it shows
whether the classifier generalises beyond what was tuned. Labels are the author's judgement, not gold: they follow
the rubric in scripts/make_query_training_data.py (KINDS). The team's own questions replace this set later.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from cgrag.config import ROOT


def main() -> None:
    raw = ROOT / "eval" / "stage1_dev_raw.txt"
    rows = [tuple(line.split("|", 2)) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    rnd = random.Random(2026)
    by: dict[tuple[str, str], list] = defaultdict(list)
    for intent, cx, q in rows:
        by[(intent, cx)].append((intent, cx, q))
    out = []
    for _, items in sorted(by.items()):
        rnd.shuffle(items)
        n_b = max(1, round(len(items) * 0.35))
        for j, (intent, cx, q) in enumerate(items):
            out.append({"question": q, "intent": intent, "complexity": cx, "split": "B" if j < n_b else "A"})
    (ROOT / "eval" / "stage1_dev.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    print(len(out), "questions;", sorted(Counter((r["intent"], r["complexity"]) for r in out).items()))
    print("split:", dict(Counter(r["split"] for r in out)), "| complexity:", dict(Counter(r["complexity"] for r in out)))


if __name__ == "__main__":
    main()
