"""Install the best checkpoint of a training run as Stage 1's classifier (data/index/query_classifier/).

  python scripts/install_query_classifier.py data/index/query_runs/iter2

QueryUnderstanding loads that folder automatically when it exists. The run's status.json and errors.json are copied next to
the weights so the numbers that justified the install travel with them.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from cgrag.pipeline.intent_classifier import checkpoint_dir


def main() -> None:
    run = Path(sys.argv[1])
    best = run / "best"
    if not (best / "weights.pt").exists():
        raise SystemExit(f"no best checkpoint in {run}")
    target = checkpoint_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name in ("weights.pt", "meta.json"):
        shutil.copy2(best / name, target / name)
    for name in ("status.json", "errors.json"):
        if (run / name).exists():
            shutil.copy2(run / name, target / name)
    print("installed", best, "->", target)


if __name__ == "__main__":
    main()
