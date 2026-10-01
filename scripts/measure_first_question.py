"""How slow is the first question, and does warm-up fix it?  (needs Ollama; stop the models first for a cold start:
`ollama stop gemma4:12b`, `ollama stop gemma4:e2b`)

  python scripts/measure_first_question.py --no-warm      # init -> first question -> second question
  python scripts/measure_first_question.py --warm         # init -> warm_up() -> first question -> second question
Prints seconds per phase and writes data/index/first_question_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import time

from cgrag.config import get_settings

QUESTIONS = ["What accuracy does mBERT get on XNLI for Hindi?", "What F1 does BERT-large get on SQuAD v2.0?"]


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--warm", action="store_true")
    g.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()

    out: dict = {"mode": "warm" if args.warm else "cold"}
    t0 = time.perf_counter()
    from cgrag.pipeline.run import Pipeline
    out["import_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    pipe = Pipeline()
    out["init_s"] = round(time.perf_counter() - t0, 1)
    if args.warm:
        t0 = time.perf_counter()
        out["warm_up_models_s"] = pipe.warm_up()
        out["warm_up_s"] = round(time.perf_counter() - t0, 1)
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.perf_counter()
        r = pipe.run(q)
        out[f"question{i}_s"] = round(time.perf_counter() - t0, 1)
        out[f"question{i}_stage_s"] = {k: round(v / 1000, 1) for k, v in r.timings_ms.items() if v > 500}
    print(json.dumps(out, indent=1))
    path = get_settings().paths.index_dir / f"first_question_{out['mode']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
