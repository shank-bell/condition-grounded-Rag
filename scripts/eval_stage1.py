"""Score a trained Stage 1 checkpoint on the dev sets A, B and C (C is the clean held-out set: look at it only at the end).

  python scripts/eval_stage1.py data/index/query_runs/iter2/best --out data/index/query_runs/final_eval.json

With --perturb N every dev question is also scored in N typo / lower-case / chatty-prefix variants (robustness probe).
Prints accuracy / macro-F1 per split, per-class recall for intent, and the errors of the chosen splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_query_classifier import per_class, predict, read, score      # noqa: E402

from cgrag.config import ROOT                                           # noqa: E402
from cgrag.pipeline.intent_classifier import TwoHeadClassifier          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="a directory with weights.pt + meta.json (best/ of a run, or the installed one)")
    ap.add_argument("--dev", type=Path, default=ROOT / "eval" / "stage1_dev.jsonl")
    ap.add_argument("--splits", default="A,B,C")
    ap.add_argument("--errors-for", default="", help="comma list of splits whose errors are printed (default none)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--perturb", type=int, default=0,
                    help="score each question N extra times typed the way a real user types (typo / lower case / chatty prefix)")
    args = ap.parse_args()

    meta = json.loads((args.checkpoint / "meta.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(meta["encoder"])
    model = TwoHeadClassifier(meta["encoder"])
    model.load_state_dict(torch.load(args.checkpoint / "weights.pt", map_location="cpu"))
    model.to(device)

    dev = read(args.dev)
    if args.perturb:
        from prepare_query_data import perturb                          # noqa: E402  (same perturbations as the training augmentation)
        import random
        rnd = random.Random(99)
        dev = [{**r, "question": perturb(r["question"], rnd)} for _ in range(args.perturb) for r in dev]
    report, show = {}, set(filter(None, args.errors_for.split(",")))
    for split in [s for s in args.splits.split(",") if s]:
        rows = [r for r in dev if r.get("split") == split]
        if not rows:
            continue
        preds = predict(model, tok, rows, device)
        report[split] = {**score(rows, preds), "per_class": per_class(rows, preds)}
        s = report[split]
        print(f"split {split}: n={s['n']}  intent acc {s['intent_acc']:.3f} F1 {s['intent_f1']:.3f} | "
              f"complexity acc {s['cx_acc']:.3f} F1 {s['cx_f1']:.3f} | both right {s['joint_acc']:.3f}")
        if split in show:
            for r, p in zip(rows, preds):
                if r["intent"] != p[0] or r["complexity"] != p[1]:
                    print(f"   x {r['question'][:100]} | gold {r['intent']}/{r['complexity']} | pred {p[0]}/{p[1]}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
