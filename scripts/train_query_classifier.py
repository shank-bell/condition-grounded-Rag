"""Train stage 1's SciBERT two-head classifier (intent + complexity) on labelled questions.

  python scripts/train_query_classifier.py                       # data/index/query_training.jsonl
  python scripts/train_query_classifier.py --data my.jsonl --epochs 5

Each line of the data file: {"question": "...", "intent": "comparison", "complexity": "complex"}.
Questions are held out 20 % for the accuracy report; the checkpoint is saved under data/index/query_classifier/.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from cgrag.config import get_settings
from cgrag.pipeline.intent_classifier import COMPLEXITIES, INTENTS, MAX_LEN, TwoHeadClassifier, save


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=cfg.paths.index_dir / "query_training.jsonl")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["intent"] in INTENTS and r["complexity"] in COMPLEXITIES]
    random.Random(args.seed).shuffle(rows)
    cut = int(len(rows) * 0.8)
    train, held = rows[:cut], rows[cut:]
    print(f"{len(rows)} questions: {len(train)} train, {len(held)} held out")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg.models.query_classifier
    tok = AutoTokenizer.from_pretrained(name)
    model = TwoHeadClassifier(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    def batches(data, shuffle):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), args.batch):
            chunk = [data[j] for j in idx[i:i + args.batch]]
            enc = tok([r["question"] for r in chunk], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            yield enc, torch.tensor([INTENTS.index(r["intent"]) for r in chunk], device=device), \
                torch.tensor([COMPLEXITIES.index(r["complexity"]) for r in chunk], device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for enc, y_intent, y_cx in batches(train, True):
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                li, lc = model(**enc)
                loss = loss_fn(li.float(), y_intent) + loss_fn(lc.float(), y_cx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        model.eval()
        ok_i = ok_c = 0
        with torch.inference_mode():
            for enc, y_intent, y_cx in batches(held, False):
                li, lc = model(**enc)
                ok_i += int((li.argmax(-1) == y_intent).sum())
                ok_c += int((lc.argmax(-1) == y_cx).sum())
        n = max(len(held), 1)
        print(f"epoch {epoch}: train loss {total:.2f} | held-out intent acc {ok_i / n:.3f}, complexity acc {ok_c / n:.3f}")

    print("saved to", save(model, name))


if __name__ == "__main__":
    main()
