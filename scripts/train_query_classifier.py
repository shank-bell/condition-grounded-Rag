"""Train stage 1's SciBERT two-head classifier (intent + complexity) on labelled questions.

  python scripts/train_query_classifier.py --train RUN/data/train.jsonl --val RUN/data/val_clean.jsonl --run-dir RUN
  python scripts/train_query_classifier.py ... --init-from PREVIOUS_RUN/best        # continue from an earlier iteration

Files written into --run-dir (what a monitor reads):
  metrics.jsonl  one JSON line per evaluation ("kind": "eval") and one every 25 steps ("kind": "train")
  best/          weights.pt + meta.json of the best evaluation so far (composite score below)
  status.json    written when the run ends: finished | stopped_by_monitor | max_minutes | crashed
  errors.json    misclassified questions of the best checkpoint (val, dev A, dev B) + per-class scores
A monitor ends the run early by creating the file STOP in --run-dir; the best checkpoint is kept.

Evaluation sets: val (held-out clean questions), dev A (decisions), dev B (never used for decisions; reported at the end).
composite = mean(val intent F1, val complexity F1, dev-A intent accuracy, dev-A complexity accuracy).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cgrag.config import ROOT, get_settings
from cgrag.pipeline.intent_classifier import COMPLEXITIES, INTENTS, MAX_LEN, TwoHeadClassifier, save


def read(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def macro_f1(gold: list[str], pred: list[str], classes: tuple[str, ...]) -> float:
    scores = []
    for c in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


@torch.inference_mode()
def predict(model: TwoHeadClassifier, tok, rows: list[dict], device: str, batch: int = 64) -> list[tuple]:
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        enc = tok([r["question"] for r in part], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            li, lc = model(**enc)
        pi, pc = li.float().softmax(-1), lc.float().softmax(-1)
        for j in range(len(part)):
            out.append((INTENTS[int(pi[j].argmax())], COMPLEXITIES[int(pc[j].argmax())], float(pi[j].max()), float(pc[j].max())))
    return out


def score(rows: list[dict], preds: list[tuple]) -> dict:
    n = max(len(rows), 1)
    gi, gc = [r["intent"] for r in rows], [r["complexity"] for r in rows]
    pi, pc = [p[0] for p in preds], [p[1] for p in preds]
    return {"n": len(rows),
            "intent_acc": round(sum(a == b for a, b in zip(gi, pi)) / n, 4), "intent_f1": round(macro_f1(gi, pi, INTENTS), 4),
            "cx_acc": round(sum(a == b for a, b in zip(gc, pc)) / n, 4), "cx_f1": round(macro_f1(gc, pc, COMPLEXITIES), 4),
            "joint_acc": round(sum(a == b and c == d for a, b, c, d in zip(gi, pi, gc, pc)) / n, 4)}


def per_class(rows: list[dict], preds: list[tuple]) -> dict:
    out = {}
    for head, classes, gold_key, idx in (("intent", INTENTS, "intent", 0), ("complexity", COMPLEXITIES, "complexity", 1)):
        gold, pred = [r[gold_key] for r in rows], [p[idx] for p in preds]
        table = {}
        for c in classes:
            tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
            fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
            fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
            table[c] = {"support": tp + fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}
        confusion: dict[str, int] = {}
        for g, p in zip(gold, pred):
            if g != p:
                confusion[f"{g}->{p}"] = confusion.get(f"{g}->{p}", 0) + 1
        out[head] = {"classes": table, "confusions": dict(sorted(confusion.items(), key=lambda kv: -kv[1]))}
    return out


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--dev", type=Path, default=ROOT / "eval" / "stage1_dev.jsonl")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--iteration", type=int, default=1)
    args = ap.parse_args()

    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    stop_file, metrics_file = run / "STOP", run / "metrics.jsonl"
    if stop_file.exists():
        stop_file.unlink()
    metrics_file.write_text("", encoding="utf-8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = [r for r in read(args.train) if r["intent"] in INTENTS and r["complexity"] in COMPLEXITIES]
    val = read(args.val)
    dev = read(args.dev) if args.dev.exists() else []
    dev_a, dev_b = [r for r in dev if r.get("split") == "A"], [r for r in dev if r.get("split") == "B"]
    print(f"train {len(train)} | val {len(val)} | dev A {len(dev_a)} B {len(dev_b)} | iteration {args.iteration}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg.models.query_classifier
    tok = AutoTokenizer.from_pretrained(name)
    model = TwoHeadClassifier(name)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from / "weights.pt", map_location="cpu"))
        print("initialised from", args.init_from, flush=True)
    model.to(device)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def log(line: dict) -> None:
        line["t"] = round(time.time() - t0, 1)
        with metrics_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    t0 = time.time()
    best = {"composite": -1.0}
    best_preds: dict = {}
    status, step, epochs_run = "finished", 0, 0
    recent: list[float] = []

    def evaluate(epoch: int) -> bool:
        nonlocal best, best_preds
        sets = {"val": val, "dev_a": dev_a, "dev_b": dev_b}
        preds = {k: predict(model, tok, v, device) for k, v in sets.items() if v}
        sc = {k: score(sets[k], p) for k, p in preds.items()}
        parts = [sc["val"]["intent_f1"], sc["val"]["cx_f1"]] + ([sc["dev_a"]["intent_acc"], sc["dev_a"]["cx_acc"]] if "dev_a" in sc else [])
        composite = round(sum(parts) / len(parts), 4)
        line = {"kind": "eval", "iteration": args.iteration, "epoch": epoch, "step": step, "composite": composite,
                "val_intent_f1": sc["val"]["intent_f1"], "val_cx_f1": sc["val"]["cx_f1"], "val_intent_acc": sc["val"]["intent_acc"], "val_cx_acc": sc["val"]["cx_acc"]}
        for k in ("dev_a", "dev_b"):
            if k in sc:
                line.update({f"{k}_intent_acc": sc[k]["intent_acc"], f"{k}_cx_acc": sc[k]["cx_acc"], f"{k}_joint_acc": sc[k]["joint_acc"]})
        improved = composite > best["composite"]
        line["improved"] = improved
        log(line)
        print(json.dumps(line), flush=True)
        if improved:
            best = {**line}
            best_preds = {k: (sets[k], p) for k, p in preds.items()}
            save(model, name, run / "best")
        return improved

    stop_reason = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        for i in range(0, len(order), args.batch):
            if stop_file.exists():
                stop_reason = "stopped_by_monitor"
                break
            if (time.time() - t0) / 60 > args.max_minutes:
                stop_reason = "max_minutes"
                break
            part = [train[j] for j in order[i:i + args.batch]]
            enc = tok([r["question"] for r in part], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            yi = torch.tensor([INTENTS.index(r["intent"]) for r in part], device=device)
            yc = torch.tensor([COMPLEXITIES.index(r["complexity"]) for r in part], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                li, lc = model(**enc)
                loss = loss_fn(li.float(), yi) + loss_fn(lc.float(), yc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            recent.append(float(loss))
            if step % 25 == 0:
                log({"kind": "train", "epoch": epoch, "step": step, "loss": round(sum(recent) / len(recent), 4)})
                recent.clear()
        epochs_run = epoch
        evaluate(epoch)                                   # also after a partial epoch, so the best checkpoint is never stale
        if stop_reason:
            status = stop_reason
            break

    minutes = round((time.time() - t0) / 60, 2)
    (run / "status.json").write_text(json.dumps({"status": status, "epochs_run": epochs_run, "minutes": minutes, "best": best,
                                                 "iteration": args.iteration}, indent=2), encoding="utf-8")
    errors = []
    for k, (rows, preds) in best_preds.items():
        for r, p in zip(rows, preds):
            if r["intent"] != p[0] or r["complexity"] != p[1]:
                errors.append({"set": k, "question": r["question"], "gold_intent": r["intent"], "pred_intent": p[0], "gold_cx": r["complexity"],
                               "pred_cx": p[1], "p_intent": round(p[2], 3), "p_cx": round(p[3], 3), "source": r.get("source", "")})
    (run / "errors.json").write_text(json.dumps({"errors": errors, "per_class": {k: per_class(rows, preds) for k, (rows, preds) in best_preds.items()}},
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {status} after {epochs_run} epochs, {minutes} min; best epoch {best.get('epoch')} composite {best.get('composite')}", flush=True)


if __name__ == "__main__":
    main()
