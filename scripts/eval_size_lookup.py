"""Table lookups by model size: does the answer carry the number of the size that the question names? (12 Oct 2026, evening)

  python scripts/eval_size_lookup.py --tag sizes_off [--n 24] [--seed 12103] [--api http://localhost:8000]

Why. GPT-3's appendix Table H.1 holds every benchmark at eight model sizes in three settings (906 results since the evening of 12 Oct). A live probe showed that a question
that names a size ("LAMBADA perplexity of GPT-3 13B, zero-shot") retrieved only main-text chunks and was answered with the 175B figure. This script makes such questions in
bulk from the stored results of that table and asks the running API; an answer counts when it contains the number printed in the cell (as 78.9 or 78.9 %).

The key is the table itself (the stored profile whose number is a cell of the printed table: `tables._scaling_table` reads only clean rows), so it is exact; the questions
are templates, one profile per (data set, metric) so that no row is asked twice, sizes other than 175B are drawn six times as often (175B also appears in the main text). It
is a check of retrieval and writing for table lookups, not of the extraction. The API must be up (`uvicorn cgrag.api.main:app --port 8000`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.stores.profile_store import ProfileStore

CHUNKS = ("2005.14165:0142", "2005.14165:0143", "2005.14165:0144")
PERCENT = {"acc": "accuracy", "em": "exact match accuracy", "f1": "F1"}
SETTING = {"zero-shot": "zero-shot", "one-shot": "one-shot", "few-shot": "few-shot"}


def build(n: int, seed: int) -> list[dict]:
    store = ProfileStore(repaired=True)
    pool = [p for p in store.all() if p.chunk_id in CHUNKS and p.model and p.model.startswith("GPT-3 ") and p.setting in SETTING and p.value is not None
            and p.dataset and p.metric and (p.metric.lower() in PERCENT or p.metric.lower() in ("ppl", "bleu"))]
    rng = random.Random(seed)
    by_key: dict[tuple, list] = {}
    for p in pool:
        by_key.setdefault((p.dataset, p.metric.lower()), []).append(p)
    keys = sorted(by_key)
    rng.shuffle(keys)
    out: list[dict] = []
    for k in keys:
        cand = by_key[k]
        small = [p for p in cand if p.model_size and p.model_size != "175B"]
        pick = rng.choice(small if small and rng.random() < 6 / 7 else cand)
        size = pick.model.replace("GPT-3 ", "")
        metric = pick.metric.lower()
        what = {"ppl": "perplexity", "bleu": "BLEU"}.get(metric) or PERCENT[metric]
        ds = pick.dataset
        text = (f"What is the {SETTING[pick.setting]} {what} of GPT-3 {size} on {ds}?" if metric not in ("bleu",)
                else f"What BLEU does GPT-3 {size} reach on {ds} in the {SETTING[pick.setting]} setting?")
        out.append({"question": text, "value": float(pick.value), "profile_id": pick.profile_id, "dataset": ds, "metric": metric, "size": size, "setting": pick.setting})
        if len(out) >= n:
            break
    return out


def contains(answer: str, value: float) -> bool:
    forms = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d])", answer) for f in forms)


def ask(api: str, q: str) -> dict:
    req = urllib.request.Request(api.rstrip("/") + "/query", data=json.dumps({"question": q}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12103)
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()
    qs = build(args.n, args.seed)
    rows = []
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        r = ask(args.api, q["question"])
        ans = r.get("answer") or ""
        ids = [s.get("chunk_id") for s in (r.get("sources") or [])]
        row = {**q, "answer": ans[:400], "sources": ids, "hit": contains(ans, q["value"]), "table_in_sources": any(c in ids for c in CHUNKS)}
        rows.append(row)
        print(f"[{i}/{len(qs)}] {'HIT ' if row['hit'] else 'miss'} table={'Y' if row['table_in_sources'] else 'n'}  {q['question']}  -> {q['value']:g}", flush=True)
    hit = sum(r["hit"] for r in rows)
    tab = sum(r["table_in_sources"] for r in rows)
    summary = {"questions": len(rows), "answer_has_the_number": hit, "table_among_sources": tab, "seconds": round(time.time() - t0)}
    print(json.dumps(summary))
    out = ROOT / "eval" / "labels" / f"size_lookup.{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
