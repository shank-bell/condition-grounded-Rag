"""Print a random sample of extracted profiles next to the table line they came from, for a hand check of the fields.

  python scripts/audit_sample.py                     # 3 profiles per paper, seed 7
  python scripts/audit_sample.py --per-paper 5 --seed 1 1911.02116 1810.04805

For each profile check: model = the row's system? dataset / metric / language / setting = what the caption and column say?
Do not run while the API or an ingest is using the index (ChromaDB is single-process).
"""
from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys

from cgrag.config import get_settings
from cgrag.stores.vector_store import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("papers", nargs="*")
    ap.add_argument("--per-paper", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(args.seed)
    db = sqlite3.connect(f"file:{get_settings().paths.profile_db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    chunks = {c.chunk_id: c for c in VectorStore().all_chunks()}
    papers = args.papers or [r[0] for r in db.execute("SELECT DISTINCT paper_id FROM profiles ORDER BY paper_id")]
    k = 0
    for pid in papers:
        rows = db.execute("SELECT * FROM profiles WHERE paper_id=?", (pid,)).fetchall()
        for r in random.sample(rows, min(args.per_paper, len(rows))):
            k += 1
            text = chunks[r["chunk_id"]].text
            num = f"{r['value']:g}"
            line = next((ln for ln in text.split("\n") if re.search(rf"(?<![\d.]){re.escape(num)}(?!\d)", ln)), "?")
            print(f"#{k} {pid} {r['chunk_id'][-4:]} | {text.split(chr(10))[0][:110]}")
            print(f"   PROFILE model={r['model']!r} dataset={r['dataset']!r} ver={r['dataset_version']!r} metric={r['metric']!r} "
                  f"value={r['value']:g} lang={r['language']!r} size={r['model_size']!r} setting={r['setting']!r}")
            print(f"   SOURCE  {line[:230]}")


if __name__ == "__main__":
    main()
