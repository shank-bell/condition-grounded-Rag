"""Which table chunks look unreadable (glued numbers, fragment cells)? Uses data/index/chunks.jsonl and the profile store.

  python scripts/garble_report.py [--threshold 0.15] [--show 3]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from cgrag.config import get_settings
from cgrag.ingestion.tables import garble_ratio, is_structured_table, parse_table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--show", type=int, default=3, help="rows of each flagged table to print")
    args = ap.parse_args()
    cfg = get_settings()
    chunks = [json.loads(ln) for ln in (cfg.paths.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    db = sqlite3.connect(f"file:{cfg.paths.profile_db}?mode=ro", uri=True)
    counts = dict(db.execute("SELECT chunk_id, COUNT(*) FROM profiles GROUP BY chunk_id").fetchall())
    scored = []
    for c in chunks:
        if is_structured_table(c["text"]):
            scored.append((garble_ratio(c["text"]), c))
    scored.sort(key=lambda x: -x[0])
    flagged = [(g, c) for g, c in scored if g >= args.threshold]
    print(f"{len(scored)} table chunks, {len(flagged)} flagged at ratio >= {args.threshold}; "
          f"profiles in flagged chunks: {sum(counts.get(c['chunk_id'], 0) for _, c in flagged)} of {sum(counts.values())}")
    print("ratio distribution (top 25):", [round(g, 2) for g, _ in scored[:25]])
    for g, c in flagged:
        t = parse_table(c["text"])
        print(f"\n{c['chunk_id']} ratio {g:.2f} profiles {counts.get(c['chunk_id'], 0)} | {t.caption[:90]}")
        for row in t.rows[: args.show]:
            print("    ", " | ".join(row)[:170])


if __name__ == "__main__":
    main()
