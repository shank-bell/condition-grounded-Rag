"""Per-table view of the extraction of one paper: which tables were read, how many decimal cells became profiles.

  python scripts/table_detail.py 2302.13971
Explains where the cell recall of scripts/ingest_metrics.py is lost: skipped statistics tables, tables beyond the per-paper
cap, or tables the LLM read but only partly returned.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from collections import defaultdict

from cgrag.config import get_settings
from cgrag.ingestion.profile_extractor import MAX_CHUNKS_PER_PAPER, _views, wants_extraction
from cgrag.ingestion.tables import _rows, is_numeric_cell, is_statistics_table, is_structured_table, parse_table
from cgrag.stores.vector_store import VectorStore

DECIMAL = re.compile(r"\d+\.\d+")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pid = sys.argv[1]
    db = sqlite3.connect(f"file:{get_settings().paths.profile_db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    by_chunk = defaultdict(list)
    for r in db.execute("SELECT * FROM profiles WHERE paper_id=?", (pid,)):
        by_chunk[r["chunk_id"]].append(r)
    chunks = VectorStore().paper_chunks(pid)
    wanted = [c for c in chunks if wants_extraction(c)]
    wanted.sort(key=lambda c: c.section not in {"results", "experiments", "discussion"})
    read = {c.chunk_id for c in wanted[:MAX_CHUNKS_PER_PAPER]}
    print(f"{pid}: {len(chunks)} chunks, {len(wanted)} want extraction, {len(read)} read (cap {MAX_CHUNKS_PER_PAPER}), "
          f"{sum(len(v) for v in by_chunk.values())} profiles")
    tot = hit = 0
    for c in chunks:
        if not is_structured_table(c.text):
            continue
        t = parse_table(c.text)
        cells = [n for r in _rows(t) for p in r.pairs for n in DECIMAL.findall(p.partition(" = ")[2]) if is_numeric_cell(p.partition(" = ")[2])]
        mine = by_chunk.get(c.chunk_id, [])
        found = sum(any(abs(p["value"] - float(n)) < 1e-6 for p in mine) for n in cells)
        why = "STATISTICS-SKIPPED" if is_statistics_table(t.caption) else ("beyond cap" if c.chunk_id not in read else
                                                                          f"read in {len(_views(c.text))} views")
        if why != "STATISTICS-SKIPPED":
            tot += len(cells)
            hit += found
        print(f"  {c.chunk_id[-4:]} p{c.page:<3} {c.section:<12} cells {len(cells):>4} found {found:>4} ({len(mine)} profiles) {why:<22} | {t.caption[:70]}")
    print(f"recall excluding statistics tables: {hit}/{tot} = {100 * hit / max(tot, 1):.0f}%")


if __name__ == "__main__":
    main()
