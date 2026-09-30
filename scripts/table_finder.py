"""Why did "Table N" of a paper not become a recognised table? Shows the layout model's boxes on the caption's page and
whether the indexed chunks contain that table.

  python scripts/table_finder.py 1907.11692 7 [more table numbers]
"""
from __future__ import annotations

import json
import re
import sys

import pymupdf as fitz
import pymupdf.layout  # noqa: F401
import pymupdf4llm

from cgrag.config import get_settings
from cgrag.ingestion.tables import is_structured_table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = get_settings()
    pid = sys.argv[1]
    path = cfg.paths.papers_dir / f"{pid}.pdf"
    doc = fitz.open(path)
    chunks = [json.loads(ln) for ln in (cfg.paths.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    mine = [c for c in chunks if c["paper_id"] == pid]
    for n in sys.argv[2:]:
        pat = re.compile(rf"^\s*Table\s+{n}\s*[:.]", re.M)
        page_no = next((i for i, p in enumerate(doc) if pat.search(p.get_text("text"))), None)
        print(f"\n##### {pid} Table {n}: page {None if page_no is None else page_no + 1}")
        hits = [c for c in mine if re.search(rf"Table\s+{n}\s*[:.]", c["text"])]
        for c in hits:
            print(f"  chunk {c['chunk_id']} section={c['section']} structured={is_structured_table(c['text'])} chars={len(c['text'])} starts: {c['text'][:70]!r}")
        if page_no is None:
            continue
        chunk = pymupdf4llm.to_markdown(str(path), pages=[page_no], page_chunks=True, header=False, footer=False, show_progress=False)[0]
        text = chunk["text"]
        for b in chunk.get("page_boxes") or []:
            s = text[b["pos"][0]:b["pos"][1]].replace("\n", " ")[:95]
            print(f"  {b['class']:<15} {[round(v) for v in b['bbox']]} | {s}")


if __name__ == "__main__":
    main()
