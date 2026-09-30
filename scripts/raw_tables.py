"""Raw layout-model markdown of the tables on one page: python scripts/raw_tables.py 2005.11401 6"""
import sys

import pymupdf.layout  # noqa: F401
import pymupdf4llm

from cgrag.config import get_settings

sys.stdout.reconfigure(encoding="utf-8")
pid, page = sys.argv[1], int(sys.argv[2])
path = get_settings().paths.papers_dir / f"{pid}.pdf"
chunk = pymupdf4llm.to_markdown(str(path), pages=[page - 1], page_chunks=True, header=False, footer=False, show_progress=False)[0]
text = chunk["text"]
for b in chunk.get("page_boxes") or []:
    if b["class"] == "table":
        print("=" * 30, [round(v) for v in b["bbox"]])
        print(text[b["pos"][0]:b["pos"][1]])
