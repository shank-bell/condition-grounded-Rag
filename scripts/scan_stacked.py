"""Count layout-model tables that came back column-wise (cells holding whole columns joined by <br>): python scripts/scan_stacked.py"""
import re
import sys

import pymupdf.layout  # noqa: F401
import pymupdf4llm

from cgrag.config import get_settings

sys.stdout.reconfigure(encoding="utf-8")
papers = sorted(get_settings().paths.papers_dir.glob("*.pdf"))
total = stacked_total = 0
for pdf in papers:
    pages = pymupdf4llm.to_markdown(str(pdf), page_chunks=True, header=False, footer=False, show_progress=False)
    n = s = 0
    notes = []
    for pno, chunk in enumerate(pages, 1):
        text = chunk["text"]
        for b in chunk.get("page_boxes") or []:
            if b["class"] != "table":
                continue
            md = text[b["pos"][0]:b["pos"][1]]
            n += 1
            stacked_rows = [ln for ln in md.split("\n") if ln.startswith("|") and sum(1 for c in ln.split("|") if c.count("<br>") >= 2) >= 2]
            if stacked_rows and len(re.findall(r"\d+\.\d+", md)) >= 4:
                s += 1
                notes.append(f"p{pno}")
    total += n
    stacked_total += s
    print(f"{pdf.stem:<12} tables {n:>3}  stacked {s:>2} {' '.join(notes)}", flush=True)
print("TOTAL tables", total, "stacked", stacked_total)
