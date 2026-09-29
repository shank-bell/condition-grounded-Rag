"""Stage A - PDF Loader: PyMuPDF text page by page, with section headings detected.

Output is an ordered list of elements (heading / text / table), each tagged with its page.
Turning headings into IMRaD section labels and splitting into chunks is the chunker's job (stage B).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from .sections import NUMBERED, classify_heading

_CAPTION = re.compile(r"^(table|figure|fig\.|algorithm)\s*\d", re.I)
_NUM_TOKEN = re.compile(r"^[\(\[]?[-+]?\d+(?:[.,]\d+)?(?:/\d+(?:\.\d+)?)?(?:±\d+(?:\.\d+)?)?[%)\]]?$")
_DECIMAL = re.compile(r"\d+\.\d+")
_TOC_PAGE = re.compile(r"[A-Za-z)]\s+\d{1,3}$")     # a heading followed by its page number


@dataclass
class Element:
    page: int                     # 1-based
    text: str
    kind: str = "text"            # "heading" | "text" | "table"


@dataclass
class LoadedPaper:
    paper_id: str
    title: str
    n_pages: int
    elements: list[Element] = field(default_factory=list)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # fi/fl ligatures, odd spaces
    return re.sub(r"[ \t]+", " ", text).strip()


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines; glue words split by a line-end hyphen."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower() and len(out) > 3 and out[-2].isalpha():
            out = out[:-1] + line
        else:
            out += " " + line
    return out


def _body_size(doc: fitz.Document) -> float:
    """Most common font size by character count over the first pages = body text size."""
    counts: Counter[float] = Counter()
    for page in list(doc)[:6]:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    counts[round(span["size"], 1)] += len(span["text"])
    return counts.most_common(1)[0][0] if counts else 10.0


@dataclass
class _Row:
    text: str
    size: float
    bold_share: float
    y0: float
    y1: float


def _rows(block: dict) -> list[_Row]:
    """Text lines of a block, with lines on the same baseline merged left to right.

    PDFs often store a section number and its title, or the cells of a table row, as separate
    lines at the same height; merging them restores "1 Introduction" and "BERT 84.6 71.2 ...".
    """
    lines = sorted(block["lines"], key=lambda ln: (round(ln["spans"][0]["origin"][1] / 1.5), ln["bbox"][0]) if ln["spans"] else (0, 0))
    grouped: list[list[dict]] = []
    for ln in lines:
        if not ln["spans"]:
            continue
        base = ln["spans"][0]["origin"][1]
        if grouped and abs(grouped[-1][0]["spans"][0]["origin"][1] - base) <= 1.5:
            grouped[-1].append(ln)
        else:
            grouped.append([ln])
    rows: list[_Row] = []
    for group in grouped:
        group.sort(key=lambda ln: ln["bbox"][0])
        total = bold = 0
        size = 0.0
        for ln in group:
            for span in ln["spans"]:
                n = len(span["text"].strip())
                total += n
                if "bold" in span["font"].lower() or span["flags"] & 16:
                    bold += n
                size = max(size, span["size"])
        text = _clean(" ".join("".join(s["text"] for s in ln["spans"]) for ln in group))
        rows.append(_Row(text, size, bold / total if total else 0.0, min(ln["bbox"][1] for ln in group), max(ln["bbox"][3] for ln in group)))
    return rows


def is_table_row(text: str) -> bool:
    """A results-table row: mostly numbers, where "-" marks an empty cell."""
    tokens = text.split()
    numeric = sum(1 for t in tokens if _NUM_TOKEN.match(t))
    empty = sum(1 for t in tokens if t in ("-", "–", "—"))
    cells = numeric + empty
    return numeric >= 1 and cells >= 2 and cells / len(tokens) >= 0.5 and (numeric >= 2 or empty >= 1)


def _heading(text: str, size: float, bold_share: float, body: float, page: int) -> str | None:
    """The heading text if this line is a section heading, else None."""
    t = text.strip()
    if not t or len(t) > 95 or _CAPTION.match(t) or "http" in t or len(t.split()) > 12:
        return None
    if page == 1 and size >= body * 1.35:
        return None                                              # the paper title, not a section heading
    emphasised = bold_share >= 0.8 or size >= body * 1.08 or t.isupper()
    if not emphasised or t.endswith((".", ",", ";")):
        return None
    if NUMBERED.match(t):
        if page <= 3 and _TOC_PAGE.search(t):
            return None                                          # table-of-contents entry: "3 Results 10"
        return t
    if classify_heading(t) is not None and len(t.split()) <= 4:
        return t
    return None


def _title(doc: fitz.Document, fallback: str) -> str:
    """Largest-font lines on page 1 (before the abstract) form the title."""
    page = doc[0]
    rows: list[tuple[float, float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = _clean("".join(s["text"] for s in line["spans"]))
            if text and line["dir"][0] > 0.9:
                rows.append((max(s["size"] for s in line["spans"]), line["bbox"][1], text))
    if not rows:
        return fallback
    top = max(size for size, _, _ in rows)
    parts = [t for size, y, t in sorted(rows, key=lambda r: r[1]) if size >= top * 0.95 and y < page.rect.height * 0.4]
    title = " ".join(parts).strip()
    return title if len(title) >= 8 else (doc.metadata.get("title") or fallback)


def _tables(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Tables on the page as (bbox, markdown). Empty when detection finds nothing usable."""
    found: list[tuple[fitz.Rect, str]] = []
    try:
        for tab in page.find_tables().tables:
            if tab.row_count < 2 or tab.col_count < 2:
                continue
            md = tab.to_markdown().strip()
            if len(_DECIMAL.findall(md)) < 4:            # diagrams are often mis-detected as tables
                continue
            found.append((fitz.Rect(tab.bbox), md))
    except Exception:  # table detection is best-effort; plain text extraction still works
        return []
    return found


def load_pdf(path: Path) -> LoadedPaper:
    doc = fitz.open(path)
    body = _body_size(doc)
    paper = LoadedPaper(paper_id=path.stem, title=_title(doc, path.stem), n_pages=len(doc))
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        tables = _tables(page)
        emitted: set[int] = set()
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            centre = fitz.Point((block["bbox"][0] + block["bbox"][2]) / 2, (block["bbox"][1] + block["bbox"][3]) / 2)
            inside = next((i for i, (box, _) in enumerate(tables) if (box + (-2, -2, 2, 2)).contains(centre)), None)
            if inside is not None:                       # a table cell: emit the whole table once, in place
                if inside not in emitted:
                    emitted.add(inside)
                    paper.elements.append(Element(pno, tables[inside][1], "table"))
                continue
            para: list[str] = []
            grid: list[str] = []                         # consecutive numeric rows = a results table

            def flush() -> None:
                if para:
                    paper.elements.append(Element(pno, _join_lines(para), "text"))
                    para.clear()
                if grid:
                    paper.elements.append(Element(pno, "\n".join(grid), "table"))
                    grid.clear()

            block = {**block, "lines": [ln for ln in block["lines"] if ln["dir"][0] >= 0.9]}   # drop rotated arXiv stamp
            for row in _rows(block):
                if not row.text or row.y1 < height * 0.045 or row.y0 > height * 0.955:
                    continue                             # running header / footer
                head = _heading(row.text, row.size, row.bold_share, body, pno)
                if head:
                    flush()
                    paper.elements.append(Element(pno, head, "heading"))
                elif is_table_row(row.text):
                    if para and not grid:                # the header row just above the numbers belongs to the table
                        header = para.pop()
                        flush()
                        grid.append(header)
                    grid.append(row.text)
                else:
                    if grid:
                        flush()
                    para.append(row.text)
            flush()
        for i, (_, md) in enumerate(tables):             # tables whose cells were not found as text blocks
            if i not in emitted:
                paper.elements.append(Element(pno, md, "table"))
    doc.close()
    return paper
