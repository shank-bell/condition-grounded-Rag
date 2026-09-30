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

import pymupdf as fitz

try:
    import pymupdf.layout
    import pymupdf4llm
except Exception:
    pymupdf4llm = None

from .sections import NUMBERED, is_bare_section_title

_CAPTION = re.compile(r"^(table|figure|fig\.|algorithm)\s*\d", re.I)
_TABLE_CAPTION = re.compile(r"^\W*table\s*[A-Z]?\d+", re.I)
_SUP = re.compile(r"<sup>.*?</sup>", re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_EMPHASIS = re.compile(r"\*\*|(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
_SEPARATOR_ROW = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
MAX_CAPTION_GAP = 70.0
_NUM_TOKEN = re.compile(r"^[\(\[]?[-+]?\d+(?:[.,]\d+)?(?:/\d+(?:\.\d+)?)?(?:±\d+(?:\.\d+)?)?[%)\]]?$")
_DECIMAL = re.compile(r"\d+\.\d+")
_TOC_PAGE = re.compile(r"[A-Za-z)]\s+\d{1,3}$")


@dataclass
class Element:
    page: int
    text: str
    kind: str = "text"


@dataclass
class LoadedPaper:
    paper_id: str
    title: str
    n_pages: int
    elements: list[Element] = field(default_factory=list)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
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
        return None
    emphasised = bold_share >= 0.8 or size >= body * 1.08 or t.isupper()
    if not emphasised or t.endswith((".", ",", ";")):
        return None
    if NUMBERED.match(t):
        if page <= 3 and _TOC_PAGE.search(t):
            return None
        return t
    if is_bare_section_title(t):
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


def _clean_markup(text: str) -> str:
    """Markdown decoration and footnote marks removed: '**89.1**' -> '89.1', 'BERT<sup>†</sup>' -> 'BERT'."""
    text = unicodedata.normalize("NFKC", text)
    text = _EMPHASIS.sub("", _BR.sub(" ", _SUP.sub("", text)))
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=\d) \. (?=\d)", ".", text)


def pipe_table(markdown: str) -> str:
    """Markdown table -> one '| cell | cell |' line per row (separator row dropped, cells cleaned)."""
    rows = []
    for line in markdown.split("\n"):
        line = line.strip()
        if not line.startswith("|") or _SEPARATOR_ROW.match(line):
            continue
        body = line[1:-1] if line.endswith("|") and len(line) > 1 else line[1:]
        rows.append("| " + " | ".join(_clean_markup(c) for c in body.split("|")) + " |")
    return "\n".join(rows)


def match_captions(tables: list[tuple], captions: list[tuple], max_gap: float = MAX_CAPTION_GAP) -> dict[int, int]:
    """Pair each table (x0, y0, x1, y1) with its caption: the nearest caption directly above or below it in the same
    column; every caption is used once. Papers put captions above or below tables, so distance decides, not a convention."""
    gaps: dict[tuple[int, int], float] = {}
    for i, t in enumerate(tables):
        for j, c in enumerate(captions):
            overlap = min(t[2], c[2]) - max(t[0], c[0])
            if overlap < 0.4 * min(t[2] - t[0], c[2] - c[0]):
                continue
            if c[3] <= t[1] + 4:
                gap = max(0.0, t[1] - c[3])
            elif c[1] >= t[3] - 4:
                gap = max(0.0, c[1] - t[3])
            else:
                continue
            if gap <= max_gap:
                gaps[(i, j)] = gap
    if len(captions) > 12:
        return {}
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(i: int, used: int) -> tuple[int, float, tuple]:
        if i == len(tables):
            return 0, 0.0, ()
        n, cost, rest = best(i + 1, used)
        top = (n, -cost, rest, None)
        for j in range(len(captions)):
            if (i, j) in gaps and not used >> j & 1:
                n2, cost2, rest2 = best(i + 1, used | 1 << j)
                cand = (n2 + 1, -(cost2 + gaps[(i, j)]), rest2, j)
                if cand[:2] > top[:2]:
                    top = cand
        n, neg_cost, rest, pick = top
        return n, -neg_cost, (((i, pick),) if pick is not None else ()) + rest

    return dict(best(0, 0)[2])


_LEADERS = re.compile(r"(?:\.\s){6,}")


def _worth_keeping(table_text: str, has_caption: bool) -> bool:
    """Diagrams are often mis-detected as tables (so is a table of contents, whose section numbers look like decimals); a
    results table has decimals (or many numbers under a caption)."""
    if _LEADERS.search(table_text):
        return False
    return len(_DECIMAL.findall(table_text)) >= 4 or (has_caption and len(re.findall(r"\d+", table_text)) >= 10)


def _layout_pages(path: Path) -> list[dict] | None:
    if pymupdf4llm is None:
        return None
    try:
        return pymupdf4llm.to_markdown(str(path), page_chunks=True, header=False, footer=False, show_progress=False)
    except Exception:
        return None


def _layout_tables(chunk: dict) -> tuple[list[tuple[fitz.Rect, str]], list[fitz.Rect]]:
    """Tables of one page from the layout model as (bbox, caption + rows), and the caption boxes they absorbed."""
    text, boxes = chunk.get("text", ""), chunk.get("page_boxes") or []
    tabs: list[tuple[tuple, str]] = []
    caps: list[tuple[tuple, str]] = []
    for b in boxes:
        body = text[b["pos"][0]:b["pos"][1]]
        if b["class"] == "table":
            tabs.append((tuple(b["bbox"]), body))
        elif b["class"] in ("caption", "text") and _TABLE_CAPTION.match(_clean_markup(body)):
            caps.append((tuple(b["bbox"]), _clean_markup(body)))
    match = match_captions([t[0] for t in tabs], [c[0] for c in caps])
    found: list[tuple[fitz.Rect, str]] = []
    swallowed: list[fitz.Rect] = []
    for i, (bbox, md) in enumerate(tabs):
        rows = pipe_table(md)
        caption = caps[match[i]][1] if i in match else ""
        if not rows or not _worth_keeping(rows, bool(caption)):
            continue
        found.append((fitz.Rect(bbox), f"{caption}\n{rows}" if caption else rows))
        if caption:
            swallowed.append(fitz.Rect(caps[match[i]][0]))
    return found, swallowed


def _tables(page: fitz.Page, layout_page: dict | None = None) -> tuple[list[tuple[fitz.Rect, str]], list[fitz.Rect]]:
    """Tables on the page as (bbox, text) plus the caption boxes already inside those texts. Empty when detection
    finds nothing usable."""
    if layout_page is not None:
        found, swallowed = _layout_tables(layout_page)
        if found:
            return found, swallowed
    found = []
    try:
        for tab in page.find_tables().tables:
            if tab.row_count < 2 or tab.col_count < 2:
                continue
            rows = pipe_table(tab.to_markdown())
            if rows and _worth_keeping(rows, False):
                found.append((fitz.Rect(tab.bbox), rows))
    except Exception:
        return [], []
    return found, []


def load_pdf(path: Path) -> LoadedPaper:
    doc = fitz.open(path)
    body = _body_size(doc)
    paper = LoadedPaper(paper_id=path.stem, title=_title(doc, path.stem), n_pages=len(doc))
    layout = _layout_pages(path)
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        tables, captions = _tables(page, layout[pno - 1] if layout and pno - 1 < len(layout) else None)
        emitted: set[int] = set()
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            centre = fitz.Point((block["bbox"][0] + block["bbox"][2]) / 2, (block["bbox"][1] + block["bbox"][3]) / 2)
            inside = next((i for i, (box, _) in enumerate(tables) if (box + (-2, -2, 2, 2)).contains(centre)), None)
            if inside is not None:
                if inside not in emitted:
                    emitted.add(inside)
                    paper.elements.append(Element(pno, tables[inside][1], "table"))
                continue
            if any((box + (-2, -2, 2, 2)).contains(centre) for box in captions):
                continue
            para: list[str] = []
            grid: list[str] = []

            def flush() -> None:
                if para:
                    paper.elements.append(Element(pno, _join_lines(para), "text"))
                    para.clear()
                if grid:
                    paper.elements.append(Element(pno, "\n".join(grid), "table"))
                    grid.clear()

            block = {**block, "lines": [ln for ln in block["lines"] if ln["dir"][0] >= 0.9]}
            for row in _rows(block):
                if not row.text or row.y1 < height * 0.045 or row.y0 > height * 0.955:
                    continue
                head = _heading(row.text, row.size, row.bold_share, body, pno)
                if head:
                    flush()
                    paper.elements.append(Element(pno, head, "heading"))
                elif is_table_row(row.text):
                    if para and not grid:
                        header = para.pop()
                        flush()
                        grid.append(header)
                    grid.append(row.text)
                else:
                    if grid:
                        flush()
                    para.append(row.text)
            flush()
        for i, (_, md) in enumerate(tables):
            if i not in emitted:
                paper.elements.append(Element(pno, md, "table"))
    doc.close()
    return paper
