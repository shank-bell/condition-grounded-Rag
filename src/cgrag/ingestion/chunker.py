"""Stage B - Section Chunker: split text inside section boundaries; tag paper, page and section.

A "section" here is a run of consecutive text with the same IMRaD label, so numbered subsections
(4.1 GLUE, 4.2 SQuAD ...) stay inside their parent instead of becoming tiny chunks.
"""
from __future__ import annotations

import re
from bisect import bisect_right

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import ChunkingConfig, get_settings
from ..schemas import Chunk, Section
from .pdf_loader import LoadedPaper
from .sections import classify_heading, parse_heading
from .tables import is_structured_table, split_table_text

MIN_CHUNK_CHARS = 60
TABLE_MAX_CHARS = 5000        # a table longer than this is cut into parts of whole rows (caption + header repeated)


def label_elements(paper: LoadedPaper) -> list[tuple[Section, "object"]]:
    """Pair every element with the section label in force where it appears."""
    has_abstract_heading = any(
        e.kind == "heading" and classify_heading(parse_heading(e.text)[1]) == "abstract" for e in paper.elements
    )
    current: Section = "other" if has_abstract_heading else "abstract"   # front matter before the first heading
    seen_intro = seen_results = seen_end = in_appendix = False
    out: list[tuple[Section, object]] = []
    for el in paper.elements:
        if el.kind == "heading":
            num, title = parse_heading(el.text)
            depth = 0 if num is None else num.count(".") + 1
            label = classify_heading(title)
            if re.search(r"\bappendix\b", title, re.I) or (num and re.fullmatch(r"[A-H]", num.split(".")[0]) and depth == 1 and seen_end):
                in_appendix = True
            if in_appendix:
                current = "other"
            elif depth <= 1:                                   # top-level (numbered or a bare keyword heading)
                if label is not None:
                    current = label
                elif depth == 1:                               # e.g. "3 BERT" between introduction and experiments
                    current = "methods" if seen_intro and not seen_results else current
            # numbered subsections (depth >= 2) inherit the label of their parent section
            seen_intro |= current == "introduction"
            seen_results |= current in ("experiments", "results")
            seen_end |= current in ("conclusion", "references")
        out.append((current, el))
    return out


def chunk_paper(paper: LoadedPaper, cfg: ChunkingConfig | None = None, *, keep_references: bool = False) -> list[Chunk]:
    cfg = cfg or get_settings().chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""], add_start_index=True,
    )
    # group consecutive elements that share a label into runs
    runs: list[tuple[Section, list]] = []
    for label, el in label_elements(paper):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(el)
        else:
            runs.append((label, [el]))

    chunks: list[Chunk] = []

    def add(label: Section, page: int, body: str) -> None:
        chunks.append(Chunk(chunk_id=f"{paper.paper_id}:{len(chunks):04d}", paper_id=paper.paper_id,
                            paper_title=paper.title, page=page, section=label, text=body))

    def add_text(label: Section, els: list) -> None:
        text, starts, pages = "", [], []
        for i, el in enumerate(els):
            if text:
                text += "\n" if (el.kind == "table" and els[i - 1].kind == "table") else "\n\n"
            starts.append(len(text))
            pages.append(el.page)
            text += el.text
        for doc in splitter.create_documents([text]):
            body = doc.page_content.strip()
            if len(body) >= MIN_CHUNK_CHARS:
                add(label, pages[max(0, bisect_right(starts, doc.metadata["start_index"]) - 1)], body)

    for label, els in runs:
        if label == "references" and not keep_references:
            # the bibliography is left out, but a results table that a float pushed in between (the top of an appendix
            # page that starts after the last reference) is not part of it
            for el in els:
                if el.kind == "table" and is_structured_table(el.text):
                    for part in split_table_text(el.text, TABLE_MAX_CHARS):
                        add("other", el.page, part)
            continue
        pending: list = []                                     # prose (and unstructured tables) waiting to be split
        for el in els:
            if el.kind == "table" and is_structured_table(el.text):
                add_text(label, pending)                       # a recognised table is never cut through a row and keeps
                pending = []                                   # its caption + header: it is a chunk of its own
                for part in split_table_text(el.text, TABLE_MAX_CHARS):
                    add(label, el.page, part)
            else:
                pending.append(el)
        add_text(label, pending)
    return chunks
