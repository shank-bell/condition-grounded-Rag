"""Offline path, run when a paper is added: A load -> B chunk -> C profiles -> D embed -> stores."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings
from ..llm import OllamaLLM
from ..models import embed
from ..stores.bm25_store import BM25Store
from ..stores.profile_store import ProfileStore
from ..stores.vector_store import VectorStore
from .chunker import chunk_paper
from .pdf_loader import load_pdf
from .profile_extractor import extract_paper


@dataclass
class IngestReport:
    paper_id: str
    title: str = ""
    skipped: bool = False
    n_pages: int = 0
    n_chunks: int = 0
    sections: dict[str, int] = field(default_factory=dict)
    chunks_extracted: int = 0
    n_profiles: int = 0
    failures: int = 0
    dropped_ungrounded: int = 0
    dropped_not_result: int = 0
    seconds: dict[str, float] = field(default_factory=dict)


class Ingestor:
    def __init__(self, llm: OllamaLLM | None = None, vectors: VectorStore | None = None, profiles: ProfileStore | None = None):
        self.llm = llm or OllamaLLM()
        self.vectors = vectors or VectorStore()
        self.profiles = profiles or ProfileStore()

    def ingest(self, pdf: Path, *, force: bool = False) -> IngestReport:
        paper_id = pdf.stem
        if not force and self.vectors.has_paper(paper_id) and self.profiles.was_extracted(paper_id):
            return IngestReport(paper_id, skipped=True)
        t = {}
        t0 = time.perf_counter()
        paper = load_pdf(pdf)                                               # A
        t["load"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunks = chunk_paper(paper)                                         # B
        t["chunk"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        profiles, stats = extract_paper(chunks, self.llm)                   # C
        t["extract"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vectors = embed([c.text for c in chunks])                           # D
        t["embed"] = time.perf_counter() - t0
        self.vectors.delete_paper(paper_id)
        self.profiles.delete_paper(paper_id)
        self.vectors.add(chunks, vectors)
        self.profiles.add_many(profiles)
        self.profiles.log_extraction(paper_id, stats.chunks_extracted, len(profiles), self.llm.cfg.model, t["extract"])
        return IngestReport(
            paper_id, paper.title, False, paper.n_pages, len(chunks), dict(Counter(c.section for c in chunks)),
            stats.chunks_extracted, len(profiles), stats.failures, stats.dropped_ungrounded, stats.dropped_not_result, t)

    def rebuild_keyword_index(self) -> int:
        """BM25 is rebuilt from all stored chunks so it always matches ChromaDB."""
        chunks = self.vectors.all_chunks()
        BM25Store.build(chunks).save(get_settings().paths.bm25_path)
        return len(chunks)
