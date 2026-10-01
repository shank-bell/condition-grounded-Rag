"""Stage 5 - Rerank + Filter: a cross-encoder scores question and chunk together; keep chunks above a threshold."""
from __future__ import annotations

from ..config import RetrievalConfig, get_settings
from ..models import rerank_scores
from ..schemas import Chunk, RetrievedChunk

MIN_KEEP = 3          # if nothing clears the threshold, the best few are kept and flagged as weak evidence


def rerank_text(chunk: Chunk, use_cards: bool = True) -> str:
    """What the cross-encoder reads: the retrieval card in front of the chunk text. A results table is rows of numbers the
    cross-encoder cannot read (it scored the Kannada NLI table -10.8); the card says what the table records in words."""
    return f"{chunk.card}\n{chunk.text}" if use_cards and chunk.card else chunk.text


def rerank_filter(question: str, candidates: list[RetrievedChunk], cfg: RetrievalConfig | None = None) -> tuple[list[RetrievedChunk], bool]:
    """(kept chunks best first, weak) where weak means no chunk cleared the relevance threshold."""
    cfg = cfg or get_settings().retrieval
    scores = rerank_scores(question, [rerank_text(c.chunk, cfg.use_cards) for c in candidates])
    scored = [c.model_copy(update={"rerank_score": s}) for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c.rerank_score, reverse=True)
    kept = [c for c in scored if c.rerank_score >= cfg.rerank_threshold][: cfg.rerank_keep]
    if kept:
        return kept, False
    return scored[:MIN_KEEP], True
