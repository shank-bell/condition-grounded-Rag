"""Stage 5 - Rerank + Filter: a cross-encoder scores question and chunk together; keep chunks above a threshold."""
from __future__ import annotations

from ..config import RetrievalConfig, get_settings
from ..models import rerank_scores
from ..schemas import RetrievedChunk

MIN_KEEP = 3          # if nothing clears the threshold, the best few are kept and flagged as weak evidence


def rerank_filter(question: str, candidates: list[RetrievedChunk], cfg: RetrievalConfig | None = None) -> tuple[list[RetrievedChunk], bool]:
    """(kept chunks best first, weak) where weak means no chunk cleared the relevance threshold."""
    cfg = cfg or get_settings().retrieval
    scores = rerank_scores(question, [c.chunk.text for c in candidates])
    scored = [c.model_copy(update={"rerank_score": s}) for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c.rerank_score, reverse=True)
    kept = [c for c in scored if c.rerank_score >= cfg.rerank_threshold][: cfg.rerank_keep]
    if kept:
        return kept, False
    return scored[:MIN_KEEP], True
