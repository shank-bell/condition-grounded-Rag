"""Stage 4 - Hybrid Retrieval: dense (ChromaDB) + BM25, fused with Reciprocal Rank Fusion; section soft boost."""
from __future__ import annotations

from ..config import RetrievalConfig, get_settings
from ..models import embed
from ..schemas import Chunk, Intent, RetrievedChunk
from ..stores.bm25_store import BM25Store
from ..stores.vector_store import VectorStore

# Sections where each kind of question is usually answered. A match gives a soft boost, never a filter,
# so an answer that spans sections is not lost.
INTENT_SECTIONS: dict[str, set[str]] = {
    "factual": {"abstract", "introduction", "methods", "results", "experiments"},
    "comparison": {"results", "experiments", "discussion"},
    "method": {"methods", "introduction"},
    "result": {"results", "experiments"},
    "survey": {"introduction", "related_work", "abstract", "conclusion"},
}


class HybridRetriever:
    def __init__(self, vectors: VectorStore, bm25: BM25Store, cfg: RetrievalConfig | None = None) -> None:
        self.vectors, self.bm25 = vectors, bm25
        self.cfg = cfg or get_settings().retrieval

    def retrieve(self, queries: list[str], intent: Intent = "factual") -> list[RetrievedChunk]:
        """Top `fused_top` chunks for one or more queries (sub-questions are fused into a single ranking)."""
        cfg = self.cfg
        rrf: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        vectors = embed(queries)
        for query, vec in zip(queries, vectors):
            for rank, (chunk, _) in enumerate(self.vectors.query(vec, cfg.dense_k), start=1):
                chunks[chunk.chunk_id] = chunk
                rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (cfg.rrf_k + rank)
            sparse = self.bm25.search(query, cfg.bm25_k)
            missing = [cid for cid, _ in sparse if cid not in chunks]
            for chunk in self.vectors.get(missing):
                chunks[chunk.chunk_id] = chunk
            for rank, (cid, _) in enumerate(sparse, start=1):
                if cid in chunks:
                    rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)
        wanted = INTENT_SECTIONS.get(intent, set())
        for cid in rrf:
            if chunks[cid].section in wanted:
                rrf[cid] *= 1.0 + cfg.section_boost
        top = sorted(rrf, key=rrf.get, reverse=True)[: cfg.fused_top]
        return [RetrievedChunk(chunk=chunks[cid], score=rrf[cid]) for cid in top]
