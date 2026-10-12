"""Stage 4 - Hybrid Retrieval: dense (ChromaDB) + BM25, fused with Reciprocal Rank Fusion; section soft boost."""
from __future__ import annotations

from ..config import RetrievalConfig, get_settings
from ..models import embed
from ..schemas import Chunk, Intent, RetrievedChunk
from ..stores.bm25_store import BM25Store
from ..stores.profile_store import ProfileStore
from ..stores.vector_store import VectorStore

INTENT_SECTIONS: dict[str, set[str]] = {
    "factual": {"abstract", "introduction", "methods", "results", "experiments"},
    "comparison": {"results", "experiments", "discussion"},
    "method": {"methods", "introduction"},
    "result": {"results", "experiments"},
    "survey": {"introduction", "related_work", "abstract", "conclusion"},
}


CARD_INTENTS = {"result", "comparison", "factual"}


class HybridRetriever:
    def __init__(self, vectors: VectorStore, bm25: BM25Store, cfg: RetrievalConfig | None = None, profiles: ProfileStore | None = None) -> None:
        self.vectors, self.bm25 = vectors, bm25
        self.cfg = cfg or get_settings().retrieval
        self.profiles = profiles

    def profile_ids(self, conditions: dict | None) -> list[str]:
        """Chunks whose profiles record the named conditions TOGETHER, one list per compared system, interleaved (Stage 4's profile signal).

        At least two of model / dataset / language must be named (one alone matches half the store). A comparison ("mBERT and XLM-R on XNLI for Hindi")
        gives every compared system its own slots, so the evidence of the one that the text ranking prefers does not crowd out the other's table."""
        if not conditions or self.profiles is None:
            return []
        shared = {f: conditions[f] for f in ("dataset", "language") if conditions.get(f)}
        models = [m for m in [conditions.get("model"), *(conditions.get("other_models") or [])] if m] or [None]
        lists: list[list[str]] = []
        for model in models:
            key = {**shared, **({"model": model} if model else {})}
            if len(key) < 2:
                continue
            counts: dict[str, int] = {}
            for p in self.profiles.profiles_matching(key, list(key)):
                counts[p.chunk_id] = counts.get(p.chunk_id, 0) + 1
            lists.append([cid for cid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))])
        out: list[str] = []
        for rank in range(max((len(x) for x in lists), default=0)):
            for lst in lists:
                if rank < len(lst) and lst[rank] not in out:
                    out.append(lst[rank])
        return out[: self.cfg.profile_k]

    def retrieve(self, queries: list[str], intent: Intent = "factual", conditions: dict | None = None) -> list[RetrievedChunk]:
        """Top `fused_top` chunks for one or more queries (sub-questions are fused into a single ranking). `conditions` (Stage 1's model / dataset /
        language, plus other_models) switch on the profile signal for result / comparison / factual questions."""
        cfg = self.cfg
        rrf: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        vectors = embed(queries)
        use_cards = cfg.use_cards and intent in CARD_INTENTS
        for query, vec in zip(queries, vectors):
            for rank, (chunk, _) in enumerate(self.vectors.query(vec, cfg.dense_k), start=1):
                chunks[chunk.chunk_id] = chunk
                rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (cfg.rrf_k + rank)
            ranked_cards = self.vectors.query_cards(vec, cfg.dense_k) if use_cards else []
            sparse = self.bm25.search(query, cfg.bm25_k)
            missing = [cid for cid in dict.fromkeys([c for c, _ in sparse] + [c for c, _ in ranked_cards]) if cid not in chunks]
            for chunk in self.vectors.get(missing):
                chunks[chunk.chunk_id] = chunk
            for rank, (cid, _) in enumerate(ranked_cards, start=1):
                if cid in chunks:
                    rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)
            for rank, (cid, _) in enumerate(sparse, start=1):
                if cid in chunks:
                    rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)
        if use_cards and cfg.profile_signal:
            ids = self.profile_ids(conditions)
            missing = [cid for cid in ids if cid not in chunks]
            for chunk in self.vectors.get(missing):
                chunks[chunk.chunk_id] = chunk
            for rank, cid in enumerate(ids, start=1):
                if cid in chunks:
                    rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)
        wanted = INTENT_SECTIONS.get(intent, set())
        for cid in rrf:
            if chunks[cid].section in wanted:
                rrf[cid] *= 1.0 + cfg.section_boost
        top = sorted(rrf, key=rrf.get, reverse=True)[: cfg.fused_top]
        return [RetrievedChunk(chunk=chunks[cid], score=rrf[cid]) for cid in top]
