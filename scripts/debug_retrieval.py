"""What do Stages 4-5 do with a question? Prints the retrieved chunks with their rerank scores, and where the chunks that the
profile store says are relevant (they record the question's conditions) ended up.

  python scripts/debug_retrieval.py "How well do models perform on Kannada NLI?" --cond language=Kannada --cond task="natural language inference"
  python scripts/debug_retrieval.py "..." --query "a rewritten query" --top 25
"""
from __future__ import annotations

import argparse

from cgrag.config import get_settings
from cgrag.models import rerank_scores
from cgrag.pipeline.rerank import rerank_text
from cgrag.pipeline.retrieval import HybridRetriever
from cgrag.schemas import RetrievedChunk
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.profile_store import ProfileStore
from cgrag.stores.vector_store import VectorStore
from cgrag.ingestion.tables import is_structured_table as is_table_chunk


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--query", action="append", default=[], help="extra retrieval queries (the rewritten question)")
    ap.add_argument("--cond", action="append", default=[], help="field=value condition; the profile store is asked which chunks record them")
    ap.add_argument("--intent", default="result")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    vectors, profiles = VectorStore(cfg.paths.chroma_dir), ProfileStore(cfg.paths.profile_db)
    retriever = HybridRetriever(vectors, BM25Store.load(cfg.paths.bm25_path), cfg.retrieval)
    queries = [args.question, *args.query]
    cands = retriever.retrieve(queries, args.intent)
    scores = rerank_scores(args.question, [rerank_text(c.chunk, cfg.retrieval.use_cards) for c in cands])
    ranked = sorted(zip(cands, scores), key=lambda cs: cs[1], reverse=True)
    print(f"{len(cands)} candidates from {len(queries)} queries; threshold {cfg.retrieval.rerank_threshold}; "
          f"{sum(1 for _, s in ranked if s >= cfg.retrieval.rerank_threshold)} above it")
    for i, (c, s) in enumerate(ranked[: args.top], 1):
        kind = "TABLE" if is_table_chunk(c.chunk.text) else "text "
        print(f"{i:>2} rerank {s:7.2f} fused {c.score:.4f} {kind} {c.chunk.chunk_id:<18} {c.chunk.section:<12} "
              f"{' '.join(c.chunk.text.split())[:70]}")

    requested = dict(kv.split("=", 1) for kv in args.cond)
    if requested:
        print(f"\nchunks whose ONE profile records all of {requested}:")
        fused_rank = {c.chunk.chunk_id: i for i, c in enumerate(cands, 1)}
        by_rerank = {c.chunk.chunk_id: (i, s) for i, (c, s) in enumerate(ranked, 1)}
        for field in requested:
            for cid in profiles.chunks_recording(requested, field, 8):
                chunk = vectors.get([cid])[0]
                s = rerank_scores(args.question, [rerank_text(chunk, cfg.retrieval.use_cards)])[0]
                where = f"retrieved #{fused_rank[cid]} (rerank rank {by_rerank[cid][0]})" if cid in fused_rank else "NOT retrieved"
                print(f"  {cid:<18} {'TABLE' if is_table_chunk(chunk.text) else 'text '} rerank {s:7.2f}  {where}  {' '.join(chunk.text.split())[:60]}")
            break


if __name__ == "__main__":
    main()
