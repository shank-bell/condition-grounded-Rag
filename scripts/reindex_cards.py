"""Build the retrieval cards of an existing index from its Condition Profile store (no LLM, ~1-2 min on a GPU).

  python scripts/reindex_cards.py                                   # the real index (nothing else may hold ChromaDB)
  python scripts/reindex_cards.py --chroma data/index_exp/chroma --bm25 data/index_exp/bm25.pkl      # an experimental copy
  python scripts/reindex_cards.py --cards raw|repaired|union ...    # which profile fields the cards are built from (default: what the system reads, i.e. repaired)

Run it again after anything that changes the profiles of a paper (reclean_profiles.py --apply, ingest of a paper is automatic).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from cgrag.config import get_settings
from cgrag.ingestion.cards import build_card
from cgrag.models import embed
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.profile_store import ProfileStore
from cgrag.stores.vector_store import VectorStore


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma", type=Path, default=cfg.paths.chroma_dir)
    ap.add_argument("--bm25", type=Path, default=cfg.paths.bm25_path)
    ap.add_argument("--profiles", type=Path, default=cfg.paths.profile_db)
    ap.add_argument("--cards", choices=["repaired", "raw", "union"], default=None,
                    help="profile fields the cards are built from: the repaired view (default when [features] profile_repair is on), the raw extraction, or both")
    args = ap.parse_args()

    t0 = time.time()
    vectors = VectorStore(args.chroma)
    ids = [c.chunk_id for c in vectors.all_chunks()]
    if args.cards == "union":      # every value of the raw and of the repaired profile counts (a card that keeps the old words and adds the corrected ones)
        raw, fixed = ProfileStore(args.profiles, repaired=False).for_chunks(ids), ProfileStore(args.profiles, repaired=True).for_chunks(ids)
        by_chunk = {cid: raw.get(cid, []) + fixed.get(cid, []) for cid in ids}
    else:
        profiles = ProfileStore(args.profiles, repaired=None if args.cards is None else args.cards == "repaired")
        by_chunk = profiles.for_chunks(ids)
    chunks = vectors.all_chunks()
    chunks = [c.model_copy(update={"card": build_card(by_chunk.get(c.chunk_id, []))}) for c in chunks]
    carded = [c for c in chunks if c.card]
    card_vectors = embed([c.card for c in carded]) if carded else None
    vectors.set_cards(chunks, card_vectors)
    BM25Store.build(chunks).save(args.bm25)
    lengths = [len(c.card) for c in carded]
    print(f"{len(chunks)} chunks, {len(carded)} with a card (mean {sum(lengths) / max(len(lengths), 1):.0f} chars); card vectors in store: "
          f"{vectors.card_count()}; BM25 rebuilt; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
