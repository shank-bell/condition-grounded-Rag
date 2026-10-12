"""Read the tables that were skipped as garbled but whose merged columns can be split without doubt (`tables.rescue_tables`), and add their profiles.

  python scripts/rescue_garbled.py             # list the chunks that can be rescued (writes nothing)
  python scripts/rescue_garbled.py --apply     # read them with the extractor model (Ollama) and add the raw profiles to the store

Why. Some tables arrive from the layout model with several columns of one header block in one cell ("SST-2 MRPC MRPC" over "97.1 93.6 91.5") and several
blocks stacked in one table (T5's Table 14). They were skipped as garbled, so the paper's main results were not in the store at all (found 11 Oct night with
the Papers with Code leaderboards: 6 of the 128 results of T5 that are printed in the paper). Only chunks that have no profile yet are read; a block that
cannot be split without doubt is left out, as before. Needs Ollama; reads the chunk texts from data/index/chunks.jsonl (no vector store), so it can run
while the API is up, but the cards must be rebuilt afterwards, which needs the vector store free:

  python scripts/resolve_datasets.py --papers <ids> --force    # the data set of the new groups, from the paper's text
  python scripts/repair_store.py --apply                       # the overlay (needs the API stopped, it opens the vector store)
  python scripts/reindex_cards.py                              # cards, vectors and BM25 from the repaired profiles
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.config import get_settings
from cgrag.ingestion.profile_extractor import extract_paper, wants_extraction
from cgrag.ingestion.tables import GARBLE_LIMIT, garble_ratio, is_structured_table, rescue_tables
from cgrag.llm import OllamaLLM
from cgrag.schemas import Chunk
from cgrag.stores.profile_store import ProfileStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--papers", nargs="*", help="paper ids (default: every paper)")
    args = ap.parse_args()
    cfg = get_settings()
    store = ProfileStore(repaired=False)
    have = {p.chunk_id for p in store.all()}
    by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for line in (cfg.paths.index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not args.papers or r["paper_id"] in args.papers:
                by_paper[r["paper_id"]].append(Chunk(chunk_id=r["chunk_id"], paper_id=r["paper_id"], paper_title=r.get("paper_title", ""), page=int(r["page"]),
                                                     section=r["section"], text=r["text"]))
    todo: dict[str, list[Chunk]] = {}
    for pid, chunks in sorted(by_paper.items()):
        found = [c for c in chunks if c.chunk_id not in have and is_structured_table(c.text) and garble_ratio(c.text) >= GARBLE_LIMIT
                 and rescue_tables(c.text) is not None and wants_extraction(c)]
        if found:
            todo[pid] = found
            for c in found:
                pieces = rescue_tables(c.text) or []
                print(f"{c.chunk_id} (p{c.page}): {len(pieces)} block(s) can be read: {c.text.splitlines()[0][:90]}")
    print(f"{sum(len(v) for v in todo.values())} chunk(s) of {len(todo)} paper(s)")
    if not args.apply or not todo:
        return
    llm = OllamaLLM(model=cfg.agents.extractor_model or None)
    added = 0
    for pid, found in todo.items():
        t0 = time.time()
        methods = [c for c in by_paper[pid] if c.section == "methods" and not is_structured_table(c.text) and not wants_extraction(c)]
        profiles, stats = extract_paper(methods + found, llm)
        mine = [p for p in profiles if p.chunk_id in {c.chunk_id for c in found}]
        store.add_many(mine)
        added += len(mine)
        print(f"{pid}: {len(mine)} profiles from {len(found)} chunk(s) ({time.time() - t0:.0f} s; failed {stats.failures}, ungrounded dropped {stats.dropped_ungrounded}, "
              f"non-results dropped {stats.dropped_not_result})", flush=True)
    print(f"{added} profiles added; profiles in the store: {store.count()}")


if __name__ == "__main__":
    main()
