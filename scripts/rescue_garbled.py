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
from cgrag.ingestion.profile_extractor import extract_paper, size_family, wants_extraction
from cgrag.ingestion.tables import GARBLE_LIMIT, garble_ratio, is_structured_table, rescue_tables, with_carried_headers
from cgrag.llm import OllamaLLM
from cgrag.schemas import Chunk
from cgrag.stores.profile_store import ProfileStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--papers", nargs="*", help="paper ids (default: every paper)")
    ap.add_argument("--replace", nargs="*", default=[], help="chunk ids that already have profiles and are re-read (a block that could not be read before): the old "
                    "profiles of the chunk are swapped for the new reading only when it has at least as many")
    ap.add_argument("--min-ratio", type=float, default=1.0, help="with --replace: the new reading replaces the old profiles when it has at least this share of their number "
                    "(default 1.0 = at least as many; 0.95 for a chunk whose table reader was corrected, where the names matter more than 5 % of the results)")
    ap.add_argument("--complete", nargs="*", default=[], help="chunk ids that are only partly read: the chunk is read again and ONLY the results whose number is missing "
                    "from the stored profiles are added, with new ids after the existing ones (no stored profile id changes: the labelled Stage 7 pairs refer to them)")
    ap.add_argument("--empty", action="store_true", help="also read every wanted chunk that has no profile at all (a first reading that failed, or tables whose headers "
                    "name no metric: those are kept as 'score' now)")
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
    all_texts: dict[str, dict[str, str]] = {}
    for pid, chunks in sorted(by_paper.items()):
        texts = dict(zip((c.chunk_id for c in chunks), with_carried_headers([c.text for c in chunks])))
        all_texts[pid] = texts
        found = [c for c in chunks if wants_extraction(c, texts[c.chunk_id])
                 and ((c.chunk_id not in have and is_structured_table(texts[c.chunk_id]) and garble_ratio(texts[c.chunk_id]) >= GARBLE_LIMIT
                       and rescue_tables(texts[c.chunk_id]) is not None)
                      or (c.chunk_id in args.replace and is_structured_table(texts[c.chunk_id])) or c.chunk_id in args.complete
                      or (args.empty and c.chunk_id not in have))]
        if found:
            todo[pid] = found
            for c in found:
                t = texts[c.chunk_id]
                pieces = (rescue_tables(t) or []) if is_structured_table(t) and garble_ratio(t) >= GARBLE_LIMIT else []
                what = f"{len(pieces)} block(s) can be read" if pieces else ("no profile yet: read again" if c.chunk_id not in have else "re-read with the current table reader")
                print(f"{c.chunk_id} (p{c.page}): {what}: {c.text.splitlines()[0][:90]}")
    print(f"{sum(len(v) for v in todo.values())} chunk(s) of {len(todo)} paper(s)")
    if not args.apply or not todo:
        return
    llm = OllamaLLM(model=cfg.agents.extractor_model or None)
    added = 0
    for pid, found in todo.items():
        t0 = time.time()
        methods = [c for c in by_paper[pid] if c.section == "methods" and not is_structured_table(c.text) and not wants_extraction(c)]
        profiles, stats = extract_paper(methods + found, llm, all_texts[pid], size_family(c.text for c in by_paper[pid]))
        mine = [p for p in profiles if p.chunk_id in {c.chunk_id for c in found}]
        for cid in [c.chunk_id for c in found if c.chunk_id in args.complete]:
            from collections import Counter
            existing = [p for p in store.all() if p.chunk_id == cid]
            have_vals = Counter(round(float(p.value), 4) for p in existing if p.value is not None)
            start = 1 + max([int(p.profile_id.rsplit("#", 1)[1]) for p in existing] or [-1])
            fresh, seen = [], Counter()
            for p in (q for q in mine if q.chunk_id == cid):
                v = round(float(p.value), 4) if p.value is not None else None
                seen[v] += 1
                if v is not None and seen[v] > have_vals.get(v, 0):
                    fresh.append(p.model_copy(update={"profile_id": f"{cid}#{start + len(fresh)}"}))
            print(f"   {cid}: {len(existing)} stored, new reading {sum(1 for q in mine if q.chunk_id == cid)}, {len(fresh)} added")
            store.add_many(fresh)
            added += len(fresh)
            mine = [p for p in mine if p.chunk_id != cid]
        for cid in [c.chunk_id for c in found if c.chunk_id in args.replace]:
            new = [p for p in mine if p.chunk_id == cid]
            old = sum(1 for p in store.all() if p.chunk_id == cid)
            if len(new) >= old * args.min_ratio:
                print(f"   {cid}: replacing {old} profiles by {len(new)}")
                store.replace_chunk(cid, new)
            else:
                print(f"   {cid}: the new reading has {len(new)} < {old} profiles: kept the old ones")
            mine = [p for p in mine if p.chunk_id != cid]
        store.add_many(mine)
        added += len(mine)
        print(f"{pid}: {len(mine)} profiles from {len(found)} chunk(s) ({time.time() - t0:.0f} s; failed {stats.failures}, ungrounded dropped {stats.dropped_ungrounded}, "
              f"non-results dropped {stats.dropped_not_result})", flush=True)
    print(f"{added} profiles added; profiles in the store: {store.count()}")


if __name__ == "__main__":
    main()
