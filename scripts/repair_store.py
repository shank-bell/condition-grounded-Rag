"""Apply (or remove) the profile repair on an existing store: the raw `profiles` table is never touched, the corrections go into the overlay table
`profile_repairs` (src/cgrag/ingestion/repair.py); `[features] profile_repair` decides whether the system reads through it.

  python scripts/repair_store.py --stats                 # what the repair would change (writes nothing)
  python scripts/repair_store.py --apply                 # write the overlay for every paper
  python scripts/repair_store.py --clear                 # remove it again (the system then reads the raw extraction)
  CGRAG_OVERLAY=config/bench_metalead.toml python scripts/repair_store.py --apply      # the benchmark store

After --apply the retrieval cards must be rebuilt from the repaired fields:  python scripts/reindex_cards.py
Do not run while the API or an ingest holds the vector store.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.ingestion.repair import repair_paper  # noqa: E402
from cgrag.stores.profile_store import ProfileStore  # noqa: E402
from cgrag.stores.vector_store import VectorStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stats", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--clear", action="store_true")
    ap.add_argument("--off", nargs="*", default=[], help="rule names to leave out (the ablation)")
    args = ap.parse_args()

    store = ProfileStore(repaired=False)
    if args.clear:
        store.clear_repairs()
        print("overlay cleared")
        return
    vs = VectorStore()
    papers = sorted({p.paper_id for p in store.all()})
    total, changed, rules, fields = 0, 0, Counter(), Counter()
    for pid in papers:
        raw = store.for_paper(pid)
        cs = vs.paper_chunks(pid)
        fixed, counts = repair_paper(raw, {c.chunk_id: c.text for c in cs}, frozenset(args.off), cs[0].paper_title if cs else "")
        rules.update(counts)
        n = 0
        for a, b in zip(raw, fixed):
            diff = [f for f in ("task", "dataset", "dataset_version", "metric", "language", "model", "model_size", "setting")
                    if (getattr(a, f) or None) != (getattr(b, f) or None)]
            fields.update(diff)
            n += bool(diff)
        total += len(raw)
        changed += n
        if args.apply:
            store.set_repairs(pid, raw, fixed)
    print(f"{len(papers)} papers, {total} profiles, {changed} changed ({100 * changed / max(1, total):.1f} %)")
    print("fields changed:", dict(fields.most_common()))
    print("rules fired:", dict(rules.most_common()))
    print("overlay rows in the store:", store.repair_count())


if __name__ == "__main__":
    main()
