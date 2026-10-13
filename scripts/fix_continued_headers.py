"""Give the continuation chunks of a long table the header rows that the chunker did not repeat.

  python scripts/fix_continued_headers.py            # list the chunks that would change (reads data/index/chunks.jsonl; writes nothing)
  python scripts/fix_continued_headers.py --apply    # update their text and vectors in ChromaDB, then re-export data/index/chunks.jsonl (the API must be stopped)

Why. A table that is longer than one chunk is cut between rows and each part starts with the caption and header (`tables.split_table_text`). The header was measured
with `header_row_count`, which stops at a names row whose cells are wider than a sub-header ("XL 2.7B 6.7B 13B 175B"), so GPT-3's per-size appendix table was cut into
three chunks of which only the first has the row that names the model sizes; the other two cannot be read (and their cards cannot be grounded). `_is_subheader` now
knows such a row, so new papers are cut correctly; this script repairs the chunks that are already stored. The chunk ids and the profiles do not change; the text of
the chunk gets the missing header rows after its first row (`tables.with_carried_headers`) and its vector is computed again. Run `scripts/reindex_cards.py` after it
(it rebuilds the keyword index from the stored chunks).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.config import get_settings
from cgrag.ingestion.tables import with_carried_headers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cfg = get_settings()
    path = cfg.paths.index_dir / "chunks.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_paper[r["paper_id"]].append(r)
    new_text: dict[str, str] = {}
    for rs in by_paper.values():
        rs.sort(key=lambda r: r["chunk_id"])
        for r, t in zip(rs, with_carried_headers([r["text"] for r in rs])):
            if t != r["text"]:
                new_text[r["chunk_id"]] = t
    for cid, t in new_text.items():
        old = next(r["text"] for r in rows if r["chunk_id"] == cid)
        print(f"{cid}: {len(old)} -> {len(t)} chars; rows added: {len(t.splitlines()) - len(old.splitlines())}")
    print(f"{len(new_text)} chunk(s) would change")
    if not args.apply or not new_text:
        return
    from cgrag.labelling.sampling import export_chunks
    from cgrag.models import embed
    from cgrag.stores.vector_store import VectorStore
    vectors = VectorStore()
    chunks = vectors.get(list(new_text))
    chunks = [c.model_copy(update={"text": new_text[c.chunk_id]}) for c in chunks]
    vectors.add(chunks, embed([c.text for c in chunks]))
    export_chunks(path)
    print(f"{len(chunks)} chunk(s) updated; chunks.jsonl re-exported")


if __name__ == "__main__":
    main()
