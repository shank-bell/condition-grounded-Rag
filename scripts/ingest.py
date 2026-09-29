"""Add papers to the system: PDF -> chunks -> condition profiles -> embeddings -> ChromaDB / BM25 / profile store.

  python scripts/ingest.py                   # every PDF in data/papers
  python scripts/ingest.py 1810.04805        # one paper (file stem or path)
  python scripts/ingest.py --force --limit 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cgrag.config import get_settings
from cgrag.ingestion.pipeline import Ingestor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("papers", nargs="*", help="file stems or PDF paths (default: all in data/papers)")
    ap.add_argument("--force", action="store_true", help="re-ingest papers that are already indexed")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    papers_dir = get_settings().paths.papers_dir
    pdfs = [Path(p) if p.endswith(".pdf") else papers_dir / f"{p}.pdf" for p in args.papers] or sorted(papers_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    ing = Ingestor()
    print(f"llm={ing.llm.cfg.model}  papers={len(pdfs)}")
    for i, pdf in enumerate(pdfs, 1):
        r = ing.ingest(pdf, force=args.force)
        if r.skipped:
            print(f"[{i}/{len(pdfs)}] {r.paper_id}: already indexed (use --force)")
            continue
        s = r.seconds
        print(f"[{i}/{len(pdfs)}] {r.paper_id} | {r.title[:60]} | {r.n_pages}pg {r.n_chunks} chunks {r.sections}")
        print(f"      profiles={r.n_profiles} from {r.chunks_extracted} chunks (failed={r.failures}, ungrounded dropped={r.dropped_ungrounded}, "
              f"non-results dropped={r.dropped_not_result})"
              f" | load {s['load']:.1f}s extract {s['extract']:.0f}s embed {s['embed']:.1f}s")
    n = ing.rebuild_keyword_index()
    print(f"BM25 rebuilt over {n} chunks; profiles in store: {ing.profiles.count()}")


if __name__ == "__main__":
    main()
