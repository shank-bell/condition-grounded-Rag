"""Does Stage 4 (hybrid retrieval) + Stage 5 (rerank) find the PAGE that holds the answer? Scored against the AI-annotated question key.

The silver test (scripts/eval_retrieval.py) makes its questions from the Condition Profile store and counts a chunk as relevant when the same
store says so, so it is circular. Here the relevant chunks are the ones on the pages the key names as evidence (the `evidence` field of
eval/labels/questions_gold.jsonl, written from the PDFs: "1810.04805 p7 Table 3 (SQuAD 2.0 results)"), whatever the profile store holds.
No LLM is used: the question text is the query (the pipeline would also rewrite it, so this is a floor), the intent is the key's.

  python scripts/eval_retrieval_gold.py                                 # prints the summary, writes eval/labels/retrieval_gold.json
  python scripts/eval_retrieval_gold.py --chroma <copy of data/index/chroma>   # read a COPY while the API is running (Chroma is single-process)

A question counts only when its evidence names a paper AND a page ("2010.11934 p15", "p1-3"); questions whose evidence is a bare paper list
or text are skipped (reported). Relevant = any chunk of that paper whose page is one of the named pages. Reports hit@20 (among the 20 fused
candidates), hit@5 / hit@kept after the reranker and its threshold, MRR of the first relevant chunk, and the same at paper level.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from cgrag.config import get_settings
from cgrag.labelling.sampling import load_chunk_index
from cgrag.models import rerank_scores
from cgrag.pipeline.rerank import rerank_text
from cgrag.pipeline.retrieval import HybridRetriever
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.vector_store import VectorStore

_TOKEN = re.compile(r"(\d{4}\.\d{4,5})|\bp(\d+)(?:\s*-\s*(\d+))?\b")


def evidence_pages(text: str) -> set[tuple[str, int]]:
    """(paper, page) pairs named in an evidence string: an arXiv id sets the current paper, each following 'p7' / 'p1-3' is a page of it."""
    out: set[tuple[str, int]] = set()
    paper = ""
    for m in _TOKEN.finditer(text or ""):
        if m.group(1):
            paper = m.group(1)
        elif paper:
            first = int(m.group(2))
            last = int(m.group(3)) if m.group(3) else first
            out.update((paper, p) for p in range(first, last + 1))
    return out


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("eval/labels/questions_gold.jsonl"))
    ap.add_argument("--chroma", type=Path, default=cfg.paths.chroma_dir)
    ap.add_argument("--bm25", type=Path, default=cfg.paths.bm25_path)
    ap.add_argument("--out", type=Path, default=Path("eval/labels/retrieval_gold.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    page_of = index.pages
    retriever = HybridRetriever(VectorStore(args.chroma), BM25Store.load(args.bm25), cfg.retrieval)
    rows, skipped = [], []
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        pages = evidence_pages(q.get("evidence") or "")
        rel = {cid for cid, pg in page_of.items() if (cid.split(":")[0], pg) in pages}
        rel_papers = {p for p, _ in pages}
        if not rel:
            skipped.append(q["id"])
            continue
        cands = retriever.retrieve([q["question"]], q.get("intent") or "result")
        scores = rerank_scores(q["question"], [rerank_text(c.chunk, cfg.retrieval.use_cards) for c in cands])
        ranked = [c for c, _ in sorted(zip(cands, scores), key=lambda cs: cs[1], reverse=True)]
        order = [c.chunk.chunk_id for c in ranked]
        kept_ids = [c.chunk.chunk_id for c, s in sorted(zip(cands, scores), key=lambda cs: cs[1], reverse=True)
                    if s >= cfg.retrieval.rerank_threshold][: cfg.retrieval.rerank_keep]
        first = next((i for i, cid in enumerate(order, 1) if cid in rel), None)
        first_paper = next((i for i, cid in enumerate(order, 1) if cid.split(":")[0] in rel_papers), None)
        rows.append({"id": q["id"], "planned_type": q.get("planned_type"), "n_relevant": len(rel), "pages": sorted(f"{p} p{n}" for p, n in pages),
                     "hit20": bool(rel & set(order)), "hit5": bool(rel & set(order[:5])), "hit_kept": bool(rel & set(kept_ids)),
                     "rr": 1 / first if first else 0.0, "rank_first": first, "paper_hit5": first_paper is not None and first_paper <= 5,
                     "kept": len(kept_ids)})

    def pct(key: str) -> float:
        return round(100 * sum(1 for r in rows if r[key]) / len(rows), 1)

    summary = {"questions_scored": len(rows), "skipped_no_page_evidence": skipped, "hit@20": pct("hit20"), "hit@5(rerank)": pct("hit5"),
               "hit@kept(rerank threshold)": pct("hit_kept"), "paper_hit@5": pct("paper_hit5"),
               "MRR(rerank order)": round(statistics.mean(r["rr"] for r in rows), 3), "mean_kept": round(statistics.mean(r["kept"] for r in rows), 2),
               "tag": args.tag, "reranker_threshold": cfg.retrieval.rerank_threshold, "use_cards": cfg.retrieval.use_cards}
    for kind in sorted({r["planned_type"] for r in rows if r["planned_type"]}):
        part = [r for r in rows if r["planned_type"] == kind]
        summary[f"hit@5 {kind} (n={len(part)})"] = round(100 * sum(r["hit5"] for r in part) / len(part), 1)
    print(json.dumps(summary, indent=1))
    misses = [(r["id"], r["rank_first"], r["pages"][:3]) for r in rows if not r["hit5"]]
    print("not in the top 5 (question, rank of the first relevant chunk, pages):", misses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
