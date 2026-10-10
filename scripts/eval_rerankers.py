"""Which cross-encoder ranks the gold evidence pages best? (10 Oct 2026; a dev measurement on 28 questions)

  python scripts/eval_rerankers.py                                    # the current reranker and the stronger ones of docs/backup_models_huggingface.md
  python scripts/eval_rerankers.py --models cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base

Same questions and the same 20 fused candidates as scripts/eval_retrieval_gold.py (the raw question is the only query, so this is a floor; no LLM);
only the cross-encoder changes. Reports hit@3 / hit@5 / hit@10, MRR and the paper-level hit@5, per question type, and the seconds per question.
Needs Chroma free: stop the API first. Writes eval/labels/rerankers_gold.json (aggregates and the rank of the first relevant chunk per question).
Each model after the first is compared question by question with the first one (sign test on hit@5).
The pipeline's keep-threshold (`rerank_threshold`) lives on the score scale of the model, so a new model needs its own threshold
(scripts/eval_retrieval.py sweep, run with CGRAG_CONFIG pointing to a config that names the model) before it is used in the pipeline.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_retrieval_gold import evidence_pages            # noqa: E402

from cgrag import models as M                              # noqa: E402
from cgrag.config import get_settings                      # noqa: E402
from cgrag.evaluation.answers import sign_test             # noqa: E402
from cgrag.labelling.sampling import load_chunk_index      # noqa: E402
from cgrag.pipeline.rerank import rerank_text              # noqa: E402
from cgrag.pipeline.retrieval import HybridRetriever       # noqa: E402
from cgrag.stores.bm25_store import BM25Store              # noqa: E402
from cgrag.stores.vector_store import VectorStore          # noqa: E402

DEFAULT_MODELS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"]


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--gold", type=Path, default=Path("eval/labels/questions_gold.jsonl"))
    ap.add_argument("--chroma", type=Path, default=cfg.paths.chroma_dir)
    ap.add_argument("--bm25", type=Path, default=cfg.paths.bm25_path)
    ap.add_argument("--out", type=Path, default=Path("eval/labels/rerankers_gold.json"))
    args = ap.parse_args()

    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    retriever = HybridRetriever(VectorStore(args.chroma), BM25Store.load(args.bm25), cfg.retrieval)
    cases = []
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        pages = evidence_pages(q.get("evidence") or "")
        rel = {cid for cid, pg in index.pages.items() if (cid.split(":")[0], pg) in pages}
        if not rel:
            continue
        cands = retriever.retrieve([q["question"]], q.get("intent") or "result")
        cases.append({"id": q["id"], "type": q.get("planned_type"), "question": q["question"], "cands": cands, "rel": rel, "papers": {p for p, _ in pages}})
    print(f"{len(cases)} gold questions with page evidence, {sum(len(c['cands']) for c in cases)} candidate passages in all", flush=True)

    results: dict[str, dict] = {}
    for name in args.models:
        cfg.models.reranker = name
        M._reranker.cache_clear()
        M.rerank_scores("warm up", ["warm up"])                    # load the model (and download it the first time) outside the timing
        rows, t0 = [], time.perf_counter()
        for c in cases:
            scores = M.rerank_scores(c["question"], [rerank_text(x.chunk, cfg.retrieval.use_cards) for x in c["cands"]])
            order = [x.chunk.chunk_id for x, _ in sorted(zip(c["cands"], scores), key=lambda xs: xs[1], reverse=True)]
            first = next((i for i, cid in enumerate(order, 1) if cid in c["rel"]), None)
            first_paper = next((i for i, cid in enumerate(order, 1) if cid.split(":")[0] in c["papers"]), None)
            rows.append({"id": c["id"], "type": c["type"], "rank_first": first, "hit3": first is not None and first <= 3, "hit5": first is not None and first <= 5,
                         "hit10": first is not None and first <= 10, "rr": 1 / first if first else 0.0, "paper_hit5": first_paper is not None and first_paper <= 5})
        seconds = (time.perf_counter() - t0) / len(cases)

        def pct(key: str, part: list[dict] = rows) -> float:
            return round(100 * sum(1 for r in part if r[key]) / len(part), 1)

        summary = {"hit@3": pct("hit3"), "hit@5": pct("hit5"), "hit@10": pct("hit10"), "paper_hit@5": pct("paper_hit5"),
                   "MRR": round(statistics.mean(r["rr"] for r in rows), 3), "seconds_per_question": round(seconds, 3)}
        for kind in sorted({r["type"] for r in rows if r["type"]}):
            part = [r for r in rows if r["type"] == kind]
            summary[f"hit@5 {kind} (n={len(part)})"] = pct("hit5", part)
        results[name] = {"summary": summary, "rows": rows}
        print(name, json.dumps(summary), flush=True)

    base = args.models[0]
    for name in args.models[1:]:
        up = sum(a["hit5"] and not b["hit5"] for a, b in zip(results[name]["rows"], results[base]["rows"]))
        down = sum(b["hit5"] and not a["hit5"] for a, b in zip(results[name]["rows"], results[base]["rows"]))
        results[name]["against_first"] = {"base": base, "hit5_gained": up, "hit5_lost": down, "sign_test_p": round(sign_test(up, down), 4)}
        print(f"{name} against {base}: hit@5 gained on {up} questions, lost on {down}, p = {results[name]['against_first']['sign_test_p']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"questions": len(cases), "candidates_per_question": 20, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
