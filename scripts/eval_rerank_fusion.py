"""Does fusing the cross-encoder score with the retrieval rank put the evidence higher? (11 Oct 2026; no LLM, a ranking-only test)

  python scripts/eval_rerank_fusion.py                       # store-made questions (the weight is chosen on them) and the 28 gold questions (the check)
  python scripts/eval_rerank_fusion.py --out eval/labels/rerank_fusion.json

Stage 5 sorts the 20 fused candidates by the cross-encoder score alone. Here each question's 20 candidates are scored once (reranker score and RRF score
of Stage 4), and the orders below are computed from those two numbers, so a variant costs nothing:

  rerank      the cross-encoder score alone (the shipped order)
  rrf         the Stage 4 fused score alone (no cross-encoder)
  z+w         z(rerank) + w * z(rrf), both z-scored inside the 20 candidates of one question, w in 0.25 / 0.5 / 1 / 2
  rank+w      Reciprocal Rank Fusion (k = 60) of the cross-encoder rank and the Stage 4 rank, the Stage 4 rank weighted by w in 0.5 / 1

The grid is fixed before the run; the weight to use is the one that is best on the store-made questions (a bigger set, but circular: the same store defines
"relevant"), and the 28 gold questions are the check, because the evidence pages there were written from the PDFs. Needs Chroma free (stop the API first).
Each variant is compared with `rerank` question by question (sign test on hit@5).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_retrieval import make_cases, make_language_cases, relevant_chunks
from eval_retrieval_gold import evidence_pages

from cgrag.config import get_settings
from cgrag.evaluation.answers import sign_test
from cgrag.labelling.sampling import load_chunk_index
from cgrag.models import rerank_scores
from cgrag.pipeline.rerank import rerank_text
from cgrag.pipeline.retrieval import HybridRetriever
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.profile_store import ProfileStore
from cgrag.stores.vector_store import VectorStore

Z_WEIGHTS = [0.25, 0.5, 1.0, 2.0]
RANK_WEIGHTS = [0.5, 1.0]
RRF_K = 60


def zscores(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) or 1.0
    return [(v - mean) / sd for v in values]


def order_by(scores: list[float], ids: list[str]) -> list[str]:
    return [cid for _, cid in sorted(zip(scores, ids), key=lambda sc: -sc[0])]


def variants(ids: list[str], rerank: list[float], rrf: list[float]) -> dict[str, list[str]]:
    out = {"rerank": order_by(rerank, ids), "rrf": order_by(rrf, ids)}
    zr, zf = zscores(rerank), zscores(rrf)
    for w in Z_WEIGHTS:
        out[f"z+{w}"] = order_by([a + w * b for a, b in zip(zr, zf)], ids)
    rank_r = {cid: i for i, cid in enumerate(out["rerank"], 1)}
    rank_f = {cid: i for i, cid in enumerate(out["rrf"], 1)}
    for w in RANK_WEIGHTS:
        out[f"rank+{w}"] = order_by([1 / (RRF_K + rank_r[cid]) + w / (RRF_K + rank_f[cid]) for cid in ids], ids)
    return out


def measure(order: list[str], rel: set[str], papers: set[str] | None = None) -> dict:
    first = next((i for i, cid in enumerate(order, 1) if cid in rel), None)
    row = {"hit3": first is not None and first <= 3, "hit5": first is not None and first <= 5, "rr": 1 / first if first else 0.0}
    if papers is not None:
        first_paper = next((i for i, cid in enumerate(order, 1) if cid.split(":")[0] in papers), None)
        row["paper_hit5"] = first_paper is not None and first_paper <= 5
    return row


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"questions": n, "hit@3": round(100 * sum(r["hit3"] for r in rows) / n, 1), "hit@5": round(100 * sum(r["hit5"] for r in rows) / n, 1),
           "MRR": round(statistics.fmean(r["rr"] for r in rows), 3)}
    if "paper_hit5" in rows[0]:
        out["paper_hit@5"] = round(100 * sum(r["paper_hit5"] for r in rows) / n, 1)
    return out


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("eval/labels/questions_gold.jsonl"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--lang-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("eval/labels/rerank_fusion.json"))
    args = ap.parse_args()

    profiles = ProfileStore(cfg.paths.profile_db)
    all_profiles = profiles.all()
    retriever = HybridRetriever(VectorStore(cfg.paths.chroma_dir), BM25Store.load(cfg.paths.bm25_path), cfg.retrieval)
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")

    def candidates(question: str, intent: str):
        cands = retriever.retrieve([question], intent)
        scores = rerank_scores(question, [rerank_text(c.chunk, cfg.retrieval.use_cards) for c in cands])
        return [c.chunk.chunk_id for c in cands], list(scores), [c.score for c in cands]

    sets: dict[str, list[dict]] = {"store_made": [], "gold": []}
    for case in make_cases(profiles, args.n, args.seed) + make_language_cases(profiles, args.lang_n, args.seed):
        rel = relevant_chunks(all_profiles, case["conditions"])
        if not rel:
            continue
        ids, rr, rf = candidates(case["question"], "result")
        sets["store_made"].append({"id": case["question"], "family": case.get("family", "model+dataset"), "rel": sorted(rel), "ids": ids, "rerank": rr, "rrf": rf})
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        pages = evidence_pages(q.get("evidence") or "")
        rel = {cid for cid, pg in index.pages.items() if (cid.split(":")[0], pg) in pages}
        if not rel:
            continue
        ids, rr, rf = candidates(q["question"], q.get("intent") or "result")
        sets["gold"].append({"id": q["id"], "family": q.get("planned_type"), "rel": sorted(rel), "papers": sorted({p for p, _ in pages}), "ids": ids, "rerank": rr, "rrf": rf})
    print({k: len(v) for k, v in sets.items()}, flush=True)

    report: dict = {"grid": {"z_weights": Z_WEIGHTS, "rank_weights": RANK_WEIGHTS, "rrf_k": RRF_K}, "sets": {}}
    for name, cases in sets.items():
        per_variant: dict[str, list[dict]] = {}
        for c in cases:
            for v, order in variants(c["ids"], c["rerank"], c["rrf"]).items():
                per_variant.setdefault(v, []).append(measure(order, set(c["rel"]), set(c["papers"]) if "papers" in c else None))
        block = {v: summarise(rows) for v, rows in per_variant.items()}
        base = per_variant["rerank"]
        for v, rows in per_variant.items():
            if v == "rerank":
                continue
            up = sum(a["hit5"] and not b["hit5"] for a, b in zip(rows, base))
            down = sum(b["hit5"] and not a["hit5"] for a, b in zip(rows, base))
            block[v].update({"gained_vs_rerank": up, "lost_vs_rerank": down, "sign_test_p": round(sign_test(up, down), 4)})
        report["sets"][name] = block
        print(f"== {name} ({len(cases)} questions)")
        for v, s in block.items():
            print(f"  {v:10s} {json.dumps(s)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
