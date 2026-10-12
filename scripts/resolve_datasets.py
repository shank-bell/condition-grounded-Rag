"""Name the data set of the tables whose profiles have none (or a task name, a metric label, a cut fragment) from the paper's own text.

  python scripts/resolve_datasets.py                       # the 28-paper store: writes data/index/dataset_context.json (resumes; --force redoes papers)
  CGRAG_OVERLAY=config/bench_metalead.toml python scripts/resolve_datasets.py     # the benchmark store
  python scripts/resolve_datasets.py --papers 1906.05012 --show                    # one paper, print every decision

Needs Ollama (the extractor model, [agents] extractor_model or [llm] model); needs the vector store free when the index has no chunks.jsonl (stop the API).
Afterwards: python scripts/repair_store.py --apply ; python scripts/reindex_cards.py   (the overlay reads the resolutions; see ingestion/dataset_context.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.config import get_settings
from cgrag.ingestion.dataset_context import (compact, groups, load_resolutions, paper_sentences, resolve_paper, save_resolutions)
from cgrag.ingestion.model_context import citation_labels, resolve_models
from cgrag.ingestion.repair import repair_paper
from cgrag.llm import OllamaLLM
from cgrag.stores.profile_store import ProfileStore


def paper_chunks(cfg, paper_ids: list[str]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    chunks: dict[str, dict[str, str]] = defaultdict(dict)
    titles: dict[str, str] = {}
    cj = cfg.paths.index_dir / "chunks.jsonl"
    if cj.exists():
        for line in cj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["paper_id"] in paper_ids:
                    chunks[r["paper_id"]][r["chunk_id"]] = r["text"]
                    titles[r["paper_id"]] = r.get("paper_title", "")
    else:
        from cgrag.stores.vector_store import VectorStore
        vs = VectorStore()
        for pid in paper_ids:
            cs = vs.paper_chunks(pid)
            chunks[pid] = {c.chunk_id: c.text for c in cs}
            titles[pid] = cs[0].paper_title if cs else ""
    return chunks, titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="*", help="paper ids (default: every paper of the store)")
    ap.add_argument("--force", action="store_true", help="redo papers that already have resolutions")
    ap.add_argument("--show", action="store_true", help="print every question's answer")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = get_settings()
    out_path = args.out or cfg.paths.index_dir / "dataset_context.json"
    store = ProfileStore(repaired=False)
    papers = args.papers or sorted({p.paper_id for p in store.all()})
    done = load_resolutions(out_path)
    chunks, titles = paper_chunks(cfg, papers)
    llm = OllamaLLM(model=cfg.agents.extractor_model or None)
    total, kept, t0 = Counter(), Counter(), time.time()
    for n, pid in enumerate(papers, 1):
        if pid in done and not args.force:
            continue
        raw = store.for_paper(pid)
        fixed, _ = repair_paper(raw, chunks.get(pid, {}), title=titles.get(pid, ""))
        todo = groups(fixed)
        shown: list[str] = []

        def progress(key: str, name: str, evidence: str) -> None:
            if args.show:
                shown.append(f"    {key:34.34} -> {name!r:30.30} | {evidence[:90]}")

        found = resolve_paper(llm, fixed, chunks.get(pid, {}), titles.get(pid, ""), progress)
        done[pid] = {k: asdict(v) for k, v in found.items()}
        labels = citation_labels(fixed)
        if labels:
            sentences = paper_sentences(chunks.get(pid, {}))
            whole = compact(" ".join(chunks.get(pid, {}).values()))
            done[pid].update(resolve_models(llm, fixed, sentences, whole, titles.get(pid, ""), progress))
        save_resolutions(out_path, done)
        total["groups"] += len(todo) + len(labels)
        kept["resolved"] += len(done[pid])
        print(f"[{n}/{len(papers)}] {pid}: {len(done[pid])} of {len(todo) + len(labels)} groups resolved  ({time.time() - t0:.0f} s)", flush=True)
        for line in shown:
            print(line)
    print("groups", dict(total), "resolved", dict(kept), "->", out_path)


if __name__ == "__main__":
    main()
