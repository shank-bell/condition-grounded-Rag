"""Make labelled questions to train stage 1's SciBERT classifier (no human labels exist yet).

For a sample of passages from the indexed papers the LLM is asked to WRITE questions of a stated intent and complexity, so
every label is known by construction. A small hand-written seed set is added.

  python scripts/make_query_training_data.py --out data/index/query_training_v1.jsonl
  python scripts/make_query_training_data.py --out data/index/query_training_v2.jsonl --styles informal,terse,multipart \
      --kinds comparison:complex,result:complex,result:simple --chunks-per-paper 3 --per-call 4 --seed 21 --no-seeds

Passages are read from data/index/chunks.jsonl (exported from ChromaDB on first use), so this script can run while another
process holds the index. Questions equal to one in eval/stage1_dev.jsonl are never used for training.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from cgrag.config import ROOT, get_settings
from cgrag.llm import OllamaLLM

# (intent, complexity, what the question looks like). Comparison and survey questions are always complex.
KINDS = [
    ("factual", "simple", "asks for one fact (a number, a name, a definition) that a single passage states"),
    ("factual", "complex", "asks for several separate facts in one question"),
    ("comparison", "complex", "compares two or more systems, datasets, languages, model sizes or settings"),
    ("method", "simple", "asks how one technique, model or training procedure works"),
    ("method", "complex", "asks about a technique and also about why it was designed that way or how it differs from another"),
    ("result", "simple", "asks for one reported score of a named model on a named dataset, with its conditions"),
    ("result", "complex", "asks for reported scores of several models, datasets, languages or settings"),
    ("survey", "complex", "asks for an overview of an approach, a family of models, or a research topic"),
]
SECTIONS = ("introduction", "methods", "results", "experiments", "related_work", "discussion")

# wording styles; "multipart" only makes sense for complex questions (a simple question asks for one thing)
STYLES = {
    "plain": "",
    "informal": " Write them the way a busy student would type into a search box: short, casual, sometimes all lower-case "
                "and without a question mark.",
    "terse": " Write them as terse keyword-style queries of 4 to 10 words.",
    "multipart": " Write them as long questions with two or three clauses joined by 'and'.",
}

SEEDS = [
    ("What batch size is used to fine-tune BERT?", "factual", "simple"),
    ("How many parameters does BERT-large have?", "factual", "simple"),
    ("Does BERT work well for Kannada question answering?", "result", "simple"),
    ("What F1 does BERT-large get on SQuAD v2.0?", "result", "simple"),
    ("How does the masked language model objective work?", "method", "simple"),
    ("Compare BERT and RoBERTa on GLUE and explain what training changes account for the gap.", "comparison", "complex"),
    ("Do papers agree on BERT-large accuracy on MNLI?", "comparison", "complex"),
    ("Give an overview of multilingual pre-trained language models.", "survey", "complex"),
    ("What are the main approaches to making BERT smaller and faster?", "survey", "complex"),
    ("Which languages does XLM-R cover and how does it perform on XNLI zero-shot versus translate-train?", "result", "complex"),
    ("What is the training corpus of RoBERTa and how large is it?", "factual", "complex"),
    ("How does ELECTRA's replaced token detection differ from masked language modelling, and why is it more efficient?", "method", "complex"),
    ("Is DeBERTa better than RoBERTa on SuperGLUE?", "comparison", "complex"),
    ("What accuracy does mBERT reach on Hindi XNLI?", "result", "simple"),
    ("Explain how dense passage retrieval is trained.", "method", "simple"),
]


class Generated(BaseModel):
    questions: list[str]


def load_chunks(path: Path) -> list[dict]:
    """Passages as dicts; exported from ChromaDB once so later runs do not touch the index."""
    if not path.exists():
        from cgrag.labelling.sampling import export_chunks
        export_chunks(path)
        print(f"exported the chunks to {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=cfg.paths.index_dir / "query_training.jsonl")
    ap.add_argument("--chunks-file", type=Path, default=cfg.paths.index_dir / "chunks.jsonl")
    ap.add_argument("--chunks-per-paper", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--styles", default="plain,plain,plain,informal", help="comma list, repeated names weight the choice")
    ap.add_argument("--kinds", default="all", help="all, or intent:complexity pairs such as comparison:complex,result:simple")
    ap.add_argument("--no-seeds", action="store_true")
    ap.add_argument("--dev", type=Path, default=ROOT / "eval" / "stage1_dev.jsonl")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    llm = OllamaLLM()

    kinds = KINDS if args.kinds == "all" else [k for k in KINDS if f"{k[0]}:{k[1]}" in args.kinds.split(",")]
    styles = [s for s in args.styles.split(",") if s in STYLES]
    dev_keys = set()
    if args.dev.exists():
        dev_keys = {" ".join(json.loads(ln)["question"].lower().split()) for ln in args.dev.read_text(encoding="utf-8").splitlines() if ln.strip()}

    by_paper: dict[str, list[dict]] = {}
    for c in load_chunks(args.chunks_file):
        if c["section"] in SECTIONS and len(c["text"]) > 600:
            by_paper.setdefault(c["paper_id"], []).append(c)
    jobs = []
    for chunks in by_paper.values():
        for chunk in rnd.sample(chunks, min(args.chunks_per_paper, len(chunks))):
            for intent, complexity, how in kinds:
                style = rnd.choice([s for s in styles if not (s == "multipart" and complexity == "simple")] or ["plain"])
                jobs.append((chunk, intent, complexity, how, style))
    print(f"{len(by_paper)} papers, {len(jobs)} generation calls, llm={llm.cfg.model}, kinds={len(kinds)}, styles={Counter(j[4] for j in jobs)}")

    def write(job):
        chunk, intent, complexity, how, style = job
        prompt = (
            f"Passage from a research paper:\n{chunk['text'][:1800]}\n\n"
            f"Write {args.per_call} different questions that a researcher might ask a research assistant covering many papers, "
            f"where each question {how}. Each question must stand on its own: never say 'the passage', 'the paper', 'the text' "
            f"or 'the authors'; name concrete models, datasets, languages or settings instead. Vary the wording.{STYLES[style]}"
        )
        try:
            out, _ = llm.structured(prompt, Generated, temperature=0.8, max_tokens=500, retries=0)
        except ValueError:
            return []
        return [{"question": q.strip(), "intent": intent, "complexity": complexity, "source": chunk["chunk_id"], "style": style}
                for q in out.questions if 12 <= len(q.strip()) <= 300]

    rows = [] if args.no_seeds else [{"question": q, "intent": i, "complexity": c, "source": "seed", "style": "plain"} for q, i, c in SEEDS]
    with ThreadPoolExecutor(max_workers=max(1, cfg.llm.parallel)) as pool:
        for batch in pool.map(write, jobs):
            rows += batch
    seen, unique = set(dev_keys), []
    for r in rows:
        key = " ".join(r["question"].lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in unique), encoding="utf-8")
    print(f"wrote {len(unique)} questions to {args.out}")
    print(sorted(Counter((r["intent"], r["complexity"]) for r in unique).items()))
    print(dict(Counter(r["style"] for r in unique)))


if __name__ == "__main__":
    main()
