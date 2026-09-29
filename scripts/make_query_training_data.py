"""Make labelled questions to train stage 1's SciBERT classifier (no human labels exist yet).

For a sample of passages from the indexed papers the LLM is asked to WRITE questions of a stated intent and complexity, so
every label is known by construction. A small hand-written seed set is added. Output: data/index/query_training.jsonl

  python scripts/make_query_training_data.py --chunks-per-paper 4 --per-call 3
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from cgrag.config import get_settings
from cgrag.llm import OllamaLLM
from cgrag.stores.vector_store import VectorStore

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-per-paper", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = get_settings()
    rnd = random.Random(args.seed)
    llm = OllamaLLM()
    store = VectorStore()

    jobs = []
    by_paper: dict[str, list] = {}
    for c in store.all_chunks():
        if c.section in SECTIONS and len(c.text) > 600:
            by_paper.setdefault(c.paper_id, []).append(c)
    for chunks in by_paper.values():
        for chunk in rnd.sample(chunks, min(args.chunks_per_paper, len(chunks))):
            for intent, complexity, how in KINDS:
                jobs.append((chunk, intent, complexity, how))
    print(f"{len(by_paper)} papers, {len(jobs)} generation calls, llm={cfg.llm.model}")

    def write(job):
        chunk, intent, complexity, how = job
        prompt = (
            f"Passage from a research paper:\n{chunk.text[:1800]}\n\n"
            f"Write {args.per_call} different questions that a researcher might ask a research assistant covering many papers, "
            f"where each question {how}. Each question must stand on its own: never say 'the passage', 'the paper', 'the text' "
            f"or 'the authors'; name concrete models, datasets, languages or settings instead. Vary the wording."
        )
        try:
            out, _ = llm.structured(prompt, Generated, temperature=0.8, max_tokens=400, retries=0)
        except ValueError:
            return []
        return [{"question": q.strip(), "intent": intent, "complexity": complexity, "source": chunk.chunk_id}
                for q in out.questions if 15 <= len(q.strip()) <= 300]

    rows = [{"question": q, "intent": i, "complexity": c, "source": "seed"} for q, i, c in SEEDS]
    with ThreadPoolExecutor(max_workers=max(1, cfg.llm.parallel)) as pool:
        for batch in pool.map(write, jobs):
            rows += batch
    seen, unique = set(), []
    for r in rows:
        key = " ".join(r["question"].lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    out_path = cfg.paths.index_dir / "query_training.jsonl"
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in unique), encoding="utf-8")
    from collections import Counter
    print(f"wrote {len(unique)} questions to {out_path}")
    print(Counter((r["intent"], r["complexity"]) for r in unique))


if __name__ == "__main__":
    main()
