"""Does Stage 4 (hybrid retrieval) + Stage 5 (rerank + threshold) find the chunks that hold the answer? No LLM needed.

Questions are made from the Condition Profile store: pick a recorded result (model, dataset, maybe language) and ask for it in
one of several wordings. The chunks that count as relevant are the ones with a profile that records ALL of the question's
conditions (model family, dataset, language). So this measures "can the pipeline find what the store says is there" - it is
not a human relevance judgement (labelling job B is), and the profile store comes from the same extractor.

  python scripts/eval_retrieval.py --n 150 --out data/index/eval_retrieval_baseline.json
  python scripts/eval_retrieval.py --n 150 --chroma data/index_exp/chroma --bm25 data/index_exp/bm25.pkl     # an experimental index
Reports hit@20 (a relevant chunk among the 20 fused candidates), hit@kept / hit@5 after the reranker and its threshold, MRR of
the first relevant chunk after reranking, the mean number of chunks kept and how often the evidence is flagged weak.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from cgrag.config import get_settings
from cgrag.models import rerank_scores
from cgrag.pipeline.conditions import observed_values, values_match
from cgrag.pipeline.retrieval import HybridRetriever
from cgrag.pipeline.rerank import MIN_KEEP, rerank_text
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.profile_store import ProfileStore
from cgrag.stores.vector_store import VectorStore

# Questions about things that are NOT in the corpus: a good threshold leaves nothing above it (the answer is flagged "weak evidence").
ABSENT = [
    "What accuracy does ResNet-50 get on ImageNet?", "What F1 does GPT-4 get on HotpotQA?", "How well does Gemini perform on MMMU?",
    "What BLEU does Mistral 7B get on WMT22 English-German?", "What accuracy does Qwen get on C-Eval?", "What is the FID of Stable Diffusion on COCO?",
    "What word error rate does Whisper get on LibriSpeech?", "How does YOLOv8 perform on object detection?", "What is the Elo of AlphaZero in chess?",
    "What accuracy does ViT-L get on CIFAR-10?", "What is the perplexity of Mamba on the Pile?", "How well does Claude 3 do on GPQA?",
    "What is the pass@1 of CodeLlama on HumanEval+?", "What recall does ColBERT get on MS MARCO?", "What accuracy does PaLM 2 get on BIG-bench Hard?",
    "How does DALL-E 3 score on human preference?", "What is the MT-Bench score of Vicuna?", "What F1 does BioBERT get on NCBI-disease?",
    "What accuracy does Wav2Vec 2.0 get on speech emotion?", "What is the MRR of DPR on Natural Questions Open dev set in Korean?",
]
THRESHOLDS = [float(t) for t in range(-9, 8)]          # -9 .. 7: wide enough for the logits of a stronger reranker (bge-reranker-base) too
WORDINGS = [
    "What {metric} does {model} get on {dataset}{lang}?",
    "{model} {dataset}{lang} {metric}",
    "How well does {model} perform on {dataset}{lang}?",
    "Report the {metric} of {model} on {dataset}{lang}.",
    "what is {model}'s {metric} on {dataset}{lang}",
    "Results of {model} on the {dataset} benchmark{lang}",
]


def make_cases(profiles: ProfileStore, n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    everything = [p for p in profiles.all() if p.model and p.dataset and p.metric and len(p.model) >= 3 and len(p.dataset) >= 3
                  and len(p.model) <= 28 and len(p.dataset) <= 28 and p.value is not None]
    # one case per (model family, dataset, language): a spread of papers, not 1,000 rows of one big table
    by_key: dict[tuple, list] = defaultdict(list)
    for p in everything:
        by_key[(p.paper_id, p.model.lower(), p.dataset.lower(), (p.language or "").lower())].append(p)
    keys = list(by_key)
    rnd.shuffle(keys)
    cases, per_paper = [], defaultdict(int)
    for key in keys:
        p = rnd.choice(by_key[key])
        if per_paper[p.paper_id] >= max(3, n // 20):
            continue
        per_paper[p.paper_id] += 1
        with_lang = bool(p.language) and rnd.random() < 0.6
        question = rnd.choice(WORDINGS).format(metric=p.metric, model=p.model, dataset=p.dataset,
                                                lang=f" for {p.language}" if with_lang else "")
        cond = {"model": p.model, "dataset": p.dataset, **({"language": p.language} if with_lang else {})}
        cases.append({"question": question, "conditions": cond, "paper": p.paper_id, "family": "model+dataset"})
        if len(cases) >= n:
            break
    return cases


TASK_SHORT = {"natural language inference": "NLI", "question answering": "QA", "named entity recognition": "NER",
              "machine translation": "machine translation", "sentiment analysis": "sentiment analysis"}
LANG_WORDINGS = [
    "How well do models perform on {language} {task}?",
    "{task} results for {language}",
    "Which models work well for {language} {task}?",
    "What accuracy is reported for {language} {task}?",
    "{language} {task} benchmark scores",
]


def make_language_cases(profiles: ProfileStore, n: int, seed: int) -> list[dict]:
    """Questions that name a language and a task, not the dataset or the table's language code (the Kannada NLI case)."""
    from cgrag.pipeline.conditions import _LANG_ALIASES
    rnd = random.Random(seed + 1)
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for p in profiles.all():
        if p.language and p.task and p.task.lower() in TASK_SHORT:
            lang = _LANG_ALIASES.get(p.language.lower(), p.language)
            if lang.isalpha() and len(lang) >= 4:
                pairs[(lang.capitalize(), p.task.lower())] += 1
    keys = sorted(pairs)
    rnd.shuffle(keys)
    cases = []
    for lang, task in keys[:n]:
        short = TASK_SHORT[task] if rnd.random() < 0.6 else task
        cases.append({"question": rnd.choice(LANG_WORDINGS).format(language=lang, task=short), "conditions": {"language": lang, "task": task},
                      "paper": "", "family": "language+task"})
    return cases


def relevant_chunks(profiles: list, cond: dict[str, str]) -> set[str]:
    out: set[str] = set()
    if not cond:
        return out                                            # an "absent" question: nothing in the corpus is relevant
    for p in profiles:
        if all(any(values_match(f, w, v) for v in observed_values(p, f)) for f, w in cond.items()):
            out.add(p.chunk_id)
    return out


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lang-n", type=int, default=40, help="how many language + task questions (the table says 'kn', the question says Kannada)")
    ap.add_argument("--chroma", type=Path, default=cfg.paths.chroma_dir)
    ap.add_argument("--bm25", type=Path, default=cfg.paths.bm25_path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cases", type=Path, default=None, help="reuse the cases of an earlier run (same questions for a fair comparison)")
    args = ap.parse_args()

    profiles = ProfileStore(cfg.paths.profile_db)
    all_profiles = profiles.all()
    if args.cases and args.cases.exists():
        cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    else:
        cases = make_cases(profiles, args.n, args.seed) + make_language_cases(profiles, args.lang_n, args.seed)
    if not any(c.get("family") == "absent" for c in cases):
        cases += [{"question": q, "conditions": {}, "paper": "", "family": "absent"} for q in ABSENT]
    retriever = HybridRetriever(VectorStore(args.chroma), BM25Store.load(args.bm25), cfg.retrieval)

    rows = []
    for case in cases:
        rel = relevant_chunks(all_profiles, case["conditions"])
        cands = retriever.retrieve([case["question"]], "result")
        scores = rerank_scores(case["question"], [rerank_text(c.chunk, cfg.retrieval.use_cards) for c in cands])
        ranked = sorted(zip(cands, scores), key=lambda cs: cs[1], reverse=True)
        kept = [c.chunk.chunk_id for c, s in ranked if s >= cfg.retrieval.rerank_threshold][: cfg.retrieval.rerank_keep]
        weak = not kept
        if weak:
            kept = [c.chunk.chunk_id for c, _ in ranked[:MIN_KEEP]]
        order = [c.chunk.chunk_id for c, _ in ranked]
        sweep = {}
        for t in THRESHOLDS:
            above = [c.chunk.chunk_id for c, s in ranked if s >= t][: cfg.retrieval.rerank_keep]
            sweep[str(t)] = (bool(rel & set(above)), len(above))
        first = next((i for i, cid in enumerate(order, 1) if cid in rel), None)
        rows.append({"question": case["question"], "family": case.get("family", "model+dataset"), "n_relevant": len(rel), "hit20": bool(rel & {c.chunk.chunk_id for c in cands}),
                     "recall20": len(rel & {c.chunk.chunk_id for c in cands}) / len(rel) if rel else 0.0,
                     "hit_kept": bool(rel & set(kept)) and not weak, "hit5": bool(rel & set(order[:5])),
                     "rr": 1 / first if first else 0.0, "kept": 0 if weak else len(kept), "weak": weak,
                     "sweep": sweep})
    absent_rows = [r for r in rows if r["family"] == "absent"]
    rows = [r for r in rows if r["n_relevant"] > 0]

    def summarise(part: list[dict]) -> dict:
        n = len(part)

        def pct(key: str) -> float:
            return round(100 * sum(1 for r in part if r[key]) / n, 1)

        return {"questions": n, "hit@20": pct("hit20"), "recall@20": round(100 * statistics.mean(r["recall20"] for r in part), 1),
                "hit@kept(rerank threshold)": pct("hit_kept"), "hit@5(rerank)": pct("hit5"),
                "MRR(rerank order)": round(statistics.mean(r["rr"] for r in part), 3),
                "mean_kept": round(statistics.mean(r["kept"] for r in part), 2), "flagged_weak_%": pct("weak")}

    summary = {"all": summarise(rows)}
    for fam in sorted({r["family"] for r in rows}):
        summary[fam] = summarise([r for r in rows if r["family"] == fam])
    # the reranker's threshold: what each value would keep (hit = a relevant chunk is kept; kept = chunks kept on average;
    # none = questions where nothing clears it and the pipeline falls back to its top 3 flagged "weak evidence")
    if absent_rows:
        summary["absent_questions"] = {"n": len(absent_rows), "flagged_weak_%_by_threshold": {
            t: round(100 * sum(1 for r in absent_rows if r["sweep"][t][1] == 0) / len(absent_rows), 1) for t in map(str, THRESHOLDS)}}
    summary["threshold_sweep"] = {
        t: {"hit_%": round(100 * sum(r["sweep"][t][0] for r in rows) / len(rows), 1),
            "mean_kept": round(statistics.mean(r["sweep"][t][1] for r in rows), 2),
            "none_kept_%": round(100 * sum(1 for r in rows if r["sweep"][t][1] == 0) / len(rows), 1)}
        for t in map(str, THRESHOLDS)}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "cases": cases, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
