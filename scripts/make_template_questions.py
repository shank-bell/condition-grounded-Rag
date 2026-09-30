"""Template questions for the Stage 1 classifier: labels are correct by construction (no LLM, no noise).

Model, dataset, language, metric and setting names are taken from the Condition Profile store so the questions name things
that really are in the corpus. Every template belongs to one (intent, complexity) class of the rubric in
make_query_training_data.KINDS; wording variants cover formal, informal and keyword-style phrasing.

  python scripts/make_template_questions.py --out data/index/query_runs/iter1/template.jsonl --per-class 260 --seed 5
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

from cgrag.config import get_settings

METRICS = ["F1", "EM", "accuracy", "BLEU", "exact match", "Spearman correlation", "perplexity", "ROUGE-L", "top-20 accuracy"]
SETTINGS = ["zero-shot", "few-shot", "fine-tuned", "translate-train", "dev", "test"]
BENCHMARKS = ["GLUE", "SuperGLUE", "XTREME", "IndicXTREME", "MMLU", "XNLI"]
TECHNIQUES = [
    "masked language modeling", "next sentence prediction", "replaced token detection", "sliding window attention",
    "knowledge distillation", "cross-layer parameter sharing", "factorized embedding parameterization", "disentangled attention",
    "retrieval-augmented generation", "dense passage retrieval", "translate-train fine-tuning", "contrastive learning",
    "siamese sentence encoders", "span corruption", "reinforcement learning from human feedback", "context distillation",
    "translation language modeling", "instruction tuning", "in-context learning", "back-translation", "SentencePiece tokenization",
    "multi-task fine-tuning", "global attention", "layer-wise learning rate decay",
]
TOPICS = [
    "pre-trained language models for Indian languages", "BERT compression", "cross-lingual transfer", "multilingual pre-training",
    "long-document transformers", "retrieval-augmented question answering", "pre-training objectives", "safety evaluation of language models",
    "model scaling", "sentence embeddings", "low-resource language modelling", "few-shot learning with large language models",
    "efficient transformers", "question answering datasets", "bias evaluation in language models", "evaluation of multilingual models",
    "distillation of language models", "human evaluation of chat models", "benchmarks for natural language understanding",
    "transfer learning for NLP", "reading comprehension benchmarks", "machine translation with pre-trained models",
]
CLEAN = re.compile(r"^[A-Za-z][A-Za-z0-9+\-\. ]{1,24}$")
GENERIC = {"baseline", "baseline model", "ours", "our model", "our", "model", "models", "random", "majority", "average", "mean",
           "single model", "ensemble", "base", "large", "small", "human", "sota", "previous sota", "best", "this work"}

# (intent, complexity) -> templates. Slots: {model} {model_b} {model_c} {dataset} {dataset_b} {version} {metric} {language}
# {language_b} {setting} {benchmark} {technique} {technique_b} {topic}
BANK: dict[tuple[str, str], list[str]] = {
    ("result", "simple"): [
        "What {metric} does {model} get on {dataset}?", "What is {model}'s {metric} on {dataset}?", "How well does {model} do on {dataset}?",
        "What {metric} does {model} reach on {dataset} in the {setting} setting?", "How does {model} perform on {dataset} for {language}?",
        "What accuracy does {model} achieve on {language} {dataset}?", "what did {model} score on {dataset}", "{model} {dataset} {metric}",
        "{model} {metric} {dataset} {version}", "Report {model}'s {metric} on {dataset}.", "What was the {metric} of {model} on {dataset} {version}?",
        "Does {model} work well on {dataset}?", "How good is {model} at {dataset}?", "what is {model} {metric} on {dataset}",
        "{metric} of {model} on {language} {dataset}", "How well does {model} handle {language}?",
    ],
    ("result", "complex"): [
        "How does {model} perform on {dataset} across all languages?", "What are the {dataset} scores of {model}, {model_b} and {model_c}?",
        "What {metric} do the different sizes of {model} get on {dataset} in the zero-shot and fine-tuned settings?",
        "List {model}'s results on every task of {benchmark}.", "What are the results of {model} and {model_b} on {dataset} and {dataset_b}?",
        "What {metric} does {model} get on {dataset} for {language} and {language_b}?", "Give the {metric} of {model} on each language of {dataset}.",
        "what do the different {model} sizes get on {dataset}, {dataset_b} and {benchmark}", "What are {model}'s scores on all the {benchmark} tasks?",
        "Report {model}'s {metric} on {dataset} for dev and test in the zero-shot and fine-tuned settings.",
        "How does {model} do on {dataset} in {language}, {language_b} and the other languages?",
    ],
    ("comparison", "complex"): [
        "How does {model} compare with {model_b} on {dataset}?", "Is {model} better than {model_b} on {dataset}?", "{model} vs {model_b}",
        "{model} vs {model_b} on {dataset}", "Which is better for {language}, {model} or {model_b}?", "Do the papers agree on {model}'s {metric} on {dataset}?",
        "Compare {model}, {model_b} and {model_c} on {dataset}.", "Which performs better on {dataset}: {model} or {model_b}?",
        "How much does {model} improve over {model_b} on {dataset}?", "What is the difference between {model} and {model_b}?",
        "{model} or {model_b} for {dataset}", "Do different papers report the same {metric} for {model} on {dataset}?",
        "Is {model} more accurate than {model_b} in the {setting} setting?", "{model} versus {model_b} in {language}",
    ],
    ("factual", "simple"): [
        "How many parameters does {model} have?", "What is the vocabulary size of {model}?", "Which optimizer was used to train {model}?",
        "What batch size does {model} use?", "how many params in {model}", "What is the maximum sequence length of {model}?",
        "Who proposed {dataset}?", "How many layers does {model} have?", "What is the hidden size of {model}?", "what learning rate is used for {model}",
        "How large is the {dataset} dataset?", "What does {dataset} stand for?", "{model} parameters", "How many languages does {model} cover?",
    ],
    ("factual", "complex"): [
        "What are the layer count, hidden size and parameter count of each {model} variant?", "Which corpora was {model} pre-trained on and how large is each?",
        "What pre-training data, batch size and number of steps were used for {model}?", "Which languages does {model} cover and how many tokens does each have?",
        "What are the sizes of the train, dev and test sets of {dataset} and how many languages does it cover?",
        "What batch size, learning rate and optimizer did {model} use?", "which datasets make up {benchmark} and what does each one test",
        "What are the number of layers, attention heads and parameters of {model} and {model_b}?",
    ],
    ("method", "simple"): [
        "How does {technique} work in {model}?", "Explain how {model} is trained.", "How is {model} pre-trained?", "Explain {technique}.",
        "how does {technique} work", "Explain how {technique} is used in {model}.", "how does {model} work", "How does {model} use {technique}?",
        "Describe how {model} is fine-tuned on {dataset}.",
    ],
    ("method", "complex"): [
        "Why does {model} use {technique}, and how does that differ from {model_b}?", "How does {technique} work, and why does it improve results?",
        "Why did the authors of {model} choose {technique}, and what are the trade-offs?", "How does {technique} differ from {technique_b}, and why is it more efficient?",
        "How does {model} apply {technique}, and why does that help on {dataset}?", "Why is {technique} used, and how does it change training compared with {technique_b}?",
    ],
    ("survey", "complex"): [
        "Give me an overview of {topic}.", "What are the main approaches to {topic}?", "Summarize what these papers say about {topic}.",
        "What techniques exist for {topic}?", "overview of {topic}", "What is known about {topic}?", "What are the main limitations of {topic} discussed across the papers?",
        "Survey of {topic}", "what do we know about {topic}", "Summarize the evidence on {topic}.",
    ],
}


def load_cases(db_path: Path) -> list[dict]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    cases = []
    for r in db.execute("SELECT model, dataset, dataset_version, metric, language, setting FROM profiles "
                        "WHERE model IS NOT NULL AND dataset IS NOT NULL"):
        if (CLEAN.match(r["model"]) and CLEAN.match(r["dataset"]) and len(r["model"]) >= 3 and len(r["dataset"]) >= 3
                and r["model"].lower() not in GENERIC and r["dataset"].lower() not in GENERIC):
            cases.append(dict(r))
    return cases


def render(template: str, slots: dict, rnd: random.Random) -> str:
    text = template.format(**slots)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?.,])", r"\1", text)                 # an empty {version} slot leaves "SWAG ?"
    if rnd.random() < 0.22:
        text = text.lower()
    if text.endswith("?") and rnd.random() < 0.25:
        text = text[:-1]
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=260)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list of intent:complexity to generate, default all")
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    cases = load_cases(get_settings().paths.profile_db)
    models = sorted({c["model"] for c in cases})
    by_dataset: dict[str, list[dict]] = {}
    for c in cases:
        by_dataset.setdefault(c["dataset"], []).append(c)
    languages = sorted({c["language"] for c in cases if c["language"] and re.fullmatch(r"[A-Za-z]{3,15}", c["language"])})
    only = {tuple(x.split(":")) for x in args.only.split(",") if x}
    print(f"{len(cases)} cases, {len(models)} models, {len(by_dataset)} datasets, {len(languages)} languages")

    def slots() -> dict:
        c = rnd.choice(cases)
        same = [x for x in by_dataset[c["dataset"]] if x["model"] != c["model"]]
        other = rnd.choice(same) if same else rnd.choice(cases)
        third = rnd.choice(same) if same else rnd.choice(cases)
        other_ds = rnd.choice(cases)["dataset"]
        version = c["dataset_version"] or ""
        return {
            "model": c["model"], "model_b": other["model"], "model_c": third["model"], "dataset": c["dataset"], "dataset_b": other_ds,
            "version": ("v" + version) if version and rnd.random() < 0.7 else "", "metric": rnd.choice(METRICS),
            "language": c["language"] if c["language"] and re.fullmatch(r"[A-Za-z]{3,15}", c["language"]) else rnd.choice(languages),
            "language_b": rnd.choice(languages), "setting": rnd.choice(SETTINGS), "benchmark": rnd.choice(BENCHMARKS),
            "technique": rnd.choice(TECHNIQUES), "technique_b": rnd.choice(TECHNIQUES), "topic": rnd.choice(TOPICS),
        }

    rows, seen = [], set()
    for (intent, complexity), templates in BANK.items():
        if only and (intent, complexity) not in only:
            continue
        made, tries = 0, 0
        while made < args.per_class and tries < args.per_class * 20:
            tries += 1
            q = render(rnd.choice(templates), slots(), rnd)
            key = q.lower()
            if key in seen or len(q) < 8:
                continue
            seen.add(key)
            rows.append({"question": q, "intent": intent, "complexity": complexity, "source": "template", "style": "template"})
            made += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} template questions to {args.out}")
    for r in rnd.sample(rows, 8):
        print("  ", r["intent"], r["complexity"], "|", r["question"])


if __name__ == "__main__":
    main()
