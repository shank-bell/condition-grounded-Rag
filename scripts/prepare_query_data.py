"""Merge question files, blind-relabel them with the LLM, keep the questions whose label the LLM confirms, and split.

  python scripts/prepare_query_data.py --inputs a.jsonl b.jsonl --out-dir data/index/query_runs/iter1/data --relabel

Why relabel: the LLM that WRITES a question "of intent X, complexity Y" is not always right about it. A second, blind pass
(the LLM sees only the question and a rubric) classifies every question; a question is "clean" when both passes agree on
both heads. Templates and the hand-written seeds are clean by construction. Training uses clean questions only, and the
validation set is a held-out slice of clean questions (val_clean) plus a slice of everything else (val_noisy, diagnostics).
Questions equal to a dev question (eval/stage1_dev.jsonl) are dropped. Blind labels are cached, so re-runs only pay for new
questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cgrag.config import ROOT, get_settings
from cgrag.llm import OllamaLLM

RUBRIC = (
    "Classify a question that a researcher asks a research assistant about computer-science papers.\n"
    "intent - exactly one of:\n"
    "- factual: asks for a stated fact (a number, a name, a definition) about a model, dataset or experimental setup. "
    "It is NOT a benchmark score.\n"
    "- method: asks how a technique, model or training procedure works, or why it was designed that way.\n"
    "- result: asks for reported performance scores of models on datasets or tasks.\n"
    "- comparison: asks to compare or contrast two or more systems, datasets or settings: which is better, how they differ, "
    "or whether papers agree.\n"
    "- survey: asks for an overview or summary of an approach, a family of models or a research topic.\n"
    "complexity - exactly one of:\n"
    "- simple: asks for exactly one thing (one fact, one score, one procedure).\n"
    "- complex: asks for several things at once, lists or aggregates (all languages, every task, different sizes, several "
    "models), compares things, or asks for an overview. Every comparison and every survey question is complex.\n"
    "Return JSON with intent and complexity."
)


class Label(BaseModel):
    intent: Literal["factual", "method", "result", "comparison", "survey"]
    complexity: Literal["simple", "complex"]


def key(question: str) -> str:
    return " ".join(question.lower().split())


def digest(question: str) -> str:
    return hashlib.sha1(key(question).encode("utf-8")).hexdigest()[:16]


PREFIXES = ["please tell me ", "quick question: ", "hey, ", "can you tell me ", "i want to know ", "so, ", "question - "]


def perturb(question: str, rnd: random.Random) -> str:
    """The way a real user types: lower case, no question mark, a typo, a chatty prefix. The label does not change."""
    q = question
    roll = rnd.random()
    if roll < 0.35:
        q = q.lower().rstrip("?").strip()
    elif roll < 0.7:
        words = q.split()
        idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
        if idx:
            i = rnd.choice(idx)
            j = rnd.randrange(1, len(words[i]) - 1)
            w = words[i]
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]                 # two neighbouring letters swapped
            q = " ".join(words)
    else:
        q = rnd.choice(PREFIXES) + (q[0].lower() + q[1:] if q[:2].isalpha() and not q[:2].isupper() else q)
    return q


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--relabel", action="store_true", help="blind LLM pass over questions not clean by construction")
    ap.add_argument("--relabel-all", action="store_true", help="also relabel templates and seeds (diagnostic; they stay clean)")
    ap.add_argument("--augment", type=float, default=0.0,
                    help="add a perturbed copy (typo / lower case / chatty prefix) of this fraction of the TRAIN questions")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--cache", type=Path, default=cfg.paths.index_dir / "query_runs" / "relabel_cache.jsonl")
    ap.add_argument("--dev", type=Path, default=ROOT / "eval" / "stage1_dev.jsonl")
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    dev_keys = {key(json.loads(ln)["question"]) for ln in args.dev.read_text(encoding="utf-8").splitlines() if ln.strip()}
    rows, seen = [], set(dev_keys)
    for path in args.inputs:
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            k = key(r["question"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    print(f"{len(rows)} unique questions from {len(args.inputs)} files (dev questions removed)")

    cache: dict[str, dict] = {}
    if args.cache.exists():
        for ln in args.cache.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                cache[d["h"]] = d
    by_construction = lambda r: r.get("source") in ("template", "seed")          # noqa: E731
    todo = [r for r in rows if (args.relabel and (args.relabel_all or not by_construction(r))) and digest(r["question"]) not in cache]
    if todo:
        llm = OllamaLLM()
        print(f"blind relabel of {len(todo)} questions with {llm.cfg.model} ({len(cache)} cached)")

        def judge(r):
            try:
                out, _ = llm.structured(f"Question: {r['question']}", Label, system=RUBRIC, temperature=0.0, max_tokens=40, retries=0)
                return {"h": digest(r["question"]), "intent": out.intent, "complexity": out.complexity}
            except Exception:                                 # unusable JSON or a timed-out request: the question stays unlabelled
                return None

        args.cache.parent.mkdir(parents=True, exist_ok=True)
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, cfg.llm.parallel)) as pool, args.cache.open("a", encoding="utf-8") as fh:
            for d in pool.map(judge, todo):
                done += 1
                if d:
                    cache[d["h"]] = d
                    fh.write(json.dumps(d) + "\n")
                if done % 200 == 0:                           # progress + durable cache: a re-run resumes where this stopped
                    fh.flush()
                    print(f"  relabelled {done}/{len(todo)}", flush=True)

    agree = defaultdict(lambda: [0, 0, 0, 0])                 # per source-kind: n, intent agree, complexity agree, both
    for r in rows:
        c = cache.get(digest(r["question"]))
        r["clean"] = True if by_construction(r) else False
        if c:
            ai, ac = c["intent"] == r["intent"], c["complexity"] == r["complexity"]
            r["llm_intent"], r["llm_complexity"] = c["intent"], c["complexity"]
            if not by_construction(r):
                r["clean"] = ai and ac
            a = agree[(r["intent"], r["complexity"], "template" if by_construction(r) else "llm")]
            a[0] += 1; a[1] += ai; a[2] += ac; a[3] += ai and ac                # noqa: E702
        elif not by_construction(r) and not args.relabel:
            r["clean"] = True                                   # no relabel requested: trust the writer

    clean = [r for r in rows if r["clean"]]
    noisy = [r for r in rows if not r["clean"]]

    def split(items: list[dict], frac: float) -> tuple[list[dict], list[dict]]:
        by: dict[tuple, list] = defaultdict(list)
        for r in items:
            by[(r["intent"], r["complexity"])].append(r)
        train, val = [], []
        for group in by.values():
            rnd.shuffle(group)
            n_val = max(1, int(len(group) * frac)) if len(group) >= 8 else 0
            val += group[:n_val]
            train += group[n_val:]
        rnd.shuffle(train)
        return train, val

    train, val_clean = split(clean, args.val_frac)
    _, val_noisy = split(noisy, args.val_frac) if noisy else ([], [])
    if args.augment > 0:                                       # only the training side: the validation sets stay as written
        copies = []
        for r in rnd.sample(train, int(len(train) * args.augment)):
            q = perturb(r["question"], rnd)
            if key(q) not in seen and q != r["question"]:
                seen.add(key(q))
                copies.append({**r, "question": q, "source": f"augment:{r.get('source', 'llm')}"})
        train += copies
        rnd.shuffle(train)
        print(f"augmented {len(copies)} train questions (typos / lower case / chatty prefixes)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val_clean", val_clean), ("val_noisy", val_noisy)):
        (args.out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data), encoding="utf-8")
    stats = {
        "questions": len(rows), "clean": len(clean), "noisy": len(noisy), "train": len(train), "val_clean": len(val_clean), "val_noisy": len(val_noisy),
        "train_by_class": {f"{i}/{c}": n for (i, c), n in sorted(Counter((r["intent"], r["complexity"]) for r in train).items())},
        "train_by_source": dict(Counter(r["source"].split(":")[0] if r.get("source", "").split(":")[0] in ("template", "seed", "augment") else "llm"
                                        for r in train)),
        "agreement": {f"{i}/{c}/{s}": {"n": a[0], "intent": round(a[1] / a[0], 3), "complexity": round(a[2] / a[0], 3), "both": round(a[3] / a[0], 3)}
                      for (i, c, s), a in sorted(agree.items())},
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "agreement"}, indent=1))
    print("agreement of the blind pass with the writer's label (both heads):")
    for k, v in stats["agreement"].items():
        print(f"  {k:<32} n={v['n']:<5} intent {v['intent']:.2f}  complexity {v['complexity']:.2f}  both {v['both']:.2f}")


if __name__ == "__main__":
    main()
