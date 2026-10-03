"""Answer quality, the fourth claim of the evaluation plan ("overall answer quality is not reduced"), scaled to our own corpus.

The architecture names Qasper / SQuAI and RAGAS with an independent judge. Both need other papers ingested and a judge from another
model family than the answer model (only Gemma is installed), so this is a DEPARTURE, listed in docs/evaluation_baselines.md: result-lookup
questions with a known numeric answer ("What F1 does BERT-large get on SQuAD?"), answer correct = a recorded value appears in the answer.
The questions are built from the profile store (silver); with the labellers' job-A verdicts they are built from rows a human judged ALL_OK,
so the answer is human-checked (gold). Compared: the shipped pipeline, the pipeline without stages C, 6 and 7, and the LLM alone.
"""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict

from ..pipeline.conditions import metric_key, norm, split_version
from ..schemas import ConditionProfile

_CITATION = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")          # standalone numbers only: the 1 in "F1" is not one
MAX_VALUES = 3                                                   # a group with more distinct values makes a lucky number too likely
VAGUE_METRICS = {"", "score", "scores", "result", "results"}


def answer_numbers(answer: str) -> list[float]:
    """The numbers an answer states: a citation such as [3] is not a number, 90.9% is 90.9."""
    return [float(m) for m in _NUMBER.findall(_CITATION.sub(" ", answer))]


def contains_value(answer: str, gold: list[float], tol: float = 0.051) -> bool:
    """Does the answer state one of the gold values (up to rounding)?"""
    numbers = answer_numbers(answer)
    return any(abs(n - g) <= tol for n in numbers for g in gold)


def value_recall(answer: str, gold: list[float], tol: float = 0.051) -> float:
    """The share of the expected numbers that the answer states."""
    numbers = answer_numbers(answer)
    return sum(any(abs(n - g) <= tol for n in numbers) for g in gold) / len(gold) if gold else 0.0


def is_correct(answer: str, case: dict) -> bool:
    """mode 'any' (store-built questions: every gold value is an acceptable answer): one of them is stated. Mode 'all' (the labellers'
    expected facts: a good answer states them all): at least half of the expected numbers are stated."""
    if case.get("mode") == "all":
        return value_recall(answer, case["gold_values"]) >= 0.5
    return contains_value(answer, case["gold_values"])


_DECIMAL = re.compile(r"(?<![\w.])\d{1,3}\.\d{1,2}(?!\d)")       # a score such as 58.6; not a paper id (2212.05409) nor a table number (16)


def gold_b_cases(rows: list[dict]) -> list[dict]:
    """Independent test questions: the ones the labellers WROTE FROM THE PAPERS and marked answerable (no scope warning expected), with the decimal
    numbers of their 'expected answer facts' as the gold values (job B's gold file, `score_labels.py B`)."""
    cases = []
    for r in rows:
        if r.get("expect_warning"):
            continue
        values = sorted({float(m) for m in _DECIMAL.findall(r.get("facts") or "")})
        if values:
            cases.append({"id": r["id"], "question": r["question"], "gold_values": values, "mode": "all", "paper": "", "profile_ids": []})
    return cases


def _usable(p: ConditionProfile) -> bool:
    return bool(p.model and p.dataset and p.value is not None and 3 <= len(p.model) <= 28 and 3 <= len(p.dataset) <= 28
                and any(c.isalpha() for c in p.model) and 0 < p.value < 1000 and norm(p.metric or "") not in VAGUE_METRICS)


def lookup_cases(profiles: list[ConditionProfile], n: int, seed: int, verified: set[str] | None = None) -> list[dict]:
    """Result-lookup questions with a known answer. One per (paper, model, dataset, version, language, metric); the gold values are every value
    recorded for that group (a table can hold one model under several settings: any of them answers the question), at most MAX_VALUES distinct
    ones. `verified`: only profiles in this set (the labellers' ALL_OK rows) are used, so the gold value is human-checked."""
    groups: dict[tuple, list[ConditionProfile]] = defaultdict(list)
    for p in profiles:
        if not _usable(p) or (verified is not None and p.profile_id not in verified):
            continue
        name, version = split_version(p.dataset)
        version = p.dataset_version or version
        groups[(p.paper_id, norm(p.model), norm(name), norm(version or ""), norm(p.language or ""), metric_key(p.metric))].append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cases, per_paper = [], defaultdict(int)
    for key in keys:
        members = groups[key]
        values = sorted({round(p.value, 2) for p in members})
        if len(values) > MAX_VALUES or per_paper[key[0]] >= max(3, n // 8):
            continue
        p = members[0]
        name, version = split_version(p.dataset)
        version = p.dataset_version or version
        question = (f"What {p.metric} does {p.model} get on {name}{(' ' + version) if version else ''}"
                    f"{(' for ' + p.language) if p.language else ''}?")
        per_paper[key[0]] += 1
        cases.append({"id": f"Q{len(cases) + 1:02d}", "question": question, "gold_values": values, "paper": p.paper_id,
                      "profile_ids": [m.profile_id for m in members]})
        if len(cases) >= n:
            break
    return cases


def sign_test(only_a: int, only_b: int) -> float:
    """Two-sided exact sign test on the questions where exactly one of two systems was right (p = 0.5 under 'no difference')."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """System A against system B on the same questions."""
    both = sum(a and b for a, b in zip(correct_a, correct_b))
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b))
    return {"both_right": both, "only_a": only_a, "only_b": only_b, "both_wrong": len(correct_a) - both - only_a - only_b,
            "sign_test_p": round(sign_test(only_a, only_b), 4)}


def score_system(rows: list[dict]) -> dict:
    """rows: correct (bool), numbers (how many numbers the answer states), supported (claims supported / checked, None without a critic),
    warned (bool or None), seconds."""
    n = len(rows)
    supported = [r["supported"] for r in rows if r.get("supported") is not None]
    warned = [r["warned"] for r in rows if r.get("warned") is not None]
    return {"questions": n, "correct": sum(r["correct"] for r in rows), "accuracy": round(sum(r["correct"] for r in rows) / n, 4) if n else None,
            "numbers_per_answer": round(sum(r["numbers"] for r in rows) / n, 2) if n else None,
            "claims_supported": round(sum(supported) / len(supported), 4) if supported else None,
            "warned": sum(warned) if warned else None, "seconds_per_question": round(sum(r["seconds"] for r in rows) / n, 2) if n else None}
