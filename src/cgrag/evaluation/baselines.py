"""Baselines for the architecture's evaluation plan ("How each claim will be tested").

They take the same inputs as the stage they are compared with and need no labels, so their outputs can be produced and frozen
before anyone's labels are known (nothing is tuned to the answer key).

  Stage 6, answers that go beyond the evidence:  plain RAG (never warns); abstention when retrieval is weak (Stage 5's own flag);
                                                 a Sufficient-Context autorater (an LLM judges whether the retrieved passages are
                                                 enough to answer the question as asked)
  Stage 7, genuine vs explained conflicts:       plain NLI on the two results; an LLM prompt built on the DRAGged-into-Conflicts taxonomy

The LLM baselines run on the same local model as the answer: the system may not call external APIs. The published Sufficient-Context
autorater used a far larger model, so this one is probably weaker than the published one; the paper must say so. How each baseline's
output maps onto the three classes of the labelling (GENUINE / EXPLAINED / NOT_COMPARABLE) is written next to the baseline.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..labelling.score import prf

PASSAGE_CHARS = 1800        # per retrieved passage shown to the autorater (five of them fit the context window)


# ------------------------------------------------------------------ Stage 6 ----

class Sufficiency(BaseModel):
    sufficient: bool
    reason: str = ""


SUFFICIENT_SYSTEM = (
    "You are an expert evaluator of retrieval-augmented question answering over research papers. You get a QUESTION and a CONTEXT: "
    "the passages retrieved for it. Decide whether the CONTEXT is sufficient to answer the QUESTION exactly as asked, using nothing "
    "but the CONTEXT. It is sufficient only if a definitive answer can be inferred from the CONTEXT for every model, dataset, "
    "dataset version, language and setting that the question names. If the CONTEXT only covers other models, datasets, languages or "
    "settings, or only related ones, it is NOT sufficient.\n"
    'Return JSON: {"sufficient": true or false, "reason": "one short sentence"}')


def sufficient_context(llm, question: str, passages: list[str]) -> tuple[bool, str]:
    """The Sufficient-Context autorater (Joren et al., 2024) with a local model: are the retrieved passages enough to answer the question
    as asked? Returns (sufficient, reason). An unusable reply counts as sufficient (the baseline stays silent) and the reason says so."""
    context = "\n\n".join(f"[{i}] {p[:PASSAGE_CHARS]}" for i, p in enumerate(passages, start=1)) or "(no passage was retrieved)"
    try:
        out, _ = llm.structured(f"QUESTION: {question}\n\nCONTEXT:\n{context}", Sufficiency, system=SUFFICIENT_SYSTEM,
                                max_tokens=150, temperature=0.0)
    except ValueError as err:
        return True, f"unusable reply: {err}"[:200]
    return out.sufficient, out.reason


def score_warnings(rows: list[dict]) -> dict:
    """Score a system's warn / do-not-warn decisions. A row: expect_warning (bool, the truth), warned (bool), expected_missing (the condition
    names the truth says are missing) and named_missing (the condition names the system reports, None for a system that only abstains).

    precision / recall / F1 of "warns" (positive = the question should get a warning); extrapolation rate = the share of should-warn
    questions that were answered with no warning at all, i.e. delivered as if the evidence covered them (= 1 - recall); false-warning rate =
    the share of should-not-warn questions that got a warning; right_condition_named = of the correct warnings, how many name a condition
    the truth names (only for systems that name one)."""
    tp = sum(r["warned"] and r["expect_warning"] for r in rows)
    fp = sum(r["warned"] and not r["expect_warning"] for r in rows)
    fn = sum((not r["warned"]) and r["expect_warning"] for r in rows)
    tn = sum((not r["warned"]) and not r["expect_warning"] for r in rows)
    correct = [r for r in rows if r["warned"] and r["expect_warning"]]
    names = all(r.get("named_missing") is not None for r in correct) and bool(correct)
    return {**prf(tp, fp, fn), "tn": tn, "questions": len(rows), "warnings_given": tp + fp, "should_warn": tp + fn, "should_not_warn": fp + tn,
            "accuracy": round((tp + tn) / len(rows), 4) if rows else None,
            "extrapolation_rate": round(fn / (tp + fn), 4) if tp + fn else None,
            "false_warning_rate": round(fp / (fp + tn), 4) if fp + tn else None,
            "right_condition_named": sum(bool(set(r["expected_missing"]) & set(r["named_missing"])) for r in correct) if names else None}


# ------------------------------------------------------------------ Stage 7 ----

# Plain NLI: the two results as statements, both directions, probabilities averaged. Contradiction -> GENUINE; neutral or entailment ->
# NOT_COMPARABLE. A plain NLI detector has no notion of a difference that conditions explain, so it never predicts EXPLAINED: that is what
# this baseline is there to show.

def plain_nli_verdict(nli, text_a: str, text_b: str) -> dict:
    (e1, n1, c1), (e2, n2, c2) = nli.probs([(text_a, text_b), (text_b, text_a)])
    e, n, c = (e1 + e2) / 2, (n1 + n2) / 2, (c1 + c2) / 2
    return {"system_verdict": "GENUINE" if c > max(e, n) else "NOT_COMPARABLE", "system_differing": [],
            "p_contradiction": round(float(c), 4), "p_neutral": round(float(n), 4), "p_entailment": round(float(e), 4)}


# LLM prompt on the taxonomy of "DRAGged into Conflicts" (Cattan et al., 2025). Mapping: complementary information -> EXPLAINED;
# conflicting research outcomes, outdated information, misinformation -> GENUINE (the same measurement differs; for this baseline that
# is a genuine disagreement, whatever its cause); no conflict (or an unrecognised answer) -> NOT_COMPARABLE.

class ConflictJudgement(BaseModel):
    category: str
    differing_conditions: list[str] = Field(default_factory=list)
    reason: str = ""


TAXONOMY_SYSTEM = (
    "Two research papers report different numbers for what looks like the same measurement. For each result you get the table caption, "
    "the header and the row (or the passage) the number was read from, with the number marked [[like this]]. Judge from that text only.\n"
    "Classify the relation between the two results with this taxonomy of conflicts between sources (DRAGged into Conflicts):\n"
    "- no conflict: the two numbers do not measure the same thing, or the text does not say enough to compare them.\n"
    "- complementary information: both are right and differ because they were obtained under different conditions: another dataset "
    "version or split, model size, language, or training / evaluation setting.\n"
    "- conflicting research outcomes: the same measurement under the same conditions gives different numbers in the two papers.\n"
    "- outdated information: the same measurement, but one number comes from an older version of the model, data or method.\n"
    "- misinformation: one of the numbers looks like an error (a misread table, a typo).\n"
    'Return JSON: {"category": one of the five names above, "differing_conditions": for complementary information the conditions that '
    'differ, taken from [dataset_version, model_size, language, setting, other], otherwise [], "reason": "one short sentence"}')

_DIFFERING_WORDS = (("dataset_version", ("version", "split", "dataset")), ("model_size", ("size", "parameter")), ("language", ("language",)),
                    ("setting", ("setting", "train", "evaluat", "zero", "few", "fine", "dev", "test", "protocol")))


def taxonomy_verdict(category: str) -> str:
    c = " ".join(re.sub(r"[^a-z ]", " ", category.lower().replace("_", " ")).split())
    if "no conflict" in c:
        return "NOT_COMPARABLE"
    if "complement" in c:
        return "EXPLAINED"
    if "outdated" in c or "misinformation" in c or "conflict" in c:
        return "GENUINE"
    return "NOT_COMPARABLE"


def differing_names(items: list[str]) -> list[str]:
    """The model's condition names in the vocabulary of Stage 7 (an unknown one is 'other', which the scorer understands)."""
    out: list[str] = []
    for item in items:
        t = item.lower().replace("_", " ")
        name = next((n for n, words in _DIFFERING_WORDS if any(w in t for w in words)), "other")
        if name not in out:
            out.append(name)
    return out


def llm_conflict_verdict(llm, header: str, evidence_a: str, evidence_b: str) -> dict:
    prompt = f"{header}\n\nRESULT A\n{evidence_a}\n\nRESULT B\n{evidence_b}"
    try:
        out, _ = llm.structured(prompt, ConflictJudgement, system=TAXONOMY_SYSTEM, max_tokens=200, temperature=0.0)
    except ValueError as err:
        return {"system_verdict": "NOT_COMPARABLE", "system_differing": [], "category": None, "reason": f"unusable reply: {err}"[:200]}
    verdict = taxonomy_verdict(out.category)
    return {"system_verdict": verdict, "system_differing": differing_names(out.differing_conditions) if verdict == "EXPLAINED" else [],
            "category": out.category, "reason": out.reason}
