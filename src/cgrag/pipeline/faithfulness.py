"""RAGAS-style faithfulness and answer relevancy, and the "write it again until it is faithful" loop of the original design.

The two numbers follow the RAGAS definitions and are implemented here (not through the `ragas` package) because a small local judge breaks that
package's output parser, while Ollama's structured output keeps the replies valid:

  faithfulness = supported / all statements. The judge (1) breaks the answer into short statements that make sense on their own and (2) says
                 for each whether it can be inferred from the context (the passages the answer was written from, plus the recorded results
                 that were shown to the writer). Statements that only say what the sources do NOT contain are left out, as the NLI critic does.
  answer relevancy = the mean cosine similarity between the question and N questions that the judge writes for the answer; 0 when the answer
                 is non-committal ("I do not know").

`improve_until_faithful` is the runtime loop (`[features] ragas_loop`, off by default): while the score is below the threshold the answer is
written again, with the unsupported statements as feedback, up to `max_retries` times; the best-scoring text is kept. The judge may be the
answer model (cheap, but the same family as the writer) or a model of another family (`[ragas] judge_model`; for evaluation).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ..llm import OllamaLLM
from .critic import _ECHO, _META
from .text import plain

SYS_STATEMENTS = ("Given a question and an answer, break the answer into short atomic statements. Each statement must make sense on its own "
                  "(replace pronouns by what they refer to) and must not add anything that is not in the answer. Keep numbers, model names, "
                  "data set names and settings exactly as written. Reply only with JSON: {\"statements\": [\"...\"]}.")
SYS_VERDICTS = ("You judge the faithfulness of statements against a context. For each statement return verdict 1 if the statement can be "
                "directly inferred from the context, otherwise 0, and give a one-sentence reason. A number must appear in the context for "
                "the same model, data set and setting. Reply only with JSON: "
                "{\"verdicts\": [{\"statement\": \"...\", \"reason\": \"...\", \"verdict\": 1}]}.")
SYS_QUESTIONS = ("Write {n} different questions that the given answer would be a direct answer to. Also say whether the answer is "
                 "non-committal (it says it does not know, or that the information is not available). Reply only with JSON: "
                 "{{\"questions\": [\"...\"], \"noncommittal\": false}}.")


class _Statements(BaseModel):
    statements: list[str] = Field(default_factory=list)


class _Verdict(BaseModel):
    statement: str = ""
    reason: str = ""
    verdict: int = 0


class _Verdicts(BaseModel):
    verdicts: list[_Verdict] = Field(default_factory=list)


class _Questions(BaseModel):
    questions: list[str] = Field(default_factory=list)
    noncommittal: bool = False


@dataclass
class Faithfulness:
    score: float | None                      # None: the answer has no statement to check (an abstention, or an unusable judge reply)
    statements: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    seconds: float = 0.0
    judge: str = ""


def _checkable(statement: str) -> bool:
    """False for a statement that only says what the sources do not contain, or restates a conflict line (nothing to verify)."""
    return not (_META.search(statement) or _ECHO.search(statement))


def faithfulness(question: str, answer: str, context: str, judge: OllamaLLM, *, max_context_chars: int = 16000) -> Faithfulness:
    t0 = time.perf_counter()
    name = judge.cfg.model
    try:
        parsed, _ = judge.structured(f"Question: {question}\nAnswer: {plain(answer)}", _Statements, system=SYS_STATEMENTS, temperature=0.0,
                                     max_tokens=700)
        statements = [s.strip() for s in parsed.statements if s and s.strip() and _checkable(s)]
        if not statements:
            return Faithfulness(None, [], [], time.perf_counter() - t0, name)
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1))
        verdicts, _ = judge.structured(f"Context:\n{context[:max_context_chars]}\n\nStatements:\n{listing}", _Verdicts, system=SYS_VERDICTS,
                                       temperature=0.0, max_tokens=1500)
    except ValueError:                                                   # the judge did not return valid JSON twice
        return Faithfulness(None, [], [], time.perf_counter() - t0, name)
    if not verdicts.verdicts:
        return Faithfulness(None, statements, [], time.perf_counter() - t0, name)
    unsupported: list[str] = []
    for i, v in enumerate(verdicts.verdicts):
        if v.verdict != 1:
            text = v.statement.strip() or (statements[i] if i < len(statements) else "")
            if text:
                unsupported.append(text)
    supported = sum(1 for v in verdicts.verdicts if v.verdict == 1)
    return Faithfulness(supported / len(verdicts.verdicts), statements, unsupported, time.perf_counter() - t0, name)


def answer_relevancy(question: str, answer: str, judge: OllamaLLM, embed: Callable[[list[str]], np.ndarray], *, n: int = 3) -> float | None:
    """RAGAS answer relevancy: mean cosine between the question and the questions the judge writes for the answer; 0 for a non-committal answer."""
    try:
        parsed, _ = judge.structured(f"Answer: {plain(answer)}", _Questions, system=SYS_QUESTIONS.format(n=n), temperature=0.0, max_tokens=300)
    except ValueError:
        return None
    if parsed.noncommittal:
        return 0.0
    questions = [q for q in parsed.questions if q.strip()][:n]
    if not questions:
        return None
    vectors = embed([question, *questions])                              # rows are L2-normalised, so a dot product is the cosine
    return float(np.clip(vectors[1:] @ vectors[0], 0.0, 1.0).mean())


def improve_until_faithful(question: str, answer: str, context: str, *, judge: OllamaLLM, regenerate: Callable[[str], str], threshold: float,
                           max_retries: int, max_context_chars: int = 16000,
                           notify: Callable[[str], None] | None = None) -> tuple[str, Faithfulness | None, int]:
    """(best answer, its faithfulness, retries used). `regenerate(feedback)` writes the answer again with the unsupported statements as
    feedback. The loop stops when the score reaches the threshold, when there is nothing to check, or after `max_retries` rewrites."""
    say = notify or (lambda _msg: None)
    best_text, best, retries = answer, None, 0
    text = answer
    for attempt in range(max_retries + 1):
        result = faithfulness(question, text, context, judge, max_context_chars=max_context_chars)
        if result.score is None:
            say("no statement to check" if attempt == 0 else "no statement to check in the rewrite")
            break
        say(f"faithfulness {result.score:.2f} ({len(result.statements) - len(result.unsupported)} of {len(result.statements)} statements supported), "
            f"judge {result.judge}")
        if best is None or result.score > (best.score or 0.0):
            best_text, best = text, result
        if result.score >= threshold or attempt == max_retries or not result.unsupported:
            break
        say(f"below {threshold:.2f}: writing the answer again ({attempt + 1} of {max_retries})")
        text = regenerate("\n".join(f"- {s}" for s in result.unsupported))
        retries += 1
    return best_text, best, retries
