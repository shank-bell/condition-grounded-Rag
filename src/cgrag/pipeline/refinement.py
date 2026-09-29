"""Stage 3 - Query Refinement: rewrite into a technical search query; decompose multi-part questions."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import OllamaLLM
from ..schemas import QueryAnalysis
from .conditions import norm

MAX_SUBQUESTIONS = 3

SYSTEM = (
    "You prepare a question about computer-science research papers for a search engine over paper text. "
    "rewritten: a precise technical query that keeps every model, dataset, version, language and number named "
    "in the question. "
    "sub_questions: if the question has several parts, split it into at most 3 self-contained sub-questions "
    "that each keep the conditions they depend on; otherwise an empty list. Do not answer the question."
)


class Refinement(BaseModel):
    rewritten: str
    sub_questions: list[str] = Field(default_factory=list, max_length=MAX_SUBQUESTIONS)


def refine(question: str, analysis: QueryAnalysis, llm: OllamaLLM, *, decompose: bool = True) -> list[str]:
    """Search queries for retrieval: the rewrite first, then any sub-questions. Never returns an empty list."""
    try:
        out, _ = llm.structured(f"Question: {question}", Refinement, system=SYSTEM, max_tokens=400, retries=0)
    except ValueError:
        return [question]
    queries = [out.rewritten.strip() or question]
    if decompose:
        queries += [s.strip() for s in out.sub_questions if s.strip()]
    # a rewrite must not lose a condition the question named: append any that went missing
    joined = norm(" ".join(queries))
    lost = [v for v in analysis.conditions.specified().values() if norm(v) and norm(v) not in joined]
    if lost:
        queries[0] = f"{queries[0]} {' '.join(lost)}"
    return queries
