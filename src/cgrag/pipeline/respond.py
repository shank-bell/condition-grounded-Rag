"""Stage 10 - Respond (UPGRADED): the JSON the API returns and the UI renders.

Adds applicability {coverage, missing, warning} and contradictions [{verdict, differing, reason}] to the usual
answer + sources, so the UI can show a coverage bar and conflict badges.
"""
from __future__ import annotations

from ..schemas import (
    ApplicabilityResult, ClaimCheck, ContradictionPair, QueryAnalysis, QueryResponse, RetrievedChunk, SourceRef,
)

PREVIEW_CHARS = 280


def source_ref(rc: RetrievedChunk) -> SourceRef:
    c = rc.chunk
    text = " ".join(c.text.split())
    return SourceRef(
        chunk_id=c.chunk_id, paper_id=c.paper_id, paper_title=c.paper_title, page=c.page, section=c.section,
        score=rc.rerank_score if rc.rerank_score is not None else rc.score,
        preview=text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else ""))


def respond(question: str, answer: str, sources: list[RetrievedChunk], analysis: QueryAnalysis | None,
            applicability: ApplicabilityResult | None, contradictions: list[ContradictionPair],
            claim_checks: list[ClaimCheck], regenerated: bool, trace: list[str], timings_ms: dict[str, float],
            retrieval_weak: bool = False, faithfulness: float | None = None) -> QueryResponse:
    return QueryResponse(
        question=question, answer=answer, sources=[source_ref(rc) for rc in sources], analysis=analysis,
        applicability=applicability, contradictions=contradictions, claim_checks=claim_checks,
        regenerated=regenerated, retrieval_weak=retrieval_weak, faithfulness=faithfulness, trace=trace,
        timings_ms={k: round(v, 1) for k, v in timings_ms.items()})
