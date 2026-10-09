"""Shared data contracts. Every stage reads and writes these types; change them deliberately."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Section = Literal[
    "abstract", "introduction", "related_work", "methods", "experiments",
    "results", "discussion", "conclusion", "references", "other",
]
Intent = Literal["factual", "comparison", "method", "result", "survey"]
Complexity = Literal["simple", "complex"]
Verdict = Literal["GENUINE", "EXPLAINED", "NOT_COMPARABLE"]

# The conditions a finding can depend on. Order matters for display only.
CONDITION_FIELDS = ("task", "dataset", "dataset_version", "language", "model", "model_size", "setting")
# What the question can constrain (metric and value are results, not conditions).
QUERY_CONDITION_FIELDS = CONDITION_FIELDS


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str = ""
    page: int
    section: Section = "other"
    text: str
    card: str = ""            # retrieval card built from the chunk's profiles (ingestion/cards.py); used by retrieval only


class ExtractedProfile(BaseModel):
    """What the LLM returns for one reported result. Kept flat and small for a small model."""
    task: str | None = None
    dataset: str | None = None
    dataset_version: str | None = None
    metric: str | None = None
    value: float | None = None
    language: str | None = None
    model: str | None = None
    model_size: str | None = None
    setting: str | None = None
    evidence: str = Field(default="", description="short verbatim quote containing the number")


class ProfileExtraction(BaseModel):
    profiles: list[ExtractedProfile] = Field(default_factory=list)


class ConditionProfile(ExtractedProfile):
    """A stored profile: one reported result plus where it came from."""
    profile_id: str
    paper_id: str
    chunk_id: str
    methods_chunk_id: str | None = None

    def conditions(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in CONDITION_FIELDS if getattr(self, f)}


class QueryConditions(BaseModel):
    """Conditions the question implies. None means the question doesn't constrain it."""
    task: str | None = None
    dataset: str | None = None
    dataset_version: str | None = None
    language: str | None = None
    model: str | None = None
    model_size: str | None = None
    setting: str | None = None
    # A question can name several systems, benchmarks or languages ("compare mBERT, XLM-R and GPT-4 on XNLI for Hindi and Kannada").
    # The first goes in model / dataset / language, the others here; Stage 6 checks each one. specified() ignores them.
    other_models: list[str] = Field(default_factory=list)
    other_datasets: list[str] = Field(default_factory=list)
    other_languages: list[str] = Field(default_factory=list)

    def specified(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in QUERY_CONDITION_FIELDS if getattr(self, f)}

    def extras(self) -> dict[str, list[str]]:
        """The further models / datasets / languages the question names, by field name."""
        found = {"model": self.other_models, "dataset": self.other_datasets, "language": self.other_languages}
        return {f: values for f, values in found.items() if values}


class QueryAnalysis(BaseModel):
    intent: Intent = "factual"
    complexity: Complexity = "simple"
    conditions: QueryConditions = Field(default_factory=QueryConditions)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rerank_score: float | None = None


class ConditionCheck(BaseModel):
    condition: str
    requested: str
    covered: bool
    observed: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class ApplicabilityResult(BaseModel):
    coverage: float = 1.0          # covered / requested; 1.0 when nothing is requested
    checks: list[ConditionCheck] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    re_retrieved: bool = False
    warning: str | None = None
    reasoning: str = ""            # the agent's short notes on the conditions the profiles could not settle
    profile_guided: bool = False   # chunks were added because their profiles record a missing condition
    joint_covered: bool | None = None   # model + dataset + language recorded TOGETHER by one result? None = not checked (fewer than two named)
    escalated: bool = False        # the larger fallback model gave a second opinion on a "not covered" verdict


class ContradictionPair(BaseModel):
    verdict: Verdict
    chunk_a: str
    chunk_b: str
    paper_a: str
    paper_b: str
    metric: str | None = None
    value_a: float | None = None
    value_b: float | None = None
    differing: list[str] = Field(default_factory=list)
    reason: str = ""
    nli_contradiction: float | None = None


class ClaimCheck(BaseModel):
    sentence: str
    supported: bool
    chunk_id: str | None = None
    entailment: float = 0.0


class SourceRef(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    score: float
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    analysis: QueryAnalysis | None = None
    applicability: ApplicabilityResult | None = None
    contradictions: list[ContradictionPair] = Field(default_factory=list)
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    regenerated: bool = False
    retrieval_weak: bool = False     # Stage 5's own verdict (no passage above the threshold), before Stage 6 may withdraw it: the evaluation's abstention baseline
    faithfulness: float | None = None   # RAGAS-style score of the final answer, only when [features] ragas_loop is on (pipeline/faithfulness.py)
    trace: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
