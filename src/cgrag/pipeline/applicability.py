"""Stage 6 - Applicability Agent (NEW, AGENT / investigator): does the evidence cover the conditions in the question?

An LLM agent working with tools:
  1. TOOL - profile lookup: for every condition the question names (language, dataset, model, size ...), code looks for a
     value recorded in the Condition Profiles of the retrieved passages. A recorded match is a fact and settles the condition.
  2. AGENT - judgement: conditions the profiles cannot settle (nothing recorded, or only a text mention such as "Kannada" in
     a list of pre-training languages) are judged by the LLM from short evidence snippets. Its "covered" verdict is accepted
     only if it is grounded: the value it cites must occur in a passage it cites. Model, dataset version and model size are
     strict fields the LLM can never override - "RoBERTa" does not cover "BERT", v1.1 does not cover v2.0.
  3. AGENT - action: if a condition is still missing, the agent writes a targeted search query and searches again (once);
     if it is still missing, the answer carries a scope warning that names it.
When the LLM is switched off (`[features] applicability_agent = false`) or its output is unusable, the deterministic
matcher decides alone (including the passage text as a fallback for unrecorded fields).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from pydantic import BaseModel, Field, create_model

from ..llm import OllamaLLM
from ..schemas import ApplicabilityResult, ConditionCheck, ConditionProfile, RetrievedChunk
from ..stores.profile_store import ProfileStore
from .conditions import norm, observed_values, values_match

MAX_OBSERVED = 6
MAX_VALUES_PER_FIELD = 4          # recorded values shown per field and passage in the LLM's evidence
MAX_VALUES_FOR_JUDGED_FIELD = 12  # ... but the fields under judgement show more so the LLM can see e.g. 'hi', 'sw'
SNIPPET_CHARS = 80
STRICT_FIELDS = {"model", "dataset_version", "model_size"}    # decided by recorded values only

SYSTEM = (
    "You are the investigator of a pipeline that answers questions about research papers. A question names experimental "
    "conditions. A profile lookup already settled some of them; for each condition under TO JUDGE, decide from the "
    "EVIDENCE whether at least one passage covers it.\n"
    "covered=true only if a passage RECORDS a value that means the same thing (for example 'hi' means Hindi) or its text "
    "states that an experiment was run under that condition. A passage that only mentions the word without evaluating "
    "anything (a list of pre-training languages, future work, related work, 'not evaluated on ...') does NOT cover it.\n"
    "Cite chunk_ids, put the exact recorded value or quoted words in matched, and give one short sentence in note. "
    "If a condition is not covered, write search_query: a short query that would find papers evaluated under the missing "
    "condition, keeping the model, dataset and task words of the question. Return JSON."
)


class Verdict(BaseModel):
    condition: str
    covered: bool = False
    chunk_ids: list[str] = Field(default_factory=list)
    matched: str = ""
    note: str = ""


class Judgement(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)
    search_query: str = ""


@lru_cache(maxsize=8)
def _judgement_model(n: int) -> type[Judgement]:
    """Judgement whose verdict list must hold exactly one verdict per condition under judgement (a small model
    otherwise tends to skip the verdicts and only write the search query)."""
    return create_model("Judgement", __base__=Judgement, verdicts=(list[Verdict], Field(min_length=n, max_length=n)))


@dataclass
class Evidence:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    recorded: dict[str, list[str]]        # field -> distinct recorded values (the passage's own profiles, else its paper's)
    paper_level: bool


class ApplicabilityAgent:
    def __init__(self, profiles: ProfileStore, max_reretrieve: int = 1, llm: OllamaLLM | None = None,
                 fallback: OllamaLLM | None = None) -> None:
        self.profiles = profiles
        self.max_reretrieve = max_reretrieve
        self.llm = llm
        self.fallback = fallback if llm is not None else None    # the bigger model that gives a second opinion (escalation)

    # ---------- tool: profile lookup ----------

    def _evidence(self, chunks: list[RetrievedChunk], fields: list[str]) -> list[Evidence]:
        by_chunk = self.profiles.for_chunks([c.chunk.chunk_id for c in chunks])
        by_paper: dict[str, list[ConditionProfile]] = {}
        out: list[Evidence] = []
        for rc in chunks:
            own = by_chunk.get(rc.chunk.chunk_id)
            paper_level = not own
            if paper_level:                  # a passage without profiles (intro, methods) speaks for its paper's results
                pid = rc.chunk.paper_id
                if pid not in by_paper:
                    by_paper[pid] = self.profiles.for_paper(pid)
                own = by_paper[pid]
            recorded = {f: list(dict.fromkeys(v for p in own for v in observed_values(p, f))) for f in fields}
            out.append(Evidence(rc.chunk.chunk_id, rc.chunk.paper_id, rc.chunk.section, rc.chunk.text, recorded, paper_level))
        return out

    @staticmethod
    def _lookup(requested: dict[str, str], evidence: list[Evidence]) -> dict[str, list[str]]:
        """Per condition: the passages whose recorded values satisfy it."""
        return {f: [e.chunk_id for e in evidence if any(values_match(f, wanted, v) for v in e.recorded.get(f, []))]
                for f, wanted in requested.items()}

    # ---------- agent: judgement ----------

    def _digest(self, todo: dict[str, str], evidence: list[Evidence]) -> str:
        lines = []
        for e in evidence:
            recorded = "; ".join(
                f"{f}: {', '.join(vals[:MAX_VALUES_FOR_JUDGED_FIELD if f in todo else MAX_VALUES_PER_FIELD])}"
                for f, vals in e.recorded.items() if vals) or "nothing recorded"
            mentions = []
            for f, wanted in todo.items():
                m = re.search(re.escape(wanted), e.text, re.I) if len(wanted) >= 3 else None
                if m:
                    lo, hi = max(0, m.start() - SNIPPET_CHARS), min(len(e.text), m.end() + SNIPPET_CHARS)
                    mentions.append(f'{f}={wanted} -> "...{" ".join(e.text[lo:hi].split())}..."')
            line = f"[{e.chunk_id}] paper {e.paper_id}, {e.section}, recorded{' (whole paper)' if e.paper_level else ''}: {recorded}"
            lines.append(line + (" | text mentions: " + " || ".join(mentions) if mentions else ""))
        return "\n".join(lines)

    def _judge(self, question: str, todo: dict[str, str], settled: dict[str, str], evidence: list[Evidence],
               llm: OllamaLLM | None = None) -> Judgement | None:
        prompt = (f"QUESTION: {question}\n"
                  f"SETTLED BY THE PROFILE LOOKUP: {', '.join(f'{f}={v}' for f, v in settled.items()) or 'nothing'}\n"
                  f"TO JUDGE: {', '.join(f'{f}={v}' for f, v in todo.items())}\n"
                  f"EVIDENCE:\n{self._digest(todo, evidence)}")
        try:
            out, _ = (llm or self.llm).structured(prompt, _judgement_model(len(todo)), system=SYSTEM, max_tokens=500, retries=0)
        except ValueError:
            return None
        return out

    @staticmethod
    def _grounded(field: str, verdict: Verdict, by_id: dict[str, Evidence]) -> list[str]:
        """Passages that back a 'covered' verdict: the value it cites really occurs in a passage it cites. Strict fields
        are never decided by the LLM."""
        matched = norm(verdict.matched)
        if field in STRICT_FIELDS or not verdict.covered or len(matched) < 3:
            return []
        return [cid for cid in verdict.chunk_ids
                if cid in by_id and matched in norm(" ".join(by_id[cid].recorded.get(field, [])) + " " + by_id[cid].text)]

    # ---------- the check ----------

    def _assess(self, requested: dict[str, str], chunks: list[RetrievedChunk], question: str,
                llm: OllamaLLM | None = None) -> tuple[ApplicabilityResult, str]:
        """Coverage of each requested condition by these passages, and the agent's search query for what is missing.
        `llm` overrides the agent's own model (the escalation's second opinion)."""
        evidence = self._evidence(chunks, list(requested))
        by_id = {e.chunk_id: e for e in evidence}
        covering = self._lookup(requested, evidence)
        todo = {f: v for f, v in requested.items() if not covering[f]}
        query, notes = "", []
        judged = self._judge(question, todo, {f: v for f, v in requested.items() if f not in todo}, evidence, llm) \
            if todo and (llm or self.llm) is not None else None
        for field, wanted in todo.items():
            # the LLM may write the condition as "language" or as "language=Kannada": compare the field name only
            verdict = next((v for v in (judged.verdicts if judged else [])
                            if norm(v.condition.split("=")[0]) == norm(field)), None)
            if judged is None:                   # no agent or unusable output: the passage text is the fallback evidence
                if len(norm(wanted)) >= 4:
                    covering[field] = [e.chunk_id for e in evidence if not e.recorded.get(field) and norm(wanted) in norm(e.text)]
            else:
                if verdict:
                    covering[field] = self._grounded(field, verdict, by_id)
                    if verdict.note:
                        notes.append(f"{field}={wanted}: {verdict.note.strip()}")
        if judged and judged.search_query.strip():
            query = judged.search_query.strip()
        checks = []
        for field, wanted in requested.items():
            seen: Counter[str] = Counter(v for e in evidence for v in e.recorded.get(field, []))
            checks.append(ConditionCheck(condition=field, requested=wanted, covered=bool(covering[field]),
                                         observed=[v for v, _ in seen.most_common(MAX_OBSERVED)], chunk_ids=covering[field]))
        covered = sum(c.covered for c in checks)
        missing = [f"{c.condition}={c.requested}" for c in checks if not c.covered]
        result = ApplicabilityResult(
            coverage=covered / len(checks) if checks else 1.0, checks=checks, missing=missing,
            warning=self._warning(checks) if missing else None, reasoning=" ".join(notes))
        return result, query

    def check(self, requested: dict[str, str], chunks: list[RetrievedChunk], question: str = "") -> ApplicabilityResult:
        return self._assess(requested, chunks, question)[0]

    @staticmethod
    def _warning(checks: list[ConditionCheck]) -> str:
        covered = sum(c.covered for c in checks)
        parts = [f"The evidence covers {covered} of {len(checks)} conditions in the question."]
        for c in checks:
            if c.covered:
                continue
            seen = f" The retrieved sources record: {', '.join(c.observed)}." if c.observed else " No retrieved source states this condition."
            parts.append(f"No retrieved source reports {c.condition.replace('_', ' ')} = {c.requested}.{seen}")
        parts.append("Do not assume the findings hold for the missing condition.")
        return " ".join(parts)

    def run(self, requested: dict[str, str], kept: list[RetrievedChunk],
            research: Callable[..., list[RetrievedChunk]], question: str = "") -> tuple[list[RetrievedChunk], ApplicabilityResult]:
        """Check coverage; while conditions are missing and retries remain, search again for the missing ones.

        `research(terms[, query])` runs retrieval + rerank with the missing condition values added (and the agent's own
        search query, when it wrote one) and returns the new kept set.
        """
        result, query = self._assess(requested, kept, question)
        retried = False
        for _ in range(self.max_reretrieve):
            if not result.missing:
                break
            terms = [c.requested for c in result.checks if not c.covered]
            kept = research(terms, query) if query else research(terms)
            result, query = self._assess(requested, kept, question)
            retried = True
        escalated = False
        if result.missing and self.fallback is not None:
            # Escalation: the small model says "not covered" (or its output was unusable). Before a scope warning is
            # shown, the bigger model judges the same evidence once; its verdict replaces the small model's.
            result, _ = self._assess(requested, kept, question, llm=self.fallback)
            escalated = True
        result.re_retrieved = retried
        result.escalated = escalated
        result.profile_guided = bool(getattr(research, "guided_ids", None))
        return kept, result
