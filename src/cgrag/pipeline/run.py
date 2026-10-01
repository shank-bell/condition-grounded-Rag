"""The online path: one question through stages 1-10.

  1 Query Understanding -> 2 Orchestrator -> 3 Query Refinement -> 4 Hybrid Retrieval -> 5 Rerank + Filter
  -> 6 Applicability Agent (may loop back to 4) -> 7 Contradiction Resolver -> 8 Generate
  -> 9 Claim-Check Critic (may loop back to 8) -> 10 Respond
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from functools import lru_cache

from ..config import Settings, get_settings
from ..llm import OllamaLLM
from ..models import rerank_scores
from ..schemas import ApplicabilityResult, ClaimCheck, ContradictionPair, QueryResponse, RetrievedChunk
from ..stores.bm25_store import BM25Store
from ..stores.profile_store import ProfileStore
from ..stores.vector_store import VectorStore
from .applicability import ApplicabilityAgent
from .contradiction import ContradictionResolver
from .critic import ClaimChecker
from .generate import generate, select_sources
from .orchestrator import OrchestratorAgent
from .query_understanding import QueryUnderstanding
from .refinement import refine
from .rerank import rerank_filter
from .respond import respond
from .retrieval import HybridRetriever

WEAK_EVIDENCE = "No retrieved passage is clearly relevant to the question, so the sources below may not answer it."
EMPTY_INDEX = "No papers are indexed yet, so there is nothing to answer from. Add papers first."
TARGETED_SLOTS = 2        # chunks found by the targeted re-search that are always kept
PROFILE_GUIDED_POOL = 6   # chunks per missing condition that the profile store proposes ...
PROFILE_GUIDED_SLOTS = 2  # ... of which the best two (by the reranker) are added


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = (time.perf_counter() - t0) * 1000


class Pipeline:
    def __init__(self, settings: Settings | None = None, llm: OllamaLLM | None = None) -> None:
        self.cfg = settings or get_settings()
        self.llm = llm or OllamaLLM(self.cfg)
        self.vectors = VectorStore(self.cfg.paths.chroma_dir)
        self.profiles = ProfileStore(self.cfg.paths.profile_db)
        self.retriever = HybridRetriever(self.vectors, BM25Store.load(self.cfg.paths.bm25_path), self.cfg.retrieval)
        self.understand = QueryUnderstanding(self._agent_llm(self.cfg.agents.understanding_model), profiles=self.profiles)
        self.refine_llm = self._agent_llm(self.cfg.agents.refinement_model)
        orchestrator_llm = self._agent_llm(self.cfg.agents.orchestrator_model)
        applicability_llm = self._agent_llm(self.cfg.agents.applicability_model) if self.cfg.features.applicability_agent else None
        self.fallback_llm = self._agent_llm(self.cfg.agents.fallback_model) if self.cfg.features.escalation else None
        self.orchestrator = OrchestratorAgent(orchestrator_llm, self.cfg.features, self._escalation_target(orchestrator_llm))
        self.applicability = ApplicabilityAgent(
            self.profiles, self.cfg.applicability.max_reretrieve, llm=applicability_llm,
            fallback=self._escalation_target(applicability_llm))
        self.contradictions = ContradictionResolver(self.profiles, self.cfg.contradiction)
        self.critic = ClaimChecker(self.cfg.critic)

    def _agent_llm(self, model: str) -> OllamaLLM:
        """The LLM an agent uses: the shared one, or its own open-weights model when the config names one."""
        if not model or model == self.llm.cfg.model:
            return self.llm
        return OllamaLLM(self.cfg, model=model)

    def _escalation_target(self, agent_llm: OllamaLLM | None) -> OllamaLLM | None:
        """The bigger model an agent hands a bad outcome to; None when escalation is off or it would be the same model."""
        if self.fallback_llm is None or agent_llm is None or agent_llm.cfg.model == self.fallback_llm.cfg.model:
            return None
        return self.fallback_llm

    def reload_indexes(self) -> None:
        """Pick up papers added since start-up (the keyword index is a file rebuilt after ingestion)."""
        self.retriever.bm25 = BM25Store.load(self.cfg.paths.bm25_path)

    def run(self, question: str, history: list[dict] | None = None) -> QueryResponse:
        t, trace = _Timer(), []
        t0 = time.perf_counter()
        if self.vectors.count() == 0:
            return respond(question, EMPTY_INDEX, [], None, None, [], [], False, ["index is empty"], {})

        with t("1_understand"):
            analysis = self.understand.analyze(question)
        with t("2_plan"):
            plan = self.orchestrator.plan(question, analysis, self.profiles.paper_count())
        trace += [f"1 {plan.notes[0]}; conditions={analysis.conditions.specified() or 'none'}", *[f"2 {n}" for n in plan.notes[1:]]]

        queries = [question]
        if plan.refine:
            with t("3_refine"):
                queries = refine(question, analysis, self.refine_llm, decompose=plan.decompose)
            trace.append(f"3 queries: {queries}")

        with t("4_retrieve"):
            candidates = self.retriever.retrieve(queries, analysis.intent)
        with t("5_rerank"):
            kept, weak = rerank_filter(question, candidates, self.cfg.retrieval)
        trace.append(f"4 retrieved {len(candidates)} -> 5 kept {len(kept)}" + (" (weak evidence)" if weak else ""))

        with t("2b_replan"):                     # the agent looks at what came back and may re-plan once
            review = self.orchestrator.review(question, plan, n_kept=len(kept), weak=weak)
        if review.action == "refine_and_retry":
            queries = refine(question, analysis, self.refine_llm, decompose=False)
            candidates = self.retriever.retrieve(queries, analysis.intent)
            kept, weak = rerank_filter(question, candidates, self.cfg.retrieval)
            trace.append(f"2 re-plan ({review.reason or 'weak evidence'}): rewrote the question {queries} and searched again "
                         f"-> kept {len(kept)}" + (" (still weak)" if weak else ""))

        applicability: ApplicabilityResult | None = None
        if plan.check_applicability:
            with t("6_applicability"):
                kept, applicability = self.applicability.run(
                    analysis.conditions.specified(), kept,
                    self._research(question, queries, analysis.intent, kept, analysis.conditions.specified()), question)
            trace.append(f"6 coverage {applicability.coverage:.2f}, missing {applicability.missing or 'none'}"
                         + (", re-retrieved" if applicability.re_retrieved else "")
                         + (", profile-guided" if applicability.profile_guided else "")
                         + (f", second opinion by {self.fallback_llm.cfg.model}" if applicability.escalated else "")
                         + (f" | agent: {applicability.reasoning}" if applicability.reasoning else ""))
        elif self.cfg.features.applicability:
            applicability = ApplicabilityResult()

        contradictions: list[ContradictionPair] = []
        if plan.check_contradictions and len(kept) >= 2:
            with t("7_contradiction"):
                contradictions = self.contradictions.resolve(question, kept, analysis.conditions.specified())
            trace.append("7 conflicts: " + (", ".join(c.verdict for c in contradictions) or "none"))

        sources = select_sources(kept, contradictions, self.cfg.retrieval.generate_top,
                                 applicability.checks if applicability else None)
        profile_map = self.profiles.for_chunks([rc.chunk.chunk_id for rc in sources])
        warning = " ".join(w for w in (applicability.warning if applicability else None, WEAK_EVIDENCE if weak else None) if w) or None
        include = self.cfg.features.profile_in_context

        with t("8_generate"):
            answer = generate(question, sources, self.llm, warning=warning, conflicts=contradictions, history=history,
                              profiles=profile_map, include_profiles=include, requested=analysis.conditions.specified())

        checks: list[ClaimCheck] = []
        regenerated = False
        if plan.check_claims:
            with t("9_critic"):
                checks = self.critic.check(answer, sources, profile_map)
                bad = self.critic.unsupported(checks)
                if bad and self.cfg.critic.max_regenerate > 0:
                    retry = generate(question, sources, self.llm, warning=warning, conflicts=contradictions, history=history,
                                     profiles=profile_map, include_profiles=include, requested=analysis.conditions.specified(),
                                     feedback="\n".join(f"- {c.sentence}" for c in bad))
                    retry_checks = self.critic.check(retry, sources, profile_map)
                    regenerated = True
                    if len(self.critic.unsupported(retry_checks)) <= len(bad):
                        answer, checks = retry, retry_checks
                answer, checks = self.critic.repair_citations(answer, checks, sources)
            trace.append(f"9 claims checked {len(checks)}, unsupported {len(self.critic.unsupported(checks))}"
                         + (", regenerated once" if regenerated else ""))

        t.ms["total"] = (time.perf_counter() - t0) * 1000
        return respond(question, answer, sources, analysis, applicability, contradictions, checks, regenerated, trace, t.ms)

    def _research(self, question: str, queries: list[str], intent: str, kept: list[RetrievedChunk],
                  requested: dict[str, str] | None = None):
        """Stage 6's re-retrieval: search again with the missing condition values added, keep the best new chunks.
        With `[features] profile_guided_retrieval` the profile store is also asked which chunks record the missing
        condition (e.g. language = Kannada together with the question's task), and those are added too."""
        def research(terms: list[str], query: str = "") -> list[RetrievedChunk]:
            extra = " ".join(terms)
            found = self.retriever.retrieve([f"{queries[0]} {extra}", extra, *([query] if query else [])], intent)
            fresh, _ = rerank_filter(question, [rc for rc in found], self.cfg.retrieval)
            have = {rc.chunk.chunk_id for rc in kept}
            guided = self._profile_guided(question, requested or {}, terms, have)
            research.guided_ids = [rc.chunk.chunk_id for rc in guided]
            taken = have | set(research.guided_ids)
            merged = list(kept) + guided + [rc for rc in fresh[:TARGETED_SLOTS] if rc.chunk.chunk_id not in taken]
            return sorted(merged, key=lambda rc: rc.rerank_score if rc.rerank_score is not None else rc.score, reverse=True)
        research.guided_ids = []
        return research

    def _profile_guided(self, question: str, requested: dict[str, str], terms: list[str], have: set[str]) -> list[RetrievedChunk]:
        """Chunks the profile store says record a missing condition, scored against the question (not thresholded:
        the profile is the evidence that the passage is on the condition)."""
        if not self.cfg.features.profile_guided_retrieval or not requested:
            return []
        ids: list[str] = []
        fields = [f for f, wanted in requested.items() if wanted in terms]
        for field in fields:
            ids += [cid for cid in self.profiles.chunks_recording(requested, field, PROFILE_GUIDED_POOL)
                    if cid not in have and cid not in ids]
        chunks = self.vectors.get(ids) if ids else []
        if not chunks:
            return []
        scores = rerank_scores(question, [c.text for c in chunks])          # the question settles ties between candidates
        best = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)[:PROFILE_GUIDED_SLOTS * len(fields)]
        return [RetrievedChunk(chunk=c, score=0.0, rerank_score=float(s)) for c, s in best]


@lru_cache(maxsize=1)
def default_pipeline() -> Pipeline:
    return Pipeline()


def run(question: str, history: list[dict] | None = None) -> QueryResponse:
    """The single entry point: question in, QueryResponse out."""
    return default_pipeline().run(question, history)
