"""The online path: one question through stages 1-10.

  1 Query Understanding -> 2 Orchestrator -> 3 Query Refinement -> 4 Hybrid Retrieval -> 5 Rerank + Filter
  -> 6 Applicability Agent (may loop back to 4) -> 7 Contradiction Resolver -> 8 Generate
  -> 9 Claim-Check Critic (may loop back to 8) -> 10 Respond
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache

from ..config import Settings, get_settings
from ..llm import OllamaLLM
from ..models import embed, get_nli, rerank_scores
from ..schemas import ApplicabilityResult, ClaimCheck, ContradictionPair, QueryResponse, RetrievedChunk
from ..stores.bm25_store import BM25Store
from ..stores.profile_store import ProfileStore
from ..stores.vector_store import VectorStore
from .applicability import ApplicabilityAgent
from .contradiction import ContradictionResolver
from .conditions import claim_text
from .critic import ClaimChecker
from .faithfulness import improve_until_faithful
from .generate import generate, select_sources
from .orchestrator import OrchestratorAgent
from .query_understanding import QueryUnderstanding
from .refinement import refine
from .rerank import rerank_filter, rerank_text
from .respond import respond
from .retrieval import HybridRetriever

WEAK_EVIDENCE = "No retrieved passage is clearly relevant to the question, so the sources below may not answer it."
EMPTY_INDEX = "No papers are indexed yet, so there is nothing to answer from. Add papers first."
TARGETED_SLOTS = 2        # chunks found by the targeted re-search that are always kept
PROFILE_GUIDED_POOL = 6   # chunks per missing condition that the profile store proposes ...
PROFILE_GUIDED_SLOTS = 2  # ... of which the best two (by the reranker) are added


# The timed steps as the web page's live "chain of execution" shows them: id -> (stage number, component, what it does in one line).
STEP_INFO: dict[str, tuple[int, str, str]] = {
    "1_understand": (1, "Query understanding", "SciBERT reads the intent and complexity; the language model extracts the conditions (model, dataset, language ...)"),
    "2_plan": (2, "Orchestrator agent", "Decides whether to refine or split the question and which checks to run"),
    "3_refine": (3, "Query refinement", "Rewrites the question into search queries"),
    "4_retrieve": (4, "Hybrid retrieval", "BGE-M3 vectors + BM25 keywords + result cards"),
    "5_rerank": (5, "Reranker", "A cross-encoder keeps the passages that really answer the question"),
    "2b_replan": (2, "Orchestrator review", "Looks at what came back and may re-plan once"),
    "6_applicability": (6, "Applicability agent", "Checks that the passages cover every condition of the question; searches again or warns"),
    "7_contradiction": (7, "Contradiction resolver", "Separates real conflicts between papers from differences caused by different conditions"),
    "8_generate": (8, "Answer generation", "The language model writes the answer from the passages and the recorded conditions"),
    "9_critic": (9, "Claim critic", "An NLI model checks every sentence against its source and regenerates once if needed"),
    "9b_ragas": (9, "Faithfulness loop (RAGAS-style)", "A judge splits the answer into statements and checks each against the passages; the answer is written again while too few are supported"),
}


class _Timer:
    """Times the steps of one question and, when a `progress` listener is given, tells it what is happening (events for the live view of
    the web page: a step starts, a step ends or is skipped, a stage reports a result). The listener never changes what the pipeline does,
    and one that raises is ignored."""

    def __init__(self, progress: Callable[[dict], None] | None = None) -> None:
        self.ms: dict[str, float] = {}
        self.progress = progress
        self._t0 = time.perf_counter()

    def emit(self, **event) -> None:
        if self.progress is None:
            return
        try:
            self.progress({**event, "t_ms": round((time.perf_counter() - self._t0) * 1000)})
        except Exception:                                    # a broken listener must never break an answer
            pass

    @contextmanager
    def __call__(self, name: str):
        stage, component, blurb = STEP_INFO.get(name, (0, name, ""))
        t0 = time.perf_counter()
        self.emit(type="step", id=name, stage=stage, component=component, blurb=blurb, status="running")
        try:
            yield
        finally:
            self.ms[name] = (time.perf_counter() - t0) * 1000
            self.emit(type="step", id=name, stage=stage, component=component, status="done", ms=round(self.ms[name]))

    def skip(self, name: str, reason: str) -> None:
        stage, component, blurb = STEP_INFO.get(name, (0, name, ""))
        self.emit(type="step", id=name, stage=stage, component=component, blurb=blurb, status="skipped", reason=reason)

    def note(self, stage: int, text: str) -> None:
        self.emit(type="note", stage=stage, text=text)


class Pipeline:
    def __init__(self, settings: Settings | None = None, llm: OllamaLLM | None = None) -> None:
        self.cfg = settings or get_settings()
        self.llm = llm or OllamaLLM(self.cfg)
        self.vectors = VectorStore(self.cfg.paths.chroma_dir)
        self.profiles = ProfileStore(self.cfg.paths.profile_db)
        self.retriever = HybridRetriever(self.vectors, BM25Store.load(self.cfg.paths.bm25_path), self.cfg.retrieval)
        self.understand = QueryUnderstanding(self._agent_llm(self.cfg.agents.understanding_model),
                                             profiles=self.profiles if self.cfg.features.use_profiles else None)
        self.refine_llm = self._agent_llm(self.cfg.agents.refinement_model)
        orchestrator_llm = self._agent_llm(self.cfg.agents.orchestrator_model)
        applicability_llm = self._agent_llm(self.cfg.agents.applicability_model) if self.cfg.features.applicability_agent else None
        self.fallback_llm = self._agent_llm(self.cfg.agents.fallback_model) if self.cfg.features.escalation else None
        self.orchestrator = OrchestratorAgent(orchestrator_llm, self.cfg.features, self._escalation_target(orchestrator_llm))
        self.applicability = ApplicabilityAgent(
            self.profiles, self.cfg.applicability.max_reretrieve, llm=applicability_llm,
            fallback=self._escalation_target(applicability_llm), joint=self.cfg.features.joint_coverage)
        self.contradictions = ContradictionResolver(self.profiles, self.cfg.contradiction)
        self.critic = ClaimChecker(self.cfg.critic)
        # the original design's runtime RAGAS loop (a switch, off by default); the judge is the answer model unless [ragas] judge_model names another
        self.judge_llm = self._agent_llm(self.cfg.ragas.judge_model) if self.cfg.features.ragas_loop else None

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

    def warm_up(self) -> dict[str, float]:
        """Load every local model once - embedder, reranker, NLI, SciBERT and each Ollama model the pipeline uses - so the first
        real question is not the slow one (cold, it takes 30-100 s). Returns seconds per model.

        The Ollama models load side by side (a separate server, in threads). The torch models load one after another in this
        thread: `from_pretrained` sets a process-wide default dtype while it builds a model, so two loads at once can leave a model
        half fp16, half fp32 ("mat1 and mat2 must have the same dtype")."""
        llms = {m.cfg.model: m for m in (self.llm, self.refine_llm, self.understand.llm, self.orchestrator.llm,
                                         self.applicability.llm, self.fallback_llm) if m is not None}
        seconds: dict[str, float] = {}

        def timed(name: str, fn) -> None:
            t0 = time.perf_counter()
            fn()
            seconds[name] = round(time.perf_counter() - t0, 1)

        torch_jobs = {"embedder": lambda: embed(["warm up"]),
                      "reranker": lambda: rerank_scores("warm up", ["warm up"]),
                      "nli": lambda: get_nli().probs([("warm up", "warm up")])}
        if self.understand.classifier:
            torch_jobs["scibert"] = lambda: self.understand.classifier.predict("warm up")
        with ThreadPoolExecutor(max_workers=max(1, len(llms))) as pool:
            futures = [pool.submit(timed, f"ollama:{name}", lambda m=m: m.chat([{"role": "user", "content": "Reply with OK."}],
                                                                              max_tokens=3, temperature=0.0))
                       for name, m in llms.items()]
            for name, fn in torch_jobs.items():
                timed(name, fn)
            for future in futures:
                future.result()
        return seconds

    def reload_indexes(self) -> None:
        """Pick up papers added since start-up (the keyword index is a file rebuilt after ingestion)."""
        self.retriever.bm25 = BM25Store.load(self.cfg.paths.bm25_path)

    def run(self, question: str, history: list[dict] | None = None, stop_after: str | None = None,
            progress: Callable[[dict], None] | None = None) -> QueryResponse:
        """The whole pipeline. `stop_after="applicability"` returns right after Stage 6 (analysis, coverage, scope warning, the kept
        passages) without generating an answer: what the evaluation of scope warnings needs, in a third of the time.
        `progress` (optional) is called with one dict per event (step running / done / skipped, a stage's result line) so a caller can show
        which component is working; it changes nothing about the answer."""
        t, trace = _Timer(progress), []
        t0 = time.perf_counter()

        def say(line: str) -> None:
            """A trace line (kept in the response as before) that is also reported to the listener as a note of its stage."""
            trace.append(line)
            m = re.match(r"(\d+)\s+(.*)", line, re.S)
            t.note(int(m.group(1)) if m else 0, m.group(2) if m else line)

        if self.vectors.count() == 0:
            return respond(question, EMPTY_INDEX, [], None, None, [], [], False, ["index is empty"], {})

        with t("1_understand"):
            analysis = self.understand.analyze(question)
        with t("2_plan"):
            plan = self.orchestrator.plan(question, analysis, self.profiles.paper_count())
        for line in [f"1 {plan.notes[0]}; conditions={analysis.conditions.specified() or 'none'}", *[f"2 {n}" for n in plan.notes[1:]]]:
            say(line)

        queries = [question]
        if plan.refine:
            with t("3_refine"):
                queries = refine(question, analysis, self.refine_llm, decompose=plan.decompose)
            say(f"3 queries: {queries}")
        else:
            t.skip("3_refine", "the orchestrator judged the question simple enough to search as it is")

        with t("4_retrieve"):
            candidates = self.retriever.retrieve(queries, analysis.intent)
        t.note(4, f"{len(candidates)} candidate passages")
        with t("5_rerank"):
            kept, weak = rerank_filter(question, candidates, self.cfg.retrieval)
        trace.append(f"4 retrieved {len(candidates)} -> 5 kept {len(kept)}" + (" (weak evidence)" if weak else ""))
        t.note(5, f"kept {len(kept)} of {len(candidates)} passages" + (" (weak evidence)" if weak else ""))

        with t("2b_replan"):                     # the agent looks at what came back and may re-plan once
            review = self.orchestrator.review(question, plan, n_kept=len(kept), weak=weak)
        if review.action == "refine_and_retry":
            queries = refine(question, analysis, self.refine_llm, decompose=False)
            candidates = self.retriever.retrieve(queries, analysis.intent)
            kept, weak = rerank_filter(question, candidates, self.cfg.retrieval)
            say(f"2 re-plan ({review.reason or 'weak evidence'}): rewrote the question {queries} and searched again "
                f"-> kept {len(kept)}" + (" (still weak)" if weak else ""))

        retrieval_weak = weak                    # Stage 5's own verdict; Stage 6 may withdraw the flag below (the evaluation's abstention baseline reads this)
        applicability: ApplicabilityResult | None = None
        if plan.check_applicability:
            with t("6_applicability"):
                kept, applicability = self.applicability.run(
                    analysis.conditions.specified(), kept,
                    self._research(question, queries, analysis.intent, kept, analysis.conditions.specified()), question,
                    joint_fetch=self._joint_fetch(question), extras=analysis.conditions.extras())
            say(f"6 coverage {applicability.coverage:.2f}, missing {applicability.missing or 'none'}"
                + (", re-retrieved" if applicability.re_retrieved else "")
                + (", profile-guided" if applicability.profile_guided else "")
                + (", conditions recorded together" if applicability.joint_covered else "")
                + (", NOT recorded together" if applicability.joint_covered is False else "")
                + (f", second opinion by {self.fallback_llm.cfg.model}" if applicability.escalated else "")
                + (f" | agent: {applicability.reasoning}" if applicability.reasoning else ""))
        elif self.cfg.features.applicability:
            applicability = ApplicabilityResult()
            t.skip("6_applicability", "the question names no conditions to check")
        if weak and applicability is not None and applicability.checks and not applicability.missing:
            # The cross-encoder cannot read results tables (even with their retrieval card it scores the Kannada NLI table -3.9, below
            # the -2.0 threshold), but the profiles can: when every condition the question names is recorded in the kept passages the
            # evidence is not "weak", and telling the generator so would put a bogus "scope warning" into the answer.
            weak = False
            say("4 weak-evidence flag withdrawn: every condition named in the question is recorded in the kept passages")
        if stop_after == "applicability":
            t.ms["total"] = (time.perf_counter() - t0) * 1000
            return respond(question, "", kept[: self.cfg.retrieval.generate_top], analysis, applicability, [], [], False, trace, t.ms,
                           retrieval_weak=retrieval_weak)

        contradictions: list[ContradictionPair] = []
        if plan.check_contradictions and len(kept) >= 2:
            with t("7_contradiction"):
                contradictions = self.contradictions.resolve(question, kept, analysis.conditions.specified())
            say("7 conflicts: " + (", ".join(c.verdict for c in contradictions) or "none"))
        else:
            t.skip("7_contradiction", "fewer than two passages, or not a question about results")

        sources = select_sources(kept, contradictions, self.cfg.retrieval.generate_top,
                                 applicability.checks if applicability else None)
        profile_map = self.profiles.for_chunks([rc.chunk.chunk_id for rc in sources]) if self.cfg.features.use_profiles else {}
        warning = " ".join(w for w in (applicability.warning if applicability else None, WEAK_EVIDENCE if weak else None) if w) or None
        include = self.cfg.features.profile_in_context

        with t("8_generate"):
            answer = generate(question, sources, self.llm, warning=warning, conflicts=contradictions, history=history,
                              profiles=profile_map, include_profiles=include, requested=analysis.conditions.specified())
        t.note(8, f"wrote {len(answer)} characters from {len(sources)} passages")

        checks: list[ClaimCheck] = []
        regenerated = False
        if plan.check_claims:
            with t("9_critic"):
                checks = self.critic.check(answer, sources, profile_map)
                bad = self.critic.unsupported(checks)
                if bad and self.cfg.critic.max_regenerate > 0:
                    t.note(9, f"{len(bad)} sentence(s) not backed by the sources: writing the answer once more")
                    retry = generate(question, sources, self.llm, warning=warning, conflicts=contradictions, history=history,
                                     profiles=profile_map, include_profiles=include, requested=analysis.conditions.specified(),
                                     feedback="\n".join(f"- {c.sentence}" for c in bad))
                    retry_checks = self.critic.check(retry, sources, profile_map)
                    regenerated = True
                    if len(self.critic.unsupported(retry_checks)) <= len(bad):
                        answer, checks = retry, retry_checks
                answer, checks = self.critic.repair_citations(answer, checks, sources)
            say(f"9 claims checked {len(checks)}, unsupported {len(self.critic.unsupported(checks))}"
                + (", regenerated once" if regenerated else ""))
        else:
            t.skip("9_critic", "the claim check is switched off")

        faith_score: float | None = None
        if self.judge_llm is not None and answer.strip():                 # the original design's runtime RAGAS loop (a switch, off by default)
            with t("9b_ragas"):
                context = "\n\n".join(rc.chunk.text for rc in sources)
                recorded = [claim_text(p) for ps in profile_map.values() for p in ps]
                if recorded:
                    context += "\n\nRecorded results shown to the writer:\n" + "\n".join(recorded[:60])
                answer, faith, retries = improve_until_faithful(
                    question, answer, context, judge=self.judge_llm, threshold=self.cfg.ragas.threshold,
                    max_retries=self.cfg.ragas.max_retries, max_context_chars=self.cfg.ragas.max_context_chars,
                    notify=lambda msg: t.note(9, msg),
                    regenerate=lambda feedback: generate(question, sources, self.llm, warning=warning, conflicts=contradictions, history=history,
                                                         profiles=profile_map, include_profiles=include, requested=analysis.conditions.specified(),
                                                         feedback=feedback))
                if retries:
                    regenerated = True
                    if plan.check_claims:                                  # keep the claim list in step with the final text
                        checks = self.critic.check(answer, sources, profile_map)
                        answer, checks = self.critic.repair_citations(answer, checks, sources)
                if faith is not None:
                    faith_score = faith.score
                    trace.append(f"9 faithfulness {faith.score:.2f}" + (f" after {retries} rewrite(s)" if retries else "") + f" (judge {faith.judge})")

        t.ms["total"] = (time.perf_counter() - t0) * 1000
        return respond(question, answer, sources, analysis, applicability, contradictions, checks, regenerated, trace, t.ms,
                       retrieval_weak=retrieval_weak, faithfulness=faith_score)

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

    def _scored_chunks(self, question: str, ids: list[str], limit: int) -> list[RetrievedChunk]:
        """The best `limit` of these chunks by the reranker (not thresholded: a profile that records the conditions is the evidence)."""
        chunks = self.vectors.get(ids) if ids else []
        if not chunks:
            return []
        scores = rerank_scores(question, [rerank_text(c, self.cfg.retrieval.use_cards) for c in chunks])      # the question settles ties
        best = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)[:limit]
        return [RetrievedChunk(chunk=c, score=0.0, rerank_score=float(s)) for c, s in best]

    def _profile_guided(self, question: str, requested: dict[str, str], terms: list[str], have: set[str]) -> list[RetrievedChunk]:
        """Chunks the profile store says record a missing condition (together with the other named ones)."""
        if not self.cfg.features.profile_guided_retrieval or not requested:
            return []
        ids: list[str] = []
        fields = [f for f, wanted in requested.items() if wanted in terms]
        for field in fields:
            ids += [cid for cid in self.profiles.chunks_recording(requested, field, PROFILE_GUIDED_POOL)
                    if cid not in have and cid not in ids]
        return self._scored_chunks(question, ids, PROFILE_GUIDED_SLOTS * len(fields))

    def _joint_fetch(self, question: str):
        """For Stage 6's joint coverage: when results that record the named conditions together exist but none of their chunks was
        retrieved, add the best of them to the evidence."""
        def fetch(kept: list[RetrievedChunk], ids: list[str]) -> list[RetrievedChunk]:
            have = {rc.chunk.chunk_id for rc in kept}
            if have & set(ids):
                return kept
            merged = list(kept) + self._scored_chunks(question, ids, PROFILE_GUIDED_SLOTS)
            return sorted(merged, key=lambda rc: rc.rerank_score if rc.rerank_score is not None else rc.score, reverse=True)
        return fetch


@lru_cache(maxsize=1)
def default_pipeline() -> Pipeline:
    return Pipeline()


def run(question: str, history: list[dict] | None = None) -> QueryResponse:
    """The single entry point: question in, QueryResponse out."""
    return default_pipeline().run(question, history)
