"""Escalation (a small-model agent hands a bad outcome to the bigger model) and profile-guided retrieval (Stage 6 asks the
profile store which chunks record a missing condition). Stub LLMs: no GPU needed."""
from types import SimpleNamespace

import pytest

from cgrag.config import FeaturesConfig
from cgrag.pipeline.applicability import ApplicabilityAgent, Judgement, Verdict
from cgrag.pipeline.orchestrator import OrchestratorAgent, PlanDecision, Review
from cgrag.schemas import Chunk, ConditionProfile, QueryAnalysis, QueryConditions, RetrievedChunk
from cgrag.stores.profile_store import ProfileStore


class StubLLM:
    """Queued results in order; an Exception in the queue is raised like OllamaLLM.structured does on invalid JSON."""

    def __init__(self, *results, model="small"):
        self.queue, self.calls = list(results), 0
        self.cfg = SimpleNamespace(model=model)

    def structured(self, prompt, model_cls, **kwargs):
        self.calls += 1
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, None


BAD = ValueError("invalid json")


def analysis(intent="result", complexity="simple", **cond) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, complexity=complexity, conditions=QueryConditions(**cond))


def rc(cid: str, paper: str, text: str = "results text") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id=paper, page=1, section="results", text=text), score=1.0, rerank_score=1.0)


def prof(cid: str, paper: str, n: int = 0, **kw) -> ConditionProfile:
    return ConditionProfile(**{**dict(profile_id=f"{cid}#{n}", paper_id=paper, chunk_id=cid, metric="accuracy", value=70.0), **kw})


# ---------- Stage 2: the planner ----------

def test_an_unusable_small_plan_is_escalated_to_the_bigger_model():
    small, big = StubLLM(BAD), StubLLM(PlanDecision(refine=True, decompose=False, check_contradictions=True, reason="vague"), model="big")
    plan = OrchestratorAgent(small, FeaturesConfig(), big).plan("q", analysis(), 28)
    assert plan.source == "agent" and plan.refine and (small.calls, big.calls) == (1, 1)
    assert any("escalated to big" in n for n in plan.notes)


def test_a_usable_small_plan_never_calls_the_bigger_model():
    small, big = StubLLM(PlanDecision(reason="ok")), StubLLM(model="big")
    OrchestratorAgent(small, FeaturesConfig(), big).plan("q", analysis(), 28)
    assert big.calls == 0


def test_when_both_models_fail_the_fixed_rules_decide():
    plan = OrchestratorAgent(StubLLM(BAD), FeaturesConfig(), StubLLM(BAD, model="big")).plan("q", analysis(complexity="complex"), 28)
    assert plan.source == "rules" and plan.refine and plan.decompose


def test_weak_retrieval_is_re_planned_by_the_bigger_model():
    small, big = StubLLM(model="small"), StubLLM(Review(action="refine_and_retry", reason="informal wording"), model="big")
    a = OrchestratorAgent(small, FeaturesConfig(), big)
    plan = OrchestratorAgent(StubLLM(PlanDecision()), FeaturesConfig()).plan("q", analysis(), 28)
    review = a.review("q", plan, n_kept=1, weak=True)
    assert review.action == "refine_and_retry" and "decided by big" in review.reason
    assert (small.calls, big.calls) == (0, 1)


def test_without_a_fallback_the_review_uses_the_agents_own_model():
    small = StubLLM(Review(action="proceed", reason="not covered"))
    plan = OrchestratorAgent(StubLLM(PlanDecision()), FeaturesConfig()).plan("q", analysis(), 28)
    review = OrchestratorAgent(small, FeaturesConfig()).review("q", plan, n_kept=1, weak=True)
    assert review.action == "proceed" and small.calls == 1 and "decided by" not in review.reason


# ---------- Stage 6: second opinion ----------

@pytest.fixture
def store(tmp_path):
    return ProfileStore(tmp_path / "p.sqlite")


TEXT = "PAN-X named entity recognition results for Kannada and Tamil are reported in Table 4."


def test_a_not_covered_verdict_gets_a_second_opinion_before_a_warning_is_shown(store):
    store.add_many([prof("a:1", "a", model="mBERT")])
    small = StubLLM(Judgement(verdicts=[Verdict(condition="language", covered=False)]))
    big = StubLLM(Judgement(verdicts=[Verdict(condition="language", covered=True, chunk_ids=["a:1"], matched="Kannada")]), model="big")
    agent = ApplicabilityAgent(store, 0, llm=small, fallback=big)
    _, res = agent.run({"language": "Kannada"}, [rc("a:1", "a", TEXT)], lambda terms, query="": [], "q")
    assert res.coverage == 1.0 and res.warning is None and res.escalated
    assert (small.calls, big.calls) == (1, 1)


def test_the_second_opinion_can_confirm_the_warning(store):
    store.add_many([prof("a:1", "a", model="mBERT")])
    no = Judgement(verdicts=[Verdict(condition="language", covered=False, note="only a pre-training language")])
    agent = ApplicabilityAgent(store, 0, llm=StubLLM(no), fallback=StubLLM(no, model="big"))
    _, res = agent.run({"language": "Kannada"}, [rc("a:1", "a", TEXT)], lambda terms, query="": [], "q")
    assert res.missing == ["language=Kannada"] and res.warning and res.escalated


def test_a_second_opinion_is_not_asked_when_everything_is_covered(store):
    store.add_many([prof("a:1", "a", model="mBERT", language="Kannada")])
    big = StubLLM(model="big")
    agent = ApplicabilityAgent(store, 0, llm=StubLLM(), fallback=big)
    _, res = agent.run({"language": "Kannada"}, [rc("a:1", "a")], lambda terms, query="": [], "q")
    assert res.coverage == 1.0 and not res.escalated and big.calls == 0


def test_the_second_opinion_is_still_grounded_and_strict_fields_stay_recorded_only(store):
    store.add_many([prof("a:1", "a", model="RoBERTa")])
    no = Judgement(verdicts=[Verdict(condition="model", covered=False)])
    yes = Judgement(verdicts=[Verdict(condition="model", covered=True, chunk_ids=["a:1"], matched="BERT")])
    agent = ApplicabilityAgent(store, 0, llm=StubLLM(no), fallback=StubLLM(yes, model="big"))
    _, res = agent.run({"model": "BERT"}, [rc("a:1", "a", "RoBERTa results")], lambda terms, query="": [], "q")
    assert res.missing == ["model=BERT"]                      # the bigger model cannot override a strict field either


def test_no_fallback_without_an_agent_llm(store):
    assert ApplicabilityAgent(store, 0, llm=None, fallback=StubLLM(model="big")).fallback is None


# ---------- Stage 6: profile-guided retrieval ----------

def test_the_profile_store_points_to_chunks_that_record_the_missing_condition_and_the_other_conditions(store):
    store.add_many([
        prof("nli:1", "p1", model="XLM-R", language="Kannada", task="natural language inference", dataset="IndicXNLI"),
        prof("nli:1", "p1", 1, model="mBERT", language="Kannada", task="natural language inference", dataset="IndicXNLI"),
        prof("ner:1", "p2", model="XLM-R", language="Kannada", task="named entity recognition", dataset="WikiAnn"),
        prof("xnli:1", "p3", model="XLM-R", language="Hindi", task="natural language inference", dataset="XNLI"),
    ])
    found = store.chunks_recording({"language": "Kannada", "task": "natural language inference"}, "language")
    assert found == ["nli:1"]                                 # Kannada AND NLI beats Kannada alone and NLI alone


def test_a_chunk_that_records_only_part_of_the_conditions_is_not_pointed_to(store):
    """XNLI has no Kannada: pointing to Kannada results of another dataset would hide the honest scope warning."""
    store.add_many([
        prof("ner:1", "p2", model="XLM-R", language="Kannada", task="NER", dataset="WikiAnn"),
        prof("xnli:1", "p3", model="XLM-R", language="Hindi", task="NLI", dataset="XNLI"),
    ])
    assert store.chunks_recording({"language": "Kannada", "dataset": "XNLI"}, "language") == []
    assert store.chunks_recording({"language": "Kannada", "dataset": "XNLI", "model": "XLM-R"}, "language") == []
    assert store.chunks_recording({"language": "Kannada", "model": "XLM-R"}, "language") == ["ner:1"]


def test_language_codes_and_the_cache_are_handled(store):
    store.add_many([prof("a:1", "a", language="kn")])
    assert store.chunks_recording({"language": "Kannada"}, "language") == ["a:1"]
    store.add_many([prof("b:1", "b", language="Kannada")])
    assert set(store.chunks_recording({"language": "Kannada"}, "language")) == {"a:1", "b:1"}      # new rows are seen


def test_nothing_recorded_gives_no_chunks(store):
    store.add_many([prof("a:1", "a", language="Hindi")])
    assert store.chunks_recording({"language": "Kannada"}, "language") == []
    assert store.chunks_recording({"language": "Kannada"}, "dataset") == []
