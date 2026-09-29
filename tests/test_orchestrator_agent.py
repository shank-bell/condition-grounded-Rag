"""The Orchestrator is an LLM agent working inside code guardrails (stub LLM: no GPU needed)."""
import pytest

from cgrag.config import FeaturesConfig
from cgrag.pipeline.orchestrator import OrchestratorAgent, PlanDecision, Review
from cgrag.schemas import QueryAnalysis, QueryConditions


class StubLLM:
    """Returns a fixed decision, or raises ValueError like OllamaLLM.structured does on invalid JSON."""

    def __init__(self, result=None, fail=False):
        self.result, self.fail, self.calls = result, fail, 0

    def structured(self, prompt, model_cls, **kwargs):
        self.calls += 1
        if self.fail:
            raise ValueError("invalid json")
        return self.result, None


def analysis(intent="result", complexity="simple", **cond) -> QueryAnalysis:
    return QueryAnalysis(intent=intent, complexity=complexity, conditions=QueryConditions(**cond))


def agent(result=None, fail=False, **features):
    llm = StubLLM(result, fail)
    return OrchestratorAgent(llm, FeaturesConfig(**features)), llm


def test_the_agent_can_add_refinement_for_a_vague_simple_question():
    a, llm = agent(PlanDecision(refine=True, decompose=False, check_contradictions=True, reason="informal wording"))
    plan = a.plan("bert kannada work?", analysis(model="BERT"), 10)
    assert plan.refine and not plan.decompose and plan.source == "agent" and llm.calls == 1
    assert any("agent added refinement" in n for n in plan.notes) and any("informal wording" in n for n in plan.notes)


def test_a_complex_question_is_always_rewritten_and_decomposed_even_if_the_agent_says_no():
    a, _ = agent(PlanDecision(refine=False, decompose=False, reason="looks easy"))
    plan = a.plan("q", analysis(intent="comparison", complexity="complex"), 10)
    assert plan.refine and plan.decompose


def test_the_agent_cannot_skip_the_applicability_check_when_conditions_are_named():
    a, _ = agent(PlanDecision(reason="no need"))
    assert a.plan("q", analysis(language="Kannada"), 10).check_applicability
    assert not a.plan("q", analysis(), 10).check_applicability                       # nothing to check


def test_ablation_switches_always_win_over_the_agent():
    a, _ = agent(PlanDecision(refine=True, check_contradictions=True), applicability=False, contradiction=False, critic=False)
    plan = a.plan("q", analysis(language="Kannada"), 10)
    assert not (plan.check_applicability or plan.check_contradictions or plan.check_claims)


def test_the_agent_may_skip_the_conflict_check_for_a_method_question_but_not_a_result_question():
    a, _ = agent(PlanDecision(check_contradictions=False, reason="how it works"))
    assert not a.plan("How does masking work?", analysis(intent="method"), 10).check_contradictions
    assert a.plan("What F1 does BERT get?", analysis(intent="result"), 10).check_contradictions


def test_unusable_agent_output_falls_back_to_the_fixed_rules():
    a, _ = agent(fail=True)
    plan = a.plan("q", analysis(complexity="complex"), 10)
    assert plan.source == "rules" and plan.refine and plan.decompose
    assert any("fixed rules used" in n for n in plan.notes)


def test_agent_switched_off_uses_the_rules_without_calling_the_llm():
    a, llm = agent(PlanDecision(refine=True), orchestrator_agent=False)
    plan = a.plan("q", analysis(), 10)
    assert plan.source == "rules" and not plan.refine and llm.calls == 0


def test_replan_only_when_evidence_is_weak_and_the_question_was_not_yet_rewritten():
    a, llm = agent(Review(action="refine_and_retry", reason="informal wording"))
    rules_plan = OrchestratorAgent(StubLLM(PlanDecision()), FeaturesConfig()).plan("q", analysis(), 10)
    assert a.review("q", rules_plan, n_kept=5, weak=False).action == "proceed" and llm.calls == 0      # evidence is fine
    assert a.review("q", rules_plan, n_kept=3, weak=True).action == "refine_and_retry" and llm.calls == 1
    rewritten = OrchestratorAgent(StubLLM(PlanDecision()), FeaturesConfig()).plan("q", analysis(complexity="complex"), 10)
    assert a.review("q", rewritten, n_kept=0, weak=True).action == "proceed"                          # already rewritten


def test_replan_falls_back_to_rules_when_the_agent_output_is_unusable():
    a, _ = agent(fail=True)
    plan = OrchestratorAgent(StubLLM(PlanDecision()), FeaturesConfig()).plan("q", analysis(), 10)
    assert a.review("q", plan, n_kept=0, weak=True).action == "refine_and_retry"
    assert a.review("q", plan, n_kept=1, weak=False).action == "proceed"
