"""Stage 2 - Orchestrator (AGENT, planner): decides the path a question takes through the pipeline.

The planner is an LLM agent: it reads the question, Stage 1's analysis and the size of the paper library, and decides
whether the question is rewritten and split, and whether the conflict check is worth running. After retrieval it looks at
what came back and can re-plan (rewrite the question and search again) when the evidence is weak.

The agent works inside guardrails that are plain code: the ablation switches in the config always win; the applicability
check cannot be skipped when the question names conditions; a complex question is always rewritten and decomposed; a
comparison or a result question always gets the conflict check. When the agent's output is unusable, or the agent is
switched off (`[features] orchestrator_agent = false`), the fixed rules in `make_plan` decide instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from ..config import FeaturesConfig
from ..llm import OllamaLLM
from ..schemas import QueryAnalysis

MIN_EVIDENCE = 2          # fewer kept passages than this counts as thin evidence

PLAN_SYSTEM = (
    "You are the planner of a pipeline that answers questions about computer-science research papers. Given a question, "
    "its analysis and the size of the paper library, decide which optional stages to run.\n"
    "refine: rewrite the question into a precise technical search query. Use it when the wording is vague or informal, or "
    "the question has several parts or compares things. Skip it for a single, clearly worded lookup.\n"
    "decompose: split the question into sub-questions that are searched separately. Only for multi-part or comparison "
    "questions; it needs refine.\n"
    "check_contradictions: compare results across papers to tell genuine conflicts from differences caused by different "
    "experimental conditions. Use it for comparisons, for a reported number that several papers may report, and for "
    "'do the papers agree' questions. Skip it for definitions and how-it-works questions.\n"
    "Return JSON with refine, decompose, check_contradictions and reason (one short sentence)."
)

REVIEW_SYSTEM = (
    "You are the planner of a pipeline that answers questions about research papers. The question was searched exactly as "
    "written and the search came back with weak or thin evidence. Decide: 'refine_and_retry' (rewrite the question into "
    "a precise technical query and search again) or 'proceed' (answer with what was found). Rewriting helps when the "
    "wording is informal or vague; it does not help when the papers simply do not cover the topic. "
    "Return JSON with action and reason (one short sentence)."
)


class PlanDecision(BaseModel):
    """What the planner agent returns."""
    refine: bool = False
    decompose: bool = False
    check_contradictions: bool = False
    reason: str = ""


class Review(BaseModel):
    action: Literal["proceed", "refine_and_retry"] = "proceed"
    reason: str = ""


@dataclass
class Plan:
    refine: bool                  # stage 3 runs: rewrite into technical form
    decompose: bool               # split a multi-part question into sub-questions
    check_applicability: bool     # stage 6 runs: the question names conditions that can be checked
    check_contradictions: bool    # stage 7 runs
    check_claims: bool            # stage 9 runs
    notes: list[str] = field(default_factory=list)
    source: str = "rules"         # "agent" or "rules"
    reason: str = ""


def make_plan(analysis: QueryAnalysis, features: FeaturesConfig) -> Plan:
    """The fixed rules: simple questions skip refinement; complex questions are rewritten and decomposed. They are the
    agent's guardrails and its fallback."""
    complex_q = analysis.complexity == "complex"
    conditions = analysis.conditions.specified()
    plan = Plan(
        refine=complex_q,
        decompose=complex_q,
        check_applicability=features.applicability and bool(conditions),
        check_contradictions=features.contradiction,
        check_claims=features.critic,
    )
    plan.notes.append(f"intent={analysis.intent} complexity={analysis.complexity}")
    plan.notes.append("refinement: " + ("rewrite + decompose" if plan.refine else "skipped (simple question)"))
    if features.applicability and not conditions:
        plan.notes.append("applicability: skipped (question names no conditions)")
    return plan


class OrchestratorAgent:
    def __init__(self, llm: OllamaLLM, features: FeaturesConfig) -> None:
        self.llm = llm
        self.features = features

    def plan(self, question: str, analysis: QueryAnalysis, n_papers: int) -> Plan:
        rules = make_plan(analysis, self.features)
        if not self.features.orchestrator_agent:
            rules.notes.append("planner: fixed rules (agent switched off)")
            return rules
        prompt = (f"QUESTION: {question}\n"
                  f"ANALYSIS: intent={analysis.intent}, complexity={analysis.complexity}, "
                  f"conditions named={analysis.conditions.specified() or 'none'}\n"
                  f"LIBRARY: {n_papers} papers")
        try:
            decision, _ = self.llm.structured(prompt, PlanDecision, system=PLAN_SYSTEM, max_tokens=160, retries=0)
        except ValueError:
            rules.notes.append("planner: agent output unusable, fixed rules used")
            return rules
        return self._within_guardrails(decision, rules, analysis)

    @staticmethod
    def _within_guardrails(decision: PlanDecision, rules: Plan, analysis: QueryAnalysis) -> Plan:
        refine = rules.refine or decision.refine
        plan = Plan(
            refine=refine,
            decompose=rules.decompose or (decision.decompose and refine),
            check_applicability=rules.check_applicability,
            check_contradictions=rules.check_contradictions and (decision.check_contradictions or analysis.intent in ("comparison", "result")),
            check_claims=rules.check_claims,
            notes=[rules.notes[0]], source="agent", reason=decision.reason.strip())
        plan.notes.append(f"planner agent: {plan.reason or 'no reason given'}")
        plan.notes.append("refinement: " + ("rewrite + decompose" if plan.decompose else "rewrite" if plan.refine else "skipped"))
        if plan.refine and not rules.refine:
            plan.notes.append("agent added refinement (the question was judged vague or multi-part)")
        if rules.check_contradictions and not plan.check_contradictions:
            plan.notes.append("agent skipped the conflict check (not a results question)")
        if len(rules.notes) > 2:
            plan.notes.append(rules.notes[2])                # e.g. applicability skipped: no conditions
        return plan

    def review(self, question: str, plan: Plan, *, n_kept: int, weak: bool) -> Review:
        """After retrieval: proceed, or rewrite the question and search again (only when the evidence is weak or thin
        and the question was searched as written)."""
        if plan.refine or not (weak or n_kept < MIN_EVIDENCE):
            return Review()
        if not self.features.orchestrator_agent:
            return Review(action="refine_and_retry", reason="weak evidence") if weak else Review()
        prompt = f"QUESTION: {question}\nOBSERVATION: the search kept {n_kept} passages; weak evidence: {weak}."
        try:
            review, _ = self.llm.structured(prompt, Review, system=REVIEW_SYSTEM, max_tokens=100, retries=0)
        except ValueError:
            return Review(action="refine_and_retry" if weak else "proceed", reason="agent output unusable")
        return review
