"""The evaluation baselines (Stage 6: autorater and the scoring of warnings; Stage 7: plain NLI and the LLM on the conflict taxonomy); stubs."""
import numpy as np

from cgrag.evaluation.baselines import (
    ConflictJudgement, Sufficiency, differing_names, llm_conflict_verdict, plain_nli_verdict, score_warnings, sufficient_context,
    taxonomy_verdict,
)
from cgrag.labelling.score import score_system_c


class StubNLI:
    def __init__(self, rows):
        self.rows = rows

    def probs(self, pairs):
        return np.array(self.rows[: len(pairs)])


class StubLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def structured(self, prompt, model_cls, **kwargs):
        self.prompts.append((prompt, kwargs.get("system")))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, None


# ---------- Stage 7 ----------

def test_plain_nli_calls_a_contradiction_genuine_and_never_predicts_explained():
    assert plain_nli_verdict(StubNLI([[0.05, 0.1, 0.85], [0.1, 0.1, 0.8]]), "A", "B")["system_verdict"] == "GENUINE"
    neutral = plain_nli_verdict(StubNLI([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]]), "A", "B")
    assert neutral["system_verdict"] == "NOT_COMPARABLE" and neutral["system_differing"] == []
    assert plain_nli_verdict(StubNLI([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]), "A", "B")["system_verdict"] == "NOT_COMPARABLE"


def test_the_taxonomy_categories_map_onto_the_three_classes():
    assert taxonomy_verdict("complementary information") == "EXPLAINED"
    assert taxonomy_verdict("Conflicting research outcomes") == "GENUINE"
    assert taxonomy_verdict("conflict due to outdated information") == "GENUINE"
    assert taxonomy_verdict("misinformation") == "GENUINE"
    assert taxonomy_verdict("no_conflict") == "NOT_COMPARABLE" and taxonomy_verdict("no conflict") == "NOT_COMPARABLE"
    assert taxonomy_verdict("something else") == "NOT_COMPARABLE"                        # an unrecognised answer: no decision
    assert differing_names(["dataset version", "model_size", "evaluation protocol", "language", "dataset_version", "the tokenizer"]) == [
        "dataset_version", "model_size", "setting", "language", "other"]


def test_the_llm_conflict_baseline_keeps_the_conditions_only_for_explained_and_survives_a_bad_reply():
    llm = StubLLM(ConflictJudgement(category="complementary information", differing_conditions=["dataset version", "setting"], reason="v1.1 vs v2.0"))
    out = llm_conflict_verdict(llm, "Both are accuracy.", "row A", "row B")
    assert out["system_verdict"] == "EXPLAINED" and out["system_differing"] == ["dataset_version", "setting"]
    assert "RESULT A\nrow A" in llm.prompts[0][0] and "DRAGged" in llm.prompts[0][1]
    genuine = llm_conflict_verdict(StubLLM(ConflictJudgement(category="conflicting research outcomes", differing_conditions=["setting"])), "h", "a", "b")
    assert genuine["system_verdict"] == "GENUINE" and genuine["system_differing"] == []
    broken = llm_conflict_verdict(StubLLM(ValueError("bad json")), "h", "a", "b")
    assert broken["system_verdict"] == "NOT_COMPARABLE" and "unusable" in broken["reason"]


def test_a_baseline_that_never_says_explained_gets_zero_f1_on_that_class_through_the_normal_scorer():
    gold = {pid: {"verdict": v, "differs": [], "extraction_error": False}
            for pid, v in (("P1", "GENUINE"), ("P2", "EXPLAINED"), ("P3", "NOT COMPARABLE"))}
    scores = score_system_c(gold, {pid: {"system_verdict": "GENUINE", "system_differing": []} for pid in gold})
    assert scores["accuracy"] == round(1 / 3, 4) and scores["per_class"]["EXPLAINED"]["f1"] == 0.0


# ---------- Stage 6 ----------

def test_the_autorater_reads_the_passages_and_an_unusable_reply_stays_silent():
    llm = StubLLM(Sufficiency(sufficient=False, reason="no Kannada"))
    sufficient, why = sufficient_context(llm, "XLM-R on XNLI for Kannada?", ["passage one", "p" * 5000])
    assert sufficient is False and why == "no Kannada"
    prompt = llm.prompts[0][0]
    assert "[1] passage one" in prompt and "[2] " in prompt and len(prompt) < 4000          # a long passage is cut
    assert sufficient_context(StubLLM(ValueError("x")), "q", [])[0] is True


def row(expect, warned, expected=None, named=None):
    return {"expect_warning": expect, "warned": warned, "expected_missing": expected or [], "named_missing": named}


def test_warning_scores_count_extrapolation_and_false_warnings():
    rows = [row(True, True, ["language"], ["language"]), row(True, True, ["language"], ["model"]), row(True, False, ["language"], []),
            row(False, False), row(False, True, [], ["dataset"]), row(False, False)]
    s = score_warnings(rows)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 2)
    assert s["precision"] == round(2 / 3, 4) and s["recall"] == round(2 / 3, 4)
    assert s["extrapolation_rate"] == round(1 / 3, 4) and s["false_warning_rate"] == round(1 / 3, 4)
    assert s["right_condition_named"] == 1                          # only the first correct warning names the condition the truth names
    never = score_warnings([row(True, False, ["language"]), row(False, False)])         # plain RAG: no warning, ever
    assert never["precision"] is None and never["recall"] == 0.0 and never["extrapolation_rate"] == 1.0
    assert never["right_condition_named"] is None
