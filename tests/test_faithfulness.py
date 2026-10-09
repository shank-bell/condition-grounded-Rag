"""RAGAS-style faithfulness, answer relevancy and the rewrite loop (9 Oct 2026). A stub judge replaces the language model; no GPU needed."""
from types import SimpleNamespace

import numpy as np
import pytest

from cgrag.config import get_settings
from cgrag.pipeline import faithfulness as F
from cgrag.pipeline.run import STEP_INFO


class StubJudge:
    """Answers the two judge calls from a script: `statements` for the first, then one verdict list per scoring call."""

    def __init__(self, statements, verdict_rounds, model="stub-judge"):
        self.cfg = SimpleNamespace(model=model)
        self.statements = statements if isinstance(statements, list) and statements and isinstance(statements[0], list) else [statements]
        self.verdict_rounds = verdict_rounds
        self.calls = 0
        self.rounds = 0

    def structured(self, prompt, model_cls, *, system=None, temperature=None, retries=1, max_tokens=None):
        name = model_cls.__name__
        if name == "_Statements":
            self.calls += 1
            batch = self.statements[min(self.rounds, len(self.statements) - 1)]
            return model_cls(statements=batch), None
        if name == "_Verdicts":
            verdicts = self.verdict_rounds[min(self.rounds, len(self.verdict_rounds) - 1)]
            self.rounds += 1
            return model_cls(verdicts=[{"statement": s, "reason": "r", "verdict": v} for s, v in verdicts]), None
        raise AssertionError(name)


def test_score_is_the_share_of_supported_statements_and_lists_the_rest():
    judge = StubJudge(["BERT-large gets 90.9 F1.", "BERT-large gets 95.0 F1."], [[("BERT-large gets 90.9 F1.", 1), ("BERT-large gets 95.0 F1.", 0)]])
    r = F.faithfulness("What F1 does BERT-large get?", "BERT-large gets 90.9 F1 [1]. It also gets 95.0 F1.", "BERT-large: 90.9 F1", judge)
    assert r.score == 0.5 and r.unsupported == ["BERT-large gets 95.0 F1."] and r.judge == "stub-judge"


def test_a_statement_that_only_says_what_the_sources_lack_is_not_checked():
    judge = StubJudge(["No results are reported for Kannada.", "XLM-R gets 71.5 on IndicXNLI."], [[("XLM-R gets 71.5 on IndicXNLI.", 1)]])
    r = F.faithfulness("q", "No results are reported for Kannada. XLM-R gets 71.5.", "ctx", judge)
    assert r.statements == ["XLM-R gets 71.5 on IndicXNLI."] and r.score == 1.0


def test_an_answer_with_nothing_to_check_has_no_score():
    judge = StubJudge(["The sources do not contain this result."], [[]])
    assert F.faithfulness("q", "The sources do not contain this result.", "ctx", judge).score is None


def test_a_judge_that_answers_with_a_string_verdict_is_still_read():
    class Lenient(StubJudge):
        def structured(self, prompt, model_cls, **kw):
            if model_cls.__name__ == "_Verdicts":
                return model_cls.model_validate({"verdicts": [{"statement": "a", "verdict": "1"}, {"statement": "b", "verdict": "0"}]}), None
            return super().structured(prompt, model_cls, **kw)

    assert F.faithfulness("q", "a b", "ctx", Lenient(["a", "b"], [[]])).score == 0.5


def test_a_judge_that_returns_invalid_json_gives_no_score_instead_of_an_error():
    class Broken:
        cfg = SimpleNamespace(model="broken")

        def structured(self, *a, **k):
            raise ValueError("LLM did not return valid _Statements")

    assert F.faithfulness("q", "answer", "ctx", Broken()).score is None


def _loop(judge, texts, threshold=0.8, max_retries=3):
    feedbacks, notes = [], []
    it = iter(texts[1:])

    def regenerate(feedback):
        feedbacks.append(feedback)
        return next(it)

    best, result, retries = F.improve_until_faithful("q", texts[0], "ctx", judge=judge, regenerate=regenerate, threshold=threshold,
                                                     max_retries=max_retries, notify=notes.append)
    return best, result, retries, feedbacks, notes


def test_the_loop_rewrites_with_the_unsupported_statements_as_feedback_until_the_score_is_high_enough():
    judge = StubJudge([["s1", "s2"], ["s1", "s2"]], [[("s1", 1), ("s2", 0)], [("s1", 1), ("s2", 1)]])
    best, result, retries, feedbacks, notes = _loop(judge, ["first", "second"])
    assert best == "second" and result.score == 1.0 and retries == 1
    assert feedbacks == ["- s2"] and any("writing the answer again (1 of 3)" in n for n in notes)


def test_the_loop_stops_after_max_retries_and_keeps_the_best_text():
    judge = StubJudge([["a", "b"]], [[("a", 1), ("b", 0)], [("a", 0), ("b", 0)], [("a", 1), ("b", 0)], [("a", 0), ("b", 0)]])
    best, result, retries, feedbacks, _ = _loop(judge, ["t0", "t1", "t2", "t3"], max_retries=3)
    assert retries == 3 and len(feedbacks) == 3 and best == "t0" and result.score == 0.5      # the first text was already the best


def test_the_loop_does_not_rewrite_an_answer_that_is_already_faithful_or_has_nothing_to_check():
    best, result, retries, feedbacks, _ = _loop(StubJudge([["a"]], [[("a", 1)]]), ["fine"])
    assert best == "fine" and retries == 0 and feedbacks == [] and result.score == 1.0
    best, result, retries, feedbacks, notes = _loop(StubJudge(["No results are reported."], [[]]), ["abstain"])
    assert best == "abstain" and result is None and retries == 0 and feedbacks == [] and notes == ["no statement to check"]


def test_answer_relevancy_is_the_mean_cosine_and_zero_for_a_noncommittal_answer():
    vectors = {"q": [1.0, 0.0], "same": [1.0, 0.0], "other": [0.0, 1.0]}

    def embed(texts):
        return np.array([vectors[t] for t in texts], dtype=np.float32)

    class Writer:
        cfg = SimpleNamespace(model="stub")

        def __init__(self, questions, noncommittal=False):
            self.questions, self.noncommittal = questions, noncommittal

        def structured(self, prompt, model_cls, **kw):
            return model_cls(questions=self.questions, noncommittal=self.noncommittal), None

    assert F.answer_relevancy("q", "an answer", Writer(["same", "other"]), embed) == pytest.approx(0.5)
    assert F.answer_relevancy("q", "I do not know", Writer(["same"], noncommittal=True), embed) == 0.0


def test_the_loop_is_off_by_default_and_has_the_designs_three_retries():
    cfg = get_settings()
    assert cfg.features.ragas_loop is False and cfg.ragas.max_retries == 3 and cfg.ragas.threshold == 0.80 and cfg.ragas.judge_model == ""
    assert STEP_INFO["9b_ragas"][0] == 9
