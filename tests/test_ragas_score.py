"""RAGAS score-only mode (14 Oct 2026): `score_answer` returns faithfulness and answer relevancy without rewriting, the response carries both.
A stub judge and a stub embedder replace the models; no GPU needed."""
import zlib
from types import SimpleNamespace

import numpy as np

from cgrag.config import get_settings
from cgrag.pipeline import faithfulness as F
from cgrag.pipeline.respond import respond
from cgrag.schemas import QueryResponse


class Judge:
    def __init__(self, statements, verdicts, questions, noncommittal=False, relevancy_reply_ok=True, model="stub-judge"):
        self.cfg = SimpleNamespace(model=model)
        self.statements, self.verdicts, self.questions = statements, verdicts, questions
        self.noncommittal, self.relevancy_reply_ok = noncommittal, relevancy_reply_ok
        self.calls: list[str] = []

    def structured(self, prompt, model_cls, *, system=None, temperature=None, retries=1, max_tokens=None):
        name = model_cls.__name__
        self.calls.append(name)
        if name == "_Statements":
            return model_cls(statements=self.statements), None
        if name == "_Verdicts":
            return model_cls(verdicts=[{"statement": s, "reason": "r", "verdict": v} for s, v in self.verdicts]), None
        if name == "_Questions":
            if not self.relevancy_reply_ok:
                raise ValueError("no valid JSON")
            return model_cls(questions=self.questions, noncommittal=self.noncommittal), None
        raise AssertionError(name)


def fake_embed(texts):
    """Bag of words hashed into 64 dimensions, L2-normalised: texts that share words are similar."""
    out = np.zeros((len(texts), 64))
    for i, t in enumerate(texts):
        for w in t.lower().replace("?", "").split():
            out[i, zlib.crc32(w.encode()) % 64] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


Q = "What F1 does BERT-large get on SQuAD v1.1?"
A = "BERT-large gets 90.9 F1 on SQuAD v1.1 [1]."


def test_score_answer_gives_both_numbers_and_calls_only_the_judge():
    judge = Judge([A], [(A, 1)], ["What F1 does BERT-large get on SQuAD v1.1?", "How well does BERT-large do on SQuAD v1.1?"])
    s = F.score_answer(Q, A, "BERT-large: 90.9 F1 on SQuAD v1.1", judge, fake_embed)
    assert s.faithfulness.score == 1.0 and s.faithfulness.judge == "stub-judge"
    assert s.relevancy is not None and s.relevancy > 0.5
    assert judge.calls == ["_Statements", "_Verdicts", "_Questions"]


def test_a_non_committal_answer_has_zero_relevancy():
    judge = Judge(["The sources do not say."], [], ["Question?"], noncommittal=True)
    s = F.score_answer(Q, "The sources do not say.", "ctx", judge, fake_embed)
    assert s.relevancy == 0.0 and s.faithfulness.score is None


def test_an_unusable_relevancy_reply_gives_none_and_keeps_faithfulness():
    judge = Judge([A], [(A, 0)], [], relevancy_reply_ok=False)
    s = F.score_answer(Q, A, "ctx", judge, fake_embed)
    assert s.faithfulness.score == 0.0 and s.relevancy is None


def test_relevancy_can_be_left_out():
    judge = Judge([A], [(A, 1)], ["x"])
    s = F.score_answer(Q, A, "ctx", judge, fake_embed, with_relevancy=False)
    assert s.relevancy is None and "_Questions" not in judge.calls


def test_the_switch_is_off_by_default_and_the_response_fields_are_optional():
    cfg = get_settings()
    assert cfg.features.ragas_score is False and cfg.features.ragas_loop is False
    r = QueryResponse(question="q", answer="a")
    assert r.faithfulness is None and r.answer_relevancy is None and r.faithfulness_judge is None


def test_respond_carries_the_scores():
    r = respond("q", "a", [], None, None, [], [], False, [], {}, faithfulness=0.9, answer_relevancy=0.8, faithfulness_judge="gemma4:12b")
    assert (r.faithfulness, r.answer_relevancy, r.faithfulness_judge) == (0.9, 0.8, "gemma4:12b")
