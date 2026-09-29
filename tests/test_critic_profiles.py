"""Stage 9 verifies table-derived claims against recorded results; stage 8 gets the passages that cover the conditions."""
import numpy as np

from cgrag.config import CriticConfig
from cgrag.pipeline.critic import ClaimChecker
from cgrag.pipeline.generate import select_sources
from cgrag.schemas import Chunk, ConditionCheck, ConditionProfile, RetrievedChunk


def rc(cid: str, text: str = "some passage text") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id="p", page=1, section="results", text=text), score=1.0, rerank_score=1.0)


def prof(cid: str, **kw) -> ConditionProfile:
    return ConditionProfile(**{**dict(profile_id=f"{cid}#0", paper_id="p", chunk_id=cid), **kw})


class ContradictingNLI:
    def probs(self, pairs):
        return np.array([[0.02, 0.08, 0.90]] * len(pairs))


CLAIM = "BERT-large (Single) achieves an F1 of 83.1 on the SQuAD 2.0 test set [1]."
RESULT = dict(model="BERTLARGE (Single)", dataset="SQuAD", metric="F1", value=83.1, setting="test set")


def check(profiles):
    return ClaimChecker(CriticConfig(), lambda: ContradictingNLI()).check(CLAIM, [rc("c1", "BERTLARGE 78.7 81.9 80.0 83.1")], {"c1": profiles})


def test_a_recorded_result_that_matches_number_metric_and_model_supports_the_claim_even_if_nli_is_confused():
    assert [c.supported for c in check([prof("c1", **RESULT)])] == [True]


def test_a_recorded_dev_result_does_not_support_a_test_set_claim():
    assert [c.supported for c in check([prof("c1", **{**RESULT, "setting": "dev set"})])] == [False]


def test_a_recorded_result_of_another_model_does_not_support_the_claim():
    assert [c.supported for c in check([prof("c1", **{**RESULT, "model": "RoBERTa"})])] == [False]


def test_a_different_number_is_not_supported():
    assert [c.supported for c in check([prof("c1", **{**RESULT, "value": 81.9})])] == [False]


def test_the_answers_restatement_of_a_conflict_is_not_checked():
    answer = ("XLM-R beats mBERT on XNLI [1].\n"
              "- Sources [4] and [5] (not comparable): The results differ (92.25 vs 79.2 accuracy) but conditions are unrecorded.")
    sources = [rc(f"c{i}", "XLM-R beats mBERT on XNLI by a wide margin.") for i in range(1, 6)]

    class Entailing:
        def probs(self, pairs):
            return np.array([[0.9, 0.05, 0.05]] * len(pairs))

    checks = ClaimChecker(CriticConfig(), lambda: Entailing()).check(answer, sources, {})
    assert [c.sentence for c in checks] == ["XLM-R beats mBERT on XNLI [1]."]


def test_a_chunk_that_covers_a_condition_is_added_to_the_sources():
    kept = [rc(f"c{i}") for i in range(8)]
    checks = [ConditionCheck(condition="language", requested="Hindi", covered=True, chunk_ids=["c6", "c7"]),
              ConditionCheck(condition="model", requested="mBERT", covered=True, chunk_ids=["c1"]),          # already in the top 5
              ConditionCheck(condition="dataset", requested="XNLI", covered=False)]
    ids = [s.chunk.chunk_id for s in select_sources(kept, [], top=5, checks=checks)]
    assert ids == ["c0", "c1", "c2", "c3", "c4", "c6"]
