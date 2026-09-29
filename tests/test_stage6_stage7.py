"""Stage 6 (applicability) and stage 7 (contradiction classes) on hand-built profiles; no LLM or GPU needed."""
import numpy as np
import pytest

from cgrag.config import ContradictionConfig
from cgrag.pipeline.applicability import ApplicabilityAgent
from cgrag.pipeline.contradiction import ContradictionResolver, classify
from cgrag.schemas import Chunk, ConditionProfile, RetrievedChunk
from cgrag.stores.profile_store import ProfileStore


def chunk(cid: str, paper: str, text: str = "results text", section: str = "results") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id=paper, page=1, section=section, text=text), score=1.0, rerank_score=1.0)


def prof(cid: str, paper: str, n: int = 0, **kw) -> ConditionProfile:
    base = dict(profile_id=f"{cid}#{n}", paper_id=paper, chunk_id=cid, metric="F1", value=90.0)
    return ConditionProfile(**{**base, **kw})


class StubNLI:
    def probs(self, pairs):
        return np.array([[0.05, 0.05, 0.90]] * len(pairs))


@pytest.fixture
def store(tmp_path):
    return ProfileStore(tmp_path / "p.sqlite")


# ---------- stage 6 ----------

def test_all_evidence_english_so_kannada_is_missing(store):
    store.add_many([prof("a:1", "a", model="BERT", dataset="SQuAD", language="English"),
                    prof("b:1", "b", model="BERT", dataset="MNLI", language="English")])
    agent = ApplicabilityAgent(store)
    res = agent.check({"model": "BERT", "language": "Kannada"}, [chunk("a:1", "a"), chunk("b:1", "b")])
    assert res.coverage == 0.5
    assert res.missing == ["language=Kannada"]
    assert "English" in res.warning and "Kannada" in res.warning
    assert [c.covered for c in res.checks] == [True, False]


def test_full_coverage_has_no_warning(store):
    store.add_many([prof("a:1", "a", model="BERT-large", dataset="SQuAD", dataset_version="2.0")])
    res = ApplicabilityAgent(store).check({"model": "BERT", "dataset": "SQuAD", "dataset_version": "v2.0"}, [chunk("a:1", "a")])
    assert res.coverage == 1.0 and res.warning is None and res.missing == []


def test_intro_chunk_speaks_for_its_papers_profiles(store):
    store.add_many([prof("a:9", "a", model="BERT", language="Hindi")])
    res = ApplicabilityAgent(store).check({"language": "Hindi"}, [chunk("a:0", "a", section="introduction")])
    assert res.coverage == 1.0


def test_unrecorded_field_falls_back_to_the_chunk_text(store):
    store.add_many([prof("a:1", "a", model="mBERT", dataset="XNLI")])            # language not recorded
    hit = ApplicabilityAgent(store).check({"language": "Swahili"}, [chunk("a:1", "a", text="results on Swahili and Hindi")])
    miss = ApplicabilityAgent(store).check({"language": "Swahili"}, [chunk("a:1", "a", text="results on English")])
    assert hit.coverage == 1.0 and miss.coverage == 0.0


def test_reretrieval_runs_once_and_can_close_the_gap(store):
    store.add_many([prof("a:1", "a", language="English"), prof("b:1", "b", language="Kannada")])
    calls = []

    def research(terms):
        calls.append(terms)
        return [chunk("a:1", "a"), chunk("b:1", "b")]

    kept, res = ApplicabilityAgent(store, max_reretrieve=1).run({"language": "Kannada"}, [chunk("a:1", "a")], research)
    assert calls == [["Kannada"]] and res.re_retrieved and res.coverage == 1.0 and len(kept) == 2


def test_reretrieval_gives_up_after_the_limit(store):
    store.add_many([prof("a:1", "a", language="English")])
    calls = []
    _, res = ApplicabilityAgent(store, max_reretrieve=1).run(
        {"language": "Kannada"}, [chunk("a:1", "a")], lambda t: calls.append(t) or [chunk("a:1", "a")])
    assert len(calls) == 1 and res.re_retrieved and res.missing == ["language=Kannada"] and res.warning


def test_no_conditions_means_full_coverage(store):
    assert ApplicabilityAgent(store).check({}, [chunk("a:1", "a")]).coverage == 1.0


# ---------- stage 7 ----------

def test_dataset_version_explains_the_difference(store):
    a = prof("a:1", "a", model="BERT", dataset="SQuAD", dataset_version="1.1", value=92.0, setting="dev set")
    b = prof("b:1", "b", model="BERT", dataset="SQuAD", dataset_version="2.0", value=85.0, setting="dev set")
    store.add_many([a, b])
    out = ContradictionResolver(store, ContradictionConfig(), lambda: StubNLI()).resolve("q", [chunk("a:1", "a"), chunk("b:1", "b")])
    assert len(out) == 1 and out[0].verdict == "EXPLAINED" and out[0].differing == ["dataset_version"]
    assert "1.1 vs 2.0" in out[0].reason and out[0].value_a == 92.0 and out[0].nli_contradiction == pytest.approx(0.9)


def test_same_conditions_different_result_is_genuine(store):
    kw = dict(model="BERT", dataset="SQuAD", dataset_version="1.1", setting="dev set", model_size="340M")
    store.add_many([prof("a:1", "a", value=90.9, **kw), prof("b:1", "b", value=84.0, **kw)])
    out = ContradictionResolver(store, ContradictionConfig(), lambda: StubNLI()).resolve("q", [chunk("a:1", "a"), chunk("b:1", "b")])
    assert [c.verdict for c in out] == ["GENUINE"] and out[0].differing == []


def test_condition_recorded_on_one_side_only_is_not_comparable(store):
    store.add_many([prof("a:1", "a", model="BERT", dataset="SQuAD", value=90.9, setting="dev set"),
                    prof("b:1", "b", model="BERT", dataset="SQuAD", value=84.0)])
    out = ContradictionResolver(store, ContradictionConfig(), lambda: StubNLI()).resolve("q", [chunk("a:1", "a"), chunk("b:1", "b")])
    assert [c.verdict for c in out] == ["NOT_COMPARABLE"] and "setting" in out[0].reason


def test_small_numeric_gap_and_same_paper_are_not_flagged(store):
    store.add_many([prof("a:1", "a", model="BERT", dataset="SQuAD", value=90.0),
                    prof("b:1", "b", model="BERT", dataset="SQuAD", value=90.5),        # 0.6 % < 2 %
                    prof("a:2", "a", model="BERT", dataset="SQuAD", value=70.0)])       # same paper as a:1
    res = ContradictionResolver(store, ContradictionConfig(nli_threshold=1.1), lambda: StubNLI())
    assert res.resolve("q", [chunk("a:1", "a"), chunk("b:1", "b")]) == []
    assert all(c.paper_a != c.paper_b for c in res.resolve("q", [chunk("a:1", "a"), chunk("a:2", "a")]))


def test_different_models_are_not_a_contradiction(store):
    store.add_many([prof("a:1", "a", model="BERT", dataset="SQuAD", value=90.0),
                    prof("b:1", "b", model="RoBERTa", dataset="SQuAD", value=80.0)])
    out = ContradictionResolver(store, ContradictionConfig(nli_threshold=1.1), lambda: StubNLI()).resolve("q", [chunk("a:1", "a"), chunk("b:1", "b")])
    assert out == []


def test_classify_names_model_size_difference():
    a = prof("a:1", "a", model="BERT-large", dataset="SQuAD", value=90.9)
    b = prof("b:1", "b", model="BERT-base", dataset="SQuAD", value=88.5)
    verdict, differing, _ = classify(a, b)
    assert verdict == "EXPLAINED" and differing == ["model_size"]
