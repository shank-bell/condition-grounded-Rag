"""Regression tests for the problems the first real end-to-end run exposed (stages 1, 7 and 9)."""
import numpy as np
import pytest

from cgrag.config import ContradictionConfig, CriticConfig
from cgrag.pipeline import contradiction as contradiction_module
from cgrag.pipeline.contradiction import ContradictionResolver
from cgrag.pipeline.critic import ClaimChecker
from cgrag.pipeline.query_understanding import vocabulary_conditions
from cgrag.pipeline.text import plain
from cgrag.schemas import Chunk, ConditionProfile, RetrievedChunk
from cgrag.stores.profile_store import ProfileStore


def rc(cid: str, paper: str, text: str = "results text") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id=paper, page=1, section="results", text=text), score=1.0, rerank_score=1.0)


def prof(cid: str, paper: str, n: int = 0, **kw) -> ConditionProfile:
    base = dict(profile_id=f"{cid}#{n}", paper_id=paper, chunk_id=cid, metric="F1", value=90.0)
    return ConditionProfile(**{**base, **kw})


class ContradictingNLI:
    def probs(self, pairs):
        return np.array([[0.05, 0.05, 0.90]] * len(pairs))


class NeutralNLI:
    def probs(self, pairs):
        return np.array([[0.20, 0.70, 0.10]] * len(pairs))


@pytest.fixture
def store(tmp_path):
    return ProfileStore(tmp_path / "p.sqlite")


# ---------- stage 1: names written literally in the question ----------

VOCAB = {"dataset": ["SQuAD", "GLUE", "Date"], "model": ["BERTLARGE", "RoBERTa", "Human"], "language": [], "task": []}


def test_dataset_version_model_and_language_are_found_in_the_question():
    got = vocabulary_conditions("What F1 does BERT-large get on SQuAD 2.0 for Kannada?", VOCAB)
    assert got == {"dataset": "SQuAD", "dataset_version": "2.0", "model": "BERT-large", "language": "Kannada"}


def test_names_the_question_does_not_contain_are_not_invented():
    assert vocabulary_conditions("How does pre-training work?", VOCAB) == {}
    assert "model" not in vocabulary_conditions("Is DistilBERT smaller?", VOCAB)          # not the BERT family


# ---------- stage 7: only conflicts about what the question asks, once per pair of papers ----------

def test_conflict_about_another_model_is_left_out_when_the_question_names_a_model(store):
    store.add_many([prof("a:1", "a", model="Human", dataset="SQuAD", value=90.0, dataset_version="1.1"),
                    prof("b:1", "b", model="Human", dataset="SQuAD", value=80.0, dataset_version="2.0"),
                    prof("a:2", "a", model="BERT", dataset="SQuAD", value=91.0, dataset_version="1.1"),
                    prof("b:2", "b", model="BERT", dataset="SQuAD", value=83.0, dataset_version="2.0")])
    chunks = [rc("a:1", "a"), rc("b:1", "b"), rc("a:2", "a"), rc("b:2", "b")]
    resolver = ContradictionResolver(store, ContradictionConfig(nli_threshold=1.1), lambda: ContradictingNLI())
    everything = resolver.resolve("q", chunks)
    only_bert = resolver.resolve("q", chunks, {"model": "BERT"})
    assert len(everything) == 2 and {c.value_a for c in everything} == {90.0, 91.0}
    assert len(only_bert) == 1 and only_bert[0].value_a == 91.0


def test_a_conflict_in_another_language_is_not_about_a_question_that_names_a_language(store):
    store.add_many([prof("a:1", "a", model="XLM-R", dataset="XNLI", language="Hindi", value=88.7),
                    prof("b:1", "b", model="XLM-R", dataset="XNLI", language="Swahili", value=66.5),
                    prof("a:2", "a", model="XLM-R", dataset="XNLI", value=85.0),               # no language recorded: English
                    prof("b:2", "b", model="XLM-R", dataset="XNLI", value=79.0)])
    chunks = [rc("a:1", "a"), rc("b:1", "b"), rc("a:2", "a"), rc("b:2", "b")]
    resolver = ContradictionResolver(store, ContradictionConfig(nli_threshold=1.1), lambda: ContradictingNLI())
    assert resolver.resolve("q", chunks, {"model": "XLM-R", "language": "Kannada"}) == []
    assert [(c.value_a, c.value_b) for c in resolver.resolve("q", chunks, {"language": "English"})] == [(85.0, 79.0)]
    # no language named: every language counts (one conflict per pair of papers and subject: the widest gap, Hindi vs Swahili)
    assert [(c.value_a, c.value_b) for c in resolver.resolve("q", chunks, {"model": "XLM-R"})] == [(88.7, 66.5)]


def test_the_same_conflict_is_reported_once(store):
    kw = dict(model="BERT", dataset="SQuAD", dataset_version="1.1", setting="dev set")
    store.add_many([prof("a:1", "a", value=90.0, **kw), prof("a:2", "a", value=90.0, **kw),
                    prof("b:1", "b", value=80.0, **kw), prof("b:2", "b", value=80.0, **kw)])
    chunks = [rc("a:1", "a"), rc("a:2", "a"), rc("b:1", "b"), rc("b:2", "b")]
    out = ContradictionResolver(store, ContradictionConfig(nli_threshold=1.1), lambda: ContradictingNLI()).resolve("q", chunks)
    assert len(out) == 1


TEXT_A = "Human performance on the benchmark reaches 91.2 accuracy across annotators."
TEXT_B = "Human performance on the benchmark stays at 80.3 accuracy across annotators."


def stub_rerank(score):
    return lambda question, passages: [score] * len(passages)


def test_text_only_disagreement_needs_relevance_shared_words_and_both_directions(store, monkeypatch):
    chunks = [rc("a:1", "a", TEXT_A), rc("b:1", "b", TEXT_B)]
    resolver = ContradictionResolver(store, ContradictionConfig(), lambda: ContradictingNLI())
    monkeypatch.setattr(contradiction_module, "rerank_scores", stub_rerank(2.0))
    assert [c.verdict for c in resolver.resolve("human performance?", chunks)] == ["NOT_COMPARABLE"]
    monkeypatch.setattr(contradiction_module, "rerank_scores", stub_rerank(-4.0))          # not about the question
    assert resolver.resolve("human performance?", chunks) == []
    monkeypatch.setattr(contradiction_module, "rerank_scores", stub_rerank(2.0))
    unrelated = [rc("a:1", "a", TEXT_A), rc("b:1", "b", "Completely different topic entirely.")]
    assert resolver.resolve("human performance?", unrelated) == []                           # no shared subject words


def test_a_year_in_a_citation_is_not_a_result(store, monkeypatch):
    chunks = [rc("a:1", "a", "We compare TinyBERT with PKD (Sun et al., 2019), BERTSMALL6 and DistilBERT (Sanh et al., 2019)."),
              rc("b:1", "b", "We compare our MobileBERT with BERTBASE and DistilBERT (Sanh et al., 2019) and DocQA (Clark, 2017).")]
    resolver = ContradictionResolver(store, ContradictionConfig(), lambda: ContradictingNLI())
    monkeypatch.setattr(contradiction_module, "rerank_scores", stub_rerank(2.0))
    assert resolver.resolve("Compare DistilBERT and BERT-base on GLUE.", chunks) == []
    percent = [rc("a:1", "a", "BERTBASE retains 97% of the performance on the benchmark."),
               rc("b:1", "b", "BERTBASE retains only 71.2% of the performance on the benchmark.")]
    assert [c.verdict for c in resolver.resolve("benchmark performance?", percent)] == ["NOT_COMPARABLE"]    # a decimal / percentage counts


def test_descriptive_sentences_without_a_number_are_never_a_text_only_conflict(store, monkeypatch):
    """NLI calls "We compare TinyBERT with ..." and "We compare our MobileBERT with ..." contradictory (different subjects); neither
    reports a result, so there is nothing to disagree about."""
    chunks = [rc("a:1", "a", "We compare TinyBERT with BERTTINY, DistilBERT and MobileBERT on the GLUE benchmark."),
              rc("b:1", "b", "We compare our MobileBERT with BERTBASE, DistilBERT and DocQA on the GLUE benchmark.")]
    resolver = ContradictionResolver(store, ContradictionConfig(), lambda: ContradictingNLI())
    monkeypatch.setattr(contradiction_module, "rerank_scores", stub_rerank(2.0))
    assert resolver.resolve("Compare DistilBERT and BERT-base on GLUE.", chunks) == []


# ---------- stage 9 ----------

def test_sentences_about_what_the_sources_lack_are_not_checked():
    answer = "The provided sources do not contain any information regarding Kannada. XNLI covers 15 languages [1]."
    sources = [rc("c1", "p", "XNLI covers 15 languages in total.")]
    checks = ClaimChecker(CriticConfig(), lambda: ContradictingNLI() if False else EntailingNLI()).check(answer, sources, {})
    assert [c.sentence for c in checks] == ["XNLI covers 15 languages [1]."]


class EntailingNLI:
    def probs(self, pairs):
        return np.array([[0.9, 0.05, 0.05]] * len(pairs))


def test_number_grounded_claim_is_accepted_when_nli_is_only_neutral():
    sources = [rc("c1", "p", "BERTLARGE (Single) 84.1 90.9\n\nSQuAD 1.1 development set")]
    answer = "BERT-large reaches 90.9 F1 on the SQuAD 1.1 development set [1]."
    (check,) = ClaimChecker(CriticConfig(), lambda: NeutralNLI()).check(answer, sources, {})
    assert check.supported


def test_number_not_in_the_source_stays_unsupported():
    sources = [rc("c1", "p", "BERTLARGE (Single) 84.1 90.9")]
    (check,) = ClaimChecker(CriticConfig(), lambda: NeutralNLI()).check("BERT-large reaches 95.5 F1 [1].", sources, {})
    assert not check.supported


def test_number_grounded_claim_is_still_rejected_when_nli_contradicts_it():
    sources = [rc("c1", "p", "BERTLARGE 90.9 F1 on the test set only")]
    (check,) = ClaimChecker(CriticConfig(), lambda: ContradictingNLI()).check("BERT-large reaches 90.9 F1 on the dev set [1].", sources, {})
    assert not check.supported


def test_markdown_is_stripped_before_a_sentence_is_checked():
    assert plain("*   **SQuAD 2.0 (dev)**: humans get 89.0 F1 [4].") == "SQuAD 2.0 (dev): humans get 89.0 F1."
