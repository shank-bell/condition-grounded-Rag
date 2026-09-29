"""Stage 6 as an LLM agent: the profile lookup settles what it can, the agent judges the rest inside guardrails (stub LLM)."""
import pytest

from cgrag.pipeline.applicability import ApplicabilityAgent, Judgement, Verdict
from cgrag.schemas import Chunk, ConditionProfile, RetrievedChunk
from cgrag.stores.profile_store import ProfileStore


def chunk(cid: str, paper: str, text: str = "results text", section: str = "results") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id=paper, page=1, section=section, text=text), score=1.0, rerank_score=1.0)


def prof(cid: str, paper: str, n: int = 0, **kw) -> ConditionProfile:
    return ConditionProfile(**{**dict(profile_id=f"{cid}#{n}", paper_id=paper, chunk_id=cid, metric="F1", value=90.0), **kw})


class StubLLM:
    """Returns queued judgements in order; an Exception in the queue is raised like OllamaLLM.structured does."""

    def __init__(self, *judgements):
        self.queue, self.prompts = list(judgements), []

    def structured(self, prompt, model_cls, **kwargs):
        self.prompts.append(prompt)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, None


@pytest.fixture
def store(tmp_path):
    return ProfileStore(tmp_path / "p.sqlite")


def agent(store, *judgements, retries=1):
    llm = StubLLM(*judgements)
    return ApplicabilityAgent(store, retries, llm=llm), llm


def test_recorded_profiles_settle_everything_so_the_llm_is_not_called(store):
    store.add_many([prof("a:1", "a", model="BERT-large", dataset="SQuAD", dataset_version="2.0")])
    a, llm = agent(store)
    res = a.check({"model": "BERT", "dataset": "SQuAD", "dataset_version": "v2.0"}, [chunk("a:1", "a")], "q")
    assert res.coverage == 1.0 and llm.prompts == []


def test_the_agent_judges_a_condition_the_profiles_cannot_settle_and_can_say_not_covered(store):
    store.add_many([prof("a:1", "a", model="mBERT", dataset="XNLI", language="en")])
    text = "We pre-train on 100 languages including Kannada, but evaluate only on English."
    a, llm = agent(store, Judgement(verdicts=[Verdict(condition="language", covered=False, note="Kannada is only a pre-training language")],
                                    search_query="mBERT XNLI Kannada evaluation"), retries=0)
    res = a.check({"model": "mBERT", "language": "Kannada"}, [chunk("a:1", "a", text)], "How does mBERT do on Kannada NLI?")
    assert res.missing == ["language=Kannada"] and "pre-training language" in res.reasoning and res.warning
    assert len(llm.prompts) == 1 and "Kannada -> " in llm.prompts[0]                  # the agent saw the text mention


def test_a_verdict_written_as_condition_equals_value_is_still_matched(store):
    store.add_many([prof("a:1", "a", model="mBERT")])
    a, _ = agent(store, Judgement(verdicts=[Verdict(condition="language=Kannada", covered=True, chunk_ids=["a:1"], matched="Kannada",
                                                    note="PAN-X results are reported for Kannada")]))
    res = a.check({"language": "Kannada"}, [chunk("a:1", "a", "PAN-X results for Kannada")], "q")
    assert res.coverage == 1.0 and "PAN-X results are reported for Kannada" in res.reasoning


def test_a_grounded_covered_verdict_is_accepted(store):
    store.add_many([prof("a:1", "a", model="mBERT")])
    text = "PAN-X named entity recognition results for Kannada and Tamil are reported in Table 4."
    a, _ = agent(store, Judgement(verdicts=[Verdict(condition="language", covered=True, chunk_ids=["a:1"], matched="Kannada")]))
    assert a.check({"language": "Kannada"}, [chunk("a:1", "a", text)], "q").coverage == 1.0


def test_a_covered_verdict_that_cites_something_the_passage_does_not_contain_is_overturned(store):
    store.add_many([prof("a:1", "a", model="mBERT")])
    a, _ = agent(store, Judgement(verdicts=[Verdict(condition="language", covered=True, chunk_ids=["a:1"], matched="Kannada")]))
    res = a.check({"language": "Kannada"}, [chunk("a:1", "a", "English results only.")], "q")
    assert res.missing == ["language=Kannada"]


def test_the_agent_can_never_override_a_strict_field(store):
    store.add_many([prof("a:1", "a", model="RoBERTa", dataset="SQuAD", dataset_version="1.1")])
    a, llm = agent(store, Judgement(verdicts=[
        Verdict(condition="model", covered=True, chunk_ids=["a:1"], matched="RoBERTa"),
        Verdict(condition="dataset_version", covered=True, chunk_ids=["a:1"], matched="1.1")]))
    res = a.check({"model": "BERT", "dataset_version": "2.0"}, [chunk("a:1", "a", "RoBERTa on SQuAD 1.1")], "q")
    assert sorted(res.missing) == ["dataset_version=2.0", "model=BERT"]


def test_unusable_agent_output_falls_back_to_the_deterministic_matcher(store):
    store.add_many([prof("a:1", "a", model="mBERT", dataset="XNLI")])
    a, _ = agent(store, ValueError("invalid json"))
    assert a.check({"language": "Swahili"}, [chunk("a:1", "a", "results on Swahili and Hindi")], "q").coverage == 1.0   # text fallback
    b, _ = agent(store, ValueError("invalid json"))
    assert b.check({"language": "Swahili"}, [chunk("a:1", "a", "results on English")], "q").coverage == 0.0


def test_the_agents_search_query_is_used_for_the_second_look_and_its_second_judgement_counts(store):
    store.add_many([prof("a:1", "a", language="English"), prof("b:1", "b", language="Kannada")])
    calls = []

    def research(terms, query=""):
        calls.append((terms, query))
        return [chunk("a:1", "a"), chunk("b:1", "b")]

    a, llm = agent(store, Judgement(verdicts=[Verdict(condition="language", covered=False)], search_query="Kannada NLI benchmark"))
    kept, res = a.run({"language": "Kannada"}, [chunk("a:1", "a")], research, "q")
    assert calls == [(["Kannada"], "Kannada NLI benchmark")] and res.re_retrieved and res.coverage == 1.0
    assert len(llm.prompts) == 1                      # the second look was settled by the recorded profile, no second LLM call


def test_without_an_llm_the_agent_is_the_deterministic_matcher(store):
    store.add_many([prof("a:1", "a", language="English")])
    res = ApplicabilityAgent(store).check({"language": "Kannada"}, [chunk("a:1", "a")], "q")
    assert res.missing == ["language=Kannada"] and "English" in res.warning
