"""Stage 6 joint coverage: each condition may be covered by a different paper while no paper covers them together."""
import pytest

from cgrag.pipeline.applicability import ApplicabilityAgent
from cgrag.schemas import Chunk, ConditionProfile, RetrievedChunk
from cgrag.stores.profile_store import ProfileStore


def rc(cid: str, paper: str = "p") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id=paper, page=1, section="results", text="a results table"), score=1.0, rerank_score=1.0)


def prof(cid: str, n: int, **kw) -> ConditionProfile:
    return ConditionProfile(**{**dict(profile_id=f"{cid}#{n}", paper_id=cid.split(":")[0], chunk_id=cid, metric="accuracy", value=70.0 + n), **kw})


@pytest.fixture
def store(tmp_path):
    s = ProfileStore(tmp_path / "p.sqlite")
    s.add_many([
        prof("xnli:1", 1, model="XLM-R", dataset="XNLI", language="English"),
        prof("xnli:1", 2, model="XLM-R", dataset="XNLI", language="Hindi"),
        prof("xnli:1", 3, model="mBERT", dataset="XNLI", language="Hindi"),
        prof("indic:1", 4, model="XLM-R", dataset="IndicXNLI", language="Kannada"),
        prof("indic:1", 5, model="mBERT", dataset="IndicXNLI", language="Kannada"),
    ])
    return s


def agent(store, joint=True):
    return ApplicabilityAgent(store, 0, llm=None, joint=joint)


NONE = lambda terms, query="": []                                   # noqa: E731  (the re-search finds nothing new)
EVIDENCE = [rc("xnli:1"), rc("indic:1")]


def test_profiles_matching_needs_one_profile_with_all_the_conditions(store):
    key = {"model": "XLM-R", "dataset": "XNLI", "language": "Hindi"}
    assert [p.profile_id for p in store.profiles_matching(key, list(key))] == ["xnli:1#2"]
    assert store.profiles_matching({"model": "XLM-R", "dataset": "XNLI", "language": "Kannada"}, ["model", "dataset", "language"]) == []
    assert store.profiles_matching({"model": "BERT"}, ["model"]) == []                     # mBERT is not BERT
    assert len(store.profiles_matching({"language": "kn"}, ["language"])) == 2             # a language code matches the name
    assert store.profiles_matching({}, ["model"]) == []


def test_without_the_guardrail_each_condition_covered_by_a_different_paper_passes(store):
    requested = {"model": "XLM-R", "dataset": "XNLI", "language": "Kannada"}
    _, result = agent(store, joint=False).run(requested, EVIDENCE, NONE, "q")
    assert result.coverage == 1.0 and result.warning is None and result.joint_covered is None


def test_the_guardrail_warns_that_xlm_r_on_xnli_is_not_recorded_for_kannada(store):
    requested = {"model": "XLM-R", "dataset": "XNLI", "language": "Kannada"}
    _, result = agent(store).run(requested, EVIDENCE, NONE, "q")
    assert result.joint_covered is False and result.missing == ["language=Kannada"]       # XNLI x XLM-R has several languages: blame language
    assert result.coverage == pytest.approx(2 / 3)
    assert "language = Kannada together with model = XLM-R, dataset = XNLI" in result.warning
    assert "English" in result.warning and "Hindi" in result.warning                       # what IS recorded for XLM-R on XNLI
    assert next(c for c in result.checks if c.condition == "language").covered is False


def test_one_condition_is_blamed_and_the_most_specific_is_preferred(store):
    store.add_many([prof("mmlu:1", 9, model="GPT-4", dataset="MMLU")])                      # GPT-4 exists, but not on XNLI
    _, result = agent(store).run({"model": "GPT-4", "dataset": "XNLI"}, [rc("xnli:1"), rc("mmlu:1")], NONE, "q")
    assert result.joint_covered is False and result.missing == ["dataset=XNLI"]             # dataset before model; says what GPT-4 IS recorded on
    assert "MMLU" in result.warning
    store.add_many([prof("ner:1", 10, model="XLM-R", dataset="WikiAnn", language="Kannada")])  # XLM-R has Kannada results, on another dataset
    both = {"model": "XLM-R", "dataset": "XNLI", "language": "Kannada"}
    _, result = agent(store).run(both, [*EVIDENCE, rc("ner:1")], NONE, "q")
    assert result.missing == ["language=Kannada"]                                           # language, though dataset is also a candidate
    _, result = agent(store).run({"model": "GPT-4", "dataset": "XNLI", "language": "Hindi"}, [rc("xnli:1"), rc("mmlu:1")], NONE, "q")
    assert result.joint_covered is False and len(result.missing) == 1


def test_recorded_together_is_covered_and_the_chunk_is_added_when_it_was_not_retrieved(store):
    store.add_many([prof("zzz:1", 7, model="XLM-R", dataset="XNLI", language="Kannada")])     # a result that records all three together
    requested = {"model": "XLM-R", "dataset": "XNLI", "language": "Kannada"}
    asked = []

    def fetch(kept, ids):
        asked.append(ids)
        return kept + [rc("zzz:1")]

    # the retrieved chunks cover each condition separately, but not the chunk that records them together
    kept, result = agent(store).run(requested, EVIDENCE, NONE, "q", joint_fetch=fetch)
    assert result.joint_covered is True and asked == [["zzz:1"]] and [r.chunk.chunk_id for r in kept][-1] == "zzz:1"
    _, again = agent(store).run(requested, [*EVIDENCE, rc("zzz:1")], NONE, "q", joint_fetch=fetch)       # already retrieved: nothing to add
    assert again.joint_covered is True and len(asked) == 2 and asked[1] == ["zzz:1"]


def test_a_benchmark_suite_named_in_the_question_is_covered_by_its_member_tasks(store):
    """GLUE tables are recorded per task (CoLA, MRPC ...), never as 'GLUE': "DistilBERT on GLUE" must not get a joint warning."""
    store.add_many([prof("glue:1", 11, model="DistilBERT", dataset="CoLA"), prof("glue:1", 12, model="DistilBERT", dataset="MRPC")])
    _, result = agent(store).run({"model": "DistilBERT", "dataset": "GLUE"}, [rc("glue:1")], NONE, "q")
    assert result.joint_covered is True and result.warning is None and result.coverage == 1.0
    _, other = agent(store).run({"model": "DistilBERT", "dataset": "SQuAD"}, [rc("glue:1")], NONE, "q")
    assert other.joint_covered is False                                                     # a task of another benchmark does not count


def test_a_second_model_that_no_paper_records_on_the_dataset_is_reported(store):
    """"Compare mBERT, XLM-R and GPT-4 on XNLI": mBERT and XLM-R are recorded on XNLI, GPT-4 is in no paper."""
    requested = {"model": "mBERT", "dataset": "XNLI"}
    _, result = agent(store).run(requested, [rc("xnli:1")], NONE, "q", extras={"model": ["GPT-4", "XLM-R"]})
    assert result.missing == ["model=GPT-4"] and result.coverage == pytest.approx(3 / 4)
    assert [c.requested for c in result.checks if c.condition == "model"] == ["mBERT", "GPT-4", "XLM-R"]
    assert next(c for c in result.checks if c.requested == "XLM-R").covered is True
    assert "model = GPT-4 together with dataset = XNLI" in result.warning and "XLM-R" in result.warning
    assert "model = XLM-R" not in result.warning                                          # the covered one is not blamed


def test_a_second_language_is_checked_with_the_model_and_dataset(store):
    requested = {"model": "XLM-R", "dataset": "XNLI", "language": "Hindi"}
    _, result = agent(store).run(requested, [rc("xnli:1")], NONE, "q", extras={"language": ["Kannada", "English"]})
    assert result.missing == ["language=Kannada"] and result.joint_covered is True       # Hindi itself is recorded together
    assert "English" in result.warning and "language = Kannada together with model = XLM-R, dataset = XNLI" in result.warning


def test_covered_extras_add_their_chunks_to_the_evidence_and_the_flag_switches_the_check_off(store):
    asked = []

    def fetch(kept, ids):
        asked.append(ids)
        return kept

    requested = {"model": "mBERT", "dataset": "XNLI"}
    _, result = agent(store).run(requested, [rc("xnli:1")], NONE, "q", joint_fetch=fetch, extras={"model": ["XLM-R"]})
    assert result.coverage == 1.0 and result.warning is None and ["xnli:1"] in asked
    _, off = agent(store, joint=False).run(requested, [rc("xnli:1")], NONE, "q", extras={"model": ["GPT-4"]})
    assert off.warning is None and len(off.checks) == 2                                   # extras are part of the joint feature


def test_an_extra_that_repeats_a_named_condition_is_not_added_twice(store):
    _, result = agent(store).run({"model": "mBERT", "dataset": "XNLI"}, [rc("xnli:1")], NONE, "q", extras={"model": ["mbert"]})
    assert len(result.checks) == 2


def test_fewer_than_two_named_conditions_are_not_checked_jointly(store):
    _, result = agent(store).run({"language": "Kannada"}, EVIDENCE, NONE, "q")
    assert result.joint_covered is None and result.warning is None
    _, result = agent(store).run({"language": "Kannada", "task": "natural language inference"}, EVIDENCE, NONE, "q")   # task is not a key field
    assert result.joint_covered is None


def test_a_condition_that_is_missing_on_its_own_is_reported_by_the_joint_verdict_too(store):
    _, result = agent(store).run({"model": "XLM-R", "language": "Tamil"}, EVIDENCE, NONE, "q")
    assert result.missing == ["language=Tamil"] and result.joint_covered is False


def test_the_joint_verdict_overrides_what_retrieval_happened_to_miss(store):
    """XLM-R on XNLI IS recorded (English, Hindi) but the retrieved chunks lack an XNLI chunk, so the per-condition check says
    "dataset not covered"; the question is really about the language, and the warning must say so."""
    requested = {"model": "XLM-R", "dataset": "XNLI", "language": "Kannada"}
    _, result = agent(store).run(requested, [rc("indic:1")], NONE, "q")                     # only the Kannada chunk was retrieved
    assert result.missing == ["language=Kannada"] and result.joint_covered is False
    assert next(c for c in result.checks if c.condition == "dataset").covered is True
    assert "English" in result.warning


def test_a_named_setting_must_be_recorded_together_with_the_model_and_the_dataset(tmp_path):
    """Found 11 Oct: "LLaMA 65B on HellaSwag in the 5-shot setting" was covered by any passage that says 5-shot (the paper has 5-shot MMLU), though the
    model's HellaSwag results are zero-shot only."""
    s = ProfileStore(tmp_path / "s.sqlite")
    s.add_many([prof("llama:1", 1, model="LLaMA 65B", dataset="HellaSwag", setting="zero-shot"),
                prof("llama:1", 2, model="LLaMA 65B", dataset="NaturalQuestions", setting="5-shot"),
                prof("bert:1", 3, model="BERT-base", dataset="MNLI", setting="fine-tuned")])
    evidence = [rc("llama:1"), rc("bert:1")]
    asked = {"model": "LLaMA 65B", "dataset": "HellaSwag", "setting": "5-shot"}
    _, result = ApplicabilityAgent(s, 0, llm=None, joint=True).run(asked, evidence, NONE, "q")
    assert result.joint_covered is False and result.missing == ["setting=5-shot"]
    assert "setting = 5-shot together with model = LLaMA 65B, dataset = HellaSwag" in result.warning and "zero-shot" in result.warning
    _, ok = ApplicabilityAgent(s, 0, llm=None, joint=True).run({**asked, "dataset": "NaturalQuestions"}, evidence, NONE, "q")
    assert ok.joint_covered is True and ok.warning is None
    _, zero = ApplicabilityAgent(s, 0, llm=None, joint=True).run({**asked, "setting": "zero-shot"}, evidence, NONE, "q")
    assert zero.joint_covered is True
    _, off = ApplicabilityAgent(s, 0, llm=None, joint=True, joint_setting=False).run(asked, evidence, NONE, "q")
    assert off.joint_covered is True and off.warning is None          # the switch restores the 10 Oct behaviour
    _, bert = ApplicabilityAgent(s, 0, llm=None, joint=True).run({"model": "BERT-base", "dataset": "MNLI", "setting": "zero-shot"}, evidence, NONE, "q")
    assert bert.joint_covered is False and bert.missing == ["setting=zero-shot"]


def test_only_a_regime_setting_joins_the_key_a_split_is_left_to_the_per_condition_check(tmp_path):
    """Probed 11 Oct: "MuRIL on IndicXNLI for Kannada on the test set" has a stored result without a setting; a split is unrecorded in many tables."""
    from cgrag.pipeline.conditions import is_regime_setting
    assert all(is_regime_setting(x) for x in ("zero-shot", "5-shot", "few-shot", "fine-tuned", "translate-train", "five shot"))
    assert not any(is_regime_setting(x) for x in ("test set", "dev set", "single model", "ensemble", None, ""))
    s = ProfileStore(tmp_path / "s.sqlite")
    s.add_many([prof("indic:1", 1, model="MuRIL", dataset="IndicXNLI", language="Kannada")])             # no setting recorded
    _, result = ApplicabilityAgent(s, 0, llm=None, joint=True).run(
        {"model": "MuRIL", "dataset": "IndicXNLI", "language": "Kannada", "setting": "test set"}, [rc("indic:1")], NONE, "q")
    assert result.joint_covered is True                                                              # the setting is not part of the joint key
    _, strict = ApplicabilityAgent(s, 0, llm=None, joint=True).run(
        {"model": "MuRIL", "dataset": "IndicXNLI", "language": "Kannada", "setting": "zero-shot"}, [rc("indic:1")], NONE, "q")
    assert strict.joint_covered is False and strict.missing == ["setting=zero-shot"]


def test_cross_lingual_transfer_is_zero_shot_for_matching_but_not_a_stage_7_tag():
    """Probed 11 Oct: mBERT on XNLI for Hindi "in the zero-shot setting" was reported as not recorded; the XLM-R paper's block heading is "Cross-lingual Transfer"."""
    from cgrag.pipeline.conditions import setting_tags, values_match
    assert values_match("setting", "zero-shot", "Cross-lingual Transfer")
    assert values_match("setting", "zero-shot", "zero-shot cross-lingual transfer")
    assert not values_match("setting", "zero-shot", "translate-train cross-lingual transfer")
    assert not values_match("setting", "zero-shot", "fine-tuned dev set")
    assert not setting_tags("Cross-lingual Transfer")          # Stage 7 compares tags: an unrecorded setting stays unrecorded there

