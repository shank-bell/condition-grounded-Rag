"""Stage 7 policy B (10 Oct 2026): a result-changing condition that only one of the two papers records is the probable explanation of the difference
(EXPLAINED, hedged wording, naming that condition) instead of NOT_COMPARABLE. Chosen after the error analysis of both pair sets (a development choice)."""
from cgrag.config import ContradictionConfig
from cgrag.pipeline.contradiction import classify
from cgrag.schemas import ConditionProfile

B_ON = ContradictionConfig(one_sided_conditions_explain=True)
B_OFF = ContradictionConfig(one_sided_conditions_explain=False)      # the 10 Oct behaviour; policy B is ON by default since 11 Oct (with table grounding)


def prof(cid: str, **kw) -> ConditionProfile:
    base = dict(profile_id=f"{cid}#0", paper_id=cid, chunk_id=f"{cid}:1", metric="accuracy", value=80.0, language="English")
    return ConditionProfile(**{**base, **kw})


def test_a_size_recorded_for_one_paper_only_is_the_probable_explanation():
    """RoBERTa 90.2 against RoBERTa_base 84.7 on MNLI-m (pairs H09 / H20): the size is in one profile only."""
    a = prof("1907.11692", model="RoBERTa", dataset="MNLI", setting="dev set", value=90.2)
    b = prof("1910.01108", model="RoBERTa_base", dataset="MNLI", setting="dev set", value=84.7, model_size="base")
    verdict, differing, reason = classify(a, b, B_ON)
    assert verdict == "EXPLAINED" and differing == ["model_size"] and reason.startswith("Probable explanation") and "model size" in reason
    assert classify(a, b, B_OFF)[0] == "NOT_COMPARABLE"                       # the frozen behaviour


def test_two_conditions_recorded_for_one_paper_only_are_both_named():
    a = prof("x", model="BERT", dataset="MRPC", value=87.8, model_size="base", setting="dev set")
    b = prof("y", model="BERT", dataset="MRPC", value=69.45)
    verdict, differing, _ = classify(a, b, B_ON)
    assert verdict == "EXPLAINED" and set(differing) == {"model_size", "setting"}


def test_recorded_and_different_conditions_are_still_a_plain_explanation():
    a = prof("x", model="BERT", dataset="SQuAD", value=90.9, setting="dev set")
    b = prof("y", model="BERT", dataset="SQuAD", value=84.0, setting="test set")
    verdict, differing, reason = classify(a, b, B_ON)
    assert (verdict, differing) == ("EXPLAINED", ["setting"]) and reason.startswith("Explained difference")      # not hedged: both papers record it


def test_nothing_recorded_on_either_side_or_equal_conditions_stays_genuine():
    a = prof("x", model="ELMo", dataset="QNLI", value=75.2, setting="dev set")
    b = prof("y", model="ELMo", dataset="QNLI", value=71.1, setting="dev set")
    assert classify(a, b, B_ON)[0] == "GENUINE"
    c = prof("x", model="ELMo", dataset="QNLI", value=75.2)
    d = prof("y", model="ELMo", dataset="QNLI", value=71.1)
    assert classify(c, d, B_ON)[0] == "GENUINE"                                # no condition recorded anywhere: policy B changes nothing


def test_human_rows_and_two_metric_tasks_are_decided_before_policy_b():
    a = prof("x", model="Human", dataset="SQuAD", value=86.8, setting="test set")
    b = prof("y", model="Human", dataset="SQuAD", value=91.2)
    assert classify(a, b, B_ON)[0] == "NOT_COMPARABLE"                         # fix A rule 1 comes first
