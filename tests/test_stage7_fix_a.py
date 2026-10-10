"""Stage 7 fix A (8 Oct 2026): three classification rules found by error analysis on the 50 AI-annotated pairs; each has a switch."""
from cgrag.config import ContradictionConfig
from cgrag.pipeline.contradiction import classify
from cgrag.schemas import ConditionProfile

OFF = ContradictionConfig(human_rows_not_comparable=False, two_metric_tasks_no_genuine=False, mnli_split_tags=False, one_sided_conditions_explain=False)
ON = ContradictionConfig(one_sided_conditions_explain=False)          # fix A alone; policy B (one-sided conditions) is tested in test_stage7_policy_b.py


def prof(cid: str, **kw) -> ConditionProfile:
    base = dict(profile_id=f"{cid}#0", paper_id=cid, chunk_id=f"{cid}:1", metric="EM", value=80.0, language="English")
    return ConditionProfile(**{**base, **kw})


def test_human_rows_are_not_comparable():
    """SQuAD paper: 77.0 / 86.8 = a second annotator against the others; BERT's 82.3 / 91.2 = a leaderboard figure (pairs P14, P36, P39)."""
    a = prof("1606.05250", model="human", dataset="SQuAD", dataset_version="v1.0", setting="test set", metric="F1", value=86.8)
    b = prof("1810.04805", model="Human", dataset="SQuAD", dataset_version="1.1", setting="baseline", metric="F1", value=91.2)
    assert classify(a, b, OFF)[0] == "EXPLAINED"           # before: the version labels were taken as the explanation
    assert classify(a, b, ON)[0] == "NOT_COMPARABLE"
    c = prof("1806.03822", model="Human", dataset="SQuAD", dataset_version="2.0", setting="false positive error", value=46.4)
    d = prof("1810.04805", model="Human", dataset="SQuAD", dataset_version="2.0", setting="baseline", value=86.9)
    assert classify(c, d, OFF)[0] == "GENUINE" and classify(c, d, ON)[0] == "NOT_COMPARABLE"


def test_a_system_compared_with_itself_is_unchanged_by_the_human_rule():
    a = prof("x", model="BERT-large", dataset="SQuAD", value=90.9)
    b = prof("y", model="BERT-base", dataset="SQuAD", value=88.5)
    assert classify(a, b, ON)[:2] == ("EXPLAINED", ["model_size"])


def test_two_metric_glue_tasks_are_never_genuine():
    """DistilBERT prints 86.2 for ELMo on QQP = the mean of GLUE's accuracy 88.0 and F1 84.3 (pair P23)."""
    a = prof("1804.07461", model="+ELMo", dataset="QQP", setting="dev set", metric="accuracy", value=88.0)
    b = prof("1910.01108", model="ELMo", dataset="QQP", setting="dev set", metric="accuracy", value=86.2)
    assert classify(a, b, OFF)[0] == "GENUINE"
    verdict, _, reason = classify(a, b, ON)
    assert verdict == "NOT_COMPARABLE" and "two official metrics" in reason
    for task in ("MRPC", "STS-B"):
        assert classify(a.model_copy(update={"dataset": task}), b.model_copy(update={"dataset": task}), ON)[0] == "NOT_COMPARABLE"


def test_single_metric_tasks_can_still_be_genuine():
    a = prof("x", model="ELMo", dataset="QNLI", setting="dev set", metric="accuracy", value=75.2)
    b = prof("y", model="ELMo", dataset="QNLI", setting="dev set", metric="accuracy", value=71.1)
    assert classify(a, b, ON)[0] == "GENUINE"


def test_two_metric_rule_does_not_hide_an_explained_difference():
    a = prof("x", model="RoBERTa", dataset="QQP", setting="dev set", metric="accuracy", value=92.2)
    b = prof("y", model="RoBERTa", dataset="QQP", setting="test set", metric="accuracy", value=90.2)
    assert classify(a, b, ON)[:2] == ("EXPLAINED", ["setting"])


def test_mnli_matched_and_mismatched_are_a_split_difference():
    """GLUE paper: +ELMo 73.4 on MNLI mismatched vs DistilBERT's ELMo 68.6 on the dev set (pair P03)."""
    a = prof("1804.07461", model="+ELMo", dataset="MNLI", setting="mismatched", metric="accuracy", value=73.4)
    b = prof("1910.01108", model="ELMo", dataset="MNLI", setting="dev set", metric="accuracy", value=68.6)
    assert classify(a, b, OFF)[0] == "NOT_COMPARABLE"
    assert classify(a, b, ON)[:2] == ("EXPLAINED", ["setting"])
    c = a.model_copy(update={"setting": "matched"})
    d = a.model_copy(update={"setting": "mismatched"})
    assert classify(c, d, ON)[:2] == ("EXPLAINED", ["setting"])
    assert classify(c, c.model_copy(update={"value": 70.0}), ON)[0] == "GENUINE"      # matched vs matched is the same split
