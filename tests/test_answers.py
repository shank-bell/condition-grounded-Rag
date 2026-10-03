"""Answer-quality scoring (the answer-quality test on our own corpus); no LLM or GPU."""
from cgrag.evaluation.answers import answer_numbers, contains_value, lookup_cases, paired, score_system, sign_test
from cgrag.schemas import ConditionProfile


def prof(pid, paper="p1", model="BERT-large", dataset="SQuAD", metric="F1", value=90.9, **kw):
    return ConditionProfile(profile_id=pid, paper_id=paper, chunk_id=f"{paper}:0", model=model, dataset=dataset, metric=metric, value=value, **kw)


def test_numbers_in_an_answer_ignore_citations_and_metric_names():
    assert answer_numbers("BERT-large gets 90.9 F1 [3] and 84.1 EM [1, 2]; BERT-base reaches 88.5%.") == [90.9, 84.1, 88.5]
    assert answer_numbers("The F1 and EM scores are not reported [4].") == []


def test_a_value_counts_when_the_answer_states_it_up_to_rounding():
    assert contains_value("It reaches 90.9 F1 on the dev set [1].", [90.9])
    assert contains_value("About 74.65 accuracy.", [74.7])                    # rounded differently
    assert not contains_value("It reaches 89.1 F1 [1].", [90.9])
    assert not contains_value("The sources do not report it.", [90.9])


def test_lookup_cases_name_the_conditions_and_keep_every_value_of_the_group():
    profiles = [prof("a", value=90.9, setting="dev set"), prof("b", value=91.5, setting="test set"),
                prof("c", model="XLM-R", dataset="XNLI", metric="accuracy", value=79.2, language="Hindi", paper="p2")]
    cases = lookup_cases(profiles, 10, seed=1)
    questions = {c["question"]: c for c in cases}
    assert set(questions) == {"What F1 does BERT-large get on SQuAD?", "What accuracy does XLM-R get on XNLI for Hindi?"}
    assert questions["What F1 does BERT-large get on SQuAD?"]["gold_values"] == [90.9, 91.5]                 # either setting answers it
    assert questions["What accuracy does XLM-R get on XNLI for Hindi?"]["profile_ids"] == ["c"]


def test_lookup_cases_skip_groups_with_many_values_vague_metrics_and_unverified_rows():
    many = [prof(f"m{i}", value=50.0 + i) for i in range(5)]                  # 5 distinct values: a lucky number would be too likely
    vague = [prof("v", model="T5", metric="score", value=26.4, paper="p3")]
    assert lookup_cases(many + vague, 10, seed=1) == []
    good = [prof("g1", value=90.9), prof("g2", model="RoBERTa", value=94.6, paper="p4")]
    only = lookup_cases(good, 10, seed=1, verified={"g2"})                      # the labellers confirmed only g2
    assert [c["profile_ids"] for c in only] == [["g2"]]


def test_the_labellers_questions_become_cases_with_the_decimal_numbers_of_their_facts():
    from cgrag.evaluation.answers import gold_b_cases, is_correct, value_recall
    rows = [{"id": "B-1", "question": "How well do models perform on Kannada NLI?", "expect_warning": False,
             "facts": "IndicXNLI (paper 2212.05409, Table 16): mBERT 58.6, XLM-R 71.5, MuRIL 74.0, 15 languages"},
            {"id": "B-2", "question": "XLM-R on XNLI for Kannada?", "expect_warning": True, "facts": "Kannada is not in XNLI: 99.5"},
            {"id": "B-3", "question": "How does ELECTRA work?", "expect_warning": False, "facts": "replaced token detection"}]
    cases = gold_b_cases(rows)
    assert [c["id"] for c in cases] == ["B-1"]                                  # a question that expects a warning, or states no score, is not a lookup
    assert cases[0]["gold_values"] == [58.6, 71.5, 74.0] and cases[0]["mode"] == "all"       # not the paper id, the table number or the language count
    assert value_recall("mBERT 58.6 and XLM-R 71.5 [1].", cases[0]["gold_values"]) == 2 / 3
    assert is_correct("mBERT gets 58.6 and XLM-R 71.5.", cases[0])              # two of three expected numbers: at least half
    assert not is_correct("mBERT gets 58.6.", cases[0])
    assert is_correct("It reaches 90.9 F1.", {"gold_values": [90.9, 91.5]})     # a store-built question: any one value is enough


def test_the_sign_test_and_the_paired_counts():
    assert sign_test(8, 0) == 2 / 256 and sign_test(5, 5) == 1.0 and sign_test(0, 0) == 1.0
    result = paired([True, True, False, True, False], [True, False, False, False, True])
    assert result["both_right"] == 1 and result["only_a"] == 2 and result["only_b"] == 1 and result["both_wrong"] == 1
    assert result["sign_test_p"] == 1.0


def test_system_scores_summarise_correctness_numbers_and_support():
    rows = [{"correct": True, "numbers": 2, "supported": 1.0, "warned": False, "seconds": 10.0},
            {"correct": False, "numbers": 4, "supported": 0.5, "warned": True, "seconds": 14.0}]
    s = score_system(rows)
    assert s["accuracy"] == 0.5 and s["numbers_per_answer"] == 3.0 and s["claims_supported"] == 0.75 and s["warned"] == 1
    assert s["seconds_per_question"] == 12.0
    closed_book = score_system([{"correct": True, "numbers": 1, "supported": None, "warned": None, "seconds": 3.0}])
    assert closed_book["claims_supported"] is None and closed_book["warned"] is None
