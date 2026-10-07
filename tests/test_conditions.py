from cgrag.pipeline.conditions import (
    claim_text, differing_conditions, metric_key, model_family, parse_params, values_match,
)
from cgrag.schemas import ConditionProfile


def prof(**kw) -> ConditionProfile:
    base = dict(profile_id="p", paper_id="a", chunk_id="a:0", value=90.0)
    return ConditionProfile(**{**base, **kw})


def test_model_match_respects_family_and_size():
    assert values_match("model", "BERT", "BERTLARGE")
    assert values_match("model", "BERT-large", "BERT (Single)")
    assert not values_match("model", "BERT-large", "BERTBASE")
    assert not values_match("model", "BERT", "RoBERTa")
    assert not values_match("model", "BERT", "DistilBERT")
    assert model_family("BERT-base") == "bert"


def test_dataset_version_and_language():
    assert values_match("dataset", "SQuAD", "SQuAD")
    assert values_match("dataset", "MNLI", "MNLI-m")
    assert not values_match("dataset", "SQuAD", "GLUE")
    assert values_match("dataset_version", "v2.0", "2.0")
    assert not values_match("dataset_version", "1.1", "2.0")
    assert values_match("language", "English", "en")
    assert not values_match("language", "Kannada", "English")


def test_size_parsing():
    assert parse_params("340M") == 340
    assert parse_params("1.5B") == 1500
    assert parse_params("large") is None
    assert values_match("model_size", "0.34B", "340M")
    assert not values_match("model_size", "110M", "340M")


def test_metric_key_merges_spellings():
    assert metric_key("Dev F1") == metric_key("F1 score") == "f1"
    assert metric_key("Acc") == metric_key("accuracy")


def test_differing_conditions_names_the_explanation():
    a = prof(model="BERT", dataset="SQuAD", dataset_version="1.1", metric="F1", setting="dev set")
    b = prof(model="BERT", dataset="SQuAD", dataset_version="2.0", metric="F1", setting="dev set", value=81.8)
    assert differing_conditions(a, b) == ["dataset_version"]
    same = b.model_copy(update={"dataset_version": "1.1"})
    assert differing_conditions(a, same) == []


def test_claim_text_is_a_sentence():
    p = prof(model="BERT-large", dataset="SQuAD", dataset_version="1.1", metric="F1", value=90.9, setting="dev set")
    assert claim_text(p) == "BERT-large obtains 90.9 F1 on SQuAD 1.1 (dev set)."


def test_a_task_abbreviation_matches_its_long_form_and_short_equal_words_match():
    assert values_match("task", "NLI", "natural language inference")
    assert values_match("task", "natural language inference", "NLI")
    assert values_match("task", "NER", "NER") and values_match("task", "NER", "named entity recognition")
    assert values_match("task", "translation", "machine translation")
    assert not values_match("task", "NLI", "question answering")
    assert not values_match("task", "QA", "natural language inference")
    assert not values_match("dataset", "XNLI", "IndicXNLI")           # different datasets: no loosening for datasets


def test_a_benchmark_suite_is_covered_by_results_on_its_member_tasks_but_never_equal_to_them():
    from cgrag.pipeline.conditions import covers
    assert covers("dataset", "GLUE", "CoLA") and covers("dataset", "GLUE", "MNLI-m") and covers("dataset", "XTREME", "XNLI")
    assert covers("dataset", "IndicXTREME", "IndicXNLI") and covers("dataset", "SuperGLUE", "BoolQ")
    assert not covers("dataset", "GLUE", "XNLI") and not covers("dataset", "XNLI", "IndicXNLI") and not covers("dataset", "XTREME", "IndicXNLI")
    assert not covers("language", "Hindi", "Tamil") and covers("model", "BERT", "BERT-large")
    assert not values_match("dataset", "GLUE", "CoLA")           # Stage 7 must never compare a GLUE score with a CoLA score


def test_a_parameter_count_after_a_model_name_is_its_size_not_part_of_the_name():
    from cgrag.pipeline.conditions import _model_core
    # 7 Oct: "LLaMA 65B" got a false scope warning although the store has LLaMA 65B results
    assert _model_core("LLaMA 65B") == ("llama", "65b")
    assert _model_core("T5-11B") == ("t5", "11b") and _model_core("OPT-1.3B") == ("opt", "1.3b") and _model_core("Llama 2-Chat 70B") == ("llama2chat", "70b")
    assert values_match("model", "LLaMA 65B", "LLaMA")                 # the size is a field of its own in the store
    assert values_match("model", "LLaMA 65B", "LLaMA-65B") and values_match("model", "LLaMA", "LLaMA 65B")
    assert not values_match("model", "LLaMA 65B", "LLaMA 7B")          # a different size is a different model
    assert not values_match("model", "LLaMA 65B", "Llama 2")
    assert values_match("model", "Llama 2-Chat 70B", "Llama 2-Chat")
    assert not values_match("model", "OPT-1.3B", "OPT-13B")             # the dot matters: 1.3B is not 13B


def test_a_number_that_is_part_of_a_name_is_not_taken_for_a_size():
    from cgrag.pipeline.conditions import _model_core
    assert _model_core("Llama 2") == ("llama2", None) and _model_core("GPT-3.5") == ("gpt35", None)
    assert _model_core("DistilBERT-6L") == ("distilbert6l", None)        # "6L" is a layer count, not a parameter count
    assert _model_core("BERT-large") == ("bert", "large") and _model_core("Our BERT") == ("bert", None)
    assert model_family("LLaMA 65B") == "llama65b"                         # Stage 7 pairing is unchanged on purpose


def test_the_form_of_a_task_a_question_uses_matches_the_name_the_tables_use():
    assert values_match("task", "translating", "translation") and values_match("task", "translate", "machine translation")
    assert values_match("task", "summarizing", "summarization") and values_match("task", "classifying", "classification")
    assert not values_match("task", "translating", "question answering")


def test_a_translation_direction_is_matched_by_either_of_its_languages():
    from cgrag.pipeline.conditions import values_match
    assert values_match("language", "English-to-German", "German")
    assert values_match("language", "English to German", "english")
    assert values_match("language", "en-de", "de")
    assert not values_match("language", "English-to-German", "French")
    assert values_match("language", "Hindi", "Hindi") and not values_match("language", "Hindi", "Tamil")
