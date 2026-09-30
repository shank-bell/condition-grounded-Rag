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


def test_a_translation_direction_is_matched_by_either_of_its_languages():
    from cgrag.pipeline.conditions import values_match
    assert values_match("language", "English-to-German", "German")
    assert values_match("language", "English to German", "english")
    assert values_match("language", "en-de", "de")
    assert not values_match("language", "English-to-German", "French")
    assert values_match("language", "Hindi", "Hindi") and not values_match("language", "Hindi", "Tamil")
