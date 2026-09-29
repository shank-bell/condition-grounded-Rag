"""Regression tests for the parts of stage C that were debugged on real papers (BERT tables)."""
from cgrag.ingestion.pdf_loader import is_table_row
from cgrag.ingestion.profile_extractor import _normalize, _views, salvage_profiles
from cgrag.ingestion.tables import linearize
from cgrag.schemas import ExtractedProfile


def lines(text: str) -> list[str]:
    return text.strip("\n").split("\n")


def test_table_row_detection():
    assert is_table_row("BERTBASE 84.6/83.4 71.2 90.5 93.5")
    assert is_table_row("Human - - 82.3 91.2")            # "-" marks an empty cell
    assert is_table_row("ESIM+GloVe 51.9 52.7")
    assert not is_table_row("We fine-tune for 3 epochs with a learning rate of 3e-5")
    assert not is_table_row("Table 3: results on SQuAD 1.1")


def test_units_line_is_applied_to_each_column():
    raw = lines("""
Dev Set Tasks MNLI-m QNLI MRPC SST-2 SQuAD

(Acc) (Acc) (Acc) (Acc) (F1)

BERTBASE 84.4 88.4 86.7 92.7 88.5
No NSP 83.9 84.9 86.5 92.6 87.9
""")
    out, rows = linearize(raw)
    assert len(rows) == 2
    assert out[rows[0]] == "BERTBASE: MNLI-m (Acc) = 84.4; QNLI (Acc) = 88.4; MRPC (Acc) = 86.7; SST-2 (Acc) = 92.7; SQuAD (F1) = 88.5"


def test_training_size_row_is_not_mistaken_for_the_header():
    raw = lines("""
System MNLI-(m/mm) QQP QNLI

392k 363k 108k
BERTBASE 84.6/83.4 71.2 90.5
""")
    out, rows = linearize(raw)
    assert out[rows[-1]] == "BERTBASE: MNLI-(m/mm) = 84.6/83.4; QQP = 71.2; QNLI = 90.5"


def test_grouped_header_and_empty_cells():
    raw = lines("""
System Dev Test EM F1 EM F1

Top Leaderboard Systems (Dec 10th, 2018)
Human - - 82.3 91.2
BERTLARGE (Sgl.+TriviaQA) 84.2 91.1 85.1 91.8
""")
    out, rows = linearize(raw)
    assert out[rows[0]] == "Human: Test EM = 82.3; Test F1 = 91.2"
    assert out[rows[1]] == "BERTLARGE (Sgl.+TriviaQA): Dev EM = 84.2; Dev F1 = 91.1; Test EM = 85.1; Test F1 = 91.8"


def test_unalignable_rows_are_left_untouched():
    raw = lines("""
Some caption text here

3 768 12 5.84 77.9
6 768 3 5.24 80.6
""")
    out, _ = linearize(raw)
    assert out == raw


def test_row_with_numbers_in_its_label_is_not_linearized():
    raw = lines("""
System Dev Test

ESIM+GloVe 51.9 52.7 ESIM+ELMo 59.1 59.2 OpenAI GPT - 78.0
""")
    out, _ = linearize(raw)
    assert out == raw


def test_long_table_is_read_a_few_rows_at_a_time():
    header = "System MNLI QQP QNLI"
    body = "\n".join(f"Model{i} {80 + i}.1 {70 + i}.2 {90 + i}.3" for i in range(7))
    views = _views(f"{header}\n\n{body}\n\nTable 1: results")
    assert len(views) == 3                                    # 7 rows, 3 per call
    assert all("Table 1: results" in v and header in v for v in views)
    assert sum("Model0:" in v for v in views) == 1            # each row is in exactly one view


def test_short_table_is_a_single_view():
    assert len(_views("System A B\n\nx 1.1 2.2\ny 3.3 4.4")) == 1


def test_truncated_json_keeps_the_complete_profiles():
    text = ('{"profiles": [{"model": "A", "metric": "F1", "value": 90.1}, '
            '{"model": "B", "metric": "F1", "value": 88.2}, {"model": "C", "metr')
    got, cut = salvage_profiles(text)
    assert cut and [p.model for p in got] == ["A", "B"]


def test_complete_json_is_not_reported_as_cut():
    got, cut = salvage_profiles('{"profiles": [{"model": "A", "value": 1.5}]}')
    assert not cut and len(got) == 1


def test_garbage_yields_nothing():
    assert salvage_profiles("not json at all") == ([], True)


def test_each_table_gets_its_own_caption_and_prose_its_own_view():
    table_a = "System A B\n\nx 1.1 2.2\ny 3.3 4.4\n\nTable 1: BLEU on translation"
    table_b = "System C D\n\nz 5.5 6.6\nw 7.7 8.8\n\nTable 2: accuracy on NLI"
    filler = "\n".join(f"filler line {i}" for i in range(12))
    prose = "BERT reaches 91.1 F1 and 84.6 accuracy in our runs."
    views = _views(f"{table_a}\n{filler}\n{table_b}\n{filler}\n{prose}")
    assert len(views) == 3
    first, second, last = views
    assert "BLEU" in first and "accuracy on NLI" not in first and "z:" not in first
    assert "accuracy on NLI" in second and "BLEU" not in second and "x:" not in second
    assert "91.1 F1" in last and "Table" not in last and "x:" not in last


def test_null_like_strings_become_none_and_statistics_are_not_results():
    from cgrag.ingestion.profile_extractor import _is_result
    p = _normalize(ExtractedProfile(dataset="not specified", language="N/A", model="BERT", metric="F1", value=1.0))
    assert p.dataset is None and p.language is None and p.model == "BERT"
    assert _is_result(p)
    assert not _is_result(ExtractedProfile(dataset="XNLI", metric="average number of tokens", model="x", value=18.7))
    assert not _is_result(ExtractedProfile(dataset="SQuAD", metric="unanswerable", value=2.6))          # no system
    assert _is_result(ExtractedProfile(model="Human", metric="F1", value=90.5))


def test_version_and_split_glued_to_the_dataset_name_are_separated():
    p = _normalize(ExtractedProfile(dataset="SQuAD 2.0 test", model="Human", metric="EM", value=86.9))
    assert (p.dataset, p.dataset_version, p.setting) == ("SQuAD", "2.0", "test set")
    q = _normalize(ExtractedProfile(dataset="SQuAD v1.1 dev", dataset_version=None, setting="single", value=1.0))
    assert (q.dataset, q.dataset_version, q.setting) == ("SQuAD", "1.1", "single")      # an explicit setting wins


def test_version_glued_to_dataset_name_is_split():
    p = _normalize(ExtractedProfile(dataset="SQuAD 1.1", value=1.0))
    assert (p.dataset, p.dataset_version) == ("SQuAD", "1.1")
    q = _normalize(ExtractedProfile(dataset="SST-2", value=1.0))
    assert (q.dataset, q.dataset_version) == ("SST-2", None)
    r = _normalize(ExtractedProfile(dataset="SQuAD", dataset_version="2.0", value=1.0))
    assert r.dataset_version == "2.0"
