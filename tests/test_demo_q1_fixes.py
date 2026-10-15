"""14 Oct 2026: fix found by the live demo (question "What F1 does BERT-large reach on SQuAD v1.1?"): the writer gave a guessed reason for a difference between
numbers ("explained by different reporting of model parameters (335M vs 334M)"). The prompt rule 6 alone did not stop it (3 of 5 live runs), so a code guard
removes a sentence that names a reason for numbers unless Stage 7's CONFLICTS lines contain those numbers (`[features] strip_guessed_reasons`).
A Stage 7 filter that skipped results of another data set version was tried the same day and REVERTED: it exposed a false GENUINE (ALBERT's Table 6 re-trains
BERT-large for 400k steps; the training budget is not a recorded condition) and Stage 7 is frozen for the paper's numbers."""
from cgrag.config import get_settings
from cgrag.pipeline.generate import SYSTEM, drop_guessed_reasons

CONFLICTS = "- Sources [2] and [4] (explained): Explained difference (66.5 vs 88.7 accuracy): language (Swahili vs English)."


def test_the_writer_is_told_not_to_guess_reasons():
    assert "Never guess why two numbers differ" in SYSTEM
    assert "never a guessed cause" in SYSTEM
    numbers = [line.split(".")[0] for line in SYSTEM.split("\n") if line[:1].isdigit()]
    assert numbers == ["1", "2", "3", "4", "5", "6", "7"]


def test_the_switch_is_on_by_default():
    assert get_settings().features.strip_guessed_reasons is True


def test_a_guessed_reason_is_removed_and_the_results_stay():
    answer = ("BERT-large reaches:\n\n* **90.9 F1** (dev set) [4].\n* **92.2 F1** (dev set, large) [6].\n\n"
              "Note: The difference between 90.9 and 92.2 is explained by different reporting of model parameters (335M vs 334M) [4].")
    out = drop_guessed_reasons(answer, "")
    assert "90.9 F1" in out and "92.2 F1" in out
    assert "explained" not in out and "Note" not in out


def test_a_note_keeps_its_other_sentences():
    answer = "Note: Only dev results are reported here [4]. The gap between 90.9 and 92.2 is due to training conditions [4]."
    out = drop_guessed_reasons(answer, "")
    assert out == "Note: Only dev results are reported here [4]."


def test_a_reason_that_stage_7_gave_stays():
    answer = "The difference between 66.5% and 88.7% accuracy is due to the language being evaluated (Swahili vs. English) [2][4]."
    assert drop_guessed_reasons(answer, CONFLICTS) == answer


def test_a_reason_with_a_number_stage_7_did_not_name_is_removed():
    answer = "The difference between 66.5 and 91.2 is due to the language [2]."
    assert drop_guessed_reasons(answer, CONFLICTS) == ""


def test_a_sentence_without_numbers_and_a_plain_answer_are_untouched():
    answer = "ELECTRA is more compute-efficient because it learns from all input tokens [1].\nIt reaches 94.9 F1 on SQuAD 1.1 dev [2]."
    assert drop_guessed_reasons(answer, "") == answer


def test_citation_numbers_are_not_counted_as_results():
    answer = "This is explained by the setting [4]."
    assert drop_guessed_reasons(answer, "") == answer
