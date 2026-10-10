"""Table grounding (10 Oct 2026): a stored result is checked against its own table cell; the block heading, caption and labels give the conditions."""
from cgrag.config import ContradictionConfig
from cgrag.ingestion.tables import parse_table, table_cells
from cgrag.pipeline.contradiction import classify
from cgrag.pipeline.grounding import block_tags, caption_protocol, caption_size, caption_split, enriched, ground, label_size
from cgrag.schemas import ConditionProfile

# Real chunk text (trimmed): ALBERT's Table 9. The second block heading arrives cut at column borders and ends in a piece that looks like a number ("019)").
ALBERT = """Table 9: State-of-the-art results on the GLUE benchmark. For single-task single-model results, we report ALBERT at 1M steps (comparable to RoBERTa) and at 1.5M steps.
| Models | MNLI | QNLI | QQP | RTE | SST | MRPC | CoLA | STS | WNLI | Avg |
| Single-task single | models on | dev |  |  |  |  |  |  |  |  |
| BERT-large | 86.6 | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 | - | - |
| RoBERTa-large | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 | - | - |
| Ensembles on test | (from lead | erboard | as of Sept. 16, 2 | 019) |  |  |  |  |  |  |
| XLNet | 90.2 | 98.6 | 90.3 | 86.3 | 96.8 | 93.0 | 67.8 | 91.6 | 90.4 | 88.4 |
| RoBERTa | 90.8 | 98.9 | 90.2 | 88.2 | 96.7 | 92.3 | 67.8 | 92.2 | 89.0 | 88.5 |"""

# XLNet's Table 5: the upper block holds STS-B 92.5 and a WNLI of "-", the lower block holds WNLI 92.5 too.
XLNET = """Table 5: Results on GLUE. The upper section shows direct comparison on dev data and the lower section shows comparison with state-of-the-art results on the public leaderboard.
| Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B | WNLI |
| Single-task single | models on de | v |  |  |  |  |  |  |  |
| RoBERTa [21] | 90.2/90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 | - |
| XLNet | 90.8/90.8 | 94.9 | 92.3 | 85.9 | 97.0 | 90.8 | 69.0 | 92.5 | - |
| Multi-task ensemb | les on test (fr | om leader | board as | of Oct 2 | 8, 2019) |  |  |  |  |
| XLNet | 90.9/90.9 | 99.0 | 90.4 | 88.5 | 97.1 | 92.9 | 70.2 | 93.0 | 92.5 |"""

RACE = """Table 10: State-of-the-art results on the SQuAD and RACE benchmarks.
| Models | SQuAD1.1 dev | SQuAD2.0 dev | RACE test(Middle/High) |
| RoBERTa | 94.6/88.9 | 89.4/86.5 | 83.2 (86.5/81.3) |"""

SBERT_CAPTION = ("Table 5: Evaluation of SBERT sentence embeddings using the SentEval toolkit. SentEval evaluates sentence embeddings on different sentence "
                 "classification tasks by training a logistic regression classifier using the sentence embeddings as features.")


def prof(model, dataset, value, setting=None, size=None, chunk="c:1") -> ConditionProfile:
    return ConditionProfile(profile_id=f"{chunk}#{value}", paper_id="p", chunk_id=chunk, model=model, dataset=dataset, metric="accuracy", value=value,
                            setting=setting, model_size=size, language="English")


def test_a_block_heading_cut_at_column_borders_still_starts_a_new_block():
    cells = table_cells(parse_table(ALBERT))
    by_row = {(c.row_label, c.column): c.block for c in cells}
    assert block_tags(by_row[("RoBERTa-large", "QNLI")]) == {"dev", "single"}
    assert block_tags(by_row[("RoBERTa", "QNLI")]) == {"test", "ensemble"}          # before the fix this row kept the heading of the dev block
    assert block_tags("Multi-task ensemb les on test (from leaderboard as of Oct 28, 2019)") == {"test", "ensemble"}
    assert block_tags("Single-task singlemodels ondev") == {"dev", "single"}


def test_grounding_reads_the_split_and_system_kind_from_the_block_heading_not_from_the_stored_setting():
    stored = prof("RoBERTa", "QNLI", 98.9, setting="single-task single-model")            # what the extraction stored for the ensemble row
    g = ground(stored, ALBERT)
    assert g and g.column == "QNLI" and g.row_label == "RoBERTa" and g.setting_tags == {"test", "ensemble"} and not g.suspect
    assert set(enriched(stored, g).setting.split()) == {"ensemble", "test"}                  # the block's tags replace the wrong ones
    dev = ground(prof("RoBERTa-large", "QNLI", 94.7, setting="single-task single-model"), ALBERT)
    assert dev.setting_tags == {"dev", "single"} and dev.size == "large"


def test_two_papers_that_differ_by_dev_against_ensemble_on_test_are_explained_with_grounding_and_not_comparable_without():
    a, b = prof("RoBERTa", "QNLI", 98.9, setting="single-task single-model"), prof("RoBERTa-large", "QNLI", 94.7, setting="single-task single-model", size="large")
    off, on = ContradictionConfig(one_sided_conditions_explain=False), ContradictionConfig(table_grounding=True, one_sided_conditions_explain=False)
    assert classify(a, b, off)[0] == "NOT_COMPARABLE"                                        # size recorded for one card only
    verdict, differing, reason = classify(a, b, on, ground(a, ALBERT), ground(b, ALBERT))
    assert verdict == "EXPLAINED" and differing == ["setting"] and "evaluation setting" in reason


def test_a_card_whose_dataset_is_not_in_its_cell_is_suspect():
    """STS-B 92.5 of XLNet's dev block stored as WNLI (the next column is '-')."""
    hybrid = prof("XLNet", "WNLI", 92.5, setting="single-task singlemodels on dev")
    g = ground(hybrid, XLNET)
    assert g and g.column == "STS-B" and g.suspect == ["dataset"]
    legit = prof("XLNet", "WNLI", 92.5, setting="multi-task ensemble")
    assert not ground(legit, XLNET).suspect and ground(legit, XLNET).column == "WNLI"
    other = prof("RoBERTa", "WNLI", 90.4)
    verdict, _, reason = classify(hybrid, other, ContradictionConfig(table_grounding=True), g, ground(other, ALBERT))
    assert verdict == "NOT_COMPARABLE" and "does not fit its own table cell" in reason


def test_a_part_inside_a_bracket_of_the_cell_is_suspect():
    middle = prof("RoBERTa", "RACE", 86.5)
    g = ground(middle, RACE)
    assert g and g.column.startswith("RACE") and "subset" in g.suspect                      # 83.2 (86.5/81.3): 86.5 is the Middle subset
    assert not ground(prof("RoBERTa", "RACE", 83.2), RACE).suspect


def test_a_variant_row_takes_the_captions_subject_as_the_model():
    text = ("Table 4: Development set results for RoBERTa as we pretrain over more data. RoBERTa matches the architecture of BERTLARGE.\n"
            "| Model | data | SQuAD | MNLI-m | SST-2 |\n| + additional data (3.2) | 160GB | 94.0/87.7 | 89.3 | 95.6 |")
    assert "model" in ground(prof("BERT-large", "SST-2", 95.6), text).suspect                # the row is a RoBERTa variant, not BERT
    assert not ground(prof("RoBERTa", "SST-2", 95.6), text).suspect


def test_generic_row_labels_and_language_columns_say_nothing():
    text = "Table 1: Results.\n| Model | MNLI-m | en | sw |\n| Top | 80.6 | 82.0 | 60.1 |\n| mBERT | 75.0 | 81.9 | 38.9 |"
    assert not ground(prof("TinyBERT", "MNLI-m", 80.6), text).suspect                        # "Top" is a block label, not a model
    assert not ground(prof("mBERT", "XNLI", 81.9), text).suspect                              # a language column does not name the data set


def test_sizes_come_from_labels_and_captions_and_a_bad_stored_size_is_ignored():
    assert label_size("BERT-large") == "large" and label_size("RoBERTa_base") == "base" and label_size("BERTTINY") == "tiny" and label_size("XLNet") is None
    assert caption_size("GLUE test-set results for large models.") == "large" and caption_size("Ablation study of the DeBERTa base model.") == "base"
    assert caption_size("All results are based on a 24-layer architecture.") == "large"
    text = "Table 3: GLUE test-set results for large models.\n| Model | CoLA | SST |\n| ELECTRA | 71.7 | 97.1 |"
    p = prof("ELECTRA", "CoLA", 71.7, size="1.75M")                                          # 1.75M is a number of steps, not a size
    assert enriched(p, ground(p, text)).model_size == "large"


def test_a_caption_for_frozen_embeddings_and_a_caption_split_are_read():
    assert caption_protocol(SBERT_CAPTION) == {"feature-based"} and not caption_protocol("Results on GLUE dev.")
    assert caption_split("Comparison of large models on the GLUE dev set.") == {"dev"} and caption_split("GLUE test-set results") == {"test"}
    assert caption_split("results on the dev and test sets") == frozenset()


def test_without_a_table_or_a_matching_cell_nothing_is_changed():
    p = prof("BERT", "SQuAD", 80.8, setting="baseline")
    assert not ground(p, "BERT-BASE achieved an exact match score of 80.8 on SQuAD.") and enriched(p, ground(p, "no table")) == p
    assert not ground(prof("BERT", "SQuAD", 12.3), ALBERT)
