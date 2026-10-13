"""Stage A/B/C on tables the layout model recognised: captions, header applied in code, tables kept whole (no PDF, no LLM)."""
from cgrag.ingestion.chunker import chunk_paper
from cgrag.ingestion.pdf_loader import Element, LoadedPaper, match_captions, pipe_table
from cgrag.ingestion.profile_extractor import _views, wants_extraction
from cgrag.ingestion.tables import is_structured_table, parse_table, split_table_text, table_views
from cgrag.schemas import Chunk

NER = """Table 2: Results on named entity recognition on CoNLL-2002 and CoNLL-2003 (F1 score).
| Model | train | #M | en | nl | es | de | Avg |
| Lample et al.(2016) | each | N | 90.74 | 81.74 | 85.75 | 78.76 | 84.25 |
| Akbik et al.(2018) | each | N | 93.18 | 90.44 | - | 88.27 | - |
| mBERT | each | N | 91.97 | 90.94 | 87.38 | 82.82 | 88.28 |
|  | en | 1 | 91.97 | 77.57 | 74.96 | 69.56 | 78.52 |
|  | each | N | 92.25 | 90.39 | 87.99 | 84.60 | 88.81 |
| XLM-RBase | en | 1 | 92.25 | 78.08 | 76.53 | 69.60 | 79.11 |
|  | all | 1 | 91.08 | 89.09 | 87.28 | 83.17 | 87.66 |"""


def test_markdown_table_becomes_clean_pipe_rows():
    md = "|**Model**|**en**|\n|---|---|\n|BERT<sup>_†_</sup><br>|**91.3**|\n|_XLM-R_|80.6 / 67.8|"
    assert pipe_table(md) == "| Model | en |\n| BERT | 91.3 |\n| XLM-R | 80.6 / 67.8 |"


def test_caption_goes_to_the_nearest_table_whether_it_sits_above_or_below():
    tables = [(70, 60, 520, 250), (70, 350, 290, 460)]
    below = [(70, 268, 520, 320), (70, 474, 290, 530)]
    assert match_captions(tables, below) == {0: 0, 1: 1}
    above = [(70, 40, 520, 58), (70, 325, 290, 345)]
    assert match_captions(tables, above) == {0: 0, 1: 1}
    assert match_captions([(70, 350, 290, 460)], [(300, 474, 520, 530)]) == {}
    stacked = [(82, 67, 280, 232), (81, 303, 281, 429)]
    assert match_captions(stacked, [(72, 251, 291, 285), (72, 448, 291, 470)]) == {0: 0, 1: 1}
    assert match_captions([(70, 60, 520, 250)], [(70, 600, 520, 640)]) == {}


def test_the_header_is_applied_to_every_number_in_code():
    views = table_views(parse_table(NER))
    text = "\n".join(views)
    assert "named entity recognition" in views[0] and "Columns: Model | train | #M | en | nl | es | de | Avg" in views[0]
    assert "Lample et al.(2016): train = each; #M = N; en = 90.74; nl = 81.74; es = 85.75; de = 78.76; Avg = 84.25" in text
    assert "Akbik et al.(2018): train = each; #M = N; en = 93.18; nl = 90.44; de = 88.27" in text


def test_rows_under_a_shared_group_label_take_the_nearest_label():
    text = "\n".join(table_views(parse_table(NER)))
    assert "XLM-RBase: train = each; #M = N; en = 92.25; nl = 90.39" in text
    assert "XLM-RBase: train = all; #M = 1; en = 91.08" in text
    assert "mBERT: train = en; #M = 1; en = 91.97; nl = 77.57" in text


def test_group_label_rows_become_context_for_the_rows_below():
    t = "Table 5: BERT-BASE.\n| Model | D | en | fr |\n| Monolingual baselines |  |  |  |\n| BERT | Wiki | 84.5 | 78.6 |\n| Multilingual (translate-train-all) |  |  |  |\n| XLM-7 | CC | 87.2 | 82.5 |"
    text = "\n".join(table_views(parse_table(t)))
    assert text.index("[Monolingual baselines]") < text.index("BERT: D = Wiki") < text.index("[Multilingual (translate-train-all)]") < text.index("XLM-7: D = CC")


def test_a_group_label_cut_at_column_borders_is_glued_back_together():
    t = ("Table 1: XNLI accuracy.\n| Model | #M | en | fr |\n| _Fine-tune multilingual model o | n English tra | ining set (Cross-lingual Transfer) |  |\n"
         "| XLM-R | 1 | 89.1 | 84.1 |")
    assert "[Fine-tune multilingual model on English training set (Cross-lingual Transfer)]" in "\n".join(table_views(parse_table(t))).replace("_", "")


def test_two_row_header_is_combined():
    t = "Table 3: SQuAD.\n| System | Dev |  | Test |  |\n|  | EM | F1 | EM | F1 |\n| Human | - | - | 82.3 | 91.2 |\n| BERT | 84.2 | 91.1 | 85.1 | 91.8 |"
    text = "\n".join(table_views(parse_table(t)))
    assert "Human: Test EM = 82.3; Test F1 = 91.2" in text
    assert "BERT: Dev EM = 84.2; Dev F1 = 91.1; Test EM = 85.1; Test F1 = 91.8" in text


def test_a_wide_table_is_read_in_calls_that_fit_the_profile_limit():
    header = "| Model | " + " | ".join(f"l{i}" for i in range(16)) + " |"
    rows = "\n".join(f"| M{r} | " + " | ".join(f"{70 + r}.{i}" for i in range(16)) + " |" for r in range(6))
    views = table_views(parse_table(f"Table 1: accuracy per language.\n{header}\n{rows}"), max_results=32)
    assert len(views) == 3 and all(v.count("=") <= 32 for v in views)
    assert all("Columns:" in v and "accuracy per language" in v for v in views)


def test_only_loader_written_tables_are_structured():
    assert is_structured_table(NER)
    assert not is_structured_table("System A B\n\nx 1.1 2.2\ny 3.3 4.4")
    assert not is_structured_table("| Model | Size |\n| BERT | large |")
    assert parse_table("just some prose about BERT 84.6") is None


def test_the_extractor_reads_a_structured_table_through_its_header_and_wants_it_in_any_body_section():
    chunk = Chunk(chunk_id="p:0001", paper_id="p", page=1, section="other", text=NER)
    assert wants_extraction(chunk)
    assert not wants_extraction(chunk.model_copy(update={"section": "references"}))
    assert _views(NER) and all("Columns:" in v and "named entity recognition" in v for v in _views(NER))


def test_a_recognised_table_is_a_chunk_of_its_own_with_its_caption():
    prose = "We evaluate on named entity recognition. " * 4
    paper = LoadedPaper("p", "T", 2, [
        Element(1, "5 Results", "heading"), Element(1, prose, "text"), Element(1, NER, "table"), Element(1, prose, "text")])
    chunks = chunk_paper(paper)
    tables = [c for c in chunks if c.text.startswith("Table 2:")]
    assert len(tables) == 1 and tables[0].text == NER and tables[0].section == "results"
    assert all("| Model |" not in c.text for c in chunks if c is not tables[0])


def test_a_blank_first_cell_stays_blank_so_the_columns_do_not_shift():
    md = "|**Model**|**train**|**en**|\n|---|---|---|\n||each|92.25|\n|XLM-R|en|92.92|"
    assert pipe_table(md) == "| Model | train | en |\n|  | each | 92.25 |\n| XLM-R | en | 92.92 |"


def test_a_decimal_point_set_in_italics_is_repaired():
    assert pipe_table("|a|b|\n|-|-|\n|x 83 _._ 6%|1.5|") == "| a | b |\n| x 83.6% | 1.5 |"


def test_a_model_name_cut_over_two_rows_is_put_back_together():
    t = "Table 2: F1.\n| Model | train | en | nl |\n| BERT | each | 91.97 | 90.94 |\n| m | en | 91.97 | 77.57 |\n| XLM-R | each | 92.9 | 92.5 |"
    text = "\n".join(table_views(parse_table(t)))
    assert "mBERT: train = each" in text and "mBERT: train = en" in text and "XLM-R: train = each" in text


def test_statistics_and_model_size_tables_are_not_sent_to_the_llm():
    from cgrag.ingestion.tables import is_statistics_table
    assert is_statistics_table("Table 6: Languages and statistics of the CC-100 corpus. We report the number of tokens.")
    assert is_statistics_table("Table 7: Details on model sizes. We show the number of layers and parameters.")
    assert not is_statistics_table("Table 1: Results on cross-lingual classification. We report the accuracy.")
    assert not is_statistics_table("Table 3: Statistics of the datasets and F1 results of each system.")
    assert not is_statistics_table("")
    chunk = Chunk(chunk_id="p:0002", paper_id="p", page=1, section="other",
                  text="Table 6: Languages and statistics of the corpus.\n| ISO | Tokens(M) | Size |\n| af | 242 | 1.3 |\n| am | 68 | 0.8 |")
    assert not wants_extraction(chunk)


def test_only_a_whole_line_section_name_is_an_unnumbered_heading():
    from cgrag.ingestion.sections import is_bare_section_title
    assert all(is_bare_section_title(t) for t in ("Conclusion", "Related Work", "References", "Results and Discussion", "Appendix"))
    assert not is_bare_section_title("Multilingual Masked Language Models")
    assert not is_bare_section_title("Model Architectures and Sizes")


def test_a_count_of_languages_is_not_a_model_size_and_avg_is_not_a_language():
    from cgrag.ingestion.profile_extractor import _normalize
    from cgrag.schemas import ExtractedProfile
    p = _normalize(ExtractedProfile(model="XLM-R", model_size="100", language="Avg", metric="accuracy", value=80.9))
    assert p.model_size is None and p.language is None
    q = _normalize(ExtractedProfile(model="XLM-R", model_size="Base", language="Hindi", metric="accuracy", value=1.0))
    assert (q.model_size, q.language) == ("Base", "Hindi")
    assert _normalize(ExtractedProfile(model="T5", model_size="11B", metric="F1", value=1.0)).model_size == "11B"


def test_pieces_of_row_labels_and_non_results_are_not_profiles():
    from cgrag.ingestion.profile_extractor import _is_result, _normalize
    from cgrag.schemas import ExtractedProfile as P
    for junk in ("L", "MA", "Fl", "lama", "acon", "eQs", "base", "9"):
        assert not _is_result(P(model=junk, metric="accuracy", value=1.0)), junk
    for real in ("T5", "mT5", "GPT-3", "ELMo", "Llama 2", "mBERT", "ada", "Human", "BERT-large", "L2"):
        assert _is_result(P(model=real, metric="accuracy", value=1.0)), real
    assert not _is_result(P(model="LLaMA-7B", metric="Carbon emitted (tCO2eq)", value=14.0))
    assert not _is_result(P(model="ReCoRD", dataset="ReCoRD", metric="f1", value=1.0))


def test_a_split_in_the_dataset_field_moves_to_setting_and_a_metric_repeating_the_dataset_becomes_score():
    from cgrag.ingestion.profile_extractor import _normalize
    from cgrag.schemas import ExtractedProfile as P
    p = _normalize(P(model="Sliding Window", dataset="Test", metric="F1", value=19.7))
    assert p.dataset is None and p.setting == "test set"
    q = _normalize(P(model="T5", dataset="GLUE", metric="GLUE", value=83.3))
    assert q.metric == "score" and q.dataset == "GLUE"


def test_a_table_whose_numbers_are_split_or_glued_is_flagged_and_not_extracted():
    from cgrag.ingestion.tables import GARBLE_LIMIT, garble_ratio
    bad = ("Table 16: Score achieved on every task.\n| Experiment | Average | MCC | Acc |\n| Baseline | 83.28 | 53.84 9 | 2.68 |\n"
           "| Equal | . 83.83 84.03 | . 56.55 9 | . 2.66 3.12 |\n| Deshuffling | 73.17 | 22.82 8 | 7.16 |\n| BERT-style | 82.96 | 52.49 9 | 2.55 |")
    assert garble_ratio(NER) == 0.0 and garble_ratio(bad) >= GARBLE_LIMIT
    assert wants_extraction(Chunk(chunk_id="p:0005", paper_id="p", page=1, section="other", text=NER))
    assert not wants_extraction(Chunk(chunk_id="p:0006", paper_id="p", page=1, section="other", text=bad))
    assert garble_ratio("| a | b |\n| x | 24 layers, 1024 hidden states |\n| y | 1.5 |") == 0.0
    assert garble_ratio("Table 9: pairs.\n| System | A | B |\n| x | 80.2 / 67.4 | 89.8/- |") == 0.0


def test_a_table_of_contents_is_not_a_table():
    from cgrag.ingestion.pdf_loader import _worth_keeping
    toc = "| 2 Pretraining | 5 |\n| 2.1 | Pretraining Data . . . . . . . . . . . . . . 5 |\n| 2.2 | Training Details . . . . . . . . . . 5 |\n| 3.1 | Setup . . . . . . . . 7 |"
    assert not _worth_keeping(toc, False)
    assert _worth_keeping("| Model | F1 |\n| A | 80.1 |\n| B | 81.2 |\n| C | 82.3 |\n| D | 83.4 |", False)


def test_a_table_pushed_between_the_references_and_an_appendix_heading_is_kept():
    paper = LoadedPaper("p", "T", 2, [
        Element(1, "6 Conclusion", "heading"), Element(1, "We conclude that things work. " * 5, "text"),
        Element(1, "References", "heading"), Element(1, "[1] Devlin et al. 2019. BERT: pre-training of deep transformers.", "text"),
        Element(2, NER, "table"), Element(2, "A Additional Results", "heading"), Element(2, "More results here. " * 10, "text")])
    chunks = chunk_paper(paper)
    assert any(c.text == NER and c.section == "other" for c in chunks)
    assert not any("Devlin" in c.text for c in chunks)


def test_figure_axis_ticks_do_not_make_a_chunk_worth_extracting():
    chunk = Chunk(chunk_id="p:0003", paper_id="p", page=1, section="methods",
                  text="Figure 3: curves.\n\n7 15 30 60 100\n\nNumber of languages\n\n40\n\n70 80 90 100 110")
    assert not wants_extraction(chunk)


def test_a_results_table_filed_under_methods_is_never_the_methods_context_of_the_llm():
    from types import SimpleNamespace
    from cgrag.ingestion.profile_extractor import extract_paper

    prompts: list[str] = []

    class StubLLM:
        cfg = SimpleNamespace(parallel=1, extract_temperature=0.0)

        def chat(self, messages, **kwargs):
            prompts.append(messages[-1]["content"])
            return SimpleNamespace(text='{"profiles": []}', wall_s=0.0, gen_tokens=5)

    setup = "We evaluate zero-shot on common sense benchmarks with the 7B to 65B models. " * 3
    table3 = NER.replace("Table 2:", "Table 3:")
    chunks = [Chunk(chunk_id="p:0001", paper_id="p", page=1, section="methods", text=setup),
              Chunk(chunk_id="p:0002", paper_id="p", page=2, section="methods", text=table3),
              Chunk(chunk_id="p:0003", paper_id="p", page=3, section="results", text=NER)]
    profiles, _ = extract_paper(chunks, StubLLM())
    assert prompts and all("common sense benchmarks" in p.split("RESULTS CHUNK:")[0] for p in prompts)
    assert all("Table 3:" not in p.split("RESULTS CHUNK:")[0] for p in prompts)


def test_a_long_table_is_cut_between_rows_and_every_part_keeps_caption_and_header():
    rows = "\n".join(f"| Model{i} | each | N | 9{i % 10}.1 | 8{i % 10}.2 | 7{i % 10}.3 | 6{i % 10}.4 | 5{i % 10}.5 |" for i in range(60))
    text = "Table 9: long table.\n| Model | train | #M | en | nl | es | de | Avg |\n" + rows
    parts = split_table_text(text, 800)
    assert len(parts) > 2 and all(len(p) <= 900 for p in parts)
    assert all(p.startswith("Table 9: long table.\n| Model | train |") for p in parts)
    assert sum(p.count("| Model") for p in parts) == 60 + len(parts)


T5_MERGED = """Table 14: Performance of our T5 variants on every task we study.
|  | GLUE | CoLA | SST-2 MRPC MRPC | STS-B | STS-B |
| Model | Average | Matthew | 's Accuracy F1 Accuracy | Pearson | Spearman |
| Previous best | 89.4 | 69.2 | 97.1 93.6 91.5 | 92.7 | 92.3 |
| T5-Small | 77.4 | 41.0 | 91.8 89.7 86.6 | 85.6 | 85.0 |
| T5-11B | 90.3 | 71.6 | 97.5 92.8 90.4 | 93.1 | 92.8 |
|  | WMT EnD | e WMT | EnFr WMT EnRo CNN/DM | CNN/DM | CNN/DM |
| Model | BLEU | BL | EU BLEU ROUGE-1 | ROUGE-2 | ROUGE-L |
| Previous best | 33.8 | 43 | .8 38.5 43.47 | 20.30 | 40.63 |
| T5-Small | 26.7 | 36 | .0 26.8 41.12 | 19.56 | 38.35 |"""


def test_merged_columns_of_a_garbled_table_are_split_block_by_block():
    from cgrag.ingestion.tables import GARBLE_LIMIT, garble_ratio, rescue_tables
    assert garble_ratio(T5_MERGED) >= GARBLE_LIMIT and not wants_extraction(Chunk(chunk_id="p:1", paper_id="p", page=1, section="other", text="x")) is None
    pieces = rescue_tables(T5_MERGED)
    assert pieces is not None and len(pieces) == 2
    t = parse_table(pieces[0])
    assert t.header == ["Model", "GLUE Average", "CoLA Matthew's", "SST-2 Accuracy", "MRPC F1", "MRPC Accuracy", "STS-B Pearson", "STS-B Spearman"]
    assert t.rows[1] == ["T5-Small", "77.4", "41.0", "91.8", "89.7", "86.6", "85.6", "85.0"]
    view = table_views(t, 36)[0]
    assert "T5-11B: GLUE Average = 90.3; CoLA Matthew's = 71.6; SST-2 Accuracy = 97.5; MRPC F1 = 92.8; MRPC Accuracy = 90.4" in view


def test_the_extractor_reads_a_rescued_table_and_still_skips_one_that_cannot_be_split():
    chunk = Chunk(chunk_id="p:1", paper_id="p", page=1, section="other", text=T5_MERGED)
    assert wants_extraction(chunk)
    views = _views(T5_MERGED)
    assert views and any("MRPC F1" in v for v in views) and any("WMT EnDe BLEU" in v for v in views)
    ragged = "Table 3: x\n|  | WMT EnD | e WMT | EnFr WMT EnRo CNN/DM | CNN/DM | CNN/DM |\n| Model | BLEU | BL | EU BLEU ROUGE-1 | ROUGE-2 | ROUGE-L |\n" \
             "| A | 33.8 | 43 | .8 38.5 43.47 | 20.30 | 40.63 |\n| B | 26.7 | 36 | .0 26.8 41.12 | 19.56 |  |"
    assert not wants_extraction(Chunk(chunk_id="p:2", paper_id="p", page=1, section="other", text=ragged))


def test_a_table_that_is_read_today_is_never_changed_by_the_rescue():
    from cgrag.ingestion.tables import rescue_tables
    assert rescue_tables(NER) is None


def test_a_block_with_cut_numbers_and_cut_header_words_is_read_as_streams():
    """T5's last block: "43" | ".8" is 43.8, "EnD" | "e" is EnDe, "BL" | "EU" is BLEU, and "WMT" + "EnDe" is one column name."""
    from cgrag.ingestion.tables import rescue_tables
    pieces = rescue_tables(T5_MERGED)
    t = parse_table(pieces[1])
    assert t.header == ["Model", "WMT EnDe BLEU", "WMT EnFr BLEU", "WMT EnRo BLEU", "CNN/DM ROUGE-1", "CNN/DM ROUGE-2", "CNN/DM ROUGE-L"]
    assert t.rows[0] == ["Previous best", "33.8", "43.8", "38.5", "43.47", "20.30", "40.63"]
    assert t.rows[1] == ["T5-Small", "26.7", "36.0", "26.8", "41.12", "19.56", "38.35"]


DEBERTA_MERGED = """Table 5: SuperGLUE test set results.
| Model | BoolQ Acc | CB F1/Acc | COPA Acc | WiC Acc | Average Score |
| RoBERTalarge | 87.1 | 90.5/95.2 | 90.6 | 69.9 | 84.6 |
| Human | 89.0 | 95.8/98.9 | 100.0 | 80.0 | 89.8 |
| DeBERTa1.5_B+SiFT DeBERTa_Ensemble | 90.4 90.4 | 94.9/97.2 95.7/97.6 | 96.8 98.4 | 76.4 77.5 | 89.9 90.3 |"""


def test_two_systems_merged_into_one_line_are_split_into_two_rows():
    """DeBERTa's SuperGLUE table (found 12 Oct): the label holds two names and every cell two values; the row used to be taken for a heading and lost."""
    from cgrag.ingestion.tables import table_cells
    t = parse_table(DEBERTA_MERGED)
    assert [r[0] for r in t.rows] == ["RoBERTalarge", "Human", "DeBERTa1.5_B+SiFT", "DeBERTa_Ensemble"]
    assert t.rows[2] == ["DeBERTa1.5_B+SiFT", "90.4", "94.9/97.2", "96.8", "76.4", "89.9"]
    assert t.rows[3] == ["DeBERTa_Ensemble", "90.4", "95.7/97.6", "98.4", "77.5", "90.3"]
    cells = {(c.row_label, c.column): c.text for c in table_cells(t)}
    assert cells[("DeBERTa_Ensemble", "WiC Acc")] == "77.5"


def test_a_single_system_with_two_numbers_in_a_cell_is_not_split():
    one = "Table 2: x\n| Model | EM F1 | Acc |\n| BERT base | 88.5 81.2 | 90.1 |\n| RoBERTa | 89.0 82.0 | 91.0 |\n| ALBERT | 90.0 83.0 | 92.0 |"
    assert [r[0] for r in parse_table(one).rows] == ["BERT base", "RoBERTa", "ALBERT"]
    two = "Table 2: x\n| Model | A | B | C |\n| X Y | 1.5 2.5 | 3.5 | 4.5 |\n| Z | 1.0 | 2.0 | 3.0 |"
    assert [r[0] for r in parse_table(two).rows] == ["X Y", "Z"]


GPT3_SUPERGLUE = """Table 3.8: Performance of GPT-3 on SuperGLUE compared to fine-tuned baselines and SOTA. All results are reported on the test set.
|  | SuperGLU Average | E BoolQ Accurac | CB y Accurac | CB y F1 | COPA Accuracy | RTE Accuracy |
| Fine-tuned SOTA | 89.0 | 91.0 | 96.9 | 93.9 | 94.8 | 92.5 |
| GPT-3 Few-Shot | 71.8 | 76.4 | 75.6 | 52.0 | 92.0 | 69.0 |
|  | WiC Accuracy | WSC Accuracy | MultiRC Accuracy | MultiRC F1a | ReCoRD Accuracy | ReCoRD F1 |
| Fine-tuned SOTA | 76.1 | 93.8 | 62.3 | 88.2 | 92.5 | 93.3 |
| GPT-3 Few-Shot | 49.4 | 80.1 | 30.5 | 75.4 | 90.2 | 91.1 |"""


def test_header_words_cut_at_a_column_border_are_put_back_together():
    t = parse_table(GPT3_SUPERGLUE)
    assert t is not None and t.header == ["", "SuperGLUE Average", "BoolQ Accuracy", "CB Accuracy", "CB F1", "COPA Accuracy", "RTE Accuracy"]


def test_a_second_header_block_of_data_set_and_measure_names_switches_the_column_names():
    text = "\n".join(table_views(parse_table(GPT3_SUPERGLUE)))
    assert "GPT-3 Few-Shot: SuperGLUE Average = 71.8; BoolQ Accuracy = 76.4" in text
    assert "GPT-3 Few-Shot: WiC Accuracy = 49.4; WSC Accuracy = 80.1; MultiRC Accuracy = 30.5; MultiRC F1a = 75.4" in text
    assert "BoolQ Accuracy = 80.1" not in text and "BoolQ Accuracy = 93.8" not in text


def test_a_row_of_model_names_under_benchmark_columns_is_not_taken_for_a_new_header():
    table = """Table 1: Benchmarks and the models we compare.
|  | XTREME | XTREME-R | XGLUE | IndicXTREME |
| Task coverage | 9 | 10 | 11 | 18 |
|  | mBERT | XLM-R | MuRIL | IndicBERT |
| #Indic / #Total langs. | 11/104 | 15/110 | 16/17 | 23/24 |
| Fertility | 2.8 | 2.2 | 1.7 | 1.7 |"""
    text = "\n".join(table_views(parse_table(table)))
    assert "XTREME = 11/104" in text and "mBERT = 11/104" not in text


def test_a_lone_letter_is_not_moved_onto_a_short_word_or_a_word_the_table_never_spells():
    table = """Table 4: Skills.
| Model | All | Grai L S | ned K ingle-T |
| BiLSTM | 21 | 16 | 16 |
| ELMo | 20 | 14 | 17 |"""
    t = parse_table(table)
    assert t is not None and "Grai L S" in t.header and "ned K ingle-T" in t.header


GLUE_BASELINES = """Table 4: Baseline performance on the GLUE task test sets.
|  |  | Single S | entence | Similar | ity and Par | aphrase | Natura | l Langua | ge Infe | rence |
| Model | Avg | CoLA | SST-2 | MRPC | QQP | STS-B | MNLI | QNLI | RTE | WNLI |
|  |  |  |  | Single- | Task Traini | ng |  |  |  |  |
| BiLSTM | 63.9 | 15.7 | 85.9 | 69.3/79.4 | 81.7/61.4 | 66.0/62.8 | 70.3/70.8 | 75.7 | 52.8 | 65.1 |
| +ELMo | 66.4 | 35.0 | 90.2 | 69.0/80.8 | 85.7/65.6 | 64.0/60.2 | 72.9/73.4 | 71.7 | 50.1 | 65.1 |
|  |  |  |  | Multi- | Task Trainin | g |  |  |  |  |
| BiLSTM | 64.2 | 11.6 | 82.8 | 74.3/81.8 | 84.2/62.5 | 70.3/67.8 | 65.4/66.1 | 74.6 | 57.4 | 65.1 |"""


def test_a_block_heading_under_the_names_row_is_a_heading_not_part_of_the_column_names():
    t = parse_table(GLUE_BASELINES)
    assert t is not None and "Task" not in " ".join(t.header) and "Traini" not in " ".join(t.header)
    text = "\n".join(table_views(t))
    assert "[Single- Task Training]" in text and "[Multi- Task Training]" in text
    single, multi = text.index("[Single- Task Training]"), text.index("[Multi- Task Training]")
    assert single < text.index("+ELMo") < multi < text.rindex("BiLSTM")


def test_a_group_label_row_over_columns_stays_a_header_row():
    table = """Table 2: Dev and test results.
|  | SQuAD | | MNLI | |
| | Single Model | | Ensemble | |
| Model | EM | F1 | m | mm |
| BERT | 84.1 | 90.9 | 86.7 | 85.9 |
| RoBERTa | 88.9 | 94.6 | 90.2 | 90.2 |"""
    t = parse_table(table)
    assert t is not None
    assert any("Single Model" in h for h in t.header) and any("Ensemble" in h for h in t.header)


def test_a_measure_in_brackets_after_a_group_label_joins_the_label_but_not_in_the_names_row():
    grouped = """Table 23: Comparison on reading comprehension.
|  |  | SQUAD | (EM) |  | QUAC | (f1) |
| Model | Size | 0-shot | 1-shot | 5-shot | 0-shot | 1-shot |
| MPT | 7B | 59.5 | 62.8 | 62.7 | 38.0 | 37.7 |"""
    t = parse_table(grouped)
    assert t is not None and "SQUAD (EM) 1-shot" in t.header and "SQUAD (EM) 5-shot" in t.header and "QUAC (f1) 1-shot" in t.header
    names_row = """Table 5: Test set performance on GLUE tasks.
|  | CoLA | SST-2 | MRPC | RTE | (Avg) |
| Google BERT | 59.3 | 95.2 | 88.5/84.3 | 71.1 | 80.4 |
| Our BERT | 58.6 | 93.9 | 90.1/86.6 | 74.7 | 81.1 |"""
    t2 = parse_table(names_row)
    assert t2 is not None and t2.header[-1] == "(Avg)" and t2.header[-2] == "RTE"


def test_the_rest_of_a_word_that_opens_the_next_cell_moves_forward_when_the_word_is_already_whole_in_this_one():
    table = """Table 24: Comparison on AGI Eval (English)
| Model | Size | Avg | LSAT-RC | SAT-en SA | T-en (w/o Psg.) | SAT-math |
| MPT | 7B | 23.5 | 20.8 | 25.2 | 32.5 | 23.6 |"""
    t = parse_table(table)
    assert t is not None and "SAT-en (w/o Psg.)" in t.header and "SAT-en" in t.header and not any(h.startswith("T-en") for h in t.header)


def test_a_blank_header_cell_over_a_smaller_model_does_not_inherit_the_label_of_the_larger_one():
    table = """Table 16: MMLU. Detailed 5-shot results per domain on the test sets.
|  |  | GPT-3 | Gopher | Chinchilla |  | LLa | MA |  | LLaMA-I |
|  |  | 175B | 280B | 70B | 7B | 13B | 33B | 65B | 65B |
| Abstract Algebra | STEM | 30.0 | 25.0 | 31.0 | 29.0 | 34.0 | 32.0 | 34.0 | 31.0 |"""
    t = parse_table(table)
    assert t is not None
    assert "Chinchilla 70B" in t.header and "LLaMA 7B" in t.header and not any("Chinchilla 7B" == h for h in t.header)
    assert "LLaMA 13B" in t.header and "LLaMA 33B" in t.header and "LLaMA 65B" in t.header and "LLaMA-I 65B" in t.header
    assert not any(h.startswith("MA ") or h.startswith("LLa ") for h in t.header)


GLUED = """Table 2: DistilBERT yields to comparable performance on downstream tasks. Comparison on downstream tasks: IMDb (test accuracy) and SQuAD 1.1 (EM/F1 on dev set).
| Model BERT-base DistilBERT DistilBERT (D) | IMDb (acc.) 93.46 92.82 - | SQuAD (EM/F1) 81.2/88.5 77.7/85.8 79.1/86.9 | Model ELMo BERT-base DistilBERT | # param. (Millions) 180 110 66 | Inf. time (seconds) 895 668 410 |

Table 3: DistilBERT is significantly smaller while being constantly faster. Inference time of a full pass of GLUE task STS-B on CPU."""


def test_a_row_of_whole_column_cells_is_unstacked_into_rows_and_the_statistics_group_is_dropped():
    t = parse_table(GLUED)
    assert t is not None and t.header == ["Model", "IMDb (acc.)", "SQuAD (EM/F1)"]
    assert t.rows == [["BERT-base", "93.46", "81.2/88.5"], ["DistilBERT", "92.82", "77.7/85.8"], ["DistilBERT (D)", "-", "79.1/86.9"]]
    assert is_structured_table(GLUED)
    assert wants_extraction(Chunk(chunk_id="1910.01108:0007", paper_id="1910.01108", paper_title="DistilBERT", page=3, section="methods", text=GLUED))
    assert "DistilBERT: IMDb (acc.) = 92.82; SQuAD (EM/F1) = 77.7/85.8" in "\n".join(table_views(t))


def test_a_glued_row_that_is_not_fully_explained_is_left_alone():
    count_mismatch = GLUED.replace("93.46 92.82 -", "93.46 92.82")
    assert parse_table(count_mismatch) is None
    no_label = GLUED.replace("| Model BERT-base DistilBERT DistilBERT (D) |", "| BERT-base DistilBERT DistilBERT (D) |")
    assert parse_table(no_label) is None
    only_statistics = "Table 3: sizes.\n| Model ELMo BERT-base DistilBERT | # param. (Millions) 180 110 66 | Inf. time (seconds) 895 668 410 |"
    assert parse_table(only_statistics) is None
    ordinary = "Table 1: Results.\n| Model | IMDb | SQuAD |\n| BERT | 93.5 | 88.5 |"
    assert parse_table(ordinary).rows == [["BERT", "93.5", "88.5"]]


SCALING = """|  |  |  |  |  |  |  | Zero-Shot |  | One-Shot |  |  | Few-Shot |  |
| Name | Metric | Split | Fine-tun SOTA | e K | Small | Med Large | XL 2.7B 6.7B 13B 175B | Small Med Large | XL 2.7B 6.7B 13B 175B | Small | Med Larg | e XL 2.7B 6.7B 13B 175B ( | 175B test server) |
| HellaSwag | acc | dev | 85.6 | 20 | 33.7 | 43.6 51.0 | 54.7 62.8 67.4 70.9 78.9 | 33.0 42.9 50.5 | 53.5 61.9 66.5 70.0 78.1 | 33.5 | 43.1 51.3 | 54.9 62.9 67.3 71.3 79.3 |  |
| LAMBADA LAMBADA | acc ppl | test test | 68.0 8.63 | 15 15 | 42.7 18.6 | 54.3 60.4 9.09 6.53 | 63.6 67.1 70.3 72.5 76.2 5.44 4.60 4.00 3.56 3.00 | 22.0 47.1 52.6 165.0 11.6 8.29 | 58.3 61.1 65.4 69.0 72.5 6.46 5.53 4.61 4.06 3.35 | 22.0 165.0 | 40.4 63.2 27.6 6.63 | 57.0 78.1 79.1 81.3 86.4 7.45 2.89 2.56 2.56 1.92 |  |
| TriviaQA Wb | acc | dev tt | 68.0 455 | 64 64 | 4.15 177 | 7.61 14.0 320433 | 19.7 31.3 38.7 41.8 64.3 463792773822144 | 4.19 12.9 20.5 256 620851 | 26.5 35.9 44.4 51.3 68.0 915145151190253 | 6.96 546 | 16.3 26.5 126159 | 32.1 42.3 51.6 57.5 71.2 196248277335415 | 71.2 |
| BoolQ | acc | dev | 91.0 | 32 | 49.7 | 60.3 58.9 | 62.4 67.1 65.4 66.2 60.5 | 52.6 61.7 60.4 | 63.7 68.4 68.7 69.0 76.7 | 43.1 | 60.6 62.0 | 64.1 70.3 70.0 70.2 77.5 | 76.4 |"""

CONTINUED = """|  |  |  |  |  |  |  | Zero-Shot |  | One-Shot |  |  | Few-Shot |  |
| CB | acc | dev | 96.9 | 32 | 0.00 | 32.1 8.93 | 19.6 19.6 28.6 19.6 46.4 | 55.4 53.6 53.6 | 48.2 57.1 33.9 55.4 64.3 | 42.9 | 58.9 53.6 | 69.6 67.9 60.7 66.1 82.1 | 75.6 |
| WiC | acc | dev | 76.1 | 32 | 0.00 | 0.00 0.00 | 0.00 0.00 0.00 0.00 0.00 | 50.0 50.3 50.3 | 49.2 49.4 50.3 50.0 48.6 | 49.8 | 55.0 53.0 | 53.0 51.6 53.1 51.1 55.3 | 49.4 |
| ANLI R1 | acc | test | 73.8 | 50 | 33.4 | 34.2 33.4 | 33.4 34.2 32.3 33.2 34.6 | 32.1 31.6 31.9 | 34.6 30.6 31.6 32.7 32.0 | 32.1 | 32.5 30.9 | 32.5 33.5 33.1 33.3 36.8 |  |"""


def _scaling_rows(piece: str) -> dict[tuple[str, str], dict[str, str]]:
    t = parse_table(piece)
    return {(r[0], r[1]): dict(zip(t.header, r)) for r in t.rows}


def test_a_scaling_table_with_merged_size_columns_is_read_row_by_row_and_unclean_rows_are_dropped():
    from cgrag.ingestion.tables import rescue_tables
    pieces = rescue_tables(SCALING)
    assert pieces is not None and len(pieces) == 1
    rows = _scaling_rows(pieces[0])
    assert ("HellaSwag", "acc") in rows and ("BoolQ", "acc") in rows
    assert not any(k[0].startswith("TriviaQA") for k in rows)
    h = rows[("HellaSwag", "acc")]
    assert h["Zero-Shot Small"] == "33.7" and h["Zero-Shot Large"] == "51.0" and h["Zero-Shot 175B"] == "78.9"
    assert h["Few-Shot Large"] == "51.3" and h["Few-Shot 175B"] == "79.3" and h["K"] == "20"
    assert rows[("BoolQ", "acc")]["Few-Shot 175B test server"] == "76.4"
    assert "Few-Shot Large" in parse_table(pieces[0]).header


def test_two_result_rows_printed_as_one_are_split_row_by_row():
    from cgrag.ingestion.tables import rescue_tables
    rows = _scaling_rows(rescue_tables(SCALING)[0])
    acc, ppl = rows[("LAMBADA", "acc")], rows[("LAMBADA", "ppl")]
    assert acc["Zero-Shot 175B"] == "76.2" and ppl["Zero-Shot 175B"] == "3.00" and acc["Zero-Shot 2.7B"] == "67.1" and ppl["Zero-Shot 13B"] == "3.56"
    assert acc["Few-Shot 175B"] == "86.4" and ppl["Few-Shot 175B"] == "1.92" and acc["One-Shot Large"] == "52.6" and ppl["One-Shot Large"] == "8.29"
    assert acc["Fine-tuned SOTA"] == "68.0" and ppl["Fine-tuned SOTA"] == "8.63" and acc["K"] == ppl["K"] == "15"


def test_a_continuation_chunk_gets_the_names_row_of_the_chunk_before_it():
    from cgrag.ingestion.tables import rescue_tables, with_carried_headers
    first, second = with_carried_headers([SCALING, CONTINUED])
    assert first == SCALING
    assert rescue_tables(CONTINUED) is None
    pieces = rescue_tables(second)
    assert pieces is not None
    rows = _scaling_rows(pieces[0])
    assert rows[("CB", "acc")]["Zero-Shot 6.7B"] == "28.6" and rows[("WiC", "acc")]["One-Shot 175B"] == "48.6" and ("ANLI R1", "acc") in rows
    assert with_carried_headers([CONTINUED, SCALING])[1] == SCALING


def test_a_table_that_is_not_a_size_scaling_table_is_not_touched_by_the_scaling_reader():
    from cgrag.ingestion.tables import _scaling_table
    plain = [["Model", "SST-2", "MRPC"], ["BERT", "93.5", "88.9"], ["RoBERTa", "96.4", "90.2"], ["XLNet", "97.0", "90.8"]]
    assert _scaling_table(plain, "Table 1: Results.") is None


def test_a_data_set_name_cut_across_two_header_cells_is_put_back_together_with_its_version():
    table = """Table 6: Results (dev) of baselines and TinyBERT on question answering tasks.
| System | SQuA EM | D 1.1 F1 | SQuA EM | D 2.0 F1 |
| BERTBASE(Teacher) | 80.7 | 88.4 | 74.5 | 77.7 |
| TinyBERT4 | 72.7 | 82.1 | 68.2 | 71.8 |"""
    t = parse_table(table)
    assert t is not None
    assert t.header == ["System", "SQuAD 1.1 EM", "SQuAD 1.1 F1", "SQuAD 2.0 EM", "SQuAD 2.0 F1"]
    plain = "Table 2: Results.\n| System | Abc EM | D 1.1 F1 |\n| X | 1.0 | 2.0 |\n| Y | 3.0 | 4.0 |"
    assert parse_table(plain).header == ["System", "Abc EM", "D 1.1 F1"]


def test_a_two_letter_piece_completes_a_known_data_set_name_in_the_cell_before_it():
    table = """Table 7: Results.
| Model | SQuAD EM | 1.1 F1 | SQu EM | AD 2.0 F1 |
| BERT | 84.1 | 90.9 | 79.0 | 81.8 |
| RoBERTa | 88.9 | 94.6 | 86.5 | 89.4 |"""
    h = parse_table(table).header
    assert h[3] == "SQuAD 2.0 EM" and h[4] == "SQuAD 2.0 F1"


def test_a_bare_number_as_row_label_takes_the_name_of_the_factor_from_the_header():
    table = """Table 7: Performance of the span-corruption objective for different average span lengths.
| Span length | GLUE | CNNDM |
| ⋆Baseline (i.i.d.) | 83.28 | 19.24 |
| 2 | 83.54 | 19.39 |
| 10 | 82.85 | 19.33 |"""
    views = "\n".join(table_views(parse_table(table)))
    assert "Span length 2: GLUE = 83.54; CNNDM = 19.39" in views and "Span length 10: GLUE = 82.85" in views and "Baseline (i.i.d.): GLUE" in views
    index = "Table 6: Results.\n| # | Acc |\n| 1 | 80.1 |\n| 2 | 81.3 |"
    assert "1: Acc = 80.1" in "\n".join(table_views(parse_table(index)))


def test_a_model_family_named_in_the_text_is_written_into_the_size_columns():
    from cgrag.ingestion.profile_extractor import size_family
    from cgrag.ingestion.tables import rescue_tables
    prose = "GPT-3 Small has 125M parameters. GPT-3 Medium has 350M. GPT-3 Large has 760M. GPT-3 175B is the largest. The Table shows Small results."
    assert size_family([prose, "unrelated text"]) == "GPT-3"
    assert size_family(["A Large model beats a Small one. The Base model is fine."]) == ""
    t = parse_table(rescue_tables(SCALING, "GPT-3")[0])
    assert "GPT-3 Small (Zero-Shot)" in t.header and "GPT-3 Medium (Few-Shot)" in t.header and "GPT-3 175B (Few-Shot)" in t.header
    assert "Fine-tuned SOTA" in t.header
    views = " ".join(table_views(t))
    assert "GPT-3 Large (Zero-Shot) = 51.0" in views


COPA_ROW = ("| CB Copa | f1 acc | dev dev | 93.9 94.8 | 32 32 | 0.00 66.0 | 29.3 11.4 68.073.0 | 17.4 22.4 25.1 20.3 42.8 77.076.080.084.091.0 | 60.1 39.8 45.6 62.0 64.066.0 "
            "| 37.5 45.7 28.5 44.6 52.5 74.076.082.086.087.0 | 26.1 67.0 | 40.4 32.6 64.072.0 | 48.3 45.7 44.6 46.0 57.2 77.083.083.086.092.0 | 52.0 92.0 |")


def test_numbers_of_two_digits_and_one_decimal_that_are_glued_are_cut_into_fixed_width_numbers():
    from cgrag.ingestion.tables import rescue_tables
    rows = _scaling_rows(rescue_tables(SCALING + "\n" + COPA_ROW)[0])
    copa, cb = rows[("Copa", "acc")], rows[("CB", "f1")]
    assert copa["Zero-Shot Large"] == "73.0" and copa["Zero-Shot 175B"] == "91.0" and copa["One-Shot Large"] == "66.0" and copa["Few-Shot 175B"] == "92.0"
    assert cb["Zero-Shot Medium"] == "29.3" and cb["Zero-Shot 175B"] == "42.8" and cb["One-Shot Small"] == "60.1"
    assert not any(k[0].startswith("TriviaQA") for k in rows)
