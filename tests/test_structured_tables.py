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
    below = [(70, 268, 520, 320), (70, 474, 290, 530)]          # ACL style: caption under its table
    assert match_captions(tables, below) == {0: 0, 1: 1}
    above = [(70, 40, 520, 58), (70, 325, 290, 345)]            # caption over its table
    assert match_captions(tables, above) == {0: 0, 1: 1}
    assert match_captions([(70, 350, 290, 460)], [(300, 474, 520, 530)]) == {}    # a caption in the other column belongs to no table
    stacked = [(82, 67, 280, 232), (81, 303, 281, 429)]                          # BERT p.7: Table 3's top is 18pt under Table 2's
    assert match_captions(stacked, [(72, 251, 291, 285), (72, 448, 291, 470)]) == {0: 0, 1: 1}    # caption, not 19pt from its own
    assert match_captions([(70, 60, 520, 250)], [(70, 600, 520, 640)]) == {}      # too far away


def test_the_header_is_applied_to_every_number_in_code():
    views = table_views(parse_table(NER))
    text = "\n".join(views)
    assert "named entity recognition" in views[0] and "Columns: Model | train | #M | en | nl | es | de | Avg" in views[0]
    assert "Lample et al.(2016): train = each; #M = N; en = 90.74; nl = 81.74; es = 85.75; de = 78.76; Avg = 84.25" in text
    assert "Akbik et al.(2018): train = each; #M = N; en = 93.18; nl = 90.44; de = 88.27" in text      # the "-" cells are dropped


def test_rows_under_a_shared_group_label_take_the_nearest_label():
    text = "\n".join(table_views(parse_table(NER)))
    assert "XLM-RBase: train = each; #M = N; en = 92.25; nl = 90.39" in text        # blank label above XLM-RBase's own row
    assert "XLM-RBase: train = all; #M = 1; en = 91.08" in text                    # blank label below it
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
    assert len(views) == 3 and all(v.count("=") <= 32 for v in views)                # 16 numbers per row: two rows per call
    assert all("Columns:" in v and "accuracy per language" in v for v in views)


def test_only_loader_written_tables_are_structured():
    assert is_structured_table(NER)
    assert not is_structured_table("System A B\n\nx 1.1 2.2\ny 3.3 4.4")
    assert not is_structured_table("| Model | Size |\n| BERT | large |")                   # no numbers
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
    assert not is_bare_section_title("Multilingual Masked Language Models")      # a paragraph title inside section 5
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
    assert not _is_result(P(model="ReCoRD", dataset="ReCoRD", metric="f1", value=1.0))          # the row names a dataset


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
    assert garble_ratio("| a | b |\n| x | 24 layers, 1024 hidden states |\n| y | 1.5 |") == 0.0        # words with numbers are not garbling
    assert garble_ratio("Table 9: pairs.\n| System | A | B |\n| x | 80.2 / 67.4 | 89.8/- |") == 0.0        # pair cells are fine


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
    assert not any("Devlin" in c.text for c in chunks)                          # the bibliography itself is still left out


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
    table3 = NER.replace("Table 2:", "Table 3:")                                    # a results table the heading heuristic filed as methods
    chunks = [Chunk(chunk_id="p:0001", paper_id="p", page=1, section="methods", text=setup),
              Chunk(chunk_id="p:0002", paper_id="p", page=2, section="methods", text=table3),
              Chunk(chunk_id="p:0003", paper_id="p", page=3, section="results", text=NER)]
    profiles, _ = extract_paper(chunks, StubLLM())
    assert prompts and all("common sense benchmarks" in p.split("RESULTS CHUNK:")[0] for p in prompts)
    assert all("Table 3:" not in p.split("RESULTS CHUNK:")[0] for p in prompts)      # the table is not offered as "methods"


def test_a_long_table_is_cut_between_rows_and_every_part_keeps_caption_and_header():
    rows = "\n".join(f"| Model{i} | each | N | 9{i % 10}.1 | 8{i % 10}.2 | 7{i % 10}.3 | 6{i % 10}.4 | 5{i % 10}.5 |" for i in range(60))
    text = "Table 9: long table.\n| Model | train | #M | en | nl | es | de | Avg |\n" + rows
    parts = split_table_text(text, 800)
    assert len(parts) > 2 and all(len(p) <= 900 for p in parts)
    assert all(p.startswith("Table 9: long table.\n| Model | train |") for p in parts)
    assert sum(p.count("| Model") for p in parts) == 60 + len(parts)              # every row once, header in every part
