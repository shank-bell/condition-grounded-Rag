"""Profile repair (ingestion/repair.py) and the benchmark list (knowledge/benchmarks.py): each rule on a small table written like the stored chunks."""
from cgrag.ingestion.repair import _order_languages, fix_name, paper_context, repair, repair_paper
from cgrag.ingestion.tables import parse_table, table_cells
from cgrag.knowledge.benchmarks import is_language, lookup, lookup_cut, task_fits
from cgrag.schemas import ConditionProfile


def _p(**kw) -> ConditionProfile:
    base = dict(profile_id="x:0001#0", paper_id="x", chunk_id="x:0001", value=1.0, metric="accuracy", model="BERT")
    base.update(kw)
    return ConditionProfile(**base)


LLAMA_TABLE = """Table 23: Comparison to open-source models on reading comprehension (SQUAD and QUAC).
| Model | Size | 0-shot | 1-shot |
| MPT | 7B | 59.5 | 62.8 |
|  | 30B | 74.7 | 74.2 |
| Falcon | 7B | 16.4 | 16.0 |
|  | 40B | 72.9 | 73.1 |
|  | 7B | 60.0 | 62.3 |
| L | 13B | 68.9 | 68.4 |
| lama 1 | 33B | 75.5 | 77.0 |
|  | 65B | 79.4 | 80.0 |"""

PROSE = "We compare Llama 1 and Llama 2 with MPT and Falcon. Llama 1 is older than Llama 2. MPT and Falcon are open models."


def test_benchmark_lookup_and_cut_names():
    assert lookup("SQuAD2.0").name == "SQuAD" and lookup("EnDe").task == "machine translation"
    assert lookup_cut("SQuA").name == "SQuAD" and lookup_cut("ellaSwag").name == "HellaSwag"
    assert lookup("NoSuchSet") is None and lookup_cut("abc") is None


def test_task_family_check():
    mrpc = lookup("MRPC")
    assert task_fits("sentiment analysis", mrpc) is False
    assert task_fits("paraphrase detection", mrpc) is True
    assert task_fits("cross-lingual classification", lookup("XNLI")) is True
    assert task_fits("general tasks", lookup("SQuAD")) is None


def test_language_names():
    assert is_language("Kannada") and is_language("hi") and is_language("transliterated Urdu") and is_language("Indian languages")
    assert not is_language("Jewish") and not is_language("STEM")


def test_vertical_group_label_is_joined_and_sizes_keep_groups():
    labels = {c.row: c.row_label for c in table_cells(parse_table(LLAMA_TABLE))}
    assert labels[0] == labels[1] == "MPT"
    assert labels[2] == labels[3] == "Falcon"
    assert {labels[4], labels[5], labels[6], labels[7]} == {"Llama 1"}


def test_garbled_names_are_matched_to_names_the_prose_writes():
    ctx = paper_context("x", [PROSE, LLAMA_TABLE], ["MPT", "Falcon", "Llama 1", "Llama 2"])
    assert fix_name("lama 1", ctx) == "Llama 1"
    assert fix_name("coFaln", ctx) == "Falcon"
    assert fix_name("Fln", ctx) == "Falcon"
    assert fix_name("MPT", ctx) is None


def test_repair_task_from_dataset_and_model_from_row():
    ctx = paper_context("x", [PROSE, LLAMA_TABLE], ["MPT", "Falcon", "Llama 1"])
    p = _p(model="lama 1", value=75.5, dataset="SQuAD", task="SQuAD", metric="EM", setting="1-shot")
    q, fired = repair(p, LLAMA_TABLE, ctx)
    assert q.model == "Llama 1" and q.task == "question answering"
    assert "model_name" in fired and "task" in fired


def test_wrong_task_family_is_replaced_unless_the_caption_says_it():
    ctx = paper_context("x", ["BERT is a model. BERT again."], ["BERT"])
    table = "Table 6: Ablation over BERT model size.\n| Model | MRPC |\n| BERT | 87.8 |"
    q, _ = repair(_p(value=87.8, dataset="MRPC", task="sentiment analysis"), table, ctx)
    assert q.task == "paraphrase identification"
    table2 = "Table 3: Zero-shot performance on Common Sense Reasoning tasks.\n| Model | BoolQ |\n| BERT | 76.5 |"
    q2, _ = repair(_p(value=76.5, dataset="BoolQ", task="common sense reasoning"), table2, ctx)
    assert q2.task == "common sense reasoning"


def test_unstated_split_is_removed_and_stated_split_is_kept():
    ctx = paper_context("x", ["BERT is a model. BERT again."], ["BERT"])
    plain = "Table 2: Results on reading comprehension.\n| Model | QuAC |\n| BERT | 41.0 |"
    q, fired = repair(_p(value=41.0, dataset="QuAC", setting="test set"), plain, ctx)
    assert not q.setting and "setting_split" in fired
    dev = "Table 4: GLUE dev results.\n| Model | MNLI |\n| BERT | 86.6 |"
    q2, _ = repair(_p(value=86.6, dataset="MNLI", setting="dev set"), dev, ctx)
    assert q2.setting == "dev set"
    tt = "Table 5: Results.\n| Model | XNLI |\n| BERT | 70.1 |"
    q3, _ = repair(_p(value=70.1, dataset="XNLI", setting="TRANSLATE TEST"), tt, ctx)
    assert q3.setting == "TRANSLATE TEST"


def test_dataset_split_prefix_cut_name_and_language_column():
    ctx = paper_context("x", ["BERT is a model. BERT again."], ["BERT"])
    table = "Table 5: Ablation over the pre-training tasks.\n| Model | Dev Set SST-2 |\n| BERT | 92.7 |"
    q, _ = repair(_p(value=92.7, dataset="Dev Set SST-2", task="sentiment analysis"), table, ctx)
    assert q.dataset == "SST-2" and "dev set" in (q.setting or "")
    q2, _ = repair(_p(value=79.2, dataset="ellaSwag", task="common sense reasoning"), "", ctx)
    assert q2.dataset == "HellaSwag"
    panx = "Table 11: PANX (F1) Results for each language.\n| Model | ur-tr | avg. |\n| BERT | 68.4 | 57.7 |"
    q3, _ = repair(_p(value=68.4, dataset="ur-tr", task="PANX", metric="F1", language="Urdu"), panx, ctx)
    assert q3.dataset == "WikiAnn" and q3.task == "named entity recognition"


def test_metric_noise_language_and_version():
    ctx = paper_context("x", ["BERT is a model. BERT again."], ["BERT"])
    q, _ = repair(_p(value=34.2, setting="P@1", language="Jewish"), "", ctx)
    assert q.setting is None and q.language is None
    squad = "Table 3: Exact Match (EM) and F1 scores on SQuAD 1.1 and 2.0.\n| System | SQuAD 2.0 dev EM |\n| BERT | 59.8 |"
    q2, _ = repair(_p(value=59.8, dataset="SQuAD", metric="EM", setting="dev set"), squad, ctx)
    assert q2.dataset_version == "2.0"
    sizes = "Table 1: GLUE test results.\n| Model | MRPC 3.5k |\n| BERT | 88.9 |"
    q3, _ = repair(_p(value=88.9, dataset="MRPC", metric="F1"), sizes, ctx)
    assert q3.dataset_version is None


def test_rules_can_be_switched_off():
    ctx = paper_context("x", ["BERT is a model. BERT again."], ["BERT"])
    q, fired = repair(_p(value=1.0, dataset="MRPC", task="sentiment analysis"), "", ctx, off=frozenset({"task"}))
    assert q.task == "sentiment analysis" and fired == []


STACKED = """Table 9: WikiAnn NER F1 scores for each language.
| Model | af | ar | bg | bn |  |
| mBERT | 77.4 | 41.1 | 77.0 | 70.0 |  |
| XLM-R | 78.9 | 53.0 | 81.4 | 78.8 |  |
|  | ka | kk | ko | ml | avg |
| mBERT | 66.4 | 57.2 | 26.3 | 59.4 | 61.2 |
| XLM-R | 71.6 | 56.2 | 60.0 | 67.8 | 65.4 |"""


def test_restated_header_gives_the_second_half_its_own_column_names():
    cols = {c.text: c.column for c in table_cells(parse_table(STACKED))}
    assert cols["77.4"] == "af" and cols["66.4"] == "ka" and cols["60.0"] == "ko" and cols["65.4"] == "avg"


def test_language_is_read_from_the_column_of_a_language_table_only():
    ctx = paper_context("x", ["mT5 and XLM-R are models. mBERT too. mBERT again. XLM-R again."], ["mBERT", "XLM-R"])
    p = _p(model="XLM-R", value=60.0, dataset="WikiAnn", language="Afrikaans", metric="F1")
    q, fired = repair(p, STACKED, ctx)
    assert q.language == "Korean" and "language_column" in fired
    sq = "Table 2: Results on SQuAD.\n| Model | Dev EM | Dev F1 | Te EM | Te F1 |\n| BERT | 84.1 | 90.9 | 85.1 | 91.8 |"
    q2, _ = repair(_p(value=85.1, dataset="SQuAD", language="English"), sq, paper_context("x", ["BERT BERT BERT"], ["BERT"]))
    assert q2.language == "English"


def test_regime_is_read_from_a_heading_with_stray_glyphs():
    table = ("Table 7: XNLI accuracy.\n| Model | en | ar |\n| Translate-tra iin (models finei-tune on Engli ish training di ata plus transli ations "
             "in all tai rget languagei s) |  |  |\n| mT5-XL | 85.5 | 70.0 |")
    ctx = paper_context("x", ["mT5-XL is a model. mT5-XL again."], ["mT5-XL"])
    q, fired = repair(_p(model="mT5-XL", value=70.0, dataset="XNLI", setting="zero-shot"), table, ctx)
    assert q.setting == "translate-train" and "setting_regime" in fired


def test_configuration_label_becomes_the_setting_and_the_papers_own_system_the_model():
    ctx = paper_context("x", ["ALBERT is a model. ALBERT again. ALBERT: A Lite BERT"], ["ALBERT"], title="ALBERT: A Lite BERT")
    table = "Table 4: Ablation study of the embedding size E.\n| Model | MNLI |\n| ALBERT: E = 64 | 80.1 |\n| ALBERT: E = 128 | 81.2 |"
    q, fired = repair(_p(model="ALBERT: E = 64", value=80.1, dataset="MNLI", setting="dev set"), table, ctx)
    assert q.model == "ALBERT" and "E = 64" in (q.setting or "") and "model_own" in fired


def test_composite_names_and_real_row_labels_are_never_replaced():
    ctx = paper_context("x", ["BiLSTM and ELMo. BiLSTM again. ELMo again. mT5-XL and mT5-XXL are models. mT5-XXL again."], ["BiLSTM+ELMo", "mT5-XL", "mT5-XXL"])
    table = "Table 1: Results.\n| Model | MNLI |\n| BiLSTM | 66.7 |\n| +ELMo | 68.6 |\n| mT5-XL | 82.9 |\n| mT5-XXL | 85.0 |"
    q, _ = repair(_p(model="BiLSTM+ELMo", value=68.6, dataset="MNLI"), table, ctx)
    assert q.model == "BiLSTM+ELMo"
    q2, _ = repair(_p(model="mT5-XL", value=82.9, dataset="MNLI"), table, ctx)
    assert q2.model == "mT5-XL"


def test_paper_defined_abbreviations_and_short_names():
    ctx = paper_context("x", ["We use OntoNotes 5.0 (ON5e) and WNUT-16 (WN16). RG is short for ROUGE. The Spearman correlation coefficient (scc) is used."], [])
    assert ctx.abbrevs["on5e"] == "OntoNotes 5.0" and ctx.abbrevs["wn16"] == "WNUT-16" and ctx.abbrevs["rg"] == "ROUGE"
    q, _ = repair(_p(value=1.0, dataset="WN16", metric="scc"), "", ctx)
    assert q.dataset == "WNUT-16" and q.metric == "Spearman correlation coefficient"
    assert lookup("mr") is None and lookup("MR").name == "MR"


def test_store_overlay_leaves_raw_rows_and_can_be_switched_off(tmp_path):
    from cgrag.stores.profile_store import ProfileStore
    raw = [_p(profile_id="a:0001#0", paper_id="a", chunk_id="a:0001", setting="test set"),
           _p(profile_id="a:0001#1", paper_id="a", chunk_id="a:0001", setting="dev set")]
    fixed = [raw[0].model_copy(update={"setting": None}), raw[1]]
    store = ProfileStore(tmp_path / "p.sqlite", repaired=True)
    store.add_many(raw)
    assert store.set_repairs("a", raw, fixed) == 1
    assert [p.setting for p in store.for_paper("a")] == [None, "dev set"]
    assert [p.setting for p in ProfileStore(tmp_path / "p.sqlite", repaired=False).for_paper("a")] == ["test set", "dev set"]
    store.delete_paper("a")
    assert store.repair_count() == 0


STACKED_LANGUAGES = """Table 17: Results on a COPA task. Metric: accuracy.
|  | as | bn | gom | gu | hi | kn |
| mBERT | 53.6 | 52.0 | 50.2 | 51.6 | 49.2 | 49.0 |
| XLMR | 58.0 | 62.6 | 56.4 | 60.7 | 59.9 | 60.8 |
|  |  | or | pa | sa | sat | ta | Avg. |
| mBERT |  | 48.8 | 51.8 | 47.2 | 52.0 | 51.8 | 50.3 |
| XLMR |  | 59.4 | 58.8 | 54.6 | 53.8 | 64.8 | 58.3 |"""


def _stacked_profiles(model: str, values: list[float], languages: list[str | None], first: int) -> list[ConditionProfile]:
    return [_p(profile_id=f"x:0001#{first + k}", chunk_id="x:0001", model=model, value=v, language=lang, dataset="IndicCOPA")
            for k, (v, lang) in enumerate(zip(values, languages))]


def test_repeated_values_in_a_stacked_language_table_are_read_by_order():
    """The extractor gave the second half the first half's names by position ("Dogri" for the gom column's place, which is the pa cell); 51.8 occurs twice in
    mBERT's second row, so the value alone cannot tell pa from ta. The order of the results (ids count along the row) can."""
    profiles = (_stacked_profiles("mBERT", [53.6, 52.0, 50.2, 51.6, 49.2, 49.0], ["Assamese", "Bengali", "Konkani", "Gujarati", "Hindi", "Kannada"], 0)
                + _stacked_profiles("XLMR", [58.0, 62.6, 56.4, 60.7, 59.9, 60.8], ["Assamese", "Bengali", "Konkani", "Gujarati", "Hindi", "Kannada"], 6)
                + _stacked_profiles("mBERT", [48.8, 51.8, 47.2, 52.0, 51.8, 50.3], ["Bengali", "Dogri", "Gujarati", "Hindi", "Kannada", "Maithili"], 12)
                + _stacked_profiles("XLMR", [59.4, 58.8, 54.6, 53.8, 64.8, 58.3], ["Bengali", "Dogri", "Gujarati", "Hindi", "Kannada", "Maithili"], 18))
    got = _order_languages(profiles, {"x:0001": STACKED_LANGUAGES})
    mbert_second = [got.get(i, profiles[i].language) for i in range(12, 18)]
    assert mbert_second == ["Odia", "Punjabi", "Sanskrit", "Santali", "Tamil", None]
    xlmr_second = [got.get(i, profiles[i].language) for i in range(18, 24)]
    assert xlmr_second == ["Odia", "Punjabi", "Sanskrit", "Santali", "Tamil", None]
    assert not any(i < 12 for i in got)


def test_the_order_rule_does_nothing_when_the_numbers_do_not_line_up():
    profiles = _stacked_profiles("mBERT", [48.8, 51.8, 47.2, 52.0], ["Bengali", "Dogri", "Gujarati", "Hindi"], 0)
    assert _order_languages(profiles, {"x:0001": STACKED_LANGUAGES}) == {}
    swapped = (_stacked_profiles("mBERT", [53.6, 52.0, 50.2, 51.6, 49.2, 49.0], ["Assamese", "Bengali", "Konkani", "Gujarati", "Hindi", "Kannada"], 0)
               + _stacked_profiles("mBERT", [51.8, 48.8, 47.2, 52.0, 51.8, 50.3], ["Bengali", "Dogri", "Gujarati", "Hindi", "Kannada", "Maithili"], 12))
    assert _order_languages(swapped, {"x:0001": STACKED_LANGUAGES}) == {}


def test_the_order_rule_can_be_switched_off():
    profiles = (_stacked_profiles("mBERT", [53.6, 52.0, 50.2, 51.6, 49.2, 49.0], ["Assamese", "Bengali", "Konkani", "Gujarati", "Hindi", "Kannada"], 0)
                + _stacked_profiles("mBERT", [48.8, 51.8, 47.2, 52.0, 51.8, 50.3], ["Bengali", "Dogri", "Gujarati", "Hindi", "Kannada", "Maithili"], 12))
    on, counts_on = repair_paper(profiles, {"x:0001": STACKED_LANGUAGES})
    off, counts_off = repair_paper(profiles, {"x:0001": STACKED_LANGUAGES}, frozenset({"language_order"}))
    assert counts_on["language_order"] > 0 and counts_off["language_order"] == 0
    assert [p.language for p in on[6:]] == ["Odia", "Punjabi", "Sanskrit", "Santali", "Tamil", None]


GLUE_TABLE = "Table 1: GLUE test results.\n| Model | CoLA | SST-2 | STS-B |\n| BERT-base | 52.1 | 93.5 | 85.8 |\n| BiLSTM | 17.6 | 82.8 | 70.0 |"


def test_the_measure_of_cola_and_stsb_is_what_the_paper_names_never_an_accuracy():
    """The extractor reads a GLUE column headed only "CoLA" as accuracy; the paper's own text names Matthews correlation (found 11 Oct night)."""
    prose = "We report Matthews correlation for CoLA and Spearman correlations for STS-B, accuracy for the other tasks."
    ctx = paper_context("x", [prose, GLUE_TABLE], ["BERT-base", "BiLSTM"])
    assert ctx.measures == {"cola": "Matthews correlation", "stsb": "Spearman correlation"}
    cola, rules = repair(_p(value=52.1, dataset="CoLA", metric="accuracy", model="BERT-base"), GLUE_TABLE, ctx)
    assert cola.metric == "Matthews correlation" and "metric_paper_measure" in rules
    sts, _ = repair(_p(value=85.8, dataset="STS-B", metric="accuracy", model="BERT-base"), GLUE_TABLE, ctx)
    assert sts.metric == "Spearman correlation"
    sst, _ = repair(_p(value=93.5, dataset="SST-2", metric="accuracy", model="BERT-base"), GLUE_TABLE, ctx)
    assert sst.metric == "accuracy"
    big, _ = repair(_p(value=84.0, dataset="CoLA", metric="accuracy", model="BERT-base"), GLUE_TABLE, ctx)
    assert big.metric == "accuracy"
    quiet = paper_context("x", ["We use the GLUE benchmark.", GLUE_TABLE], ["BERT-base"])
    kept, rules_kept = repair(_p(value=52.1, dataset="CoLA", metric="accuracy", model="BERT-base"), GLUE_TABLE, quiet)
    assert kept.metric == "Matthews correlation" and "metric_paper_measure" in rules_kept
    sts_quiet, _ = repair(_p(value=85.8, dataset="STS-B", metric="accuracy", model="BERT-base"), GLUE_TABLE, quiet)
    assert sts_quiet.metric == "correlation"
    sst_quiet, _ = repair(_p(value=93.5, dataset="SST-2", metric="accuracy", model="BERT-base"), GLUE_TABLE, quiet)
    assert sst_quiet.metric == "accuracy"
    both = paper_context("x", ["Pearson and Spearman correlations are reported.", GLUE_TABLE], ["BERT-base"])
    assert both.measures == {"stsb": "correlation"}
    off, _ = repair(_p(value=52.1, dataset="CoLA", metric="accuracy", model="BERT-base"), GLUE_TABLE, ctx, frozenset({"metric_paper_measure"}))
    assert off.metric == "accuracy"


def test_a_number_that_the_sentence_presents_as_a_difference_is_not_typed_as_a_result():
    """"RACE (+8.4%)" and "outperforms mBERT by 13.0% F1" were stored as results (accuracy 8.4, F1 13.0): 4 of them cost Stage 7 not-comparable pairs (12 Oct)."""
    from cgrag.ingestion.repair import states_a_difference
    assert states_a_difference("RACE (+8.4%)", 8.4) and states_a_difference("XLM-R also outperforms mBERT by 13.0% F1-score", 13.0)
    assert states_a_difference("outperform the previous state of the art by ... 6.3%", 6.3) and states_a_difference("a solid gain of 2.0% F1", 2.0)
    assert states_a_difference("1 point lower top-k accuracy", 1.0)
    assert not states_a_difference("BERT-large obtains 86.7 and outperforms the baseline by 3.5", 86.7)
    assert not states_a_difference("sw = 50.4", 50.4) and not states_a_difference(None, 3.0)
    ctx = paper_context("x", ["ALBERT is a model. ALBERT again."], ["ALBERT-xxlarge"])
    q, rules = repair(_p(model="ALBERT-xxlarge", dataset="RACE", metric="accuracy", value=8.4, evidence="RACE (+8.4%)", chunk_id="x:0016"), "", ctx)
    assert q.metric == "difference in accuracy" and "metric_difference" in rules
    table = "Table 2: RACE results.\n| Model | RACE |\n| ALBERT-xxlarge | 8.4 |\n| BERT | 7.1 |"
    t, _ = repair(_p(model="ALBERT-xxlarge", dataset="RACE", metric="accuracy", value=8.4, evidence="RACE (+8.4%)", chunk_id="x:0016"), table, ctx)
    assert t.metric == "accuracy"


def test_a_cut_metric_word_is_completed_only_when_one_known_metric_fits():
    from cgrag.ingestion.repair import complete_metric
    assert complete_metric("Exac") == "Exact Match" and complete_metric("Match") == "Exact Match"
    assert complete_metric("Spea") == "Spearman correlation" and complete_metric("Pear.") == "Pearson correlation"
    assert complete_metric("Exact") is None and complete_metric("accuracy") is None and complete_metric("Acc") is None
    assert complete_metric("Pre") is None and complete_metric("Corr") is None and complete_metric("BLEU") is None
    q, rules = repair(_p(metric="Exac", model="BERT", value=80.0), "", paper_context("x", ["BERT BERT"], ["BERT"]))
    assert q.metric == "Exact Match" and "metric_cut" in rules


MT5_ABLATION = """Table 14: XQuAD zero-shot F1/EM of various ablations on our mT5-Large model.
| Model | en | ar | de | avg |
| Baseline (mT5-large) | 88.4 / 77.3 | 75.2 / 56.7 | 80.0 / 62.9 | 77.8 / 61.5 |
| Dropout0.1 | 87.3 / 76.0 | 54.9 / 33.9 | 77.6 / 60.2 | 70.4 / 52.1 |
| Sequence length512 | 88.0 / 76.9 | 77.0 / 59.6 | 80.2 / 62.4 | 78.6 / 61.7 |"""
MT5_PROSE = "mT5 is a multilingual variant of T5. We pre-train mT5 on mC4. The mT5 models reach strong results. mT5-Large is our main ablation model. mT5 transfers well."


def test_an_ablation_label_is_a_setting_of_the_paper_s_own_system_not_the_model():
    """mT5's ablation table: the row 'Dropout0.1' is mT5-Large with another dropout, not a model; Stage 7 must see it as an ablation (12 Oct, fresh sample)."""
    ctx = paper_context("x", [MT5_PROSE, MT5_ABLATION], ["mT5", "mT5-Large", "Dropout0.1"], title="mT5: A massively multilingual pre-trained text-to-text transformer")
    assert ctx.own == "mT5" and ctx.main == "mT5"
    kept, rules = repair(_p(value=33.9, model="mT5-Large", dataset="XQuAD", metric="EM", language="Arabic", setting="zero-shot", chunk_id="x:0001"), MT5_ABLATION, ctx)
    assert kept.model == "mT5-Large" and kept.setting == "zero-shot; ablation: Dropout0.1" and "setting_ablation" in rules
    as_label, _ = repair(_p(value=33.9, model="Dropout0.1", dataset="XQuAD", metric="EM", language="Arabic", setting="zero-shot"), MT5_ABLATION, ctx)
    assert as_label.model == "mT5" and as_label.setting == "zero-shot; ablation: Dropout0.1"
    base, rules_base = repair(_p(value=56.7, model="mT5-Large", dataset="XQuAD", metric="EM", language="Arabic", setting="zero-shot"), MT5_ABLATION, ctx)
    assert base.setting == "zero-shot" and "setting_ablation" not in rules_base
    off, _ = repair(_p(value=33.9, model="Dropout0.1", dataset="XQuAD", metric="EM", language="Arabic", setting="zero-shot"), MT5_ABLATION, ctx,
                    frozenset({"setting_ablation"}))
    assert off.model == "Dropout0.1"


T5_FACTOR = """Table 5: Comparison of variants of the BERT-style pre-training objective.
| Objective | GLUE | CNNDM | SQuAD |
| BERT-style (Devlin et al., 2018) | 82.96 | 19.17 | 80.65 |
| MASS-style (Song et al., 2019) | 82.32 | 19.16 | 80.10 |
| ⋆Replace corrupted spans | 83.28 | 19.24 | 80.88 |"""


def test_a_factor_table_names_its_rows_by_the_thing_that_varies():
    prose = "We study T5, our text-to-text model. " * 10 + "T5-Base is the baseline."
    ctx = paper_context("x", [prose, T5_FACTOR], ["T5-Base", "T5-Large", "T5-11B", "BERT-style (Devlin et al., 2018)"], title="Exploring the limits of transfer learning")
    assert ctx.own is None and ctx.main == "T5"
    q, rules = repair(_p(value=80.65, model="BERT-style (Devlin et al., 2018)", dataset="SQuAD", metric="score", setting="pre-training"), T5_FACTOR, ctx)
    assert q.model == "T5" and "ablation: BERT-style (Devlin et al., 2018)" in q.setting and "setting_ablation" in rules


def test_a_system_row_of_an_ablation_table_is_left_alone():
    table = "Table 9: Ablation on the dev sets of GLUE benchmark.\n|  | MNLI-m | QNLI |\n| BERTLARGE | 86.6 | 92.1 |\n| + PD | 81.1 | 88.9 |"
    ctx = paper_context("x", ["MobileBERT is compact. " * 10, table], ["MobileBERT", "BERTLARGE"], title="MobileBERT: a compact task-agnostic BERT")
    teacher, rules = repair(_p(value=86.6, model="BERTLARGE", dataset="MNLI-m", setting="dev set"), table, ctx)
    assert teacher.model == "BERTLARGE" and "setting_ablation" not in rules
    variant, _ = repair(_p(value=81.1, model="+ PD", dataset="MNLI-m", setting="dev set"), table, ctx)
    assert variant.model == "MobileBERT" and variant.setting == "dev set; ablation: + PD"


def test_a_decimal_point_lost_by_the_layout_is_put_back_only_in_a_table_of_decimals():
    table = """Table 1: Open-domain QA test scores.
| Model | NQ | TQA | FVR-2 Accuracy |
| BART | 15.1 | 19.7 | 81.1 |
| RAG-Tok. | 17.3 | 22.2 | 895 |
| RAG-Seq. | 14.7 | 21.4 | 891 |
| REALM | 40.4 | 57.9 | 76.8 |
| DPR | 41.5 | 57.9 | 73.2 |"""
    ctx = paper_context("x", ["RAG is our model. " * 3, table], ["RAG-Tok.", "BART"], title="Retrieval-augmented generation")
    q, rules = repair(_p(value=895.0, model="RAG-Tok.", dataset="FVR-2", metric="accuracy"), table, ctx)
    assert q.value == 89.5 and "value_lost_decimal" in rules
    kept, _ = repair(_p(value=81.1, model="BART", dataset="FVR-2", metric="accuracy"), table, ctx)
    assert kept.value == 81.1
    ppl, _ = repair(_p(value=895.0, model="RAG-Tok.", dataset="FVR-2", metric="perplexity"), table, ctx)
    assert ppl.value == 895.0


def test_the_second_number_of_a_matched_mismatched_pair_is_the_mismatched_set_and_conll_follows_the_language():
    glue = "Table 4: GLUE dev results.\n| Model | MNLI-m/mm | QNLI |\n| XLM-R | 88.9/89.0 | 93.8 |\n| RoBERTa | 87.6/- | 92.8 |"
    ctx = paper_context("x", ["We use GLUE. " * 3, glue], ["XLM-R"], title="XLM-R")
    first, _ = repair(_p(value=88.9, model="XLM-R", dataset="MNLI-m", metric="accuracy"), glue, ctx)
    second, rules = repair(_p(value=89.0, model="XLM-R", dataset="MNLI-m", metric="accuracy"), glue, ctx)
    assert first.dataset == "MNLI-m" and second.dataset == "MNLI-mm" and "dataset_pair" in rules
    ner = "Table 2: Results on NER on CoNLL-2002 and CoNLL-2003.\n| Model | en | nl | es | de |\n| mBERT | 91.97 | 90.94 | 87.38 | 82.82 |\n| XLM-R | 92.92 | 92.0 | 89.0 | 85.8 |"
    ctx2 = paper_context("x", ["NER. " * 3, ner], ["mBERT", "XLM-R"], title="XLM-R")
    en, _ = repair(_p(value=91.97, model="mBERT", dataset="CoNLL-2002", metric="F1", language="English"), ner, ctx2)
    nl, _ = repair(_p(value=90.94, model="mBERT", dataset="CoNLL-2002", metric="F1", language="Dutch"), ner, ctx2)
    assert en.dataset == "CoNLL-2003" and nl.dataset == "CoNLL-2002"


def test_stage_7_reads_an_ablation_setting_as_something_the_other_paper_does_not_record():
    from cgrag.pipeline.conditions import setting_tags, unrecorded_conditions
    assert "ablation" in setting_tags("zero-shot; ablation: Dropout0.1") and "ablation" not in setting_tags("zero-shot")
    a = _p(value=59.5, model="mT5-Large", dataset="XQuAD", metric="EM", setting="zero-shot; ablation: Dropout0.1")
    b = _p(value=56.7, model="mT5-Large", dataset="XQuAD", metric="EM", setting="zero-shot", paper_id="y", chunk_id="y:0001", profile_id="y:0001#0")
    assert "setting" in unrecorded_conditions(a, b)


def test_a_number_that_two_cells_carry_is_taken_from_the_cell_where_it_stands_at_the_position_of_the_measure():
    """TyDi QA: 37.2 is bn's F1 and fi's EM under "(F1/EM)"; a result stored with F1 and a wrong language could not be corrected (12 Oct, E4)."""
    table = """Table 12: TyDi QA GoldP results (F1/EM) for each language.
| Model | en | ar | bn | fi | avg |
| mT5-Small | 57.1 / 46.6 | 56.8 / 39.7 | 37.2 / 21.2 | 50.9 / 37.2 | 50.5 / 36.2 |"""
    ctx = paper_context("x", ["mT5 is our model. " * 3, table], ["mT5-Small"], title="mT5: a massively multilingual model")
    f1, _ = repair(_p(value=37.2, model="mT5-Small", dataset="TyDi QA GoldP", metric="F1", language="Hindi"), table, ctx)
    em, _ = repair(_p(value=37.2, model="mT5-Small", dataset="TyDi QA GoldP", metric="EM", language="Hindi"), table, ctx)
    assert f1.language == "Bengali" and em.language == "Finnish"


def test_results_averaged_across_languages_have_no_language_even_when_the_columns_are_tasks():
    table = """Table 4: Results averaged across languages from the IndicXTREME benchmark. We report F1 for QA.
|  | Sentiment | XNLI | QA |
| IndicBERT v2 | 88.3 | 73.0 | 47.7 |"""
    ctx = paper_context("x", ["IndicBERT. " * 3, table], ["IndicBERT v2"], title="Towards leaving no Indic language behind")
    q, rules = repair(_p(value=47.7, model="IndicBERT v2", dataset="Indic QA", metric="F1", language="Hindi"), table, ctx)
    assert q.language is None and "language_average" in rules


def test_a_variant_that_the_paper_writes_in_its_prose_is_still_a_variant_unless_a_results_table_has_it_as_a_row():
    prose = "MobileBERT uses Auxiliary Knowledge Transfer (AKT), JKT and PKT. AKT is the simplest. We compare AKT with JKT. " * 3
    ablation = "Table 8: Ablation study of MobileBERT on the GLUE dev sets with AKT, JKT and PKT.\n|  | MNLI-m | SST-2 |\n| AKT | 83.0 | 91.9 |\n| + BiLSTM | 82.1 | 91.6 |"
    results = "Table 3: Results on GLUE.\n| Model | MNLI-m | SST-2 |\n| BERTBASE | 84.6 | 93.5 |\n| MobileBERT | 83.9 | 92.1 |"
    ctx = paper_context("x", [prose, ablation, results], ["MobileBERT", "AKT", "BERTBASE"], title="MobileBERT: a compact task-agnostic BERT")
    assert "bertbase" in ctx.main_rows and "akt" not in ctx.main_rows
    akt, _ = repair(_p(value=91.9, model="AKT", dataset="SST-2", setting="dev set", chunk_id="x:0002"), ablation, ctx)
    assert akt.model == "MobileBERT" and akt.setting == "dev set; ablation: AKT"
    plus, _ = repair(_p(value=91.6, model="+ BiLSTM", dataset="SST-2", setting="dev set", chunk_id="x:0002"), ablation, ctx)
    assert plus.model == "MobileBERT" and "ablation: + BiLSTM" in plus.setting


def test_m_and_mm_are_splits_not_measures_and_a_train_column_does_not_make_a_dev_number_a_train_one():
    table = "Table 5: GLUE.\n| Model | MNLI m/mm | QNLI |\n| DAE | 84.88/84.45 | 92.6 |\n| RoBERTa | 87.6/- | 92.8 |"
    ctx = paper_context("x", ["GLUE. " * 3, table], ["DAE"], title="XLNet")
    q, rules = repair(_p(value=84.45, model="DAE", dataset="MNLI-mm", metric="mm"), table, ctx)
    assert q.metric == "accuracy" and "metric_split_word" in rules
    feat = "Table 6: Performance with feature ablations.\n|  | Train | F1 Dev |\n| Logistic Regression | 91.7% | 50.6% |\n| - Root Match | 91.7% | 49.9% |"
    ctx2 = paper_context("x", ["SQuAD. " * 3, feat], ["Logistic Regression"], title="SQuAD")
    s, _ = repair(_p(value=50.6, model="Logistic Regression", dataset="SQuAD", metric="F1", setting="train dev set"), feat, ctx2)
    assert s.setting == "dev set"


def test_a_figure_after_and_in_a_sentence_that_says_outperforms_is_a_difference_too():
    from cgrag.ingestion.repair import sentence_of, states_a_difference
    chunk = "XLM-R is evaluated on NER. On this task, XLM-R also outperforms mBERT by 2.42 F1 on average for cross-lingual transfer, and 1.86 F1 when trained on each language. Next sentence."
    whole = sentence_of("and 1.86 F1 when trained on each language", chunk)
    assert whole.startswith("On this task") and states_a_difference(whole, 1.86) and states_a_difference(whole, 2.42)
    assert not states_a_difference("XLM-R reaches 89.43 F1 and 1.86 F1 on another set", 89.43)


WMT_TABLE = "Table 8: Ablations.\n| Model | GLUE | EnDe | EnFr |\n| Baseline | 83.28 | 26.98 | 39.82 |\n| No pre-training | 66.22 | 25.86 | 39.77 |"
AGI_TABLE = ("Table 24: Comparison to open source models on AGI Eval (English)\n| Model | Size | Avg | AQuA-RAT | LogiQA | SAT-math |\n"
             "| Falcon | 7B | 21.2 | 21.7 | 22.3 | 26.4 |\n| Falcon | 40B | 37.0 | 18.5 | 36.4 | 32.7 |")


def test_a_wmt_column_that_names_no_measure_is_scored_by_bleu_and_other_data_sets_are_left_alone():
    ctx = paper_context("x", ["We pre-train T5.", WMT_TABLE], ["Baseline"])
    fixed, rules = repair(_p(value=26.98, dataset="WMT En-De", metric="score", model="Baseline"), WMT_TABLE, ctx)
    assert fixed.metric == "BLEU" and "metric_benchmark" in rules
    glue, _ = repair(_p(value=83.28, dataset="GLUE", metric="score", model="Baseline"), WMT_TABLE, ctx)
    assert glue.metric == "score"
    named, _ = repair(_p(value=39.82, dataset="WMT En-Fr", metric="BLEU", model="Baseline"), WMT_TABLE, ctx)
    assert named.metric == "BLEU"
    off, _ = repair(_p(value=26.98, dataset="WMT En-De", metric="score", model="Baseline"), WMT_TABLE, ctx, frozenset({"metric_benchmark"}))
    assert off.metric == "score"


def test_the_avg_metric_of_a_data_set_column_is_not_the_tables_average_column():
    ctx = paper_context("x", ["We evaluate Falcon.", AGI_TABLE], ["Falcon"])
    cell, rules = repair(_p(value=21.7, dataset="AQuA-RAT", metric="Avg", model="Falcon"), AGI_TABLE, ctx)
    assert cell.metric == "score" and "metric_avg_column" in rules
    avg, _ = repair(_p(value=21.2, dataset="AGI Eval", metric="Avg", model="Falcon"), AGI_TABLE, ctx)
    assert avg.metric == "Avg"


def test_the_leaderboard_spellings_of_yelp_and_wmt_are_known_names_of_the_same_benchmarks():
    from cgrag.knowledge.benchmarks import lookup
    assert lookup("Yelp Fine-grained classification").name == lookup("Yelp-5").name == "Yelp"
    assert lookup("WMT2014 English-German").name == lookup("EnDe").name == "WMT En-De"
    assert lookup("WMT2016 English-Romanian").name == "WMT En-Ro" and lookup("WMT2016 German-English").name == "WMT De-En"


SPAN_EFFECTS = """Table 7: The effects of different auxiliary objectives, given MLM over random spans as the primary objective.
|  | SQuAD 2.0 | NewsQA | TriviaQA |
| Span Masking (1seq) | 86.7 | 73.4 | 80.0 |
| + NSP | 86.2 | 73.1 | 79.9 |"""


def test_a_caption_that_names_the_effects_of_something_marks_an_ablation_table():
    prose = "We present SpanBERT, a pre-training method. SpanBERT masks spans. SpanBERT reaches strong results."
    ctx = paper_context("x", [prose, SPAN_EFFECTS], ["SpanBERT", "Span Masking (1seq)", "+ NSP"], title="SpanBERT: Improving Pre-training by Representing and Predicting Spans")
    row, rules = repair(_p(value=80.0, model="Span Masking (1seq)", dataset="TriviaQA", metric="F1", setting="pre-training"), SPAN_EFFECTS, ctx)
    assert row.model == "SpanBERT" and "ablation: Span Masking (1seq)" in (row.setting or "") and "setting_ablation" in rules


def test_a_run_of_header_letters_glued_into_a_model_label_is_cut_off():
    table = "Table 23: Results on FLORES sentence retrieval task. Metric: accuracy.\n|  | as | bn |\n| +Back-Trans | 75.8 | 85.8 |\n| +Samanantar | 70.2 | 80.1 |"
    ctx = paper_context("x", ["IndicBERT is our model.", table], ["IndicBERT", "+Back-Trans"])
    fixed, rules = repair(_p(value=75.8, model="neorpasasattateur Avg. +Back-Trans", dataset="FLORES", metric="accuracy", language="Assamese"), table, ctx)
    assert "neorpasasattateur" not in (fixed.model or "") and "Back-Trans" in (fixed.model or "") + (fixed.setting or "") and "model_name" in rules
    plain, _ = repair(_p(value=85.8, model="mBERT", dataset="FLORES", metric="accuracy", language="Bengali"), table, ctx)
    assert plain.model == "mBERT"


def test_a_spelling_of_a_known_benchmark_that_differs_from_the_tables_own_label_is_corrected():
    table = "Table 3.2: Results.\n| Name | Metric | Small | Large |\n| LAMBADA | acc | 42.7 | 60.4 |\n| StoryCloze | acc | 63.3 | 72.4 |"
    ctx = paper_context("x", ["We evaluate GPT-3.", table], ["GPT-3 Small"])
    fixed, rules = repair(_p(value=42.7, dataset="LAMBDA", metric="accuracy", model="GPT-3 Small"), table, ctx)
    assert fixed.dataset == "LAMBADA" and "dataset_spelling" in rules
    other, _ = repair(_p(value=63.3, dataset="StoryCloze", metric="accuracy", model="GPT-3 Small"), table, ctx)
    assert other.dataset == "StoryCloze"
