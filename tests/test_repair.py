"""Profile repair (ingestion/repair.py) and the benchmark list (knowledge/benchmarks.py): each rule on a small table written like the stored chunks."""
from cgrag.ingestion.repair import fix_name, paper_context, repair
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
    assert task_fits("general tasks", lookup("SQuAD")) is None               # no known family: left alone


def test_language_names():
    assert is_language("Kannada") and is_language("hi") and is_language("transliterated Urdu") and is_language("Indian languages")
    assert not is_language("Jewish") and not is_language("STEM")


def test_vertical_group_label_is_joined_and_sizes_keep_groups():
    labels = {c.row: c.row_label for c in table_cells(parse_table(LLAMA_TABLE))}
    assert labels[0] == labels[1] == "MPT"                    # 30B continues MPT (sizes grow), it does not belong to Falcon
    assert labels[2] == labels[3] == "Falcon"
    assert {labels[4], labels[5], labels[6], labels[7]} == {"Llama 1"}


def test_garbled_names_are_matched_to_names_the_prose_writes():
    ctx = paper_context("x", [PROSE, LLAMA_TABLE], ["MPT", "Falcon", "Llama 1", "Llama 2"])
    assert fix_name("lama 1", ctx) == "Llama 1"
    assert fix_name("coFaln", ctx) == "Falcon"                # same letters
    assert fix_name("Fln", ctx) == "Falcon"                   # a piece of the name
    assert fix_name("MPT", ctx) is None                       # written in the prose: not garbled


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
    assert q3.setting == "TRANSLATE TEST"                     # translate-test is a regime, not a split


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
    assert q3.dataset_version is None                         # a training-set size is not a version


def test_rules_can_be_switched_off():
    ctx = paper_context("x", ["BERT is a model. BERT again."], ["BERT"])
    q, fired = repair(_p(value=1.0, dataset="MRPC", task="sentiment analysis"), "", ctx, off=frozenset({"task"}))
    assert q.task == "sentiment analysis" and fired == []


# ---- 11 Oct, second round: stacked tables, language tables, regimes, variants, abbreviations ----------------------------------------------------

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
    assert q2.language == "English"                          # "Te" is the test column, not Telugu


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
    assert q2.model == "mT5-XL"                              # a prefix of another name, but a row label of its own


def test_paper_defined_abbreviations_and_short_names():
    ctx = paper_context("x", ["We use OntoNotes 5.0 (ON5e) and WNUT-16 (WN16). RG is short for ROUGE. The Spearman correlation coefficient (scc) is used."], [])
    assert ctx.abbrevs["on5e"] == "OntoNotes 5.0" and ctx.abbrevs["wn16"] == "WNUT-16" and ctx.abbrevs["rg"] == "ROUGE"
    q, _ = repair(_p(value=1.0, dataset="WN16", metric="scc"), "", ctx)
    assert q.dataset == "WNUT-16" and q.metric == "Spearman correlation coefficient"
    assert lookup("mr") is None and lookup("MR").name == "MR"      # "mr" is Marathi, "MR" the review data set


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
