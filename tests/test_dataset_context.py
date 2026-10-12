"""The data set of a table from the paper's own text (ingestion/dataset_context.py) and its use by the repair: the decision logic, with a fake LLM."""
import json

from cgrag.ingestion.dataset_context import accept, compact, group_key, groups, label_kind, resolve_paper
from cgrag.ingestion.repair import repair_paper
from cgrag.schemas import ConditionProfile


def _p(i: int, chunk: str = "x:0001", **kw) -> ConditionProfile:
    base = dict(profile_id=f"{chunk}#{i}", paper_id="x", chunk_id=chunk, value=40.0 + i, metric="ROUGE-1", model="BiSET", task="summarization")
    base.update(kw)
    return ConditionProfile(**base)


class FakeLLM:
    def __init__(self, answers: dict[str, tuple[str, str]]):
        self.answers, self.prompts = answers, []

    def chat(self, messages, **kw):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        for needle, (name, evidence) in self.answers.items():
            if needle in prompt:
                return type("R", (), {"text": json.dumps({"dataset": name, "evidence": evidence})})()
        return type("R", (), {"text": json.dumps({"dataset": "UNKNOWN", "evidence": ""})})()


CHUNKS = {
    "x:0003": "Extensive experiments were conducted on the Gigaword dataset (Rush et al., 2015), a public dataset for abstractive sentence summarization.",
    "x:0001": "Table 2: Results of different interaction approaches.\n| Interaction method | ROUGE-1 | ROUGE-2 |\n| Concatenation | 32.26 | 15.30 |\n| Bi-selective layer | 39.11 | 19.78 |",
}


def test_which_values_cannot_be_trusted():
    assert label_kind(None) == "none" and label_kind("  ") == "none"
    assert label_kind("F-score") == "metric_label" and label_kind("AVG") == "metric_label"
    assert label_kind("Chunking") == "task_name" and label_kind("NER") == "task_name"
    assert label_kind("CoN h") == "fragment" and label_kind("SQuAD") is None and label_kind("Gigaword") is None


def test_an_answer_must_be_carried_by_the_paper():
    text = compact("on the Gigaword dataset and the CoNLL-2003 shared task")
    assert accept("Gigaword", "experiments were conducted on the Gigaword dataset", text, "none")
    assert not accept("Gigaword", "", text, "none")
    assert not accept("DUC-2004", "evaluated on DUC-2004", text, "none")
    assert not accept("UNKNOWN", "x", text, "none") and not accept("dataset", "the dataset", text, "none")
    assert not accept("Gigaword and CoNLL-2003", "Gigaword and CoNLL-2003", text, "none")
    assert not accept("Chunking", "the Chunking task", compact("the Chunking task"), "task_name")
    assert not accept("Hindi", "Hindi", compact("Hindi"), "none")


def test_groups_are_keyed_by_chunk_and_stored_value():
    ps = [_p(0, dataset=None), _p(1, dataset=None), _p(2, dataset="CNN/DailyMail"), _p(3, chunk="x:0002", dataset="Chunking")]
    g = groups(ps)
    assert set(g) == {group_key("x:0001", None), group_key("x:0002", "Chunking")} and len(g[group_key("x:0001", None)]) == 2


def test_resolution_is_asked_once_per_group_and_kept_only_when_grounded():
    ps = [_p(0, dataset=None), _p(1, dataset=None)]
    llm = FakeLLM({"TABLE:": ("Gigaword", "experiments were conducted on the Gigaword dataset")})
    found = resolve_paper(llm, ps, CHUNKS, "BiSET")
    assert list(found) == [group_key("x:0001", None)] and found[group_key("x:0001", None)].dataset == "Gigaword" and len(llm.prompts) == 1
    assert "Gigaword" in llm.prompts[0] and "Table 2" in llm.prompts[0]
    liar = FakeLLM({"TABLE:": ("DUC-2004", "evaluated on DUC-2004")})
    assert resolve_paper(liar, ps, CHUNKS, "BiSET") == {}


def test_the_repair_applies_the_decision_and_can_be_switched_off():
    ps = [_p(0, dataset=None), _p(1, dataset="SQuAD")]
    res = {group_key("x:0001", None): {"dataset": "Gigaword", "evidence": "conducted on the Gigaword dataset", "mode": "table", "kind": "none"}}
    on, counts = repair_paper(ps, CHUNKS, resolutions=res)
    assert on[0].dataset == "Gigaword" and on[1].dataset == "SQuAD" and counts["dataset_context"] == 1
    off, counts_off = repair_paper(ps, CHUNKS, frozenset({"dataset_context"}), resolutions=res)
    assert off[0].dataset is None and counts_off["dataset_context"] == 0
    plain, _ = repair_paper(ps, CHUNKS)
    assert plain[0].dataset is None


def test_a_replacement_must_not_lose_what_the_stored_label_says():
    text = compact("MASSIVE StoryCloze SuperGLUE Penn Treebank XSum CNN/DailyMail MLQA SQuAD")
    assert not accept("MASSIVE", "the MASSIVE dataset", text, "task_name", "MASSIVE (Intent)")
    assert not accept("StoryCloze", "the StoryCloze dataset", text, "task_name", "StoryCloze 2016")
    assert not accept("SuperGLUE", "SuperGLUE", text, "fragment", "CB")
    assert not accept("MRPC", "MRPC", compact("MRPC STS"), "task_name", "MRPC STS")
    assert accept("SuperGLUE", "SuperGLUE", text, "metric_label", "Average")


def test_a_cut_fragment_is_resolved_only_to_the_name_it_is_a_piece_of():
    text = compact("XSum CNN/DailyMail NYT")
    assert accept("XSum", "XSum", text, "fragment", "XS") and accept("XSum", "XSum", text, "fragment", "um")
    assert not accept("XSum", "XSum", text, "fragment", "CNN") and not accept("XSum", "XSum", text, "fragment", "/DM")


def test_a_known_benchmark_of_another_task_is_not_accepted():
    text = compact("MLQA question answering")
    assert not accept("MLQA", "MLQA", text, "none", None, "", ["sequence labeling"])
    assert accept("MLQA", "MLQA", text, "none", None, "", ["question answering"])


def test_a_citation_after_a_name_is_not_part_of_the_name():
    from cgrag.ingestion.model_context import is_pure_citation, strip_citation
    assert strip_citation("Adaptive (Sukhbaatar et al.,2019)") == "Adaptive"
    assert strip_citation("BM25+BERT (Lee et al.,2019)") == "BM25+BERT"
    assert strip_citation("ELMo (Peters et al.,2018a)") == "ELMo"
    assert strip_citation("Devlin et al.(2018)") is None and is_pure_citation("Devlin et al.(2018)")
    assert is_pure_citation("Lample and Conneau (2019)") and is_pure_citation("Lee et. (2018)")
    assert strip_citation("BERT-base") is None and strip_citation("Lee et. (2018)") is None
    assert not is_pure_citation("BERT (2018)") and not is_pure_citation("mBERT")


def test_a_citation_is_resolved_to_the_name_the_paper_writes():
    from cgrag.ingestion.model_context import accept as accept_model, model_key, resolve_models
    text = compact("multilingual BERT (mBERT; Devlin et al., 2018) and XLM (Lample and Conneau, 2019)")
    assert accept_model("mBERT", "multilingual BERT (mBERT; Devlin et al., 2018)", text, "Devlin et al.(2018)")
    assert not accept_model("Devlin", "Devlin et al.", text, "Devlin et al.(2018)")
    assert not accept_model("GPT-4", "GPT-4", text, "Devlin et al.(2018)")
    ps = [_p(0, model="Devlin et al.(2018)", dataset="XNLI"), _p(1, model="mBERT")]
    sentences = ["We compare with multilingual BERT (mBERT; Devlin et al., 2018) trained on 104 languages."]

    class Fake:
        def chat(self, messages, **kw):
            return type("R", (), {"text": json.dumps({"model": "mBERT", "evidence": "multilingual BERT (mBERT; Devlin et al., 2018)"})})()

    found = resolve_models(Fake(), ps, sentences, text, "XLM-R")
    assert list(found) == [model_key("Devlin et al.(2018)")] and found[model_key("Devlin et al.(2018)")]["model"] == "mBERT"
    fixed, counts = repair_paper(ps, {}, resolutions=found)
    assert fixed[0].model == "mBERT" and counts["model_context"] == 1 and fixed[1].model == "mBERT"
    off, _ = repair_paper(ps, {}, frozenset({"model_context"}), resolutions=found)
    assert off[0].model == "Devlin et al.(2018)"
