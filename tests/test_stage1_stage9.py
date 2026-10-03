"""Query-condition grounding (stage 1), the planner (stage 2) and the claim critic (stage 9); stubs, no LLM/GPU."""
import numpy as np

from cgrag.config import CriticConfig, FeaturesConfig
from cgrag.pipeline.critic import ClaimChecker
from cgrag.pipeline.generate import describe_conflicts, select_sources
from cgrag.pipeline.orchestrator import make_plan
from cgrag.pipeline.query_understanding import clean_conditions, heuristic_analysis
from cgrag.pipeline.text import citations, split_sentences, strip_citations
from cgrag.schemas import Chunk, ConditionProfile, ContradictionPair, QueryAnalysis, QueryConditions, RetrievedChunk


def rc(cid: str, text: str = "x " * 20, paper: str = "p") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id=paper, page=1, section="results", text=text), score=1.0, rerank_score=1.0)


# ---------- stage 1 ----------

def test_invented_conditions_are_dropped():
    q = "Does BERT work well for Kannada?"
    llm_says = QueryConditions(model="BERT", language="Kannada", dataset="SQuAD", task="question answering")
    kept = clean_conditions(llm_says, q).specified()
    assert kept == {"model": "BERT", "language": "Kannada"}


def test_version_glued_to_the_dataset_is_split_and_task_is_not_a_dataset_name():
    q = "What is the human performance on SQuAD 2.0 compared with the best system?"
    got = clean_conditions(QueryConditions(task="SQuAD 2.0", dataset="SQuAD 2.0"), q).specified()
    assert got == {"dataset": "SQuAD", "dataset_version": "2.0"}
    real_task = clean_conditions(QueryConditions(task="question answering", dataset="SQuAD"), "How does BERT do on SQuAD question answering?")
    assert real_task.specified() == {"task": "question answering", "dataset": "SQuAD"}


def test_a_task_abbreviation_the_llm_put_in_the_dataset_field_is_moved_to_task():
    got = clean_conditions(QueryConditions(dataset="NLI", language="Kannada"), "How well do models perform on Kannada NLI?")
    assert got.specified() == {"task": "NLI", "language": "Kannada"}
    kept = clean_conditions(QueryConditions(task="natural language inference", dataset="NLI"),
                            "How do models do on NLI (natural language inference)?").specified()
    assert kept == {"task": "natural language inference"}          # an existing task wins, the abbreviation is not a dataset


def test_further_models_and_languages_are_kept_only_when_the_question_names_them():
    q = "Compare mBERT, XLM-R and GPT-4 on XNLI for Hindi and Kannada."
    said = QueryConditions(model="mBERT", other_models=["XLM-R", "GPT-4", "mBERT", "Llama"], dataset="XNLI", language="Hindi",
                           other_languages=["Kannada", "Tamil"])
    got = clean_conditions(said, q)
    assert got.other_models == ["XLM-R", "GPT-4"] and got.other_languages == ["Kannada"]       # no repeat of the first, nothing invented
    assert got.extras() == {"model": ["XLM-R", "GPT-4"], "language": ["Kannada"]}
    assert got.specified() == {"model": "mBERT", "dataset": "XNLI", "language": "Hindi"}      # the single-value view is unchanged


def test_the_only_dataset_the_llm_listed_among_the_others_is_the_first_one():
    """Seen on 1 Oct with the 12B model: "mT5 on XQuAD for Arabic" came back as dataset = null, other_datasets = ["XQuAD"]."""
    got = clean_conditions(QueryConditions(model="mT5", language="Arabic", other_datasets=["XQuAD"]),
                           "What accuracy does mT5 get on XQuAD for Arabic?")
    assert got.specified() == {"model": "mT5", "dataset": "XQuAD", "language": "Arabic"} and got.extras() == {}
    several = clean_conditions(QueryConditions(model="BERT-base", other_datasets=["MLQA", "SQuAD 2.0"]),
                               "How does BERT-base do on SQuAD 2.0 and MLQA?")
    assert several.specified() == {"model": "BERT-base", "dataset": "SQuAD", "dataset_version": "2.0"}   # the one named first, split as usual
    assert several.other_datasets == ["MLQA"]


class StubLLM:
    def __init__(self, said: QueryAnalysis):
        self.said = said

    def structured(self, prompt, model_cls, **kwargs):
        return self.said, None


class StubProfiles:
    def vocabulary(self):
        return {"model": ["mT5", "mBERT", "XLM-R"], "dataset": ["XQuAD", "XNLI", "TyDi QA GoldP"], "language": [], "task": []}


def test_ordinary_words_that_a_table_row_is_labelled_with_are_not_further_models():
    """"embeddings", "baseline", "memory" are stored as models (row labels); a question using the plain word must not get a fake further model."""
    from cgrag.pipeline.query_understanding import extras_from_vocabulary
    vocab = {"model": ["BERT", "Embeddings", "baseline", "memory", "human", "Llama", "mBERT"], "dataset": [], "language": [], "task": []}
    primary = {"model": "BERT"}
    assert extras_from_vocabulary("How much memory does BERT need compared with the baseline and human embeddings?", vocab, primary) == {}
    assert extras_from_vocabulary("Compare BERT and Llama on SQuAD", vocab, primary) == {"other_models": ["Llama"]}        # a capitalised name
    assert extras_from_vocabulary("Compare BERT and mBERT on SQuAD", vocab, primary) == {"other_models": ["mBERT"]}        # a capital after the first letter


def test_the_words_of_the_first_models_own_name_are_not_further_models():
    from cgrag.pipeline.query_understanding import extras_from_vocabulary
    vocab = {"model": ["GloVe", "GloVe embeddings", "Embeddings"], "dataset": [], "language": [], "task": []}
    assert extras_from_vocabulary("What Spearman rank correlation does GloVe embeddings get on SICK-R?", vocab, {"model": "GloVe embeddings"}) == {}


def test_a_multi_word_model_name_the_llm_cut_short_is_restored_from_the_question():
    from cgrag.pipeline.query_understanding import QueryUnderstanding, fuller_model_name
    names = ["GloVe", "GloVe embeddings", "Avg. GloVe embeddings", "BERT embeddings"]
    q = "What Spearman rank correlation does GloVe embeddings get on SICK-R?"
    assert fuller_model_name("GloVe", q, names) == "GloVe embeddings"                  # the longest name the question writes literally
    assert fuller_model_name("GloVe", "How does GloVe do on SICK-R?", names) is None   # the question does not write the longer name
    assert fuller_model_name(None, q, names) is None
    said = QueryAnalysis(intent="result", complexity="simple", conditions=QueryConditions(model="GloVe", dataset="SICK-R"))

    class Profiles(StubProfiles):
        def vocabulary(self):
            return {"model": names, "dataset": ["SICK-R"], "language": [], "task": []}
    got = QueryUnderstanding(StubLLM(said), classifier=None, profiles=Profiles()).analyze(q).conditions
    assert got.specified() == {"model": "GloVe embeddings", "dataset": "SICK-R"} and got.extras() == {}
    # the LLM also filed the word "embeddings" under task: it is part of the model's name, and Stage 6 would warn that no source covers such a task
    said = QueryAnalysis(intent="result", complexity="simple", conditions=QueryConditions(model="GloVe", dataset="SICK-R", task="embeddings"))
    got = QueryUnderstanding(StubLLM(said), classifier=None, profiles=Profiles()).analyze(q).conditions
    assert got.specified() == {"model": "GloVe embeddings", "dataset": "SICK-R"}
    real = QueryAnalysis(intent="result", complexity="simple", conditions=QueryConditions(model="GloVe", dataset="SICK-R", task="semantic textual similarity"))
    assert QueryUnderstanding(StubLLM(real), classifier=None, profiles=Profiles()).analyze(
        "What semantic textual similarity does GloVe get on SICK-R?").conditions.task == "semantic textual similarity"          # a real task stays


def test_a_question_with_one_of_each_has_no_further_entities_even_when_the_llm_misfiles_one():
    from cgrag.pipeline.query_understanding import QueryUnderstanding
    said = QueryAnalysis(intent="result", complexity="simple",
                         conditions=QueryConditions(model="mT5", language="Arabic", other_datasets=["XQuAD"]))     # dataset left empty
    got = QueryUnderstanding(StubLLM(said), classifier=None, profiles=StubProfiles()).analyze("What accuracy does mT5 get on XQuAD for Arabic?")
    assert got.conditions.specified() == {"model": "mT5", "dataset": "XQuAD", "language": "Arabic"}
    assert got.conditions.extras() == {}


def test_a_part_of_a_dataset_name_is_replaced_by_the_full_name_the_question_writes():
    """With the new prompt the 12B model answered "GoldP" for "TyDi QA GoldP": the part matches nothing in the store (false warning)."""
    from cgrag.pipeline.query_understanding import QueryUnderstanding
    said = QueryAnalysis(intent="result", complexity="simple",
                         conditions=QueryConditions(model="mBERT", language="Swahili", other_datasets=["GoldP"]))
    qu = QueryUnderstanding(StubLLM(said), classifier=None, profiles=StubProfiles())
    got = qu.analyze("What accuracy does mBERT get on TyDi QA GoldP for Swahili?").conditions
    assert got.specified() == {"model": "mBERT", "dataset": "TyDi QA GoldP", "language": "Swahili"} and got.extras() == {}
    other = QueryAnalysis(intent="result", complexity="simple", conditions=QueryConditions(model="mT5", dataset="XQuAD", language="Arabic"))
    assert QueryUnderstanding(StubLLM(other), classifier=None, profiles=StubProfiles()).analyze(
        "What accuracy does mT5 get on XQuAD for Arabic?").conditions.dataset == "XQuAD"          # an equal name is left alone


def test_names_written_literally_are_added_as_further_entities():
    from cgrag.pipeline.query_understanding import extras_from_vocabulary
    vocab = {"model": ["mBERT", "XLM-R", "BERT-large"], "dataset": ["XNLI"], "language": [], "task": []}
    got = extras_from_vocabulary("Compare mBERT and XLM-R for Hindi and Kannada", vocab, {"model": "mBERT", "language": "Hindi"})
    assert got == {"other_models": ["XLM-R"], "other_languages": ["Kannada"]}
    assert extras_from_vocabulary("How does mBERT do on Hindi?", vocab, {"model": "mBERT", "language": "Hindi"}) == {}


def test_version_and_size_must_be_in_the_question():
    q = "What is BERT-large F1 on SQuAD v2.0?"
    got = clean_conditions(QueryConditions(model="BERT-large", dataset="SQuAD", dataset_version="v2.0", model_size="340M"), q)
    assert got.specified() == {"model": "BERT-large", "dataset": "SQuAD", "dataset_version": "v2.0"}


def test_heuristic_fallback_is_usable():
    a = heuristic_analysis("Compare BERT and RoBERTa on GLUE and explain why they differ?")
    assert a.intent == "comparison" and a.complexity == "complex"
    assert heuristic_analysis("What is the batch size used for BERT?").complexity == "simple"


# ---------- stage 2 ----------

def test_planner_skips_refinement_for_simple_questions():
    simple = make_plan(QueryAnalysis(complexity="simple", conditions=QueryConditions(model="BERT")), FeaturesConfig())
    assert not simple.refine and not simple.decompose and simple.check_applicability
    hard = make_plan(QueryAnalysis(complexity="complex"), FeaturesConfig())
    assert hard.refine and hard.decompose and not hard.check_applicability          # no conditions to check


def test_ablation_flags_switch_stages_off():
    plan = make_plan(QueryAnalysis(conditions=QueryConditions(model="BERT")),
                     FeaturesConfig(applicability=False, contradiction=False, critic=False))
    assert not (plan.check_applicability or plan.check_contradictions or plan.check_claims)


# ---------- text helpers ----------

def test_an_abbreviation_does_not_end_a_sentence():
    from cgrag.pipeline.text import split_sentences
    text = ("DistilBERT is much smaller (66 million parameters vs. BERT-base at 110 million). Devlin et al. (2019) report 79.5 on GLUE. "
            "Languages differ, e.g. Hindi and Tamil. See Fig. 3 for details.")
    assert [s[:12] for s in split_sentences(text, min_chars=8)] == ["DistilBERT i", "Devlin et al", "Languages di", "See Fig. 3 f"]
    assert split_sentences(text, min_chars=8)[0].endswith("110 million).")


def test_citation_parsing():
    assert citations("BERT scores 90.9 F1 [1][3]. Also [2, 4].") == [1, 3, 2, 4]
    assert strip_citations("BERT scores 90.9 F1 [1].") == "BERT scores 90.9 F1."
    assert split_sentences("First sentence here is long. Second sentence here is long too.") == [
        "First sentence here is long.", "Second sentence here is long too."]


# ---------- stage 8 helpers ----------

def test_conflict_chunks_are_added_to_the_sources():
    kept = [rc(f"c{i}") for i in range(8)]
    pair = ContradictionPair(verdict="EXPLAINED", chunk_a="c1", chunk_b="c7", paper_a="a", paper_b="b", reason="version differs")
    sources = select_sources(kept, [pair], top=5)
    assert [s.chunk.chunk_id for s in sources] == ["c0", "c1", "c2", "c3", "c4", "c7"]
    assert "[2] and [6]" in describe_conflicts([pair], sources)


# ---------- stage 9 ----------

class StubNLI:
    """Entails a hypothesis only if every number in it occurs in the premise."""

    def probs(self, pairs):
        import re
        rows = []
        for premise, hyp in pairs:
            ok = all(n in premise for n in re.findall(r"\d+\.\d+", hyp))
            rows.append([0.9, 0.05, 0.05] if ok else [0.05, 0.05, 0.9])
        return np.array(rows)


def test_unsupported_number_is_caught_and_table_claims_use_profiles():
    chunk_text = "BERTLARGE (Single) 84.1 90.9 - -\n\nWe report results on the SQuAD 1.1 development set."
    sources = [rc("c1", chunk_text)]
    profiles = {"c1": [ConditionProfile(profile_id="c1#0", paper_id="p", chunk_id="c1", model="BERT-large", dataset="SQuAD",
                                        metric="F1", value=90.9, setting="dev set")]}
    answer = ("BERT-large obtains 90.9 F1 on SQuAD [1]. It also reaches 95.5 F1 on the same set [1]. "
              "The evidence does not cover Kannada.")
    checks = ClaimChecker(CriticConfig(), lambda: StubNLI()).check(answer, sources, profiles)
    assert [c.supported for c in checks] == [True, False]         # scope sentence makes no claim and is skipped
    assert ClaimChecker.unsupported(checks)[0].sentence.startswith("It also reaches 95.5")


def test_sentences_that_say_a_result_is_missing_make_no_claim():
    """The scope warning asks the answer to say what is not covered; the critic must not call that unsupported (1 Oct)."""
    sources = [rc("c1", "Model A reaches 70.5 accuracy on the test set.")]
    answer = ("Model A reaches 70.5 accuracy on the test set [1]. No results are reported for Kannada for any of the models. "
              "No results are reported for GPT-4 on the XNLI dataset for any language. GPT-4 was not evaluated on XNLI. "
              "There are no reported scores for Tamil. None of the sources give a Hindi number.")
    checks = ClaimChecker(CriticConfig(), lambda: StubNLI()).check(answer, sources, {})
    assert [c.sentence for c in checks] == ["Model A reaches 70.5 accuracy on the test set [1]."]
    wrong = ClaimChecker(CriticConfig(), lambda: StubNLI()).check("Model A reaches 95.5 accuracy on the test set [1].", sources, {})
    assert [c.supported for c in wrong] == [False]                  # a real claim with a wrong number is still caught


def test_answer_without_citations_is_checked_against_all_sources():
    sources = [rc("c1", "Model A reaches 70.5 accuracy on the test set."), rc("c2", "Other text about training details only.")]
    checks = ClaimChecker(CriticConfig(), lambda: StubNLI()).check("Model A reaches 70.5 accuracy on the test set.", sources, {})
    assert len(checks) == 1 and checks[0].supported and checks[0].chunk_id == "c1"
