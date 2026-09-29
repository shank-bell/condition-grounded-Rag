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


def test_answer_without_citations_is_checked_against_all_sources():
    sources = [rc("c1", "Model A reaches 70.5 accuracy on the test set."), rc("c2", "Other text about training details only.")]
    checks = ClaimChecker(CriticConfig(), lambda: StubNLI()).check("Model A reaches 70.5 accuracy on the test set.", sources, {})
    assert len(checks) == 1 and checks[0].supported and checks[0].chunk_id == "c1"
