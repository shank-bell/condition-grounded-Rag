"""Stage 9: a sentence supported by a source it did not cite is a mis-citation, and is repaired."""
import numpy as np

from cgrag.config import CriticConfig
from cgrag.pipeline.critic import ClaimChecker
from cgrag.schemas import Chunk, RetrievedChunk


def rc(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, paper_id="p", page=1, section="results", text=text), score=1.0, rerank_score=1.0)


class NeutralNLI:
    def probs(self, pairs):
        return np.array([[0.2, 0.7, 0.1]] * len(pairs))


def test_a_mis_cited_sentence_is_supported_and_its_citation_is_repaired():
    sources = [rc("c1", "BERT-large reaches 84.1 EM on SQuAD 1.1."), rc("c2", "On SQuAD 2.0 test, BERT-large gets 83.1 F1.")]
    answer = "BERT-large gets 83.1 F1 on SQuAD 2.0 test [1]."
    checker = ClaimChecker(CriticConfig(), lambda: NeutralNLI())
    checks = checker.check(answer, sources, {})
    assert [c.supported for c in checks] == [True] and checks[0].chunk_id == "c2"
    fixed_answer, fixed_checks = checker.repair_citations(answer, checks, sources)
    assert fixed_answer == "BERT-large gets 83.1 F1 on SQuAD 2.0 test [2]."
    assert fixed_checks[0].sentence == fixed_answer


def test_a_sentence_no_source_supports_stays_unsupported_and_is_not_repaired():
    sources = [rc("c1", "BERT-large reaches 84.1 EM."), rc("c2", "Nothing about that here 12.5.")]
    answer = "BERT-large gets 99.9 F1 on SQuAD 2.0 test [1]."
    checker = ClaimChecker(CriticConfig(), lambda: NeutralNLI())
    checks = checker.check(answer, sources, {})
    assert [c.supported for c in checks] == [False]
    assert checker.repair_citations(answer, checks, sources)[0] == answer


def test_a_correct_citation_is_left_alone():
    sources = [rc("c1", "BERT-large gets 83.1 F1 on SQuAD 2.0 test."), rc("c2", "Other text.")]
    answer = "BERT-large gets 83.1 F1 on SQuAD 2.0 test [1]."
    checker = ClaimChecker(CriticConfig(), lambda: NeutralNLI())
    checks = checker.check(answer, sources, {})
    assert checker.repair_citations(answer, checks, sources)[0] == answer
