"""Retrieval cards: a chunk's recorded conditions in a question's words, used by keyword search, dense search and the reranker."""
import numpy as np

from cgrag.config import RetrievalConfig
from cgrag.ingestion.cards import build_card
from cgrag.pipeline import retrieval as retrieval_module
from cgrag.pipeline.rerank import rerank_text
from cgrag.pipeline.retrieval import HybridRetriever
from cgrag.schemas import Chunk, ConditionProfile
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.vector_store import VectorStore


def prof(i: int, **kw) -> ConditionProfile:
    return ConditionProfile(**{**dict(profile_id=f"p{i}", paper_id="a", chunk_id="a:1", metric="accuracy", value=50.0), **kw})


def chunk(cid: str, text: str, card: str = "", paper: str = "a") -> Chunk:
    return Chunk(chunk_id=cid, paper_id=paper, paper_title="t", page=1, section="results", text=text, card=card)


def test_a_card_lists_what_the_chunk_records_in_words():
    card = build_card([prof(1, task="natural language inference", dataset="IndicXNLI", language="Kannada", model="mBERT"),
                       prof(2, task="natural language inference", dataset="IndicXNLI", language="Hindi", model="MuRIL", setting="test set")])
    assert card.startswith("Reported results - ")
    for word in ("natural language inference", "IndicXNLI", "Kannada", "Hindi", "mBERT", "MuRIL", "accuracy", "test set"):
        assert word in card


def test_most_frequent_values_come_first_and_lists_are_capped():
    profiles = [prof(i, language=f"Lang{i}") for i in range(40)] + [prof(100 + i, language="Kannada") for i in range(5)]
    card = build_card(profiles)
    assert card.index("Kannada") < card.index("Lang0")
    assert card.count("Lang") <= 24


def test_no_profiles_or_only_junk_values_give_no_card():
    assert build_card([]) == ""
    assert build_card([prof(1, model="x" * 80)]).count("models") == 0                # a 80-character "model name" is junk


def test_the_keyword_index_finds_a_chunk_by_its_card():
    plain = chunk("a:1", "Table 16 | as | bn | kn | 74.7 |")
    carded = chunk("a:2", "Table 16 | as | bn | kn | 74.7 |", card="Reported results - languages: Kannada.")
    filler = [chunk(f"b:{i}", f"unrelated paragraph number {i} about attention heads") for i in range(6)]
    store = BM25Store.build([plain, carded, *filler])
    assert [cid for cid, _ in store.search("Kannada results", 5)] == ["a:2"]


def test_the_vector_store_keeps_a_card_vector_per_carded_chunk(tmp_path):
    vs = VectorStore(tmp_path / "chroma")
    a, b = chunk("a:1", "text one"), chunk("a:2", "text two", card="Reported results - languages: Kannada.")
    vs.add([a, b], np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), np.array([[0, 0, 1.0, 0]]))
    assert vs.card_count() == 1
    assert [cid for cid, _ in vs.query_cards(np.array([0, 0, 1.0, 0]), 3)] == ["a:2"]
    assert vs.get(["a:2"])[0].card.startswith("Reported results")             # the card travels with the chunk
    vs.delete_paper("a")
    assert vs.card_count() == 0 and vs.count() == 0


def test_set_cards_updates_existing_chunks_without_losing_their_metadata(tmp_path):
    vs = VectorStore(tmp_path / "chroma")
    vs.add([chunk("a:1", "text one"), chunk("a:2", "text two")], np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    assert vs.card_count() == 0
    new = [chunk("a:1", "text one", card="Reported results - tasks: NER."), chunk("a:2", "text two")]
    vs.set_cards(new, np.array([[0, 0, 1.0, 0]]))
    got = {c.chunk_id: c for c in vs.get(["a:1", "a:2"])}
    assert got["a:1"].card.startswith("Reported results") and got["a:1"].paper_id == "a" and got["a:1"].section == "results"
    assert got["a:2"].card == "" and vs.card_count() == 1


def test_the_reranker_reads_the_card_in_front_of_the_text_unless_cards_are_off():
    c = chunk("a:1", "| as | bn |", card="Reported results - languages: Kannada.")
    assert rerank_text(c).startswith("Reported results") and rerank_text(c).endswith("| as | bn |")
    assert rerank_text(c, use_cards=False) == "| as | bn |"
    assert rerank_text(chunk("a:2", "plain")) == "plain"


def test_dense_search_over_cards_finds_a_table_its_text_vector_misses(tmp_path, monkeypatch):
    vs = VectorStore(tmp_path / "chroma")
    table = chunk("a:1", "Table 16 | as | bn | kn |", card="Reported results - languages: Kannada.")
    prose = chunk("a:2", "A paragraph about something else entirely.")
    vs.add([table, prose], np.array([[0, 1.0, 0, 0], [1.0, 0, 0, 0]]), np.array([[0, 0, 1.0, 0]]))
    query = np.array([[0.1, 0, 1.0, 0]])                                         # close to the card vector only
    monkeypatch.setattr(retrieval_module, "embed", lambda qs: np.repeat(query, len(qs), axis=0))
    on = HybridRetriever(vs, BM25Store([], []), RetrievalConfig(dense_k=1, bm25_k=1, fused_top=5, use_cards=True))
    off = HybridRetriever(vs, BM25Store([], []), RetrievalConfig(dense_k=1, bm25_k=1, fused_top=5, use_cards=False))
    assert [r.chunk.chunk_id for r in on.retrieve(["Kannada results"], "result")][:2] == ["a:2", "a:1"] or \
        "a:1" in [r.chunk.chunk_id for r in on.retrieve(["Kannada results"], "result")]
    assert "a:1" not in [r.chunk.chunk_id for r in off.retrieve(["Kannada results"], "result")]
    assert "a:1" not in [r.chunk.chunk_id for r in on.retrieve(["How does it work?"], "method")]     # method questions skip the cards
