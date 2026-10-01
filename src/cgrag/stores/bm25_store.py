"""BM25 (Okapi) keyword index over the same chunks as ChromaDB, persisted next to it."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..config import get_settings
from ..schemas import Chunk

# keeps version strings and hyphenated model names whole: "v1.1", "bert-large", "84.6"
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Store:
    def __init__(self, ids: list[str], tokens: list[list[str]]) -> None:
        self.ids = ids
        self.tokens = tokens
        self.index = BM25Okapi(tokens) if tokens else None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Store":
        """The keyword index sees the chunk text and, when there is one, its retrieval card (ingestion/cards.py)."""
        return cls([c.chunk_id for c in chunks], [tokenize(f"{c.card} {c.text}" if c.card else c.text) for c in chunks])

    def save(self, path: Path | None = None) -> None:
        path = path or get_settings().paths.bm25_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"ids": self.ids, "tokens": self.tokens}, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Store":
        path = path or get_settings().paths.bm25_path
        if not path.exists():
            return cls([], [])
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(data["ids"], data["tokens"])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]
