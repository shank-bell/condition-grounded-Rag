"""ChromaDB: 1024-d BGE-M3 vectors with paper / page / section metadata (chunk text kept as the document)."""
from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings
from ..schemas import Chunk


class VectorStore:
    def __init__(self, path: Path | None = None) -> None:
        path = path or get_settings().paths.chroma_dir
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _to_chunk(id_: str, doc: str, meta: dict) -> Chunk:
        return Chunk(chunk_id=id_, paper_id=meta["paper_id"], paper_title=meta.get("paper_title", ""),
                     page=int(meta["page"]), section=meta["section"], text=doc)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"paper_id": c.paper_id, "paper_title": c.paper_title, "page": c.page, "section": c.section} for c in chunks],
        )

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Nearest chunks with cosine similarity (1 - distance)."""
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[embedding.tolist()], n_results=min(k, n))
        return [(self._to_chunk(i, d, m), 1.0 - dist)
                for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def get(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        res = self.col.get(ids=ids)
        found = {i: self._to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])}
        return [found[i] for i in ids if i in found]

    def paper_chunks(self, paper_id: str) -> list[Chunk]:
        res = self.col.get(where={"paper_id": paper_id})
        chunks = [self._to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: c.chunk_id)

    def all_chunks(self) -> list[Chunk]:
        res = self.col.get()
        chunks = [self._to_chunk(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: c.chunk_id)

    def has_paper(self, paper_id: str) -> bool:
        return bool(self.col.get(where={"paper_id": paper_id}, limit=1)["ids"])

    def delete_paper(self, paper_id: str) -> None:
        self.col.delete(where={"paper_id": paper_id})

    def count(self) -> int:
        return self.col.count()
