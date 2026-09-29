"""Local model loaders (embedder, reranker, NLI), each created once and reused. All run locally."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .config import get_settings


def _dtype(device: str):
    return torch.float16 if device.startswith("cuda") else torch.float32


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    cfg = get_settings()
    model = SentenceTransformer(cfg.models.embedder, device=cfg.devices.embedder,
                                model_kwargs={"torch_dtype": _dtype(cfg.devices.embedder)})
    model.max_seq_length = 1024          # chunks are <= 2000 chars (~500 tokens)
    return model


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """L2-normalised BGE-M3 dense vectors, shape (n, 1024)."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    return _embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    cfg = get_settings()
    return CrossEncoder(cfg.models.reranker, device=cfg.devices.reranker, max_length=512)


def rerank_scores(question: str, passages: list[str]) -> list[float]:
    """Cross-encoder relevance logits for (question, passage) pairs."""
    if not passages:
        return []
    return [float(s) for s in _reranker().predict([(question, p) for p in passages], batch_size=32, show_progress_bar=False)]


class NLI:
    """Entailment / neutral / contradiction probabilities. Label order is read from the model config."""

    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = get_settings()
        self.device = cfg.devices.nli
        self.tok = AutoTokenizer.from_pretrained(cfg.models.nli)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.models.nli, torch_dtype=_dtype(self.device))
        self.model.to(self.device).eval()
        labels = {i: name.lower() for i, name in self.model.config.id2label.items()}
        self.idx = {name: i for i, name in labels.items()}
        for needed in ("entailment", "neutral", "contradiction"):
            if needed not in self.idx:
                raise ValueError(f"NLI model labels {labels} lack '{needed}'")

    @torch.inference_mode()
    def probs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        """(n, 3) array of probabilities in the order [entailment, neutral, contradiction]."""
        out: list[np.ndarray] = []
        order = [self.idx["entailment"], self.idx["neutral"], self.idx["contradiction"]]
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation=True, max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, order].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


@lru_cache(maxsize=1)
def get_nli() -> NLI:
    return NLI()
