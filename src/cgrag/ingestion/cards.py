"""Retrieval cards: a short, plain-words summary of what a chunk records, built from its Condition Profiles.

Why: a results table says "kn" in its header and "IndicXNLI" in its caption, while a question says "Kannada NLI". Keyword search
cannot connect them, a dense embedding of a 5,000-character table is mostly numbers, and the cross-encoder cannot read pipe-table
rows (it scored the Kannada NLI table -10.8, below the -2.0 threshold, so the table was dropped). The card states the same facts
in the words a question uses:

    Reported results - tasks: natural language inference; datasets: IndicXNLI; languages: Assamese, Bengali, ..., Kannada, ...;
    models: mBERT, XLM-R, MuRIL; metrics: accuracy; settings: test set.

The card is used only for retrieval (it is embedded and keyword-indexed next to the chunk, and the reranker reads it in front of the
chunk text); the answer, the sources shown and every other stage still use the chunk text. It is built from the profile store, so
it can be rebuilt without an LLM (`scripts/reindex_cards.py`).
"""
from __future__ import annotations

from collections import Counter

from ..schemas import ConditionProfile

MAX_PER_FIELD = {"task": 4, "dataset": 6, "language": 24, "model": 14, "metric": 5, "setting": 4}
_LABEL = {"task": "tasks", "dataset": "datasets", "language": "languages", "model": "models", "metric": "metrics", "setting": "settings"}
MAX_CARD_CHARS = 700


def _is_name(value: str | None) -> bool:
    return bool(value) and 1 < len(value.strip()) <= 40


def _dataset_label(p: ConditionProfile) -> str | None:
    """The data set with its version when one is recorded ("SQuAD 2.0"): two tables of one benchmark differ in the version only."""
    name = (p.dataset or "").strip()
    if not _is_name(name):
        return None
    version = (p.dataset_version or "").strip()
    return f"{name} {version}" if version and version.lower() not in name.lower() else name


def build_card(profiles: list[ConditionProfile]) -> str:
    """'' when the chunk has no profiles (prose without results gets no card)."""
    if not profiles:
        return ""
    parts = []
    for field, limit in MAX_PER_FIELD.items():
        if field == "dataset":
            counts = Counter(label for label in map(_dataset_label, profiles) if label)
        else:
            counts = Counter(getattr(p, field).strip() for p in profiles if _is_name(getattr(p, field)))
        values = [v for v, _ in counts.most_common(limit)]
        if values:
            parts.append(f"{_LABEL[field]}: {', '.join(values)}")
    card = "Reported results - " + "; ".join(parts) + "." if parts else ""
    return card[:MAX_CARD_CHARS]


def cards_for(profile_map: dict[str, list[ConditionProfile]], chunk_ids: list[str]) -> dict[str, str]:
    return {cid: build_card(profile_map.get(cid, [])) for cid in chunk_ids}
