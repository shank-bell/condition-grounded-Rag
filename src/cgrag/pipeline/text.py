"""Small text helpers shared by the contradiction resolver and the claim-check critic."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(\"'“])")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_WORD = re.compile(r"[a-z0-9]+")


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Sentences of a text; every line of a table is kept as its own unit."""
    out: list[str] = []
    for line in re.split(r"\n+", text):
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if len(s.strip()) >= min_chars)
    return out


def citations(sentence: str) -> list[int]:
    """1-based source numbers cited in a sentence: '... [1, 3]' -> [1, 3]."""
    return [int(n) for m in _CITATION.findall(sentence) for n in re.split(r"\s*,\s*", m)]


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s*" + _CITATION.pattern, "", sentence).strip()


def plain(sentence: str) -> str:
    """Sentence without citation markers, markdown emphasis and list bullets (what an NLI model should read)."""
    text = re.sub(r"[*_`#>]+", "", strip_citations(sentence))
    return re.sub(r"^\s*[-•]\s*", "", text).strip()


def overlap(a: str, b: str) -> float:
    """Share of a's distinct words that also occur in b (a cheap relevance signal)."""
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(wa & wb) / len(wa) if wa else 0.0
