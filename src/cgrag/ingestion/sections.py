"""Map a section heading to one of the IMRaD-style labels in schemas.Section."""
from __future__ import annotations

import re

from ..schemas import Section

# Checked in order; the first rule whose pattern matches the heading wins.
_RULES: list[tuple[Section, re.Pattern[str]]] = [
    ("references", re.compile(r"\b(references|bibliography)\b")),
    ("conclusion", re.compile(r"\b(conclusions?|future work|concluding remarks|summary)\b")),
    ("abstract", re.compile(r"\babstract\b")),
    ("introduction", re.compile(r"\bintroduction\b")),
    ("related_work", re.compile(r"\b(related work|background|preliminar\w*|literature review|prior work|previous work)\b")),
    ("results", re.compile(r"\b(results?|findings|ablations?|main results)\b")),
    ("experiments", re.compile(r"\b(experiments?|experimental|evaluations?|setup|implementation details|datasets?|benchmarks?|training details|hyper-?parameters?|baselines?)\b")),
    ("discussion", re.compile(r"\b(discussion|analysis|limitations?)\b")),
    ("methods", re.compile(r"\b(methods?|methodology|approach|architecture|framework|algorithm|proposed|model|models)\b")),
    ("other", re.compile(r"\b(acknowledge?ments?|appendix|appendices|supplementary|ethics?|broader impacts?)\b")),
]


# "3", "3.1", "A", "A.2", "IV." followed by a capitalised title
NUMBERED = re.compile(r"^(?P<num>(?:\d{1,2}|[A-H]|[IVX]{1,4})(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{1,90})$")


def parse_heading(text: str) -> tuple[str | None, str]:
    """Split a heading into (section number or None, title)."""
    m = NUMBERED.match(text.strip())
    return (m.group("num"), m.group("title")) if m else (None, text.strip())


def classify_heading(title: str) -> Section | None:
    """Label for a heading's title, or None if no keyword matches."""
    t = title.lower()
    for label, pattern in _RULES:
        if pattern.search(t):
            return label
    return None
