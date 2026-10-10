"""Optional data set name list from the archived Papers with Code (built by scripts/build_pwc_names.py; empty when the file is missing).

Used by the profile repair to recognise a data set named in a caption ("Results on the CoNLL-03 test set") when the extractor left the data set empty."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "data/bench/pwc/dataset_names.json"

# words that are data set names in the archive but, in a caption, only mean what they say
_COMMON = frozenset("""train test dev data text sentence sentences results result table model models score scores baseline baselines average word words
english german french chinese task tasks language languages corpus dataset datasets benchmark human large small base all none other mixed random
news web wiki wikipedia dialog dialogue speech audio image images video question questions answer answers""".split())


@lru_cache(maxsize=1)
def _names() -> dict[str, dict]:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def available() -> bool:
    return bool(_names())


def find(name: str | None) -> dict | None:
    """The archive entry of a data set name ("CoNLL 2003", "conll2003", "Gigaword"), or None."""
    k = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    if len(k) < 3 or k in _COMMON:
        return None
    return _names().get(k)


def in_text(text: str | None, min_papers: int = 5) -> list[str]:
    """Distinct data set names that occur in a text as 1-4 consecutive words (the canonical names of the archive, most-cited first)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*", text or "")
    seen: dict[str, int] = {}
    for n in (4, 3, 2, 1):
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i:i + n])
            if n == 1 and not (re.search(r"[A-Z]", chunk[1:]) or re.search(r"\d", chunk)):
                continue                                    # a single plain word is not a data set name ("Gigaword" and "CoNLL-03" have capitals / digits)
            e = find(chunk)
            if e and e["n"] >= min_papers:
                seen[e["name"]] = max(seen.get(e["name"], 0), e["n"])
    return [n for n, _ in sorted(seen.items(), key=lambda kv: -kv[1])]
