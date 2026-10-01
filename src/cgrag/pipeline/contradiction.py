"""Stage 7 - Condition-Aware Contradiction Resolver (NEW, agent/analyst).

Two signals flag a candidate pair of retrieved chunks from different papers:
  * numeric: the two chunks report the same metric of the same model on the same dataset with values that differ
    by more than a relative threshold (read from their Condition Profiles);
  * NLI: the most question-relevant sentences of the two chunks contradict each other (strict: both sentences must be
    about the question, share subject words, and contradict in both directions).
The profiles of a flagged pair then decide the class:
  GENUINE         same recorded conditions, different result
  EXPLAINED       a recorded condition differs (dataset version, model size, setting ...) and is named
  NOT_COMPARABLE  the conditions are not recorded on both sides (profile missing), so no verdict can be given
When the question names a model or dataset, only conflicts about that model / dataset are reported, once per pair of
papers and subject.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Callable

from ..config import ContradictionConfig, get_settings
from ..models import NLI, get_nli, rerank_scores
from ..schemas import ConditionProfile, ContradictionPair, RetrievedChunk
from ..stores.profile_store import ProfileStore
from .conditions import (
    claim_text, differing_conditions, metric_key, model_family, norm, observed_values, unrecorded_conditions,
    values_match,
)
from .text import split_sentences

MAX_REPORTED = 6
MAX_TEXTUAL = 2
MAX_SENTENCES_PER_CHUNK = 30
# a reported result: a decimal ("83.6") or a percentage ("97%"). Not "2019" in a citation, not the 6 of "BERT6" or "66 million".
_NUMBER = re.compile(r"(?<![\w.])\d+\.\d+(?!\w)|(?<![\w.])\d+(?:\.\d+)?\s?%")
TEXT_RELEVANCE = 0.0          # cross-encoder score a key sentence must reach to count as being about the question
MIN_SHARED_WORDS = 2          # two sentences can only contradict each other if they talk about the same things
_STOP = frozenset(
    "the and for with that this from are was were has have been which their these those than then into also such more "
    "most some each both when where while about would could there them they its not but can will our over only other "
    "between".split())
_LABEL = {"dataset_version": "dataset version", "model_size": "model size", "language": "language",
          "setting": "evaluation setting", "task": "task"}


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def same_subject(a: ConditionProfile, b: ConditionProfile) -> bool:
    """Both profiles report the same metric of the same model family on the same dataset."""
    return bool(
        a.dataset and b.dataset and values_match("dataset", a.dataset, b.dataset)
        and metric_key(a.metric) and metric_key(a.metric) == metric_key(b.metric)
        and model_family(a.model) and model_family(a.model) == model_family(b.model)
    )


def relevant(x: ConditionProfile, y: ConditionProfile, requested: dict[str, str] | None) -> bool:
    """When the question names a model, a dataset or a language, a conflict matters only if both results are about it (a question
    about Kannada is not answered by a disagreement between two Hindi results). Results with no recorded language are taken to
    be English, so they still count when the question asks about English."""
    for field in ("model", "dataset", "language"):
        want = (requested or {}).get(field)
        if not want:
            continue
        for p in (x, y):
            seen = observed_values(p, field)
            if field == "language" and not seen and norm(want) == "english":
                continue
            if not any(values_match(field, want, v) for v in seen):
                return False
    return True


def classify(a: ConditionProfile, b: ConditionProfile) -> tuple[str, list[str], str]:
    """(verdict, differing condition names, reason) for two profiles of the same subject with different values."""
    values = f"{a.value:g} vs {b.value:g} {a.metric or ''}".strip()
    differing = differing_conditions(a, b)
    if differing:
        detail = "; ".join(f"{_LABEL[f]} ({_val(a, f)} vs {_val(b, f)})" for f in differing)
        return "EXPLAINED", differing, f"Explained difference ({values}): {detail}."
    missing = unrecorded_conditions(a, b)
    if missing:
        names = ", ".join(_LABEL[f] for f in missing)
        return "NOT_COMPARABLE", [], (f"The results differ ({values}) but {names} is recorded for only one of the two "
                                      f"papers, so it cannot be shown that the conditions are the same.")
    return "GENUINE", [], f"Same model, dataset, metric and recorded conditions, but different results ({values})."


def _val(p: ConditionProfile, field: str) -> str:
    v = observed_values(p, field)
    return v[0] if v else "unrecorded"


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in _STOP}


class ContradictionResolver:
    def __init__(self, profiles: ProfileStore, cfg: ContradictionConfig | None = None,
                 nli_factory: Callable[[], NLI] = get_nli) -> None:
        self.profiles = profiles
        self.cfg = cfg or get_settings().contradiction
        self._nli_factory = nli_factory

    def _both_directions(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        """P(contradiction) premise->hypothesis and hypothesis->premise for each pair."""
        if not pairs:
            return []
        nli = self._nli_factory()
        forward = nli.probs(pairs)[:, 2]
        backward = nli.probs([(h, p) for p, h in pairs])[:, 2]
        return [(float(f), float(b)) for f, b in zip(forward, backward)]

    def resolve(self, question: str, chunks: list[RetrievedChunk], requested: dict[str, str] | None = None) -> list[ContradictionPair]:
        pairs = [(a, b) for a, b in combinations(chunks, 2) if a.chunk.paper_id != b.chunk.paper_id][: self.cfg.max_pairs]
        by_chunk = self.profiles.for_chunks([c.chunk.chunk_id for c in chunks])
        best: dict[tuple, tuple[RetrievedChunk, RetrievedChunk, ConditionProfile, ConditionProfile]] = {}
        rest: list[tuple[RetrievedChunk, RetrievedChunk]] = []
        for a, b in pairs:
            hits = [(x, y) for x in by_chunk.get(a.chunk.chunk_id, []) for y in by_chunk.get(b.chunk.chunk_id, [])
                    if same_subject(x, y) and rel_diff(x.value, y.value) > self.cfg.numeric_rel_diff and relevant(x, y, requested)]
            if not hits:
                rest.append((a, b))
                continue
            x, y = max(hits, key=lambda h: rel_diff(h[0].value, h[1].value))
            key = (frozenset((a.chunk.paper_id, b.chunk.paper_id)), metric_key(x.metric), norm(x.dataset), model_family(x.model),
                   tuple(sorted((x.dataset_version or "", y.dataset_version or ""))))
            if key not in best or rel_diff(x.value, y.value) > rel_diff(best[key][2].value, best[key][3].value):
                best[key] = (a, b, x, y)          # one conflict per pair of papers and subject
        flagged = list(best.values())
        found: list[ContradictionPair] = []
        probs = self._both_directions([(claim_text(x), claim_text(y)) for _, _, x, y in flagged])
        for (a, b, x, y), (fw, bw) in zip(flagged, probs):
            verdict, differing, reason = classify(x, y)
            found.append(ContradictionPair(
                verdict=verdict, chunk_a=a.chunk.chunk_id, chunk_b=b.chunk.chunk_id, paper_a=a.chunk.paper_id,
                paper_b=b.chunk.paper_id, metric=x.metric, value_a=x.value, value_b=y.value, differing=differing,
                reason=reason, nli_contradiction=max(fw, bw)))
        found += self._textual(question, rest)
        order = {"GENUINE": 0, "EXPLAINED": 1, "NOT_COMPARABLE": 2}
        found.sort(key=lambda c: (order[c.verdict], -(c.nli_contradiction or 0.0)))
        return found[:MAX_REPORTED]

    def _textual(self, question: str, pairs: list[tuple[RetrievedChunk, RetrievedChunk]]) -> list[ContradictionPair]:
        """NLI over the most question-relevant sentence of each chunk, for pairs the numbers did not flag."""
        if not pairs:
            return []
        chunk_by_id = {rc.chunk.chunk_id: rc.chunk for pair in pairs for rc in pair}
        # Only sentences that report a number can contradict each other as RESULTS: "We compare TinyBERT with ..." and "We
        # compare our MobileBERT with ..." make NLI say "contradiction" (different subjects), and that is not a conflict.
        sentences = {cid: [s for s in split_sentences(c.text)[:MAX_SENTENCES_PER_CHUNK] if _NUMBER.search(s)]
                     for cid, c in chunk_by_id.items()}
        flat = [(cid, s) for cid, ss in sentences.items() for s in ss]
        if not flat:
            return []
        scores = rerank_scores(question, [s for _, s in flat])
        best: dict[str, tuple[float, str]] = {}
        for (cid, s), score in zip(flat, scores):
            if cid not in best or score > best[cid][0]:
                best[cid] = (score, s)
        usable = [
            (a, b) for a, b in pairs
            if a.chunk.chunk_id in best and b.chunk.chunk_id in best
            and best[a.chunk.chunk_id][0] >= TEXT_RELEVANCE and best[b.chunk.chunk_id][0] >= TEXT_RELEVANCE
            and len(_content_words(best[a.chunk.chunk_id][1]) & _content_words(best[b.chunk.chunk_id][1])) >= MIN_SHARED_WORDS
        ]
        probs = self._both_directions([(best[a.chunk.chunk_id][1], best[b.chunk.chunk_id][1]) for a, b in usable])
        out = [
            ContradictionPair(
                verdict="NOT_COMPARABLE", chunk_a=a.chunk.chunk_id, chunk_b=b.chunk.chunk_id, paper_a=a.chunk.paper_id,
                paper_b=b.chunk.paper_id, reason=("The sources disagree in wording, but there is no pair of recorded "
                                                  "results with the same subject to compare (profile missing)."),
                nli_contradiction=min(fw, bw))
            for (a, b), (fw, bw) in zip(usable, probs) if min(fw, bw) >= self.cfg.nli_threshold     # contradiction both ways
        ]
        out.sort(key=lambda c: -(c.nli_contradiction or 0.0))
        return out[:MAX_TEXTUAL]
