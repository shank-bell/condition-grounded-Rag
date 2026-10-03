"""Stage 9 - Claim-Check Critic (agent/critic): is every sentence of the answer entailed by the chunk it cites?

Each sentence is checked with NLI against its cited chunk (or against all sources when it cites none). The premise is the
few sentences of the chunk that best match the claim, plus the chunk's recorded results when the claim quotes one of
their numbers, so a claim read off a table can still be verified. NLI models are weak on tables, so a sentence whose
every number occurs in the cited source is also accepted when the NLI model does not contradict it. A sentence that its
cited sources do not support but another source does is a mis-citation: it counts as supported and its citation number
is repaired. Any sentence still unsupported triggers one regeneration.
"""
from __future__ import annotations

import re
from typing import Callable

from ..config import CriticConfig, get_settings
from ..models import NLI, get_nli
from ..schemas import ClaimCheck, ConditionProfile, RetrievedChunk
from .conditions import claim_text, metric_key, model_family, norm, setting_tags
from .text import citations, overlap, plain, split_sentences, strip_citations

# Sentences that say what the sources do NOT contain make no claim about them, so there is nothing to verify.
_META = re.compile(
    r"\b(scope warning|not covered|does not cover|do not cover|cannot be determined|unresolved|"
    r"(?:do|does|did) not (?:contain|mention|provide|include|report|address|state|specify|cover)|"
    r"no (?:(?:retrieved|reported|recorded|available) )?(?:information|evidence|data|sources?|results?|scores?|numbers?|findings?|"
    r"reports?|studies|papers?)|"                                   # "No results are reported for Kannada ..." (what the scope warning asks for)
    r"(?:(?:is|are|was|were) not|(?:has|have) not been) (?:reported|recorded|evaluated|tested|available|included|mentioned|provided|found)|"
    r"none of the (?:sources|papers|studies)|"
    r"not (?:enough|sufficient)|insufficient|cannot answer|unable to|"
    r"not reported|do not assume|does not assume)\b", re.I)
# The answer restating a CONFLICTS line ("Sources [4] and [5] (not comparable): ...") is not a claim about the papers.
_ECHO = re.compile(r"\bsources?\s*\[\d+\]\s*(?:and|&|,)\s*\[\d+\]", re.I)
WINDOW = 3
MIN_WORDS = 4
GROUNDED_CONTRADICTION_MAX = 0.30      # a number-grounded sentence is still rejected if NLI contradicts it this much
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")       # standalone numbers only: the 1 in "F1" is not one


def _premise(claim: str, chunk_text: str, profiles: list[ConditionProfile]) -> str:
    """The chunk sentences most like the claim (in reading order), plus profile lines that quote its numbers."""
    sentences = split_sentences(chunk_text, min_chars=8) or [chunk_text]
    top = sorted(sorted(range(len(sentences)), key=lambda i: overlap(claim, sentences[i]), reverse=True)[:WINDOW])
    premise = " ".join(sentences[i] for i in top)
    numbers = set(_NUMBER.findall(claim))
    quoted = [claim_text(p) for p in profiles if f"{p.value:g}" in numbers]
    return " ".join([premise, *quoted[:4]])


def _numbers_grounded(claim: str, chunk_text: str, profiles: list[ConditionProfile]) -> bool:
    """True when the claim has numbers and every one of them occurs in the source (its text or recorded results)."""
    numbers = set(_NUMBER.findall(claim))
    return bool(numbers) and numbers <= set(_NUMBER.findall(chunk_text)) | {f"{p.value:g}" for p in profiles}


_GENERIC_METRICS = {"accuracy", "acc", "f1", "em", "exactmatch", "bleu", "rouge", "perplexity", "spearman", "pearson", "auc", "recall",
                    "precision", "mrr"}


def _metric_named(metric: str, flat: str, words: set[str]) -> bool:
    m = norm(metric)
    return (len(m) >= 3 and m in flat) or metric_key(metric) in words or m in words


def _profile_supports(claim: str, profiles: list[ConditionProfile]) -> bool:
    """A recorded result of the source that matches the claim's number, metric and model (and does not contradict its
    dev/test setting). This is the strongest evidence a table-derived claim can have, and NLI cannot judge tables.

    A list item such as "- mBERT: 58.6 (test set)" does not repeat the metric - the sentence above it ("... accuracy
    scores:") did - so when the claim names no metric at all (neither one the source records nor a common one), number +
    model + setting decide; a claim that does name a metric must match the recorded one."""
    numbers, words, flat, tags = set(_NUMBER.findall(claim)), set(re.findall(r"[a-z0-9]+", claim.lower())), norm(claim), setting_tags(claim)
    names_a_metric = bool(words & _GENERIC_METRICS) or any(p.metric and _metric_named(p.metric, flat, words) for p in profiles)
    for p in profiles:
        if f"{p.value:g}" not in numbers or not p.metric:
            continue
        if names_a_metric and not _metric_named(p.metric, flat, words):
            continue
        family = model_family(p.model)
        if p.model and len(family) >= 3 and family not in flat:
            continue
        recorded = setting_tags(p.setting)
        if tags and recorded and not (tags & recorded):
            continue                                            # e.g. the claim says dev set, the result is a test result
        return True
    return False


def _recite(sentence: str, number: int) -> str:
    """The sentence with its citation replaced by [number], placed before the closing punctuation."""
    bare = strip_citations(sentence).rstrip()
    end = bare[-1] if bare and bare[-1] in ".!?" else ""
    return f"{bare[:-1] if end else bare} [{number}]{end}"


class ClaimChecker:
    def __init__(self, cfg: CriticConfig | None = None, nli_factory: Callable[[], NLI] = get_nli) -> None:
        self.cfg = cfg or get_settings().critic
        self._nli_factory = nli_factory

    def _best(self, claims: list[tuple[str, str, list[int]]], sources: list[RetrievedChunk],
              profiles: dict[str, list[ConditionProfile]], todo: dict[int, list[int]]) -> dict[int, tuple[float, float, float, int]]:
        """Per claim: (supported 1/0, entailment, contradiction, source index) of its best source among `todo`."""
        pairs, owner = [], []
        for ci, indices in todo.items():
            text = claims[ci][1]
            for si in indices:
                chunk = sources[si].chunk
                pairs.append((_premise(text, chunk.text, profiles.get(chunk.chunk_id, [])), text))
                owner.append((ci, si))
        if not pairs:
            return {}
        best: dict[int, tuple[float, float, float, int]] = {}
        for (ci, si), (entail, _, contra) in zip(owner, self._nli_factory().probs(pairs)):
            chunk = sources[si].chunk
            recorded = profiles.get(chunk.chunk_id, [])
            ok = entail >= self.cfg.entail_threshold or _profile_supports(claims[ci][1], recorded) or (
                contra < GROUNDED_CONTRADICTION_MAX and _numbers_grounded(claims[ci][1], chunk.text, recorded))
            cand = (float(ok), float(entail), float(contra), si)
            if ci not in best or cand[:2] > best[ci][:2]:          # supported first, then higher entailment
                best[ci] = cand
        return best

    @staticmethod
    def _is_bare_condition(text: str, sources: list[RetrievedChunk], profiles: dict[str, list[ConditionProfile]]) -> bool:
        """Is a short line only a condition value with a number - "CoLA: 56.3" under a model heading - rather than a system with a
        number ("MuRIL: 67.8")? The first has no subject to verify; the second does, and must be backed by a recorded result."""
        label = norm(text.split(":")[0]) if ":" in text else norm(re.sub(r"[\d.%]+", " ", text))
        if not label:
            return True
        known = {norm(getattr(p, f)) for rc in sources for p in profiles.get(rc.chunk.chunk_id, [])
                 for f in ("dataset", "language", "metric", "task", "setting") if getattr(p, f)}
        return label in known

    def check(self, answer: str, sources: list[RetrievedChunk], profiles: dict[str, list[ConditionProfile]]) -> list[ClaimCheck]:
        claims: list[tuple[str, str, list[int]]] = []        # (original sentence, plain claim, cited source indices)
        for sentence in split_sentences(answer, min_chars=1):
            text = plain(sentence)
            words = text.split()
            if len(words) < MIN_WORDS and (not _NUMBER.search(text) or self._is_bare_condition(text, sources, profiles)):
                continue                       # "MuRIL: 67.8" is a claim; "CoLA: 56.3" under a model heading has no subject to verify
            if _META.search(text) or _ECHO.search(sentence) or text.rstrip().endswith(":"):
                continue                       # headings, hedges, list lead-ins ("... achieved these scores:"), the scope warning and
                                               # conflict echoes make no claim; the list items that follow are checked one by one
            cited = [n - 1 for n in citations(sentence) if 1 <= n <= len(sources)]
            claims.append((sentence, text, cited or list(range(len(sources)))))
        if not claims or not sources:
            return []
        best = self._best(claims, sources, profiles, {ci: idx for ci, (_, _, idx) in enumerate(claims)})
        # a sentence its cited sources do not support may still be supported by another source (a mis-citation)
        others = {ci: [si for si in range(len(sources)) if si not in claims[ci][2]] for ci in best if not best[ci][0]}
        for ci, cand in self._best(claims, sources, profiles, {ci: idx for ci, idx in others.items() if idx}).items():
            if cand[0]:
                best[ci] = cand
        return [
            ClaimCheck(sentence=sentence, supported=bool(best[ci][0]), chunk_id=sources[best[ci][3]].chunk.chunk_id,
                       entailment=best[ci][1])
            for ci, (sentence, _, _) in enumerate(claims)
        ]

    @staticmethod
    def repair_citations(answer: str, checks: list[ClaimCheck], sources: list[RetrievedChunk]) -> tuple[str, list[ClaimCheck]]:
        """A sentence that is supported, but by a source it did not cite, gets the number of the source that supports it."""
        index = {rc.chunk.chunk_id: i for i, rc in enumerate(sources, start=1)}
        repaired: list[ClaimCheck] = []
        for c in checks:
            cited, actual = citations(c.sentence), index.get(c.chunk_id or "")
            if c.supported and cited and actual and actual not in cited:
                fixed = _recite(c.sentence, actual)
                answer = answer.replace(c.sentence, fixed, 1)
                c = c.model_copy(update={"sentence": fixed})
            repaired.append(c)
        return answer, repaired

    @staticmethod
    def unsupported(checks: list[ClaimCheck]) -> list[ClaimCheck]:
        return [c for c in checks if not c.supported]
