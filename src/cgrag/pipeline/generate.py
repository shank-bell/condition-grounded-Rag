"""Stage 8 - Generate (UPGRADED): answer from the top chunks and chat history.

The prompt now also carries the scope warning from stage 6 and the explanations for conflicts from stage 7.
"""
from __future__ import annotations

import re

from ..config import get_settings
from ..llm import OllamaLLM
from ..schemas import ConditionCheck, ConditionProfile, ContradictionPair, RetrievedChunk
from .conditions import claim_text, observed_values, values_match
from .text import _SENTENCE_END, strip_citations

MAX_ANSWER_TOKENS = 500
MAX_PROFILE_LINES = 6
MAX_EXTRA_CONFLICT_SOURCES = 2
MAX_EXTRA_COVERING_SOURCES = 2

SYSTEM = (
    "You are a careful research assistant answering questions about computer-science papers.\n"
    "1. Use ONLY the numbered SOURCES. Never use outside knowledge.\n"
    "2. Put the source number after every factual sentence, like [1] or [2][3].\n"
    "3. When you give a result, say the conditions it holds under (dataset and version, model and size, language, "
    "evaluation setting) exactly as the sources record them.\n"
    "4. Only if the message has a SCOPE WARNING section: tell the reader plainly which condition the evidence does not "
    "cover, and do not claim the result holds for it. Without that section never write the words 'scope warning'; "
    "write any other caveat as a sentence starting with 'Note:'.\n"
    "5. Only if the message has a CONFLICTS section: describe each conflict. A difference explained by a named condition "
    "is not a real contradiction: say which condition explains it. A genuine conflict must be reported as unresolved.\n"
    "6. Never guess why two numbers differ. Give each number once, with its own source and the conditions the source "
    "records. Name a reason only when a CONFLICTS line gives it or the cited text itself states it; a different paper, "
    "a re-run or a different model size is a condition to name, not an explanation to invent. A 'Note:' states a fact "
    "from the sources, never a guessed cause.\n"
    "7. Be concise: at most 180 words, no preamble."
)

_REASON = re.compile(r"\b(?:explained|attributed|accounted\s+for|due\s+to|owing\s+to|because\s+of|caused\s+by|results?\s+from|stems?\s+from|reflects?)\b", re.I)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def drop_guessed_reasons(answer: str, conflict_text: str) -> str:
    """Remove each sentence that gives a reason for numbers when Stage 7 gave none for them. A sentence without numbers, and every sentence that only
    repeats a CONFLICTS line, stays; a line that is left with nothing but "Note:" is removed."""
    allowed = set(_NUMBER.findall(conflict_text))
    lines: list[str] = []
    for line in answer.split("\n"):
        if not line.strip():
            lines.append(line)
            continue
        kept = []
        for sentence in _SENTENCE_END.split(line):
            numbers = set(_NUMBER.findall(strip_citations(sentence)))
            if _REASON.search(sentence) and numbers and not numbers <= allowed:
                continue
            kept.append(sentence)
        text = " ".join(x.strip() for x in kept if x.strip())
        if text.rstrip(":* ").strip().lower() not in ("", "note"):
            lines.append(text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def select_sources(kept: list[RetrievedChunk], conflicts: list[ContradictionPair], top: int,
                   checks: list[ConditionCheck] | None = None) -> list[RetrievedChunk]:
    """The top chunks for the prompt, plus a chunk that stage 6 found to cover each requested condition (so the answer can
    use the evidence that matches the question's conditions) and the chunks behind reported conflicts (so those can be
    explained)."""
    sources = list(kept[:top])
    have = {rc.chunk.chunk_id for rc in sources}
    added = 0
    for check in checks or []:
        if check.covered and not have & set(check.chunk_ids) and added < MAX_EXTRA_COVERING_SOURCES:
            match = next((rc for rc in kept if rc.chunk.chunk_id in check.chunk_ids), None)
            if match:
                sources.append(match)
                have.add(match.chunk.chunk_id)
                added += 1
    extra = 0
    for c in conflicts:
        for cid in (c.chunk_a, c.chunk_b):
            if cid not in have and extra < MAX_EXTRA_CONFLICT_SOURCES:
                match = next((rc for rc in kept if rc.chunk.chunk_id == cid), None)
                if match:
                    sources.append(match)
                    have.add(cid)
                    extra += 1
    return sources


def _relevance(p: ConditionProfile, requested: dict[str, str | list[str]]) -> int:
    """How many of the question's conditions this recorded result satisfies. A condition the question names with several values ("Hindi and Tamil",
    "MuRIL and mBERT") is satisfied by any of them, so the result of each named language or system is listed before the results of the others
    (found 11 Oct: with one wanted value per field, mBERT's Tamil result of a 19-column table fell behind the six results that matched only the model and
    the data set, and the answer said that no Tamil result is recorded)."""
    total = 0
    for f, wanted in requested.items():
        values = [w for w in (wanted if isinstance(wanted, (list, tuple)) else [wanted]) if w]
        total += any(values_match(f, w, v) for w in values for v in observed_values(p, f))
    return total


def format_sources(sources: list[RetrievedChunk], profiles: dict[str, list[ConditionProfile]], include_profiles: bool,
                   requested: dict[str, str | list[str]] | None = None) -> str:
    """The numbered sources for the prompt. When a passage has many recorded results, the ones that match the question's
    conditions are listed first, so the answer can find e.g. the Hindi result of a 15-language table."""
    blocks = []
    for i, rc in enumerate(sources, start=1):
        c = rc.chunk
        head = f"[{i}] {c.paper_title or c.paper_id} (page {c.page}, {c.section})"
        recorded = ""
        if include_profiles and profiles.get(c.chunk_id):
            ranked = sorted(profiles[c.chunk_id], key=lambda p: -_relevance(p, requested)) if requested else profiles[c.chunk_id]
            recorded = "\nRecorded results: " + " ".join(claim_text(p) for p in ranked[:MAX_PROFILE_LINES])
        blocks.append(f"{head}{recorded}\n{c.text}")
    return "\n\n".join(blocks)


def describe_conflicts(conflicts: list[ContradictionPair], sources: list[RetrievedChunk]) -> str:
    index = {rc.chunk.chunk_id: i for i, rc in enumerate(sources, start=1)}
    lines = []
    for c in conflicts:
        a, b = index.get(c.chunk_a), index.get(c.chunk_b)
        if a and b:
            lines.append(f"- Sources [{a}] and [{b}] ({c.verdict.replace('_', ' ').lower()}): {c.reason}")
    return "\n".join(lines)


def generate(question: str, sources: list[RetrievedChunk], llm: OllamaLLM, *, warning: str | None = None,
             conflicts: list[ContradictionPair] | None = None, history: list[dict] | None = None,
             profiles: dict[str, list[ConditionProfile]] | None = None, include_profiles: bool = True,
             requested: dict[str, str | list[str]] | None = None, feedback: str | None = None) -> str:
    parts = [f"QUESTION: {question}"]
    if warning:
        parts.append(f"SCOPE WARNING: {warning}")
    conflict_text = describe_conflicts(conflicts or [], sources)
    if conflict_text:
        parts.append(f"CONFLICTS:\n{conflict_text}")
    parts.append("SOURCES:\n" + format_sources(sources, profiles or {}, include_profiles, requested))
    if feedback:
        parts.append("Your previous answer contained sentences the sources do not support:\n" + feedback +
                     "\nWrite the answer again. Keep only statements that a cited source supports.")
    messages = [{"role": "system", "content": SYSTEM}, *(history or []), {"role": "user", "content": "\n\n".join(parts)}]
    answer = llm.chat(messages, max_tokens=MAX_ANSWER_TOKENS).text.strip()
    if get_settings().features.strip_guessed_reasons:
        answer = drop_guessed_reasons(answer, conflict_text)
    return answer
