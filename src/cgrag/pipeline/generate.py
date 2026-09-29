"""Stage 8 - Generate (UPGRADED): answer from the top chunks and chat history.

The prompt now also carries the scope warning from stage 6 and the explanations for conflicts from stage 7.
"""
from __future__ import annotations

from ..llm import OllamaLLM
from ..schemas import ConditionCheck, ConditionProfile, ContradictionPair, RetrievedChunk
from .conditions import claim_text, observed_values, values_match

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
    "6. Be concise: at most 180 words, no preamble."
)


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


def _relevance(p: ConditionProfile, requested: dict[str, str]) -> int:
    """How many of the question's conditions this recorded result satisfies."""
    return sum(any(values_match(f, wanted, v) for v in observed_values(p, f)) for f, wanted in requested.items())


def format_sources(sources: list[RetrievedChunk], profiles: dict[str, list[ConditionProfile]], include_profiles: bool,
                   requested: dict[str, str] | None = None) -> str:
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
             requested: dict[str, str] | None = None, feedback: str | None = None) -> str:
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
    return llm.chat(messages, max_tokens=MAX_ANSWER_TOKENS).text.strip()
