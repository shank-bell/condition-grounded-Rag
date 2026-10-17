"""Follow-up resolution (before Stage 1): a question that leans on the chat ("Can you detail more on it?") is rewritten into one that stands alone.
Stages 1 to 7 read the question only, so without this a follow-up names no condition and finds no passage; before, only the writer saw the history.
The same reading says whether the message asks for MORE DETAIL on the last answer, so that the writer adds to it instead of repeating it."""
from __future__ import annotations

import re
from difflib import get_close_matches
from typing import NamedTuple

from pydantic import BaseModel

from ..llm import OllamaLLM

MAX_TURNS = 4
MAX_TURN_CHARS = 600
MAX_REWRITE_CHARS = 400
SHORT_QUESTION_WORDS = 6

CUES = re.compile(
    r"\b(it|its|they|them|their|theirs|these|those|same|above|previous|earlier|former|latter|more|further|"
    r"elaborate|elaboration|detail|details|detailed|expand|continue|else|"
    r"what about|how about|and for|and in|and on|and with)\b",
    re.I,
)

WORD = re.compile(r"\w[\w+\-./]*")
NAME_MATCH_CUTOFF = 0.85

SYSTEM = (
    "You rewrite the last message of a chat about computer-science research papers into one question that can be understood without the chat. "
    "If the message refers to the earlier conversation (it, they, that, the same, more detail, \"what about Hindi?\"), replace each reference by "
    "what it stands for, and keep every model, data set, version, language, setting and number that the message depends on. "
    "If the message already stands on its own, return it unchanged. "
    "Use only what the conversation says. Do not answer the question and do not add facts. Return one question. "
    "Also set more_detail to true when the message asks the assistant to explain, elaborate on or give more detail about something it already said, "
    "and to false for any other message."
)


class Standalone(BaseModel):
    question: str
    more_detail: bool = False


class Resolved(NamedTuple):
    question: str
    more_detail: bool


def needs_chat(question: str) -> bool:
    """A short question, or one that points back (it, they, the same, more detail, what about ...), may depend on the chat."""
    return len(question.split()) <= SHORT_QUESTION_WORDS or bool(CUES.search(question))


def transcript(history: list[dict]) -> str:
    """The last few turns of the chat as plain text, each cut to a few hundred characters."""
    turns = [t for t in history if str(t.get("content") or "").strip()][-MAX_TURNS:]
    return "\n".join(f"{'User' if t.get('role') == 'user' else 'Assistant'}: {str(t['content']).strip()[:MAX_TURN_CHARS]}" for t in turns)


def _is_name(token: str) -> bool:
    """A model or data set name: four or more characters with a letter, and a capital inside it, a digit or a plus sign (IndicBERT+Samanantar, SQuAD)."""
    return len(token) >= 4 and any(c.isalpha() for c in token) and (any(c.isdigit() or c == "+" for c in token) or any(c.isupper() for c in token[1:]))


def restore_names(rewritten: str, *known_text: str) -> str:
    """A name in the rewrite that the chat and the question never wrote, but that is a near miss of one they did, is the model's slip of the pen
    ("IndicBERT+Samanyar" for "IndicBERT+Samanantar"): it is put back as the chat spelled it, so that Stage 1 and Stage 6 look for a name that exists."""
    known: dict[str, str] = {}
    for text in known_text:
        for word in WORD.findall(text):
            token = word.rstrip(".-/")
            if _is_name(token):
                known.setdefault(token.lower(), token)

    def fix(match: re.Match) -> str:
        word = match.group(0)
        token = word.rstrip(".-/")
        if not _is_name(token) or token.lower() in known:
            return word
        close = get_close_matches(token.lower(), list(known), n=1, cutoff=NAME_MATCH_CUTOFF)
        return known[close[0]] + word[len(token):] if close else word

    return WORD.sub(fix, rewritten)


def resolve(question: str, history: list[dict] | None, llm: OllamaLLM) -> Resolved:
    """The question as a question that stands alone, and whether it asks for more detail on the last answer. Anything unusable (no chat, a model
    error, an empty, long or multi-line rewrite) returns the question unchanged and no wish for more detail, which is what the pipeline did before."""
    chat = transcript(history or [])
    if not chat:
        return Resolved(question, False)
    try:
        out, _ = llm.structured(f"Conversation so far:\n{chat}\n\nLast message: {question}", Standalone, system=SYSTEM,
                                max_tokens=200, retries=0, temperature=0.0)
    except Exception:
        return Resolved(question, False)
    rewritten = out.question.strip().strip('"').strip()
    if not rewritten or "\n" in rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        return Resolved(question, False)
    return Resolved(restore_names(rewritten, chat, question), bool(out.more_detail))


def resolve_followup(question: str, history: list[dict] | None, llm: OllamaLLM) -> str:
    """Only the stand-alone question of `resolve`."""
    return resolve(question, history, llm).question
