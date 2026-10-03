"""Stage 1 - Query Understanding (UPGRADED): intent, complexity, and the conditions the question implies.

Intent and complexity come from the SciBERT two-head classifier and the implied conditions (language, dataset, model
size ...) from the LLM. Until that classifier is trained the LLM produces all three zero-shot (approved deviation), and a
keyword heuristic covers LLM failure so this stage can never return nothing.

A small model sometimes misses a condition that is written plainly in the question, and stage 6 depends on them, so any
dataset, version, model or language named literally in the question is also picked up deterministically (dataset and
model names come from what the paper store already contains; languages from a fixed list).
"""
from __future__ import annotations

import re

from ..llm import OllamaLLM
from ..schemas import QUERY_CONDITION_FIELDS, QueryAnalysis, QueryConditions
from ..stores.profile_store import ProfileStore
from .conditions import TASK_WORDS, model_family, norm, setting_tags, split_version
from .intent_classifier import IntentClassifier

SYSTEM = (
    "You analyse a research question about computer-science papers. Return JSON with:\n"
    "intent: factual (a single fact), comparison (compare systems or settings), method (how something works), "
    "result (a reported score or outcome), or survey (overview of a topic).\n"
    "complexity: 'complex' if the question has several parts, compares several things, or needs multiple "
    "look-ups; otherwise 'simple'.\n"
    "conditions: the experimental conditions the question ITSELF names or clearly implies. "
    "task: the kind of problem, e.g. question answering or natural language inference - never a dataset or model name. "
    "dataset: the dataset name without its version, e.g. SQuAD. dataset_version: e.g. 2.0. "
    "language: a human language the models are evaluated on. model: e.g. BERT-large. model_size: e.g. 340M. "
    "setting: zero-shot, few-shot, fine-tuned, dev set, test set. "
    "If the question names several models, datasets or languages, put the first one named in model / dataset / language and the others in "
    "other_models / other_datasets / other_languages (never repeat the first one). A single one always goes in model / dataset / "
    "language, never in an other_* list.\n"
    "Use null for anything the question does not state. Never add a condition the question does not mention.\n"
    "Examples (question -> JSON):\n"
    'Does BERT work well for Kannada question answering? -> {"intent":"result","complexity":"simple","conditions":'
    '{"task":"question answering","language":"Kannada","model":"BERT"}}\n'
    'What F1 does BERT-large get on SQuAD v2.0? -> {"intent":"result","complexity":"simple","conditions":'
    '{"dataset":"SQuAD","dataset_version":"2.0","model":"BERT-large"}}\n'
    'Compare RoBERTa and XLNet on GLUE and explain what accounts for the gap. -> {"intent":"comparison",'
    '"complexity":"complex","conditions":{"dataset":"GLUE"}}\n'
    'How does the masked language model objective work? -> {"intent":"method","complexity":"simple","conditions":{}}\n'
    'Compare ALBERT and ELECTRA on RACE and SQuAD for English and Tamil. -> {"intent":"comparison","complexity":"complex","conditions":'
    '{"model":"ALBERT","other_models":["ELECTRA"],"dataset":"RACE","other_datasets":["SQuAD"],"language":"English",'
    '"other_languages":["Tamil"]}}'
)

LANGUAGES = (
    "English French German Spanish Italian Portuguese Russian Chinese Japanese Korean Arabic Hindi Bengali Urdu Tamil "
    "Telugu Kannada Malayalam Marathi Gujarati Punjabi Odia Assamese Nepali Sinhala Swahili Turkish Greek Bulgarian "
    "Vietnamese Thai Indonesian Malay Persian Hebrew Dutch Polish Czech Swedish Finnish Hungarian Romanian Ukrainian "
    "Yoruba Amharic Hausa Zulu").split()

_INTENT_KEYWORDS = [
    ("comparison", r"\b(compare|compared|versus|vs\.?|better than|worse than|difference|differ|outperform)\b"),
    ("method", r"\b(how does|how do|how is|method|approach|architecture|algorithm|technique|trained|training)\b"),
    ("result", r"\b(accuracy|score|f1|exact match|result|perform|performance|benchmark|achieve|obtain)\b"),
    ("survey", r"\b(survey|overview|review|what are the|main (ideas|approaches)|state of the art)\b"),
]


def heuristic_analysis(question: str) -> QueryAnalysis:
    q = question.lower()
    intent = next((name for name, pat in _INTENT_KEYWORDS if re.search(pat, q)), "factual")
    parts = q.count("?") > 1 or len(re.findall(r"\b(and|also|as well as)\b", q)) >= 2 or len(q.split()) > 30
    return QueryAnalysis(intent=intent, complexity="complex" if parts or intent == "comparison" else "simple")


def _in_question(field: str, value: str, question: str) -> bool:
    """Keep a condition only if the question really contains it (guards against an LLM inventing one)."""
    if norm(value) and norm(value) in norm(question):
        return True
    words = [w for w in re.findall(r"[a-z0-9.]+", value.lower()) if len(w) >= 3]
    if not words:
        return False
    text = question.lower()
    if field in ("task", "setting"):                 # free text: any content word of it in the question is enough
        return any(w in text for w in words)
    return all(w in text for w in words)


def _position(value: str, question: str) -> int:
    """Where the question first mentions `value` (a value it does not contain literally sorts last)."""
    at = question.lower().find(value.lower())
    return at if at >= 0 else len(question)


def clean_conditions(conditions: QueryConditions, question: str) -> QueryConditions:
    """Keep only conditions the question really states, and put each one in its proper field."""
    kept = {f: v for f, v in conditions.specified().items() if _in_question(f, v, question)}
    further: dict[str, list[str]] = {}
    for field, values in conditions.extras().items():           # further models / datasets / languages: only what the question really says
        values = [v for v in values if norm(v) and _in_question(field, v, question)]
        if values and not kept.get(field):                      # the model listed the only one among "the others" (seen with QA datasets):
            values.sort(key=lambda v: _position(v, question))   # the one named first is THE one
            kept[field] = values.pop(0)
        further[field] = values
    if kept.get("dataset") and norm(kept["dataset"]) in TASK_WORDS:
        kept.setdefault("task", kept.pop("dataset"))           # "NLI" / "question answering" is a task, not a dataset
    dataset = kept.get("dataset")
    if dataset and "dataset_version" not in kept:              # "SQuAD 2.0" is the dataset SQuAD, version 2.0
        name, version = split_version(dataset)
        if version:
            kept["dataset"], kept["dataset_version"] = name, version
    task = kept.get("task")
    if task and any(kept.get(f) and norm(kept[f]) in norm(task) for f in ("dataset", "model")):
        del kept["task"]                                        # a task is a kind of problem, not a dataset or model name
    if kept.get("setting") and not setting_tags(kept["setting"]):
        del kept["setting"]                                     # "human" is not an evaluation setting
    extras = {}
    for field, values in further.items():
        seen = {norm(kept.get(field))}
        extras[f"other_{field}s"] = [v for v in values if not (norm(v) in seen or seen.add(norm(v)))]
    return QueryConditions(**{f: kept.get(f) for f in QUERY_CONDITION_FIELDS}, **extras)


def _find(name: str, text: str) -> re.Match | None:
    return re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", text, re.I)


def vocabulary_conditions(question: str, vocab: dict[str, list[str]]) -> dict[str, str]:
    """Dataset, version, model and language written literally in the question."""
    found: dict[str, str] = {}
    for name in sorted({d for d in vocab.get("dataset", []) if len(d) >= 3 and norm(d) not in TASK_WORDS}, key=len, reverse=True):
        m = _find(name, question)
        if m:
            found["dataset"] = name
            version = re.match(r"\s*(?:v|version\s*)?(\d+(?:\.\d+)+)", question[m.end():], re.I)
            if version:
                found["dataset_version"] = version.group(1)
            break
    families = {model_family(m) for m in vocab.get("model", [])} - {""}
    for word in re.findall(r"[A-Za-z][\w+\-]*", question):
        family = model_family(word)
        if len(family) >= 3 and family in families:
            found["model"] = word
            break
    for language in LANGUAGES:
        if _find(language, question):
            found["language"] = language
            break
    return found


def extras_from_vocabulary(question: str, vocab: dict[str, list[str]], primary: dict[str, str]) -> dict[str, list[str]]:
    """Further models and languages written literally in the question (the LLM may name only the first)."""
    out: dict[str, list[str]] = {}
    families = {model_family(m) for m in vocab.get("model", [])} - {""}
    seen = {model_family(primary["model"])} if primary.get("model") else set()
    models = []
    for word in re.findall(r"[A-Za-z][\w+\-]*", question):
        family = model_family(word)
        if len(family) >= 3 and family in families and family not in seen:
            seen.add(family)
            models.append(word)
    if models:
        out["other_models"] = models
    languages = [lang for lang in LANGUAGES if _find(lang, question) and lang != primary.get("language")]
    if languages:
        out["other_languages"] = languages
    return out


class QueryUnderstanding:
    """SciBERT (intent + complexity) when a trained checkpoint exists, LLM zero-shot otherwise; the LLM always
    supplies the conditions, backed up by names found literally in the question."""

    def __init__(self, llm: OllamaLLM | None = None, classifier: "IntentClassifier | None | str" = "auto",
                 profiles: ProfileStore | None = None) -> None:
        self.llm = llm or OllamaLLM()
        self.classifier = IntentClassifier.load() if classifier == "auto" else classifier
        self.profiles = profiles

    @property
    def source(self) -> str:
        return "scibert+llm" if self.classifier else "llm-zero-shot"

    def analyze(self, question: str) -> QueryAnalysis:
        try:
            out, _ = self.llm.structured(f"Question: {question}", QueryAnalysis, system=SYSTEM, max_tokens=300, retries=0)
        except ValueError:
            out = heuristic_analysis(question)
        intent, complexity = out.intent, out.complexity
        if self.classifier:
            pred = self.classifier.predict(question)
            intent, complexity = pred.intent, pred.complexity
        cleaned = clean_conditions(out.conditions, question)
        conditions = cleaned.specified()
        extras = {f"other_{field}s": list(values) for field, values in cleaned.extras().items()}
        if self.profiles is not None:
            vocab = self.profiles.vocabulary()
            for field, value in vocabulary_conditions(question, vocab).items():
                said = conditions.setdefault(field, value)   # the LLM's reading wins; names it missed are added
                if field == "dataset" and said != value and norm(said) in norm(value):
                    conditions[field] = value                # ... unless it kept only a part of a name the question writes in full
                                                             # ("GoldP" for "TyDi QA GoldP": the part matches nothing in the store)
            for key, values in extras_from_vocabulary(question, vocab, conditions).items():
                have = {norm(v) for v in extras.get(key, [])}
                extras[key] = extras.get(key, []) + [v for v in values if norm(v) not in have]
        return QueryAnalysis(intent=intent, complexity=complexity,
                             conditions=QueryConditions(**{f: conditions.get(f) for f in QUERY_CONDITION_FIELDS}, **extras))
