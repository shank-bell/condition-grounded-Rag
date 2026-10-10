"""Benchmarks and the task they measure (11 Oct 2026, written for the profile repair, ingestion/repair.py).

Why. On the 320 hand-checked profiles (docs/evaluation_ai_annotated.md, 4.1) the most frequent extraction error is the task field: 67 profiles hold a data set
name as the task ("EnDe", "GLUE", "CNNDM", "diagnostic set") or a task that the data set does not measure (MRPC as sentiment analysis, QQP or QNLI as question
answering, Tatoeba as translation in a retrieval table). A benchmark's task is a fact about the benchmark, so a list can settle it.

Scope, on purpose. The list holds the benchmarks of the 28-paper development corpus (the stored dataset names with at least a few profiles) and nothing chosen by
looking at a test set: a data set that is not in the list is left as the extractor stored it. Each entry has the canonical task (written into an empty task, or a
task that is a data set name) and the task FAMILIES that are right for the benchmark: a stored task is replaced only when it belongs to a known family that is
not one of them (MRPC + "sentiment analysis"); a task in no known family ("cross-lingual classification", "multiple choice") is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


def key(text: str | None) -> str:
    """Comparison form of a name: lower case, letters and digits only ("SQuAD 2.0" -> "squad20", "MNLI-m" -> "mnlim")."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# task families: a stored task belongs to every family whose pattern it matches
FAMILIES: dict[str, str] = {
    "nli": r"\binferenc|\bentail|\bnli\b|\brte\b",
    "qa": r"question[- ]answer|\bqa\b|reading comprehension|machine reading",
    "paraphrase": r"paraphras",
    "sentiment": r"sentiment|polarity|opinion",
    "acceptability": r"acceptab|grammatical",
    "similarity": r"similarit|relatedness|\bsts\b",
    "ner": r"named[- ]entit|\bner\b|entity recogn",
    "pos": r"part[- ]of[- ]speech|\bpos\b",
    "translation": r"translat|\bmt\b",
    "retrieval": r"retriev|\bmining\b|bitext",
    "summarization": r"summari",
    "lm": r"language model|perplexity|\blm\b|\bmlm\b",
    "commonsense": r"common ?sense|sentence completion|\bcloze\b|plausib|story",
    "math": r"\bmath|arithmetic|quantitative",
    "code": r"\bcode\b|coding|program",
    "toxicity": r"toxic|hate",
    "bias": r"\bbias|stereotyp|demographic",
    "truthfulness": r"truthful",
    "nlu": r"understanding|\bglue\b|benchmark",
    "intent": r"\bintent|slot",
    "coref": r"co-?reference|winograd|pronoun",
    "classification": r"classif|categori",
    "multiple_choice": r"multiple[- ]choice|\bexam",
    "knowledge": r"\bknowledge",
    "chunking": r"\bchunk",
    "parsing": r"\bpars(?:e|er|ing)\b|dependency|constituen",
    "reasoning": r"reasoning",
    "tagging": r"sequence label|tagging|structured prediction|structure prediction",
}
_FAMILY_RE = {f: re.compile(p, re.I) for f, p in FAMILIES.items()}


def families(task: str | None) -> frozenset[str]:
    return frozenset(f for f, rx in _FAMILY_RE.items() if rx.search(task or ""))


@dataclass(frozen=True)
class Benchmark:
    name: str                 # canonical name
    task: str                 # the task it measures, as written into the profile
    families: frozenset[str]  # task families that are right for it
    aliases: tuple[str, ...]  # other spellings found in the papers (compared by key())


def _b(name: str, task: str, fams: str, *aliases: str) -> Benchmark:
    return Benchmark(name, task, frozenset(fams.split()), aliases)


_NLI = "nli classification nlu"
_PARA = "paraphrase similarity classification nlu"
_QA = "qa retrieval"
_CS = "commonsense reasoning multiple_choice qa"
_CS_NOQA = "commonsense reasoning multiple_choice"
BENCHMARKS: tuple[Benchmark, ...] = (
    # natural language inference
    _b("XNLI", "natural language inference", _NLI),
    _b("IndicXNLI", "natural language inference", _NLI, "Indic XNLI"),
    _b("MNLI", "natural language inference", _NLI, "MultiNLI", "MNLI-m", "MNLI-mm", "MNLI-matched", "MNLI-mismatched"),
    _b("RTE", "natural language inference", _NLI),
    _b("WNLI", "natural language inference", _NLI + " coref"),
    _b("QNLI", "natural language inference", _NLI),
    _b("CB", "natural language inference", _NLI, "CommitmentBank"),
    _b("SNLI", "natural language inference", _NLI),
    _b("GLUE diagnostic", "natural language inference", _NLI, "diagnostic set", "diagnostic", "AX", "AXb", "AXg"),
    # paraphrase, similarity, acceptability, sentiment
    _b("MRPC", "paraphrase identification", _PARA),
    _b("QQP", "paraphrase identification", _PARA, "Quora Question Pairs"),
    _b("PAWS-X", "paraphrase identification", _PARA, "PAWS"),
    _b("IndicXParaphrase", "paraphrase identification", _PARA, "Indic XParaphrase", "Indic XPara."),
    _b("STS-B", "semantic textual similarity", "similarity nlu", "STSb", "STS Benchmark", "STS"),
    _b("SICK-R", "semantic textual similarity", "similarity", "SICK"),
    _b("CoLA", "linguistic acceptability", "acceptability classification nlu"),
    _b("SST-2", "sentiment analysis", "sentiment classification nlu", "SST", "SST2", "SST-5"),
    _b("IMDB", "sentiment analysis", "sentiment classification"),
    _b("IndicSentiment", "sentiment analysis", "sentiment classification", "Indic Sentiment"),
    _b("Yelp", "sentiment analysis", "sentiment classification", "Yelp-2", "Yelp-5"),
    _b("Amazon", "sentiment analysis", "sentiment classification", "Amazon-2", "Amazon-5"),
    _b("MR", "sentiment analysis", "sentiment classification"),
    _b("CR", "sentiment analysis", "sentiment classification"),
    _b("TREC", "question classification", "classification"),
    _b("DBpedia", "topic classification", "classification"),
    _b("AG News", "topic classification", "classification", "AG"),
    # question answering and reading comprehension
    _b("SQuAD", "question answering", _QA, "SQuAD1.1", "SQuAD 1.1", "SQuAD2.0", "SQuAD 2.0", "SQuADv2", "SQuAD v1.1", "SQuAD v2.0"),
    _b("XQuAD", "question answering", _QA),
    _b("MLQA", "question answering", _QA),
    _b("TyDi QA GoldP", "question answering", _QA, "TyDiQA-GoldP", "TyDiQA", "TyDi QA"),
    _b("IndicQA", "question answering", _QA, "Indic QA"),
    _b("TriviaQA", "question answering", _QA + " knowledge", "TQA"),
    _b("Natural Questions", "question answering", _QA + " knowledge", "NQ", "NaturalQuestions", "NaturalQS"),
    _b("WebQuestions", "question answering", _QA + " knowledge", "WQ", "WebQS"),
    _b("HotpotQA", "question answering", _QA + " reasoning"),
    _b("NewsQA", "question answering", _QA),
    _b("SearchQA", "question answering", _QA),
    _b("CoQA", "question answering", _QA),
    _b("QuAC", "question answering", _QA),
    _b("DROP", "question answering", _QA + " reasoning math"),
    _b("WikiHop", "question answering", _QA + " reasoning"),
    _b("BoolQ", "question answering", "qa classification"),
    _b("MultiRC", "reading comprehension", _QA),
    _b("ReCoRD", "reading comprehension", _QA + " commonsense"),
    _b("RACE", "reading comprehension", _QA + " multiple_choice", "RACE-middle", "RACE-high", "RACE-m", "RACE-h"),
    _b("MS MARCO", "passage retrieval", "retrieval qa", "MSMARCO", "MS-MARCO"),
    # sequence labelling
    _b("WikiAnn", "named entity recognition", "ner tagging", "PANX", "PAN-X", "WikiANN NER"),
    _b("Naamapadam", "named entity recognition", "ner tagging"),
    _b("CoNLL-2003", "named entity recognition", "ner tagging", "CoNLL 2003", "CoNLL03"),
    _b("CoNLL-2002", "named entity recognition", "ner tagging", "CoNLL 2002"),
    _b("UDPOS", "part-of-speech tagging", "pos tagging", "UD-POS"),
    _b("MASSIVE", "intent classification and slot filling", "intent classification tagging nlu"),
    _b("TACRED", "relation extraction", "classification"),
    # translation and retrieval
    _b("FLORES", "machine translation", "translation", "FLORES-200", "FLORES-101"),
    _b("WMT En-De", "machine translation", "translation", "EnDe", "En-De", "WMT'14 English-German", "WMT14 En-De"),
    _b("WMT En-Fr", "machine translation", "translation", "EnFr", "En-Fr", "WMT'14 English-French", "WMT14 En-Fr", "WMT'14 Fr ↔ En", "WMT'14 En ↔ Fr"),
    _b("WMT En-Ro", "machine translation", "translation", "EnRo", "En-Ro", "WMT'16 Romanian-English", "WMT'16 Ro ↔ En", "WMT'16 En ↔ Ro"),
    _b("WMT De-En", "machine translation", "translation", "WMT'16 German-English", "WMT'16 De ↔ En", "WMT'16 En ↔ De"),
    _b("Tatoeba", "sentence retrieval", "retrieval"),
    _b("BUCC", "bitext mining", "retrieval"),
    # summarization, language modelling
    _b("CNN/Daily Mail", "summarization", "summarization", "CNNDM", "CNN/DM", "CNN-DM", "CNN DailyMail"),
    _b("Penn Treebank", "language modeling", "lm", "PTB"),
    _b("WikiText-103", "language modeling", "lm", "Wikitext-103", "WikiText103"),
    _b("enwik8", "language modeling", "lm"),
    _b("text8", "language modeling", "lm"),
    _b("LAMBADA", "language modeling", "lm commonsense"),
    # benchmark suites
    _b("GLUE", "natural language understanding", "nlu classification nli paraphrase similarity sentiment acceptability"),
    _b("SuperGLUE", "natural language understanding", "nlu classification nli qa coref commonsense", "SGLUE"),
    _b("IndicXTREME", "natural language understanding", "nlu classification nli qa ner", "IndicXTREME benchmark"),
    _b("MMLU", "multitask language understanding", "nlu knowledge multiple_choice qa reasoning", "Massive Multitask Language Understanding"),
    _b("AGIEval", "exam question answering", "multiple_choice qa reasoning knowledge", "AGI Eval", "AGI-Eval"),
    _b("BIG-Bench Hard", "reasoning", "reasoning", "BBH"),
    # commonsense
    _b("HellaSwag", "commonsense reasoning", _CS_NOQA, "Hella-Swag"),
    _b("PIQA", "commonsense reasoning", _CS, "PhysicalQA", "PhysicalQA (PIQA)"),
    _b("SIQA", "commonsense reasoning", _CS, "Social IQA"),
    _b("WinoGrande", "commonsense reasoning", _CS + " coref", "Winogrande (XL)"),
    _b("ARC", "commonsense reasoning", _CS + " knowledge", "ARC-e", "ARC-c", "ARC (Easy)", "ARC (Challenge)", "ARC-Easy", "ARC-Challenge"),
    _b("OpenBookQA", "commonsense reasoning", _CS + " knowledge", "OBQA"),
    _b("COPA", "commonsense reasoning", _CS_NOQA),
    _b("IndicCOPA", "commonsense reasoning", _CS_NOQA, "Indic COPA"),
    _b("StoryCloze", "commonsense reasoning", _CS_NOQA, "Story Cloze"),
    _b("SWAG", "commonsense reasoning", _CS_NOQA + " nli"),
    _b("CommonsenseQA", "commonsense reasoning", _CS, "CSQA"),
    _b("WSC", "coreference resolution", "coref commonsense reasoning", "WSC273", "Winograd Schema Challenge"),
    _b("WinoGender", "coreference resolution", "coref bias"),
    _b("WiC", "word sense disambiguation", "classification"),
    # math and code
    _b("GSM8K", "mathematical reasoning", "math reasoning"),
    _b("MATH", "mathematical reasoning", "math reasoning"),
    _b("ASDiv", "mathematical reasoning", "math reasoning"),
    _b("SVAMP", "mathematical reasoning", "math reasoning"),
    _b("MAWPS", "mathematical reasoning", "math reasoning"),
    _b("AQuA-RAT", "mathematical reasoning", "math reasoning multiple_choice"),
    _b("HumanEval", "code generation", "code", "Human-Eval"),
    _b("MBPP", "code generation", "code"),
    # safety
    _b("TruthfulQA", "truthfulness", "truthfulness qa"),
    _b("ToxiGen", "toxicity evaluation", "toxicity"),
    _b("RealToxicityPrompts", "toxicity evaluation", "toxicity"),
    _b("CrowS-Pairs", "bias evaluation", "bias"),
)


@lru_cache(maxsize=1)
def _index() -> dict[str, Benchmark]:
    out: dict[str, Benchmark] = {}
    for b in BENCHMARKS:
        for name in (b.name, *b.aliases):
            out.setdefault(key(name), b)
    return out


def lookup(name: str | None) -> Benchmark | None:
    """The benchmark a stored name refers to: an exact name or alias ("SQuAD2.0", "EnDe", "MNLI-m"), else None. A name of up to three characters must be
    spelled exactly as listed ("MR" the review data set, not "mr" = Marathi; "CB", "RTE")."""
    k = key(name)
    b = _index().get(k) if k else None
    if b is not None and len(k) <= 3 and (name or "").strip() not in {b.name, *b.aliases}:
        return None
    return b


def lookup_cut(name: str | None) -> Benchmark | None:
    """A name cut at a column border ("SQuA", "ellaSwag", "arco") that is the beginning or the end of exactly one known name (at least 4 letters)."""
    k = key(name)
    if len(k) < 4 or lookup(name):
        return lookup(name)
    hits = {b for n, b in _index().items() if len(n) > len(k) and (n.startswith(k) or n.endswith(k))}
    return next(iter(hits)) if len(hits) == 1 else None


def task_fits(task: str | None, bench: Benchmark) -> bool | None:
    """True / False when the stored task belongs to a known family (inside / outside the benchmark's families), None when it is in no known family."""
    fams = families(task)
    return None if not fams else bool(fams & bench.families)


NOT_TASKS = frozenset("""multitask multitasktraining generaltasks downstreamtasks averagetasks tasks benchmark benchmarks finetuning zeroshottransfer
crosslingualtransfer crosslingualzeroshottransfer crosslingualzeroshot transfer zeroshot fewshot academicbenchmarks standardbenchmarks average mixed various""".split())

# column abbreviations used as data set names in a table header (T5's tables: "EnDe", "EnFr", "EnRo", "CNNDM", "SGLUE"): the benchmark they stand for
ABBREVIATIONS: dict[str, str] = {"ende": "WMT En-De", "enfr": "WMT En-Fr", "enro": "WMT En-Ro", "cnndm": "CNN/Daily Mail", "sglue": "SuperGLUE"}


LANGUAGES = frozenset("""afrikaans albanian amharic arabic armenian assamese azerbaijani basque belarusian bengali bangla bodo bosnian bulgarian burmese catalan
cebuano chinese mandarin cantonese croatian czech danish dogri dutch english esperanto estonian filipino tagalog finnish french galician georgian german greek
gujarati haitian hausa hebrew hindi hungarian icelandic igbo indonesian irish italian japanese javanese kannada kashmiri kazakh khmer konkani korean kurdish
kyrgyz lao latin latvian lithuanian luxembourgish macedonian maithili malagasy malay malayalam maltese manipuri marathi meitei mongolian nepali norwegian odia
oriya pashto persian farsi polish portuguese punjabi romanian russian sanskrit santali scottish serbian sindhi sinhala slovak slovenian somali spanish
sundanese swahili swedish tajik tamil telugu thai tibetan tigrinya turkish turkmen ukrainian urdu uyghur uzbek vietnamese welsh xhosa yiddish yoruba zulu
multilingual indic indian dravidian""".split())

# ISO-639 codes (and the Indic codes of IndicXTREME) with the English name the papers and the extractor use.
CODE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali",
    "bd": "Bodo", "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "ga": "Irish", "gl": "Galician", "gom": "Konkani", "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "ig": "Igbo", "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jv": "Javanese",
    "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "ks": "Kashmiri", "ku": "Kurdish", "ky": "Kyrgyz",
    "la": "Latin", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mai": "Maithili", "mg": "Malagasy", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mni": "Manipuri", "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
    "no": "Norwegian", "or": "Odia", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "ug": "Uyghur", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "xh": "Xhosa",
    "yo": "Yoruba", "zh": "Chinese", "zu": "Zulu",
}
_TR = frozenset(f"{c}-tr" for c in ("bn", "hi", "ml", "mr", "ta", "te", "ur", "gu", "kn"))
_CODES = frozenset(CODE_NAMES) | _TR

# The language columns of a benchmark identify it when a caption is lost (found 11 Oct: mT5's XQuAD / MLQA / TyDi QA tables lost their captions and were
# stored with the language code as the data set and "translation" as the task). The sets are those of the benchmark papers (XQuAD 11 languages,
# MLQA 7, TyDi QA GoldP 9, XNLI 15, PAWS-X 7); a table whose language columns equal one set exactly is that benchmark.
LANGUAGE_SETS: dict[str, frozenset[str]] = {
    "XQuAD": frozenset("en ar de el es hi ru th tr vi zh".split()),
    "MLQA": frozenset("en ar de es hi vi zh".split()),
    "TyDi QA GoldP": frozenset("en ar bn fi id ko ru sw te".split()),
    "XNLI": frozenset("en fr es de el bg ru tr ar vi th zh hi sw ur".split()),
    "PAWS-X": frozenset("en de es fr ja ko zh".split()),
}


def language_code(token: str | None) -> str | None:
    """The code a column token stands for ("ur-tr" -> "ur", "Swahili" -> "sw"); None for anything else."""
    t = (token or "").strip().strip(".").lower()
    if not t:
        return None
    if t in CODE_NAMES:
        return t
    if t in _TR:
        return t[:-3]
    for code, name in CODE_NAMES.items():
        if t == name.lower():
            return code
    return None


def language_name(code: str | None) -> str | None:
    return CODE_NAMES.get(code or "")


def is_language(value: str | None) -> bool:
    """A language name or code ("Kannada", "hi", "transliterated Urdu", "English-German"); a demographic group ("Jewish", "Asian Americans") or a category
    is not."""
    v = (value or "").strip().lower()
    if not v:
        return False
    if v in _CODES:
        return True
    words = re.findall(r"[a-z]+", v)
    return any(w in LANGUAGES for w in words)
