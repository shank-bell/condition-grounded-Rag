"""Comparing experimental conditions: shared by stage 6 (question vs evidence) and stage 7 (paper vs paper).

Everything is deterministic string logic on the fields of a Condition Profile, so a verdict can always be
traced back to the exact values that produced it.
"""
from __future__ import annotations

import re

from ..schemas import ConditionProfile

_DECOR = ("base", "large", "small", "tiny", "mini", "medium", "xlarge", "xxlarge", "xl", "xxl", "huge", "cased",
          "uncased", "single", "ensemble", "distilled", "multilingual", "finetuned", "pretrained")
_SIZE_LABELS = {"base", "large", "small", "tiny", "mini", "medium", "xlarge", "xxlarge", "xl", "xxl", "huge"}
_LANG_ALIASES = {"en": "english", "eng": "english", "hi": "hindi", "kn": "kannada", "ta": "tamil", "te": "telugu",
                 "de": "german", "fr": "french", "es": "spanish", "zh": "chinese", "ru": "russian", "ar": "arabic",
                 "sw": "swahili", "ja": "japanese", "ko": "korean", "mr": "marathi", "bn": "bengali"}
_TASK_ALIASES = {"nli": "naturallanguageinference", "qa": "questionanswering", "ner": "namedentityrecognition",
                 "mt": "machinetranslation", "mrc": "readingcomprehension", "sts": "semantictextualsimilarity",
                 "pos": "partofspeechtagging", "rte": "recognizingtextualentailment",
                 # the form a question uses ("translating German into English") is the task the tables call "translation" (found 7 Oct)
                 "translating": "translation", "translate": "translation", "summarizing": "summarization", "summarising": "summarization",
                 "classifying": "classification"}
TASK_WORDS = frozenset(_TASK_ALIASES) | frozenset(_TASK_ALIASES.values())     # "nli", "naturallanguageinference", ...
_METRIC_ALIASES = {"acc": "accuracy", "exactmatch": "em", "f1score": "f1", "fscore": "f1", "rougel": "rougel"}
_METRIC_NOISE = ("score", "dev", "test", "val", "validation", "avg", "average")


def norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


NULLISH = {"not specified", "unspecified", "n/a", "na", "none", "null", "unknown", "not stated", "not applicable",
           "not reported", "not mentioned", "not available", "-", "--", "—", ""}


def is_nullish(text: str | None) -> bool:
    """'not specified', 'N/A' and friends: what a language model writes instead of leaving a field empty."""
    return text is None or text.strip().strip(".").lower() in NULLISH


_SPLIT_LABEL = {"dev": "dev set", "development": "dev set", "val": "dev set", "validation": "dev set",
                "test": "test set", "train": "train set"}
_DATASET_PARTS = re.compile(
    r"^(?P<name>.+?)\s+v?(?P<ver>\d+(?:\.\d+)+)(?:\s+(?P<split>dev(?:elopment)?|test|train|validation|val))?(?:\s+set)?$", re.I)


def split_dataset(name: str) -> tuple[str, str | None, str | None]:
    """'SQuAD 2.0 test' -> ('SQuAD', '2.0', 'test set'); names without a dotted version ('SST-2') are unchanged."""
    m = _DATASET_PARTS.match(name.strip())
    if not m:
        return name, None, None
    split = m["split"].lower() if m["split"] else None
    return m["name"], m["ver"], _SPLIT_LABEL.get(split) if split else None


def split_version(name: str) -> tuple[str, str | None]:
    """'SQuAD 2.0' -> ('SQuAD', '2.0')."""
    base, version, _ = split_dataset(name)
    return base, version


def norm_version(text: str | None) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"^v(?:ersion)?\s*", "", t)
    return t


def parse_params(text: str | None) -> float | None:
    """Parameter count in millions from '340M', '1.5B', '110 million'; None if it is a label like 'large'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|m|million|b|billion)\b", (text or "").lower())
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    return val * {"k": 0.001, "m": 1, "million": 1, "b": 1000, "billion": 1000}[unit]


def _strip_decor(n: str) -> tuple[str, str | None]:
    """(family, size label) for a normalised model name: 'bertlarge' -> ('bert', 'large')."""
    size = None
    changed = True
    while changed:
        changed = False
        for d in sorted(_DECOR, key=len, reverse=True):
            if n.endswith(d) and len(n) > len(d):
                if d in _SIZE_LABELS and size is None:
                    size = d
                n = n[: -len(d)]
                changed = True
                break
    return n, size


def model_family(model: str | None) -> str:
    return _strip_decor(norm(model))[0]


# A table row can label a known system with WHOSE or WHICH version it is: "Our BERT", "Google BERT", "Baseline (mT5-large)", "Avg. GloVe embeddings".
# 149 of the 10,275 stored results are labelled so, mostly BERT (70) and mT5-large (40): the system people ask about by its plain name. The word must be
# followed by a separator, so "GoogLeNet" is not "Google" + "Net". Used by `values_match` (coverage) only: `model_family` - and with it which two
# results Stage 7 pairs - is left as it was.
_PREFIX_DECOR = re.compile(r"^\s*(?:google|ours?|published|baseline|vanilla|original|proposed|avg\.?|average)[\s:(\-]+", re.I)


# A parameter count written after the name ("LLaMA 65B", "T5-11B") is the model's SIZE, not part of its name: a question about "LLaMA 65B" is satisfied
# by a row recorded as "LLaMA" or "LLaMA-65B", not by "LLaMA 7B". Found on 7 Oct on the gold questions: "LLaMA 65B" got a false scope warning although the
# store holds LLaMA 65B results. Like the prefix above this is used by `values_match` (coverage) only; `model_family` is unchanged.
_NUMERIC_SIZE = re.compile(r"^(?P<name>.*?[A-Za-z].*?)[\s\-_]*(?P<size>\d+(?:\.\d+)?\s?[MBK])\s*$", re.I)


def _model_core(text: str | None) -> tuple[str, str | None]:
    """(family, size) of a model name without a leading whose-version word ('Our BERT' -> 'bert') and without a trailing parameter count
    ('LLaMA 65B' -> ('llama', '65b')); a size label such as 'large' is the size when there is one."""
    stripped = _PREFIX_DECOR.sub("", text or "", count=1)
    base = stripped if len(norm(stripped)) >= 2 else (text or "")
    count = None
    m = _NUMERIC_SIZE.match(base.strip())
    if m and len(norm(m["name"])) >= 2:
        base, count = m["name"], re.sub(r"\s", "", m["size"]).lower()      # keep the dot: 1.3B is not 13B
    family, label = _strip_decor(norm(base))
    return family, label or count


def model_size_label(p: ConditionProfile) -> str | None:
    """Size recorded for the profile: the explicit field, else a label carried by the model name (BERT-large)."""
    if p.model_size:
        return p.model_size
    return _strip_decor(norm(p.model))[1]


def metric_key(metric: str | None) -> str:
    n = norm(metric)
    for noise in _METRIC_NOISE:
        n = n.replace(noise, "")
    return _METRIC_ALIASES.get(n, n)


_PAIR_SPLIT = re.compile(r"\s*(?:-to-|\bto\b|→|->|–|—|/|-)\s*", re.I)


def _language_pair(text: str) -> list[str]:
    """['English', 'German'] for 'English-to-German' / 'en-de' / 'English → German'; [] when the text is not a pair."""
    parts = [p for p in _PAIR_SPLIT.split(text.strip()) if p]
    return parts if len(parts) == 2 else []


def values_match(field: str, requested: str, observed: str) -> bool:
    """Does an observed value satisfy a requested condition? 'BERT' is satisfied by 'BERT-large', not by 'RoBERTa'."""
    if not requested or not observed:
        return False
    if field == "model":
        rf, rs = _model_core(requested)
        of, os_ = _model_core(observed)
        return rf == of and (rs is None or os_ is None or rs == os_)
    if field == "dataset":
        r, o = norm(requested), norm(observed)
        return len(r) >= 3 and len(o) >= 3 and (r == o or o.startswith(r) or r.startswith(o))
    if field == "dataset_version":
        return norm_version(requested) == norm_version(observed) or _num_equal(requested, observed)
    if field == "language":
        r, o = norm(requested), norm(observed)
        if _LANG_ALIASES.get(r, r) == _LANG_ALIASES.get(o, o):
            return True
        # a translation direction ("English-to-German") names two languages; a source that records either one is on topic
        return any(_LANG_ALIASES.get(norm(p), norm(p)) == _LANG_ALIASES.get(o, o) for p in _language_pair(requested))
    if field == "model_size":
        pr, po = parse_params(requested), parse_params(observed)
        if pr is not None and po is not None:
            return abs(pr - po) <= 0.05 * max(pr, po)
        return norm(requested) == norm(observed)
    if field == "setting":
        tr, to = setting_tags(requested), _recorded_regime_tags(observed)
        if tr and to:
            return tr <= to
    r, o = norm(requested), norm(observed)          # task, setting: free text, so containment either way
    if field == "task":
        r, o = _TASK_ALIASES.get(r, r), _TASK_ALIASES.get(o, o)
    if r and r == o:
        return True                                 # the same words, however short ("NER" = "NER")
    return len(r) >= 4 and len(o) >= 4 and (r in o or o in r)


# A question names a benchmark SUITE ("GLUE") while the tables record its member tasks (CoLA, MRPC, ...). Used for COVERAGE only
# (Stage 6, profile lookups): comparing a GLUE score with an MNLI accuracy (Stage 7) must keep using values_match.
DATASET_SUITES = {
    "glue": ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli", "diagnostic"),
    "superglue": ("boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"),
    "xtreme": ("xnli", "pawsx", "panx", "wikiann", "udpos", "xquad", "mlqa", "tydiqa", "bucc", "tatoeba"),
    "indicxtreme": ("indicxnli", "indiccopa", "indicsentiment", "indicxpara", "indicqa", "indicner", "indicxquad", "flores"),
}


def dataset_in_suite(requested: str, observed: str) -> bool:
    """'CoLA' / 'MNLI-m' are tasks of the GLUE suite; 'XNLI' is not part of 'IndicXTREME' (nor 'IndicXNLI' of 'XTREME')."""
    members = DATASET_SUITES.get(norm(requested))
    o = norm(observed)
    return bool(members) and o != norm(requested) and any(o == m or (len(m) >= 4 and o.startswith(m)) for m in members)


def covers(field: str, requested: str, observed: str) -> bool:
    """Does a recorded value cover a requested condition? `values_match`, plus: a suite named in the question is covered by
    results on its member tasks."""
    return values_match(field, requested, observed) or (field == "dataset" and dataset_in_suite(requested, observed))


def _num_equal(a: str, b: str) -> bool:
    try:
        return float(norm_version(a)) == float(norm_version(b))
    except ValueError:
        return False


def observed_values(profile: ConditionProfile, field: str) -> list[str]:
    """Values a profile records for a condition field (model gets its size label as a model_size value too)."""
    if field == "model_size":
        size = model_size_label(profile)
        return [size] if size and not is_nullish(size) else []
    value = getattr(profile, field, None)
    return [value] if value and not is_nullish(value) else []


def _recorded_regime_tags(text: str | None) -> frozenset[str]:
    """`setting_tags` of a RECORDED setting for matching a requested one: "cross-lingual transfer" (fine-tune on English, test on the others) is zero-shot
    transfer unless the setting says translate-train / translate-test."""
    tags = set(setting_tags(text))
    if re.search(r"cross[- ]?lingual transfer", (text or "").lower()) and not tags & {"translate-train", "translate-test"}:
        tags.add("zero-shot")
    return frozenset(tags)


_SETTING_TAGS = (
    ("zero-shot", r"zero[- ]?shot"), ("few-shot", r"few[- ]?shot"), ("fine-tuned", r"fine[- ]?tun"),
    ("translate-train", r"translate[- ]?train"), ("translate-test", r"translate[- ]?test"),
    ("dev", r"\b(?:dev|development|validation|val)\b"), ("test", r"\btest\b"), ("train", r"\btrain(?:ing)?\b"),
    ("single", r"\bsingle\b"), ("ensemble", r"\bensemble"), ("in-domain", r"in[- ]?domain"),
    ("out-of-domain", r"out[- ]?of[- ]?domain"), ("feature-based", r"feature[- ]?based"),
)


def setting_tags(text: str | None) -> frozenset[str]:
    """Evaluation-setting categories found in a free-text setting ('zero-shot transfer' -> {'zero-shot'}).

    Free-text settings such as 'each N' carry no tag: they are too noisy to name as the explanation of a difference."""
    t = (text or "").lower()
    return frozenset(tag for tag, pattern in _SETTING_TAGS if re.search(pattern, t))


NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
N_SHOT = re.compile(r"(?<![\w-])(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")[- ]?shots?(?![\w-])", re.I)      # "5-shot", "5 shot", "five-shot", "1-shot"
_REGIME_TAGS = frozenset({"zero-shot", "few-shot", "fine-tuned", "translate-train", "translate-test"})


def is_regime_setting(text: str | None) -> bool:
    """A setting that says HOW the model was used (zero-shot, 5-shot, fine-tuned, translate-train) and not which split was scored (dev / test)."""
    return bool(text) and (bool(setting_tags(text) & _REGIME_TAGS) or bool(N_SHOT.search(text)))


def differing_conditions(a: ConditionProfile, b: ConditionProfile) -> list[str]:
    """Condition fields that are recorded on both profiles with different values (dataset and metric are
    the shared subject of the comparison, so they are not listed). The free-text task is not compared: labels
    such as 'cross-lingual classification' vs 'sentence pair question answering' are extraction noise, not conditions."""
    diffs: list[str] = []
    for f in ("dataset_version", "model_size", "language"):
        va, vb = observed_values(a, f), observed_values(b, f)
        if va and vb and not values_match(f, va[0], vb[0]):
            diffs.append(f)
    ta, tb = setting_tags(a.setting), setting_tags(b.setting)
    if ta and tb and ta != tb:
        diffs.append("setting")
    return diffs


# Conditions that change a result. If exactly one side records one of these we cannot say the two results were
# obtained under the same conditions, so the pair must not be called a genuine contradiction.
_RESULT_CHANGING = ("dataset_version", "model_size", "setting")


def _recorded(p: ConditionProfile, field: str) -> bool:
    return bool(setting_tags(p.setting)) if field == "setting" else bool(observed_values(p, field))


def unrecorded_conditions(a: ConditionProfile, b: ConditionProfile) -> list[str]:
    """Result-changing conditions recorded on one profile but missing on the other."""
    return [f for f in _RESULT_CHANGING if _recorded(a, f) != _recorded(b, f)]


def claim_text(p: ConditionProfile) -> str:
    """One natural-language sentence for a profile, used as NLI input and as evidence in prompts."""
    parts = [f"{p.model or 'The system'} obtains {p.value:g} {p.metric or 'score'}"]
    if p.dataset:
        parts.append(f"on {p.dataset}{(' ' + p.dataset_version) if p.dataset_version else ''}")
    extra = [x for x in (p.setting, p.language, p.model_size) if x]
    return " ".join(parts) + (f" ({', '.join(extra)})" if extra else "") + "."
