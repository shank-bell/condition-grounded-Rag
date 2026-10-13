"""Profile repair (11 Oct 2026): correct the fields of stored Condition Profiles with code, from the table cell each number came from and from two
small hand-written lists (benchmark -> task, language names). No language model; the profile ids do not change, so every labelled set keeps working.

Why. The hand check of 320 stored profiles (docs/evaluation_ai_annotated.md, 4.1) found that the numbers are right 99.7 % of the time and the names are the
weak part. The errors repeat in a few patterns, each of which the table or a benchmark list settles:

* task: a data set name stored as the task ("EnDe", "GLUE", "diagnostic set"), or a task the data set does not measure (MRPC as sentiment analysis);
* model: a name printed vertically beside a group of rows arrives in pieces ("L" + "lama 1", "LLMA" + "a"); the table parser now joins the pieces and a
  piece that is not written anywhere in the paper's prose is matched to a system name that is (same letters, or the piece is a part of it);
* setting: a split the table never states ("test set" on a table of validation scores), a "single model" the caption gives only for other rows, a metric
  name or "baseline" stored as the setting; a split or a training regime that the block heading does state ("Translate-train-all") replaces a vague one;
* dataset: "Dev Set SST-2" (the split belongs to the setting), a name cut at a column border ("SQuA", "ellaSwag"), a column of languages ("ur-tr", "avg.")
  or a task ("NER") stored as the data set;
* metric: a shot count ("0-shot") or an MMLU category ("STEM") stored as the metric, when the column or the caption names the metric;
* language: a demographic group ("Jewish") stored as the language;
* dataset_version: a version that the column or the caption states ("SQuAD 2.0").

Every rule only acts when the table, the caption or the list states the value; otherwise the field is left as the extractor stored it. `repair` returns the
new profile and the names of the rules that changed it, so each rule can be counted and switched off.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from ..knowledge.benchmarks import (ABBREVIATIONS, FAMILIES as FAMILIES_RX, LANGUAGE_SETS, NOT_TASKS, Benchmark, families, is_language, key,
                                    language_code, language_name, lookup, lookup_cut, task_fits)
from ..knowledge import pwc
from ..pipeline.conditions import _SIZE_LABELS, metric_key, split_dataset
from ..pipeline.grounding import (Grounding, _numbers, _size_like, block_tags, candidates, caption_split, ground, model_fits)
from ..schemas import ConditionProfile
from .dataset_context import group_key, label_kind
from .model_context import is_pure_citation, model_key, strip_citation
from .tables import parse_table, table_cells

RULES = ("task", "model_name", "model_own", "setting_split", "setting_system", "setting_noise", "setting_regime", "dataset_split",
         "dataset_cut", "dataset_column", "dataset_signature", "metric", "language", "language_column", "language_average",
         "model_size", "dataset_version", "dataset_abbrev", "task_filler", "setting_fragment", "dataset_caption", "dataset_paper_abbrev", "metric_paper_abbrev", "task_paper",
         "model_heading", "language_order", "dataset_context", "model_citation", "model_context", "metric_paper_measure", "metric_difference", "metric_cut",
         "setting_ablation", "value_lost_decimal", "dataset_pair", "dataset_conll", "metric_split_word", "dataset_row_label", "metric_benchmark",
         "metric_avg_column", "dataset_spelling")


_ABLATION = re.compile(r"\bablation|\beffects? of\b|\bdifferent (?:masking|pre-?training|mixing|training) (?:strateg|objective|data)", re.I)
_CONFIG_LABEL = re.compile(r"%|=|^\s*\d")


@dataclass
class PaperContext:
    paper_id: str
    prose: str
    names: dict[str, str] = field(default_factory=dict)
    own: str | None = None
    main: str | None = None
    row_labels: set[str] = field(default_factory=set)
    main_rows: set[str] = field(default_factory=set)
    abbrevs: dict[str, str] = field(default_factory=dict)
    measures: dict[str, str] = field(default_factory=dict)
    datasets: dict[str, str] = field(default_factory=dict)
    task: str | None = None

    def written(self, name: str | None) -> bool:
        """The name occurs in the prose as a word (case-insensitive)."""
        n = (name or "").strip()
        return len(n) >= 2 and re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", self.prose, re.I) is not None


_HEADING_MODELS = frozenset({"stateoftheart", "sota", "previousstateoftheart", "previoussota", "previousbest", "ours", "ourmodel", "ourmethod", "proposed",
                             "proposedmodel", "proposedmethod", "baselines", "publishedresults"})
_SIZE_KEYS = frozenset(_SIZE_LABELS)
_SIZE_KEPT = re.compile(r"(?<![\w.])[≈~]?\d+(?:\.\d+)?\s?[KMBT]\b(?!\s*(?:steps?|epochs?|tokens?|examples?))|\bL\s*=\s*\d|\d+\s*layers?\b", re.I)
_NAME_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*(?:\s(?:\d+(?:\.\d+)?|[A-Z][a-z]+))?\b")


_TASKISH = frozenset("chunk chunking parse parsing dep dependency dependen translate translation tagging ner pos intent slot segmentation overall average".split())


def _dataset_like(name: str) -> bool:
    """A stored data set value that is a name and not a task / column word ("Chunking", "Dep.", "Translation", "avg"): known benchmarks always are."""
    k = key(name)
    if len(k) < 3 or k in _NOT_A_NAME or k in NOT_TASKS or k in _TASKISH or _LANG_COLUMN.match(name.strip()):
        return False
    if lookup(name) is not None or pwc.find(name) is not None:
        return True
    return not families(name)


_PAPER_TASKS = {"ner": "named entity recognition", "pos": "part-of-speech tagging", "translation": "machine translation", "summarization": "summarization",
                "qa": "question answering", "nli": "natural language inference", "paraphrase": "paraphrase identification", "sentiment": "sentiment analysis",
                "similarity": "semantic textual similarity", "lm": "language modeling", "intent": "intent detection and slot filling",
                "coref": "coreference resolution", "toxicity": "toxicity evaluation", "code": "code generation", "retrieval": "retrieval"}

_DEFINITION = [
    re.compile(r"([A-Z][\w\-.’']*(?:\s+[\w\-.’']+){0,5})\s*\(\s*(?:the\s+)?([A-Za-z][A-Za-z0-9\-]{1,11})\s*\)"),
    re.compile(r"\b([A-Za-z][A-Za-z0-9\-]{1,11})\s+(?:is|are|stands?|denotes?)\s+(?:short\s+for|an?\s+abbreviation\s+(?:of|for)|for)\s+([A-Z][\w\- ]{2,40})"),
    re.compile(r"\b([a-z][a-z0-9\-]{1,6}|[A-Z][A-Za-z0-9\-]{1,6})\s*=\s*([A-Z][A-Za-z\- ]{3,40})(?=[,;.)]|\s*$)"),
]


def _plausible_abbrev(abbr: str, full: str) -> bool:
    """The abbreviation's letters occur in the full name in order (WN16 / WNUT-16, scc / Spearman correlation coefficient, MTR / METEOR)."""
    a = re.sub(r"[^a-z]", "", abbr.lower())
    f = re.sub(r"[^a-z]", "", full.lower())
    if len(a) < 2 or len(a) >= len(f) or abbr.lower() == full.lower():
        return False
    it = iter(f)
    return all(ch in it for ch in a) and f[0] == a[0]


def paper_abbreviations(prose: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for rx in _DEFINITION:
        for m in rx.finditer(prose or ""):
            if rx is _DEFINITION[0]:
                full, abbr = re.split(r"(?<=[.!?])\s+", m.group(1).strip())[-1], m.group(2)
                while True:
                    shorter = re.sub(r"^(?:the|a|an|and|of|in|on|for|with|our|we|use|used|using|called|named|dataset|datasets|benchmark|corpus|is|are)\s+", "", full, flags=re.I)
                    if shorter == full:
                        break
                    full = shorter
            else:
                abbr, full = m.group(1), m.group(2).strip()
            full = re.sub(r"\s+", " ", re.sub(r"([\-/])\s+", r"\1", full)).strip()
            if _plausible_abbrev(abbr, full):
                out.setdefault(abbr.lower(), full)
    return out


_MCC_WORD = re.compile(r"\bmatthews?\b|\bmatthew['’]s\b|\bmcc\b", re.I)
_PEARSON_WORD = re.compile(r"\bpearson", re.I)
_SPEARMAN_WORD = re.compile(r"\bspearman", re.I)


def paper_measures(prose: str) -> dict[str, str]:
    """The measure the paper itself names for the two GLUE tasks that are not scored by accuracy: CoLA by Matthews correlation, STS-B by a correlation
    (Pearson or Spearman, named only when the paper names exactly one of them). The extractor reads such a column as "accuracy" when the header says nothing
    more than the task's name (found 11 Oct night with the Papers with Code leaderboards: 101 CoLA and 78 STS-B results)."""
    out: dict[str, str] = {}
    if _MCC_WORD.search(prose):
        out["cola"] = "Matthews correlation"
    pearson, spearman = bool(_PEARSON_WORD.search(prose)), bool(_SPEARMAN_WORD.search(prose))
    if pearson or spearman:
        out["stsb"] = "Pearson correlation" if pearson and not spearman else ("Spearman correlation" if spearman and not pearson else "correlation")
    return out


_BENCHMARK_MEASURE = {"cola": "Matthews correlation", "stsb": "correlation"}

_KNOWN_METRICS = ("Exact Match", "Spearman correlation", "Pearson correlation", "Matthews correlation", "accuracy", "precision", "recall", "perplexity")


def complete_metric(metric: str | None) -> str | None:
    """A metric label that is a header word cut at a column border ("Exac", "Match" of "Exact Match"; "Spea", "Pear"): the beginning (4 letters or more) or the end (5 letters or
    more) of exactly one well-known metric name, and not a name itself. None when nothing is cut or the completion is ambiguous."""
    m = key(re.sub(r"[.\s]+$", "", metric or ""))
    if len(m) < 4 or m in {key(k) for k in _KNOWN_METRICS} | {"acc", "em", "f1", "exact"}:
        return None
    heads = [k for k in _KNOWN_METRICS if key(k).startswith(m)]
    tails = [k for k in _KNOWN_METRICS if len(m) >= 5 and key(k).endswith(m)]
    found = heads or tails
    return found[0] if len(found) == 1 else None


_N = r"[+\-−]?\s*(\d+(?:\.\d+)?)"
_DIFF_CUE = r"(?:outperform\w*|exceed\w*|surpass\w*|beat\w*|improv\w+|better|higher|lower|worse|gain\w*|boost\w*|increas\w+|drop\w*|decreas\w+|reduc\w+|above|below|ahead|margin)"
_DIFFERENCE_PATTERNS = (
    re.compile(rf"\b{_DIFF_CUE}\b[^.;]{{0,80}}?\bby\s*(?:\.\.\.\s*)?(?:a\s+margin\s+of\s+)?{_N}", re.I),
    re.compile(rf"\(\s*[+\-−]\s*(\d+(?:\.\d+)?)"),
    re.compile(rf"(?<![\w.])\+\s*(\d+(?:\.\d+)?)\s*(?:%|points?|pts)"),
    re.compile(rf"\b(?:gain|improvement|increase|boost|margin|difference|drop|decrease|reduction)\s+of\s+{_N}", re.I),
    re.compile(rf"{_N}\s*(?:absolute\s+)?(?:points?|pts)\s+(?:higher|lower|better|worse|above|below|over|more|less)\b", re.I),
    re.compile(rf"\b{_DIFF_CUE}\b(?:[^.;]|(?<=\d)\.(?=\d)){{0,120}}?\band\s+(?:by\s+)?{_N}\s*(?:%|points?|pts|F1|EM|BLEU|accuracy)", re.I),
)


def sentence_of(evidence: str | None, chunk_text: str | None) -> str | None:
    """The whole sentence of the chunk that contains the stored evidence snippet (the extractor keeps a fragment: "and 1.86 F1 when trained on each language"),
    so that a cue earlier in the sentence is seen. None when the snippet is not found."""
    if not evidence or not chunk_text:
        return None
    flat = re.sub(r"\s+", " ", chunk_text)
    snippet = re.sub(r"\s+", " ", evidence).strip()
    i = flat.find(snippet)
    if len(snippet) < 12 or i < 0:
        return None
    start = max(flat.rfind(". ", 0, i), flat.rfind("\n", 0, i)) + 1
    end = flat.find(". ", i + len(snippet))
    return flat[start:(end + 1 if end >= 0 else len(flat))].strip()


def states_a_difference(evidence: str | None, value: float | None) -> bool:
    """The sentence a profile was read from presents its number as a DIFFERENCE between two systems ("outperforms mBERT by 13.0% F1", "RACE (+8.4%)", "a gain of
    2.0%"), not as a result: the number that follows the cue is the stored value. A result read from a table never matches (it has no such sentence)."""
    if not evidence or value is None:
        return False
    for rx in _DIFFERENCE_PATTERNS:
        for m in rx.finditer(evidence):
            try:
                if abs(float(m.group(m.lastindex or 1)) - abs(value)) < 1e-6:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def paper_context(paper_id: str, chunk_texts: list[str], model_names: list[str], title: str = "",
                  datasets: list[str] | None = None) -> PaperContext:
    """The prose of a paper and the system names it writes: stored model names and row labels that occur at least twice in the prose."""
    prose = "\n".join(ln for t in chunk_texts for ln in t.split("\n") if not ln.lstrip().startswith("|"))
    ctx = PaperContext(paper_id, prose)
    ctx.abbrevs = paper_abbreviations(prose)
    ctx.measures = paper_measures(prose)
    from collections import Counter as _C
    counts = _C(d for d in (datasets or []) if d and len(key(d)) >= 3)
    ctx.datasets = {key(d): d for d, n in counts.items() if n >= 2 and _dataset_like(d)}
    head = " ".join([title or ""] + chunk_texts[:3])[:6000].lower()
    scores = sorted(((len(re.findall(FAMILIES_RX[f], head)), f) for f in _PAPER_TASKS), reverse=True)
    if scores and scores[0][0] >= 2 and (len(scores) < 2 or scores[0][0] >= 1.5 * scores[1][0]):
        ctx.task = _PAPER_TASKS[scores[0][1]]
    in_title = {m for m in set(model_names) if len(key(m)) >= 3 and re.search(rf"(?<![A-Za-z0-9]){re.escape(m)}(?![A-Za-z0-9])", title or "")}
    in_title = {m for m in in_title if not any(o != m and key(m) in key(o) for o in in_title)}
    ctx.own = in_title.pop() if len(in_title) == 1 else None
    ctx.main = ctx.own
    if ctx.main is None:
        from collections import Counter as _Counter
        firsts = _Counter()
        for n in set(model_names):
            m = re.match(r"^([A-Za-z][A-Za-z0-9]*(?:-\d+(?:\.\d+)?(?![A-Za-z0-9]))?)(?=[-\s_]|$)", (n or "").strip())
            if m and (n or "").strip() != m.group(1):
                firsts[m.group(1)] += 1
        for word, count in firsts.most_common(4):
            if count >= 3 and len(key(word)) >= 2 and key(word) not in _NOT_A_NAME and re.search(r"[A-Z0-9]", word) and not is_language(word) and lookup(word) is None                     and len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", prose)) >= 10:
                ctx.main = word
                break
    labels = set(model_names)
    for t in chunk_texts:
        table = parse_table(t)
        if table:
            rows = {c.row_label for c in table_cells(table)}
            labels |= rows
            ctx.row_labels |= {key(r) for r in rows if r}
            if not _ABLATION.search(table.caption or "") and not re.search(r"\bvariants?\b", table.caption or "", re.I) \
                    and not _FACTOR_HEADER.search((table.header[0] if table.header else "") or ""):
                ctx.main_rows |= {key(r) for r in rows if r}
    for name in labels:
        n = (name or "").strip()
        if len(key(n)) < 3 or key(n) in _NOT_A_NAME or not re.search(r"[A-Z]", n):
            continue
        hits = len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", prose))
        if hits >= 2:
            ctx.names.setdefault(key(n), n)
    return ctx


_NOT_A_NAME = frozenset({"all", "average", "avg", "mean", "total", "overall", "human", "humans", "ours", "baseline", "baselines", "random", "majority",
                         "model", "models", "pretrained", "finetuned", "chat", "base", "large", "small", "test", "dev", "sota"})


def _subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def fix_name(label: str | None, ctx: PaperContext) -> str | None:
    """The system name a garbled label stands for: a name the prose writes with the same letters ("aLLMA" -> LLaMA, "coFaln" -> Falcon) or of which the
    label is a part ("lama 2hat" -> Llama 2-Chat, "Fln" -> Falcon). None when the label is written in the prose itself or nothing fits uniquely."""
    if not label or ctx.written(label):
        return None
    k = key(label)
    if len(k) < 3:
        return None
    anagram = {n for kk, n in ctx.names.items() if len(k) >= 4 and sorted(kk) == sorted(k) and kk != k}
    if len(anagram) == 1:
        return anagram.pop()
    if k in ctx.row_labels:
        return None
    if not (label.strip()[:1].islower() or len(k) <= 4):
        return None
    part = {n for kk, n in ctx.names.items() if len(kk) > len(k) and len(kk) - len(k) <= 4 and _subsequence(k, kk) and kk[0] == k[0]}
    part = {n for n in part if not any(o != n and key(n).startswith(key(o)) for o in part)} or part
    if len(part) == 1:
        return part.pop()
    return None


_PARAMS = re.compile(r"^(\d+(?:\.\d+)?)\s?([KMBT])$", re.I)


def _run_label(chunk_text: str, g: Grounding) -> str | None:
    """All the pieces of a group label printed vertically beside a run of rows (sizes 7B, 13B, 34B, 70B), in row order: "L -C" + "lama 2hat"."""
    t = parse_table(chunk_text)
    if t is None:
        return None
    rows: dict[int, tuple[str, float | None]] = {}
    for c in table_cells(t):
        m = _PARAMS.match(c.text.strip())
        label, size = rows.get(c.row, (c.row_label, None))
        rows[c.row] = (c.row_label, size if size is not None else (float(m.group(1)) * {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[m.group(2).lower()] if m else None))
    target = next((r for r, (lab, _) in rows.items() if lab == g.row_label), None)
    if target is None or rows[target][1] is None:
        return None
    order = sorted(rows)
    start = end = order.index(target)
    while start > 0 and rows[order[start - 1]][1] is not None and rows[order[start - 1]][1] < rows[order[start]][1]:
        start -= 1
    while end + 1 < len(order) and rows[order[end + 1]][1] is not None and rows[order[end + 1]][1] > rows[order[end]][1]:
        end += 1
    pieces: list[str] = []
    for r in order[start:end + 1]:
        if rows[r][0] and rows[r][0] not in pieces:
            pieces.append(rows[r][0])
    return "".join(pieces) if len(pieces) > 1 else None


def _row_size(chunk_text: str, g: Grounding) -> str | None:
    t = parse_table(chunk_text)
    if t is None:
        return None
    for c in table_cells(t):
        if c.row_label == g.row_label and c.block == g.block and _PARAMS.match(c.text.strip()):
            return c.text.strip().upper().replace(" ", "")
    return None


_METRIC_WORD = re.compile(r"\b(EM|exact match|F1|acc(?:uracy)?|BLEU(?:-\d)?|ROUGE[- ]?(?:1|2|L)|R-L|pass@\d+|MCC|matthews|spearman|pearson|perplexity|PPL|"
                          r"precision|recall|top-\d+|P@\d+|BPC)\b", re.I)
_NOT_A_METRIC = re.compile(r"^(?:\d+|zero|one|few)[- ]?shots?$|^(?:avg\.?|average|overall|stem|humanities|social sciences|other)$", re.I)
_SPLIT_WORDS = re.compile(r"(?<!translate[ -])\b(?:dev(?:elopment)?|test|validation|val)(?:[- ]?set)?\b", re.I)
_ANY_SPLIT = re.compile(r"(?<!translate[ -])\b(?:dev|development|validation|valid|val|test)(?:[- ]?sets?)?\b", re.I)
_ONLY_TEST = re.compile(r"\b(?:only|just)\s+(?:create|creat\w+|provid\w+|releas\w+|have|has|use|using|available)?\s*(?:a\s+|the\s+)?test[- ]sets?\b", re.I)
_TABLE_NO = re.compile(r"^\s*Table\s+(\d+(?:\.\d+)?)", re.I)
_SYSTEM_WORDS = re.compile(r"\bsingle[- ]?model\b|\bensemble\b", re.I)
_GENERIC_SETTING = {"fine-tuned", "finetuned", "fine-tuning", "fine tuned", "multilingual", "monolingual", "pretrained", "pre-trained", "pre-training"}
_REGIME = re.compile(r"translate[- ]train(?:[- ]all)?|translate[- ]test|cross[- ]lingual transfer|zero[- ]shot|few[- ]shot|\d+[- ]shot|single[- ]task|"
                     r"multi[- ]task|feature[- ]based", re.I)
_DATASET_SPLIT = re.compile(r"^(?:(?P<s1>dev(?:elopment)?|test|validation)\s+set\s+(?P<n1>.+)|(?P<n2>.+?)\s+(?P<s2>dev(?:elopment)?|test|validation)(?:\s+set)?)$", re.I)
_LANG_COLUMN = re.compile(r"^(?:[a-z]{2,3}(?:-tr)?|avg\.?|average|all|overall)$", re.I)


def _split_label(word: str) -> str:
    w = word.lower()
    return "test set" if w.startswith("test") else "dev set"


def _table_splits(g: Grounding, caption: str) -> set[str]:
    """dev / test that the cell's block heading, its column name or (where neither does) the caption states."""
    tags = set(block_tags(g.block)) | set(block_tags(g.column))
    tags &= {"dev", "test"}
    return tags or set(caption_split(caption))


def _squash(text: str | None) -> str:
    """Lower-case letters only and without "i": a block heading reaches the parser with stray "i" glyphs at column borders ("Translate-tra iin (models
    finei-tune on Engli ish training di ata ..."), so words are compared with their i's removed on both sides."""
    return re.sub(r"[^a-z]", "", (text or "").lower()).replace("i", "")


_REGIMES = tuple((out, tuple(_squash(w) for w in words)) for out, words in (
    ("translate-train-all", ("translate-train-all", "fine-tune multilingual model on all training sets", "translate train all")),
    ("translate-test", ("translate-test", "translate test")),
    ("translate-train", ("translate-train", "translate train")),
    ("in-language multitask (fine-tuned)", ("in-language multitask", "gold data in all target languages")),
    ("zero-shot cross-lingual transfer", ("zero-shot", "cross-lingual transfer", "zero shot")),
))
_AVERAGED = re.compile(r"averag\w*\s+(?:\w+\s+){0,3}?(?:across|over)\s+(?:all\s+|the\s+|\w+\s+)?languages|average\s+(?:performance|scores?|results?)\s+(?:across|over)", re.I)


def _regime(text: str | None) -> str | None:
    s = _squash(text)
    if not s:
        return None
    for out, words in _REGIMES:
        if any(w in s for w in words):
            return out
    return None


def _column_code(column: str | None) -> str | None:
    """The language code of a column name: its first token that is a code or a language name ("ar ot trans i" -> ar, "XQuAD en" -> en)."""
    for tok in re.split(r"[\s/,;()]+", column or ""):
        c = language_code(tok)
        if c:
            return c
    return None


_AVG_COLUMN = re.compile(r"^\s*(?:avg|average|mean|overall)\b", re.I)


_PERCENT_METRIC = re.compile(r"accuracy|\bacc\b|f1|exact match|\bem\b|precision|recall|rouge|bleu|\bauc\b|top-?\d|pass@|correlation|\bmcc\b|truthful", re.I)
_NOT_PERCENT = re.compile(r"perplex|loss|error|\bppl\b|param|size|flops|latency|seconds|speed", re.I)
_FACTOR_HEADER = re.compile(r"strateg|objective|architecture|number of|hidden size|batch size|corruption|context length|sequence length|span length|masking|data pct|"
                            r"dropout|learning rate|variant|configuration|\bsim\b|fine-tuning method|\bablation\b|\bsteps\b", re.I)
_BASELINE_ROW = re.compile(r"^[⋆*★\s]*(?:baseline|full\b|ours?\b|proposed)", re.I)


_SYSTEM_LIKE = re.compile(r"BERT|GPT|T5\b|Net\b|ELMo|ELECTRA|XLM|LSTM|CNN|Transformer|Longformer|Llama|LLaMA|MPT\b|Falcon|Chinchilla|PaLM|\bDPR|\bRAG|BART|MASS\b|CoVe|GloVe|BiDAF|"
                          r"DrQA|QANet|Human|SOTA|state[- ]of[- ]the[- ]art", re.I)


def _is_system_row(label: str, ctx: PaperContext) -> bool:
    """The row of an ablation table that is a system, not a configuration. A label that is also a row of one of the paper's results tables is a system; a label
    that opens with "+", "-", "w/o" or "No" is a modification of one; else it is a system when it looks like a system name (the teacher models "BERT-large" of
    MobileBERT's ablation). (12 Oct: a first version asked whether the label was a name the prose writes, which skipped "LTR & No NSP", "AKT" and "+ BiLSTM":
    the paper writes the names of its variants too.)"""
    if key(label) in ctx.main_rows:
        return True
    if re.match(r"^\s*(?:[+\-–−]|w/o\b|without\b|no\b)", label or "", re.I):
        return False
    return bool(_SYSTEM_LIKE.search(label or ""))


def _name_like_label(label: str) -> bool:
    """A row label that names a system (letters, maybe digits) and not a configuration ("E = 64", "K = 2^18", "80%: ... = 10%")."""
    return not re.search(r"[=%^]|^\s*\d", label or "")


def _language_table(table) -> bool:
    """Most columns of the table are languages: at least three distinct language codes / names and at least 40 % of the columns."""
    cols = {c.column for c in table_cells(table)}
    langs = {_column_code(c) for c in cols} - {None}
    return len(langs) >= 3 and len(langs) >= 0.4 * max(1, len(cols))


def _prose_supports(split: str, caption: str, ctx: PaperContext) -> bool:
    """The paper's prose backs a stored split that the table does not state: a sentence names this table and the split, or the prose speaks of one kind
    of split only (MuRIL: "results on the test sets", never a dev set)."""
    m = _TABLE_NO.match(caption or "")
    words = ("dev", "development", "validation") if split == "dev" else ("test",)
    if m:
        for sent in re.split(r"(?<=[.!?])\s+", ctx.prose):
            if re.search(rf"\bTable\s+{re.escape(m.group(1))}\b", sent) and any(re.search(rf"\b{w}\b", sent, re.I) for w in words):
                return True
    if split == "test" and _ONLY_TEST.search(ctx.prose):
        return True
    dev = len(re.findall(r"\b(?:dev|development|validation)[- ]sets?\b", ctx.prose, re.I))
    test = len(re.findall(r"(?<!translate[ -])\btest[- ]sets?\b", ctx.prose, re.I))
    return (test > 0 and dev == 0) if split == "test" else (dev > 0 and test == 0)


def _caption_benchmark(caption: str) -> Benchmark | None:
    hits = {lookup(w) for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+(?:\s\d\.\d)?", caption or "")} - {None}
    return hits.pop() if len(hits) == 1 else None


def _caption_dataset(caption: str, ctx: PaperContext) -> str | None:
    """The one data set a caption names: first among the data sets this paper's own profiles name, then the benchmark list, then the archive list."""
    cap = key(caption)
    own = {n for k, n in ctx.datasets.items() if len(k) >= 3 and k in cap}
    own = {n for n in own if not any(o != n and key(n) in key(o) for o in own)}
    if len(own) == 1:
        return own.pop()
    b = _caption_benchmark(caption)
    if b is not None and not own:
        return b.name
    found = pwc.in_text(caption)
    return found[0] if len(found) == 1 and not own and b is None else None


def repair(p: ConditionProfile, chunk_text: str, ctx: PaperContext, off: frozenset[str] = frozenset()) -> tuple[ConditionProfile, list[str]]:
    """The profile with every field the table or the benchmark list contradicts corrected, and the rules that fired."""
    upd: dict = {}
    fired: list[str] = []

    def put(rule: str, **fields) -> None:
        if rule in off:
            return
        changed = {k: v for k, v in fields.items() if (upd.get(k, getattr(p, k)) or None) != (v or None)}
        if changed:
            upd.update(changed)
            fired.append(rule)

    table = parse_table(chunk_text or "")
    caption = table.caption if table else ""
    g = ground(p, chunk_text or "") if table else Grounding()
    ablation_table = bool(table and _ABLATION.search(caption))
    fit = [c for c in candidates(p, chunk_text)
           if ablation_table or model_fits(p.model, c.row_label) is not False or model_fits(p.model, c.column) is True] if table else []
    stored_code = language_code(p.language)
    narrowed = [c for c in fit if stored_code and _column_code(c.column) == stored_code]
    cells = narrowed or fit

    ds = p.dataset
    if ds and key(ds) not in ABBREVIATIONS and lookup(ds) is None and ds.strip().lower() in ctx.abbrevs:
        full = ctx.abbrevs[ds.strip().lower()]
        if len(full) <= 40:
            put("dataset_paper_abbrev", dataset=full)
            ds = full
    if ds and key(ds) in ABBREVIATIONS:
        put("dataset_abbrev", dataset=ABBREVIATIONS[key(ds)])
        ds = ABBREVIATIONS[key(ds)]
    m = _DATASET_SPLIT.match((ds or "").strip())
    if m and lookup(m.group("n1") or m.group("n2")):
        name, split = m.group("n1") or m.group("n2"), m.group("s1") or m.group("s2")
        setting = p.setting if p.setting and _SPLIT_WORDS.search(p.setting) else " ".join(x for x in (p.setting, _split_label(split)) if x)
        put("dataset_split", dataset=name, setting=setting)
        ds = name
    if ds and not lookup(ds) and lookup_cut(ds):
        put("dataset_cut", dataset=lookup_cut(ds).name)
        ds = lookup_cut(ds).name
    if ds and not lookup(ds) and (is_language(ds) or _LANG_COLUMN.match(ds.strip())) and len(ds.strip()) <= 7 and _caption_benchmark(caption):
        put("dataset_column", dataset=_caption_benchmark(caption).name)
        ds = _caption_benchmark(caption).name
    if not ds and table:
        named = _caption_dataset(caption, ctx)
        if named:
            put("dataset_caption", dataset=named)
            ds = named
    if cells and ds and g and g.row_label and key(ds) == key(g.row_label) and lookup(ds) is None and _caption_benchmark(caption) is not None:
        put("dataset_row_label", dataset=_caption_benchmark(caption).name)
        ds = _caption_benchmark(caption).name
    if cells and ds and g and lookup(ds) is None and len(key(ds)) >= 5:
        for label in (g.row_label, g.column):
            named = lookup(re.sub(r"\s+", " ", label or "").strip())
            if named is not None and key(named.name) != key(ds) and difflib.SequenceMatcher(None, key(ds), key(label)).ratio() >= 0.85:
                put("dataset_spelling", dataset=named.name)
                ds = named.name
                break
    if cells and ds and p.value is not None and key(ds) in ("mnli", "mnlim", "mnlimm"):
        targets = set()
        for c in cells:
            nums = re.findall(r"\d+(?:\.\d+)?", c.text or "")
            if re.search(r"m\s*/\s*mm", c.column or "", re.I) and len(nums) == 2 and abs(float(nums[0]) - float(nums[1])) > 1e-6:
                targets.add("MNLI-mm" if abs(p.value - float(nums[1])) < 1e-6 else ("MNLI-m" if abs(p.value - float(nums[0])) < 1e-6 else None))
            else:
                targets.add(None)
        if len(targets) == 1 and next(iter(targets)) and key(next(iter(targets))) != key(ds):
            put("dataset_pair", dataset=next(iter(targets)))
            ds = next(iter(targets))
    if cells and ds and key(ds) in ("conll2002", "conll2003"):
        codes = {_column_code(c.column) for c in cells}
        if codes and codes <= {"en", "de"} and key(ds) != "conll2003":
            put("dataset_conll", dataset="CoNLL-2003")
            ds = "CoNLL-2003"
        elif codes and codes <= {"nl", "es"} and key(ds) != "conll2002":
            put("dataset_conll", dataset="CoNLL-2002")
            ds = "CoNLL-2002"

    task = upd.get("task", p.task)
    bench = lookup(ds)
    said = (caption or "").lower()
    if bench is not None:
        stated = bool(task) and any(re.search(FAMILIES_RX[f], said) for f in families(task))
        filler = bool(task) and key(task) in NOT_TASKS
        if not task or lookup(task) or filler or (task_fits(task, bench) is False and not stated):
            put("task_filler" if filler else "task", task=bench.task)
    elif task and lookup(task):
        put("task", task=lookup(task).task)
    elif task and key(task) in NOT_TASKS:
        put("task_filler", task=None)
    if not (upd.get("task", p.task)) and ctx.task:
        put("task_paper", task=ctx.task)

    cited = strip_citation(p.model)
    if cited:
        put("model_citation", model=cited)
    if p.model and key(p.model) in _HEADING_MODELS and cells:
        rows = {c.row_label for c in cells if c.row_label}
        if len(rows) == 1:
            label = next(iter(rows))
            if label and _name_like_label(label) and key(label) not in _HEADING_MODELS and key(label) not in _NOT_A_NAME:
                put("model_heading", model=label)
    glued = re.compile(r"\s*(?<![A-Za-z])[a-z]{15,}(?![A-Za-z])\s*(?:Avg\.?\s*)?")
    if p.model and glued.search(p.model):
        cleaned = glued.sub(" ", p.model).strip()
        if len(cleaned) >= 2 and cleaned != p.model:
            put("model_name", model=cleaned)
    paren = re.search(r"\(([^)]*[=%^][^)]*)\)\s*$", p.model or "")
    if p.model and paren:
        base = re.sub(r"\s*\([^)]*\)\s*$", "", p.model).strip()
        if base and (ctx.written(base) or key(base) in ctx.row_labels):
            conf = re.sub(r"\s+", " ", paren.group(1)).strip()
            put("model_name", model=base, setting=(f"{p.setting}; {conf}" if p.setting and conf.lower() not in p.setting.lower() else (p.setting or conf))[:120])
    if p.model and not ctx.written(p.model) and "model_name" not in fired:
        fragment = len(key(p.model)) <= 4 or (p.model[:1].islower() and (" " in p.model or bool(re.search(r"\d", p.model))))
        name = fix_name(p.model, ctx)
        if not name and fragment and g and key(p.model) not in ctx.row_labels:
            label = g.row_label
            if label and key(label) in ctx.names and key(label) not in key(p.model) and key(label) not in _SIZE_KEYS \
                    and (key(label).startswith(key(p.model)) or key(label).endswith(key(p.model))):
                name = label
            if not name:
                name = fix_name(_run_label(chunk_text, g), ctx)
        if name and key(name) != key(p.model) and key(p.model) not in key(name) or (name and key(name) != key(p.model) and fragment):
            put("model_name", model=name, model_size=p.model_size or (_row_size(chunk_text, g) if g else None))
    if p.model and ctx.own and "model_name" not in fired and _CONFIG_LABEL.search(p.model) \
            and (_ABLATION.search(caption or "") or (cells and not _name_like_label(p.model))):
        variant = "ablation: " + re.sub(r"\s+", " ", p.model).strip()
        carried = f"{p.setting}; {variant}" if p.setting and variant.lower() not in p.setting.lower() else (p.setting or variant)
        put("model_own", model=ctx.own, setting=carried[:120])

    setting = upd.get("setting", p.setting) or ""
    if cells:
        per_cell = [(set(block_tags(c.block)) | set(block_tags(c.column))) & {"dev", "test"} or set(caption_split(caption)) for c in cells]
        agreed = per_cell[0] if all(s == per_cell[0] for s in per_cell) else None
        stored = {"dev" if w.lower().startswith(("dev", "val")) else "test" for w in _SPLIT_WORDS.findall(setting)}
        mentioned = _ANY_SPLIT.search(" ".join([caption, " ".join(table.header)] + [f"{c.block} {c.column}" for c in cells]))
        if stored and agreed is not None and not agreed and not mentioned and not any(_prose_supports(s, caption, ctx) for s in stored):
            setting = re.sub(r"\s{2,}", " ", _SPLIT_WORDS.sub("", setting)).strip(" ,;")
            put("setting_split", setting=setting or None)
        elif agreed is not None and len(agreed) == 1 and stored != agreed:
            rest = re.sub(r"\s{2,}", " ", _SPLIT_WORDS.sub("", setting)).strip(" ,;")
            setting = " ".join(x for x in (rest, _split_label(next(iter(agreed)))) if x)
            put("setting_split", setting=setting)
        if _SYSTEM_WORDS.search(setting):
            says = " ".join([caption] + [f"{c.block} {c.column} {c.row_label}" for c in cells]).lower().replace("-", " ")
            if not re.search(r"single model|single models|singlemodel|ensemble", says):
                setting = re.sub(r"\s{2,}", " ", _SYSTEM_WORDS.sub("", setting)).strip(" ,;")
                put("setting_system", setting=setting or None)
        if re.search(r"single[- ]?model", setting, re.I) and all(re.search(r"ensemble", c.block or "", re.I) for c in cells) \
                and not any(re.search(r"single[- ]?model", f"{c.block} {c.column}", re.I) for c in cells):
            setting = re.sub(r"\s{2,}", " ", re.sub(r"single[- ]?task single[- ]?model|single[- ]?model", "ensemble", setting, flags=re.I)).strip(" ,;")
            put("setting_system", setting=setting)
        if re.search(r"\btrain(?:ing)?\b", setting, re.I) and re.search(r"\bdev", setting, re.I) and all(re.search(r"\bdev", c.column or "", re.I) for c in cells):
            setting = re.sub(r"\s{2,}", " ", re.sub(r"\btrain(?:ing)?\b", "", setting, flags=re.I)).strip(" ,;")
            put("setting_split", setting=setting)
        mixed = any(re.search(r"fine[- ]?tun", c.column or "", re.I) and re.search(r"feature", c.column or "", re.I) for c in cells)
        columns = {c.column for c in cells}
        regime = _REGIME.search(next(iter(columns))) if len(columns) == 1 else None
        if regime is None and not mixed and len({c.block for c in cells}) == 1:
            regime = _REGIME.search(cells[0].block or "")
        if regime and (not setting or setting.strip().lower() in _GENERIC_SETTING):
            setting = regime.group(0)
            put("setting_regime", setting=setting)
        named_set = {_regime(c.block) for c in cells} if not mixed else set()
        named = next(iter(named_set)) if len(named_set) == 1 else None
        if named and (not setting or setting.strip().lower() in _GENERIC_SETTING or "fine" in setting.lower()
                      or (_regime(setting) is not None and _regime(setting) != named)):
            setting = named
            put("setting_regime", setting=setting)
    if cells and setting and setting == setting.lower() and len(setting.split()) <= 2 and len(setting) <= 10 \
            and not re.search(r"\d|shot|dev|test|train|tun|ensemble|single|base", setting, re.I):
        named_set = {_regime(c.block) for c in cells}
        named = next(iter(named_set)) if len(named_set) == 1 else None
        if named:
            setting = named
            put("setting_fragment", setting=setting)
        elif len(setting) <= 8 and not re.search(r"[aeiou]{1}.*[aeiou]", setting.lower()):
            setting = ""
            put("setting_fragment", setting=None)
    if setting and _METRIC_WORD.fullmatch(setting.strip()):
        put("setting_noise", setting=None)
    elif setting and re.search(r"\bbaselines?\b", setting, re.I):
        setting = re.sub(r"\s{2,}", " ", re.sub(r"\bbaselines?\b", "", setting, flags=re.I)).strip(" ,;")
        put("setting_noise", setting=setting or None)

    main = ctx.main
    if table and cells and main and not {"model_name", "model_own"} & set(fired):
        rows = {c.row_label for c in cells if c.row_label}
        label = next(iter(rows)) if len(rows) == 1 else ""
        head = (table.header[0] if table.header else "") or ""
        factor = bool(_FACTOR_HEADER.search(head))
        if label and key(label) not in ctx.main_rows and (factor or (ablation_table and not _is_system_row(label, ctx))):
            plain = re.sub(r"\s+", " ", re.sub(r"^[⋆*★\s]+", "", label)).strip()
            if not re.search(r"[A-Za-z]{2,}", plain):
                plain = f"{head.strip().lower()} {plain}" if re.fullmatch(r"[A-Za-z][A-Za-z ]{3,40}", head.strip()) else ""
            model_is_label = bool(p.model) and key(p.model) == key(label)
            if _BASELINE_ROW.match(label):
                if model_is_label or key(p.model) in ("baseline", ""):
                    put("model_own", model=main)
            elif key(plain) and (model_is_label or (p.model and key(main) in key(p.model) and key(plain) not in key(p.model))):
                current = upd.get("setting", p.setting) or ""
                variant = f"ablation: {plain}"
                merged = current if key(plain) in key(current) else (f"{current}; {variant}" if current else variant)
                fields = {"setting": merged[:120]}
                if model_is_label:
                    fields["model"] = main
                put("setting_ablation", **fields)

    if table and (not ds or not lookup(ds)) and not caption.strip():
        cols = {_column_code(c.column) for c in table_cells(table)} - {None}
        sig = [name for name, langs in LANGUAGE_SETS.items() if cols == langs]
        if len(sig) == 1 and (not ds or is_language(ds) or _LANG_COLUMN.match((ds or "").strip())):
            put("dataset_signature", dataset=sig[0])
            ds = sig[0]
            bench = lookup(ds)
            if bench is not None and (not task or task_fits(task, bench) is not True):
                put("task", task=bench.task)

    if p.model_size and not _size_like(p.model_size) and not _SIZE_KEPT.search(p.model_size):
        sized = re.search(r"\b(tiny|mini|small|base|medium|large|xlarge|xxlarge|xl|xxl)\s*$", p.model_size.strip(), re.I)
        put("model_size", model_size=sized.group(1).lower() if sized and len(p.model_size.split()) <= 4 else None)

    if p.metric and p.metric.strip().lower() in ctx.abbrevs and len(ctx.abbrevs[p.metric.strip().lower()]) <= 40 \
            and metric_key(ctx.abbrevs[p.metric.strip().lower()]) != metric_key(p.metric):
        put("metric_paper_abbrev", metric=ctx.abbrevs[p.metric.strip().lower()])
    elif p.metric:
        m2 = re.match(r"^([A-Za-z]{2,5})[\-\s]?(\d|[lL]\b)$", p.metric.strip())
        if m2 and m2.group(1).lower() in ctx.abbrevs and key(ctx.abbrevs[m2.group(1).lower()]) in ("rouge", "bleu", "meteor"):
            put("metric_paper_abbrev", metric=f"{ctx.abbrevs[m2.group(1).lower()]}-{m2.group(2).upper()}")
    measure_bench = lookup(upd.get("dataset", p.dataset))
    measure = (ctx.measures.get(key(measure_bench.name)) or _BENCHMARK_MEASURE.get(key(measure_bench.name))) if measure_bench else None
    if measure and key(upd.get("metric", p.metric)) in ("accuracy", "acc") and not re.search(r"\bacc", g.column or "", re.I) \
            and (key(measure_bench.name) != "cola" or (p.value is not None and p.value <= 80)):
        put("metric_paper_measure", metric=measure)
    if measure_bench is not None and measure_bench.name.startswith("WMT") and key(upd.get("metric", p.metric)) in ("score", "accuracy", "acc", "avg", "translation"):
        put("metric_benchmark", metric="BLEU")
    if (p.metric and key(p.metric) in ("avg", "average", "mean") and g is not None and g.column and not _AVG_COLUMN.match(g.column) and table and ds
            and any(_AVG_COLUMN.match(h or "") for h in table.header) and key(ds) and key(ds) in key(g.column)):
        put("metric_avg_column", metric="score")
    if p.metric and re.fullmatch(r"(?:m|mm|m\s?acc|mm\s?acc)\.?", p.metric.strip(), re.I):
        put("metric_split_word", metric="accuracy")
    cut_metric = complete_metric(upd.get("metric", p.metric))
    if cut_metric:
        put("metric_cut", metric=cut_metric)
    metric_now = upd.get("metric", p.metric) or ""
    if table and cells and p.value is not None and 100 < p.value < 1000 and float(p.value).is_integer() and metric_now \
            and (_PERCENT_METRIC.search(metric_now) or re.search(r"ccuracy", metric_now)) and not _NOT_PERCENT.search(metric_now):
        if all(re.fullmatch(r"\d{3}", c.text.strip()) for c in cells) and sum(1 for c in table_cells(table) if re.search(r"\d\.\d", c.text)) >= 5:
            put("value_lost_decimal", value=round(p.value / 10, 1))
    if p.metric and not cells and (states_a_difference(p.evidence, p.value) or states_a_difference(sentence_of(p.evidence, chunk_text), p.value)) \
            and not key(upd.get("metric", p.metric)).startswith("difference"):
        put("metric_difference", metric=f"difference in {upd.get('metric', p.metric)}")
    if p.metric and _NOT_A_METRIC.match(p.metric.strip()):
        found = _METRIC_WORD.findall(g.column) if g else []
        if not found:
            found = list(dict.fromkeys(w.lower() for w in _METRIC_WORD.findall(caption)))
        if len(found) == 1:
            put("metric", metric=found[0])

    if p.language and not is_language(p.language):
        put("language", language=None)
    elif p.language and table and fit and _language_table(table):
        lang_columns = {_column_code(c.column) for c in table_cells(table)} - {None}
        stored_norm = key(p.language)
        same = lambda c: (stored_code and _column_code(c.column) == stored_code) or key(language_name(_column_code(c.column))) == stored_norm
        if not any(same(c) for c in fit):
            codes = {_column_code(c.column) for c in fit}
            if len(codes) == 1 and None not in codes:
                put("language_column", language=language_name(next(iter(codes))))
            elif stored_code and len(lang_columns) >= 2 and all(_AVG_COLUMN.match(c.column or "") for c in fit):
                put("language_average", language=None)
            elif stored_code and _AVERAGED.search(caption) and None in codes and len(codes) == 1:
                put("language_average", language=None)
    elif p.language and table and cells and _AVERAGED.search(caption) and not _language_table(table):
        put("language_average", language=None)

    if ds and not p.dataset_version and g:
        base = re.escape(ds)
        version = rf"{base}\s*v?(\d{{1,2}}(?:\.\d+)+)(?![\w.])"
        found = [m.group(1) for text in (g.column, g.block) for m in [re.search(version, text or "", re.I)] if m]
        if not found:
            found = list(dict.fromkeys(re.findall(version, caption or "", re.I)))
        if found and len(set(found)) == 1:
            put("dataset_version", dataset_version=found[0])

    return (p.model_copy(update=upd) if upd else p), fired


def _order_languages(profiles: list[ConditionProfile], chunks: dict[str, str]) -> dict[int, str | None]:
    """The languages of the results of a stacked table of languages, read from the ORDER of the extraction (11 Oct; found on IndicCOPA, Table 17 of
    2212.05409, where mBERT's 51.8 stands in the pa, ta and te columns of the second half).

    The extractor read the second half with the first half's header, so a result carries the language of the column at the same position in the first
    half ("Dogri" = gom for the 51.8 of pa). The value rule cannot choose between three cells with one number. But the extractor wrote the results of a row
    in the order of the cells (the profile ids of one model in one chunk count along the rows), so when a system's results, in that order, have exactly
    the numbers of the cells of the rows that carry its name, the k-th result IS the k-th cell, and the column of that cell names the language. Used
    only for a table of languages, for a model whose row labels name it, and only when the whole sequence agrees number by number (otherwise nothing
    is changed). Returns {profile index: language or None (the column is an average)} for the profiles whose stored language differs."""
    def seq(p: ConditionProfile) -> int:
        n = p.profile_id.rpartition("#")[2]
        return int(n) if n.isdigit() else 10 ** 9

    by_chunk: dict[str, list[int]] = {}
    for i, p in enumerate(profiles):
        if p.value is not None and p.model and p.chunk_id in chunks:
            by_chunk.setdefault(p.chunk_id, []).append(i)
    out: dict[int, str | None] = {}
    for chunk_id, idx in by_chunk.items():
        table = parse_table(chunks[chunk_id])
        if table is None or not _language_table(table):
            continue
        cells = table_cells(table)
        models: dict[str, list[int]] = {}
        for i in idx:
            models.setdefault(profiles[i].model.lower(), []).append(i)
        for members in models.values():
            members.sort(key=lambda i: seq(profiles[i]))
            rows = [c for c in cells if model_fits(profiles[members[0]].model, c.row_label) is True]
            if len(rows) != len(members):
                continue
            if not all(any(abs(n - profiles[i].value) < 1e-6 for n in _numbers(c.text)) for i, c in zip(members, rows)):
                continue
            for i, c in zip(members, rows):
                col = _column_code(c.column)
                if profiles[i].language and language_code(profiles[i].language) != col:
                    out[i] = language_name(col) if col else None
    return out


def repair_paper(profiles: list[ConditionProfile], chunks: dict[str, str], off: frozenset[str] = frozenset(),
                 title: str = "", resolutions: dict[str, dict] | None = None) -> tuple[list[ConditionProfile], Counter]:
    """Repair every profile of one paper; `chunks` maps chunk id -> text. Returns the profiles (repaired or not, same order) and the rule counts.
    `resolutions` are the paper's data set decisions of ingestion/dataset_context.py (group key -> {dataset, evidence, ...}); without them the repair is code only."""
    if not profiles:
        return [], Counter()
    ctx = paper_context(profiles[0].paper_id, [chunks[k] for k in sorted(chunks)], [p.model for p in profiles if p.model], title,
                        [p.dataset for p in profiles if p.dataset])
    out, counts = [], Counter()
    for p in profiles:
        q, fired = repair(p, chunks.get(p.chunk_id, ""), ctx, off)
        out.append(q)
        counts.update(fired)
    if "language_order" not in off:
        for i, language in _order_languages(out, chunks).items():
            if (out[i].language or None) != (language or None):
                out[i] = out[i].model_copy(update={"language": language})
                counts.update(["language_order"])
    if resolutions and "dataset_context" not in off:
        for i, q in enumerate(out):
            if label_kind(q.dataset):
                found = resolutions.get(group_key(q.chunk_id, q.dataset))
                if found and found.get("dataset") and found["dataset"] != q.dataset:
                    out[i] = q.model_copy(update={"dataset": found["dataset"]})
                    counts.update(["dataset_context"])
    if resolutions and "model_context" not in off:
        for i, q in enumerate(out):
            if is_pure_citation(q.model):
                found = resolutions.get(model_key(q.model))
                if found and found.get("model") and found["model"] != q.model:
                    out[i] = q.model_copy(update={"model": found["model"]})
                    counts.update(["model_context"])
    return out, counts


__all__ = ["RULES", "PaperContext", "paper_context", "fix_name", "repair", "repair_paper", "split_dataset"]
