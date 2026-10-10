"""Table grounding (10 Oct 2026): check a stored result against the table cell it came from, and read the conditions that the table itself states.

Why. Stage 7 compares the stored cards (Condition Profiles) of two results. On 150 labelled pairs its mistakes are mostly the cards, not the rules:
a block heading ("Ensembles on test (from leaderboard ...)") that never reached the rows under it, a size or setting recorded for one paper only, a RoBERTa
row stored as "BERT", STS-B stored as WNLI. The table is in the chunk the card points to, so the evidence can be read again by code, deterministically:

* the cell: the numbers of the card's chunk table are searched for the card's value, and the cell that fits the card's model (row label) and dataset (column
  name) best is taken;
* the block heading above the cell and the caption give the evaluation split (dev / test / leaderboard) and the system kind (single / ensemble);
* the row label, the block heading or the caption give the model size (BERT-large, RoBERTaBASE, "large models", "24-layer");
* a card whose model or dataset does not fit its own cell is SUSPECT (the number belongs to another row or column).

Nothing here guesses: a condition is only filled when the table states it, and a card is only suspect when the table contradicts it. Everything is a pure
function of the stored profile and the chunk text. Used by Stage 7 (`[contradiction] table_grounding`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from ..ingestion.tables import Cell, ParsedTable, parse_table, table_cells
from ..schemas import ConditionProfile
from .conditions import _SIZE_LABELS, _strip_decor, model_family, norm

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]|[†‡*§¶♠♣♦♥]+")
_SPLIT_TAGS = ("dev", "test", "train")
_SYSTEM_TAGS = ("single", "ensemble")


@dataclass
class Grounding:
    found: bool = False
    row_label: str = ""
    block: str = ""
    column: str = ""
    setting_tags: frozenset[str] = frozenset()      # what the block heading (else the caption) says: dev / test / single / ensemble
    size: str | None = None                          # size label stated by the row label, the block heading or the caption
    suspect: list[str] = field(default_factory=list)  # why the card does not fit its cell ("dataset", "model"); empty = fits or unknown

    def __bool__(self) -> bool:
        return self.found


# --------------------------------------------------------------------------------------------------------------------------------------------------
# what a heading or a caption says
# --------------------------------------------------------------------------------------------------------------------------------------------------

def block_tags(text: str) -> frozenset[str]:
    """Split and system kind named by a block heading. Headings arrive cut at column borders and glued ("Single-task singlemodels ondev",
    "Ensembles on test (from leaderboardas of Sept. 16, 2 019)"), so the words are also searched in the text without spaces."""
    t = (text or "").lower()
    s = re.sub(r"[^a-z0-9]", "", t)
    tags: set[str] = set()
    if re.search(r"\b(?:dev|development|validation)\b", t) or re.search(r"(?:on|the)(?:dev|development|validation)(?:set|data)?", s) or s.endswith("dev"):
        tags.add("dev")
    if re.search(r"\btest\b", t) or re.search(r"(?:on|the)test(?:set|data)?", s) or "leaderboard" in s:
        tags.add("test")
    if "ensemble" in s:
        tags.add("ensemble")
    elif re.search(r"single(?:model|task)", s):
        tags.add("single")
    return frozenset(tags)


_CAPTION_DEV = re.compile(r"\b(?:dev(?:elopment)?|validation)(?:[- ]set| data| sets)\b|\bon the (?:glue )?dev\b", re.I)
_CAPTION_TEST = re.compile(r"\btest[- ]set\b|\btest[- ]sets\b|\bon the (?:glue )?test\b|\btest results\b|\bresults on the test\b", re.I)


def caption_split(caption: str) -> frozenset[str]:
    """dev / test when the caption names exactly one of them; both or neither give nothing (a table can mix them)."""
    dev, test = bool(_CAPTION_DEV.search(caption or "")), bool(_CAPTION_TEST.search(caption or ""))
    return frozenset({"dev"}) if dev and not test else frozenset({"test"}) if test and not dev else frozenset()


_CAPTION_FEATURES = re.compile(r"logistic regression|as features|feature[- ]based|\bSentEval\b|linear (?:probe|probing|classifier)|frozen (?:encoder|embedding|representation)", re.I)


def caption_protocol(caption: str) -> frozenset[str]:
    """feature-based when the caption says the numbers come from a classifier trained on frozen embeddings (SBERT's SentEval table: "training a logistic
    regression classifier using the sentence embeddings as features"); a fine-tuned model is a different protocol and not the same quantity."""
    return frozenset({"feature-based"}) if _CAPTION_FEATURES.search(caption or "") else frozenset()


_CAPTION_SIZE = re.compile(r"\b(large|base|small|tiny|mini)\s+(?:models?|architectures?|configurations?)\b|\bthe\s+\w+\s+(large|base|small|tiny)\s+model\b", re.I)
_LAYERS = re.compile(r"\b(\d{1,2})[- ]layer\b", re.I)


def caption_size(caption: str) -> str | None:
    """A size the caption states for every row: "results for large models", "ablation study of the DeBERTa base model", "a 24-layer architecture"."""
    m = _CAPTION_SIZE.search(caption or "")
    if m:
        return (m.group(1) or m.group(2)).lower()
    m = _LAYERS.search(caption or "")
    if m:
        return {"24": "large", "12": "base"}.get(m.group(1))
    return None


def label_size(label: str) -> str | None:
    """Size label carried by a row label or a heading: "BERT-large", "RoBERTa_base", "BERTLARGE", "BERTTINY" -> large / base / tiny."""
    return _strip_decor(norm(_CITATION.sub("", label or "")))[1]


_SUBJECT = re.compile(r"\b(?:results?|performance|comparison|ablation(?: study)?|study|evaluation)\s+(?:of|for|on|with)\s+(?:the\s+)?([A-Z][A-Za-z0-9_.\-]*)")


def caption_subject(caption: str) -> str | None:
    """The system a caption is about ("Development set results for RoBERTa ...", "Ablation study of the DeBERTa base model"), when it says so."""
    m = _SUBJECT.search(caption or "")
    return m.group(1) if m and not re.fullmatch(r"GLUE|SQuAD|MNLI|SuperGLUE|RACE|SWAG|Table", m.group(1)) else None


# --------------------------------------------------------------------------------------------------------------------------------------------------
# finding the cell
# --------------------------------------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cells(chunk_text: str) -> tuple[ParsedTable | None, tuple[Cell, ...]]:
    t = parse_table(chunk_text)
    return (t, tuple(table_cells(t))) if t else (None, ())


def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUMBER.findall(text or "")]


_GENERIC = frozenset({"top", "bottom", "ours", "our", "baseline", "baselines", "human", "humans", "average", "avg", "mean", "total", "single", "ensemble", "ensembles",
                      "test", "dev", "development", "validation", "train", "model", "models", "system", "systems", "method", "methods", "task", "tasks", "row", "rows",
                      "overall", "published", "unpublished", "previous", "state", "art", "sota", "best", "worst", "random", "majority"})


def _name_like(label: str) -> bool:
    """A row label that names a system (BERT, RoBERTa-large, XLNet-Base (K = 7), mT5) and not a variant of the table's own system ("+ additional data
    (3.2)", "Equal mixing", "3") nor a generic word ("Top", "Bottom", "Ours", "Human")."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-_.]*", _CITATION.sub("", label or "")) if w.lower().strip(".") not in _GENERIC]
    return any(sum(c.isupper() for c in w) >= 2 or (any(c.isdigit() for c in w) and any(c.isalpha() for c in w)) or (w[:1].isupper() and len(w) >= 3) for w in words) \
        and not (label or "").lstrip().startswith("+")


def _model_fits(model: str | None, label: str) -> bool | None:
    """True / False when the row label names a system, None when it does not (no verdict)."""
    if not model or not _name_like(label):
        return None
    fm, nl = model_family(model), norm(_CITATION.sub("", label))
    if not fm or len(fm) < 2:
        return None
    return fm == model_family(label) or fm in nl or (len(nl) >= 3 and nl in norm(model))


# Columns can be tasks (GLUE: "MNLI QNLI QQP ... WNLI"), languages ("en sw ur-tr"), metrics ("EM F1") or subsets ("Middle High"); only a column that is itself a
# benchmark task says which data set a number belongs to, so only two such names that differ are a contradiction (found 10 Oct: STS-B stored as WNLI).
_TASKS = ("cola", "sst", "sst2", "mrpc", "sts", "stsb", "qqp", "mnli", "qnli", "rte", "wnli", "boolq", "cb", "copa", "multirc", "record", "wic", "wsc")


def _task_of(name: str | None) -> str | None:
    n = norm(name)
    for task in sorted(_TASKS, key=len, reverse=True):
        if n == task or (n.startswith(task) and n[len(task):] in ("", "m", "mm", "mmm", "b", "acc", "f1", "2", "2acc")):
            return {"sst2": "sst", "stsb": "sts"}.get(task, task)
    return None


def _column_fits(dataset: str | None, column: str) -> bool | None:
    """True / False when both the stored data set and the column are benchmark tasks (equal or different); None otherwise (the column says nothing)."""
    d, c = _task_of(dataset), _task_of(column)
    return None if d is None or c is None else d == c


def candidates(p: ConditionProfile, chunk_text: str) -> list[Cell]:
    """Every cell of the card's chunk table that carries the card's number (before any fit is judged): the evidence the repair uses to decide whether
    a stored field is inconsistent with ALL of them."""
    t, cells = _cells(chunk_text or "")
    if t is None or p.value is None:
        return []
    return [c for c in cells if any(abs(n - p.value) < 1e-6 for n in _numbers(c.text))]


def model_fits(model: str | None, label: str) -> bool | None:
    """Public form of `_model_fits`: True / False when the row label names a system, None when it does not."""
    return _model_fits(model, label)


def ground(p: ConditionProfile, chunk_text: str) -> Grounding:
    """Find the table cell of a stored result in its chunk and read what the table says about it (a falsy Grounding when the chunk holds no table or no
    cell with the card's number)."""
    t, cells = _cells(chunk_text or "")
    if t is None or p.value is None:
        return Grounding()
    hits = [c for c in cells if any(abs(n - p.value) < 1e-6 for n in _numbers(c.text))]
    if not hits:
        return Grounding()

    def score(c: Cell) -> int:
        s = 0
        s += 2 * (_model_fits(p.model, c.row_label) is True) - 2 * (_model_fits(p.model, c.row_label) is False)
        s += 2 * (_column_fits(p.dataset, c.column) is True) - 1 * (_column_fits(p.dataset, c.column) is False)
        if _column_fits(p.dataset, c.column) is None and len(norm(p.dataset)) >= 3 and norm(p.dataset) in norm(c.column):
            s += 2                                                # "RACE test (Middle/High)" is the RACE cell, "SQuAD2.0 dev" is not
        blk = set(block_tags(c.block)) & set(_SPLIT_TAGS)
        stored = _split_of(p.setting)
        if blk and stored:
            s += 3 if blk == stored else -3
        return s

    best = max(hits, key=score)
    ties = [c for c in hits if score(c) == score(best)]
    if len(hits) > 1 and score(best) <= 0:
        return Grounding()                                  # several cells carry the number and none fits the card: do not guess
    if len(ties) > 1 and len({(c.row_label, c.column) for c in ties}) > 1 and score(best) < 2:
        return Grounding()
    g = Grounding(found=True, row_label=best.row_label, block=best.block, column=best.column)
    tags = block_tags(best.block)
    if not (tags & set(_SPLIT_TAGS)):
        tags = tags | caption_split(t.caption)               # the caption's split only where no heading states one
    tags = tags | caption_protocol(t.caption)
    g.setting_tags = frozenset(tags)
    g.size = label_size(best.row_label) or label_size(best.block) or caption_size(t.caption)
    fit_model, fit_data = _model_fits(p.model, best.row_label), _column_fits(p.dataset, best.column)
    if _model_fits(p.model, best.column) is True:             # a transposed table: the systems are the columns, the data sets the rows
        fit_model = True
        d, r = _task_of(p.dataset), _task_of(best.row_label)
        fit_data = None if d is None or r is None else d == r
    sub = re.match(r"\s*(\d+(?:\.\d+)?)\s*\(([^)]*)\)", best.text)         # "83.2 (86.5/81.3)": the overall figure and a breakdown (Middle / High)
    if sub and abs(float(sub.group(1)) - p.value) > 1e-6 and any(abs(n - p.value) < 1e-6 for n in _numbers(sub.group(2))):
        g.suspect.append("subset")                               # the card holds a part of the cell's figure
    if fit_data is False:
        g.suspect.append("dataset")
    if fit_model is False:
        g.suspect.append("model")
    elif fit_model is None and p.model:                       # a variant row: the table's subject is the system
        subject = caption_subject(t.caption)
        if subject and model_family(subject) and model_family(p.model) != model_family(subject) and model_family(subject) not in norm(p.model) \
                and not _name_like(best.row_label):
            g.suspect.append("model")
    return g


def _split_of(setting: str | None) -> set[str]:
    return set(block_tags(setting or "")) & set(_SPLIT_TAGS)


# --------------------------------------------------------------------------------------------------------------------------------------------------
# the enriched card
# --------------------------------------------------------------------------------------------------------------------------------------------------

def _size_like(value: str | None) -> bool:
    """A stored model_size that is a size: a label (large, base ...) or a parameter count with a unit; not "ensemble", "1M steps", "1.75M"."""
    if not value:
        return False
    v = value.strip().lower()
    if v in _SIZE_LABELS:
        return True
    return bool(re.fullmatch(r"\d+(?:\.\d+)?\s?(?:k|m|b)(?:\s?(?:params|parameters))?", v)) and "step" not in v and not re.fullmatch(r"\d\.\d+m", v)


def enriched(p: ConditionProfile, g: Grounding) -> ConditionProfile:
    """The card with the conditions the table states: its setting tags are replaced, category by category (split, system kind), by the block heading's tags,
    and an unusable or missing model size by the size the row label, heading or caption states."""
    if not g:
        return p
    stored = set(block_tags(p.setting or ""))
    from .conditions import setting_tags
    other = set(setting_tags(p.setting)) - {"dev", "test", "train", "single", "ensemble"}      # zero-shot, fine-tuned, translate-train ... stay as stored
    tags = set(g.setting_tags)
    for group in (_SPLIT_TAGS, _SYSTEM_TAGS):
        if not (tags & set(group)):
            tags |= stored & set(group)
    update: dict = {"setting": " ".join(sorted(tags | other)) or p.setting}
    if not _size_like(p.model_size):                          # "ensemble", "1M steps", "1.75M" are not sizes; a missing size is read from the table
        update["model_size"] = label_size(p.model or "") or g.size
    return p.model_copy(update=update)
