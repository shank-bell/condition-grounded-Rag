"""Stage C - Condition Profile Extractor (NEW): one profile per reported result, linked to its Methods chunk.

For every Results / Experiments chunk the LLM (temperature 0, JSON-schema constrained) reads that chunk plus the
Methods chunk of the same paper that best matches it, and returns {task, dataset, dataset_version, metric, value,
language, model, model_size, setting}. A profile is kept only if its number really appears in the chunk text.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field, ValidationError
from rank_bm25 import BM25Okapi

from ..llm import OllamaLLM
from ..pipeline.conditions import is_nullish, norm, split_dataset
from ..schemas import Chunk, ConditionProfile, ExtractedProfile, ProfileExtraction
from ..stores.bm25_store import tokenize
from .pdf_loader import is_table_row
from .tables import GARBLE_LIMIT, garble_ratio, is_statistics_table, is_structured_table, linearize, parse_table, table_views

EXTRACT_SECTIONS = {"results", "experiments", "discussion"}
MAX_RESULTS_PER_CALL = 36        # numbers per table view; the JSON list is capped at MAX_PROFILES (40)
SKIP_SECTIONS = {"abstract", "introduction", "conclusion", "references"}
MAX_CHUNKS_PER_PAPER = 60        # main-body result chunks come first; bounds time on 70-page appendix-heavy papers
MAX_OUTPUT_TOKENS = 3000
MAX_PROFILES = 40                # bounds the JSON list so decoding cannot run away
MAX_ROWS = 4                     # a chunk with more numeric table rows than this is read ROWS_PER_CALL rows at a time
ROWS_PER_CALL = 3
WINDOW_ABOVE = 8                 # lines of header / units / caption kept above a table in its own view
WINDOW_BELOW = 4
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class _BoundedExtraction(ProfileExtraction):
    """Same shape as ProfileExtraction; the schema sent to the LLM just carries maxItems."""
    profiles: list[ExtractedProfile] = Field(default_factory=list, max_length=MAX_PROFILES)


_SCHEMA = _BoundedExtraction.model_json_schema()

SYSTEM = (
    "You extract experimental results from computer-science research papers. "
    "Output one profile for every distinct reported result: one metric value of one system on one dataset. "
    "In a table, rows are usually systems and columns are datasets, languages or metrics; every numeric cell is one "
    "result, and a cell like 84.6/83.4 or 80.2 / 67.4 is two results. A table is given as its caption, a 'Columns:' "
    "line and rows written 'row label: column = value; ...', which already pairs each number with its column; a line in "
    "[brackets] is a group label that applies to the rows below it (for example the training setting). READ THE CAPTION: "
    "it says what the numbers measure (metric, dataset, language); the metrics of a two-number cell follow the order in "
    "the caption ('F1 and EM' means first F1, then EM). The row label is the system that got the score: keep it as "
    "written, including a cited author such as 'Lample and Conneau (2019)'. Columns that describe the system or its "
    "training (for example D, #M, #lg, train) are not results. A column named by a language code (en, fr, hi) is the "
    "evaluation language: write it as the language name (English, French, Hindi). "
    "Otherwise match numbers to the column headers by position, "
    "and a unit line under the headers such as (Acc) (F1) gives the metric of each column. "
    "Report only results of evaluating a system (a model, a method, or a human baseline) with an evaluation metric such "
    "as accuracy, F1, EM, BLEU or perplexity. Do NOT report dataset statistics (counts, averages, category "
    "percentages), hyperparameters or training details. "
    "The METHODS chunk only describes the setup (datasets, model sizes, settings): never take a result number from it - "
    "every number you report must be written in the RESULTS chunk. "
    "Use only facts stated in the METHODS or RESULTS text. If a field is not stated, omit it - never guess. "
    "task: the task, e.g. question answering or natural language inference. "
    "dataset: dataset name only. dataset_version: version if stated, e.g. v1.1 or v2.0. "
    "metric: e.g. F1, EM, accuracy. value: the number only, no % sign. "
    "language: the language of the evaluation data. "
    "model: the system that obtained the score, e.g. BERT-large. "
    "model_size: parameter count or size label of the system, e.g. 340M, base, large - never a number of languages or "
    "models. "
    "setting: the evaluation regime, e.g. zero-shot, few-shot, fine-tuned, dev set, test set, single model, ensemble. "
    "evidence: a verbatim quote of at most 8 words that contains the number. "
    "Include baseline results that the text reports."
)


@dataclass
class ExtractionStats:
    chunks_seen: int = 0
    chunks_extracted: int = 0
    calls: int = 0
    truncated: int = 0            # replies cut at the token cap (complete profiles are still kept)
    failures: int = 0             # replies with no usable profile at all
    dropped_ungrounded: int = 0
    dropped_not_result: int = 0   # dataset statistics and other numbers that are not an evaluation result
    llm_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "ExtractionStats") -> None:
        for f in ("chunks_extracted", "calls", "truncated", "failures", "dropped_ungrounded", "dropped_not_result",
                  "llm_seconds"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.errors += other.errors


def _table_rows(text: str) -> int:
    """Rows of numbers with at least one decimal (axis ticks of a figure, "7 15 30 60 100", are not results)."""
    return sum(is_table_row(ln) and bool(re.search(r"\d+\.\d+", ln)) for ln in text.split("\n"))


def wants_extraction(chunk: Chunk) -> bool:
    """Chunks that report results: Results / Experiments / Discussion text with a few decimals, or - because
    heading-based section tags are heuristic - a table-dense chunk in any other section (not abstract,
    introduction, conclusion or references, which only repeat results)."""
    if chunk.section == "references":
        return False
    if is_structured_table(chunk.text):          # a table the loader recognised reports results wherever it sits
        return not is_statistics_table(parse_table(chunk.text).caption) and garble_ratio(chunk.text) < GARBLE_LIMIT
    if chunk.section in EXTRACT_SECTIONS:
        return len(re.findall(r"\d+\.\d+", chunk.text)) >= 3
    if chunk.section in SKIP_SECTIONS:
        return False
    return _table_rows(chunk.text) >= 3


def link_methods_chunk(target: Chunk, methods: list[Chunk]) -> Chunk | None:
    """The Methods chunk of the same paper whose wording is closest to the results chunk."""
    if not methods:
        return None
    if len(methods) == 1:
        return methods[0]
    scores = BM25Okapi([tokenize(m.text) for m in methods]).get_scores(tokenize(target.text))
    return methods[int(np.argmax(scores))]


_TEXT_FIELDS = ("task", "dataset", "dataset_version", "metric", "language", "model", "model_size", "setting")
_NOT_A_LANGUAGE = {"avg", "average", "mean", "overall", "all", "multiple", "various", "total"}
_NOT_A_SIZE = re.compile(r"^(?:\d+(?:\.\d+)?|[A-Za-z])$")
_STATISTIC = re.compile(
    r"\b(number of|count|average (?:number|length)|tokens?\b|percentage|proportion|fraction|distribution|carbon|tco2|mwh|kwh|gpu hours)", re.I)
_SPLITS = {"dev", "development", "test", "train", "validation", "val", "dev set", "test set", "train set", "validation set"}
_SIZE_WORDS = {"base", "large", "small", "medium", "tiny", "xl", "xxl"}
_REAL_LOWERCASE_MODELS = {"ada"}


def _junk_model(name: str) -> bool:
    """A "model" that is a piece of a row label cut at a column border ('L', 'MA', 'Fl', 'lama', 'acon'), a size word or a
    number. Real short names (T5, L2, mT5, ada) pass."""
    m = name.strip()
    if len(m) <= 1 or m.isdigit() or m.lower() in _SIZE_WORDS:
        return True
    if len(m) == 2 and not re.fullmatch(r"[A-Za-z]\d", m):
        return True
    return bool(re.fullmatch(r"[a-z][A-Za-z]{1,3}", m)) and m.lower() not in _REAL_LOWERCASE_MODELS


def _normalize(p: ExtractedProfile) -> ExtractedProfile:
    """Turn "not specified"-style strings into null, and split a version (and dev/test split) glued to the dataset name
    ("SQuAD 2.0 test" -> SQuAD + 2.0 + test set) because stage 7 compares versions and settings."""
    p = p.model_copy(update={f: None for f in _TEXT_FIELDS if getattr(p, f) is not None and is_nullish(getattr(p, f))})
    if p.dataset and p.dataset.strip().lower() in _SPLITS:                     # "Dev" / "Test" is the split, not the dataset
        p = p.model_copy(update={"dataset": None, "setting": p.setting or p.dataset.strip().lower()
                                 + ("" if p.dataset.strip().lower().endswith(" set") else " set")})
    if p.metric and p.dataset and norm(p.metric) == norm(p.dataset):           # a table whose header names the task ("GLUE")
        p = p.model_copy(update={"metric": "score"})
    if p.language and p.language.strip().lower() in _NOT_A_LANGUAGE:            # the "Avg" column is not a language
        p = p.model_copy(update={"language": None})
    if p.model_size and _NOT_A_SIZE.match(p.model_size.strip()):                # "100" (languages) or "N" (models), not a size
        p = p.model_copy(update={"model_size": None})
    if p.dataset:
        name, version, split = split_dataset(p.dataset)
        if version:
            return p.model_copy(update={"dataset": name, "dataset_version": p.dataset_version or version,
                                        "setting": p.setting or split})
    return p


def _junk_metric(metric: str) -> bool:
    """A metric that is a run of letters from a mis-read header ('mniorpasasatsdtateuravg'): one token of 20+ letters, or 45+ characters."""
    m = metric.strip()
    return len(m) > 45 or any(len(w) > 20 and w.isalpha() for w in m.split())


def _is_result(p: ExtractedProfile) -> bool:
    """An evaluation result needs a system that got the score and a metric; dataset statistics have neither."""
    if not (p.model and p.metric) or _STATISTIC.search(p.metric) or _junk_model(p.model) or _junk_metric(p.metric):
        return False
    return not (p.dataset and norm(p.model) == norm(p.dataset))              # a row that names a dataset, not a system


def _grounded(profile: ExtractedProfile, numbers_in_chunk: set[float]) -> bool:
    return profile.value is not None and any(abs(profile.value - n) < 1e-6 for n in numbers_in_chunk)


def _views(text: str) -> list[str]:
    """What the LLM reads. A chunk with one short table (or none) is read whole. Otherwise each table gets its own view,
    with only its own header and caption (a window of lines around it) so one table's caption cannot leak into another
    table's numbers, a long table is read a few rows at a time so one call never emits dozens of results, and the
    remaining prose gets a view of its own. A table the loader recognised (caption + header + rows) is read through
    table_views: the header is applied to every number in code."""
    table = parse_table(text)
    if table is not None:
        views = table_views(table, MAX_RESULTS_PER_CALL)
        if views:
            return views
    lines, rows = linearize(text.split("\n"))
    runs: list[list[int]] = []
    for r in rows:
        if runs and r - runs[-1][-1] <= 2:
            runs[-1].append(r)
        else:
            runs.append([r])
    if not runs or (len(runs) == 1 and len(rows) <= MAX_ROWS):
        return ["\n".join(lines)]
    views: list[str] = []
    covered: set[int] = set()
    row_set = set(rows)
    for run in runs:
        lo, hi = max(0, run[0] - WINDOW_ABOVE), min(len(lines), run[-1] + 1 + WINDOW_BELOW)
        covered.update(range(lo, hi))
        groups = [run] if len(run) <= MAX_ROWS else [run[i:i + ROWS_PER_CALL] for i in range(0, len(run), ROWS_PER_CALL)]
        for group in groups:
            drop = row_set - set(group)
            views.append("\n".join(lines[i] for i in range(lo, hi) if i not in drop))
    prose = [ln for i, ln in enumerate(lines) if i not in covered and i not in row_set and ln.strip()]
    if len(re.findall(r"\d+\.\d+", " ".join(prose))) >= 2:
        views.append("\n".join(prose))
    return views


def salvage_profiles(text: str) -> tuple[list[ExtractedProfile], bool]:
    """Every complete profile object in a JSON reply that may have been cut off; True if it was cut off."""
    i = text.find("[", max(text.find('"profiles"'), 0))
    if i < 0:
        return [], True
    decoder, out = json.JSONDecoder(), []
    i += 1
    while True:
        while i < len(text) and text[i] in " \n\r\t,":
            i += 1
        if i >= len(text):
            return out, True
        if text[i] == "]":
            return out, False
        if text[i] != "{":
            return out, True
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return out, True
        try:
            out.append(ExtractedProfile.model_validate(obj))
        except ValidationError:
            pass


NO_METHODS = "(no methods section found)"


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text)}


def _read_view(chunk: Chunk, methods_text: str, view: str, numbers: set[float],
               llm: OllamaLLM) -> tuple[list[ExtractedProfile], ExtractionStats]:
    """One LLM call: the profiles it returned that are grounded in the chunk and are evaluation results."""
    local = ExtractionStats()
    prompt = f"METHODS CHUNK:\n{methods_text}\n\nRESULTS CHUNK:\n{view}\n\nExtract all reported results."
    res = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                   schema=_SCHEMA, temperature=llm.cfg.extract_temperature, max_tokens=MAX_OUTPUT_TOKENS)
    local.calls += 1
    local.llm_seconds += res.wall_s
    profiles, cut = salvage_profiles(res.text)
    if cut:
        local.truncated += 1
    if not profiles and cut:
        local.failures += 1
        local.errors.append(f"{chunk.chunk_id}: no complete profile in {res.gen_tokens}-token reply")
    kept: list[ExtractedProfile] = []
    for p in map(_normalize, profiles):
        if not _grounded(p, numbers):
            local.dropped_ungrounded += 1
        elif not _is_result(p):
            local.dropped_not_result += 1
        else:
            kept.append(p)
    return kept, local


def _dedupe(profiles: list[ExtractedProfile]) -> list[ExtractedProfile]:
    seen: set[tuple] = set()
    out: list[ExtractedProfile] = []
    for p in profiles:
        key = (p.model, p.dataset, p.dataset_version, p.metric, p.value, p.setting, p.language)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def extract_chunk(chunk: Chunk, methods: Chunk | None, llm: OllamaLLM, stats: ExtractionStats) -> list[ExtractedProfile]:
    """Profiles of one chunk, its views read one after another (extract_paper reads them in parallel)."""
    methods_text = methods.text if methods else NO_METHODS
    numbers = _numbers(chunk.text)
    found: list[ExtractedProfile] = []
    for view in _views(chunk.text):
        kept, local = _read_view(chunk, methods_text, view, numbers, llm)
        stats.merge(local)
        found += kept
    return _dedupe(found)


def extract_paper(chunks: list[Chunk], llm: OllamaLLM) -> tuple[list[ConditionProfile], ExtractionStats]:
    """Profiles for every extractable chunk of one paper.

    Every view of every chunk is one LLM call and all calls of the paper run in parallel, so a chunk with several
    tables does not hold up a worker while the others sit idle.
    """
    stats = ExtractionStats(chunks_seen=len(chunks))
    # prose only: a results table that the heading heuristic filed under "methods" is not the setup description, and the
    # LLM would read its numbers instead of the table it was asked about (every one of them then fails the grounding check)
    methods = [c for c in chunks if c.section == "methods" and not is_structured_table(c.text)]
    wanted = [c for c in chunks if wants_extraction(c)]
    wanted.sort(key=lambda c: c.section not in EXTRACT_SECTIONS)        # stable: main-body result chunks first
    jobs = [(c, link_methods_chunk(c, methods)) for c in wanted[:MAX_CHUNKS_PER_PAPER]]

    calls: list[tuple[int, Chunk, str, str, set[float]]] = []           # (chunk index, chunk, methods text, view, numbers)
    for ci, (chunk, linked) in enumerate(jobs):
        methods_text, numbers = (linked.text if linked else NO_METHODS), _numbers(chunk.text)
        calls += [(ci, chunk, methods_text, view, numbers) for view in _views(chunk.text)]

    with ThreadPoolExecutor(max_workers=max(1, llm.cfg.parallel)) as pool:
        results = list(pool.map(lambda c: _read_view(c[1], c[2], c[3], c[4], llm), calls))

    per_chunk: dict[int, list[ExtractedProfile]] = {}
    for (ci, *_), (kept, local) in zip(calls, results):
        stats.merge(local)
        per_chunk.setdefault(ci, []).extend(kept)
    profiles: list[ConditionProfile] = []
    for ci, (chunk, linked) in enumerate(jobs):
        stats.chunks_extracted += 1
        for i, p in enumerate(_dedupe(per_chunk.get(ci, []))):
            profiles.append(ConditionProfile(
                **p.model_dump(), profile_id=f"{chunk.chunk_id}#{i}", paper_id=chunk.paper_id,
                chunk_id=chunk.chunk_id, methods_chunk_id=linked.chunk_id if linked else None))
    return profiles, stats
