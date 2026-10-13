"""Make table cells explicit for the extractor: "BERTBASE 84.4 88.4 86.7" becomes
"BERTBASE: MNLI-m (Acc) = 84.4; QNLI (Acc) = 88.4; MRPC (Acc) = 86.7".

A small LLM is unreliable at matching a number to its column by position, so the column header is applied
here. Rows whose cells cannot be matched to a header unambiguously are left exactly as they were.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ..knowledge.benchmarks import lookup
from .pdf_loader import is_table_row

_NUM = re.compile(r"^[\(\[]?[-+]?\d+(?:[.,]\d+)?(?:/\d+(?:\.\d+)?)?(?:±\d+(?:\.\d+)?)?[%)\]]?$")
_EMPTY = {"-", "–", "—"}
_UNIT = re.compile(r"^\(.+\)$")


def _split_row(line: str) -> tuple[str, list[str]] | None:
    """(row label, cells): trailing numeric / empty tokens are cells, whatever comes before is the label."""
    tokens = line.split()
    n = len(tokens)
    while n > 0 and (_NUM.match(tokens[n - 1]) or tokens[n - 1] in _EMPTY):
        n -= 1
    label, cells = " ".join(tokens[:n]), tokens[n:]
    if not label or len(cells) < 2 or sum(bool(_NUM.match(t)) for t in tokens[:n]) > 1:
        return None
    return label, cells


_NAME = re.compile(r"^[A-Za-z#][\w\-+./*()#]*$")
_COUNT = re.compile(r"^\d[\d.,]*[kKmMbB]?$|^-$")


def _name_like(token: str) -> bool:
    """A plausible column name: MNLI-m, F1, SST-2, MNLI-(m/mm). Not 10th, 2018) or (Dec."""
    return bool(_NAME.match(token)) and token.count("(") == token.count(")")


def _header_above(lines: list[str], first_row: int, n_cells: int) -> list[str] | None:
    """Column names (with units) from the nearest header line above a run of rows; None unless unambiguous."""
    units: list[str] = []
    j = first_row - 1
    while j >= 0 and first_row - j <= 8:
        line = lines[j].strip()
        toks = line.split()
        if not toks:
            j -= 1
        elif all(_UNIT.match(t) for t in toks):
            units = units or toks
            j -= 1
        elif sum(bool(_COUNT.match(t)) for t in toks) >= 0.6 * len(toks):
            j -= 1
        elif is_table_row(line):
            return None
        elif len(toks) < n_cells or not all(_name_like(t) for t in toks[-n_cells:]):
            j -= 1
        else:
            names, lead = toks[-n_cells:], toks[:-n_cells]
            for k in range(2, 5):
                size = n_cells // k
                if n_cells % k == 0 and len(lead) >= k and names == names[:size] * k:
                    names = [f"{g} {n}" for g in lead[-k:] for n in names[:size]]
                    break
            if len(units) == n_cells:
                names = [f"{n} {u}" for n, u in zip(names, units)]
            return names
    return None


def linearize(lines: list[str]) -> tuple[list[str], list[int]]:
    """Copy of the lines with aligned numeric rows rewritten, and the indices of all numeric table rows."""
    out = list(lines)
    rows = [i for i, ln in enumerate(lines) if is_table_row(ln)]
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] - rows[j] <= 2:
            j += 1
        run = rows[i:j + 1]
        first = _split_row(lines[run[0]])
        header = _header_above(lines, run[0], len(first[1])) if first else None
        if header:
            for r in run:
                split = _split_row(lines[r])
                if split and len(split[1]) == len(header):
                    cells = "; ".join(f"{h} = {c}" for h, c in zip(header, split[1]) if c not in _EMPTY)
                    out[r] = f"{split[0]}: {cells}"
        i = j + 1
    return out, rows


_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_CELL_NUM = re.compile(r"^[\(\[]?[-+]?\d+(?:[.,]\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?[%)\]]?$")
_CELL_PAIR = re.compile(r"^[\(\[]?[-+]?\d+(?:\.\d+)?\s*/\s*(?:[-+]?\d+(?:\.\d+)?|[-–—])[%)\]]?$|^[-–—]\s*/\s*[-+]?\d+(?:\.\d+)?$")
_DASHES = {"", "-", "–", "—"}
_FOOTNOTE = re.compile(r"[\s†‡*§¶♠♣♦♥]+$")
_PLAIN_NUMBER = re.compile(r"\d+(?:\.\d+)?")
MAX_SUBHEADER_CELL = 14


def is_pipe_row(line: str) -> bool:
    return bool(_PIPE_ROW.match(line))


def is_numeric_cell(cell: str) -> bool:
    c = cell.strip()
    return bool(_CELL_NUM.match(c) or _CELL_PAIR.match(c))


def split_cells(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return [c.strip() for c in body.split("|")]


@dataclass
class ParsedTable:
    caption: str
    header: list[str]
    rows: list[list[str]]
    vocab: dict[str, str] = field(default_factory=dict)


def _is_subheader(row: list[str]) -> bool:
    filled = [c for c in row if c]
    if _scale_names_row(_repair_scale_header(row)):
        return True
    return (len(filled) >= 2 and not any(is_numeric_cell(c) for c in filled)
            and all(len(c) <= MAX_SUBHEADER_CELL for c in filled))


def _names_row(row: list[str], width: int) -> bool:
    """A row that names most of the columns: at least 60 % of the cells after the first are text."""
    filled = [c for c in row[1:] if c]
    return width >= 3 and len(filled) >= 0.6 * (width - 1) and not any(is_numeric_cell(c) for c in filled)


def _block_heading(row: list[str], width: int) -> bool:
    """A heading of a block of rows that the layout model cut at column borders ("Single-" | "Task Traini" | "ng"; mT5's "Cross-lingua" | "l zero-s" | "hot tran" |
    "sfer (mo" | ...): a few adjacent cells, under 60 % of the columns, that read as one sentence of at least three words. It heads the rows BELOW it. Taken for a
    header row, it was glued to the column names (GLUE's single-task block lost "Single-Task Training", mT5's WikiAnn table lost "Cross-lingual zero-shot
    transfer": found 12 Oct on a fresh random sample), so its rows had no regime."""
    idx = [i for i, c in enumerate(row) if c]
    if len(idx) < 2 or len(idx) > 0.6 * width:
        return False
    if any(b - a > 2 for a, b in zip(idx, idx[1:])):
        return False
    pieces = [row[i] for i in idx]
    if not (any(p[:1].islower() for p in pieces[1:]) or any(p.endswith("-") for p in pieces[:-1])):
        return False
    text = _join_fragments(pieces)
    return len(text) >= 18 and sum(ch.isalpha() for ch in text) >= 14 and len(text.split()) >= 2


def header_row_count(grid: list[list[str]]) -> int:
    width = max((len(r) for r in grid[:12]), default=0)
    n = 1
    names_seen = _names_row(grid[0], width)
    while n < len(grid) and _is_subheader(grid[n]):
        if names_seen and _block_heading(grid[n], width):
            break
        names_seen = names_seen or _names_row(grid[n], width)
        n += 1
    return n if n < len(grid) else 1


_UNIT_CELL = re.compile(r"^\([A-Za-z0-9@%./+\- ]{1,14}\)$")


_SIZE_CELL = re.compile(r"^(\d+(?:\.\d+)?)\s?([KMBT])$", re.I)
_SIZE_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _header_sizes(rows: list[list[str]], width: int) -> list[float | None] | None:
    """The parameter counts of a header row of sizes ("175B | 280B | 70B | 7B | 13B | 33B | 65B"), per column; None when no header row is one."""
    for r in rows:
        cells = r + [""] * (width - len(r))
        vals = []
        for c in cells:
            m = _SIZE_CELL.match(c.strip())
            vals.append(float(m.group(1)) * _SIZE_SCALE[m.group(2).lower()] if m else None)
        if sum(v is not None for v in vals) >= 3:
            return vals
    return None


def _compose_header(rows: list[list[str]]) -> list[str]:
    width = max(len(r) for r in rows)
    parts: list[list[str]] = [[] for _ in range(width)]
    sizes = _header_sizes(rows, width)
    for k, row in enumerate(rows):
        row = row + [""] * (width - len(row))
        if k < len(rows) - 1:
            for j in range(1, width - 1):
                a, b = row[j], row[j + 1]
                if (a and b and a.isalpha() and b.isalpha() and len(a) <= 4 and len(b) <= 4
                        and any(c and c not in (a, b) and len(c) > len(a + b) and c.lower().startswith((a + b).lower()) for c in row)):
                    start = j
                    if sizes and j > 1 and not row[j - 1] and sizes[j - 1] is not None and sizes[j] is not None and sizes[j - 1] < sizes[j]:
                        start = j - 1
                    row[j] = row[j + 1] = ""
                    row[start] = a + b
                    break
            for j in range(1, width):
                if row[j] and _UNIT_CELL.match(row[j]) and row[j - 1] and not _UNIT_CELL.match(row[j - 1]) and not is_numeric_cell(row[j - 1]):
                    row[j - 1], row[j] = f"{row[j - 1]} {row[j]}", ""
            last = ""
            for j in range(1, width):
                if row[j]:
                    last = row[j]
                elif last:
                    if sizes and sizes[j] is not None and sizes[j - 1] is not None and sizes[j] < sizes[j - 1]:
                        last = ""
                        pieces = row[j + 1:j + 3]
                        if len(pieces) == 2 and all(pieces) and all(len(x) <= 4 and x[:1].isalpha() for x in pieces):
                            row[j] = pieces[0] + pieces[1]
                            last = row[j]
                    else:
                        row[j] = last
        for j, c in enumerate(row):
            if c and c not in parts[j]:
                parts[j].append(c)
    return [" ".join(p) for p in parts]


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")


def _vocabulary(caption: str, grid: list[list[str]]) -> dict[str, str]:
    """The words the table itself spells out whole: its caption and every row that carries no number (the header rows, a restated header, group labels)."""
    vocab: dict[str, str] = {}
    texts = [caption] + [c for row in grid if not any(is_numeric_cell(x) for x in row[1:]) for c in row]
    for text in texts:
        for w in _WORD.findall(text or ""):
            vocab.setdefault(w.lower(), w)
    return vocab


def _stitch_cut_words(cells: list[str], vocab: dict[str, str]) -> list[str]:
    """The layout model cuts a header word that is wider than its column and puts the last letter into the NEXT cell ("SuperGLU Average" | "E BoolQ Accurac" |
    "CB y Accurac" | "CB y F1" for SuperGLUE / BoolQ / CB Accuracy / CB F1: GPT-3's Table 3.8, found 12 Oct when 12 of its results carried the data sets "CB y" and "E
    BoolQ"). A lone letter inside a cell is moved back onto a word of the previous cell when that word, with the letter, is a word the table spells whole elsewhere
    (its caption or another header row). Nothing else changes."""
    out = list(cells)
    for i in range(1, len(out)):
        toks = out[i].split()
        piece = len(toks) >= 2 and 2 <= len(toks[0]) <= 3 and toks[0].isalpha() and toks[0][:1].isupper()
        if len(toks) < 2 or not (piece or any(len(t) == 1 and t.isalpha() for t in toks)):
            continue
        j = i - 1
        while j >= 0 and not out[j]:
            j -= 1
        if j < 0:
            continue
        prev, kept = out[j].split(), []
        completed: str | None = None
        for t in toks:
            if len(t) == 1 and t.isalpha():
                for m in range(len(prev) - 1, -1, -1):
                    full = vocab.get((prev[m] + t).lower()) or ((prev[m] + t) if len(prev[m] + t) >= 5 and lookup(prev[m] + t) else None)
                    if full and len(prev[m]) >= 3 and len(full) >= 5:
                        prev[m] = full
                        completed = full if lookup(full) else completed
                        break
                else:
                    kept.append(t)
            else:
                kept.append(t)
        if not completed and piece:
            for m in range(len(prev) - 1, -1, -1):
                joined = prev[m] + toks[0]
                if len(prev[m]) >= 3 and len(joined) >= 5 and lookup(joined) is not None:
                    prev[m], completed = joined, joined
                    kept = kept[1:] if kept and kept[0] == toks[0] else kept
                    break
        if completed and kept and re.fullmatch(r"\d+(?:\.\d+)?", kept[0]):
            prev.insert(prev.index(completed) + 1, kept[0])
            kept = [completed] + kept
        out[j], out[i] = " ".join(prev), " ".join(kept)
    for i in range(1, len(out)):
        toks = out[i].split()
        j = i - 1
        while j >= 0 and not out[j]:
            j -= 1
        if j < 0 or not toks:
            continue
        prev = out[j].split()
        if len(prev) < 2:
            continue
        full = vocab.get((prev[-1] + toks[0]).lower())
        if full and len(prev[-1]) >= 2 and len(full) >= 5 and full.lower() in {w.lower() for w in prev[:-1]}:
            out[j], toks[0] = " ".join(prev[:-1]), full
            out[i] = " ".join(toks)
    return out


_GLUED_VALUE = re.compile(r"^[-+]?\d+(?:\.\d+)?%?(?:/[-+]?\d+(?:\.\d+)?%?)?$|^[-–—]$")
_LABEL_WORDS = {"model", "models", "method", "methods", "system", "systems", "approach", "approaches", "architecture", "name"}
_LABEL_CONTINUES = re.compile(r"^[(\[+]|[)\]]$")
_LABEL_SIZE_WORDS = {"base", "large", "small", "medium", "tiny", "mini", "xl", "xxl", "xlarge"}
_STATISTIC_HEADER = re.compile(r"param|time|speed|latency|memory|flops|throughput|cost|^#|\bsize\b|number of|count", re.I)


def _glued_column(cell: str) -> tuple[str, list[str]] | None:
    """('IMDb (acc.)', ['93.46', '92.82', '-']) for a cell made of a header and at least two values, else None."""
    toks = cell.split()
    k = 0
    while k < len(toks) - 1 and _GLUED_VALUE.match(toks[len(toks) - 1 - k]):
        k += 1
    head = toks[:len(toks) - k]
    if k < 2 or not any(re.search(r"[A-Za-z]", t) for t in head):
        return None
    return " ".join(head), toks[len(toks) - k:]


def _glued_labels(cell: str, k: int) -> tuple[str, list[str]] | None:
    """('Model', ['BERT-base', 'DistilBERT', 'DistilBERT (D)']) when the cell is a label word followed by exactly k labels, else None."""
    toks = cell.split()
    if len(toks) < 3 or toks[0].lower() not in _LABEL_WORDS:
        return None
    labels: list[str] = []
    for t in toks[1:]:
        if labels and (_LABEL_CONTINUES.search(t) or t.lower() in _LABEL_SIZE_WORDS):
            labels[-1] += " " + t
        else:
            labels.append(t)
    return (toks[0], labels) if len(labels) == k else None


def unglue_columns(text: str) -> str:
    """The text with a single row of whole-column cells replaced by a header and one row per label (see above), or the text itself."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
    if len(idx) != 1:
        return text
    cells = split_cells(lines[idx[0]])
    if len(cells) < 3 or any(not c for c in cells):
        return text
    groups: list[tuple[str, list[tuple[str, list[str]]]]] = []
    for c in cells:
        col = _glued_column(c)
        if col is None:
            groups.append((c, []))
        elif groups:
            groups[-1][1].append(col)
        else:
            return text
    kept: list[tuple[str, list[str], list[tuple[str, list[str]]]]] = []
    for label_cell, cols in groups:
        if not cols:
            return text
        ks = {len(v) for _, v in cols}
        parsed = _glued_labels(label_cell, ks.pop()) if len(ks) == 1 else None
        if parsed is None:
            return text
        if not all(_STATISTIC_HEADER.search(h) for h, _ in cols):
            kept.append((parsed[0], parsed[1], cols))
    if len(kept) != 1:
        return text
    head, labels, cols = kept[0]
    rows = [[head] + [h for h, _ in cols]] + [[lab] + [v[i] for _, v in cols] for i, lab in enumerate(labels)]
    out = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines[:idx[0]] + out + lines[idx[0] + 1:])


def parse_table(text: str) -> ParsedTable | None:
    """A loader-written table (caption, then pipe rows) or None for anything else."""
    text = unglue_columns(text)
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
    if len(idx) < 2:
        return None
    caption = " ".join(ln.strip() for ln in lines[:idx[0]] if ln.strip())
    grid = [split_cells(lines[i]) for i in idx]
    if sum(is_numeric_cell(c) for c in grid[0][1:]) >= 2:
        return ParsedTable(caption, [], grid)
    n_head = header_row_count(grid)
    vocab = _vocabulary(caption, grid)
    return ParsedTable(caption, _stitch_cut_words(_compose_header(grid[:n_head]), vocab), _split_merged_rows(grid[n_head:]), vocab)


_MERGED_UNIT = re.compile(r"^(?:[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?|[-–—])$")
_NOT_A_SYSTEM_WORD = frozenset({"base", "large", "small", "tiny", "mini", "medium", "xlarge", "xxlarge", "xl", "xxl", "single", "ensemble", "model", "models", "test", "dev",
                                "avg", "average", "human", "ours", "baseline", "ensembles", "results", "score", "best", "previous", "the", "and", "with", "without"})


def _split_merged_rows(rows: list[list[str]]) -> list[list[str]]:
    """The layout model sometimes prints two or three result rows as ONE line: the label holds the k system names ("DeBERTa1.5_B+SiFT DeBERTa_Ensemble") and every
    cell holds k values ("90.4 90.4", "94.9/97.2 95.7/97.6"; found 12 Oct on DeBERTa's SuperGLUE table, where the row was taken for a heading and lost).
    Such a row is split into k rows only when the label has exactly k distinct name-like words, at least three cells are filled, and EVERY filled cell holds exactly k
    numbers (or pairs, or dashes), one of them with a decimal. A single system with two numbers in a cell ("BERT base | 88.5 81.2") has a size word as its second
    "name" and is left alone."""
    out: list[list[str]] = []
    for row in rows:
        names = row[0].split() if row else []
        k = len(names)
        cells = [c.strip() for c in row[1:]]
        filled = [c for c in cells if c]
        if (k >= 2 and len(set(names)) == k and not any(n.lower().strip(".,;:") in _NOT_A_SYSTEM_WORD or len(n) < 2 for n in names)
                and all(re.search(r"[A-Za-z]", n) for n in names) and len(filled) >= 3
                and all(len(c.split()) == k and all(_MERGED_UNIT.match(t) for t in c.split()) for c in filled)
                and any(re.search(r"\d\.\d", c) for c in filled)):
            for i, name in enumerate(names):
                out.append([name] + [c.split()[i] if c else "" for c in cells])
        else:
            out.append(row)
    return out


def is_structured_table(text: str) -> bool:
    """A loader-written table with at least one row of numbers."""
    t = parse_table(text)
    return t is not None and any(is_numeric_cell(c) for row in t.rows for c in row[1:])


_NON_RESULT_CAPTION = re.compile(
    r"\b(statistics|hyper-?parameters?|model sizes?|dataset (?:sizes?|details|splits?)|data sizes?|corpus|corpora|training data)\b", re.I)
_RESULT_CUE = re.compile(
    r"\b(results?|accuracy|accuracies|f1|exact match|em|bleu|rouge|perplexity|scores?|performance|error|ablations?|"
    r"comparison|compared?|outperform\w*|evaluations?|evaluated)\b", re.I)


def is_statistics_table(caption: str) -> bool:
    """Captions of tables that describe data or models ("Languages and statistics of the CC-100 corpus", "Details on
    model sizes") and say nothing about results: not worth an LLM call."""
    return bool(_NON_RESULT_CAPTION.search(caption)) and not _RESULT_CUE.search(caption)


_ONLY_NUMBERS = re.compile(r"^[\d.,%\s\-–/]+$")
_GLUED = re.compile(r"\d\.\d+\.\d|\.\d{2,}\.\d")
_SPLIT_NUMBER = re.compile(r"^\d+\.\d+ \d$")
GARBLE_LIMIT = 0.10


def garble_ratio(text: str) -> float:
    """Share of a table's data cells that show the PDF text layer fell apart: a cell holding three or more separate numbers
    (a column of a stacked table read as one cell), numbers glued together ("0.000.10"), or a fragment that starts with
    ". ". A table with a high ratio cannot be read reliably, so its numbers are not turned into profiles."""
    t = parse_table(text)
    if t is None:
        return 0.0
    cells = [c.strip() for row in t.rows for c in row[1:] if c.strip()]
    if not cells:
        return 0.0
    bad = 0
    for c in cells:
        if _CELL_PAIR.match(c):
            continue
        if ((len(_PLAIN_NUMBER.findall(c)) >= 3 and _ONLY_NUMBERS.match(c) and " " in c) or _GLUED.search(c) or c.startswith(". ")
                or _SPLIT_NUMBER.match(c)):
            bad += 1
    return bad / len(cells)


def _join_fragments(cells: list[str]) -> str:
    """A label that spans several columns arrives cut at column borders ("... model o", "n English tra", "ining"):
    glue a fragment that starts with a lower-case letter onto one that ends in a letter."""
    out = ""
    for c in cells:
        if out and not (out[-1].isalpha() and c[:1].islower()):
            out += " "
        out += c
    return out.strip()


def _is_cut_heading(cells: list[str]) -> bool:
    """A group label that the layout cut at column borders can end in a piece that looks like a number ("(from leaderboard as of Sept. 16, 2" + "019)"
    cut "2019)" in two), which made the row a result row and lost the block heading ("Ensembles on test") for every row below it (found 10 Oct on ALBERT's
    GLUE table: four ensemble rows kept the heading of the dev block). A heading has several text pieces, no decimal and at most a lone integer piece."""
    filled = [c for c in cells[1:] if c not in _DASHES]
    numeric = [is_numeric_cell(c) for c in filled]
    decimals = sum(1 for c, ok in zip(filled, numeric) if ok and ("." in c or "/" in c))
    words = sum(1 for ok in numeric if not ok)
    return bool(cells[0]) and decimals == 0 and words >= 2 and words >= 2 * sum(numeric)


def _vertical_pieces(first: str, second: str) -> bool:
    """Two labels of neighbouring rows that are one name printed vertically beside a group of rows (Llama 2's tables: "L" over "lama 1"): the first a
    short run of letters, the second starting in lower case with no capital in it ("lama 1", "lama 2-hat"). A real name such as "mT5" or "mBERT"
    has a capital, so it is never glued."""
    return bool(re.fullmatch(r"[A-Za-z]{1,4}", first)) and bool(re.match(r"[a-z]", second)) and not re.search(r"[A-Z]", second)


@dataclass
class _Row:
    label: str
    context: str
    pairs: list[str]
    n_results: int
    index: int = -1


_YES_NO = {"yes", "no", "y", "n", "true", "false", "none", "n/a", "na"}


def _restated_header(cells: list[str], header: list[str]) -> list[str] | None:
    """A wide table printed in two halves one above the other repeats a header row of NEW column names in the middle ("| | ka | kk | ko | ... | avg |"
    under "| Model | af | ar | ... | jv |"; IndicXTREME's "| | | or | pa | sa | sat | ..."). Rows below it belong to the new names, but they were read with the
    first header (so a value of the second half got the language of the first half's column: found 11 Oct on mT5's WikiAnn / TyDi QA tables and IndicCOPA).
    Returns the new header, the current header when the row only repeats it, or None when the row is not a header: its cells must be short single-word
    names (not numbers, not a sentence cut at column borders), at least three, all different, and the row label cell empty or the header's own."""
    if not header or len(cells) < 4 or abs(len(cells) - len(header)) > 1:
        return None
    toks = [c for c in cells[1:] if c]
    if len(toks) < 3 or len(set(t.lower() for t in toks)) != len(toks):
        return None
    if any(len(c) > 8 or not re.fullmatch(r"[A-Za-z][A-Za-z.\-]*", c) or c.lower() in _YES_NO for c in toks):
        return None
    if cells[0] and cells[0].lower() != (header[0] or "").lower():
        return None
    known = {h.lower() for h in header if h}
    if sum(1 for c in toks if c.lower() in known) > 0.3 * len(toks):
        return list(header)
    return (cells + [""] * max(0, len(header) - len(cells)))[:max(len(header), len(cells))]


_MEASURE_WORD = re.compile(r"\b(?:acc(?:uracy)?|f1a?|em|exact match|bleu|rouge[\w\-]*|score|avg|average|precision|recall|mcc|matthews|pearson|spearman|correlation|"
                           r"error|perplexity|ppl|mrr|map|auc|top-?\d+|bpc|loss)\b", re.I)


def _named_header_row(cells: list[str], header: list[str], vocab: dict[str, str]) -> list[str] | None:
    """A SECOND header block printed under the first in a table whose columns are named by a data set and a measure ("WiC Accuracy" | "WSC Accuracy" | "MultiRC F1a"
    under "SuperGLUE Average" | "BoolQ Accuracy" | "CB Accuracy"; GPT-3's Table 3.8, found 12 Oct: its second block was read with the first block's column names, so WSC
    was stored as BoolQ). `_restated_header` takes only single short words; this one takes names of up to three words, and only a row with no number at all in which
    the cells name most of the columns (a group label over a few columns, or a sentence, is not one), the label cell is empty or the header's own, and at least
    three names differ. The new names are mended like the first header (`_stitch_cut_words`)."""
    if not header or len(cells) < 4 or abs(len(cells) - len(header)) > 1:
        return None
    if cells[0] and cells[0].lower() != (header[0] or "").lower():
        return None
    named = [c for c in cells[1:] if c]
    if len(named) < 3 or len(named) < 0.7 * (len(cells) - 1) or len({c.lower() for c in named}) < 3:
        return None
    for c in named:
        if (len(c.split()) > 3 or len(c) > 30 or not re.search(r"[A-Za-z]{2}", c) or is_numeric_cell(c) or c.lower() in _YES_NO
                or re.search(r"\d\.\d|\d{3}", c) or c.endswith((":", "."))):
            return None
    if sum(1 for c in named if _MEASURE_WORD.search(c)) < 0.7 * len(named):
        return None
    return _stitch_cut_words(list(cells) + [""] * max(0, len(header) - len(cells)), vocab)


def _rows(t: ParsedTable) -> list[_Row]:
    parsed: list[list] = []
    context, ctx_id = "", 0
    header = list(t.header)
    for cells in t.rows:
        restated = _restated_header(cells, header) or _named_header_row(cells, header, t.vocab)
        if restated is not None:
            header = restated
            continue
        numeric = [is_numeric_cell(c) for c in cells[1:]]
        if not any(numeric) or _is_cut_heading(cells):
            text = _join_fragments([c for c in cells if c])
            if text:
                context, ctx_id = text, ctx_id + 1
            continue
        pairs = []
        for j in range(1, len(cells)):
            if cells[j] in _DASHES:
                continue
            name = header[j] if j < len(header) and header[j] else f"column {j + 1}"
            pairs.append(f"{name} = {cells[j]}")
        n = sum(len(_PLAIN_NUMBER.findall(c)) for c, ok in zip(cells[1:], numeric) if ok)
        label = _FOOTNOTE.sub("", cells[0])
        factor = header[0].strip() if header else ""
        if label and re.fullmatch(r"\d+(?:\.\d+)?%?", label.strip()) and len(re.findall(r"[A-Za-z]", factor)) >= 4 and factor.lower() not in ("no.", "rank", "index"):
            label = f"{factor} {label.strip()}"
        parsed.append([label, context, pairs, n, ctx_id])
    for i in range(1, len(parsed)):
        cur, prev = parsed[i], parsed[i - 1]
        if cur[0] and len(cur[0]) <= 2 and cur[0].isalpha() and cur[0].islower() and prev[0][:1].isupper() and prev[4] == cur[4]:
            prev[0] = cur[0] = cur[0] + prev[0]
        elif cur[0] and prev[0] and prev[4] == cur[4] and _vertical_pieces(prev[0], cur[0]):
            prev[0] = cur[0] = prev[0] + cur[0]
    labelled = [i for i, r in enumerate(parsed) if r[0]]
    run = _size_runs([_row_size(r[2]) for r in parsed])
    for i, r in enumerate(parsed):
        if r[0]:
            continue
        same = [k for k in labelled if parsed[k][4] == r[4]]
        in_run = {parsed[k][0] for k in same if run[k] is not None and run[k] == run[i]}
        if len(in_run) == 1:
            r[0] = next(iter(in_run))
            continue
        before = max((k for k in same if k < i), default=None)
        after = min((k for k in same if k > i), default=None)
        pick = after if after is not None and (before is None or after - i <= i - before) else before
        r[0] = parsed[pick][0] if pick is not None else ""
    return [_Row(r[0], r[1], r[2], r[3], i) for i, r in enumerate(parsed)]


_PARAM_COUNT = re.compile(r"^(\d+(?:\.\d+)?)\s?([KMBT])$", re.I)
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def _row_size(pairs: list[str]) -> float | None:
    """The parameter count written in a row's own size column ("7B", "540B", "110M"), when the row has one."""
    for pair in pairs:
        m = _PARAM_COUNT.match(pair.rpartition(" = ")[2].strip())
        if m:
            return float(m.group(1)) * _SCALE[m.group(2).lower()]
    return None


def _size_runs(sizes: list[float | None]) -> list[int | None]:
    """Group consecutive rows whose model sizes grow (7B, 13B, 33B, 65B | 7B, 13B ...): a smaller size than the row above starts a new run. Rows
    without a size belong to no run."""
    out: list[int | None] = []
    run, last = 0, None
    for s in sizes:
        if s is None:
            out.append(None)
            last = None
            continue
        if last is not None and s <= last:
            run += 1
        out.append(run)
        last = s
    return out


def table_views(t: ParsedTable, max_results: int = 30) -> list[str]:
    """What the LLM reads for a table: caption + column names + a few rows at a time, each written as
    "row label: column = value; ...". Empty when the table has no numeric row."""
    rows = _rows(t)
    head = ([t.caption] if t.caption else []) + (["Columns: " + " | ".join(t.header)] if t.header else [])
    views: list[str] = []
    batch: list[_Row] = []
    count = 0

    def flush() -> None:
        nonlocal batch, count
        if not batch:
            return
        body, shown = [], None
        for r in batch:
            if r.context != shown:
                if r.context:
                    body.append(f"[{r.context}]")
                shown = r.context
            body.append(f"{r.label}: {'; '.join(r.pairs)}" if r.label else "; ".join(r.pairs))
        views.append("\n".join(head + body))
        batch, count = [], 0

    for r in rows:
        if batch and count + r.n_results > max_results:
            flush()
        batch.append(r)
        count += r.n_results
    flush()
    return views


def split_table_text(text: str, max_chars: int) -> list[str]:
    """A table longer than max_chars is cut into parts of whole rows, each part starting with the caption and header."""
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
    if len(idx) >= 2:
        grid = [split_cells(lines[i]) for i in idx]
        head_end = idx[header_row_count(grid) - 1] + 1
    else:
        head_end = 0
    head, body = lines[:head_end], lines[head_end:]
    parts: list[str] = []
    cur: list[str] = []
    size = sum(len(x) + 1 for x in head)
    for ln in body:
        if cur and size + len(ln) + 1 > max_chars:
            parts.append("\n".join(head + cur))
            cur, size = [], sum(len(x) + 1 for x in head)
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        parts.append("\n".join(head + cur))
    return parts


@dataclass
class Cell:
    """One result cell of a table with everything that says what it is: the row's label, the block heading above the row, the column name."""
    row_label: str
    block: str
    column: str
    text: str
    row: int = -1


def table_cells(t: ParsedTable) -> list[Cell]:
    """Every non-empty result cell of a parsed table (the same row labels, block headings and column names that `table_views` shows the LLM),
    kept as data so that a stored result can be checked against the cell it came from (pipeline/grounding.py)."""
    out: list[Cell] = []
    for row in _rows(t):
        for pair in row.pairs:
            column, _, text = pair.rpartition(" = ")
            out.append(Cell(row.label, row.context, column, text, row.index))
    return out


_NUMTOK = r"[-+]?\d+(?:\.\d+)?"
_MULTI = re.compile(rf"^(?:{_NUMTOK}|[-–—])(?:\s+(?:{_NUMTOK}|[-–—]))+$")
_APOSTROPHE = re.compile(r"^[’'`]")
_DECIMAL_TOKEN = re.compile(r"(?<![A-Za-z\-])\d+\.\d+")


def _row_kind(cells: list[str]) -> str:
    """data (a cell with numbers), head (two or more filled cells and no number: a header row) or label (a group label)."""
    body = cells[1:]
    if any(is_numeric_cell(c) or _MULTI.match(c) or _DECIMAL_TOKEN.search(c) for c in body):
        return "data"
    return "head" if sum(1 for c in body if c) >= 2 else "label"


def _decimals(token: str) -> int | None:
    m = re.fullmatch(r"[-+]?\d+(?:\.(\d+))?", token)
    return None if m is None else len(m.group(1) or "")


def _split_block(heads: list[list[str]], body: list[tuple[str, list[str]]], caption: str) -> str | None:
    """One header block with merged cells as one clean table text, or None when the block is not of this layout or anything in it is doubtful (a block
    that cannot be read cleanly is left alone, as before)."""
    width = len(heads[0])
    data = [r for kind, r in body if kind == "data"]
    if len(data) < 2:
        return None
    expanded: dict[int, int] = {}
    for j in range(1, width):
        counts: list[int] = []
        texts = 0
        for r in data:
            c = r[j].strip()
            if _MULTI.match(c):
                counts.append(len(c.split()))
            elif is_numeric_cell(c):
                counts.append(1)
            elif c not in _DASHES:
                texts += 1
        if texts and counts:
            return None
        if texts:
            continue
        if len(set(counts)) > 1:
            return None
        if counts and counts[0] >= 2:
            expanded[j] = counts[0]
    if not expanded:
        return None
    head_toks = [[h[j].split() for j in range(width)] for h in heads]
    for j in sorted(expanded):
        m = expanded[j]
        supported = False
        for hi in range(len(heads)):
            toks = head_toks[hi][j]
            if toks and _APOSTROPHE.match(toks[0]) and j > 1 and len(toks) - 1 == m and head_toks[hi][j - 1]:
                head_toks[hi][j - 1][-1] += toks[0]
                toks = head_toks[hi][j] = toks[1:]
            if len(toks) == m:
                supported = True
                continue
            if len(toks) <= 1:
                head_toks[hi][j] = (toks or [""]) * m
                continue
            return None
        if not supported:
            return None
        if any(len(t) == 1 and t.isalpha() for hi in range(len(heads)) for t in head_toks[hi][j] if t):
            return None
    rows: list[list[str]] = []
    for hi, h in enumerate(heads):
        row = [h[0]]
        for j in range(1, width):
            row += head_toks[hi][j] if j in expanded else [" ".join(head_toks[hi][j])]
        rows.append(row)
    for kind, r in body:
        row = [r[0]]
        for j in range(1, width):
            c = r[j].strip()
            if j not in expanded:
                row.append(c)
            elif kind == "data":
                parts = c.split() if _MULTI.match(c) else ([c] if is_numeric_cell(c) else [])
                row += (parts + ["-"] * expanded[j])[:expanded[j]]
            else:
                row += [c] + [""] * (expanded[j] - 1)
        rows.append(row)
    if len({len(r) for r in rows}) != 1:
        return None
    for j in range(1, len(rows[0])):
        values = [r[j] for r in rows[len(heads):] if r[j] not in _DASHES]
        dec = [d for d in (_decimals(v) for v in values) if d is not None]
        if dec and any(d == 0 for d in dec) and sum(1 for d in dec if d > 0) >= 0.5 * len(dec):
            return None
    text = (caption + "\n" if caption else "") + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return text if garble_ratio(text) < GARBLE_LIMIT and is_structured_table(text) else None


_DOT_DIGITS = re.compile(r"^\.(\d+)(?:\s+(.*))?$", re.S)
_PLAIN_TOKEN = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _stream_block(heads: list[list[str]], body: list[tuple[str, list[str]]], caption: str) -> str | None:
    """A block whose cell borders are unreliable (a number cut in two at a border: "43" | ".8"; header words cut in the middle: "EnD" | "e") but whose rows are
    regular as STREAMS (T5's Table 14, last block: WMT and CNN/DM). Every data row is read as the list of its numbers once the cut decimals are joined; all rows
    must give the same count k; the header rows are read as streams of words (a cut word is joined, a word that repeats and the one after it that does not are one
    name: "WMT" "EnDe") and must give exactly k names. Anything else: None, the block stays unread, as before."""
    data = [r for kind, r in body if kind == "data"]
    if len(data) < 2:
        return None
    width = len(heads[0])
    cut = Counter()
    for r in data:
        for j in range(1, width - 1):
            if re.fullmatch(r"\d+", r[j].strip()) and _DOT_DIGITS.match(r[j + 1].strip()):
                cut[j] += 1
    borders = {j for j, n in cut.items() if n >= 0.8 * len(data)}
    if not borders:
        return None
    streams: list[list[str]] = []
    labels: list[str] = []
    for kind, r in body:
        if kind != "data":
            continue
        cells = [c.strip() for c in r]
        for j in sorted(borders, reverse=True):
            m = _DOT_DIGITS.match(cells[j + 1])
            if not (re.fullmatch(r"\d+", cells[j]) and m):
                return None
            cells[j] = f"{cells[j]}.{m.group(1)}"
            cells[j + 1] = (m.group(2) or "").strip()
        toks = [t for c in cells[1:] for t in c.split()]
        if any(not (_PLAIN_TOKEN.match(t) or t in _DASHES) for t in toks):
            return None
        streams.append(toks)
        labels.append(cells[0])
    counts = {len(x) for x in streams}
    if len(counts) != 1:
        return None
    k = counts.pop()
    names: list[list[str]] = []
    for h in heads:
        toks = [t for c in h[1:] for t in c.split()]
        joined: list[str] = []
        for t in toks:
            if joined and re.fullmatch(r"[a-z]{1,2}", t) and joined[-1][-1:].isalpha():
                joined[-1] += t
            else:
                joined.append(t)
        toks = joined
        i = 0
        while i + 1 < len(toks):
            if toks[i] + toks[i + 1] in toks and len(toks[i]) <= 3 and len(toks[i + 1]) <= 3:
                toks[i:i + 2] = [toks[i] + toks[i + 1]]
            else:
                i += 1
        if len(toks) > k:
            freq = Counter(toks)
            out, i = [], 0
            while i < len(toks):
                if i + 1 < len(toks) and freq[toks[i]] >= 2 and freq[toks[i + 1]] == 1 and len(toks) - len(out) - 1 >= k - len(out):
                    out.append(toks[i] + " " + toks[i + 1])
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            toks = out
        if len(toks) != k:
            return None
        names.append(toks)
    header = [" ".join(dict.fromkeys(n[c] for n in names if n[c])) for c in range(k)]
    rows = [[heads[-1][0] or "Model", *header]] + [[lab, *s] for lab, s in zip(labels, streams)]
    text = (caption + "\n" if caption else "") + "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return text if garble_ratio(text) < GARBLE_LIMIT and is_structured_table(text) else None


_SCALE_WORD = re.compile(r"^(?:Small|Med|Medium|Large|XL|XXL|Base|Tiny|Mini|\d+(?:\.\d+)?[KMBT])$")
_SCALE_VALUE = re.compile(r"^\d{1,3}\.\d{1,2}$")
_SCALE_COUNT = re.compile(r"^\d{1,3}$")
_GLUED_TENTHS = re.compile(r"^(?:\d{2}\.\d){2,}$")
_SCALE_WORDS = {"small", "med", "medium", "large", "xl", "xxl", "base", "tiny", "mini"}


def _is_scale_cell(cell: str) -> bool:
    toks = cell.split()
    return bool(toks) and all(_SCALE_WORD.match(t) for t in toks)


def _scale_names_row(row: list[str]) -> bool:
    """A header row of model sizes with several names in a cell ("Med Large", "XL 2.7B 6.7B 13B 175B")."""
    cells = [c for c in row[1:] if c]
    multi = [c for c in cells if len(c.split()) >= 2 and _is_scale_cell(c)]
    return len(multi) >= 2 and sum(_is_scale_cell(c) for c in cells) >= 4


def _repair_scale_header(cells: list[str]) -> list[str]:
    """"Med Larg" | "e XL 2.7B ... (" | "175B test server)" -> "Med Large" | "XL 2.7B ..." | "(175B test server)": a lone lower-case fragment goes back onto the size
    word it completes, and a "(" that ends a cell opens the next one."""
    out = list(cells)
    for j in range(1, len(out)):
        toks = out[j].split()
        if len(toks) >= 2 and re.fullmatch(r"[a-z]{1,2}", toks[0]):
            p = j - 1
            while p >= 0 and not out[p]:
                p -= 1
            prev = out[p].split() if p >= 0 else []
            if prev and (prev[-1] + toks[0]).lower() in _SCALE_WORDS:
                prev[-1] += toks[0]
                out[p], out[j] = " ".join(prev), " ".join(toks[1:])
    for j in range(len(out) - 1):
        if out[j].endswith("(") and out[j + 1]:
            out[j], out[j + 1] = out[j][:-1].rstrip(), "(" + out[j + 1]
    return out


def _scaling_table(grid: list[list[str]], caption: str, family: str = "") -> str | None:
    """One clean table text (header with "Zero-Shot Small" ... names, one row per readable result row) or None; see the comment above. With `family`
    ("GPT-3": the model family the paper's text names with its sizes) a column is written "GPT-3 Small (Zero-Shot)" so that the extractor names the model, not "Small"."""
    width = len(grid[0])
    h = next((i for i, r in enumerate(grid[:6]) if _scale_names_row(_repair_scale_header(r))), None)
    if h is None or h + 2 >= len(grid):
        return None
    names = _repair_scale_header(grid[h])
    value_cols = [j for j in range(1, width) if names[j] and _is_scale_cell(names[j])]
    if len(value_cols) < 4:
        return None
    first, last = value_cols[0], value_cols[-1]
    if value_cols != list(range(first, last + 1)):
        return None
    extra = [j for j in range(last + 1, width) if names[j]]
    cols = value_cols + extra
    n_names = {j: names[j].split() if j in value_cols else [names[j].strip("() ")] for j in cols}
    labels = grid[h - 1] if h >= 1 and not any(re.search(r"\d", c) for c in grid[h - 1]) else [""] * width
    marks = [j for j in range(width) if labels[j]]
    if marks:
        group = {}
        for j in cols:
            after = [p for p in marks if p >= j]
            group[j] = labels[after[0]] if after else labels[marks[-1]]
    else:
        group = {j: "" for j in cols}
    per_group: dict[str, list[str]] = {}
    for j in cols:
        per_group.setdefault(group[j], []).extend(n_names[j])
    lists = list(per_group.values())
    sizes = lists[0]
    if len(lists) < 2 or any(l[:len(sizes)] != sizes for l in lists) or any(len(l) > len(sizes) + 1 for l in lists):
        return None
    def column_name(grp: str, size: str) -> str:
        size = {"Med": "Medium"}.get(size, size)
        if not family:
            return f"{grp} {size}".strip()
        return f"{family} {size} ({grp})" if grp else f"{family} {size}"

    expanded = [(column_name(group[j], n), j, len(n_names[j])) for j in cols for n in n_names[j]]
    pre = list(range(width))[:first]
    last_pre = pre[-1]
    header = [re.sub(r"\bFine-tun\b", "Fine-tuned", re.sub(r"^[a-z]\s+(?=\S)", "", names[j])).strip() for j in pre] + [e[0] for e in expanded]
    header = [re.sub(r"\s+", " ", x).strip() for x in header]

    def numeric(j: int, toks: list[str]) -> bool:
        return bool(toks) and all(_SCALE_VALUE.match(t) or (j == last_pre and _SCALE_COUNT.match(t)) for t in toks)

    rows: list[list[str]] = []
    for r in grid[h + 1:]:
        cells = (r + [""] * width)[:width]
        toks_of = [c.split() for c in cells]
        for j in cols:
            toks_of[j] = [x for t in toks_of[j] for x in (re.findall(r"\d{2}\.\d", t) if _GLUED_TENTHS.match(t) else [t])]
        k_set: set[int] = set()
        ok = True
        for j in range(1, width):
            toks = toks_of[j]
            if not toks:
                continue
            if j in cols:
                m = len(n_names[j])
                if not all(_SCALE_VALUE.match(t) for t in toks) or len(toks) % m:
                    ok = False
                    break
                k_set.add(len(toks) // m)
            elif j in pre and numeric(j, toks):
                k_set.add(len(toks))
        if not ok or len(k_set) != 1:
            continue
        k = next(iter(k_set))
        out: list[list[str]] = [[] for _ in range(k)]
        for j in range(width):
            toks = toks_of[j]
            if j in pre:
                if not toks:
                    parts: list[str] | None = [""] * k
                elif numeric(j, toks):
                    parts = toks if len(toks) == k else None
                elif k == 1:
                    parts = [" ".join(toks)]
                elif len(toks) % k == 0:
                    s = len(toks) // k
                    parts = [" ".join(toks[i * s:(i + 1) * s]) for i in range(k)]
                else:
                    parts = None
                if parts is None:
                    ok = False
                    break
                for i in range(k):
                    out[i].append(parts[i])
            elif j in cols:
                m = len(n_names[j])
                flat = toks if toks else [""] * (k * m)
                if len(flat) != k * m:
                    ok = False
                    break
                for i in range(k):
                    out[i].extend(flat[i * m:(i + 1) * m])
        if ok and all(len(o) == len(header) for o in out):
            rows.extend(out)
    if len(rows) < 3:
        return None
    hint = "Results of the paper's models: each row is one benchmark; each of the later columns is one model (by size) in one setting."
    text = (caption + " " if caption else "") + hint + "\n" + "\n".join("| " + " | ".join(x) + " |" for x in [header] + rows)
    return text if garble_ratio(text) < GARBLE_LIMIT and is_structured_table(text) else None


def _header_rows(lines: list[str]) -> list[str]:
    """The pipe rows of a table text that come before its first row of results (a row with a label and two or more numeric cells)."""
    out: list[str] = []
    for ln in lines:
        if not is_pipe_row(ln):
            continue
        cells = split_cells(ln)
        if sum(is_numeric_cell(c) or bool(_MULTI.match(c)) for c in cells[1:]) >= 2 and cells[0]:
            break
        out.append(ln)
    return out


def with_carried_headers(texts: list[str]) -> list[str]:
    """The chunk texts of one paper, in order, with the header rows of a table that a chunk continues. The chunker repeats only the first header row of a long table in
    each part, so a part whose names row did not make it ("XL 2.7B 6.7B 13B 175B" under "Zero-Shot") cannot be read. A text whose first pipe row equals the first pipe
    row of the text before it, and that does not carry the further header rows itself, gets them after its first row. Nothing else changes."""
    out: list[str] = []
    prev_head: list[str] = []
    for text in texts:
        lines = text.split("\n")
        idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
        if idx and len(prev_head) > 1 and lines[idx[0]].strip() == prev_head[0].strip():
            own = _header_rows(lines)
            if len(own) < len(prev_head):
                lines = lines[:idx[0] + 1] + prev_head[1:] + lines[idx[0] + 1:]
                text = "\n".join(lines)
        out.append(text)
        prev_head = _header_rows(text.split("\n"))
    return out


def rescue_tables(text: str, family: str = "") -> list[str] | None:
    """For a table that `garble_ratio` skips: one clean table text per header block whose merged columns could be split without doubt, or None."""
    if garble_ratio(text) < GARBLE_LIMIT:
        return None
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
    if len(idx) < 3:
        return None
    caption = " ".join(ln.strip() for ln in lines[:idx[0]] if ln.strip())
    width = max(len(split_cells(lines[i])) for i in idx)
    scaled = _scaling_table([(split_cells(lines[i]) + [""] * width)[:width] for i in idx], caption, family)
    if scaled is not None:
        return [scaled]
    blocks: list[tuple[list[list[str]], list[tuple[str, list[str]]]]] = []
    for i in idx:
        cells = (split_cells(lines[i]) + [""] * width)[:width]
        kind = _row_kind(cells)
        if kind == "head":
            if blocks and not blocks[-1][1]:
                blocks[-1][0].append(cells)
            else:
                blocks.append(([cells], []))
        elif blocks:
            blocks[-1][1].append((kind, cells))
        else:
            return None
    pieces = [p for p in ((_split_block(h, b, caption) or _stream_block(h, b, caption)) for h, b in blocks if b) if p]
    return pieces or None
