"""Make table cells explicit for the extractor: "BERTBASE 84.4 88.4 86.7" becomes
"BERTBASE: MNLI-m (Acc) = 84.4; QNLI (Acc) = 88.4; MRPC (Acc) = 86.7".

A small LLM is unreliable at matching a number to its column by position, so the column header is applied
here. Rows whose cells cannot be matched to a header unambiguously are left exactly as they were.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

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
    if not label or len(cells) < 2 or sum(bool(_NUM.match(t)) for t in tokens[:n]) > 1:   # label full of numbers = merged rows
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
        if not toks:                                                # blank separator
            j -= 1
        elif all(_UNIT.match(t) for t in toks):                     # units line: (Acc) (Acc) (F1)
            units = units or toks
            j -= 1
        elif sum(bool(_COUNT.match(t)) for t in toks) >= 0.6 * len(toks):   # e.g. training-set sizes "392k 363k"
            j -= 1
        elif is_table_row(line):
            return None
        elif len(toks) < n_cells or not all(_name_like(t) for t in toks[-n_cells:]):
            j -= 1                                                  # a section label inside the table ("Published", "Ours")
        else:
            names, lead = toks[-n_cells:], toks[:-n_cells]          # leading tokens: row-label header, group labels
            for k in range(2, 5):                                   # "Dev Test" over "EM F1 EM F1" -> Dev EM, Dev F1, ...
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
            j += 1                                                  # rows[i..j] form one table
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


# ---------------------------------------------------------------------------------------------------------------------
# Structured tables. The loader writes a table it recognised as   <caption>\n| header | ... |\n| row | ... |   so the
# column of every number is known exactly; the header is applied here, in code, instead of trusting the LLM to count.
# ---------------------------------------------------------------------------------------------------------------------
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_CELL_NUM = re.compile(r"^[\(\[]?[-+]?\d+(?:[.,]\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?[%)\]]?$")
_CELL_PAIR = re.compile(r"^[\(\[]?[-+]?\d+(?:\.\d+)?\s*/\s*(?:[-+]?\d+(?:\.\d+)?|[-–—])[%)\]]?$|^[-–—]\s*/\s*[-+]?\d+(?:\.\d+)?$")
_DASHES = {"", "-", "–", "—"}
_FOOTNOTE = re.compile(r"[\s†‡*§¶♠♣♦♥]+$")
_PLAIN_NUMBER = re.compile(r"\d+(?:\.\d+)?")
MAX_SUBHEADER_CELL = 14        # a header continuation row ("EM F1 EM F1") has short cells; a group label is a long sentence


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
    header: list[str]             # one name per column ("Dev EM"), "" where the column has none
    rows: list[list[str]]         # every row after the header rows, cells as written


def _is_subheader(row: list[str]) -> bool:
    filled = [c for c in row if c]
    return (len(filled) >= 2 and not any(is_numeric_cell(c) for c in filled)
            and all(len(c) <= MAX_SUBHEADER_CELL for c in filled))


def header_row_count(grid: list[list[str]]) -> int:
    n = 1
    while n < len(grid) and _is_subheader(grid[n]):
        n += 1
    return n if n < len(grid) else 1


def _compose_header(rows: list[list[str]]) -> list[str]:
    width = max(len(r) for r in rows)
    parts: list[list[str]] = [[] for _ in range(width)]
    for k, row in enumerate(rows):
        row = row + [""] * (width - len(row))
        if k < len(rows) - 1:                                      # "Dev" over "EM F1": a group label spans its blank cells
            last = ""
            for j in range(1, width):
                if row[j]:
                    last = row[j]
                elif last:
                    row[j] = last
        for j, c in enumerate(row):
            if c and c not in parts[j]:
                parts[j].append(c)
    return [" ".join(p) for p in parts]


def parse_table(text: str) -> ParsedTable | None:
    """A loader-written table (caption, then pipe rows) or None for anything else."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
    if len(idx) < 2:
        return None
    caption = " ".join(ln.strip() for ln in lines[:idx[0]] if ln.strip())
    grid = [split_cells(lines[i]) for i in idx]
    if sum(is_numeric_cell(c) for c in grid[0][1:]) >= 2:           # no header row at all
        return ParsedTable(caption, [], grid)
    n_head = header_row_count(grid)
    return ParsedTable(caption, _compose_header(grid[:n_head]), grid[n_head:])


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
_SPLIT_NUMBER = re.compile(r"^\d+\.\d+ \d$")             # "53.84 9": the next cell starts with the rest of a split number
GARBLE_LIMIT = 0.10                                        # tables at or above this ratio are not read


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
    index: int = -1               # position among the result rows (cells of one row share it)


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
        return list(header)                                         # the same names again: only a repeat
    return (cells + [""] * max(0, len(header) - len(cells)))[:max(len(header), len(cells))]


def _rows(t: ParsedTable) -> list[_Row]:
    parsed: list[list] = []                   # [label or "", context, pairs, n_results, context id]
    context, ctx_id = "", 0
    header = list(t.header)
    for cells in t.rows:
        restated = _restated_header(cells, header)
        if restated is not None:
            header = restated
            continue
        numeric = [is_numeric_cell(c) for c in cells[1:]]
        if not any(numeric) or _is_cut_heading(cells):             # a group label ("Monolingual baselines", "Ours")
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
        parsed.append([_FOOTNOTE.sub("", cells[0]), context, pairs, n, ctx_id])
    for i in range(1, len(parsed)):                                # "mBERT" centred over two rows arrives as "BERT" / "m"
        cur, prev = parsed[i], parsed[i - 1]
        if cur[0] and len(cur[0]) <= 2 and cur[0].isalpha() and cur[0].islower() and prev[0][:1].isupper() and prev[4] == cur[4]:
            prev[0] = cur[0] = cur[0] + prev[0]
        elif cur[0] and prev[0] and prev[4] == cur[4] and _vertical_pieces(prev[0], cur[0]):
            prev[0] = cur[0] = prev[0] + cur[0]                    # "Llama 1" written vertically over a group arrives as "L" / "lama 1" (11 Oct)
    labelled = [i for i, r in enumerate(parsed) if r[0]]
    run = _size_runs([_row_size(r[2]) for r in parsed])
    for i, r in enumerate(parsed):                                 # a group label centred over its rows sits on one of them
        if r[0]:
            continue
        same = [k for k in labelled if parsed[k][4] == r[4]]
        in_run = {parsed[k][0] for k in same if run[k] is not None and run[k] == run[i]}
        if len(in_run) == 1:                                       # sizes 7B, 13B, 33B, 65B under one label: a smaller size starts the next model (11 Oct)
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
    block: str            # "Single-task single models on dev": the group label above the row ("" when none)
    column: str           # the column name, with its group label ("Dev EM")
    text: str             # the cell as written ("86.6/-", "90.2")
    row: int = -1         # index of the result row (the cells of one row share it: a size column, the other values)


def table_cells(t: ParsedTable) -> list[Cell]:
    """Every non-empty result cell of a parsed table (the same row labels, block headings and column names that `table_views` shows the LLM),
    kept as data so that a stored result can be checked against the cell it came from (pipeline/grounding.py)."""
    out: list[Cell] = []
    for row in _rows(t):
        for pair in row.pairs:
            column, _, text = pair.rpartition(" = ")
            out.append(Cell(row.label, row.context, column, text, row.index))
    return out
