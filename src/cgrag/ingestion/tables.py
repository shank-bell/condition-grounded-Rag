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
        if k < len(rows) - 1:
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
    if sum(is_numeric_cell(c) for c in grid[0][1:]) >= 2:
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


def _rows(t: ParsedTable) -> list[_Row]:
    parsed: list[list] = []
    context, ctx_id = "", 0
    header = list(t.header)
    for cells in t.rows:
        restated = _restated_header(cells, header)
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
        parsed.append([_FOOTNOTE.sub("", cells[0]), context, pairs, n, ctx_id])
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


def rescue_tables(text: str) -> list[str] | None:
    """For a table that `garble_ratio` skips: one clean table text per header block whose merged columns could be split without doubt, or None."""
    if garble_ratio(text) < GARBLE_LIMIT:
        return None
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if is_pipe_row(ln)]
    if len(idx) < 3:
        return None
    caption = " ".join(ln.strip() for ln in lines[:idx[0]] if ln.strip())
    width = max(len(split_cells(lines[i])) for i in idx)
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
    pieces = [p for p in (_split_block(h, b, caption) for h, b in blocks if b) if p]
    return pieces or None
