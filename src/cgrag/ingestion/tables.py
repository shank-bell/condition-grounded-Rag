"""Make table cells explicit for the extractor: "BERTBASE 84.4 88.4 86.7" becomes
"BERTBASE: MNLI-m (Acc) = 84.4; QNLI (Acc) = 88.4; MRPC (Acc) = 86.7".

A small LLM is unreliable at matching a number to its column by position, so the column header is applied
here. Rows whose cells cannot be matched to a header unambiguously are left exactly as they were.
"""
from __future__ import annotations

import re

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
