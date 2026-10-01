"""Builds the Excel workbooks the team fills in. One workbook per person and job:

  A  profile check   (Adarsh, Shashank)   - 'Profiles' sheet, one row per extracted profile, yellow verdict cells
  B  questions       (everyone)           - 'Questions' sheet + the Corpus map lookup sheets
  C  result pairs    (Aditya, Tarun)      - 'Pairs' sheet, one row per pair of conflicting results, yellow verdict cells

Every workbook starts with a READ ME sheet (what to do, definitions, colour legend, an example row, a live progress counter).
Grey cells are given, yellow cells are the labeller's, sheets are protected (no password) so a grey cell cannot be overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment
from openpyxl.worksheet.worksheet import Worksheet

from ..schemas import ConditionProfile
from .sampling import Pair, arxiv_link, evidence_excerpt
from .xl import (CENTER, F_BASE, F_LINK, FILL_FIXED, WRAP, col, dropdown, fit_row, header_row, highlight_done, new_workbook,
                 protect, put, readme_sheet, wrapped_lines)

DEADLINE = "3 October 2026"

# ---------------------------------------------------------------- job A ----

A_FIELDS = ["model", "dataset", "dataset_version", "metric", "value", "language", "task", "model_size", "setting"]
A_VERDICTS = ["ALL_OK", "SOME_WRONG", "NOT_A_RESULT", "CANNOT_CHECK"]
A_FLAGS = ["WRONG", "MISSING"]
A_SOURCE = "source (caption, header, the row the number came from)"
A_FIXED = ["#", "profile_id", "paper", "page", "open"]
A_NOTE = "note / correct value"


def a_flag_header(field: str) -> str:
    return f"{field} wrong/missing?"


A_HEADERS = [*A_FIXED, *A_FIELDS, A_SOURCE, "verdict", *[a_flag_header(f) for f in A_FIELDS], A_NOTE]
A_WIDTHS = [4, 12, 28, 6, 8, 17, 14, 9, 12, 8, 11, 16, 9, 13, 62, 14, *([10] * len(A_FIELDS)), 34]

A_README = [
    ("What this is", [
        "The system read the results tables of 28 papers and wrote one 'profile' for every reported number: which model got which score, "
        "on which dataset, in which language and setting. We need to know how often those profiles are right. Your marks become the "
        "accuracy numbers (precision, recall, F1) of the extractor in the paper.",
        "Each row of the 'Profiles' sheet is one profile. You compare the grey cells with the paper and give a verdict. About 1 minute per row.",
    ]),
    ("What to do for each row", [
        "1. Click the blue 'open' link: it opens the paper at the page the number came from. The 'source' column already shows the table "
        "caption, its header and the table row (the number is marked [[like this]]), so many rows can be checked without opening the PDF.",
        "2. Read the grey cells (model, dataset, dataset_version, metric, value, language, task, model_size, setting). Are they what the "
        "paper says for that number?",
        "3. Pick a verdict in the yellow 'verdict' column:",
        "    ALL_OK - every filled-in cell is right. Cells the paper does not state may be empty - that is fine.",
        "    SOME_WRONG - a filled-in cell is wrong, or the paper clearly states something the profile left empty. Then mark WHICH cells in "
        "the yellow 'wrong/missing?' columns: WRONG = filled in but wrong, MISSING = the paper states it but the cell is empty. "
        "Cells you leave blank are counted as right.",
        "    NOT_A_RESULT - this row is not a reported result at all (a dataset size, a count, a header, nonsense).",
        "    CANNOT_CHECK - you cannot verify it from the paper (say why in the note).",
    ]),
    ("What each cell means", [
        "model: the system that got the score (BERT-large, XLM-R, mT5-XL ...), as the table names it. dataset: the benchmark or test set "
        "without its version (SQuAD, XNLI, GLUE, MMLU). dataset_version: only if the paper states one (1.1, 2.0).",
        "metric: what the number measures (accuracy, F1, EM, BLEU ...). value: the number itself - check it against the right ROW and COLUMN.",
        "language: the language this number is for (Hindi, Kannada ...). A language code in the table header (hi, kn) must have become the "
        "language name. Empty is right for English-only or unstated results; 'Avg' is not a language.",
        "task: the kind of problem (natural language inference, question answering ...). model_size: only if stated (110M, 7B). "
        "setting: how it was evaluated (dev set, test set, zero-shot, fine-tuned, translate-train ...).",
    ]),
    ("Things that often go wrong (look for these)", [
        "The number belongs to a different column than the metric/language the profile names - especially when the same number appears "
        "twice in a row.",
        "The row label is a piece of a name ('L', 'MA') or the wrong model: a group label in the first column applies to several rows.",
        "A baseline's number is attributed to the paper's own model, or the dev number is recorded as a test number.",
    ]),
    ("Rules", [
        "Change only the yellow cells. Do not look at the system's answers while you label and do not paste profiles or system output into "
        "chat tools. If a table layout confuses you, you may ask Gemini to explain it from PASTED PDF TEXT - you still decide the verdict.",
        "The last rows (the ones marked in the 'paper' column) are also labelled by another team member to measure how much two people "
        "agree. Label them on your own and do not compare.",
        f"Save often. When you are finished, save a copy named like this one with _DONE added and send it back. Due: {DEADLINE}.",
    ]),
]
A_EXAMPLE = (["model", "dataset", "metric", "value", "language", "verdict", "language wrong/missing?", "note / correct value"],
             [["BERT-large", "SQuAD", "F1", "90.9", "(empty)", "ALL_OK", "", ""],
              ["XLM-R", "XNLI", "accuracy", "71.5", "Hindi", "SOME_WRONG", "WRONG", "the number is in the Kannada column: language should be Kannada"]])


@dataclass
class ARow:
    profile: ConditionProfile
    title: str
    page: int
    excerpt: str
    shared: bool = False           # also given to the other labeller (agreement check)


def a_rows(profiles: list[ConditionProfile], texts: dict[str, str], pages: dict[str, int], titles: dict[str, str],
           shared: set[str] = frozenset()) -> list[ARow]:
    return [ARow(p, titles.get(p.paper_id, ""), pages.get(p.chunk_id, 1), evidence_excerpt(texts.get(p.chunk_id, ""), p),
                 p.profile_id in shared) for p in profiles]


def build_job_a(path: Path, labeller: str, rows: list[ARow]) -> Path:
    wb = new_workbook()
    ws = wb.create_sheet("Profiles")
    header_row(ws, A_HEADERS, A_WIDTHS)
    n_fix, n_f = len(A_FIXED), len(A_FIELDS)
    source_c, verdict_c = n_fix + n_f + 1, n_fix + n_f + 2
    flag0_c, note_c = verdict_c + 1, verdict_c + 1 + n_f
    for i, r in enumerate(rows, start=2):
        p = r.profile
        put(ws, i, 1, i - 1, center=True)
        put(ws, i, 2, p.profile_id)
        put(ws, i, 3, f"{p.paper_id}  {r.title}" + ("\n(also labelled by someone else)" if r.shared else ""))
        put(ws, i, 4, r.page, center=True)
        put(ws, i, 5, "open", link=arxiv_link(p.paper_id, r.page), center=True)
        for k, f in enumerate(A_FIELDS):
            v = getattr(p, f)
            put(ws, i, n_fix + 1 + k, v if v not in ("", None) else None)
        put(ws, i, source_c, r.excerpt)
        put(ws, i, verdict_c, None, kind="input", center=True)
        for k in range(n_f):
            put(ws, i, flag0_c + k, None, kind="input", center=True)
        put(ws, i, note_c, None, kind="input")
        fit_row(ws, i, [(r.excerpt, A_WIDTHS[source_c - 1]), (f"{p.paper_id}  {r.title}", A_WIDTHS[2])])
    last = len(rows) + 1
    vl = col(verdict_c)
    dropdown(ws, f"{vl}2:{vl}{last}", A_VERDICTS, title="Verdict", prompt="ALL_OK, SOME_WRONG, NOT_A_RESULT or CANNOT_CHECK")
    dropdown(ws, f"{col(flag0_c)}2:{col(flag0_c + n_f - 1)}{last}", A_FLAGS, title="Which cell?",
             prompt="Only for SOME_WRONG: WRONG = filled but wrong, MISSING = the paper states it but the cell is empty")
    highlight_done(ws, f"{vl}2:{vl}{last}", f"{vl}2")
    ws.freeze_panes = ws.cell(row=2, column=n_fix + 1)
    ws.auto_filter.ref = f"A1:{col(len(A_HEADERS))}{last}"
    protect(ws, hidden=(2,))
    progress = (f'"Rows done: "&COUNTA(Profiles!{vl}2:{vl}{last})&" of "&ROWS(Profiles!{vl}2:{vl}{last})')
    readme_sheet(wb, f"Job A - profile check - {labeller}", A_README, example=A_EXAMPLE, progress=progress)
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# ---------------------------------------------------------------- job B ----

B_TYPES = ["covered", "one missing", "partly covered"]
B_INTENTS = ["factual", "method", "result", "comparison", "survey"]
B_COMPLEXITY = ["simple", "complex"]
B_CONDITIONS = ["task", "dataset", "dataset_version", "model", "model_size", "language", "setting"]
B_HEADERS = ["id", "author", "planned type", "question", "intent", "complexity", *B_CONDITIONS, "expected scope warning",
             "missing conditions", "evidence (paper, page, table)", "expected answer facts", "notes"]
B_WIDTHS = [9, 10, 13, 54, 11, 11, 18, 14, 10, 14, 10, 12, 12, 12, 24, 34, 44, 28]
B_README = [
    ("What this is", [
        "You write test questions for the finished system and say what the RIGHT behaviour is. These questions are the exam: does the "
        "system answer correctly and - the key idea of the project - does it WARN when the papers do not cover a condition named in the "
        "question (a language, a dataset, a model size ...)? About 5 minutes per question.",
        "The 'planned type' of each row is fixed (grey) so that the team ends up with 10 covered, 10 one-condition-missing and "
        "10 partly-covered questions.",
    ]),
    ("The three kinds of question", [
        "covered - at least one paper reports what the question asks for ALL the conditions it names together. Example: 'How well do "
        "models perform on Kannada NLI?' IS covered: IndicXNLI (paper 2212.05409, Table 16) reports mBERT 58.6, XLM-R 71.5, MuRIL 74.0 on Kannada.",
        "one missing - everything is covered except ONE condition, so the system must warn about it. Example: 'What accuracy does XLM-R "
        "get on XNLI for Kannada?' - XNLI has 15 languages and Kannada is not one of them. Never assume coverage: look it up.",
        "partly covered - the question has several parts or models and only some are covered. Example: 'Compare mBERT, XLM-R and GPT-4 on "
        "XNLI' - GPT-4 is in no paper; or a result exists only in a different setting than the one asked for.",
    ]),
    ("What to do for each row", [
        "1. Write the question the way a student would ask it (informal is fine, one or two sentences).",
        "2. Pick intent and complexity (definitions below).",
        "3. Fill in the conditions the question NAMES (task, dataset, dataset_version, model, model_size, language, setting). Leave blank "
        "what it does not name.",
        "4. Look up every named condition in the lookup sheets (Papers, Datasets, Dataset x Language, Models, Model x Dataset; use Ctrl+F). "
        "They are built from the system's own reading of the papers, so they can be wrong or incomplete: CONFIRM IN THE PAPER (the 'Papers' "
        "sheet has links). A blank cell does not prove a paper lacks something - search the paper.",
        "5. Decide the expected behaviour: 'expected scope warning' = YES if some named condition has no evidence in the papers (together "
        "with the others), otherwise NO. If YES, write the missing condition(s) like language=Kannada.",
        "6. Evidence: paper id, page and table where the answer is (or where you looked and found nothing). Expected answer facts: the "
        "numbers or facts a correct answer must contain, with their source.",
    ]),
    ("Intent (what kind of question)", [
        "factual - asks for a stated fact (a number, a name, a definition) that is NOT a benchmark score: 'How many parameters does mT5-XL have?'",
        "method - how or why a technique works: 'How does ELECTRA's discriminator training work?'",
        "result - asks for reported scores: 'What F1 does BERT-large get on SQuAD 2.0?'",
        "comparison - compare or contrast, 'which is better', 'do the papers agree': 'Is DeBERTa better than RoBERTa on MNLI?'",
        "survey - an overview of a topic or a family of models: 'What are the main ways to make BERT smaller?'",
    ]),
    ("Complexity", [
        "simple - asks for exactly ONE thing (one fact, one score, one procedure). complex - asks for several things, lists or aggregates "
        "(all languages, every task, different sizes), compares things, or asks for an overview. Every comparison and every survey "
        "question is complex.",
    ]),
    ("Rules", [
        "Change only the yellow cells. Do not run your question through the system before you have written the expected behaviour. "
        "Gemini may help you draft or rephrase questions, but the labels (intent, complexity, coverage) are yours.",
        f"Save often; when you are finished save a copy named like this one with _DONE added and send it back. Due: {DEADLINE}.",
    ]),
]
B_EXAMPLE = (["question", "intent", "complexity", "language", "task", "expected scope warning", "missing conditions", "evidence"],
             [["How well do models perform on Kannada NLI?", "result", "simple", "Kannada", "natural language inference", "NO", "",
               "2212.05409 p20 Table 16: mBERT 58.6, XLM-R 71.5, MuRIL 74.0"],
              ["What accuracy does XLM-R get on XNLI for Kannada?", "result", "simple", "Kannada", "(dataset XNLI, model XLM-R)", "YES",
               "language=Kannada", "1809.05053 Table 1: XNLI covers 15 languages, no Kannada"]])


def add_table_sheet(wb, name: str, headers: list[str], rows: list[list], *, heat: bool = False, first_numeric_col: int = 2) -> Worksheet:
    """A lookup sheet: header, rows, filter, frozen header; ("text", url) cells become hyperlinks; heat=True shades a count matrix."""
    ws = wb.create_sheet(name)
    widths = []
    for i, h in enumerate(headers):
        longest = max([len(str(h))] + [len(str(r[i][0] if isinstance(r[i], tuple) else r[i])) for r in rows if i < len(r) and r[i] is not None])
        widths.append(min(max(8, longest + 2), 60) if not heat or i < first_numeric_col else 5)
    header_row(ws, headers, widths)
    if heat:
        ws.row_dimensions[1].height = 80
        for i in range(first_numeric_col, len(headers)):
            ws.cell(row=1, column=i + 1).alignment = Alignment(wrap_text=True, vertical="bottom", horizontal="center", textRotation=90)
    for r, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            if isinstance(v, tuple):
                put(ws, r, c, v[0], kind="plain", link=v[1], center=True)
            else:
                put(ws, r, c, v, kind="plain", center=heat and c > first_numeric_col - 1 or isinstance(v, (int, float)))
        if not heat:
            fit_row(ws, r, [(str(v), widths[c]) for c, v in enumerate(row) if isinstance(v, str)], minimum=18, maximum=120)
    if rows:
        ws.auto_filter.ref = f"A1:{col(len(headers))}{len(rows) + 1}"
        if heat:
            rng = f"{col(first_numeric_col + 1)}2:{col(len(headers))}{len(rows) + 1}"
            ws.conditional_formatting.add(rng, ColorScaleRule(start_type="num", start_value=1, start_color="E2EFDA",
                                                              end_type="max", end_color="548235"))
    ws.freeze_panes = ws.cell(row=2, column=2)
    protect(ws)
    return ws


def build_job_b(path: Path, author: str, code: str, planned: list[str], lookup: dict[str, tuple[list[str], list[list]]]) -> Path:
    wb = new_workbook()
    ws = wb.create_sheet("Questions")
    header_row(ws, B_HEADERS, B_WIDTHS)
    for i, kind in enumerate(planned, start=2):
        put(ws, i, 1, f"B-{code}-{i - 1:02d}", center=True)
        put(ws, i, 2, author, center=True)
        put(ws, i, 3, kind, center=True)
        for c in range(4, len(B_HEADERS) + 1):
            put(ws, i, c, None, kind="input", center=c in (5, 6, 14))
        ws.row_dimensions[i].height = 62
    last = len(planned) + 1
    dropdown(ws, f"E2:E{last}", B_INTENTS, title="Intent", prompt="factual, method, result, comparison or survey (see READ ME)")
    dropdown(ws, f"F2:F{last}", B_COMPLEXITY, title="Complexity", prompt="simple = exactly one thing; complex = several things / compare / overview")
    n_cond = len(B_CONDITIONS)
    warn_col = col(6 + n_cond + 1)
    dropdown(ws, f"{warn_col}2:{warn_col}{last}", ["YES", "NO"], title="Scope warning expected?",
             prompt="YES if some condition named in the question has no evidence in the papers")
    highlight_done(ws, f"D2:D{last}", "D2")
    ws.freeze_panes = "E2"
    protect(ws)
    for name, (headers, rows) in lookup.items():
        add_table_sheet(wb, name, headers, rows, heat=name in ("Dataset x Language", "Model x Dataset"), first_numeric_col=2 if name == "Dataset x Language" else 1)
    progress = f'"Questions written: "&COUNTA(Questions!D2:D{last})&" of "&ROWS(Questions!D2:D{last})'
    readme_sheet(wb, f"Job B - write test questions - {author}", B_README, example=B_EXAMPLE, progress=progress)
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# ---------------------------------------------------------------- job C ----

C_VERDICTS = ["EXPLAINED", "GENUINE", "NOT COMPARABLE"]
C_DIFFERS = ["dataset version / split", "model size", "language", "setting", "other"]
C_HEADERS = ["pair", "metric", "model (paper A | paper B)", "dataset (paper A | paper B)",
             "paper A", "page A", "open A", "value A", "source A", "paper B", "page B", "open B", "value B", "source B",
             "verdict", *[f"differs: {d}" for d in C_DIFFERS], "other: what", "extraction error?", "confidence", "explanation (quote the paper)"]
C_WIDTHS = [7, 11, 22, 20, 26, 6, 8, 8, 52, 26, 6, 8, 8, 52, 16, 11, 9, 9, 9, 9, 18, 11, 10, 40]
C_README = [
    ("What this is", [
        "Two papers report different numbers for what looks like the same result: the same model, dataset and metric. The system has to "
        "tell a REAL conflict from a difference that is EXPLAINED by different conditions. You decide what the right answer is, from the "
        "papers - not from the system. About 5 minutes per pair.",
    ]),
    ("Your three verdicts", [
        "EXPLAINED - the numbers differ because the two papers ran the experiment under different conditions, so there is no real "
        "disagreement. Then tick (type Y) WHICH condition differs: dataset version / split (SQuAD 1.1 vs 2.0, dev vs test split), model size "
        "(base vs large, 340M vs 1.5B), language, setting (zero-shot vs fine-tuned, translate-train vs zero-shot, single model vs ensemble, "
        "dev vs test set), or other (say what in 'other: what'). Several can differ at once.",
        "GENUINE - same conditions, still different results: a real disagreement between the papers. Be strict: only say GENUINE after you "
        "checked that the conditions really match in both papers.",
        "NOT COMPARABLE - you cannot tell: the metric is defined differently, the task differs, or a paper does not state the conditions "
        "well enough to judge.",
    ]),
    ("What to do for each pair", [
        "1. Look at both sources (grey): caption, header, and the table row; the number is marked [[like this]]. Click 'open A' / 'open B' "
        "to open the paper at that page. Read the setup around the table if the caption does not say how it was run.",
        "2. Decide the verdict and fill the yellow cells. 'confidence' = high or low. In 'explanation' quote the words of the paper that "
        "decided it (for example 'Table 2 is the dev set, Table 5 the test set').",
        "3. If one of the two numbers in the sheet is not what the paper says (the system read the table wrongly), put Y in "
        "'extraction error?' and judge the rest as well as you can.",
    ]),
    ("Rules", [
        "Do this on your own. Two people (Aditya and Tarun) label ALL pairs independently; do not discuss until both are finished. A third "
        "team member settles the pairs where you disagree, and we report Cohen's kappa between you two (target 0.6 or higher).",
        "Judge from the papers. Do not run the pair through the system first, and do not paste the system's output into chat tools. "
        "Gemini may explain a confusing table from pasted PDF text; the verdict is yours.",
        f"Save often; when you are finished save a copy named like this one with _DONE added and send it back. Due: {DEADLINE}.",
    ]),
    ("Two worked examples", [
        "Human EM on SQuAD: 82.3 in one paper, 86.9 in another. The first measured SQuAD 1.1, the second SQuAD 2.0 -> EXPLAINED, differs: "
        "dataset version / split.",
        "XLM-R XNLI average: 83.6 in the XLM-R paper (translate-train-all) and 79.2 in the mT5 paper (zero-shot) -> EXPLAINED, differs: setting.",
    ]),
]
C_EXAMPLE = (["pair", "verdict", "differs: setting", "other: what", "extraction error?", "confidence", "explanation"],
             [["P00", "EXPLAINED", "Y", "", "", "high", "Paper A Table 3 reports zero-shot transfer; paper B Table 5 is translate-train-all."]])


def _model_cell(a: ConditionProfile, b: ConditionProfile, field: str) -> str:
    x, y = getattr(a, field) or "", getattr(b, field) or ""
    return x if x == y else f"{x} | {y}"


def build_job_c(path: Path, labeller: str, pairs: list[tuple[str, Pair]], texts: dict[str, str], pages: dict[str, int],
                titles: dict[str, str]) -> Path:
    """pairs: (pair_id, pair) in the order this labeller sees them (shuffled per labeller; the ids are the same for everyone)."""
    wb = new_workbook()
    ws = wb.create_sheet("Pairs")
    header_row(ws, C_HEADERS, C_WIDTHS)
    for i, (pid, pair) in enumerate(pairs, start=2):
        x, y = pair.x, pair.y
        px, py = pages.get(x.chunk_id, 1), pages.get(y.chunk_id, 1)
        sx, sy = evidence_excerpt(texts.get(x.chunk_id, ""), x), evidence_excerpt(texts.get(y.chunk_id, ""), y)
        put(ws, i, 1, pid, center=True)
        put(ws, i, 2, x.metric)
        put(ws, i, 3, _model_cell(x, y, "model"))
        put(ws, i, 4, _model_cell(x, y, "dataset"))
        put(ws, i, 5, f"{x.paper_id}  {titles.get(x.paper_id, '')}")
        put(ws, i, 6, px, center=True)
        put(ws, i, 7, "open A", link=arxiv_link(x.paper_id, px), center=True)
        put(ws, i, 8, x.value, center=True)
        put(ws, i, 9, sx)
        put(ws, i, 10, f"{y.paper_id}  {titles.get(y.paper_id, '')}")
        put(ws, i, 11, py, center=True)
        put(ws, i, 12, "open B", link=arxiv_link(y.paper_id, py), center=True)
        put(ws, i, 13, y.value, center=True)
        put(ws, i, 14, sy)
        for c in range(15, len(C_HEADERS) + 1):
            put(ws, i, c, None, kind="input", center=c < len(C_HEADERS) and c != 21)
        fit_row(ws, i, [(sx, C_WIDTHS[8]), (sy, C_WIDTHS[13]), (f"{x.paper_id}  {titles.get(x.paper_id, '')}", C_WIDTHS[4])], minimum=70)
    last = len(pairs) + 1
    dropdown(ws, f"O2:O{last}", C_VERDICTS, title="Verdict", prompt="EXPLAINED, GENUINE or NOT COMPARABLE (see READ ME)")
    first_d, last_d = 16, 15 + len(C_DIFFERS)
    dropdown(ws, f"{col(first_d)}2:{col(last_d)}{last}", ["Y"], title="Differs?", prompt="Type Y for each condition that differs between the two papers")
    err = col(last_d + 2)
    dropdown(ws, f"{err}2:{err}{last}", ["Y"], title="Extraction error?", prompt="Y if one of the two numbers is not what the paper says")
    conf = col(last_d + 3)
    dropdown(ws, f"{conf}2:{conf}{last}", ["high", "low"], title="Confidence", prompt="how sure are you")
    highlight_done(ws, f"O2:O{last}", "O2")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{col(len(C_HEADERS))}{last}"
    protect(ws)
    progress = f'"Pairs done: "&COUNTA(Pairs!O2:O{last})&" of "&ROWS(Pairs!O2:O{last})'
    readme_sheet(wb, f"Job C - do these two papers really disagree? - {labeller}", C_README, example=C_EXAMPLE, progress=progress)
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
