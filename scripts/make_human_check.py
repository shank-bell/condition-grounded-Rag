"""Build the HUMAN SPOT-CHECK pack (14 Oct 2026): three Excel workbooks a person who did not write the AI keys fills in, blind to every AI / system verdict.

  python scripts/make_human_check.py            # -> data/labelling/human_check/  (git-ignored) + data/labelling/private/human_check_map.json

  H1  can the papers answer these 30 questions?        the 30 gold questions (11 covered / 19 need a warning), shuffled, ids Q01-Q30; a raw-text search aid
  H2  is this stored profile right?                    30 random profiles of the 100-profile clean sample E6 (final store, AI-labelled lenient / strict)
  H3  do these two papers really disagree?             20 of the 300 labelled pairs: all 4 GENUINE + 6 NOT COMPARABLE + 10 EXPLAINED, shuffled, ids X01-X20

The AI labels of the same items stay in the private files; scripts/score_human_check.py compares them with the human's after the workbooks come back.
The search aid in H1 is a plain text search of the 28 PDFs (words on the same page); it is not the system's store and not the AI's opinion.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import fitz

from cgrag.config import get_settings
from cgrag.labelling import score as S
from cgrag.labelling.sampling import Pair, arxiv_link, evidence_excerpt, load_chunk_index
from cgrag.labelling.xl import (F_BASE, F_BOLD, FILL_FIXED, dropdown, fit_row, header_row, highlight_done, new_workbook, protect, put,
                                readme_sheet)
from cgrag.stores.profile_store import ProfileStore

ROOT = Path(".")
DONE = ROOT / "data" / "labelling" / "done"
PRIVATE = ROOT / "data" / "labelling" / "private"
OUT = ROOT / "data" / "labelling" / "human_check"
PAPERS = ROOT / "data" / "papers"
LABELLER = "Shashank"

EXAMPLE_PAIR_VALUES = [{83.6, 79.2}, {82.3, 86.9}]

H1_README = [
    ("Why you are doing this", [
        "The system warns a user when the 28 papers do not cover what a question asks (for example a language, a model size or a setting that no paper reports "
        "together with the rest). The answer key for the 30 questions below was written by an AI. YOU are the independent check: for each question you decide, "
        "from the papers only, whether the question can be answered fully. Afterwards we count how often you and the AI agree.",
        "Honest answers matter more than agreement. If you think the AI is wrong, that is exactly what this check is for.",
    ]),
    ("What to do for each question (about 4 minutes)", [
        "1. Read the question and the grey column 'the question names'. Everything named there must be reported by a paper TOGETHER (same table or experiment) "
        "for the question to be fully answerable: the model, the data set (and its version), the language, the model size, the setting (5-shot, zero-shot ...).",
        "2. Open the sheet 'Search hits'. For each question it lists the pages where those words appear together in the PDFs. It is a plain text search, not the "
        "system's opinion: a hit only means the words are on that page, so you must look at the table. 'No hit' means the words do not appear together in "
        "this spelling; try other spellings with Ctrl+F in the paper.",
        "3. Open the paper at that page (sheet 'Papers': an online link and a local file) and read the table with its caption and footnotes.",
        "4. Pick YES or NO in the yellow column 'fully answerable?'. YES = at least one paper reports what the question asks (a number, or the fact) for ALL "
        "the named things together. NO = something named is not reported together with the rest. If the question has several parts and only some are "
        "reported, answer NO.",
        "5. If NO: write what is missing, like language=Kannada or model=GPT-4 (several: separate with commas). If YES: no need.",
        "6. 'evidence': paper id, page and table of the answer (YES), or where you looked and what you found instead (NO).",
        "7. 'how sure': high or low. Then the next question.",
        "For every NO, do not trust the search alone: open the one or two papers that are most likely to report it and read their results tables yourself "
        "(a table can use language codes such as hi / ta / kn, another spelling, or be an image). A NO means you looked.",
    ]),
    ("Rules of thumb (the same definitions the AI key used)", [
        "A different version, size, language or setting is NOT the same condition: SQuAD 1.1 is not 2.0, BERT-base is not BERT-large, English is not Hindi, "
        "zero-shot is not few-shot. A similar data set with another name is not the same data set.",
        "An average over many languages is not a result for one language. A sentence like 'we leave X for future work' is not a result. A result in an appendix "
        "table or a table in a different paper counts. A number you can only read from a plot does not.",
        "Name variants are the same thing: mBERT = multilingual BERT, XLM-R = XLM-RoBERTa, Llama / LLaMA.",
        "For questions that ask 'how does X work' or 'how many parameters does X have', YES means a paper in the set says it.",
        "Do not use the system or ask it. Do not paste the questions or the sheets into a chat tool. You may use Ctrl+F in the PDFs. If you cannot decide, "
        "write your best answer, set 'how sure' to low and say why in the note.",
    ]),
]
H1_EXAMPLE = (["question", "fully answerable?", "what is missing", "evidence", "how sure"],
              [["What F1 does BERT-large get on SQuAD v1.1?", "YES", "", "BERT paper 1810.04805, Table 2", "high"],
               ["What accuracy does FooBERT-large get on BarNLI for Swahili? (imaginary)", "NO", "model=FooBERT, dataset=BarNLI",
                "no paper contains either word", "high"]])

H2_README = [
    ("Why you are doing this", [
        "The system read the result tables of 28 papers and stored one 'profile' for every number: which model got which score on which data set, in which "
        "language and setting. Everything else in the system trusts these profiles. An AI also checked a sample of them; YOU are the independent check on "
        "30 of them. About 2 minutes per row.",
    ]),
    ("What to do for each row", [
        "1. Click the blue 'open' link (it opens the paper at the page the number came from). The grey 'source' column already shows the table caption, its "
        "header and the row, with the number marked [[like this]]; many rows can be checked from it, but look at the PDF if anything is unclear.",
        "2. Compare the grey cells (model, dataset, dataset_version, metric, value, language, task, model_size, setting) with the paper.",
        "3. Pick a verdict in the yellow 'verdict' column: ALL_OK (every filled cell is right; a cell the paper does not state may be empty), SOME_WRONG "
        "(a filled cell is wrong, or the paper clearly states something the profile left empty), NOT_A_RESULT (the row is not a reported result at all: a "
        "data set size, a count, a header ...), CANNOT_CHECK (you cannot verify it; say why).",
        "4. For SOME_WRONG mark WHICH cells: WRONG = filled in but wrong; MISSING = the paper states it but the cell is empty; VAGUE = true but less precise "
        "than the paper (for example setting 'fine-tuned' where the table block says 'translate-train-all', or a data set name without its subset). "
        "Cells you leave blank count as right. Write the correct value in the note.",
    ]),
    ("What each cell means", [
        "model: the system that got the score, as the table names it. dataset: the benchmark without its version. dataset_version: only if the paper "
        "states one. metric: what the number measures (accuracy, F1, EM, BLEU ...). value: the number - check it against the right ROW and COLUMN. "
        "language: the language of this number (a code like hi in a header must be the language name; 'Avg' is not a language). task: the kind of problem. "
        "model_size: only if stated (110M, 7B). setting: how it was evaluated (dev set, test set, zero-shot, fine-tuned, translate-train ...).",
    ]),
    ("Things that often go wrong (look for these)", [
        "The number belongs to a different column than the metric or language the profile names, especially when the same number appears twice in a row.",
        "The row label is a piece of a name ('L', 'MA') or the wrong model: a group label in the first column applies to several rows.",
        "A baseline's number is attributed to the paper's own model, or a dev number is recorded as a test number, or two stacked tables were mixed up.",
        "Do not use the system or ask it. Do not paste profiles or sheet contents into a chat tool. You may ask a chat tool to explain a table layout from "
        "pasted PDF text; the verdict is yours.",
    ]),
]
H2_EXAMPLE = (["model", "dataset", "metric", "value", "language", "verdict", "language wrong/missing?", "note / correct value"],
              [["FooBERT-large", "BarQA", "F1", "90.9", "(empty)", "ALL_OK", "", ""],
               ["FooBERT-large", "BazNLI", "accuracy", "71.5", "Hindi", "SOME_WRONG", "WRONG", "the number is in the Tamil column: language should be Tamil (imaginary)"]])

H3_README = [
    ("Why you are doing this", [
        "Two papers report different numbers for what looks like the same result (same model, data set and metric). The system has to tell a REAL conflict from a "
        "difference that different conditions EXPLAIN. An AI labelled 300 such pairs; YOU are the independent check on 20 of them. About 6 minutes per pair. "
        "You decide from the papers, not from the system.",
    ]),
    ("Your three verdicts", [
        "EXPLAINED - the numbers differ because the two papers ran the experiment under different conditions, so there is no real disagreement. Then type Y in "
        "each 'differs:' column that applies: dataset version / split (SQuAD 1.1 vs 2.0, dev vs test split), model size (base vs large, 340M vs 1.5B), "
        "language, setting (zero-shot vs fine-tuned, translate-train vs zero-shot, single model vs ensemble, dev vs test set), or other (say what). "
        "Several can differ at once.",
        "GENUINE - same conditions in both papers and still different numbers: a real disagreement. Be strict: say GENUINE only after you checked that the "
        "conditions really match in both papers.",
        "NOT COMPARABLE - the two numbers are not the same quantity (a different metric definition, a different task, a part of a bigger number, a "
        "human-performance row) or the papers do not say enough to compare them.",
    ]),
    ("What to do for each pair", [
        "1. Read both grey sources: caption, header and row, the number marked [[like this]]. Click 'open A' and 'open B' to see the papers at those pages. "
        "Read the paragraph that describes the experiment if the caption does not say how it was run.",
        "2. Pick the verdict and fill the yellow cells. 'confidence' = high or low. In 'explanation' quote the words of the paper that decided it "
        "(for example 'Table 2 is the dev set, Table 5 the test set').",
        "3. If one of the two numbers in the sheet is not what the paper says (the system read the table wrongly: wrong row, wrong column, a mean of two "
        "metrics), put Y in 'extraction error?' and judge the pair as well as you can.",
        "Do not use the system or ask it. Do not paste the pairs into a chat tool.",
    ]),
]
H3_EXAMPLE = (["pair", "verdict", "differs: setting", "other: what", "extraction error?", "confidence", "explanation"],
              [["X00", "EXPLAINED", "Y", "", "", "high", "Paper A Table 3 is zero-shot transfer; paper B Table 5 is translate-train (imaginary example)."]])

GUIDE = """HUMAN CHECK - HOW TO LABEL (read this first; 5 minutes)
=========================================================

What you are doing
  An AI wrote the answer keys the paper is scored against. You are the independent human who checks a sample of them, without seeing the AI's answers.
  Three Excel files, about 6 hours in total:

    H1_coverage_questions.xlsx   30 questions    ~4 min each  ~2 h   "can the 28 papers answer this question fully?"
    H3_result_pairs.xlsx         20 pairs        ~6 min each  ~2 h   "do these two papers really disagree, or do conditions explain it?"
    H2_profile_check.xlsx        30 profiles     ~2 min each  ~1 h   "is what the system stored for this number right?"
    + breaks and the search for hard cases                    ~1 h

  Do them in this order (H1 first: it teaches you the papers). Fill only the YELLOW cells. Save often (Ctrl+S). When you finish a file, save a copy named
  like the file with _DONE added (for example H1_coverage_questions_DONE.xlsx) and keep it in this folder.

The five rules
  1. Judge from the PAPERS, never from the system. Do not run the questions through the system, do not look at its answers.
  2. Do not paste sheet contents into a chat tool. (You may ask one to explain a table layout from pasted PDF text.)
  3. Do not look for 'the answer the AI would give'. Disagreeing with the AI is useful; a wrong agreement is not.
  4. If you cannot decide: write your best answer, mark 'how sure' low, say why in the note. Do not leave rows blank unless you really cannot judge.
  5. Work alone. Do not ask a teammate who has seen the AI's keys.

How to read a results table (2 minutes)
  - The CAPTION says what the table is (data set, split, setting). Read it first, and the footnotes under the table.
  - Rows are usually models, columns are metrics or languages or tasks. Several blocks in one table are separated by a heading row
    ("Dev set", "Test set", "Zero-shot", "Translate-train-all"): the heading applies to all rows below it until the next heading.
  - A number is only a result for the model in ITS ROW and the column header ABOVE it. When two numbers look equal, check the column again.
  - "Avg" or "Average" is not a language and not a data set.
  - Same name, different thing: SQuAD 1.1 and 2.0, BERT-base and BERT-large, a zero-shot row and a translate-train row, dev and test.

H1 - questions (the question names things; are they all reported TOGETHER by one paper?)
  YES   = at least one paper reports the asked result (or the asked fact) for everything the question names, together.
  NO    = something named is missing together with the rest (a language, a model, a size, a setting, a data set version), or only some parts of a
          multi-part question are reported. Then write what is missing, like  language=Kannada  or  model=GPT-4.
  Use the sheet 'Search hits' (a plain text search: where the words appear on the same page) to find the page, then LOOK at the table.
  Imaginary examples:
    "What F1 does BERT-large get on SQuAD v1.1?"                            YES  evidence: BERT paper, Table 2
    "What accuracy does FooBERT-large get on BarNLI for Swahili?"           NO   missing: model=FooBERT, dataset=BarNLI

H3 - pairs (same model, data set and metric, two papers, different numbers)
  EXPLAINED       the conditions differ (version/split, model size, language, setting, other): tick which with Y.
  GENUINE         the conditions are the same in both papers and the numbers still differ. Rare. Be strict.
  NOT COMPARABLE  not the same quantity, or the papers do not say enough.
  'extraction error? = Y' when one of the two numbers shown is not what the paper says.
  Real examples from the literature: human EM on SQuAD 82.3 vs 86.9 -> EXPLAINED (v1.1 vs v2.0). An average XNLI accuracy of 83.6 (translate-train-all)
  vs 79.2 (zero-shot) -> EXPLAINED (setting).

H2 - profiles (the system stored: model | data set | metric | value | language | setting ...; is it what the table says?)
  ALL_OK        every filled cell is right (empty cells are fine when the paper does not state them)
  SOME_WRONG    a filled cell is wrong or the paper clearly states something the cell left empty -> mark the cell WRONG / MISSING / VAGUE
  NOT_A_RESULT  the row is not a reported result (a count, a size, a header)
  CANNOT_CHECK  you cannot verify it (say why)

When you are done
  Tell the AI assistant "done" and say which files. The AI assistant then scores your labels against the AI's and the system's: agreement and kappa (a number that says how
  much two labellers agree beyond chance; 0.6 and above is good), a confusion table, and the places where you disagree so we can look at them together.
  The paper will then say what was human-checked and how big the sample was.
"""

TERM_ALIASES = {
    "BERT": ["bert"], "BERT-base": ["bert-base", "bert base", "bertbase"], "BERT-large": ["bert-large", "bert large", "bertlarge"], "DistilBERT": ["distilbert"],
    "RoBERTa": ["roberta"], "ELECTRA": ["electra"], "XLM-R": ["xlm-r", "xlmr", "xlm-roberta", "xlm roberta"], "mBERT": ["mbert", "multilingual bert", "m-bert"],
    "MuRIL": ["muril"], "mT5-Large": ["mt5-large", "mt5 large"], "mT5-XL": ["mt5-xl", "mt5 xl"], "mT5-XXL": ["mt5-xxl", "mt5 xxl"],
    "GPT-3": ["gpt-3", "gpt3"], "GPT-4": ["gpt-4", "gpt4"], "LLaMA": ["llama"], "Llama 2-Chat": ["llama 2-chat", "llama-2-chat", "llama 2 chat", "llama2-chat"],
    "Mistral": ["mistral"], "Qwen": ["qwen"],
    "SQuAD": ["squad"], "XNLI": ["xnli"], "MLQA": ["mlqa"], "MNLI": ["mnli", "multinli"], "SST-2": ["sst-2", "sst2"], "XQuAD": ["xquad"], "Tatoeba": ["tatoeba"],
    "IndicCOPA": ["indiccopa"], "IndicXParaphrase": ["indicxparaphrase", "indic xparaphrase"], "TyDi QA GoldP": ["tydi qa", "tydiqa", "goldp"],
    "NaturalQuestions": ["naturalquestions", "natural questions"], "HumanEval": ["humaneval", "human eval"], "MMLU": ["mmlu"], "HellaSwag": ["hellaswag"],
    "TruthfulQA": ["truthfulqa", "truthful qa"],
    "5-shot": ["5-shot", "5 shot", "five-shot"], "zero-shot": ["zero-shot", "zero shot"], "few-shot": ["few-shot", "few shot"],
    "Hindi": ["hindi"], "Tamil": ["tamil"], "Kannada": ["kannada"], "Swahili": ["swahili"], "German": ["german"], "English": ["english"],
}
LANG_CODES = {"Hindi": "hi", "Tamil": "ta", "Kannada": "kn", "Swahili": "sw", "German": "de", "English": "en"}
VERSION_FORMS = {("SQuAD", "2.0"): ["squad 2.0", "squad v2.0", "squad2.0", "squadv2", "squad v2"], ("SQuAD", "1.1"): ["squad 1.1", "squad v1.1", "squad1.1", "squadv1.1"]}


def norm_text(s: str) -> str:
    return s.replace("‑", "-").replace("‐", "-").replace("−", "-").replace("­", "").lower()


def alias_regex(alias: str) -> re.Pattern:
    tokens = [t for t in re.split(r"[\s\-]+", alias.lower().strip()) if t]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-]?".join(re.escape(t) for t in tokens) + r"(?![a-z0-9])")


class Term:
    def __init__(self, label: str, aliases: list[str], code: str | None = None) -> None:
        self.label = label
        self.patterns = [alias_regex(a) for a in aliases]
        self.code = re.compile(r"(?<![A-Za-z])" + re.escape(code) + r"(?![A-Za-z])") if code else None

    def count(self, page_text_norm: str, page_text_raw: str) -> int:
        n = sum(len(p.findall(page_text_norm)) for p in self.patterns)
        if self.code is not None:
            n += len(self.code.findall(page_text_raw))
        return n


def make_term(kind: str, value: str, version: str | None = None) -> Term:
    if kind == "dataset" and version:
        forms = VERSION_FORMS.get((value, version)) or [f"{value} {version}", f"{value} v{version}", f"{value}{version}"]
        return Term(f"{value} {version}", forms)
    aliases = TERM_ALIASES.get(value) or [value.lower()]
    return Term(value, aliases, LANG_CODES.get(value) if kind == "language" else None)


def load_pages() -> dict[tuple[str, int], tuple[str, str]]:
    name_map = json.loads((PRIVATE / "paper_name_map.json").read_text(encoding="utf-8"))
    pages: dict[tuple[str, int], tuple[str, str]] = {}
    for pdf in sorted(PAPERS.glob("*.pdf")):
        arxiv = name_map.get(pdf.stem, {}).get("arxiv", pdf.stem)
        with fitz.open(pdf) as doc:
            for i, page in enumerate(doc, start=1):
                raw = page.get_text()
                pages[(arxiv, i)] = (norm_text(raw), raw)
    return pages


def combos_for(q: dict) -> list[tuple[list[Term], list[Term]]]:
    """(required terms, optional terms) for every model x data set x language combination the question names."""
    c = q["conditions"]
    version = (c.get("dataset_version") or [None])[0]
    models = [make_term("model", m) for m in c.get("model", [])] or [None]
    datasets = [make_term("dataset", d, version if len(c.get("dataset", [])) == 1 else None) for d in c.get("dataset", [])] or [None]
    langs = [make_term("language", g) for g in c.get("language", [])] or [None]
    settings = [make_term("setting", s) for s in c.get("setting", [])]
    optional = [make_term("size", s) for s in c.get("model_size", [])]
    out = []
    for m, d, g in product(models, datasets, langs):
        out.append(([t for t in (m, d, g) if t is not None] + settings, optional))
    return out


def fmt_pages(hits: list[tuple[str, int]], limit: int = 10) -> str:
    by: dict[str, list[int]] = defaultdict(list)
    for arxiv, page in hits[:limit]:
        by[arxiv].append(page)
    return "; ".join(f"{a} p{','.join(str(p) for p in sorted(ps))}" for a, ps in by.items())


def search_rows(q: dict, qid: str, pages: dict[tuple[str, int], tuple[str, str]]) -> list[list[str]]:
    rows = []
    for required, optional in combos_for(q):
        label = " + ".join(t.label for t in required) or "(no named term)"
        scored = []
        for key, (norm, raw) in pages.items():
            counts = [t.count(norm, raw) for t in required]
            present = sum(1 for n in counts if n)
            scored.append((key, present, sum(counts), counts))
        full = sorted([s for s in scored if s[1] == len(required) and required], key=lambda s: -s[2])
        extras = ""
        if full and optional:
            with_size = [s for s in full if all(t.count(*pages[s[0]]) for t in optional)]
            extras = f" | with {', '.join(t.label for t in optional)} on the same page: {fmt_pages([s[0] for s in with_size], 6) or 'none'}"
        if full:
            rows.append([qid, q["question"], label, fmt_pages([s[0] for s in full]) + f"  ({len(full)} pages in all)" + extras, ""])
        else:
            near = sorted([s for s in scored if s[1] >= max(1, len(required) - 1) and s[1] >= 1], key=lambda s: (-s[1], -s[2]))[:6]
            lines = []
            for key, present, _, counts in near:
                missing = [t.label for t, n in zip(required, counts) if not n]
                lines.append(f"{key[0]} p{key[1]} (missing: {', '.join(missing)})")
            rows.append([qid, q["question"], label, "NO page has all these words", "; ".join(lines) if lines else "(no page has even most of the words)"])
    return rows


def build_h1(path: Path, questions: list[tuple[str, dict]], pages, titles: dict[str, str], pdf_of: dict[str, str]) -> None:
    wb = new_workbook()
    ws = wb.create_sheet("Questions")
    headers = ["#", "id", "question", "the question names", "fully answerable from the papers?", "what is missing (condition=value, ...)",
               "evidence: paper id, page, table (or where you looked)", "how sure", "note"]
    widths = [4, 6, 58, 34, 16, 30, 46, 10, 30]
    header_row(ws, headers, widths)
    for i, (qid, q) in enumerate(questions, start=2):
        c = q["conditions"]
        names = "; ".join(f"{k.replace('_', ' ')}: {', '.join(v)}" for k, v in c.items() if v)
        put(ws, i, 1, i - 1, center=True)
        put(ws, i, 2, qid, center=True)
        put(ws, i, 3, q["question"])
        put(ws, i, 4, names)
        for col_i in range(5, 10):
            put(ws, i, col_i, None, kind="input", center=col_i in (5, 8))
        fit_row(ws, i, [(q["question"], widths[2]), (names, widths[3])], minimum=48)
    last = len(questions) + 1
    dropdown(ws, f"E2:E{last}", ["YES", "NO"], title="Fully answerable?", prompt="YES = one paper reports ALL the named things together; NO = something is missing")
    dropdown(ws, f"H2:H{last}", ["high", "low"], title="How sure", prompt="high or low")
    highlight_done(ws, f"E2:E{last}", "E2")
    ws.freeze_panes = "D2"
    protect(ws)
    sh = wb.create_sheet("Search hits")
    sh_headers = ["id", "question", "words searched together", "pages where ALL these words appear (paper id, pages)", "closest pages when none has all of them"]
    sh_widths = [6, 46, 26, 70, 60]
    header_row(sh, sh_headers, sh_widths)
    r = 2
    for qid, q in questions:
        for row in search_rows(q, qid, pages):
            for c_i, v in enumerate(row, start=1):
                put(sh, r, c_i, v, kind="plain", center=c_i == 1)
            fit_row(sh, r, [(row[1], sh_widths[1]), (row[3], sh_widths[3]), (row[4], sh_widths[4])], minimum=30)
            r += 1
    sh.freeze_panes = "C2"
    sh.auto_filter.ref = f"A1:E{r - 1}"
    protect(sh)
    pp = wb.create_sheet("Papers")
    p_headers = ["arXiv id", "title", "open online", "local file (this PC)"]
    p_widths = [14, 90, 14, 52]
    header_row(pp, p_headers, p_widths)
    for i, (arxiv, title) in enumerate(sorted(titles.items()), start=2):
        put(pp, i, 1, arxiv, kind="plain")
        put(pp, i, 2, title, kind="plain")
        put(pp, i, 3, "open", kind="plain", link=arxiv_link(arxiv, 1), center=True)
        local = pdf_of.get(arxiv)
        put(pp, i, 4, (f"{Path(local).name}  (folder data\\papers)" if local else ""), kind="plain",
            link=(Path(local).resolve().as_uri() if local else None))
    pp.freeze_panes = "A2"
    protect(pp)
    progress = f'"Questions done: "&COUNTA(Questions!E2:E{last})&" of "&ROWS(Questions!E2:E{last})'
    readme_sheet(wb, "Human check 1 of 3 - can the papers answer these 30 questions?", H1_README, example=H1_EXAMPLE, progress=progress)
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


A_FIELDS = ["model", "dataset", "dataset_version", "metric", "value", "language", "task", "model_size", "setting"]
A_FLAGS = ["WRONG", "MISSING", "VAGUE"]


def build_h2(path: Path, rows: list, texts: dict[str, str], pages: dict[str, int], titles: dict[str, str]) -> None:
    wb = new_workbook()
    ws = wb.create_sheet("Profiles")
    fixed = ["#", "id", "paper", "page", "open"]
    headers = [*fixed, *A_FIELDS, "source (caption, header, the row the number came from)", "verdict", *[f"{f} wrong/missing/vague?" for f in A_FIELDS],
               "note / correct value"]
    widths = [4, 6, 30, 6, 8, 17, 14, 9, 12, 8, 11, 16, 9, 13, 62, 14, *([11] * len(A_FIELDS)), 34]
    header_row(ws, headers, widths)
    n_fix, n_f = len(fixed), len(A_FIELDS)
    source_c, verdict_c = n_fix + n_f + 1, n_fix + n_f + 2
    flag0_c, note_c = verdict_c + 1, verdict_c + 1 + n_f
    for i, (rid, p) in enumerate(rows, start=2):
        excerpt = evidence_excerpt(texts.get(p.chunk_id, ""), p)
        title = f"{p.paper_id}  {titles.get(p.paper_id, '')}"
        page = pages.get(p.chunk_id, 1)
        put(ws, i, 1, i - 1, center=True)
        put(ws, i, 2, rid, center=True)
        put(ws, i, 3, title)
        put(ws, i, 4, page, center=True)
        put(ws, i, 5, "open", link=arxiv_link(p.paper_id, page), center=True)
        for k, f in enumerate(A_FIELDS):
            v = getattr(p, f)
            put(ws, i, n_fix + 1 + k, v if v not in ("", None) else None)
        put(ws, i, source_c, excerpt)
        put(ws, i, verdict_c, None, kind="input", center=True)
        for k in range(n_f):
            put(ws, i, flag0_c + k, None, kind="input", center=True)
        put(ws, i, note_c, None, kind="input")
        fit_row(ws, i, [(excerpt, widths[source_c - 1]), (title, widths[2])])
    last = len(rows) + 1
    vl = chr(64 + verdict_c) if verdict_c <= 26 else "P"
    from cgrag.labelling.xl import col as colname
    vl = colname(verdict_c)
    dropdown(ws, f"{vl}2:{vl}{last}", ["ALL_OK", "SOME_WRONG", "NOT_A_RESULT", "CANNOT_CHECK"], title="Verdict",
             prompt="ALL_OK, SOME_WRONG, NOT_A_RESULT or CANNOT_CHECK")
    dropdown(ws, f"{colname(flag0_c)}2:{colname(flag0_c + n_f - 1)}{last}", A_FLAGS, title="Which cell?",
             prompt="Only for SOME_WRONG: WRONG = filled but wrong; MISSING = the paper states it, the cell is empty; VAGUE = true but less precise than the paper")
    highlight_done(ws, f"{vl}2:{vl}{last}", f"{vl}2")
    ws.freeze_panes = ws.cell(row=2, column=n_fix + 1)
    ws.auto_filter.ref = f"A1:{colname(len(headers))}{last}"
    protect(ws)
    progress = f'"Rows done: "&COUNTA(Profiles!{vl}2:{vl}{last})&" of "&ROWS(Profiles!{vl}2:{vl}{last})'
    readme_sheet(wb, "Human check 2 of 3 - is what the system stored right? (30 stored results)", H2_README, example=H2_EXAMPLE, progress=progress)
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def build_h3(path: Path, pairs: list[tuple[str, Pair]], texts: dict[str, str], pages: dict[str, int], titles: dict[str, str]) -> None:
    from cgrag.labelling.sheets import C_DIFFERS, C_HEADERS, C_VERDICTS, C_WIDTHS, _model_cell
    from cgrag.labelling.xl import col as colname
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
    dropdown(ws, f"{colname(first_d)}2:{colname(last_d)}{last}", ["Y"], title="Differs?", prompt="Type Y for each condition that differs between the two papers")
    err = colname(last_d + 2)
    dropdown(ws, f"{err}2:{err}{last}", ["Y"], title="Extraction error?", prompt="Y if one of the two numbers is not what the paper says")
    conf = colname(last_d + 3)
    dropdown(ws, f"{conf}2:{conf}{last}", ["high", "low"], title="Confidence", prompt="how sure are you")
    highlight_done(ws, f"O2:O{last}", "O2")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{colname(len(C_HEADERS))}{last}"
    protect(ws)
    progress = f'"Pairs done: "&COUNTA(Pairs!O2:O{last})&" of "&ROWS(Pairs!O2:O{last})'
    readme_sheet(wb, "Human check 3 of 3 - do these two papers really disagree?", H3_README, example=H3_EXAMPLE, progress=progress)
    wb.active = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def all_pair_gold() -> list[dict]:
    """Every labelled pair of the six sets with its AI gold verdict and the two profile ids (from the private key files)."""
    def first_only(fname: str) -> dict:
        first, _ = S.parse_job_c(DONE / fname)
        return {p: {"verdict": v["verdict"], "differs": v["differs"], "extraction_error": v["extraction_error"]} for p, v in first.items()}

    def settled(a: str, b: str, c: str) -> dict:
        pa, _ = S.parse_job_c(DONE / a)
        pb, _ = S.parse_job_c(DONE / b)
        pc, _ = S.parse_job_c(DONE / c)
        return S.gold_job_c(pa, pb, pc)

    sets = {
        "first": (settled("C_result_pairs_Aditya_AIannotated-pass1_DONE.xlsx", "C_result_pairs_Tarun_AIannotated-second_ai_DONE.xlsx",
                          "C_result_pairs_Tarun_AIannotated-third_DONE.xlsx"), "job_C_key.json"),
        "heldout": (settled("C_heldout_pairs_first_AIannotated-heldout-first_DONE.xlsx", "C_heldout_pairs_second_AIannotated-heldout-second_ai_DONE.xlsx",
                            "C_heldout_pairs_first_AIannotated-heldout-third_DONE.xlsx"), "job_C_heldout_key.json"),
        "fresh": (first_only("C_fresh_pairs_first_AIannotated-fresh-first_DONE.xlsx"), "job_C_fresh_key.json"),
        "fresh2": (first_only("C_fresh2_pairs_first_AIannotated-fresh2-first_DONE.xlsx"), "job_C_fresh2_key.json"),
        "fresh3": (first_only("C_fresh3_pairs_first_AIannotated-fresh3-first_DONE.xlsx"), "job_C_fresh3_key.json"),
        "fresh4": (first_only("C_fresh4_pairs_first_AIannotated-fresh4-first_DONE.xlsx"), "job_C_fresh4_key.json"),
    }
    out = []
    for name, (gold, keyfile) in sets.items():
        key = json.loads((PRIVATE / keyfile).read_text(encoding="utf-8"))
        for pid, g in gold.items():
            k = key[pid]
            out.append({"set": name, "pair": pid, "verdict": str(g["verdict"]).replace(" ", "_"), "differs": g.get("differs"),
                        "extraction_error": g.get("extraction_error"), "a": k["profile_a"], "b": k["profile_b"], "va": k.get("value_a"), "vb": k.get("value_b")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1410)
    ap.add_argument("--profiles", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    cfg = get_settings()
    store = ProfileStore(cfg.paths.profile_db, repaired=True)
    profile_by_id = {p.profile_id: p for p in store.all()}
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    name_map = json.loads((PRIVATE / "paper_name_map.json").read_text(encoding="utf-8"))
    titles = {v["arxiv"]: " ".join(v["title"].split()) for v in name_map.values()}
    pdf_of = {v["arxiv"]: str((PAPERS / f"{k}.pdf")) for k, v in name_map.items() if (PAPERS / f"{k}.pdf").exists()}
    mapping: dict = {"seed": args.seed, "questions": {}, "profiles": {}, "pairs": {}}

    questions = [json.loads(line) for line in Path("eval/labels/questions_gold.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    order = list(range(len(questions)))
    rnd.shuffle(order)
    h1 = []
    for n, idx in enumerate(order, start=1):
        qid = f"Q{n:02d}"
        h1.append((qid, questions[idx]))
        mapping["questions"][qid] = {"id": questions[idx]["id"], "question": questions[idx]["question"], "ai_expect_warning": questions[idx]["expect_warning"],
                                     "ai_missing": questions[idx]["missing"]}
    print("loading page texts of the PDFs ...")
    pages_text = load_pages()
    build_h1(args.out / "H1_coverage_questions.xlsx", h1, pages_text, titles, pdf_of)
    print(f"H1: 30 questions, {len(pages_text)} PDF pages searched")

    items = [json.loads(line) for line in (PRIVATE / "extraction_test6" / "items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [it for it in items if it["profile_id"] in profile_by_id]
    pick = rnd.sample(items, args.profiles)
    e6_labels = {r["id"]: r for r in (json.loads(line) for line in (PRIVATE / "extraction_test6" / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = []
    for n, it in enumerate(pick, start=1):
        rid = f"R{n:02d}"
        rows.append((rid, profile_by_id[it["profile_id"]]))
        lab = e6_labels.get(it["id"], {})
        mapping["profiles"][rid] = {"profile_id": it["profile_id"], "e6_id": it["id"], "ai_wrong": lab.get("wrong", []), "ai_missing": lab.get("missing", []),
                                    "ai_imprecise": lab.get("imprecise", []), "ai_labelled": it["id"] in e6_labels}
    build_h2(args.out / "H2_profile_check.xlsx", rows, index.texts, index.pages, {**titles, **index.titles})
    print(f"H2: {len(rows)} profiles from {len({p.paper_id for _, p in rows})} papers")

    gold = [g for g in all_pair_gold() if g["a"] in profile_by_id and g["b"] in profile_by_id]
    gold = [g for g in gold if {g["va"], g["vb"]} not in EXAMPLE_PAIR_VALUES]
    by = defaultdict(list)
    for g in gold:
        by[g["verdict"]].append(g)
    print("labelled pairs available:", {k: len(v) for k, v in by.items()})
    chosen = list(by["GENUINE"]) + rnd.sample(by["NOT_COMPARABLE"], 6) + rnd.sample(by["EXPLAINED"], 10)
    rnd.shuffle(chosen)
    pairs = []
    for n, g in enumerate(chosen, start=1):
        xid = f"X{n:02d}"
        pairs.append((xid, Pair(profile_by_id[g["a"]], profile_by_id[g["b"]], "", [], "", 0.0)))
        mapping["pairs"][xid] = {"set": g["set"], "pair": g["pair"], "profile_a": g["a"], "profile_b": g["b"], "ai_verdict": g["verdict"],
                                 "ai_differs": g["differs"], "ai_extraction_error": g["extraction_error"]}
    build_h3(args.out / "H3_result_pairs.xlsx", pairs, index.texts, index.pages, {**titles, **index.titles})
    print(f"H3: {len(pairs)} pairs")

    (args.out / "GUIDE_how_to_label.txt").write_text(GUIDE, encoding="utf-8")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "human_check_map.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False), encoding="utf-8")

    per_paper: Counter = Counter()
    for _, p in rows:
        per_paper[p.paper_id] += 1
    pair_papers: Counter = Counter()
    for _, pr in pairs:
        pair_papers[pr.x.paper_id] += 1
        pair_papers[pr.y.paper_id] += 1
    summary = {"profiles_per_paper": dict(per_paper), "pair_sides_per_paper": dict(pair_papers)}
    (args.out / "papers_you_will_read.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("files in", args.out)


if __name__ == "__main__":
    main()
