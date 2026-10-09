"""Put the JSON lines of ONE AI labeller (answers to the pair prompt) into a copy of a blank job-C workbook and record where they came from.

Each answer line:  {"pair": "P07", "verdict": "EXPLAINED", "differs": ["model size"], "other": "", "extraction_error": "N", "confidence": "high", "explanation": "..."}
Lines starting with # and blank lines are ignored. Every run starts from the untouched blank workbook, so it can be repeated when more batches arrive.

  python scripts/fill_job_c_from_ai.py --answers data/labelling/ai_fiesta/answers_gemini_b*.jsonl --model "Gemini 3.1 Pro (web app)" --date "8 Oct 2026" --pdfs unknown --tag gemini31pro

Output: data/labelling/done/<sheet name>_AIannotated-<tag>_DONE.xlsx (+ .provenance.json). The READ ME sheet says in red that the labels are AI-made.
Pairs without an answer stay empty (the scorer works on the pairs both labellers did)."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

from cgrag.config import ROOT
from cgrag.labelling import score as S

LAB = ROOT / "data" / "labelling"
VERDICTS = {"EXPLAINED", "GENUINE", "NOT COMPARABLE"}
DIFFERS = {"dataset version / split": "differs: dataset version / split", "model size": "differs: model size", "language": "differs: language",
           "setting": "differs: setting", "other": "differs: other"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", required=True, help="answer files (JSON lines); wildcards are expanded")
    ap.add_argument("--sheet", type=Path, default=LAB / "C_result_pairs_Tarun.xlsx", help="the untouched blank workbook")
    ap.add_argument("--model", required=True, help="exact model name and version of the labeller")
    ap.add_argument("--date", required=True)
    ap.add_argument("--pdfs", default="unknown", help="were the PDFs attached to the chat: yes / no / unknown")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--role", default="SECOND PASS", help="written into the title of the READ ME sheet (e.g. 'THIRD LOOK')")
    ap.add_argument("--note", default="", help="replaces the default READ ME note (use it when the labeller was NOT blind)")
    args = ap.parse_args()

    files = sorted({Path(p) for pat in args.answers for p in glob.glob(pat)})
    answers, problems = {}, []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{f.name} line {i}: not JSON ({e.msg})")
                continue
            pid = str(r.get("pair", "")).strip()
            verdict = str(r.get("verdict", "")).strip().upper()
            bad = [d for d in r.get("differs", []) if d not in DIFFERS]
            if verdict not in VERDICTS or bad:
                problems.append(f"{f.name} {pid}: verdict {verdict!r} / differs {bad}")
                continue
            if pid in answers:
                problems.append(f"{pid} answered twice (the later one wins)")
            answers[pid] = {**r, "verdict": verdict}

    wb = load_workbook(args.sheet)
    ws = wb["Pairs"]
    hdr = {str(c.value).strip(): i for i, c in enumerate(ws[1], start=1) if c.value is not None}
    need = ["pair", "verdict", *DIFFERS.values(), "other: what", "extraction error?", "confidence", "explanation (quote the paper)"]
    missing_cols = [c for c in need if c not in hdr]
    if missing_cols:
        raise SystemExit(f"columns not found in the sheet: {missing_cols}")
    done, unknown = [], [p for p in answers if p not in {str(ws.cell(r, hdr['pair']).value) for r in range(2, ws.max_row + 1)}]
    for r in range(2, ws.max_row + 1):
        pid = str(ws.cell(r, hdr["pair"]).value)
        a = answers.get(pid)
        if not a:
            continue
        ws.cell(r, hdr["verdict"]).value = a["verdict"]
        for name, column in DIFFERS.items():
            ws.cell(r, hdr[column]).value = "Y" if name in a.get("differs", []) else None
        ws.cell(r, hdr["other: what"]).value = (a.get("other") or None) if "other" in a.get("differs", []) else None
        ws.cell(r, hdr["extraction error?"]).value = "Y" if str(a.get("extraction_error", "N")).upper() == "Y" else None
        ws.cell(r, hdr["confidence"]).value = a.get("confidence")
        ws.cell(r, hdr["explanation (quote the paper)"]).value = a.get("explanation")
        done.append(pid)
    all_ids = [str(ws.cell(r, hdr["pair"]).value) for r in range(2, ws.max_row + 1)]
    left = [p for p in all_ids if p not in done]

    rd = wb["READ ME"]
    rd["B1"] = f"{rd['B1'].value or ''} - AI-ANNOTATED, {args.role} by {args.model} (not a human)"
    rd["B1"].font = Font(bold=True, color="C00000", size=rd["B1"].font.size or 12)
    rd["B3"] = args.note or (f"Filled on {args.date} from the pasted answers of an AI chat model ({args.model}); PDFs attached to the chat: {args.pdfs}. The model got only the text of the BLANK pair sheet and a written task "
                             f"(instructions + 5 pairs per message); it never saw the first-pass labels or the system's verdicts. {len(done)} of {len(all_ids)} pairs are labelled; the others are empty"
                             f"{' (' + ', '.join(left) + ')' if left else ''}. Not written by the person named in the file name.")
    rd["B3"].font = Font(color="C00000", bold=True)
    rd["B3"].alignment = Alignment(wrap_text=True, vertical="top")
    out = LAB / "done" / f"{args.sheet.stem}_AIannotated-{args.tag}_DONE.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    (out.with_suffix(".provenance.json")).write_text(json.dumps({
        "workbook": out.name, "template": args.sheet.name, "labeller": args.model, "date": args.date, "pdfs_attached": args.pdfs, "written_at": datetime.now().isoformat(timespec="seconds"),
        "pairs_labelled": len(done), "pairs_empty": left, "answer_files": {f.name: sha(f) for f in files}, "problems": problems,
        "independence": "the labeller never saw the first-pass labels, the system's verdicts or the architecture document; one AI model, not a human"}, indent=1), encoding="utf-8")
    parsed, p2 = S.parse_job_c(out)
    print(f"{len(done)} of {len(all_ids)} pairs written to {out.name}; empty: {left}; unknown pair ids in the answers: {unknown or 'none'}; answer problems: {problems or 'none'}")
    print(f"the scorer reads {len(parsed)} labelled pairs from it; problems: {p2 or 'none'}")


if __name__ == "__main__":
    main()
