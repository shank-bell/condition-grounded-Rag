"""Score the filled-in labelling workbooks (see docs/labelling_guide.md).

  python scripts/score_labels.py status data/labelling/*.xlsx          # how far is everybody?
  python scripts/score_labels.py A <A_*_DONE.xlsx ...>                  # field-level F1 of the profile extractor
  python scripts/score_labels.py B <B_*_DONE.xlsx ...>                  # the gold question file for scripts/eval_questions.py
  python scripts/score_labels.py C --first C_Aditya_DONE.xlsx --second C_Tarun_DONE.xlsx [--third C_adjudication_DONE.xlsx]
        [--make-adjudication data/labelling/C_adjudication.xlsx]        # kappa, the pairs to settle, and the system's accuracy vs the gold

Results go to eval/labels/ (small JSON files; commit them after the labelling day, they are the paper's answer key).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openpyxl import load_workbook

from cgrag.config import ROOT, get_settings
from cgrag.labelling import score as S
from cgrag.labelling.sheets import A_FIELDS

OUT = ROOT / "eval" / "labels"
PRIVATE = ROOT / "data" / "labelling" / "private"


def _pct(v) -> str:
    return "  -  " if v is None else f"{100 * v:5.1f}"


def status(paths: list[Path]) -> None:
    for path in paths:
        if path.name.startswith("~$") or path.suffix != ".xlsx":
            continue
        sheet, column = {"A": ("Profiles", "verdict"), "B": ("Questions", "question"), "C": ("Pairs", "verdict")}.get(path.name[:1], (None, None))
        if not sheet:
            continue
        rows = S.read_rows(path, sheet)
        done = sum(1 for r in rows if S._clean(r.get(column)))
        print(f"{path.name:<44} {done:>4} / {len(rows):<4} {'done' if done == len(rows) else ''}")


def job_a(paths: list[Path], out: Path) -> None:
    rows, problems = [], []
    for p in paths:
        r, pr = S.parse_job_a(p)
        rows += r
        problems += pr
    assignment = PRIVATE / "job_A_assignment.json"
    group_of = json.loads(assignment.read_text(encoding="utf-8"))["group_of_paper"] if assignment.exists() else None
    scores = S.score_job_a(rows, group_of)
    scores["agreement_of_two_labellers"] = S.agreement_job_a(rows)
    scores["problems"] = problems
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in problems:
        print("PROBLEM", p)
    for name, block in scores.items():
        if not isinstance(block, dict) or "per_field" not in block:
            continue
        print(f"\n== {name}: {block['profiles_labelled']} profiles {block['verdicts']}")
        print(f"   profile-level precision {_pct(block['profile_precision'])} %   fully correct (of real results) {_pct(block['fully_correct_of_results'])} %")
        print(f"   {'field':<16}{'precision':>10}{'recall':>8}{'F1':>7}   tp  fp  fn")
        for f in A_FIELDS:
            v = block["per_field"][f]
            print(f"   {f:<16}{_pct(v['precision']):>10}{_pct(v['recall']):>8}{_pct(v['f1']):>7}  {v['tp']:>3} {v['fp']:>3} {v['fn']:>3}")
        m = block["micro"]
        print(f"   {'ALL (micro)':<16}{_pct(m['precision']):>10}{_pct(m['recall']):>8}{_pct(m['f1']):>7}  {m['tp']:>3} {m['fp']:>3} {m['fn']:>3}")
    print("\nagreement of the two labellers on the shared profiles:", scores["agreement_of_two_labellers"])
    print("wrote", out)


def job_b(paths: list[Path], out: Path) -> None:
    questions, problems = [], []
    for p in paths:
        q, pr = S.parse_job_b(p)
        questions += q
        problems += pr
    for p in problems:
        print("PROBLEM", p)
    S.write_jsonl(questions, out)
    from collections import Counter
    print(f"{len(questions)} questions -> {out}")
    print("planned types:", dict(Counter(q["planned_type"] for q in questions)), "| intents:", dict(Counter(q["intent"] for q in questions)),
          "| complexity:", dict(Counter(q["complexity"] for q in questions)), "| warning expected:", sum(q["expect_warning"] for q in questions))


def job_c(args) -> None:
    first, p1 = S.parse_job_c(args.first)
    second, p2 = S.parse_job_c(args.second)
    third, p3 = S.parse_job_c(args.third) if args.third else (None, [])
    for p in [*p1, *p2, *p3]:
        print("PROBLEM", p)
    agreement = S.agreement_job_c(first, second)
    print("agreement:", {k: v for k, v in agreement.items() if k != "disagreements"})
    print(f"kappa {agreement.get('kappa')} (target >= 0.6)  -> {'MET' if agreement.get('meets_target_0.6') else 'NOT met'}"
          if agreement.get("pairs_labelled_by_both") else "no pair was labelled by both")
    result = {"agreement": agreement, "problems": [*p1, *p2, *p3]}
    gold = S.gold_job_c(first, second, third)
    unsettled = [p for p in agreement.get("disagreements", []) if p not in gold]
    if unsettled:
        print(f"{len(unsettled)} pair(s) still need a third labeller: {', '.join(unsettled)}")
    if args.make_adjudication and unsettled:
        make_adjudication(args.make_adjudication, agreement["disagreements"], Path(args.key))
    result["gold"] = gold
    if args.key and Path(args.key).exists() and gold:
        key = json.loads(Path(args.key).read_text(encoding="utf-8"))
        result["system"] = S.score_system_c(gold, key)
        s = result["system"]
        print(f"\nthe system's verdict vs the gold ({s['pairs']} pairs): accuracy {_pct(s['accuracy'])} %, macro-F1 {_pct(s['macro_f1'])} %")
        for c, v in s["per_class"].items():
            print(f"   {c:<15} precision {_pct(v['precision'])}  recall {_pct(v['recall'])}  F1 {_pct(v['f1'])}  (gold support {v['support']})")
        print("   confusion (rows = gold, columns = system):", json.dumps(s["confusion_gold_rows_system_columns"]))
        print("   which condition explains it:", s["condition_attribution"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


def make_adjudication(path: Path, pair_ids: list[str], key_path: Path) -> None:
    """A workbook with only the pairs the two labellers disagree on, for the third labeller (blind: no earlier verdict is shown)."""
    from cgrag.labelling.sampling import Pair, load_chunk_index
    from cgrag.labelling.sheets import build_job_c
    from cgrag.stores.profile_store import ProfileStore
    cfg = get_settings()
    key = json.loads(key_path.read_text(encoding="utf-8"))
    by_id = {p.profile_id: p for p in ProfileStore(cfg.paths.profile_db).all()}
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    pairs = [(pid, Pair(by_id[key[pid]["profile_a"]], by_id[key[pid]["profile_b"]], key[pid]["system_verdict"], key[pid]["system_differing"],
                        key[pid]["reason"], key[pid]["rel_diff"])) for pid in pair_ids if pid in key]
    build_job_c(path, "adjudication", pairs, index.texts, index.pages, index.titles)
    print(f"wrote {path} with {len(pairs)} pair(s) for the third labeller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["status", "A", "B", "C"])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--first", type=Path)
    ap.add_argument("--second", type=Path)
    ap.add_argument("--third", type=Path)
    ap.add_argument("--key", default=str(PRIVATE / "job_C_key.json"))
    ap.add_argument("--make-adjudication", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.job == "status":
        status(args.files)
    elif args.job == "A":
        job_a(args.files, args.out or OUT / "job_A_scores.json")
    elif args.job == "B":
        job_b(args.files, args.out or OUT / "questions_gold.jsonl")
    else:
        if not (args.first and args.second):
            sys.exit("job C needs --first and --second")
        args.out = args.out or OUT / "job_C_scores.json"
        job_c(args)


if __name__ == "__main__":
    main()
