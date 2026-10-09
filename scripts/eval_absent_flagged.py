"""Questions about things that are NOT in the corpus: does the system end with a warning? End to end, through the running API.

The reranker's own "weak evidence" flag catches about 70-75 % of such questions (scripts/eval_retrieval.py, whatever the threshold), but the
architecture has two mechanisms: the weak-evidence flag (Stage 5) and the scope warning (Stage 6: the model, data set or language the question
names is recorded nowhere). A question counts as flagged when either fires. The 20 questions are those of scripts/eval_retrieval.py (ABSENT).

  uvicorn cgrag.api.main:app --port 8000      # then
  python scripts/eval_absent_flagged.py       # 20 questions x about 12 s; writes eval/labels/absent_flagged.json

The false-alarm side (a question the papers DO cover that gets a warning) is measured by the gold questions: eval/labels/questions_gold.*.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from eval_retrieval import ABSENT


def ask(url: str, question: str) -> dict:
    req = urllib.request.Request(url, data=json.dumps({"question": question, "history": []}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/query")
    ap.add_argument("--out", type=Path, default=Path("eval/labels/absent_flagged.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    for q in ABSENT:
        r = ask(args.url, q)
        a = r.get("applicability") or {}
        row = {"question": q, "weak_flag": bool(r.get("retrieval_weak")), "scope_warning": bool(a.get("warning")), "coverage": a.get("coverage"),
               "sources": len(r.get("sources", []))}
        row["flagged"] = row["weak_flag"] or row["scope_warning"]
        rows.append(row)
        print(f"{'FLAGGED' if row['flagged'] else 'not flagged':<12} weak={row['weak_flag']!s:<5} scope={row['scope_warning']!s:<5} {q}", flush=True)
    n = len(rows)
    summary = {"questions": n, "flagged_by_either_%": round(100 * sum(r["flagged"] for r in rows) / n, 1),
               "weak_flag_only_%": round(100 * sum(r["weak_flag"] for r in rows) / n, 1),
               "scope_warning_only_%": round(100 * sum(r["scope_warning"] for r in rows) / n, 1), "tag": args.tag}
    print(json.dumps(summary, indent=1))
    print("not flagged:", [r["question"] for r in rows if not r["flagged"]])
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
