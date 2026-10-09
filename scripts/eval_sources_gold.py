"""Does the context the answer is written from contain the PAGE that holds the answer? End to end, through the running API, scored against the
AI-annotated question key (the `evidence` field of eval/labels/questions_gold.jsonl, written from the PDFs).

  uvicorn cgrag.api.main:app --port 8000        # in another window, then:
  python scripts/eval_sources_gold.py            # 28 questions x about 12 s; writes eval/labels/sources_gold.json

Unlike scripts/eval_retrieval_gold.py (the raw question as the only query, no LLM) this runs the whole online path, so the question is
understood, refined and decomposed, and Stage 6 may search again and add passages that cover a missing condition. A source counts as the
evidence when its (paper, page) is one the key names. Reports the share of questions with the evidence page among the sources, the same at
paper level, and the number of sources.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from eval_retrieval_gold import evidence_pages


def ask(url: str, question: str) -> dict:
    req = urllib.request.Request(url, data=json.dumps({"question": question, "history": []}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=Path("eval/labels/questions_gold.jsonl"))
    ap.add_argument("--url", default="http://localhost:8000/query")
    ap.add_argument("--out", type=Path, default=Path("eval/labels/sources_gold.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = []
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        pages = evidence_pages(q.get("evidence") or "")
        if not pages:
            continue
        t0 = time.perf_counter()
        resp = ask(args.url, q["question"])
        got = {(s["paper_id"], s["page"]) for s in resp["sources"]}
        papers = {p for p, _ in pages}
        rows.append({"id": q["id"], "planned_type": q.get("planned_type"), "evidence_in_sources": bool(got & pages),
                     "paper_in_sources": bool({p for p, _ in got} & papers), "n_sources": len(resp["sources"]),
                     "seconds": round(time.perf_counter() - t0, 1), "warned": bool(resp.get("applicability") and resp["applicability"].get("warning"))})
        print(f"{q['id']}: evidence page in the sources: {rows[-1]['evidence_in_sources']}  (paper: {rows[-1]['paper_in_sources']}, {rows[-1]['n_sources']} sources, {rows[-1]['seconds']} s)", flush=True)

    def pct(key: str) -> float:
        return round(100 * sum(1 for r in rows if r[key]) / len(rows), 1)

    summary = {"questions": len(rows), "evidence_page_in_sources_%": pct("evidence_in_sources"), "evidence_paper_in_sources_%": pct("paper_in_sources"),
               "mean_sources": round(statistics.mean(r["n_sources"] for r in rows), 2), "median_seconds": round(statistics.median(r["seconds"] for r in rows), 1), "tag": args.tag}
    for kind in sorted({r["planned_type"] for r in rows if r["planned_type"]}):
        part = [r for r in rows if r["planned_type"] == kind]
        summary[f"evidence_page_in_sources_% {kind} (n={len(part)})"] = round(100 * sum(r["evidence_in_sources"] for r in part) / len(part), 1)
    print(json.dumps(summary, indent=1))
    print("missing:", [r["id"] for r in rows if not r["evidence_in_sources"]])
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
