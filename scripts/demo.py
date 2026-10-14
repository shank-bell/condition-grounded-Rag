"""A small live demo: ask the running API a handful of questions and print what the pipeline decided at each stage.

  python scripts/demo.py [--api http://localhost:8000] [--only 2,5] [--json]

Each question shows one thing: a plain lookup with its recorded number, a question the corpus covers only for another condition (a scope warning that says what IS recorded),
a question the corpus covers through a different data set (no warning), a table lookup by model size, two papers that disagree because their conditions differ (Stage 7), a
model that is not in the corpus at all, and a question that needs no numbers. Questions 2 and 6 are from the 30 gold questions (development items); the others were written
for the demo. The API must be up (`uvicorn cgrag.api.main:app --port 8000`; the page at http://localhost:5173 shows the same run live).
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
import urllib.request

DEMOS = [
    ("1", "A plain lookup, answered from a recorded result",
     "What F1 does BERT-large reach on SQuAD v1.1?"),
    ("2", "Scope warning: the corpus has XLM-R and XNLI, but no Kannada result for that pair (gold question B-AI-11)",
     "What is the accuracy of XLM-R on the XNLI dataset for the Kannada language?"),
    ("3", "No false alarm: Kannada NLI is covered, through IndicXNLI",
     "How well do models perform on Kannada NLI?"),
    ("4", "A table lookup by model size (the appendix table of GPT-3, read by the table reader)",
     "What is the zero-shot accuracy of GPT-3 13B on HellaSwag?"),
    ("5", "Two papers report different numbers: is it a contradiction or a difference of conditions?",
     "What is XLM-R's average accuracy on XNLI?"),
    ("6", "A model that is not in the corpus (gold question B-AI-23)",
     "Compare mBERT, XLM-R and Mistral 7B on XNLI."),
    ("7", "A question that needs no numbers",
     "How does ELECTRA's replaced token detection pre-training work?"),
]


def ask(api: str, question: str) -> tuple[dict, float]:
    req = urllib.request.Request(api.rstrip("/") + "/query", data=json.dumps({"question": question}).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode()), time.time() - t0


def wrap(text: str, indent: str = "    ", width: int = 110, limit: int = 900) -> str:
    text = " ".join((text or "").split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def show(number: str, title: str, question: str, r: dict, seconds: float) -> None:
    print("\n" + "=" * 112)
    print(f"[{number}] {title}")
    print(f"Q: {question}")
    a = r.get("analysis") or {}
    cond = a.get("conditions") or {}
    named = ", ".join(f"{k}={v}" for k, v in cond.items() if v not in (None, "", [], {}) and k != "raw")
    print(f"STAGE 1  intent={a.get('intent')}  complexity={a.get('complexity')}  conditions: {named or '-'}")
    app = r.get("applicability") or {}
    if app:
        again = "  (searched again)" if app.get("re_retrieved") else ""
        print(f"STAGE 6  coverage {float(app.get('coverage') or 0):.0%}{again}")
        for c in app.get("checks") or []:
            print(f"           {'OK  ' if c.get('covered') else 'MISS'} {c.get('condition')}={c.get('requested')}   recorded: {str(c.get('observed'))[:90]}")
        if app.get("warning"):
            print("  SCOPE WARNING:")
            print(wrap(app["warning"], indent="    "))
    for c in r.get("contradictions") or []:
        print(f"STAGE 7  {c.get('verdict')}: {c.get('paper_a')} vs {c.get('paper_b')}   {c.get('value_a')} vs {c.get('value_b')} {c.get('metric') or ''}")
        print(wrap(c.get("reason") or "", indent="           ", limit=420))
    print("ANSWER")
    print(wrap(r.get("answer") or "", indent="    "))
    checks = r.get("claim_checks") or []
    if checks:
        ok = sum(1 for c in checks if c.get("supported"))
        print(f"STAGE 9  {ok}/{len(checks)} claims supported by their sources" + ("  (answer regenerated once)" if r.get("regenerated") else ""))
    for i, s in enumerate((r.get("sources") or [])[:4], 1):
        print(f"  source [{i}] {str(s.get('paper_title'))[:72]}  p{s.get('page')}  {s.get('section')}")
    t = r.get("timings_ms") or {}
    print(f"TIME     {seconds:.1f} s total" + ("   (" + ", ".join(f"{k} {v / 1000:.1f}" for k, v in list(t.items())[:6]) + ")" if t else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--only", default="", help="comma-separated demo numbers, e.g. 2,5")
    ap.add_argument("--json", action="store_true", help="also dump the raw response of every question")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    for number, title, question in DEMOS:
        if only and number not in only:
            continue
        r, secs = ask(args.api, question)
        show(number, title, question, r, secs)
        if args.json:
            print(json.dumps(r, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
