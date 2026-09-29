"""Ask one question through all ten stages and print everything the pipeline decided.

  python scripts/ask.py "Does BERT work well for Kannada question answering?"
  python scripts/ask.py --json "..."          # the raw QueryResponse
"""
from __future__ import annotations

import argparse

from cgrag import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="print the raw QueryResponse JSON")
    args = ap.parse_args()

    r = run(args.question)
    if args.json:
        print(r.model_dump_json(indent=2))
        return
    print(f"\nQ: {r.question}")
    if r.analysis:
        print(f"   intent={r.analysis.intent} complexity={r.analysis.complexity} conditions={r.analysis.conditions.specified()}")
    print(f"\n{r.answer}\n")
    if r.applicability:
        a = r.applicability
        print(f"COVERAGE {a.coverage:.0%}" + (" (searched again)" if a.re_retrieved else ""))
        for c in a.checks:
            print(f"  {'OK ' if c.covered else 'MISS'} {c.condition}={c.requested}  observed={c.observed}")
        if a.warning:
            print(f"  WARNING: {a.warning}")
    for c in r.contradictions:
        print(f"\nCONFLICT {c.verdict}: {c.paper_a} vs {c.paper_b}  {c.value_a} vs {c.value_b} {c.metric or ''}\n  {c.reason}")
    if r.claim_checks:
        bad = [c for c in r.claim_checks if not c.supported]
        print(f"\nCLAIMS {len(r.claim_checks) - len(bad)}/{len(r.claim_checks)} supported" + (" (regenerated once)" if r.regenerated else ""))
        for c in bad:
            print(f"  UNSUPPORTED ({c.entailment:.2f}): {c.sentence}")
    print("\nSOURCES")
    for i, s in enumerate(r.sources, 1):
        print(f"  [{i}] {s.paper_title[:70]} p{s.page} {s.section} score={s.score:.2f}")
    print("\nTRACE  " + "\n       ".join(r.trace))
    print("TIMES  " + "  ".join(f"{k}={v / 1000:.1f}s" for k, v in r.timings_ms.items()))


if __name__ == "__main__":
    main()
