"""Run questions through the whole pipeline and print what every stage decided, with timings per stage.

  python scripts/smoke_questions.py                                   # the default question set
  python scripts/smoke_questions.py "What F1 does BERT-large get on SQuAD v2.0?"
  python scripts/smoke_questions.py --set orchestrator_model=gemma4:12b --set applicability_model=gemma4:12b
  python scripts/smoke_questions.py --brief                            # one line per question + a timing table

--set KEY=VALUE overrides a key of [agents] (orchestrator_model, applicability_model, understanding_model, refinement_model,
extractor_model) or of [llm] (model) for this run only, to compare model sizes per use case.
Do not run while an ingest is using the index (ChromaDB is single-process).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from cgrag.config import get_settings
from cgrag.pipeline.run import Pipeline

QUESTIONS = [
    "How well do models perform on Kannada natural language inference?",
    "What is the human performance on SQuAD?",
    "What F1 does BERT-large get on SQuAD v2.0?",
    "How does XLM-R compare with mBERT on XNLI?",
    "What accuracy does mBERT get on XNLI for Hindi?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    for item in args.set:
        key, _, value = item.partition("=")
        if key in settings.agents.model_fields:
            settings = settings.model_copy(update={"agents": settings.agents.model_copy(update={key: value})})
        elif key == "model":
            settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={"model": value})})
        else:
            raise SystemExit(f"unknown key {key}")
    t0 = time.perf_counter()
    pipe = Pipeline(settings)
    a = settings.agents
    print(f"pipeline ready ({time.perf_counter() - t0:.1f}s) answer model={settings.llm.model} | orchestrator={a.orchestrator_model or 'same'} "
          f"applicability={a.applicability_model or 'same'} understanding={a.understanding_model or 'same'} "
          f"refinement={a.refinement_model or 'same'}", flush=True)
    totals, per_stage = [], {}
    for q in args.questions or QUESTIONS:
        t0 = time.perf_counter()
        r = pipe.run(q)
        total = time.perf_counter() - t0
        totals.append(total)
        for k, v in r.timings_ms.items():
            per_stage.setdefault(k, []).append(v / 1000)
        if args.brief:
            print(f"[{total:5.1f}s] {q}\n        {r.answer[:220].replace(chr(10), ' ')}", flush=True)
            continue
        print(f"\n=== Q: {q}\n(total {total:.1f}s)")
        if r.analysis:
            extras = r.analysis.conditions.extras()
            print(f"ANALYSIS: intent={r.analysis.intent} complexity={r.analysis.complexity} conditions={r.analysis.conditions.specified()}"
                  + (f" also named={extras}" if extras else ""))
        print("ANSWER:", r.answer)
        if r.applicability:
            ap_ = r.applicability
            print(f"COVERAGE {ap_.coverage:.0%}{' (searched again)' if ap_.re_retrieved else ''}")
            for c in ap_.checks:
                print(f"   {'OK  ' if c.covered else 'MISS'} {c.condition}={c.requested} observed={c.observed}")
            if ap_.warning:
                print("   WARNING:", ap_.warning)
        for c in r.contradictions:
            print(f"CONFLICT {c.verdict}: {c.paper_a} vs {c.paper_b} | {c.value_a} vs {c.value_b} {c.metric} | differing={c.differing}\n   {c.reason}")
        if r.claim_checks:
            bad = [c for c in r.claim_checks if not c.supported]
            print(f"CLAIMS {len(r.claim_checks) - len(bad)}/{len(r.claim_checks)} supported{' (regenerated once)' if r.regenerated else ''}")
            for c in bad:
                print(f"   UNSUPPORTED ({c.entailment:.2f}): {c.sentence}")
        print("SOURCES:", [f"{s.paper_id} p{s.page} {s.section}" for s in r.sources])
        print("TRACE:", *r.trace, sep="\n   ")
        print("TIMES:", {k: round(v / 1000, 1) for k, v in r.timings_ms.items()}, flush=True)

    print("\n--- timing (seconds per question) ---")
    print(f"total: first {totals[0]:.1f} (includes model warm-up) | others median {statistics.median(totals[1:]) if len(totals) > 1 else totals[0]:.1f}"
          f" min {min(totals[1:] or totals):.1f} max {max(totals[1:] or totals):.1f}")
    for k in sorted(per_stage):
        vals = per_stage[k][1:] or per_stage[k]
        print(f"  {k:<16} median {statistics.median(vals):5.2f}  max {max(vals):5.2f}")


if __name__ == "__main__":
    main()
