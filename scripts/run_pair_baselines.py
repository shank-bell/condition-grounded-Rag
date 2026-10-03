"""Run Stage 7's baselines on the result pairs of labelling job C and FREEZE what every system said, before any label is known.

  python scripts/run_pair_baselines.py                 # -> data/labelling/private/job_C_baselines.json  (git-ignored)
  python scripts/run_pair_baselines.py --limit 5       # a quick check

Systems (architecture doc, "How each claim will be tested"):
  plain_nli     DeBERTa NLI on the two results as statements; contradiction -> GENUINE, otherwise NOT_COMPARABLE (never EXPLAINED)
  llm_taxonomy  the same local Gemma as the answer, zero-shot, with a prompt built on the DRAGged-into-Conflicts taxonomy, shown exactly what
                the labellers see (caption, header, the row with the number marked); complementary information -> EXPLAINED,
                conflicting outcomes / outdated / misinformation -> GENUINE, no conflict -> NOT_COMPARABLE
  stage7_rerun  Stage 7's own classifier run again on the stored profiles: a check that the key built for the sheets (private/job_C_key.json,
                what the paper reports as "the system") still says what the current code says
Needs Ollama; do not run it while an ingest or the API holds the index. Score against the gold:
  python scripts/score_labels.py C --first ... --second ... --baselines data/labelling/private/job_C_baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from cgrag.config import get_settings
from cgrag.evaluation.baselines import llm_conflict_verdict, plain_nli_verdict
from cgrag.evaluation.freeze import provenance
from cgrag.labelling.sampling import evidence_excerpt, load_chunk_index
from cgrag.llm import OllamaLLM
from cgrag.models import get_nli
from cgrag.pipeline.conditions import claim_text
from cgrag.pipeline.contradiction import classify
from cgrag.stores.profile_store import ProfileStore

PRIVATE = Path("data/labelling/private")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, default=PRIVATE / "job_C_key.json")
    ap.add_argument("--out", type=Path, default=PRIVATE / "job_C_baselines.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = get_settings()
    key = json.loads(args.key.read_text(encoding="utf-8"))
    by_id = {p.profile_id: p for p in ProfileStore(cfg.paths.profile_db).all()}
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    llm, nli = OllamaLLM(cfg), get_nli()
    systems: dict[str, dict] = {"stage7_rerun": {}, "plain_nli": {}, "llm_taxonomy": {}}
    ids = sorted(key)[: args.limit or None]
    t0 = time.perf_counter()
    for n, pid in enumerate(ids, start=1):
        k = key[pid]
        a, b = by_id[k["profile_a"]], by_id[k["profile_b"]]
        verdict, differing, reason = classify(a, b)
        systems["stage7_rerun"][pid] = {"system_verdict": verdict, "system_differing": differing, "reason": reason}
        systems["plain_nli"][pid] = plain_nli_verdict(nli, claim_text(a), claim_text(b))
        header = (f"Both results are reported as {k['metric']}. Result A: {k['model_a']} on {k['dataset_a']}. "
                  f"Result B: {k['model_b']} on {k['dataset_b']}.")
        systems["llm_taxonomy"][pid] = llm_conflict_verdict(llm, header, evidence_excerpt(index.texts[a.chunk_id], a),
                                                            evidence_excerpt(index.texts[b.chunk_id], b))
        print(f"{n}/{len(ids)} {pid}: key {k['system_verdict']:<14} rerun {verdict:<14} nli {systems['plain_nli'][pid]['system_verdict']:<14} "
              f"llm {systems['llm_taxonomy'][pid]['system_verdict']:<14} ({time.perf_counter() - t0:.0f}s)", flush=True)

    changed = [pid for pid in ids if systems["stage7_rerun"][pid]["system_verdict"] != key[pid]["system_verdict"]
               or set(systems["stage7_rerun"][pid]["system_differing"]) != set(key[pid]["system_differing"])]
    summary = {name: dict(Counter(v["system_verdict"] for v in rows.values())) for name, rows in systems.items()}
    summary["key"] = dict(Counter(key[pid]["system_verdict"] for pid in ids))
    out = {**provenance(), "pairs": len(ids), "llm_model": llm.cfg.model, "nli_model": cfg.models.nli,
           "stage7_rerun_differs_from_key": changed, "class_counts": summary, "systems": systems}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(ids)} pairs, {time.perf_counter() - t0:.0f}s)")
    print("class counts:", json.dumps(summary))
    print(f"Stage 7 re-run differs from the key on {len(changed)} pair(s): {changed[:10]}")


if __name__ == "__main__":
    main()
