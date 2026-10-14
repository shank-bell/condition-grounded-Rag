"""The Stage 7 judge baseline again, but SHOWN what Stage 7 sees (13 Oct 2026): does the rule add anything beyond the store?

  python scripts/run_pair_baselines_profiles.py                 # six pair sets -> data/labelling/private/job_C_*_baselines_profiles.json  (git-ignored)
  python scripts/run_pair_baselines_profiles.py --limit 5       # a quick check

`run_pair_baselines.py` froze what a plain local LLM says about a pair when it sees only what the labellers see (caption, header, the row with the number marked) and a prompt built
on the DRAGged-into-Conflicts taxonomy. Stage 7 however reads the pair through the Condition Profile store: the recorded conditions of each result, checked against its own table
cell (`pipeline/grounding.py`). This script gives the SAME model, prompt and excerpts those recorded conditions as well (the profile fields read through the repair overlay, the
block heading, the size the table states, and a note when the stored record does not fit its cell) and asks for the same verdict. If this judge reaches the typing step's score,
the store carries the result and the rules add little; if it stays far below, the rules add value on top of the store. Nothing is trained on labels and no label is read here.
Score it with: PROFILE_REPAIR=on PAIR_BASELINE_SUFFIX=_profiles STAGE7_OUT_TAG=_judge_profiles python scripts/score_stage7_variants.py
Needs Ollama (the answer model); the API may stay up (this script loads no other model).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cgrag.config import get_settings
from cgrag.evaluation.baselines import TAXONOMY_SYSTEM, ConflictJudgement, differing_names, taxonomy_verdict
from cgrag.evaluation.freeze import provenance
from cgrag.labelling.sampling import evidence_excerpt, load_chunk_index
from cgrag.llm import OllamaLLM
from cgrag.pipeline.grounding import ground
from cgrag.stores.profile_store import ProfileStore

PRIVATE = Path("data/labelling/private")
SETS = {
    "first 50": ("job_C_key.json", "job_C_baselines.json"),
    "held-out 50": ("job_C_heldout_key.json", "job_C_heldout_baselines.json"),
    "fresh 50": ("job_C_fresh_key.json", "job_C_fresh_baselines.json"),
    "fresh-2 50": ("job_C_fresh2_key.json", "job_C_fresh2_baselines.json"),
    "fresh-3 50": ("job_C_fresh3_key.json", "job_C_fresh3_baselines.json"),
    "fresh-4 50": ("job_C_fresh4_key.json", "job_C_fresh4_baselines.json"),
}
SYSTEM = TAXONOMY_SYSTEM.replace(
    "Judge from that text only.",
    "Each result also comes with the conditions that a reader of the table recorded for it (model, size, data set and version, language, setting, "
    "and what the table's heading says). Judge from the text and from these recorded conditions.")


def record(p, g) -> str:
    parts = [f"{name.replace('_', ' ')}: {value}" for name, value in p.conditions().items()]
    parts.append(f"metric: {p.metric}")
    parts.append(f"value: {p.value}")
    if g.found:
        if g.block:
            parts.append(f"the table's block heading: {g.block}")
        if g.setting_tags:
            parts.append("the heading or caption says: " + ", ".join(sorted(g.setting_tags)))
        if g.size:
            parts.append(f"size stated by the table: {g.size}")
        if g.suspect:
            parts.append("the stored record does not fit its table cell (" + ", ".join(g.suspect) + ")")
    return "; ".join(parts)


def judge(llm, header: str, evidence_a: str, evidence_b: str, rec_a: str, rec_b: str) -> dict:
    prompt = (f"{header}\n\nRESULT A\n{evidence_a}\nRecorded conditions of A: {rec_a}\n\nRESULT B\n{evidence_b}\nRecorded conditions of B: {rec_b}")
    try:
        out, _ = llm.structured(prompt, ConflictJudgement, system=SYSTEM, max_tokens=200, temperature=0.0)
    except ValueError as err:
        return {"system_verdict": "NOT_COMPARABLE", "system_differing": [], "category": None, "reason": f"unusable reply: {err}"[:200]}
    verdict = taxonomy_verdict(out.category)
    return {"system_verdict": verdict, "system_differing": differing_names(out.differing_conditions) if verdict == "EXPLAINED" else [],
            "category": out.category, "reason": out.reason}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs of each set")
    ap.add_argument("--workers", type=int, default=2, help="parallel LLM calls (the Ollama server has two slots)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = get_settings()
    by_id = {p.profile_id: p for p in ProfileStore(repaired=True).all()}
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    llm = OllamaLLM(cfg)
    t0 = time.perf_counter()
    for set_name, (key_file, base_file) in SETS.items():
        key = json.loads((PRIVATE / key_file).read_text(encoding="utf-8"))
        base = json.loads((PRIVATE / base_file).read_text(encoding="utf-8"))
        ids = sorted(key)[: args.limit or None]

        def one(pid: str) -> tuple[str, dict]:
            k = key[pid]
            a, b = by_id[k["profile_a"]], by_id[k["profile_b"]]
            header = (f"Both results are reported as {k['metric']}. Result A: {k['model_a']} on {k['dataset_a']}. Result B: {k['model_b']} on {k['dataset_b']}.")
            ta, tb = index.texts[a.chunk_id], index.texts[b.chunk_id]
            return pid, judge(llm, header, evidence_excerpt(ta, a), evidence_excerpt(tb, b), record(a, ground(a, ta)), record(b, ground(b, tb)))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = dict(pool.map(one, ids))
        counts = Counter(v["system_verdict"] for v in rows.values())
        out = {**provenance(), "pairs": len(ids), "llm_model": llm.cfg.model, "set": set_name, "variant": "judge shown the recorded conditions (store fields + grounding)",
               "class_counts": {"llm_taxonomy_profiles": dict(counts)},
               "systems": {"plain_nli": {p: base["systems"]["plain_nli"][p] for p in ids}, "llm_taxonomy": rows}}
        target = PRIVATE / (Path(base_file).stem + "_profiles.json")
        target.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{set_name}] {len(ids)} pairs -> {target.name}  {dict(counts)}  ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
