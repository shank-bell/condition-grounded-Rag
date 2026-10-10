"""Job C held-out set: NEW result pairs for a clean test of Stage 7 after the fixes found on the first 50 pairs.

  python scripts/make_heldout_pairs.py --pairs 50 --seed 1016

Same mining and sampling as the first set (scripts/make_label_sheets.py --jobs C), with three guards:
  * no pair of papers + subject (model family, dataset, metric) and no profile of the first set (data/labelling/private/job_C_key.json) is reused;
  * the strata that steer the sample come from the FROZEN Stage 7 (every fix-A switch off), so the fixes do not choose their own test;
  * the system's verdicts (frozen and current) are written to a private key file and never printed, so the labels can be made blind.
Ids are H01..Hnn so the two sets cannot be mixed up. Writes two blank workbooks (first and second labeller, own order each).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cgrag.config import ContradictionConfig, get_settings
from cgrag.labelling.sampling import load_chunk_index, mine_pairs, paper_titles, pick_pairs
from cgrag.labelling.sheets import build_job_c
from cgrag.pipeline.conditions import metric_key, model_family, norm, split_version
from cgrag.pipeline.contradiction import classify
from cgrag.stores.profile_store import ProfileStore

FROZEN = ContradictionConfig(human_rows_not_comparable=False, two_metric_tasks_no_genuine=False, mnli_split_tags=False, one_sided_conditions_explain=False, table_grounding=False)


def subject(p) -> tuple:
    return (model_family(p.model), norm(split_version(p.dataset)[0]), metric_key(p.metric))


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1016)
    ap.add_argument("--out", type=Path, default=Path("data/labelling"))
    ap.add_argument("--first-key", type=Path, default=Path("data/labelling/private/job_C_key.json"))
    ap.add_argument("--people", default="first,second")
    ap.add_argument("--allow-profile-reuse", action="store_true", help="when no unused profile pairs are left: exclude only pairs (two profiles) that earlier sets used, so a profile may appear in a new pair")
    ap.add_argument("--reuse-subjects", action="store_true", help="exclude only profiles that earlier sets used, not their paper pair + subject (when the pool of new subjects is empty)")
    ap.add_argument("--prefix", default="H", help="pair id prefix (H = the held-out set; use another letter for a further set)")
    ap.add_argument("--name", default="heldout", help="set name in the file names: C_<name>_pairs_<person>.xlsx and job_C_<name>_key.json")
    ap.add_argument("--also-exclude", nargs="*", type=Path, default=[], help="keys of further earlier sets whose profiles and subjects must not be reused (e.g. job_C_heldout_key.json)")
    args = ap.parse_args()
    private = args.out / "private"

    store = ProfileStore(cfg.paths.profile_db)
    profiles = store.all()
    by_id = {p.profile_id: p for p in profiles}
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    titles = {**paper_titles(cfg.paths.papers_dir, args.out / "paper_titles.json"), **index.titles}

    old = json.loads(args.first_key.read_text(encoding="utf-8"))
    for extra in args.also_exclude:
        old.update({f"{extra.stem}:{k}": v for k, v in json.loads(extra.read_text(encoding="utf-8")).items()})
    used_profiles = {k[f"profile_{s}"] for k in old.values() for s in "ab"}
    used_subjects = {(frozenset((k["paper_a"], k["paper_b"])), subject(by_id[k["profile_a"]])) for k in old.values()}

    mined = mine_pairs(profiles)
    used_pairs = {frozenset((k["profile_a"], k["profile_b"])) for k in old.values()}
    fresh = [p for p in mined
             if (frozenset((p.x.profile_id, p.y.profile_id)) not in used_pairs if args.allow_profile_reuse
                 else p.x.profile_id not in used_profiles and p.y.profile_id not in used_profiles)
             and (args.reuse_subjects or (frozenset((p.x.paper_id, p.y.paper_id)), subject(p.x)) not in used_subjects)]
    for p in fresh:                                    # strata from the frozen rules, not from the fixes being tested
        p.verdict, p.differing, p.reason = classify(p.x, p.y, FROZEN)
    rnd = random.Random(args.seed)
    picked = pick_pairs(fresh, args.pairs, rnd)
    rnd.shuffle(picked)
    ids = [f"{args.prefix}{i:02d}" for i in range(1, len(picked) + 1)]
    chosen = dict(zip(ids, picked))

    key = {}
    for pid, p in chosen.items():
        now = classify(p.x, p.y)
        key[pid] = {"system_verdict_frozen": p.verdict, "system_differing_frozen": p.differing, "reason_frozen": p.reason,
                    "system_verdict": now[0], "system_differing": now[1], "reason": now[2], "stratum_frozen": p.stratum,
                    "rel_diff": round(p.rel, 4), "profile_a": p.x.profile_id, "profile_b": p.y.profile_id,
                    "paper_a": p.x.paper_id, "paper_b": p.y.paper_id, "value_a": p.x.value, "value_b": p.y.value,
                    "metric": p.x.metric, "model_a": p.x.model, "model_b": p.y.model, "dataset_a": p.x.dataset, "dataset_b": p.y.dataset}
    private.mkdir(parents=True, exist_ok=True)
    key_path = private / f"job_C_{args.name}_key.json"
    key_path.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    for person in args.people.split(","):
        order = list(chosen.items())
        random.Random(f"{args.seed}-{person}").shuffle(order)
        path = build_job_c(args.out / f"C_{args.name}_pairs_{person}.xlsx", f"{args.name} ({person} labeller)", order, index.texts, index.pages, titles)
        print(f"wrote {path} ({len(order)} pairs)")
    print(f"{len(mined)} conflicting pairs mined, {len(fresh)} not used in the first set, {len(chosen)} picked "
          f"({len({frozenset((p.x.paper_id, p.y.paper_id)) for p in picked})} paper pairs). Key (NOT printed): {key_path}")


if __name__ == "__main__":
    main()
