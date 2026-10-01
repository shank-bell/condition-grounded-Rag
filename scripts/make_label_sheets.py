"""Build the Excel workbooks for labelling day (jobs A, B, C) from the current profile store.

  python scripts/make_label_sheets.py                 # all three jobs -> data/labelling/
  python scripts/make_label_sheets.py --jobs C --pairs 50 --seed 5

Output (data/labelling/, git-ignored; send the workbooks to the people named in the file names):
  A_profile_check_<name>.xlsx      Adarsh, Shashank   300 random profiles of 10 papers (6 easy, 4 hard), 150 each + 20 shared
  B_questions_<name>.xlsx          all four           7-8 rows each, planned mix 10 covered / 10 one missing / 10 partly + corpus map
  C_result_pairs_<name>.xlsx       Aditya, Tarun      50 pairs of conflicting results, both label all of them
  private/                         the answer keys (the system's verdict per pair, who got which profile): do NOT send these around
Nothing here needs Ollama. It reads data/index/profiles.sqlite and data/index/chunks.jsonl (re-exported from ChromaDB the first time,
so run it when no other process uses the index).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cgrag.config import get_settings
from cgrag.labelling.sampling import EASY_PAPERS, HARD_PAPERS, corpus_map, load_chunk_index, mine_pairs, paper_titles, pick_pairs, sample_profiles
from cgrag.labelling.sheets import B_TYPES, a_rows, build_job_a, build_job_b, build_job_c
from cgrag.stores.profile_store import ProfileStore

JOB_A_PEOPLE = ["Adarsh", "Shashank"]
JOB_C_PEOPLE = ["Aditya", "Tarun"]
# (covered, one missing, partly covered) per person: 10 + 10 + 10 over the four of them
JOB_B_PEOPLE = {"Aditya": (3, 3, 2), "Adarsh": (3, 3, 2), "Shashank": (2, 2, 3), "Tarun": (2, 2, 3)}


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="A,B,C")
    ap.add_argument("--out", type=Path, default=Path("data/labelling"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--profiles", type=int, default=300, help="job A: profiles to check in total (split between the labellers)")
    ap.add_argument("--overlap", type=int, default=20, help="job A: extra profiles that BOTH labellers get (agreement check)")
    ap.add_argument("--pairs", type=int, default=50, help="job C: result pairs")
    ap.add_argument("--people-a", default=",".join(JOB_A_PEOPLE))
    ap.add_argument("--people-c", default=",".join(JOB_C_PEOPLE))
    args = ap.parse_args()
    out, private = args.out, args.out / "private"
    jobs = set(args.jobs.upper().split(","))

    profiles = ProfileStore(cfg.paths.profile_db).all()
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    titles = {**paper_titles(cfg.paths.papers_dir, out / "paper_titles.json"), **index.titles}
    print(f"{len(profiles)} profiles, {len(index.texts)} chunks, {len(titles)} papers")

    if "A" in jobs:
        rnd = random.Random(args.seed)
        papers = [*EASY_PAPERS, *HARD_PAPERS]
        per_paper = args.profiles // len(papers)
        chosen = sample_profiles(profiles, papers, per_paper, rnd)
        shared = sample_profiles(profiles, papers, -(-args.overlap // len(papers)), rnd, exclude={p.profile_id for p in chosen})[: args.overlap]
        rnd.shuffle(chosen)
        people = args.people_a.split(",")
        halves = [chosen[i::len(people)] for i in range(len(people))]
        shared_ids = {p.profile_id for p in shared}
        group_of = {**{p: "easy" for p in EASY_PAPERS}, **{p: "hard" for p in HARD_PAPERS}}
        assignment = {}
        for person, mine in zip(people, halves):
            rows = a_rows([*mine, *shared], index.texts, index.pages, titles, shared_ids)
            path = build_job_a(out / f"A_profile_check_{person}.xlsx", person, rows)
            print(f"A: {path}  ({len(mine)} own + {len(shared)} shared rows)")
            for p in [*mine, *shared]:
                assignment.setdefault(p.profile_id, []).append(person)
        private.mkdir(parents=True, exist_ok=True)
        (private / "job_A_assignment.json").write_text(json.dumps({"assignment": assignment, "group_of_paper": group_of, "easy": EASY_PAPERS,
                                                                   "hard": HARD_PAPERS}, indent=1), encoding="utf-8")

    if "B" in jobs:
        rnd = random.Random(args.seed + 1)
        lookup = corpus_map(profiles, titles)
        for person, (c, m, p) in JOB_B_PEOPLE.items():
            planned = [B_TYPES[0]] * c + [B_TYPES[1]] * m + [B_TYPES[2]] * p
            rnd.shuffle(planned)
            path = build_job_b(out / f"B_questions_{person}.xlsx", person, person[:2].upper(), planned, lookup)
            print(f"B: {path}  ({len(planned)} rows: {c} covered, {m} one missing, {p} partly covered)")
        (out / "corpus_map.json").write_text(json.dumps({k: {"headers": h, "rows": [[(c[0] if isinstance(c, tuple) else c) for c in r] for r in rows]}
                                                         for k, (h, rows) in lookup.items()}, indent=1, ensure_ascii=False), encoding="utf-8")

    if "C" in jobs:
        rnd = random.Random(args.seed + 2)
        mined = mine_pairs(profiles)
        picked = pick_pairs(mined, args.pairs, rnd)
        rnd.shuffle(picked)
        ids = [f"P{i:02d}" for i in range(1, len(picked) + 1)]
        by_id = dict(zip(ids, picked))
        key = {pid: {"system_verdict": p.verdict, "system_differing": p.differing, "reason": p.reason, "stratum": p.stratum,
                     "rel_diff": round(p.rel, 4), "profile_a": p.x.profile_id, "profile_b": p.y.profile_id,
                     "paper_a": p.x.paper_id, "paper_b": p.y.paper_id, "value_a": p.x.value, "value_b": p.y.value,
                     "metric": p.x.metric, "model_a": p.x.model, "model_b": p.y.model, "dataset_a": p.x.dataset, "dataset_b": p.y.dataset}
               for pid, p in by_id.items()}
        private.mkdir(parents=True, exist_ok=True)
        (private / "job_C_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
        for person in args.people_c.split(","):
            order = list(by_id.items())
            random.Random(f"{args.seed}-{person}").shuffle(order)
            path = build_job_c(out / f"C_result_pairs_{person}.xlsx", person, order, index.texts, index.pages, titles)
            print(f"C: {path}  ({len(order)} pairs, own order)")
        from collections import Counter
        print("C: the system's classes in the sample (private):", dict(Counter(p.verdict for p in picked)),
              "| strata:", dict(Counter(p.stratum for p in picked)))
        print(f"C: {len(mined)} conflicting pairs were available; the key is in {private / 'job_C_key.json'}")


if __name__ == "__main__":
    main()
