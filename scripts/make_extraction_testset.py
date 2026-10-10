"""Fresh test set for the profile repair (11 Oct 2026): a random sample of stored profiles that the repair was NOT designed on, written out so that a
labeller can write the gold fields from the table BEFORE seeing either the stored or the repaired fields (gold first, blind to both versions).

  python scripts/make_extraction_testset.py --n-unseen 80 --n-dev 40 --seed 1110

* "unseen": profiles of the 18 papers that had no profile on the job-A sheets (no rule was written while looking at them);
* "dev": new profiles of the 10 job-A papers (their papers' tables were read while the rules were written, these rows were not).
The sample excludes every job-A profile. Output (private, git-ignored): data/labelling/private/extraction_test/
  items.jsonl     - id, profile id, paper, the cell (value, row label, column, block, caption), the table text; NO stored field
  key_template.jsonl - one line per item for the gold fields (the labeller fills it)
  versions.jsonl  - the stored and the repaired profile of each item (read only by the scorer, never by the labeller)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.ingestion.repair import repair_paper  # noqa: E402
from cgrag.ingestion.tables import parse_table  # noqa: E402
from cgrag.pipeline.grounding import ground  # noqa: E402
from cgrag.stores.profile_store import ProfileStore  # noqa: E402
from cgrag.stores.vector_store import VectorStore  # noqa: E402

FIELDS = ["task", "dataset", "dataset_version", "metric", "value", "language", "model", "model_size", "setting"]


def job_a_ids() -> set[str]:
    ids = set()
    for name in ("A_profile_check_Shashank.xlsx", "A_profile_check_Adarsh.xlsx"):
        wb = openpyxl.load_workbook(ROOT / "data/labelling" / name, read_only=True)
        for r in wb["Profiles"].iter_rows(min_row=2, values_only=True):
            if r[1]:
                ids.add(r[1])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unseen", type=int, default=80)
    ap.add_argument("--n-dev", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1110)
    ap.add_argument("--out", default=str(ROOT / "data/labelling/private/extraction_test"))
    args = ap.parse_args()

    store, vs = ProfileStore(repaired=False), VectorStore()      # the RAW extraction: the repair is applied (and scored) on top
    seen = job_a_ids()
    dev_papers = {pid.split(":")[0] for pid in seen}
    allp = [p for p in store.all() if p.profile_id not in seen]
    rng = random.Random(args.seed)
    unseen = [p for p in allp if p.paper_id not in dev_papers]
    dev = [p for p in allp if p.paper_id in dev_papers]
    sample = [("unseen", p) for p in rng.sample(unseen, args.n_unseen)] + [("dev", p) for p in rng.sample(dev, args.n_dev)]
    rng.shuffle(sample)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks_by_paper: dict[str, dict[str, str]] = {}
    titles: dict[str, str] = {}
    repaired: dict[str, object] = {}
    for paper in sorted({p.paper_id for _, p in sample}):
        cs = vs.paper_chunks(paper)
        chunks_by_paper[paper] = {c.chunk_id: c.text for c in cs}
        titles[paper] = cs[0].paper_title if cs else ""
        fixed, _ = repair_paper(store.for_paper(paper), chunks_by_paper[paper], title=titles[paper])
        repaired.update({q.profile_id: q for q in fixed})

    with open(out / "items.jsonl", "w", encoding="utf-8") as fi, open(out / "key_template.jsonl", "w", encoding="utf-8") as fk, \
            open(out / "versions.jsonl", "w", encoding="utf-8") as fv:
        for i, (group, p) in enumerate(sample, 1):
            text = chunks_by_paper[p.paper_id].get(p.chunk_id, "")
            table = parse_table(text)
            g = ground(p, text) if table else None
            item = {"id": f"E{i:03d}", "group": group, "profile_id": p.profile_id, "paper": p.paper_id, "title": titles[p.paper_id],
                    "value": p.value, "evidence": p.evidence,
                    "cell": {"row_label": g.row_label, "column": g.column, "block": g.block} if g else None,
                    "caption": table.caption if table else "", "chunk_text": text[:6000]}
            fi.write(json.dumps(item, ensure_ascii=False) + "\n")
            fk.write(json.dumps({"id": item["id"], **{f: None for f in FIELDS if f != "value"}, "not_a_result": False, "note": ""}) + "\n")
            fv.write(json.dumps({"id": item["id"], "stored": p.model_dump(include=set(FIELDS)),
                                 "repaired": repaired[p.profile_id].model_dump(include=set(FIELDS))}, ensure_ascii=False) + "\n")
    print(f"{len(sample)} items -> {out}  (unseen {args.n_unseen}, dev {args.n_dev}); papers: {len(chunks_by_paper)}")


if __name__ == "__main__":
    main()
