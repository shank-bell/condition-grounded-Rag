"""Apply the extractor's current clean-up rules to profiles that are already stored, without running the LLM again.

  python scripts/reclean_profiles.py            # dry run: prints what would change
  python scripts/reclean_profiles.py --apply    # writes the changes

Rules (profile_extractor._normalize / _is_result): "Dev"/"Test" in the dataset field moves to setting, a metric that repeats
the dataset name becomes "score", and profiles whose model is a cut-off piece of a row label ("L", "MA", "acon"), whose
model is really a dataset, that report carbon / energy / cost figures, or that come from a garbled table (numbers split or
glued by the PDF text layer, see tables.garble_ratio) are removed.
Do not run while the API or an ingest is using the index.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter

from cgrag.config import get_settings
from cgrag.ingestion.profile_extractor import _is_result, _normalize
from cgrag.ingestion.tables import GARBLE_LIMIT, garble_ratio, is_structured_table
from cgrag.schemas import ExtractedProfile
from cgrag.stores.vector_store import VectorStore

FIELDS = ("task", "dataset", "dataset_version", "metric", "value", "language", "model", "model_size", "setting", "evidence")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    db = sqlite3.connect(get_settings().paths.profile_db)
    db.row_factory = sqlite3.Row
    rows = db.execute(f"SELECT profile_id, paper_id, chunk_id, {','.join(FIELDS)} FROM profiles").fetchall()
    garbled = {c.chunk_id for c in VectorStore().all_chunks() if is_structured_table(c.text) and garble_ratio(c.text) >= GARBLE_LIMIT}
    updates: list[tuple[str, dict]] = []
    deletes: list[tuple[str, str]] = []
    changed_fields: Counter[str] = Counter()
    removed_models: Counter[str] = Counter()
    from_garbled = 0
    for r in rows:
        if r["chunk_id"] in garbled:                            # the table's text layer fell apart: its numbers cannot be trusted
            deletes.append((r["profile_id"], r["paper_id"]))
            from_garbled += 1
            continue
        p = ExtractedProfile(**{f: r[f] for f in FIELDS})
        q = _normalize(p)
        if not _is_result(q):
            deletes.append((r["profile_id"], r["paper_id"]))
            removed_models[str(r["model"])] += 1
            continue
        diff = {f: getattr(q, f) for f in FIELDS if getattr(q, f) != getattr(p, f)}
        if diff:
            updates.append((r["profile_id"], diff))
            changed_fields.update(diff.keys())
    print(f"{len(rows)} profiles: {len(updates)} would be changed {dict(changed_fields)}, {len(deletes)} would be removed "
          f"({from_garbled} of them from {len(garbled)} garbled table chunks)")
    print("most removed model names:", removed_models.most_common(12))
    print("removed per paper:", Counter(pid for _, pid in deletes).most_common(8))
    if not args.apply:
        print("dry run - nothing written (use --apply)")
        return
    with db:
        for pid, diff in updates:
            db.execute(f"UPDATE profiles SET {','.join(f'{k}=?' for k in diff)} WHERE profile_id=?", (*diff.values(), pid))
        db.executemany("DELETE FROM profiles WHERE profile_id=?", [(pid,) for pid, _ in deletes])
    print(f"applied. profiles now: {db.execute('SELECT COUNT(*) FROM profiles').fetchone()[0]}")


if __name__ == "__main__":
    main()
