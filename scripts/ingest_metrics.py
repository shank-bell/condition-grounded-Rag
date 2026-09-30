"""Metrics of the offline path after ingestion (no LLM): what is in the stores, how complete the profiles are, and how
well the profiles agree with the tables they were read from.

  python scripts/ingest_metrics.py            # prints a report
  python scripts/ingest_metrics.py --json out.json

Automatic checks (no hand labels needed):
  * store integrity: ChromaDB / BM25 / profile store agree, every link resolves, vectors are 1024-d.
  * field fill rates per condition field.
  * grounding: the profile's value (and its evidence quote) occurs in the chunk it points to.
  * table cell recall: every decimal number in a recognised table should appear as a profile value of that chunk.
  * row-label accuracy: the profile carrying that number should name the system written in the table row.
  * language accuracy on tables whose columns are languages.
This is NOT the doc's hand-checked field-level F1 (that needs the team's labels); it measures value coverage and the
row/column attribution, which is where the extractor failed before.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict

from cgrag.config import get_settings
from cgrag.ingestion.tables import _rows, is_numeric_cell, is_statistics_table, is_structured_table, parse_table
from cgrag.pipeline.conditions import norm
from cgrag.stores.bm25_store import BM25Store
from cgrag.stores.vector_store import VectorStore

FIELDS = ("task", "dataset", "dataset_version", "metric", "language", "model", "model_size", "setting")
DECIMAL = re.compile(r"\d+\.\d+")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
LANG = {"en": "english", "fr": "french", "es": "spanish", "de": "german", "el": "greek", "bg": "bulgarian", "ru": "russian",
        "tr": "turkish", "ar": "arabic", "vi": "vietnamese", "th": "thai", "zh": "chinese", "hi": "hindi", "sw": "swahili",
        "ur": "urdu", "nl": "dutch", "ja": "japanese", "ko": "korean", "fi": "finnish", "id": "indonesian", "te": "telugu",
        "bn": "bengali", "it": "italian", "pt": "portuguese", "pl": "polish", "ta": "tamil", "he": "hebrew", "af": "afrikaans"}


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def label_ok(label: str, model: str | None) -> bool:
    if not model:
        return False
    a, b = norm(label), norm(model)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    cfg = get_settings()
    vectors = VectorStore()
    chunks = vectors.all_chunks()
    by_id = {c.chunk_id: c for c in chunks}
    db = sqlite3.connect(f"file:{cfg.paths.profile_db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    profiles = [dict(r) for r in db.execute("SELECT * FROM profiles")]
    log = {r["paper_id"]: dict(r) for r in db.execute("SELECT * FROM extraction_log")}
    report: dict = {}

    # ---- corpus and stores
    papers = sorted({c.paper_id for c in chunks})
    n_pdfs = len(list(cfg.paths.papers_dir.glob("*.pdf")))
    bm25 = BM25Store.load(cfg.paths.bm25_path)
    bm25_n = len(getattr(bm25, "chunks", getattr(bm25, "ids", [])) or [])
    sections = Counter(c.section for c in chunks)
    table_chunks = [c for c in chunks if is_structured_table(c.text)]
    dim = len(vectors.col.get(limit=1, include=["embeddings"])["embeddings"][0])
    report["corpus"] = {"pdfs": n_pdfs, "papers_indexed": len(papers), "chunks": len(chunks), "table_chunks": len(table_chunks),
                        "sections": dict(sections), "vector_dim": dim, "chroma_count": vectors.count(), "bm25_docs": bm25_n,
                        "profiles": len(profiles), "papers_with_profiles": len({p["paper_id"] for p in profiles})}
    broken_chunk = [p["profile_id"] for p in profiles if p["chunk_id"] not in by_id]
    broken_methods = [p["profile_id"] for p in profiles if p["methods_chunk_id"] and p["methods_chunk_id"] not in by_id]
    report["integrity"] = {"profiles_pointing_to_a_missing_chunk": len(broken_chunk),
                           "methods_links_to_a_missing_chunk": len(broken_methods),
                           "papers_without_extraction_log": sorted(set(papers) - set(log)),
                           "papers_with_zero_profiles": sorted(pid for pid in papers if not any(p["paper_id"] == pid for p in profiles))}

    # ---- fill rates
    n = len(profiles)
    fill = {f: sum(1 for p in profiles if p[f] not in (None, "")) for f in FIELDS + ("methods_chunk_id", "evidence")}
    report["fill_rates"] = {f: round(100 * v / n, 1) if n else 0 for f, v in fill.items()}
    report["distinct"] = {f: len({p[f] for p in profiles if p[f]}) for f in ("model", "dataset", "metric", "language", "setting")}
    report["top_metrics"] = Counter(p["metric"] for p in profiles).most_common(8)
    report["top_datasets"] = Counter(p["dataset"] for p in profiles if p["dataset"]).most_common(10)

    # ---- grounding and sanity
    grounded = evid = 0
    for p in profiles:
        text = by_id[p["chunk_id"]].text if p["chunk_id"] in by_id else ""
        nums = {float(x) for x in NUMBER.findall(text)}
        grounded += any(abs(p["value"] - x) < 1e-6 for x in nums)
        evid += bool(p["evidence"]) and norm(p["evidence"]) in norm(text)
    bounded = ("accuracy", "f1", "em", "exact match", "precision", "recall", "score")
    out_of_range = [p for p in profiles if p["metric"] and norm(p["metric"]) in bounded and not (0 <= p["value"] <= 100)]
    dupes = n - len({(p["chunk_id"], p["model"], p["dataset"], p["dataset_version"], p["metric"], p["value"], p["setting"], p["language"])
                     for p in profiles})
    report["grounding"] = {"value_found_in_chunk": pct(grounded, n), "evidence_quote_found_in_chunk": pct(evid, n),
                           "out_of_range_percent_metrics": len(out_of_range), "exact_duplicates": dupes}

    # ---- table cell recall and attribution
    prof_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for p in profiles:
        prof_by_chunk[p["chunk_id"]].append(p)
    per_paper: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # cells, found, label_checked, label_ok, lang_checked, lang_ok
    worst: list[tuple[float, str, int, int]] = []
    skipped_statistics = 0
    for c in table_chunks:
        t = parse_table(c.text)
        if t is None:
            continue
        if is_statistics_table(t.caption):                      # not sent to the LLM on purpose (data / model-size tables)
            skipped_statistics += 1
            continue
        cells = found = lab_n = lab_ok = lg_n = lg_ok = 0
        rows = _rows(t)
        mine = prof_by_chunk.get(c.chunk_id, [])
        for r in rows:
            for pair in r.pairs:
                name, _, cell = pair.partition(" = ")
                if not is_numeric_cell(cell):
                    continue
                for num in DECIMAL.findall(cell):
                    cells += 1
                    hits = [p for p in mine if abs(p["value"] - float(num)) < 1e-6]
                    if not hits:
                        continue
                    found += 1
                    lab_n += 1
                    good = [p for p in hits if label_ok(r.label, p["model"])]
                    lab_ok += bool(good)
                    code = norm(name)
                    if code in LANG:
                        lg_n += 1
                        lg_ok += any(p["language"] and norm(p["language"]) in (LANG[code], code) for p in (good or hits))
        s = per_paper[c.paper_id]
        for k, v in enumerate((cells, found, lab_n, lab_ok, lg_n, lg_ok)):
            s[k] += v
        if cells:
            worst.append((found / cells, c.chunk_id, cells, found))
    tot = [sum(v[k] for v in per_paper.values()) for k in range(6)]
    report["tables"] = {"result_tables_checked": len(table_chunks) - skipped_statistics,
                        "statistics_tables_left_out_on_purpose": skipped_statistics,
                        "table_cells_with_a_decimal": tot[0], "cells_that_became_a_profile": tot[1],
                        "cell_recall": pct(tot[1], tot[0]),
                        "row_label_matches_profile_model": pct(tot[3], tot[2]),
                        "language_column_matches_profile_language": pct(tot[5], tot[4]),
                        "language_cells_checked": tot[4]}
    report["tables_by_paper"] = {pid: {"cells": v[0], "recall": pct(v[1], v[0]), "model_ok": pct(v[3], v[2])}
                                 for pid, v in sorted(per_paper.items())}
    report["worst_tables"] = [{"chunk": cid, "cells": cells_, "found": f_} for _, cid, cells_, f_ in sorted(worst)[:12]]

    # ---- speed
    secs = [v["seconds"] for v in log.values()]
    report["speed"] = {"extraction_seconds_total": round(sum(secs)), "per_paper_median": round(sorted(secs)[len(secs) // 2]) if secs else 0,
                       "per_paper_max": round(max(secs)) if secs else 0,
                       "profiles_per_minute": round(60 * n / sum(secs), 1) if secs else 0,
                       "llm": sorted({v["llm"] for v in log.values()})}
    report["per_paper"] = {pid: {"chunks": sum(1 for c in chunks if c.paper_id == pid),
                                 "profiles": log.get(pid, {}).get("profiles", 0), "seconds": round(log.get(pid, {}).get("seconds", 0))}
                           for pid in papers}

    for key, val in report.items():
        print(f"\n## {key}")
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {val}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
