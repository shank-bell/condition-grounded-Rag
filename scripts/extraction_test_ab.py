"""Blind A/B sheet and scorer for the profile-repair test set (11 Oct 2026); see scripts/make_extraction_testset.py.

  python scripts/extraction_test_ab.py sheet --seed 7      # writes ab_sheet.txt (fields of both versions as A / B in random order) + ab_map.json (private)
  python scripts/extraction_test_ab.py score               # reads ab_labels.jsonl (the labeller's verdicts) and scores stored vs repaired

The labeller judges every field of A and of B against the paper (OK / WRONG / MISSING, with `imprecise` for a true but vague value, exactly the rules of
job A), without knowing which letter is the stored and which the repaired profile; the map is read only by `score`. Identical fields are shown once and
get one verdict for both. Scores: field-level precision / recall / F1 (lenient: imprecise = right; strict: imprecise = wrong) and the share of fully right
profiles, for both versions, with a paired sign test on the items whose fully-right verdict differs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data/labelling/private/extraction_test"          # --dir changes it (extraction_test2 = the clean set)
FIELDS = ["model", "dataset", "dataset_version", "metric", "value", "language", "task", "model_size", "setting"]


def sheet(seed: int) -> None:
    rng = random.Random(seed)
    versions = [json.loads(x) for x in open(DIR / "versions.jsonl", encoding="utf-8")]
    ab, lines = {}, []
    for v in versions:
        a_is_stored = rng.random() < 0.5
        ab[v["id"]] = "stored" if a_is_stored else "repaired"
        A, B = (v["stored"], v["repaired"]) if a_is_stored else (v["repaired"], v["stored"])
        parts = []
        for f in FIELDS:
            if (A.get(f) or None) == (B.get(f) or None):
                parts.append(f"{f}={A.get(f)!r}")
            else:
                parts.append(f"{f}: A={A.get(f)!r} | B={B.get(f)!r}")
        lines.append(f"{v['id']}: " + "; ".join(parts))
    (DIR / "ab_map.json").write_text(json.dumps(ab, indent=0), encoding="utf-8")
    (DIR / "ab_sheet.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(lines)} items -> {DIR / 'ab_sheet.txt'} (map kept in ab_map.json, not to be read before scoring)")


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _sign_p(plus: int, minus: int) -> float:
    n = plus + minus
    if n == 0:
        return 1.0
    k = min(plus, minus)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def score() -> None:
    ab = json.loads((DIR / "ab_map.json").read_text(encoding="utf-8"))
    items = {json.loads(x)["id"]: json.loads(x) for x in open(DIR / "items.jsonl", encoding="utf-8")}
    labels = [json.loads(x) for x in open(DIR / "ab_labels.jsonl", encoding="utf-8") if x.strip()]
    res = {}
    for strict in (False, True):
        tally = {"stored": {f: [0, 0, 0] for f in FIELDS}, "repaired": {f: [0, 0, 0] for f in FIELDS}}
        full = {"stored": [], "repaired": []}
        groups = {"stored": {}, "repaired": {}}
        for lab in labels:
            if lab.get("not_a_result"):
                continue
            for letter in ("A", "B"):
                version = ab[lab["id"]] if letter == "A" else ("repaired" if ab[lab["id"]] == "stored" else "stored")
                bad = set(lab.get(f"{letter}_wrong", [])) | set(lab.get("both_wrong", []))
                miss = set(lab.get(f"{letter}_missing", [])) | set(lab.get("both_missing", []))
                vague = set(lab.get(f"{letter}_imprecise", [])) | set(lab.get("both_imprecise", []))
                if strict:
                    bad |= vague
                for f in FIELDS:
                    t = tally[version][f]
                    if f in bad:
                        t[1] += 1
                        t[2] += 1
                    elif f in miss:
                        t[2] += 1
                    elif lab.get(f"{letter}_values", {}).get(f, True) is not None:
                        t[0] += 1
                ok = not (bad | miss)
                full[version].append(ok)
                groups[version].setdefault(items[lab["id"]]["group"], []).append(ok)
        out = {}
        for version in ("stored", "repaired"):
            tp = sum(t[0] for t in tally[version].values())
            fp = sum(t[1] for t in tally[version].values())
            fn = sum(t[2] for t in tally[version].values())
            out[version] = {"micro": [round(100 * x, 1) for x in _f1(tp, fp, fn)],
                            "per_field_f1": {f: round(100 * _f1(*tally[version][f])[2], 1) for f in FIELDS},
                            "fully_right": f"{sum(full[version])}/{len(full[version])} = {100 * sum(full[version]) / max(1, len(full[version])):.1f} %",
                            "fully_right_by_group": {g: f"{sum(v)}/{len(v)}" for g, v in groups[version].items()}}
        plus = sum(1 for s, r in zip(full["stored"], full["repaired"]) if r and not s)
        minus = sum(1 for s, r in zip(full["stored"], full["repaired"]) if s and not r)
        out["fully_right_sign_test"] = {"repaired_better": plus, "stored_better": minus, "p": round(_sign_p(plus, minus), 4)}
        res["strict" if strict else "lenient"] = out
    res["n_items"] = len(labels)
    res["not_a_result"] = sum(1 for lab in labels if lab.get("not_a_result"))
    (ROOT / f"eval/labels/extraction_repair_{DIR.name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "score"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dir", default=None, help="folder of the test set (default data/labelling/private/extraction_test)")
    a = ap.parse_args()
    if a.dir:
        DIR = Path(a.dir)
    sheet(a.seed) if a.cmd == "sheet" else score()
