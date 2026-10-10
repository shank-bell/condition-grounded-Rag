"""Build the optional data set name list used by the profile repair (src/cgrag/knowledge/pwc.py) from the archived Papers with Code data set list.

  python scripts/build_pwc_names.py            # downloads jul-28-datasets.json.gz (8 MB) from the pwc-archive on Hugging Face if it is not there

Source: https://huggingface.co/datasets/pwc-archive/files (the last public snapshot of paperswithcode.com, 28 July 2025; CC-BY-SA-4.0). Only names and the
task labels are kept (no descriptions). The list is optional: without it the repair still works with the hand-written list of src/cgrag/knowledge/
benchmarks.py; with it, more captions are recognised ("Results on the CoNLL-03 test set" -> CoNLL-2003). It is independent of MetaLead and of the 28
development papers. Output: data/bench/pwc/dataset_names.json (git-ignored, rebuilt by this script).
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/bench/pwc/datasets.json.gz"
OUT = ROOT / "data/bench/pwc/dataset_names.json"
URL = "https://huggingface.co/datasets/pwc-archive/files/resolve/main/jul-28-datasets.json.gz"


def main() -> None:
    SRC.parent.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        urllib.request.urlretrieve(URL, SRC)
    data = json.load(gzip.open(SRC, "rt", encoding="utf-8"))
    names: dict[str, dict] = {}
    for d in data:
        mods = d.get("modalities") or []
        if "Texts" not in mods:
            continue
        tasks = [t["task"] if isinstance(t, dict) else str(t) for t in (d.get("tasks") or [])]
        entry = {"name": d["name"], "tasks": tasks[:8], "n": d.get("num_papers") or 0}
        for alias in [d["name"], d.get("full_name") or ""] + [v for v in (d.get("variants") or []) if isinstance(v, str)][:6]:
            k = re.sub(r"[^a-z0-9]", "", alias.lower())
            if len(k) >= 3 and (k not in names or names[k]["n"] < entry["n"]):
                names[k] = entry
    OUT.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"{len(names)} normalised names from {len(data)} data sets -> {OUT}")


if __name__ == "__main__":
    main()
