"""Download the public MetaLead benchmark (Timmer, Bölücü, Wan; EACL 2026; https://github.com/RoelTim/metalead): the human annotations and the 43 papers.

  python scripts/download_metalead.py            # -> data/bench/metalead/annotations/metalead_annotations.json and data/bench/metalead/papers/*.pdf
  CGRAG_OVERLAY=config/bench_metalead.toml python scripts/ingest.py        # extract them into their own index (about 2 h on the college PC)
  CGRAG_OVERLAY=config/bench_metalead.toml python scripts/eval_metalead.py  # coverage / agreement against the human gold

The papers are third-party documents: they are downloaded for evaluation only and are NOT part of this repository (data/bench/ is git-ignored). The paper
URLs come from the annotation file itself (arXiv, ACL Anthology, PMLR, FLAIRS).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/bench/metalead"
ANNOTATIONS = "https://raw.githubusercontent.com/RoelTim/metalead/main/annotations/metalead_annotations.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (research benchmark download; condition-grounded-rag)"}


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90).read()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"   retry after {type(e).__name__}", flush=True)
            time.sleep(3)
    raise RuntimeError("unreachable")


def main() -> None:
    (BASE / "annotations").mkdir(parents=True, exist_ok=True)
    (BASE / "papers").mkdir(parents=True, exist_ok=True)
    ann_path = BASE / "annotations/metalead_annotations.json"
    if not ann_path.exists():
        ann_path.write_bytes(fetch(ANNOTATIONS))
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    print(f"{len(ann)} papers, {sum(len(v['TDMs']) for v in ann.values())} annotated results")
    ok = bad = 0
    for name, v in ann.items():
        out = BASE / "papers" / name
        if out.exists() and out.stat().st_size > 50_000:
            ok += 1
            continue
        url = v["PaperURL"].replace("https://arxiv.org/pdf/", "https://export.arxiv.org/pdf/").replace("/article/view/", "/article/download/")
        try:
            data = fetch(url)
            if not data.startswith(b"%PDF"):
                raise ValueError("not a PDF")
            out.write_bytes(data)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print("FAILED", name, url, e, flush=True)
            bad += 1
        time.sleep(1.0)
    print(f"downloaded or present: {ok}, failed: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
