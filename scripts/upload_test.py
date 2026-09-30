"""Exercise the running API end to end: /health, /papers, /upload a PDF, poll /jobs, /query about it, /papers again.

  uvicorn cgrag.api.main:app --port 8000        # in another terminal (needs no ingest running)
  python scripts/upload_test.py path/to/paper.pdf "A question the new paper can answer"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pdf, question = Path(sys.argv[1]), sys.argv[2]
    t = time.time()
    print("HEALTH", requests.get(f"{BASE}/health", timeout=600).json(), f"({time.time() - t:.0f}s incl. model load)")
    before = requests.get(f"{BASE}/papers", timeout=60).json()
    print("PAPERS before:", len(before))
    with pdf.open("rb") as fh:
        r = requests.post(f"{BASE}/upload", files={"file": (pdf.name, fh, "application/pdf")}, timeout=60)
    print("UPLOAD", r.status_code, r.json())
    job = r.json()["job_id"]
    t = time.time()
    while True:
        s = requests.get(f"{BASE}/jobs/{job}", timeout=30).json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(5)
    print(f"JOB after {time.time() - t:.0f}s:", s)
    after = requests.get(f"{BASE}/papers", timeout=60).json()
    print("PAPERS after:", len(after), [p for p in after if p["paper_id"] == s.get("paper_id")])
    t = time.time()
    q = requests.post(f"{BASE}/query", json={"question": question}, timeout=600).json()
    print(f"QUERY ({time.time() - t:.1f}s):", q["answer"][:900])
    print("SOURCES:", [(x["paper_id"], x["page"], x["section"]) for x in q["sources"]])
    if q.get("applicability"):
        print("APPLICABILITY:", {k: q["applicability"][k] for k in ("coverage", "missing")})


if __name__ == "__main__":
    main()
