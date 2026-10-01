"""Read every file of the packages this project imports, in parallel, so the first `import torch` / `import transformers` of the day
does not take 20 minutes.

Why: on this PC the first read of a file costs ~0.25-0.3 s (a scanner checks every file once; the same on C: and D:, a warm read is
~0 ms). `import transformers` opens thousands of files, one after the other, so a cold start took ~25 minutes on 2026-10-01. The
delay is per-file latency, not disk throughput, so reading with many threads at once removes most of it.

  python scripts/prewarm_files.py                 # the packages below, 48 threads
  python scripts/prewarm_files.py --threads 96 --all    # everything under site-packages
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKAGES = ["torch", "transformers", "sentence_transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy", "scipy", "sklearn",
            "chromadb", "onnxruntime", "onnx", "pymupdf", "pymupdf4llm", "pymupdf_layout", "fitz", "pydantic", "pydantic_core", "ollama", "httpx",
            "httpcore", "anyio", "fastapi", "starlette", "uvicorn", "rank_bm25", "openpyxl", "pandas", "regex", "yaml", "requests", "urllib3",
            "charset_normalizer", "tqdm", "filelock", "fsspec", "packaging", "typing_extensions", "sympy", "networkx", "jinja2", "markupsafe",
            "mpmath", "psutil", "grpc", "opentelemetry", "posthog", "overrides", "tenacity", "orjson", "bcrypt", "kubernetes", "pypika", "mmh3"]
SUFFIXES = {".py", ".pyc", ".pyd", ".dll", ".so", ".json", ".txt", ".pth", ".cfg", ".toml", ".pkl"}


def site_packages() -> Path:
    for p in sys.path:
        if p.endswith("site-packages") and Path(p).is_dir():
            return Path(p)
    raise SystemExit("site-packages not found")


def touch(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return len(fh.read(4096))                        # opening is what costs; 4 KB is enough to trigger the scan
    except OSError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--all", action="store_true", help="every package under site-packages, not only the project's")
    ap.add_argument("--dir", type=Path, default=None, help="warm this directory instead (a test)")
    args = ap.parse_args()
    t0 = time.time()
    roots = [args.dir] if args.dir else ([site_packages()] if args.all else [site_packages() / name for name in PACKAGES])
    files = [f for root in roots if root.exists() for f in (root.rglob("*") if root.is_dir() else [root]) if f.is_file()
             and (args.all or args.dir or f.suffix.lower() in SUFFIXES)]
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        n = sum(1 for _ in pool.map(touch, files, chunksize=16))
    print(f"{n} files touched with {args.threads} threads in {time.time() - t1:.1f} s (listing {t1 - t0:.1f} s)")


if __name__ == "__main__":
    main()
