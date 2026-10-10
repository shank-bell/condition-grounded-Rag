"""Plain-LLM extraction baseline (11 Oct 2026): what a user gets by simply giving the paper to the same local model.

Same model as the pipeline's extractor (gemma4:12b), MetaLead's own extraction prompt (github.com/RoelTim/metalead, prompts.yaml `first_extraction`), raw
PyMuPDF text with the reference list cut off (as MetaLead did), NO table recognition, NO retrieval cards, NO repair. The text is cut into windows of
~6,000 characters because the model is served with an 8k context (MetaLead's API models read the whole paper at once; the original SciLead baseline used
text filtering for the same reason). Output: {paper_id: [ {task, dataset, train_dataset, metric, value, type} ]}, scored by `scripts/eval_metalead.py --pred`.

  python scripts/baseline_llm_extract.py --papers data/bench/metalead/papers --out data/bench/metalead/baseline_gemma.json --workers 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.llm import OllamaLLM  # noqa: E402

PROMPT = """You are given several parts of a research paper as input.
Extract these tuples for all experimental results.
Each tuple must contain:
- a single task, single train dataset, single test dataset, single metric, single numeric result, experiment-type label

Ensure tuples are not aggregated and contain a single value for each field.

Use the following JSON format:
{{"results": [
    {{
        "Task": "Task name",
        "Train-Dataset": "Train dataset name",
        "Test-Dataset": "Test dataset name",
        "Metric": "Metric name",
        "Result": "Result score",
        "Experiment-Type": "Experiment type label (\\"proposed method\\", \\"baseline\\", or \\"variation of proposed method\\")"
    }}
]}}

Paper Text: {paper_text}"""


class _Tuple(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    task: str = Field("", alias="Task")
    train_dataset: str = Field("", alias="Train-Dataset")
    test_dataset: str = Field("", alias="Test-Dataset")
    metric: str = Field("", alias="Metric")
    result: str = Field("", alias="Result")
    experiment_type: str = Field("", alias="Experiment-Type")


class _Out(BaseModel):
    results: list[_Tuple] = []


_REFS = re.compile(r"\n\s*(?:references|bibliography)\s*\n", re.I)


def paper_text(pdf: Path) -> str:
    doc = fitz.open(pdf)
    text = "\n".join(page.get_text("text", sort=True) for page in doc)
    cut = None
    for m in _REFS.finditer(text):
        if m.start() > 0.4 * len(text):
            cut = m.start()
    return text[:cut] if cut else text


def windows(text: str, size: int = 6000) -> list[str]:
    out, cur, n = [], [], 0
    for line in text.split("\n"):
        if n + len(line) > size and cur:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def number(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def run_window(llm: OllamaLLM, text: str, depth: int = 0) -> list[dict]:
    """One window; when the model's answer is cut off (invalid JSON: a dense table window has more results than the output limit) the window is split
    in two halves at a line border and each half is asked again (at most three levels), so no result is lost to the output limit."""
    try:
        out, _ = llm.structured(PROMPT.format(paper_text=text), _Out, temperature=0.0, retries=1, max_tokens=3500)
    except Exception as e:  # noqa: BLE001
        if depth < 3 and len(text) > 1200:
            lines = text.split("\n")
            mid = len(lines) // 2
            print(f"   window split (depth {depth}): {str(e)[:60]}", flush=True)
            return run_window(llm, "\n".join(lines[:mid]), depth + 1) + run_window(llm, "\n".join(lines[mid:]), depth + 1)
        print("   window failed:", str(e)[:120], flush=True)
        return []
    rows = []
    for t in out.results:
        v = number(t.result)
        if v is None:
            continue
        rows.append({"task": t.task or None, "dataset": t.test_dataset or None, "train_dataset": t.train_dataset or None, "metric": t.metric or None,
                     "value": v, "type": t.experiment_type or None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(ROOT / "data/bench/metalead/papers"))
    ap.add_argument("--out", default=str(ROOT / "data/bench/metalead/baseline_gemma.json"))
    ap.add_argument("--model", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--window", type=int, default=6000)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    out_path = Path(args.out)
    done: dict[str, list[dict]] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    pdfs = [p for p in sorted(Path(args.papers).glob("*.pdf")) if p.stem not in done and (not args.only or p.stem in args.only)]
    llm = OllamaLLM(model=args.model or None)
    print(f"model={llm.cfg.model} papers to do={len(pdfs)} already done={len(done)}", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    jobs = []
    meta: dict[str, int] = {}
    for pdf in pdfs:
        ws = windows(paper_text(pdf), args.window)
        meta[pdf.stem] = len(ws)
        jobs.extend((pdf.stem, i, w) for i, w in enumerate(ws))
    print(f"{len(jobs)} windows", flush=True)
    results: dict[str, dict[int, list[dict]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_window, llm, w): (stem, i) for stem, i, w in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            stem, i = futs[fut]
            results.setdefault(stem, {})[i] = fut.result()
            if len(results[stem]) == meta[stem]:
                rows, seen = [], set()
                for k in sorted(results[stem]):
                    for r in results[stem][k]:
                        sig = (str(r["dataset"]).lower(), str(r["metric"]).lower(), round(r["value"], 3))
                        if sig not in seen:
                            seen.add(sig)
                            rows.append(r)
                with lock:
                    done[stem] = rows
                    out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
                print(f"[{n}/{len(jobs)} windows] {stem}: {len(rows)} results  ({time.time() - t0:.0f}s)", flush=True)
    print("finished", len(done), "papers", flush=True)


if __name__ == "__main__":
    main()
