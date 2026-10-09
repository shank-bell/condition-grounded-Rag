"""RAGAS-style evaluation of the answers (faithfulness and answer relevancy) with an independent judge, and the effect of the runtime loop.

  python scripts/eval_ragas.py collect --tag loop_off                                  # answers of the full pipeline, loop off
  python scripts/eval_ragas.py collect --tag loop_same --loop                          # loop on, the answer model judges (same family)
  python scripts/eval_ragas.py collect --tag loop_llama --loop --judge-model llama3.1:8b   # loop on, an independent judge resident
  python scripts/eval_ragas.py judge --tag loop_off --judge llama3.1:8b                # score a stored answer set with a judge
  python scripts/eval_ragas.py report --tags loop_off loop_same loop_llama --judge llama3.1:8b

collect runs the 30 gold questions through `Pipeline` IN THIS PROCESS: stop the API first (two pipelines do not fit in 24 GB of GPU memory).
judge needs only the judge model (stop the pipeline, free the GPU): the faithfulness context is the text of the passages the answer was
written from plus the recorded results that the writer was shown; answer relevancy uses the BGE-M3 embedder. report writes aggregates to
eval/labels/ragas_report.json (the per-question files stay in data/labelling/private/ragas/).
Judging the answers with the answer model's own family (--judge gemma4:12b) shows how much a same-family judge flatters the writer.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from cgrag.config import get_settings
from cgrag.evaluation.answers import sign_test
from cgrag.labelling.sampling import load_chunk_index
from cgrag.llm import OllamaLLM
from cgrag.models import embed
from cgrag.pipeline.conditions import claim_text
from cgrag.pipeline.faithfulness import answer_relevancy, faithfulness
from cgrag.stores.profile_store import ProfileStore

STORE = Path("data/labelling/private/ragas")
GOLD = Path("eval/labels/questions_gold.jsonl")
OUT = Path("eval/labels/ragas_report.json")


def free_gpu_gb() -> float | None:
    """Free GPU memory in GB according to nvidia-smi (None when it cannot be read)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        return None


def preflight(label: str, need_gb: float) -> None:
    """Refuse to start when the GPU cannot hold what is about to be loaded: a full card makes Ollama swap models in and out (slow, and it can
    time out requests) instead of failing cleanly. Stop the API and other jobs first; `nvidia-smi` shows who holds the memory."""
    free = free_gpu_gb()
    print(f"pre-flight: {free:.1f} GB of GPU memory free, {label} needs about {need_gb:.0f} GB" if free is not None else "pre-flight: GPU memory unknown", flush=True)
    if free is not None and free < need_gb:
        sys.exit(f"STOP: only {free:.1f} GB free, {label} needs about {need_gb:.0f} GB. Stop the API / other pipeline jobs first.")


def _load_partial(path: Path, resume: bool) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if resume and path.exists() else []


def gold_questions(limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def collect(args: argparse.Namespace) -> None:
    from cgrag.pipeline.run import Pipeline
    cfg = get_settings()
    preflight("the whole pipeline", 14)
    if args.loop:
        cfg.features.ragas_loop = True
        if args.judge_model is not None:
            cfg.ragas.judge_model = args.judge_model
    STORE.mkdir(parents=True, exist_ok=True)
    partial = STORE / f"{args.tag}.partial.json"
    records = _load_partial(partial, args.resume)
    done = {r["id"] for r in records}
    pipe = Pipeline(cfg)
    pipe.warm_up()
    free = free_gpu_gb()
    print(f"after warm-up: {free:.1f} GB of GPU memory free" if free is not None else "after warm-up: GPU memory unknown", flush=True)
    if free is not None and free < 0.4:
        sys.exit("STOP: the GPU is full after loading the models (below 0.4 GB free); the run would swap models. Use OLLAMA_NUM_PARALLEL=2 or an offline judge.")
    for q in gold_questions(args.limit):
        if q["id"] in done:
            continue
        t0 = time.perf_counter()
        resp = pipe.run(q["question"])
        records.append({"id": q["id"], "planned_type": q.get("planned_type"), "question": q["question"], "answer": resp.answer,
                        "chunks": [s.chunk_id for s in resp.sources], "seconds": round(time.perf_counter() - t0, 1),
                        "loop_faithfulness": resp.faithfulness, "regenerated": resp.regenerated,
                        "rewrites": next((t for t in resp.trace if "faithfulness" in t), "")})
        print(f"{q['id']}: {records[-1]['seconds']} s, loop score {resp.faithfulness}", flush=True)
        partial.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")          # a hang or a crash loses at most one question
    (STORE / f"{args.tag}.json").write_text(json.dumps({"tag": args.tag, "loop": args.loop, "judge_model": cfg.ragas.judge_model or cfg.llm.model,
                                                       "records": records}, indent=1, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print("wrote", STORE / f"{args.tag}.json")


def judge(args: argparse.Namespace) -> None:
    cfg = get_settings()
    preflight("the judge model and the embedder", 8)
    stored = json.loads((STORE / f"{args.tag}.json").read_text(encoding="utf-8"))
    idx = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    profiles = ProfileStore(cfg.paths.profile_db)
    llm = OllamaLLM(cfg, model=args.judge)
    llm.cfg = llm.cfg.model_copy(update={"keep_alive": "2m"})            # the judge must not stay in GPU memory for hours (the machine default is 2 h)
    safe = args.judge.replace(":", "_").replace("/", "_")
    partial = STORE / f"{args.tag}.{safe}.partial.json"
    rows = _load_partial(partial, args.resume)
    done = {r["id"] for r in rows}
    for rec in stored["records"]:
        if rec["id"] in done:
            continue
        context = "\n\n".join(idx.texts.get(cid, "") for cid in rec["chunks"])
        recorded = [claim_text(p) for ps in profiles.for_chunks(rec["chunks"]).values() for p in ps]
        if recorded:
            context += "\n\nRecorded results shown to the writer:\n" + "\n".join(recorded[:60])
        t0 = time.perf_counter()
        f = faithfulness(rec["question"], rec["answer"], context, llm, max_context_chars=cfg.ragas.max_context_chars)
        rel = answer_relevancy(rec["question"], rec["answer"], llm, embed)
        rows.append({"id": rec["id"], "planned_type": rec["planned_type"], "faithfulness": f.score, "statements": len(f.statements),
                     "unsupported": len(f.unsupported), "answer_relevancy": rel, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{rec['id']}: faithfulness {f.score if f.score is None else round(f.score, 2)} ({len(f.statements)} statements), relevancy "
              f"{rel if rel is None else round(rel, 2)}  [{rows[-1]['seconds']} s]", flush=True)
        partial.write_text(json.dumps(rows), encoding="utf-8")
    (STORE / f"{args.tag}.{safe}.json").write_text(json.dumps({"tag": args.tag, "judge": args.judge, "rows": rows}, indent=1), encoding="utf-8")
    partial.unlink(missing_ok=True)
    try:                                                                  # free the GPU for the next phase at once
        llm.client.generate(model=args.judge, prompt="", keep_alive=0)
    except Exception:
        pass


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def report(args: argparse.Namespace) -> None:
    safe = args.judge.replace(":", "_").replace("/", "_")
    out: dict = {"judge": args.judge, "tags": {}}
    per_tag: dict[str, dict[str, float]] = {}
    for tag in args.tags:
        stored = json.loads((STORE / f"{tag}.json").read_text(encoding="utf-8"))
        rows = json.loads((STORE / f"{tag}.{safe}.json").read_text(encoding="utf-8"))["rows"]
        f = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
        rel = [r["answer_relevancy"] for r in rows if r["answer_relevancy"] is not None]
        secs = [r["seconds"] for r in stored["records"]]
        per_tag[tag] = {r["id"]: r["faithfulness"] for r in rows if r["faithfulness"] is not None}
        out["tags"][tag] = {
            "questions": len(rows), "scored": len(f), "faithfulness_mean": _mean(f), "faithfulness_median": round(statistics.median(f), 3) if f else None,
            "share_at_or_above_0.80": round(sum(v >= 0.8 for v in f) / len(f), 3) if f else None, "share_fully_faithful": round(sum(v == 1.0 for v in f) / len(f), 3) if f else None,
            "answer_relevancy_mean": _mean(rel), "seconds_median": round(statistics.median(secs), 1), "seconds_p90": round(sorted(secs)[int(0.9 * (len(secs) - 1))], 1),
            "loop": stored.get("loop"), "writer_judge": stored.get("judge_model"),
            "by_type": {kind: _mean([r["faithfulness"] for r in rows if r["planned_type"] == kind and r["faithfulness"] is not None])
                        for kind in sorted({r["planned_type"] for r in rows if r["planned_type"]})}}
    base = args.tags[0]
    for tag in args.tags[1:]:
        shared = sorted(set(per_tag[base]) & set(per_tag[tag]))
        up = sum(per_tag[tag][i] > per_tag[base][i] for i in shared)
        down = sum(per_tag[tag][i] < per_tag[base][i] for i in shared)
        out["tags"][tag]["against_" + base] = {"questions": len(shared), "higher": up, "lower": down, "sign_test_p": round(sign_test(up, down), 4)}
    print(json.dumps(out, indent=1))
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    existing[f"judge_{safe}"] = out
    OUT.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tag", required=True)
    c.add_argument("--loop", action="store_true", help="switch the runtime RAGAS loop on")
    c.add_argument("--judge-model", default=None, help="the loop's judge (Ollama tag); empty/omitted = the answer model")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--resume", action="store_true", help="continue a run that stopped (reads <tag>.partial.json)")
    j = sub.add_parser("judge")
    j.add_argument("--tag", required=True)
    j.add_argument("--judge", required=True)
    j.add_argument("--resume", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--tags", nargs="+", required=True)
    r.add_argument("--judge", required=True)
    args = ap.parse_args()
    {"collect": collect, "judge": judge, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
