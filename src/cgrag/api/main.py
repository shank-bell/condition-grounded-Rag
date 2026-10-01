"""FastAPI: POST /query -> QueryResponse JSON (answer, sources, applicability, contradictions), plus paper upload.

  uvicorn cgrag.api.main:app --port 8000
"""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_settings
from ..schemas import QueryResponse

_warm: dict = {"status": "starting", "seconds": None, "models": {}, "error": None}


def _warm_up() -> None:
    """Build the pipeline and load every model in the background, so the server answers /health at once and the first question
    finds the models loaded (a cold first question took 30-100 s)."""
    t0 = time.perf_counter()
    try:
        _warm["models"] = _pipeline().warm_up()
        _warm.update(status="ready", seconds=round(time.perf_counter() - t0, 1))
    except Exception as err:                       # a failed warm-up must not stop the server: questions still load models on demand
        _warm.update(status="failed", error=f"{type(err).__name__}: {err}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True, name="warm-up").start()
    yield


app = FastAPI(title="Condition-Grounded Scientific RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"])

_run_lock = threading.Lock()        # one question at a time: the local models share one GPU
_ingest_lock = threading.Lock()
_jobs: dict[str, dict] = {}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


_pipeline_lock = threading.Lock()
_pipeline_obj = None


def _pipeline():
    """One shared pipeline; the lock stops a question that arrives during warm-up from building a second one."""
    global _pipeline_obj
    with _pipeline_lock:
        if _pipeline_obj is None:
            from ..pipeline.run import Pipeline
            _pipeline_obj = Pipeline()
        return _pipeline_obj


@app.get("/health")
def health() -> dict:
    """Light: reports the warm-up state; counts appear once the pipeline exists (the UI can show "warming up")."""
    if _warm["status"] == "starting":
        return {"status": "starting", "warm": _warm}
    p = _pipeline()
    return {"status": "ok", "llm": p.cfg.llm.model, "chunks": p.vectors.count(), "profiles": p.profiles.count(),
            "papers": len(_paper_rows()), "warm": _warm}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    with _run_lock:
        return _pipeline().run(req.question, [t.model_dump() for t in req.history] or None)


def _paper_rows() -> list[dict]:
    p = _pipeline()
    papers: dict[str, dict] = {}
    for c in p.vectors.all_chunks():
        row = papers.setdefault(c.paper_id, {"paper_id": c.paper_id, "title": c.paper_title, "chunks": 0, "profiles": 0})
        row["chunks"] += 1
    for prof in p.profiles.all():
        if prof.paper_id in papers:
            papers[prof.paper_id]["profiles"] += 1
    return sorted(papers.values(), key=lambda r: r["paper_id"])


@app.get("/papers")
def papers() -> list[dict]:
    return _paper_rows()


def _ingest_job(job_id: str, pdf: Path) -> None:
    from ..ingestion.pipeline import Ingestor
    with _ingest_lock:
        _jobs[job_id]["status"] = "running"
        try:
            p = _pipeline()
            ing = Ingestor(p._agent_llm(p.cfg.agents.extractor_model), p.vectors, p.profiles)
            report = ing.ingest(pdf, force=True)
            ing.rebuild_keyword_index()
            p.reload_indexes()
            _jobs[job_id] = {"status": "done", "paper_id": report.paper_id, "title": report.title,
                             "chunks": report.n_chunks, "profiles": report.n_profiles}
        except Exception as err:            # surface the failure to the client instead of a silent dead job
            _jobs[job_id] = {"status": "failed", "error": f"{type(err).__name__}: {err}"}


@app.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES or not data.startswith(b"%PDF"):
        raise HTTPException(400, "Not a PDF, or larger than 40 MB.")
    stem = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in Path(file.filename).stem)[:80] or "paper"
    target = get_settings().paths.papers_dir / f"{stem}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "queued", "paper_id": stem}
    background.add_task(_ingest_job, job_id, target)
    return {"job_id": job_id, "paper_id": stem}


@app.get("/jobs/{job_id}")
def job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job.")
    return _jobs[job_id]
