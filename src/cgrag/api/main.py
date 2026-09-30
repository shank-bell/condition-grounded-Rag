"""FastAPI: POST /query -> QueryResponse JSON (answer, sources, applicability, contradictions), plus paper upload.

  uvicorn cgrag.api.main:app --port 8000
"""
from __future__ import annotations

import threading
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_settings
from ..schemas import QueryResponse

app = FastAPI(title="Condition-Grounded Scientific RAG")
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


@lru_cache(maxsize=1)
def _pipeline():
    from ..pipeline.run import Pipeline
    return Pipeline()


@app.get("/health")
def health() -> dict:
    p = _pipeline()
    return {"status": "ok", "llm": p.cfg.llm.model, "chunks": p.vectors.count(), "profiles": p.profiles.count(),
            "papers": len(_paper_rows())}


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
