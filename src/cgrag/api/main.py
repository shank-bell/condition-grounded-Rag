"""FastAPI: POST /query -> QueryResponse JSON (answer, sources, applicability, contradictions), plus paper upload.

  uvicorn cgrag.api.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    except Exception as err:
        _warm.update(status="failed", error=f"{type(err).__name__}: {err}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True, name="warm-up").start()
    yield


LOCAL_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"]
OPEN_PATHS = {"/health"}
UPLOAD_PATHS = ("/upload", "/jobs")


async def _refuse(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class AccessGuard:
    """Public demo guard (see `[server]` in config.toml; off by default). Pure ASGI so streamed answers pass through untouched. It sits
    INSIDE the CORS layer, so a refusal still carries the CORS headers and the page can read it."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["method"] != "OPTIONS":
            cfg = get_settings().server
            path = scope["path"]
            if cfg.public_mode and path.startswith(UPLOAD_PATHS):
                return await _refuse(send, 403, "Uploads are switched off on the public demo.")
            if cfg.access_code and path not in OPEN_PATHS:
                given = dict(scope["headers"]).get(b"x-access-code", b"")
                if not hmac.compare_digest(given, cfg.access_code.encode("utf-8")):
                    return await _refuse(send, 401, "Access code required.")
        await self.app(scope, receive, send)


app = FastAPI(title="Condition-Grounded Scientific RAG", lifespan=lifespan)
app.add_middleware(AccessGuard)
app.add_middleware(
    CORSMiddleware, allow_origins=[*LOCAL_ORIGINS, *get_settings().server.allowed_origins],
    allow_methods=["*"], allow_headers=["*"])

_run_lock = threading.Lock()
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
    auth_required = bool(get_settings().server.access_code)
    if _warm["status"] == "starting":
        return {"status": "starting", "warm": _warm, "auth_required": auth_required}
    p = _pipeline()
    return {"status": "ok", "llm": p.cfg.llm.model, "chunks": p.vectors.count(), "profiles": p.profiles.count(),
            "papers": len(_paper_rows()), "warm": _warm, "auth_required": auth_required}


@app.get("/auth/check")
def auth_check() -> dict:
    """Reaches the handler only when the access code (if one is set) was right: the page uses it to test a typed code."""
    return {"ok": True}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    with _run_lock:
        return _pipeline().run(req.question, [t.model_dump() for t in req.history] or None)


@app.post("/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """The same answer as POST /query, as server-sent events so the page can show which component is working while it works:
    `queued` (another question is running), `step` (a component starts / ends / is skipped), `note` (a stage's result line), then exactly
    one `result` (the QueryResponse JSON) or an `error`. Each message is `event: <type>` + `data: <json>` + a blank line."""
    events: queue.Queue = queue.Queue()
    history = [t.model_dump() for t in req.history] or None

    def work() -> None:
        try:
            if _run_lock.locked():
                events.put({"type": "queued"})
            with _run_lock:
                response = _pipeline().run(req.question, history, progress=events.put)
            events.put({"type": "result", "response": response.model_dump(mode="json")})
        except Exception as err:
            events.put({"type": "error", "message": f"{type(err).__name__}: {err}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True, name="query-stream").start()

    async def stream():
        while True:
            event = await asyncio.to_thread(events.get)
            if event is None:
                break
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
        except Exception as err:
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
