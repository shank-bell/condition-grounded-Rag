"""The live "chain of execution" (9 Oct 2026): `Pipeline.run(progress=...)` reports each component as it works, and POST /query/stream sends
those events as server-sent events. No model is needed: the timer and the endpoint are tested with stand-ins."""
import inspect
import json

import pytest
from fastapi.testclient import TestClient

from cgrag.pipeline import run as run_module
from cgrag.pipeline.run import STEP_INFO, Pipeline, _Timer
from cgrag.schemas import QueryResponse


def test_every_timed_step_has_a_name_and_a_description():
    for step_id, (stage, component, blurb) in STEP_INFO.items():
        assert stage >= 1 and component and blurb, step_id
    assert set(STEP_INFO) == {"1_understand", "2_plan", "3_refine", "4_retrieve", "5_rerank", "2b_replan", "6_applicability",
                              "7_contradiction", "8_generate", "9_critic", "9b_ragas"}      # 9b_ragas only runs with [features] ragas_loop = true


def test_timer_reports_a_step_start_and_end_and_still_times_it():
    events: list[dict] = []
    t = _Timer(events.append)
    with t("4_retrieve"):
        pass
    assert [(e["type"], e["status"], e["id"]) for e in events] == [("step", "running", "4_retrieve"), ("step", "done", "4_retrieve")]
    assert events[0]["component"] == "Hybrid retrieval" and events[0]["stage"] == 4 and events[0]["blurb"]
    assert events[1]["ms"] >= 0 and "4_retrieve" in t.ms
    assert all("t_ms" in e for e in events)


def test_timer_reports_skips_and_notes_and_ignores_a_broken_listener():
    events: list[dict] = []
    t = _Timer(events.append)
    t.skip("3_refine", "simple question")
    t.note(6, "coverage 0.67")
    assert events[0]["status"] == "skipped" and events[0]["reason"] == "simple question"
    assert events[1] == {"type": "note", "stage": 6, "text": "coverage 0.67", "t_ms": events[1]["t_ms"]}

    def broken(_event):
        raise RuntimeError("the page went away")

    quiet = _Timer(broken)
    with quiet("8_generate"):                      # must neither raise nor lose the timing
        pass
    assert "8_generate" in quiet.ms


def test_timer_without_a_listener_is_what_it_was():
    t = _Timer()
    with t("1_understand"):
        pass
    t.skip("3_refine", "x")
    t.note(1, "x")
    assert list(t.ms) == ["1_understand"]


def test_run_accepts_a_progress_listener_and_keeps_its_old_arguments():
    params = inspect.signature(Pipeline.run).parameters
    assert list(params)[:4] == ["self", "question", "history", "stop_after"] and params["progress"].default is None
    assert run_module.re is not None


class _StubPipeline:
    def run(self, question, history=None, stop_after=None, progress=None):
        if progress is not None:
            progress({"type": "step", "id": "1_understand", "stage": 1, "component": "Query understanding", "status": "running", "t_ms": 0})
            progress({"type": "step", "id": "1_understand", "stage": 1, "component": "Query understanding", "status": "done", "ms": 5, "t_ms": 5})
            progress({"type": "note", "stage": 1, "text": "intent=result", "t_ms": 6})
        return QueryResponse(question=question, answer="ok", sources=[], analysis=None, applicability=None, contradictions=[],
                             claim_checks=[], regenerated=False, trace=["1 intent=result"], timings_ms={"total": 6.0})


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def client(monkeypatch):
    from cgrag.api import main
    monkeypatch.setattr(main, "_pipeline", lambda: _StubPipeline())
    monkeypatch.setattr(main, "_warm_up", lambda: None)
    return TestClient(main.app)


def test_stream_sends_steps_notes_and_exactly_one_result_last(client):
    with client.stream("POST", "/query/stream", json={"question": "What is BERT?", "history": []}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _events("".join(r.iter_text()))
    assert [e for e, _ in events] == ["step", "step", "note", "result"]
    assert events[0][1]["status"] == "running" and events[1][1]["ms"] == 5
    final = events[-1][1]
    assert final["type"] == "result" and final["response"]["answer"] == "ok" and final["response"]["trace"] == ["1 intent=result"]


def test_stream_reports_an_error_instead_of_a_dead_stream(client, monkeypatch):
    from cgrag.api import main

    class Broken:
        def run(self, *args, **kwargs):
            raise ValueError("boom")

    monkeypatch.setattr(main, "_pipeline", lambda: Broken())
    with client.stream("POST", "/query/stream", json={"question": "What is BERT?", "history": []}) as r:
        events = _events("".join(r.iter_text()))
    assert events == [("error", {"type": "error", "message": "ValueError: boom"})]


def test_plain_query_endpoint_is_unchanged(client):
    r = client.post("/query", json={"question": "What is BERT?", "history": []})
    assert r.status_code == 200 and r.json()["answer"] == "ok"
