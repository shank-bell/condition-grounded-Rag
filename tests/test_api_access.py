"""Public demo guard: an access code on every route except /health, uploads switched off, CORS headers on refusals. Off by default."""
import importlib

import pytest
from fastapi.testclient import TestClient

from cgrag.config import ServerConfig, get_settings

ORIGIN = "https://cgrag-demo.vercel.app"


@pytest.fixture
def make_client(monkeypatch):
    """The allowed origins are read when the module loads, so every client reloads it with the patched settings; the real ones come back after."""
    from cgrag.api import main

    def make(**server):
        settings = get_settings().model_copy(update={"server": ServerConfig(**server)})
        monkeypatch.setattr("cgrag.config.get_settings", lambda: settings)
        importlib.reload(main)
        monkeypatch.setattr(main, "_warm_up", lambda: None)
        return TestClient(main.app)

    yield make
    monkeypatch.undo()
    importlib.reload(main)


def test_default_settings_change_nothing(make_client):
    c = make_client()
    assert c.get("/auth/check").json() == {"ok": True}
    assert c.get("/health").json()["auth_required"] is False
    assert c.get("/jobs/none").status_code == 404


def test_access_code_is_required_everywhere_but_health(make_client):
    c = make_client(access_code="open-sesame-123")
    assert c.get("/health").status_code == 200 and c.get("/health").json()["auth_required"] is True
    assert c.get("/auth/check").status_code == 401
    assert c.get("/auth/check", headers={"X-Access-Code": "wrong"}).status_code == 401
    assert c.post("/query", json={"question": "What is BERT?"}).status_code == 401
    assert c.post("/query/stream", json={"question": "What is BERT?"}).status_code == 401
    assert c.get("/auth/check", headers={"X-Access-Code": "open-sesame-123"}).json() == {"ok": True}


def test_a_refusal_still_carries_the_cors_headers_so_the_page_can_read_it(make_client):
    c = make_client(access_code="open-sesame-123", allowed_origins=[ORIGIN])
    r = c.get("/auth/check", headers={"Origin": ORIGIN})
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == ORIGIN
    assert r.json() == {"detail": "Access code required."}


def test_preflight_passes_without_the_code_and_only_for_allowed_origins(make_client):
    c = make_client(access_code="open-sesame-123", allowed_origins=[ORIGIN])
    ask = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-access-code,content-type"}
    ok = c.options("/query/stream", headers={"Origin": ORIGIN, **ask})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert "x-access-code" in ok.headers["access-control-allow-headers"].lower()
    bad = c.options("/query/stream", headers={"Origin": "https://evil.example", **ask})
    assert bad.status_code == 400


@pytest.mark.parametrize("method,path", [("post", "/upload"), ("get", "/jobs/abc12345")])
def test_public_mode_switches_uploads_off_even_with_the_right_code(make_client, method, path):
    c = make_client(access_code="open-sesame-123", public_mode=True)
    r = getattr(c, method)(path, headers={"X-Access-Code": "open-sesame-123"})
    assert r.status_code == 403


def test_public_mode_without_a_code_still_blocks_uploads_and_leaves_queries_open(make_client):
    c = make_client(public_mode=True)
    assert c.post("/upload").status_code == 403
    assert c.get("/auth/check").status_code == 200
