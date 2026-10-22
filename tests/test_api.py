"""HTTP API and live event stream, with the scripted model in place of the LLM."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from lib.config import Settings
from lib.trace import Trace
from server.main import app
from server.runs import Run, RunStore
from tests.conftest import SAMPLE_PDF, ScriptedLLM
from tests.test_pipeline import _happy_script


@pytest.fixture
def client(job_text):
    saved = (app.state.settings_loader, app.state.llm_factory, app.state.background, app.state.store)
    app.state.settings_loader = lambda: Settings(api_key="test", provider="test", default_model="scripted")
    app.state.llm_factory = lambda _settings: ScriptedLLM(_happy_script())
    app.state.background = False
    app.state.store = RunStore()
    with TestClient(app) as c:
        yield c
    app.state.settings_loader, app.state.llm_factory, app.state.background, app.state.store = saved


def _post(client, pdf: bytes | None = None, **form):
    files = {"resume": ("resume.pdf", pdf if pdf is not None else SAMPLE_PDF.read_bytes(), "application/pdf")}
    return client.post("/api/runs", files=files, data=form)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["llm_configured"] is True
    assert body["agents"] == ["orchestrator", "resume_parser", "job_analyst", "matcher", "critic", "career_coach"]


def test_frontend_and_samples_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "HireLens" in page.text
    assert client.get("/app.js").status_code == 200
    assert "Must have" in client.get("/api/sample").json()["job_description"]
    assert client.get("/api/sample/resume.pdf").content.startswith(b"%PDF")


def test_full_run_streams_events_and_returns_report(client, job_text):
    res = _post(client, job_description=job_text)
    assert res.status_code == 202
    run_id = res.json()["run_id"]

    status = client.get(f"/api/runs/{run_id}").json()
    assert status["status"] == "completed" and status["report"]["overall_score"] == 74

    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        body = "".join(stream.iter_text())
    trace_events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and '"step"' in line]
    assert {e["agent"] for e in trace_events} >= {"orchestrator", "resume_parser", "job_analyst",
                                                   "matcher", "critic", "career_coach"}
    assert "event: done" in body

    trace = client.get(f"/api/runs/{run_id}/trace?format=json")
    assert trace.headers["content-disposition"].endswith(f'hirelens_trace_{run_id}.json"')
    assert json.loads(trace.text)["event_count"] == len(trace_events)
    assert "| orchestrator |" in client.get(f"/api/runs/{run_id}/trace?format=md").text
    assert client.get(f"/api/runs/{run_id}/report").json()["run_id"] == run_id


def test_event_stream_resumes_after_reconnect(client, job_text):
    run_id = _post(client, job_description=job_text).json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "10"}) as stream:
        body = "".join(stream.iter_text())
    first_id = next(line for line in body.splitlines() if line.startswith("id: "))
    assert first_id == "id: 11"


@pytest.mark.parametrize("pdf, form, status, message", [
    (b"not a pdf", {"job_description": "Python developer"}, 400, "not a PDF"),
    (None, {}, 400, "job description"),
    (None, {"job_url": "ftp://jobs"}, 400, "http"),
    (b"%PDF" + b"0" * (5 * 1024 * 1024 + 10), {"job_description": "x"}, 413, "larger than 5 MB"),
], ids=["not-pdf", "missing-job", "invalid-job-url", "oversized-pdf"])
def test_bad_input_gets_clear_errors(client, pdf, form, status, message):
    res = _post(client, pdf, **form)
    assert res.status_code == status
    assert message in res.json()["error"]["message"]


def test_missing_api_key_is_reported(client, job_text):
    app.state.settings_loader = lambda: Settings(api_key=None)
    res = _post(client, job_description=job_text)
    assert res.status_code == 503 and "API key" in res.json()["error"]["message"]


def test_one_active_run_per_visitor(client, job_text):
    busy = Run(id="busy", client="testclient", created=time.time(), trace=Trace())
    app.state.store._runs["busy"] = busy
    res = _post(client, job_description=job_text)
    assert res.status_code == 429


def test_failed_run_reports_error_instead_of_hanging(client, job_text):
    def broken_factory(_settings):
        raise RuntimeError("model client could not start")

    app.state.llm_factory = broken_factory
    run_id = _post(client, job_description=job_text).json()["run_id"]
    status = client.get(f"/api/runs/{run_id}").json()
    assert status["status"] == "failed" and "could not start" in status["error"]


def test_unknown_run_is_404(client):
    res = client.get("/api/runs/nope")
    assert res.status_code == 404 and "Start a new one" in res.json()["error"]["message"]
