"""HireLens API + web app (one service, one URL).

Endpoints
  GET  /api/health                     is the model configured, which agents exist
  POST /api/runs                       multipart: resume (PDF), job_description, job_url → {run_id}
  GET  /api/runs/{id}                  status, report, error
  GET  /api/runs/{id}/events           Server-Sent Events: live trace events, then "done"
  GET  /api/runs/{id}/trace?format=…   download the execution trace (json | md)
  GET  /api/runs/{id}/report           download the fit report (json)
  GET  /api/sample                     sample job description for the "try the sample" button
  /                                    the web frontend (web/)

Run with:  uvicorn server.main:app --reload
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agents.base import AGENTS
from lib.config import Settings, load_settings
from lib.llm import LLMClient
from lib.schemas import HireLensInput
from server.runs import RunStore
from tools.errors import ToolError
from tools.pdf_tool import MAX_PDF_BYTES, extract_pdf_text

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
SAMPLES = ROOT / "samples" / "inputs"
MAX_JD_CHARS = 20000

app = FastAPI(title="HireLens", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.state.store = RunStore(max_active=int(os.getenv("HIRELENS_MAX_ACTIVE_RUNS", "3")))
app.state.settings_loader = load_settings
app.state.llm_factory = LLMClient          # tests replace this with a scripted model
app.state.background = True                # tests run synchronously


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"message": message})


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health() -> dict:
    settings: Settings = app.state.settings_loader()
    return {
        "ok": True,
        "llm_configured": bool(settings.api_key),
        "provider": settings.provider,
        "model": settings.default_model,
        "agents": ["orchestrator", *AGENTS],
        "active_runs": app.state.store.active(),
    }


@app.get("/api/sample")
def sample() -> dict:
    return {
        "job_description": (SAMPLES / "job_backend_intern.txt").read_text(),
        "vague_job_description": (SAMPLES / "job_vague.txt").read_text().strip(),
        "resume_url": "/api/sample/resume.pdf",
    }


@app.get("/api/sample/resume.pdf")
def sample_resume() -> Response:
    return Response((SAMPLES / "sample_resume.pdf").read_bytes(), media_type="application/pdf",
                    headers={"Content-Disposition": 'inline; filename="sample_resume.pdf"'})


@app.post("/api/runs", status_code=202)
async def create_run(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(""),
    job_url: str = Form(""),
) -> dict:
    settings: Settings = app.state.settings_loader()
    if not settings.api_key:
        raise _error(503, "The server has no model API key configured. Add GEMINI_API_KEY to the environment.")

    job_description, job_url = job_description.strip()[:MAX_JD_CHARS], job_url.strip()
    if not job_description and not job_url:
        raise _error(400, "Paste a job description or give a job posting URL.")
    if job_url and not job_url.lower().startswith(("http://", "https://")):
        raise _error(400, "The job URL should start with http:// or https://.")

    data = await resume.read(MAX_PDF_BYTES + 1)
    try:
        resume_text, pages = extract_pdf_text(data)
    except ToolError as exc:
        status = 413 if len(data) > MAX_PDF_BYTES else 400
        raise _error(status, str(exc)) from exc

    store: RunStore = app.state.store
    client = _client_id(request)
    if store.active_for(client) >= 1:
        raise _error(429, "You already have an analysis running. Wait for it to finish.")
    if store.active() >= store.max_active:
        raise _error(429, "HireLens is busy with other analyses. Try again in a minute.")

    inp = HireLensInput(resume_filename=resume.filename or "resume.pdf", resume_text=resume_text,
                        resume_pages=pages, job_description=job_description, job_url=job_url)
    run = store.start(inp, client, settings, app.state.llm_factory, background=app.state.background)
    return {"run_id": run.id, "events_url": f"/api/runs/{run.id}/events"}


def _get_run(run_id: str):
    run = app.state.store.get(run_id)
    if run is None:
        raise _error(404, "This analysis no longer exists. Start a new one.")
    return run


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = _get_run(run_id)
    return {"run_id": run.id, "status": run.status, "report": run.report, "error": run.error,
            "event_count": len(run.events)}


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    run = _get_run(run_id)
    last = request.headers.get("last-event-id")
    start = int(last) if last and last.isdigit() else 0

    async def generator():
        index, idle = start, 0.0
        yield "retry: 2000\n\n"
        while True:
            if await request.is_disconnected():
                return
            new = run.events_since(index)
            for event in new:
                index += 1
                yield f"id: {index}\nevent: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if new:
                idle = 0.0
            elif run.finished:
                payload = {"status": run.status, "report": run.report, "error": run.error}
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            else:
                await asyncio.sleep(0.25)
                idle += 0.25
                if idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs/{run_id}/trace")
def download_trace(run_id: str, format: str = "json") -> Response:
    run = _get_run(run_id)
    if format == "md":
        return PlainTextResponse(run.trace.to_markdown(), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="hirelens_trace_{run.id}.md"'})
    return Response(run.trace.to_json(), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="hirelens_trace_{run.id}.json"'})


@app.get("/api/runs/{run_id}/report")
def download_report(run_id: str) -> Response:
    run = _get_run(run_id)
    if not run.report:
        raise _error(409, "The report is not ready yet.")
    return Response(json.dumps(run.report, indent=2, ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="hirelens_report_{run.id}.json"'})


# the web app (mounted last so /api/* wins)
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
