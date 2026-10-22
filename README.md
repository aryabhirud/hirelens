# HireLens

**A multi-agent resume-to-job fit checker.** Upload a resume (PDF), paste a job description or give a job URL, and HireLens returns a verified 0–100 fit score with a section-wise breakdown, skill gaps, improved resume bullets and likely interview questions.

Six LangGraph agents do the work: an **Orchestrator** that plans and re-plans, a **Resume Parser**, a **Job Analyst** that searches the web when a posting is vague, a **Matcher** that scores with evidence quotes, a **Critic** that verifies those quotes and sends inflated scores back for revision, and a **Career Coach**. The web app shows the agents working live, and every step is recorded in a downloadable execution trace.

- Synopsis: [`docs/SYNOPSIS.md`](docs/SYNOPSIS.md)
- Design (architecture diagram, agent roles, orchestration pattern, tech choices): [`docs/DESIGN.md`](docs/DESIGN.md)
- Input/output definitions: [`docs/IO_DEFINITIONS.md`](docs/IO_DEFINITIONS.md)
- Team plan: [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md)
- API reference (when running): `/api/docs`

**Team:** Arya Sanjay Bhirud · Aryan Choudhary · Arun Tati

## Architecture at a glance

```
Browser (web/)  ──POST /api/analyze──►  FastAPI (server/)  ──►  LangGraph agents (agents/)  ──►  Gemini
      ▲                                     │                       │
      └──── Server-Sent Events: live ◄──────┴──── trace events ◄────┘
            agent activity + final report
```

One service serves both the API and the frontend, so there is one deployment and one URL. Each analysis runs inside a single streaming request, which is what lets it run on Vercel's serverless functions.

## Run locally

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then put your key in .env
uvicorn server.main:app --reload
```

Open <http://127.0.0.1:8000>. Use **Try the sample** to run without your own resume, or **try a one-line job post** to watch the Job Analyst search the web for the role.

Get a free API key at <https://aistudio.google.com/apikey> (Gemini). To use Groq instead, set `HIRELENS_PROVIDER=groq` and `GROQ_API_KEY` (free key at <https://console.groq.com/keys>). On a free tier, setting `HIRELENS_MIN_CALL_INTERVAL=4` avoids most rate-limit retries.

### Command line (captures the real execution trace)

```bash
python scripts/run_cli.py samples/inputs/sample_resume.pdf \
    --job-file samples/inputs/job_backend_intern.txt --out samples/outputs/submitted
python scripts/run_cli.py my_resume.pdf --job-file samples/inputs/job_vague.txt   # triggers careers-page search
python scripts/run_cli.py my_resume.pdf --job-url https://company.com/careers/123
```

Each run writes `input.json`, `report.json`, `trace.json` and `trace.md`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Model configured? Which agents exist? |
| POST | `/api/analyze` | Multipart: `resume` (PDF), `job_description`, `job_url`. Runs the analysis in this request and streams Server-Sent Events: `started`, one `trace` event per agent step, then `done` with the report and the full trace. Used by the web app. |
| POST | `/api/runs` | Same input → `{run_id}`; the run continues in the background (for long-running servers) |
| GET | `/api/runs/{id}/events` | Server-Sent Events for a background run. Resumes with `Last-Event-ID`. |
| GET | `/api/runs/{id}` | Status, report, error |
| GET | `/api/runs/{id}/trace?format=json\|md` | Download the execution trace |
| GET | `/api/runs/{id}/report` | Download the fit report |

Errors are returned as `{"error": {"message": "..."}}` with 400 (bad input), 413 (file too large), 429 (busy), 503 (no API key) or 404.

## Tests

```bash
pytest -q
```

87 tests, no API calls and no network. The suite runs the whole multi-agent graph with a scripted model: critic rejection and revision, the critic's policy override, orchestrator arbitration, a failed URL fetch, careers-page search for a vague job, a non-resume upload, a parser retry with a hint, agent failures. It also covers the HTTP API end to end, including the live event stream and reconnection, plus the tools, schemas and scoring. CI runs it on every push.

## Deploy (free, on Vercel)

1. Sign in at <https://vercel.com> with GitHub, then **Add New → Project** and import `aryabhirud/hirelens`. Vercel detects FastAPI and uses `index.py` as the entrypoint; `vercel.json` allows a run up to 300 seconds.
2. Before deploying, open **Environment Variables** and add `GEMINI_API_KEY` (and optionally `HIRELENS_MIN_CALL_INTERVAL` = `4`).
3. Click **Deploy**. The app is live at the project's `*.vercel.app` URL, and every push to `main` redeploys it.

Limits on the free plan: one analysis must finish within 5 minutes, and uploads must be under 4.5 MB (the app asks for 4 MB or less).

**Docker (any other host):** `docker build -t hirelens . && docker run -p 7860:7860 -e GEMINI_API_KEY=... hirelens`

## Project structure

```
index.py              Vercel entrypoint (imports the FastAPI app)
vercel.json           Vercel function settings (300 s limit, files left out of the bundle)
server/
  main.py             FastAPI app: upload, single-request streaming analysis, background runs, downloads, serves web/
  runs.py             background runs and buffered trace events
web/
  index.html          intake, live agent map, report
  styles.css
  app.js              upload, live stream, report rendering (no build step)
agents/
  orchestrator.py     plans, inspects results, re-plans, arbitrates Matcher/Critic disagreement
  resume_parser.py    resume text → ResumeProfile
  job_analyst.py      reason–act loop over fetch_url / web_search → JobProfile
  matcher.py          evidence-backed section scores
  critic.py           verifies evidence with tools, approves or requests revision
  career_coach.py     bullets, interview questions, gap actions
  graph.py            LangGraph wiring and report assembly
  base.py             shared state, context, logged tool calls
tools/                pdf_tool, web_tool, text_tools
lib/                  config, llm client, schemas, scoring, trace
tests/                pytest suite
scripts/              run_cli.py, make_sample_resume.py
docs/                 synopsis, design, I/O definitions, JSON schemas, team plan
samples/              sample inputs and the submitted real run
```

## Notes

- No keys are committed: `.env` is git-ignored; the deployment reads the key from Vercel's environment variables.
- Resumes are processed in memory for a single run and are not stored. Finished runs are kept in memory for a few hours so the trace can be downloaded, then discarded.
- The sample resume is of a fictional person.
- HireLens is an honest first check, not a guarantee of selection.
- Built with the help of AI coding assistants; the team reviewed, tested and can explain every component.
