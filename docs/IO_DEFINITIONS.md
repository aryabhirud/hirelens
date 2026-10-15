# HireLens — Input and Output Definitions

Both contracts are Pydantic models in `lib/schemas.py`. The exact JSON Schemas are generated from them:
`docs/schemas/input.schema.json` and `docs/schemas/output.schema.json`.

## Input

What the user provides in the web app: a resume PDF (≤ 5 MB, text-based) and a job description, a job posting URL, or both. The browser sends them to `POST /api/runs` as `multipart/form-data` (fields `resume`, `job_description`, `job_url`). The backend extracts the PDF text and passes this object to the agents:

| Field | Type | Required | Notes |
|---|---|---|---|
| `resume_filename` | string | yes | Original file name |
| `resume_text` | string (non-empty) | yes | Text extracted from the PDF by `tools/pdf_tool.py` |
| `resume_pages` | integer | no (default 1) | Page count |
| `job_description` | string | one of these two | Pasted job description |
| `job_url` | string (http/https) | one of these two | Job posting URL; the Job Analyst fetches it |

Validation: the PDF must be a real, unencrypted, text-based PDF; at least one of `job_description` / `job_url` must be given; private/local URLs are rejected.

**Sample input** (real, used for the submitted run): `samples/inputs/sample_input.json`, built from `samples/inputs/sample_resume.pdf` (a fictional candidate) and `samples/inputs/job_backend_intern.txt`. Abridged:

```json
{
  "resume_filename": "sample_resume.pdf",
  "resume_text": "Riya Deshmukh\nB.Tech Computer Science (Final Year) · Pune, India · ...\nSummary\nFinal-year computer science student focused on backend development ...",
  "resume_pages": 1,
  "job_description": "Backend Engineering Intern — Nimbus Analytics (Pune, hybrid, 6 months)\n...\nMust have\n- Strong Python programming skills\n- Experience building REST APIs (FastAPI, Django or Flask)\n...",
  "job_url": ""
}
```

A second sample, `samples/inputs/job_vague.txt`, is a one-line posting that makes the Job Analyst search the web for the company's careers page.

## Output

A `FitReport`:

| Field | Type | Meaning |
|---|---|---|
| `run_id` | string | Links the report to its execution trace |
| `status` | `"completed"` \| `"aborted"` | Aborted runs explain why in `message` and have no score |
| `message` | string | Verification status or abort reason |
| `candidate_name`, `job_title`, `company` | string | As extracted |
| `overall_score` | integer 0–100 \| null | Weighted score (see Design §5) |
| `verdict` | string | Strong / Moderate / Weak / Poor fit |
| `section_scores` | object | `skills`, `experience`, `projects`, `education`, `keywords` → `{score: 0–100, reasoning, evidence: [{claim, resume_quote}]}` |
| `weights` | object | Section weights used |
| `strengths` | string[] | |
| `matched_must_have` | string[] | Must-haves the resume demonstrates |
| `skill_gaps` | string[] | Missing must-haves |
| `nice_to_have_gaps` | string[] | Missing nice-to-haves |
| `critic_rounds` | integer | How many times the Critic reviewed the scores |
| `critic_approved` | boolean | False means the Orchestrator arbitrated or the Critic was unavailable |
| `critic_summary` | string | Critic's last verdict |
| `critic_history` | object[] | Each Critic review: `{round, verdict, summary, issues[{section, claim, problem, suggested_score}]}` |
| `score_history` | object | Per section, the score after each Matcher round (and after arbitration), e.g. `{"skills": [95, 78]}` |
| `job_sources` | string[] | Where the job requirements came from (user text, URLs, searches) |
| `coaching` | object \| null | `bullet_suggestions[{original, improved, targets}]`, `interview_questions[{question, category, why_asked}]`, `gap_actions[{gap, action}]` |
| `warnings` | string[] | Degradations during the run (e.g. coach unavailable) |

The browser receives the report at the end of the live event stream (`GET /api/runs/{id}/events`, event `done`) and can download it from `/api/runs/{id}/report`. `critic_history` and `score_history` let the UI show which scores the Critic corrected.

Alongside the report, every run produces an **execution trace** (`trace.json` / `trace.md`) with each event: `step, timestamp, elapsed_s, agent, event (input|plan|decision|tool_call|tool_result|message|handoff|critique|revision|warning|error|final), summary, data`.

**Sample output** (real run): `samples/outputs/submitted/report.json`, with its trace in `samples/outputs/submitted/trace.json`.

> TODO before submitting: run
> `python scripts/run_cli.py samples/inputs/sample_resume.pdf --job-file samples/inputs/job_backend_intern.txt --out samples/outputs/submitted`
> with your API key set, then paste an abridged copy of the resulting `report.json` here. Do not edit the values: the sample must come from a real run.
