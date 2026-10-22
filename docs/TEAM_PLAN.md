# HireLens — Team Plan

The project was built in three daily stages. Each day, each member built and committed one part from their own laptop and GitHub account, through a pull request reviewed by a teammate. At the end of every day the repository installed and passed its tests.

## Who built what

| | Arya Sanjay Bhirud | Aryan Choudhary | Arun Tati |
|---|---|---|---|
| **Day 1: foundations** | Repo setup, execution trace logger, scoring rules, evidence-verification and keyword tools, their tests | LLM client, provider settings, schemas for every agent hand-off, their tests | PDF extraction, web fetch/search tools, sample resume and jobs, test fixtures, their tests |
| **Day 2: the agents** | Matcher agent, Critic agent (tool checks, LLM review, approval policy) | Shared agent context and logged tool calls, Resume Parser agent, Job Analyst agent (reason–act loop) | Career Coach agent, unit tests for every worker agent, CI workflow |
| **Day 3: orchestration and app** | Orchestrator agent (plan, re-plan, arbitrate), LangGraph wiring, end-to-end pipeline tests, design doc, synopsis | FastAPI backend with live event stream, API tests, CLI trace capture, I/O definitions and JSON schemas | Web frontend (intake, live agent map, report), Dockerfile, README, team plan |

## Areas of ownership for Q&A

**Arya**: the orchestration pattern (orchestrator–worker with plan-and-replan plus the Matcher⇄Critic loop), when the orchestrator calls the LLM versus follows its plan, how arbitration works after two revisions, how `verify_evidence` tolerates PDF line breaks but rejects invented quotes, why the overall score is computed in code.

**Aryan**: how the Job Analyst decides between fetching, searching and finishing; what happens when a URL fails; why every hand-off is a Pydantic schema; how the LLM client handles rate limits and malformed JSON; switching between Gemini and Groq; how the backend runs analyses in the background and streams events over SSE, and resumes after a dropped connection.

**Arun**: how the tests run the whole multi-agent system without calling an LLM; which failure cases are tested; how the frontend turns the event stream into the live agent map; why model output is inserted with `textContent` and never `innerHTML`; how the report shows the critic's corrections; how the API key stays out of the repo and the Docker image.

**Everyone**: why HireLens needs multiple agents rather than one prompt. One prompt grades its own work. Here the critic checks the matcher's claims against the resume with tools, the analyst goes and finds missing job details, and the orchestrator recovers when something fails.
