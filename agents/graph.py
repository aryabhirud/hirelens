"""LangGraph wiring for HireLens.

            ┌──────────────────────── orchestrator ◄──────────────────────┐
            │  (plans, inspects every result, re-plans, arbitrates)       │
            ▼               ▼                ▼                 ▼        │
     resume_parser     job_analyst        matcher ──► critic   career_coach
            │               │  (tool loop)    ▲          │           │
            └───────────────┴─────────────────┼──────────┤───────────┘
                                              └─revise───┘  (Critic → Matcher)
                                                     approve / budget used → orchestrator

Workers always report back to the orchestrator, except the Matcher, which hands
its scores straight to the Critic, and the Critic, which can send them back to
the Matcher for revision.
"""
from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from agents import career_coach, critic, job_analyst, matcher, orchestrator, resume_parser
from agents.base import AgentContext, PipelineState
from lib.llm import LLMError, QuotaExhausted
from lib.schemas import CoachingPlan, FitReport, HireLensInput, SectionScore
from lib.scoring import WEIGHTS, overall_score, verdict_label
from tools.errors import ToolError

WORKERS = {
    "resume_parser": resume_parser.run,
    "job_analyst": job_analyst.run,
    "matcher": matcher.run,
    "critic": critic.run,
    "career_coach": career_coach.run,
}


def _worker_node(name: str, fn: Callable, ctx: AgentContext) -> Callable[[PipelineState], dict]:
    """Wrap a worker so a failure is reported to the orchestrator instead of crashing the run."""

    def node(state: PipelineState) -> dict:
        attempts = dict(state.get("attempts", {}))
        attempts[name] = attempts.get(name, 0) + 1
        try:
            updates = fn(ctx, state)
        except QuotaExhausted:
            raise  # no agent can work without the model; stop the whole run now
        except (LLMError, ToolError, KeyError, ValueError) as exc:
            ctx.trace.log(name, "error", f"{name} failed: {exc}", {"error": str(exc), "attempt": attempts[name]})
            ctx.trace.handoff(name, "orchestrator", "reporting failure")
            return {"attempts": attempts, "last": name, "failed": {"agent": name, "error": str(exc)[:300]}}
        completed = list(state.get("completed", []))
        if name not in completed:
            completed.append(name)
        if name not in {"matcher", "critic"}:
            ctx.trace.handoff(name, "orchestrator", "done")
        return {**updates, "attempts": attempts, "last": name, "completed": completed, "failed": {}}

    return node


def _finalize_node(ctx: AgentContext) -> Callable[[PipelineState], dict]:
    def node(state: PipelineState) -> dict:
        report = build_report(ctx, state)
        ctx.trace.log("orchestrator", "final",
                      f"Report ready: {report.status}"
                      + (f", score {report.overall_score} ({report.verdict})" if report.overall_score is not None else ""),
                      {"report": report.model_dump()})
        return {"report": report.model_dump(), "status": report.status}

    return node


def build_graph(ctx: AgentContext):
    graph = StateGraph(PipelineState)
    graph.add_node("orchestrator", lambda s: orchestrator.run(ctx, s))
    for name, fn in WORKERS.items():
        graph.add_node(name, _worker_node(name, fn, ctx))
    graph.add_node("finalize", _finalize_node(ctx))

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator", lambda s: s["next"], {**{n: n for n in WORKERS}, "finalize": "finalize"}
    )
    for name in ("resume_parser", "job_analyst", "career_coach"):
        graph.add_edge(name, "orchestrator")
    graph.add_conditional_edges(
        "matcher", lambda s: "orchestrator" if s.get("failed") else "critic",
        {"critic": "critic", "orchestrator": "orchestrator"},
    )
    graph.add_conditional_edges(
        "critic", lambda s: critic.route_after(s, ctx.settings.max_critic_revisions),
        {"matcher": "matcher", "orchestrator": "orchestrator"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_pipeline(inp: HireLensInput, ctx: AgentContext) -> FitReport:
    if not inp.job_description.strip() and not inp.job_url.strip():
        raise ValueError("Provide a job description or a job posting URL.")
    app = build_graph(ctx)
    try:
        final_state = _invoke(app, inp)
    except QuotaExhausted as exc:
        ctx.trace.log("orchestrator", "error", f"Run stopped: {exc}", {"reason": "daily_quota_exhausted"})
        raise
    return FitReport.model_validate(final_state["report"])


def _invoke(app, inp: HireLensInput) -> dict:
    return app.invoke(
        {
            "resume_filename": inp.resume_filename,
            "resume_text": inp.resume_text,
            "resume_pages": inp.resume_pages,
            "job_description": inp.job_description,
            "job_url": inp.job_url,
        },
        config={"recursion_limit": 60},
    )


def build_report(ctx: AgentContext, state: PipelineState) -> FitReport:
    base = {"run_id": ctx.trace.run_id, "warnings": state.get("warnings", [])}
    resume = state.get("resume_profile") or {}
    job = state.get("job_profile") or {}
    base.update(candidate_name=resume.get("name", ""), job_title=job.get("title", ""),
                company=job.get("company", ""), job_sources=job.get("sources", []))

    match = state.get("match")
    if state.get("status") == "aborted" or not match:
        return FitReport(**base, status="aborted",
                         message=state.get("message") or "The run stopped before a score could be produced.")

    sections = {k: SectionScore.model_validate(v) for k, v in match["sections"].items()}
    score = overall_score(sections, len(match.get("matched_must_have", [])), len(match.get("missing_must_have", [])))
    critique = state.get("critique") or {}
    coaching = CoachingPlan.model_validate(state["coaching"]) if state.get("coaching") else None
    arbitration = state.get("arbitration") or {}
    message = "Scores verified by the critic." if state.get("critic_approved") else (
        "Matcher and critic disagreed; the orchestrator resolved it "
        f"({arbitration.get('action', 'unverified').replace('_', ' ')})." )
    history: dict[str, list[int]] = {}
    for past in state.get("match_history", []):
        for name, sec in past["sections"].items():
            history.setdefault(name, []).append(sec["score"])
    for name, sec in sections.items():  # arbitration may have changed the final scores
        if history.get(name, [None])[-1] != sec.score:
            history.setdefault(name, []).append(sec.score)
    critic_history = [{**c, "round": i} for i, c in enumerate(state.get("critique_history", []), 1)]
    return FitReport(
        **base,
        status="completed",
        score_history=history,
        critic_history=critic_history,
        message=message,
        overall_score=score,
        verdict=verdict_label(score),
        section_scores=sections,
        weights=WEIGHTS,
        strengths=match.get("strengths", []),
        matched_must_have=match.get("matched_must_have", []),
        skill_gaps=match.get("missing_must_have", []),
        nice_to_have_gaps=match.get("missing_nice_to_have", []),
        critic_rounds=state.get("critic_rounds", 0),
        critic_approved=bool(state.get("critic_approved")),
        critic_summary=critique.get("summary", ""),
        coaching=coaching,
    )
