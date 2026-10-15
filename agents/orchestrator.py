"""Orchestrator agent: plans the run, routes work, and re-plans when something goes wrong.

Pattern: orchestrator-worker with plan-and-replan. The orchestrator makes an
LLM plan at the start, then after every worker it inspects the result. When a
result is fine it moves to the next planned step without spending an LLM call;
when something unexpected happens (weak resume extraction, a vague job, a
failed agent, an unresolved Matcher/Critic disagreement) it asks the LLM what
to do and records the reasoning in the trace.
"""
from __future__ import annotations

from agents import job_analyst, resume_parser
from agents.base import AGENTS, AgentContext, PipelineState, to_prompt_json
from lib.llm import LLMError

NAME = "orchestrator"
MAX_ATTEMPTS = 2  # per worker, including the first run

PLAN_SYSTEM = f"""You are the Orchestrator of HireLens, a multi-agent system that tells a job seeker how well
their resume fits a specific job. You coordinate these agents:
- resume_parser: turns resume text into structured data
- job_analyst: works out the job's requirements; can fetch the job URL and search the web for missing details
- matcher: scores the fit per section with evidence quotes from the resume
- critic: verifies the matcher's evidence and sends inflated scores back for revision
- career_coach: writes improved resume bullets, interview questions and gap actions

Look at the input summary and plan the run. Return JSON:
{{"reasoning": "...",
  "plan": ["agent names in order"],
  "notes_for_agents": {{"resume_parser": "...", "job_analyst": "..."}}  // specific guidance, or empty strings
}}
The plan must use these agents: {", ".join(AGENTS)}. The parser and analyst must run before the matcher,
the critic after the matcher, the coach last. Use notes_for_agents to pass on anything you notice
(e.g. the job text is very short so the analyst should look for the careers page, or only a URL was given)."""

DECIDE_SYSTEM = """You are the Orchestrator of HireLens, a multi-agent resume-to-job matching system.
Something in the run needs a decision. Read the situation and pick exactly one of the allowed actions.
Return JSON: {"reasoning": "...", "action": "<one allowed action>", "note_for_agent": "instruction for the agent that runs next, if any"}
Prefer recovering (retrying with better guidance) over giving up, but abort when continuing would give the user
a misleading result (e.g. the document is not a resume, or nothing at all is known about the job)."""


# --------------------------------------------------------------------- entry
def run(ctx: AgentContext, state: PipelineState) -> dict:
    if not state.get("plan"):
        return _initial_plan(ctx, state)

    updates: dict = {"failed": {}}
    warnings = list(state.get("warnings", []))
    completed = list(state.get("completed", []))
    attempts = dict(state.get("attempts", {}))
    notes = dict(state.get("agent_notes", {}))
    failed = state.get("failed") or {}
    last = state.get("last", "")

    def go(agent: str, reason: str) -> dict:
        ctx.trace.handoff(NAME, agent, reason)
        return {**updates, "next": agent, "warnings": warnings, "completed": completed,
                "attempts": attempts, "agent_notes": notes}

    def abort(message: str) -> dict:
        ctx.trace.log(NAME, "decision", f"Aborting run: {message}", {"message": message})
        return {**updates, "next": "finalize", "status": "aborted", "message": message,
                "warnings": warnings, "completed": completed}

    # ---- 1. a worker failed --------------------------------------------
    if failed:
        agent, error = failed["agent"], failed["error"]
        if agent in {"resume_parser", "job_analyst", "matcher"}:
            if attempts.get(agent, 0) < MAX_ATTEMPTS:
                ctx.trace.log(NAME, "plan", f"Re-plan: {agent} failed ({error}); retrying it once", {"failed": failed})
                return go(agent, "retry after failure")
            return abort(f"The {agent.replace('_', ' ')} could not complete ({error}). Please try again in a minute.")
        if agent == "critic":
            warnings.append("The critic could not verify the scores this time; treat them as unverified.")
        elif agent == "career_coach":
            warnings.append("Coaching suggestions could not be generated this time; the fit scores are complete.")
        ctx.trace.log(NAME, "plan", f"Re-plan: {agent} failed ({error}); continuing without it", {"failed": failed})
        completed.append(agent)

    # ---- 2. inspect the last result and re-plan if needed ----------------
    elif last == "resume_parser":
        problems = resume_parser.quality_problems(state["resume_profile"])
        if problems:
            not_resume = "document does not look like a resume" in problems
            options = ["abort", "continue"] if not_resume else (
                (["retry_resume_parser"] if attempts.get(last, 0) < MAX_ATTEMPTS else []) + ["continue", "abort"])
            action, note = _decide(ctx, "The resume parser's output looks weak.", {
                "problems": problems,
                "missing_sections": state["resume_profile"].get("missing_sections", []),
                "resume_preview": state["resume_text"][:1200],
            }, options, default="abort" if not_resume else options[0])
            if action == "abort":
                return abort("The uploaded file does not look like a resume, so a fit score would be meaningless."
                             if not_resume else "The resume could not be read reliably. Try a text-based PDF.")
            if action == "retry_resume_parser":
                completed.remove("resume_parser")
                notes["resume_parser"] = note or "Re-read carefully; previous extraction missed sections: " + "; ".join(problems)
                return go("resume_parser", "re-parse with guidance")
            warnings.append("Resume extraction was incomplete (" + "; ".join(problems) + "); scores may be lower than deserved.")

    elif last == "job_analyst":
        job = state["job_profile"]
        if job.get("completeness", 1) < 0.4 or not job.get("must_have"):
            options = (["retry_job_analyst"] if attempts.get(last, 0) < MAX_ATTEMPTS else []) + ["continue", "abort"]
            action, note = _decide(ctx, "The job analyst could not establish clear requirements.", {
                "job_profile": job, "had_url": bool(state.get("job_url")),
                "pasted_words": len((state.get("job_description") or "").split()),
            }, options, default=options[0])
            if action == "abort":
                return abort("Not enough is known about this job to score a fit. Paste the full job description.")
            if action == "retry_job_analyst":
                completed.remove("job_analyst")
                notes["job_analyst"] = note or "Requirements are still unclear: search the company's careers page for this role."
                return go("job_analyst", "look for missing requirements")
            warnings.append("The job posting had few concrete requirements, so the score is less certain.")

    elif last == "critic" and not state.get("critic_approved"):
        updates.update(_arbitrate(ctx, state))

    # ---- 3. continue with the plan ---------------------------------------
    for agent in state["plan"]:
        if agent not in completed:
            if agent == "critic" and "matcher" in completed:
                continue  # critic runs straight after matcher via the graph
            return go(agent, "next step in plan")
    ctx.trace.handoff(NAME, "finalize", "all planned agents finished")
    return {**updates, "next": "finalize", "warnings": warnings, "completed": completed,
            "attempts": attempts, "agent_notes": notes}


# ------------------------------------------------------------------ helpers
def _initial_plan(ctx: AgentContext, state: PipelineState) -> dict:
    summary = {
        "resume_chars": len(state["resume_text"]),
        "resume_pages": state.get("resume_pages", 1),
        "job_description_words": len((state.get("job_description") or "").split()),
        "job_description_preview": (state.get("job_description") or "")[:600],
        "job_url": state.get("job_url") or None,
    }
    ctx.trace.log(NAME, "input", "Received resume and job input", summary)
    plan, notes, reasoning = list(AGENTS), {}, "default plan"
    try:
        raw = ctx.llm.chat_json(NAME, PLAN_SYSTEM, "Input summary:\n" + to_prompt_json(summary), max_tokens=800)
        candidate = [a for a in raw.get("plan", []) if a in AGENTS]
        if _valid_plan(candidate):
            plan = candidate
            reasoning = str(raw.get("reasoning", ""))
            notes = {k: str(v) for k, v in (raw.get("notes_for_agents") or {}).items() if k in AGENTS and v}
        else:
            ctx.trace.log(NAME, "warning", f"LLM plan {raw.get('plan')} was invalid; using default order", {})
    except LLMError as exc:
        ctx.trace.log(NAME, "warning", f"Planning call failed ({exc}); using default plan", {})

    ctx.trace.log(NAME, "plan", f"Plan: {' → '.join(plan)}. {reasoning}", {"plan": plan, "notes_for_agents": notes})
    ctx.trace.handoff(NAME, plan[0], "start of plan")
    return {"plan": plan, "completed": [], "agent_notes": notes, "attempts": {}, "status": "running",
            "warnings": [], "next": plan[0], "failed": {}, "critic_rounds": 0, "match_history": [],
            "critique_history": []}


def _valid_plan(plan: list[str]) -> bool:
    if sorted(plan) != sorted(AGENTS):
        return False
    idx = {a: i for i, a in enumerate(plan)}
    return (idx["resume_parser"] < idx["matcher"] and idx["job_analyst"] < idx["matcher"]
            and idx["matcher"] < idx["critic"] < idx["career_coach"])


def _decide(ctx: AgentContext, situation: str, context: dict, options: list[str], default: str) -> tuple[str, str]:
    user = f"Situation: {situation}\n\nDetails:\n{to_prompt_json(context, 4000)}\n\nAllowed actions: {options}"
    try:
        raw = ctx.llm.chat_json(NAME, DECIDE_SYSTEM, user, max_tokens=600)
        action, reasoning = str(raw.get("action", "")).strip(), str(raw.get("reasoning", ""))
        note = str(raw.get("note_for_agent") or "")
        if action not in options:
            ctx.trace.log(NAME, "warning", f"Chose invalid action {action!r}; falling back to {default}", {})
            action = default
    except LLMError as exc:
        action, reasoning, note = default, f"decision call failed ({exc}); using safe default", ""
    ctx.trace.log(NAME, "plan", f"Re-plan: {situation} → {action}. {reasoning}",
                  {"situation": situation, "options": options, "action": action, "note": note})
    return action, note


def _arbitrate(ctx: AgentContext, state: PipelineState) -> dict:
    """Matcher and Critic still disagree after the revision budget: settle it."""
    critique, match = state["critique"], state["match"]
    action, _ = _decide(ctx, "The Matcher and Critic still disagree after the maximum revision rounds.", {
        "matcher_scores": {k: v["score"] for k, v in match["sections"].items()},
        "matcher_revision_notes": match.get("revision_notes", []),
        "critic_issues": critique.get("issues", []),
    }, ["apply_critic_scores", "keep_matcher_scores"], default="apply_critic_scores")

    if action == "keep_matcher_scores":
        return {"arbitration": {"action": action, "changes": []}}

    sections = {k: dict(v) for k, v in match["sections"].items()}
    matched = list(match.get("matched_must_have", []))
    missing = list(match.get("missing_must_have", []))
    changes = []
    for issue in critique.get("issues", []):
        sec, suggested = issue.get("section"), issue.get("suggested_score")
        if sec in sections and suggested is not None and suggested < sections[sec]["score"]:
            changes.append(f"{sec}: {sections[sec]['score']} → {suggested}")
            sections[sec]["score"] = suggested
        claim = (issue.get("claim") or "").strip()
        if sec == "skills" and claim in matched:
            matched.remove(claim)
            missing.append(claim)
            changes.append(f"'{claim}' moved to missing must-haves")
    new_match = {**match, "sections": sections, "matched_must_have": matched, "missing_must_have": missing}
    ctx.trace.log(NAME, "decision", "Applied critic's adjustments: " + ("; ".join(changes) or "no numeric changes"),
                  {"changes": changes})
    return {"match": new_match, "arbitration": {"action": action, "changes": changes}}
