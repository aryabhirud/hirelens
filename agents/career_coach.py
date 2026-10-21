"""Career Coach agent: turns the verified gaps into concrete next steps.

It only runs after the Critic has approved (or the Orchestrator has arbitrated)
the scores, so its advice targets gaps that were actually verified.
"""
from __future__ import annotations

from pydantic import ValidationError

from agents.base import AgentContext, PipelineState, to_prompt_json
from lib.llm import LLMError
from lib.schemas import CoachingPlan

NAME = "career_coach"

SYSTEM = """You are the Career Coach agent in HireLens, a multi-agent resume-to-job matching system.
The candidate's fit for a role has already been scored and verified. Help them improve their application.

Return JSON:
{
  "bullet_suggestions": [{"original": "existing resume bullet or empty", "improved": "rewritten bullet", "targets": "requirement it helps with"}],
  "interview_questions": [{"question": "...", "category": "technical|project|gap|behavioral", "why_asked": "..."}],
  "gap_actions": [{"gap": "missing requirement", "action": "specific, realistic step to close it before applying or interviewing"}]
}

Rules:
- 4-6 bullet suggestions. Rewrite REAL bullets from the resume to foreground the role's requirements:
  strong action verb, the technology, and an outcome. Never add skills, numbers or achievements the resume does
  not contain; where a number would help, write a placeholder like [X%] for the candidate to fill in truthfully.
- 8 interview questions: about 3 technical on the must-haves, 2 deep-dives on the candidate's own projects,
  2 probing the gaps (what an interviewer would ask because something is missing), 1 behavioral.
- One gap_action per important missing requirement; prefer small projects or targeted learning that can be
  shown on a resume. Do not suggest lying or keyword stuffing."""


def run(ctx: AgentContext, state: PipelineState) -> dict:
    match = state["match"]
    resume = state["resume_profile"]
    gaps = match.get("missing_must_have", []) + match.get("missing_nice_to_have", [])[:4]
    weak = sorted(match["sections"].items(), key=lambda kv: kv[1]["score"])[:2]

    ctx.trace.log(
        NAME, "decision",
        f"Coaching on {len(gaps)} gap(s); weakest sections: " + ", ".join(f"{k} ({v['score']})" for k, v in weak),
        {"gaps": gaps, "weakest": [k for k, _ in weak]},
    )
    bullets = [h for exp in resume.get("experience", []) for h in exp.get("highlights", [])]
    bullets += [p.get("description", "") for p in resume.get("projects", []) if p.get("description")]

    user = "\n\n".join([
        "ROLE:\n" + to_prompt_json(state["job_profile"], 5000),
        "VERIFIED ASSESSMENT:\n" + to_prompt_json({
            "section_scores": {k: v["score"] for k, v in match["sections"].items()},
            "matched_must_have": match.get("matched_must_have", []),
            "missing_must_have": match.get("missing_must_have", []),
            "missing_nice_to_have": match.get("missing_nice_to_have", []),
            "critic_summary": (state.get("critique") or {}).get("summary", ""),
        }, 3000),
        "CANDIDATE SKILLS: " + ", ".join(resume.get("skills", [])),
        "CANDIDATE PROJECTS:\n" + to_prompt_json(resume.get("projects", []), 3000),
        "EXISTING RESUME BULLETS:\n" + "\n".join(f"- {b}" for b in bullets[:20]),
    ])
    raw = ctx.llm.chat_json(NAME, SYSTEM, user, max_tokens=3000)
    try:
        plan = CoachingPlan.model_validate(raw)
    except ValidationError as exc:
        raise LLMError(f"career coach returned invalid structure: {exc.errors()[:3]}") from exc

    ctx.trace.log(
        NAME, "message",
        f"Wrote {len(plan.bullet_suggestions)} bullet suggestions, {len(plan.interview_questions)} interview "
        f"questions, {len(plan.gap_actions)} gap actions",
        {"coaching": plan.model_dump()},
    )
    return {"coaching": plan.model_dump()}
