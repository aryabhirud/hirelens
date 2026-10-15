"""Matching agent: compares the candidate to the role and scores each section.

Every section score must be backed by quotes copied from the resume, because
the Critic checks each quote against the real resume text. On a revision round
the Matcher receives the Critic's issues and must either lower the score or
supply evidence that actually exists.
"""
from __future__ import annotations

from pydantic import ValidationError

from agents.base import AgentContext, PipelineState, call_tool, to_prompt_json
from lib.llm import LLMError
from lib.schemas import MatchResult
from tools.errors import ToolError

NAME = "matcher"
MAX_RESUME_CHARS = 10000

SYSTEM = """You are the Matching agent in HireLens, a multi-agent resume-to-job matching system.
Compare a candidate's structured resume with a role's requirements and score the fit honestly.

Score each section from 0 to 100:
- skills:     coverage of the must-have skills first, nice-to-have second
- experience: relevant experience versus the role's seniority and minimum years (internships count for intern/entry roles)
- projects:   how directly the candidate's projects demonstrate the required skills/responsibilities
- education:  fit with the education requirement (degree, field, level)
- keywords:   literal ATS keyword coverage; stay within ±10 of the measured keyword coverage you are given

Return JSON:
{
  "sections": {
    "skills":     {"score": int, "reasoning": "1-3 sentences", "evidence": [{"claim": "...", "resume_quote": "..."}]},
    "experience": {...}, "projects": {...}, "education": {...}, "keywords": {...}
  },
  "matched_must_have": ["..."], "missing_must_have": ["..."],
  "matched_nice_to_have": ["..."], "missing_nice_to_have": ["..."],
  "strengths": ["short phrases"],
  "revision_notes": ["only on revision: how you handled each critic issue"]
}

Hard rules:
- Every resume_quote must be copied word-for-word from the RESUME TEXT (3-20 words). Never paraphrase a quote.
- A requirement counts as matched only if the resume shows it. Related-but-different skills are not a match
  (e.g. MySQL does not satisfy MongoDB); you may mention them in reasoning.
- Do not reward a section without evidence. No evidence → score at most 40.
- Be calibrated: a typical student applying to a role that fits reasonably scores 55-75, not 90+."""


def run(ctx: AgentContext, state: PipelineState) -> dict:
    job = state["job_profile"]
    resume = state["resume_profile"]
    resume_text = state["resume_text"]
    critique = state.get("critique") if state.get("last") == "critic" else None
    previous = state.get("match") if critique else None
    round_no = len(state.get("match_history", [])) + 1

    coverage = state.get("keyword_coverage")
    if coverage is None:
        try:
            coverage = call_tool(ctx, NAME, "keyword_coverage", keywords=job.get("keywords", []), text=resume_text)
        except ToolError:
            coverage = {"found": [], "missing": [], "coverage": None}
            ctx.trace.log(NAME, "warning", "Keyword coverage unavailable; scoring keywords by judgement", {})

    if critique:
        ctx.trace.log(
            NAME, "revision",
            f"Revision round {round_no}: addressing {len(critique.get('issues', []))} critic issue(s)",
            {"issues": critique.get("issues", [])},
        )
    else:
        ctx.trace.log(NAME, "decision", "Scoring candidate against role requirements", {"round": round_no})

    user = "\n\n".join([
        "ROLE REQUIREMENTS:\n" + to_prompt_json(job, 6000),
        "CANDIDATE (structured):\n" + to_prompt_json(resume, 7000),
        f"RESUME TEXT (quote evidence from here only):\n<<<\n{resume_text[:MAX_RESUME_CHARS]}\n>>>",
        "MEASURED KEYWORD COVERAGE (from the keyword_coverage tool):\n" + to_prompt_json(coverage, 2000),
    ])
    if critique:
        user += (
            "\n\nYOUR PREVIOUS ASSESSMENT:\n" + to_prompt_json(previous, 6000)
            + "\n\nTHE CRITIC REJECTED IT WITH THESE ISSUES:\n" + to_prompt_json(critique, 4000)
            + "\n\nRevise. For each issue either lower the score or give a quote that really is in the resume. "
              "Explain each decision in revision_notes."
        )

    result = _validated(ctx, user)
    history = list(state.get("match_history", [])) + [result]
    ctx.trace.log(
        NAME, "message",
        "Section scores: " + ", ".join(f"{k} {v['score']}" for k, v in result["sections"].items())
        + f"; missing must-haves: {', '.join(result['missing_must_have']) or 'none'}",
        {"match": result, "round": round_no},
    )
    ctx.trace.handoff(NAME, "critic", "scores ready for verification")
    return {"match": result, "match_history": history, "keyword_coverage": coverage}


def _validated(ctx: AgentContext, user: str) -> dict:
    raw = ctx.llm.chat_json(NAME, SYSTEM, user, max_tokens=3500)
    try:
        return MatchResult.model_validate(raw).model_dump()
    except ValidationError as exc:
        ctx.trace.log(NAME, "warning", f"Output failed schema check; retrying once: {exc.errors()[0]['msg']}", {})
        raw = ctx.llm.chat_json(
            NAME, SYSTEM,
            user + f"\n\nYour previous reply was rejected by the schema check: {exc.errors()[:3]}. "
                   "Include all five sections with integer scores.",
            max_tokens=3500,
        )
        try:
            return MatchResult.model_validate(raw).model_dump()
        except ValidationError as exc2:
            raise LLMError(f"matcher output invalid twice: {exc2.errors()[:3]}") from exc2
