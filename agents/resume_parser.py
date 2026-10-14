"""Resume Parser agent: raw resume text → structured ResumeProfile.

It also judges its own output (parse_confidence, missing_sections, is_resume)
so the Orchestrator can decide whether to retry it with a hint, continue with a
warning, or stop because the file is not a resume at all.
"""
from __future__ import annotations

from pydantic import ValidationError

from agents.base import AgentContext, PipelineState, to_prompt_json
from lib.llm import LLMError
from lib.schemas import ResumeProfile

NAME = "resume_parser"
MAX_RESUME_CHARS = 15000

SYSTEM = """You are the Resume Parser agent in HireLens, a multi-agent resume-to-job matching system.
Turn the raw text of a resume (extracted from a PDF, so layout may be messy) into structured data.

Return JSON with exactly these keys:
{
  "is_resume": true|false,            // false if this is clearly not a resume/CV
  "name": "", "headline": "",
  "skills": ["..."],                  // individual skills/tools/languages, deduplicated
  "experience": [{"title": "", "organization": "", "duration": "", "months": int|null, "highlights": ["bullet text as written"]}],
  "total_experience_months": int,     // sum of professional experience incl. internships
  "projects": [{"name": "", "description": "", "technologies": ["..."]}],
  "education": [{"degree": "", "institution": "", "year": "", "score": ""}],
  "certifications": ["..."],
  "parse_confidence": 0.0-1.0,        // how sure you are that the extraction is complete and correct
  "missing_sections": ["..."]         // standard sections you could not find, e.g. "experience"
}

Rules:
- Extract only what the text supports. Never invent skills, dates, employers or projects.
- Keep highlight bullets close to the original wording; they are used later as evidence.
- Skills mentioned only inside project or job descriptions still count as skills.
- If dates are unclear, set months to null and lower parse_confidence."""


def run(ctx: AgentContext, state: PipelineState) -> dict:
    hint = state.get("agent_notes", {}).get(NAME, "")
    attempt = state.get("attempts", {}).get(NAME, 0) + 1
    resume_text = state["resume_text"][:MAX_RESUME_CHARS]

    ctx.trace.log(
        NAME, "decision",
        f"Parsing resume ({len(state['resume_text'])} chars, {state.get('resume_pages', 1)} pages), attempt {attempt}"
        + (f" with orchestrator hint: {hint}" if hint else ""),
        {"attempt": attempt, "hint": hint},
    )

    user = f"Resume text:\n<<<\n{resume_text}\n>>>"
    if hint:
        user += f"\n\nThe Orchestrator reviewed a previous extraction and asks: {hint}"

    raw = ctx.llm.chat_json(NAME, SYSTEM, user, max_tokens=3000)
    try:
        profile = ResumeProfile.model_validate(raw)
    except ValidationError as exc:
        raise LLMError(f"resume parser returned invalid structure: {exc.errors()[:3]}") from exc

    ctx.trace.log(
        NAME, "message",
        f"Extracted {len(profile.skills)} skills, {len(profile.experience)} roles, "
        f"{len(profile.projects)} projects, {len(profile.education)} education entries "
        f"(confidence {profile.parse_confidence:.2f})",
        {"profile": profile.model_dump(), "missing_sections": profile.missing_sections},
    )
    return {"resume_profile": profile.model_dump()}


def quality_problems(profile: dict) -> list[str]:
    """Signals the Orchestrator uses to decide whether to re-plan."""
    problems = []
    if not profile.get("is_resume", True):
        problems.append("document does not look like a resume")
    if profile.get("parse_confidence", 1.0) < 0.5:
        problems.append(f"low parse confidence ({profile.get('parse_confidence')})")
    if len(profile.get("skills", [])) < 3:
        problems.append("fewer than 3 skills extracted")
    if not profile.get("experience") and not profile.get("projects"):
        problems.append("no experience or projects extracted")
    return problems


def summary_for_prompt(profile: dict) -> str:
    return to_prompt_json(profile, limit=7000)
