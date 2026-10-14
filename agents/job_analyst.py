"""Job Analyst agent: job description and/or URL → structured JobProfile.

This agent runs its own reason-act loop. Each turn the LLM chooses one action:
fetch the posting URL, search the web (e.g. the company's careers page) to fill
gaps in a vague description, or finish with the structured profile. What it
does depends on what it has found so far; nothing about the order is fixed.
"""
from __future__ import annotations

from pydantic import ValidationError

from agents.base import AgentContext, PipelineState, call_tool, to_prompt_json
from lib.llm import LLMError
from lib.schemas import JobProfile
from tools.errors import ToolError

NAME = "job_analyst"
MAX_JD_CHARS = 9000
MAX_OBSERVATION_CHARS = 4000

SYSTEM = """You are the Job Analyst agent in HireLens, a multi-agent resume-to-job matching system.
Work out what a role actually requires. You can use tools. Each turn, reply with ONE JSON action:

{"thought": "...", "action": "fetch_url", "url": "https://..."}
{"thought": "...", "action": "web_search", "query": "..."}
{"thought": "...", "action": "finish", "job_profile": {
    "title": "", "company": "",
    "seniority": "intern|entry|mid|senior|unknown",
    "min_years_experience": number|null,
    "must_have": ["Python", "REST APIs"],      // explicitly required skills/qualifications, short names (1-4 words)
    "nice_to_have": ["..."],                    // preferred / bonus / 'good to have'
    "responsibilities": ["..."],
    "education_requirement": "",
    "keywords": ["..."],                        // 8-20 concrete terms an ATS would scan for
    "completeness": 0.0-1.0,                    // how fully the requirements are now known
    "sources": ["where each part came from: 'user description', a URL, or 'web search: <query>'"]
}}

How to work:
- If a job URL was given and you have not read it yet, fetch it first.
- If what you have is vague (no concrete skills, no experience level, very short, or only a title and company),
  search the web for the company's careers page or the same role's posting, then fetch the most relevant result.
- Stop as soon as the requirements are clear. Do not use tools you do not need.
- Never invent requirements no source supports. If something stays unknown, leave it out and lower completeness.
- Separate must-have from nice-to-have using the posting's own wording ("required", "must", vs "preferred", "plus")."""


def run(ctx: AgentContext, state: PipelineState) -> dict:
    jd_text = (state.get("job_description") or "").strip()
    job_url = (state.get("job_url") or "").strip()
    hint = state.get("agent_notes", {}).get(NAME, "")
    max_steps = ctx.settings.max_analyst_steps

    observations: list[str] = []
    used_urls: list[str] = []
    searched = False

    ctx.trace.log(
        NAME, "decision",
        f"Analysing job: {len(jd_text.split())} words of pasted text, URL {'given' if job_url else 'not given'}"
        + (f"; orchestrator note: {hint}" if hint else ""),
        {"jd_words": len(jd_text.split()), "job_url": job_url, "hint": hint},
    )

    for step in range(1, max_steps + 2):
        force_finish = step > max_steps
        user = _build_prompt(jd_text, job_url, observations, hint, step, max_steps, force_finish)
        action = ctx.llm.chat_json(NAME, SYSTEM, user, max_tokens=2500)
        kind = str(action.get("action", "")).strip().lower()
        thought = str(action.get("thought", ""))[:400]

        if kind == "finish" or force_finish:
            if kind != "finish":
                raise LLMError("job analyst did not finish within its step budget")
            ctx.trace.log(NAME, "decision", f"Finishing: {thought}", {"step": step, "thought": thought})
            return {"job_profile": _finalise(ctx, action.get("job_profile") or {}, jd_text, job_url, used_urls, searched)}

        if kind == "fetch_url":
            url = str(action.get("url") or job_url)
            ctx.trace.log(NAME, "decision", f"Step {step}: fetch {url} — {thought}", {"step": step, "thought": thought})
            try:
                page = call_tool(ctx, NAME, "fetch_url", url=url)
                used_urls.append(url)
                observations.append(f"[fetch_url {url}]\n{page[:MAX_OBSERVATION_CHARS]}")
            except ToolError as exc:
                observations.append(f"[fetch_url {url}] FAILED: {exc}. Try another approach.")
        elif kind == "web_search":
            query = str(action.get("query", ""))
            ctx.trace.log(NAME, "decision", f"Step {step}: search '{query}' — {thought}", {"step": step, "thought": thought})
            try:
                results = call_tool(ctx, NAME, "web_search", query=query)
                searched = True
                lines = [f"- {r['title']} | {r['url']}\n  {r['snippet']}" for r in results]
                observations.append(f"[web_search {query!r}]\n" + "\n".join(lines))
            except ToolError as exc:
                observations.append(f"[web_search {query!r}] FAILED: {exc}. Try another approach or finish.")
        else:
            ctx.trace.log(NAME, "warning", f"Step {step}: unknown action {kind!r}; asking again", {"action": action})
            observations.append(f"Your last reply had an unknown action {kind!r}. Use fetch_url, web_search or finish.")

    raise LLMError("job analyst loop ended unexpectedly")  # pragma: no cover


def _build_prompt(jd_text, job_url, observations, hint, step, max_steps, force_finish) -> str:
    parts = [f"Pasted job description:\n<<<\n{jd_text[:MAX_JD_CHARS] or '(none provided)'}\n>>>"]
    parts.append(f"Job posting URL: {job_url or '(none provided)'}")
    if hint:
        parts.append(f"Note from the Orchestrator: {hint}")
    if observations:
        parts.append("What your tools returned so far:\n" + "\n\n".join(observations))
    parts.append(f"This is step {step} of at most {max_steps} tool steps.")
    if force_finish:
        parts.append("You have used all tool steps. You MUST reply with the finish action now.")
    return "\n\n".join(parts)


def _finalise(ctx: AgentContext, raw: dict, jd_text: str, job_url: str, used_urls: list[str], searched: bool) -> dict:
    try:
        profile = JobProfile.model_validate(raw)
    except ValidationError as exc:
        raise LLMError(f"job analyst returned invalid profile: {exc.errors()[:3]}") from exc

    sources = list(profile.sources)
    if jd_text and "user description" not in " ".join(sources).lower():
        sources.insert(0, "user description")
    for url in used_urls:
        if url not in sources:
            sources.append(url)
    profile.sources = sources
    # fetching the URL the user gave is not enrichment; searching or reading other pages is
    profile.enrichment_used = searched or any(u != job_url for u in used_urls)
    # keywords must at least include the must-haves so coverage is meaningful
    profile.keywords = list(dict.fromkeys(profile.keywords + profile.must_have))

    ctx.trace.log(
        NAME, "message",
        f"Role '{profile.title or 'unknown'}' at {profile.company or 'unknown company'}: "
        f"{len(profile.must_have)} must-have, {len(profile.nice_to_have)} nice-to-have, "
        f"seniority {profile.seniority}, completeness {profile.completeness:.2f}"
        + (" (enriched from web)" if profile.enrichment_used else ""),
        {"job_profile": profile.model_dump()},
    )
    return profile.model_dump()


def summary_for_prompt(profile: dict) -> str:
    return to_prompt_json(profile, limit=6000)
