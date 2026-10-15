"""Critic agent: checks the Matcher's work and sends it back if it is inflated.

Two layers:
1. Tools: every evidence quote is checked against the real resume text, and
   every claimed must-have match is searched for in the resume.
2. LLM review: with those check results in hand, the critic judges whether the
   scores are justified and lists concrete issues.

A small policy layer stops the critic from approving work whose evidence
mostly failed verification, even if the LLM says "approve".
"""
from __future__ import annotations

from pydantic import ValidationError

from agents.base import AgentContext, PipelineState, call_tool, to_prompt_json
from lib.llm import LLMError
from lib.schemas import Critique, CritiqueIssue

NAME = "critic"
UNSUPPORTED_RATIO_LIMIT = 0.3
HIGH_SCORE = 60

SYSTEM = """You are the Critic agent in HireLens, a multi-agent resume-to-job matching system.
Another agent (the Matcher) scored how well a candidate fits a role. Your job is to catch scores
that are inflated or not supported by the resume. Be strict but fair: do not lower scores that are
well supported, and do not nitpick wording.

You get: the role, the candidate profile, the Matcher's assessment, and automatic verification results
(which evidence quotes really appear in the resume, which claimed must-have matches appear in it).

Return JSON:
{
  "verdict": "approve" | "revise",
  "summary": "one or two sentences",
  "issues": [{"section": "skills|experience|projects|education|keywords",
              "claim": "the claim you dispute", "problem": "why it is unsupported or inflated",
              "suggested_score": int|null}]
}

Revise when: a score relies on quotes that failed verification; a must-have is marked matched but the
resume does not show it; experience is scored high despite clearly missing the seniority/years required;
or a score is out of line with the evidence. Otherwise approve with an empty issues list."""


def run(ctx: AgentContext, state: PipelineState) -> dict:
    match = state["match"]
    resume_text = state["resume_text"]
    round_no = state.get("critic_rounds", 0) + 1

    # ---- layer 1: tool-based verification ---------------------------------
    checks = []
    for section, data in match["sections"].items():
        for ev in data.get("evidence", []):
            result = ctx.tools["verify_evidence"](quote=ev["resume_quote"], resume_text=resume_text)
            checks.append({"section": section, "claim": ev["claim"], "quote": ev["resume_quote"], **result})
    unsupported = [c for c in checks if not c["supported"]]
    ctx.trace.log(
        NAME, "tool_call", f"verify_evidence on {len(checks)} quote(s) from the matcher",
        {"tool": "verify_evidence", "count": len(checks)},
    )
    ctx.trace.log(
        NAME, "tool_result",
        f"verify_evidence: {len(checks) - len(unsupported)} supported, {len(unsupported)} not found in resume",
        {"tool": "verify_evidence", "ok": True, "unsupported": unsupported},
    )

    claimed = match.get("matched_must_have", [])
    must_check = call_tool(ctx, NAME, "keyword_coverage", keywords=claimed, text=resume_text) if claimed else {"missing": []}
    skills_lower = {s.lower() for s in state["resume_profile"].get("skills", [])}
    doubtful_must = [m for m in must_check.get("missing", []) if m.lower() not in skills_lower]

    # ---- layer 2: LLM judgement ------------------------------------------
    user = "\n\n".join([
        "ROLE:\n" + to_prompt_json(state["job_profile"], 5000),
        "CANDIDATE:\n" + to_prompt_json(state["resume_profile"], 6000),
        "MATCHER'S ASSESSMENT:\n" + to_prompt_json(match, 6000),
        "AUTOMATIC CHECKS:\n" + to_prompt_json({
            "evidence_quotes_not_found_in_resume": unsupported,
            "claimed_must_haves_not_found_in_resume_or_skills": doubtful_must,
        }, 3000),
        f"This is review round {round_no}.",
    ])
    raw = ctx.llm.chat_json(NAME, SYSTEM, user, max_tokens=2000)
    try:
        critique = Critique.model_validate(raw)
    except ValidationError as exc:
        raise LLMError(f"critic returned invalid structure: {exc.errors()[:3]}") from exc

    # ---- policy: the LLM may not wave through mostly-unverified evidence --
    critique = _apply_policy(ctx, critique, match, checks, unsupported, doubtful_must)

    history = list(state.get("critique_history", [])) + [critique.model_dump()]
    approved = critique.verdict == "approve"
    ctx.trace.log(
        NAME, "critique",
        f"Round {round_no}: {critique.verdict.upper()} — {critique.summary}"
        + (f" ({len(critique.issues)} issue(s))" if critique.issues else ""),
        {"critique": critique.model_dump(), "round": round_no},
    )
    revisions_done = len(state.get("match_history", [])) - 1
    if not approved and revisions_done < ctx.settings.max_critic_revisions:
        ctx.trace.handoff(NAME, "matcher", "revision requested")
    else:
        ctx.trace.handoff(NAME, "orchestrator", "approved" if approved else "revision budget used up; needs arbitration")
    return {
        "critique": critique.model_dump(),
        "critique_history": history,
        "critic_rounds": round_no,
        "critic_approved": approved,
    }


def _apply_policy(ctx, critique: Critique, match, checks, unsupported, doubtful_must) -> Critique:
    if critique.verdict == "revise":
        return critique
    reasons, extra = [], []
    if checks and len(unsupported) / len(checks) > UNSUPPORTED_RATIO_LIMIT:
        reasons.append(f"{len(unsupported)}/{len(checks)} evidence quotes were not found in the resume")
        for c in unsupported:
            extra.append(CritiqueIssue(section=c["section"], claim=c["claim"],
                                       problem=f"quote not found in resume: '{c['quote']}'"))
    for section, data in match["sections"].items():
        if data["score"] >= HIGH_SCORE and not data.get("evidence"):
            reasons.append(f"{section} scored {data['score']} with no evidence")
            extra.append(CritiqueIssue(section=section, problem="high score with no supporting evidence",
                                       suggested_score=40))
    # Only short skill names can be checked literally; long requirement phrases are left to the LLM's judgement.
    for item in (m for m in doubtful_must if len(m.split()) <= 3):
        reasons.append(f"must-have '{item}' marked matched but not in resume")
        extra.append(CritiqueIssue(section="skills", claim=item, problem="claimed must-have not shown in resume"))
    if not reasons:
        return critique
    ctx.trace.log(NAME, "decision", "Policy override: LLM approved, but " + "; ".join(reasons), {"reasons": reasons})
    return Critique(verdict="revise", issues=list(critique.issues) + extra,
                    summary="Automatic checks found unsupported claims. " + critique.summary)


def route_after(state: PipelineState, max_revisions: int) -> str:
    """Critic → Matcher for another round, or back to the Orchestrator."""
    revisions_done = len(state.get("match_history", [])) - 1
    if not state.get("critic_approved") and revisions_done < max_revisions and not state.get("failed"):
        return "matcher"
    return "orchestrator"
