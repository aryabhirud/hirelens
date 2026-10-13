"""Typed contracts for everything that passes between agents.

Every LLM reply is validated against one of these models before another agent
sees it, so a malformed answer fails loudly at the boundary instead of
corrupting the next step. Models are lenient about missing fields (defaults)
but strict about types and ranges.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SECTIONS = ("skills", "experience", "projects", "education", "keywords")


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def _none_to_default(cls, value, info):
        # LLMs often send null for empty lists; treat it as the field default.
        if value is None:
            field = cls.model_fields.get(info.field_name)
            if field is not None and field.default_factory is list:
                return []
        return value


# ---------------------------------------------------------------- input ----
class HireLensInput(_Base):
    """What the system accepts. At least one of job_description / job_url is required."""

    resume_filename: str
    resume_text: str = Field(min_length=1)
    resume_pages: int = 1
    job_description: str = ""
    job_url: str = ""


# ---------------------------------------------------------- resume parser --
class ExperienceItem(_Base):
    title: str = ""
    organization: str = ""
    duration: str = ""
    months: int | None = None
    highlights: list[str] = Field(default_factory=list)


class ProjectItem(_Base):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class EducationItem(_Base):
    degree: str = ""
    institution: str = ""
    year: str = ""
    score: str = ""


class ResumeProfile(_Base):
    is_resume: bool = True
    name: str = ""
    headline: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    total_experience_months: int = 0
    projects: list[ProjectItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    parse_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_sections: list[str] = Field(default_factory=list)


# ------------------------------------------------------------ job analyst --
class JobProfile(_Base):
    title: str = ""
    company: str = ""
    seniority: Literal["intern", "entry", "mid", "senior", "unknown"] = "unknown"
    min_years_experience: float | None = None
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    education_requirement: str = ""
    keywords: list[str] = Field(default_factory=list)
    completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    enrichment_used: bool = False

    @field_validator("seniority", mode="before")
    @classmethod
    def _normalise_seniority(cls, value):
        value = (value or "unknown").lower().strip()
        aliases = {"internship": "intern", "junior": "entry", "fresher": "entry", "graduate": "entry",
                   "mid-level": "mid", "lead": "senior", "staff": "senior"}
        value = aliases.get(value, value)
        return value if value in {"intern", "entry", "mid", "senior", "unknown"} else "unknown"


# ---------------------------------------------------------------- matcher --
class Evidence(_Base):
    claim: str
    resume_quote: str


class SectionScore(_Base):
    score: int = Field(ge=0, le=100)
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def _clamp(cls, value):
        try:
            return max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError):
            return 0


class MatchResult(_Base):
    sections: dict[str, SectionScore]
    matched_must_have: list[str] = Field(default_factory=list)
    missing_must_have: list[str] = Field(default_factory=list)
    matched_nice_to_have: list[str] = Field(default_factory=list)
    missing_nice_to_have: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)

    @field_validator("sections")
    @classmethod
    def _all_sections(cls, value):
        missing = [s for s in SECTIONS if s not in value]
        if missing:
            raise ValueError(f"matcher omitted sections: {missing}")
        return {k: v for k, v in value.items() if k in SECTIONS}


# ----------------------------------------------------------------- critic --
class CritiqueIssue(_Base):
    section: str
    claim: str = ""
    problem: str
    suggested_score: int | None = None


class Critique(_Base):
    verdict: Literal["approve", "revise"]
    issues: list[CritiqueIssue] = Field(default_factory=list)
    summary: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value):
        value = str(value).lower().strip()
        return "approve" if value.startswith("approv") or value == "accept" else "revise"


# ----------------------------------------------------------- career coach --
class BulletSuggestion(_Base):
    original: str = ""
    improved: str
    targets: str = ""


class InterviewQuestion(_Base):
    question: str
    category: str = "technical"
    why_asked: str = ""


class GapAction(_Base):
    gap: str
    action: str


class CoachingPlan(_Base):
    bullet_suggestions: list[BulletSuggestion] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    gap_actions: list[GapAction] = Field(default_factory=list)


# ----------------------------------------------------------------- output --
class CriticRound(_Base):
    round: int
    verdict: Literal["approve", "revise"]
    summary: str = ""
    issues: list[CritiqueIssue] = Field(default_factory=list)


class FitReport(_Base):
    """What the system produces."""

    run_id: str
    status: Literal["completed", "aborted"]
    message: str = ""
    candidate_name: str = ""
    job_title: str = ""
    company: str = ""
    overall_score: int | None = None
    verdict: str = ""
    section_scores: dict[str, SectionScore] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    matched_must_have: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    nice_to_have_gaps: list[str] = Field(default_factory=list)
    critic_rounds: int = 0
    critic_approved: bool = False
    critic_summary: str = ""
    critic_history: list[CriticRound] = Field(default_factory=list)
    score_history: dict[str, list[int]] = Field(default_factory=dict)  # per section, one entry per matcher round (+ arbitration)
    job_sources: list[str] = Field(default_factory=list)
    coaching: CoachingPlan | None = None
    warnings: list[str] = Field(default_factory=list)
