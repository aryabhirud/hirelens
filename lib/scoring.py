"""Deterministic scoring rules.

The LLM agents judge each section (0-100) and must back it with evidence; the
overall number is computed here in plain code so it is transparent and the
same inputs always give the same total.
"""
from __future__ import annotations

from lib.schemas import SectionScore

WEIGHTS: dict[str, float] = {
    "skills": 0.35,
    "experience": 0.20,
    "projects": 0.20,
    "education": 0.10,
    "keywords": 0.15,
}

# If a candidate is missing most of the must-have requirements, a high score in
# the other sections should not make them look like a strong fit.
MUST_HAVE_CAP_RATIO = 0.5
MUST_HAVE_CAP_SCORE = 55


def overall_score(sections: dict[str, SectionScore], matched_must: int, missing_must: int) -> int:
    total = sum(WEIGHTS[name] * sections[name].score for name in WEIGHTS if name in sections)
    score = int(round(total))
    must_total = matched_must + missing_must
    if must_total and missing_must / must_total > MUST_HAVE_CAP_RATIO:
        score = min(score, MUST_HAVE_CAP_SCORE)
    return score


def verdict_label(score: int) -> str:
    if score >= 75:
        return "Strong fit"
    if score >= 55:
        return "Moderate fit"
    if score >= 35:
        return "Weak fit"
    return "Poor fit"
