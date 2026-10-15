"""Deterministic text checks used by the Matcher and the Critic.

These are the critic's "ground truth": an LLM can claim anything, but these
functions check the claim against the actual resume text.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-/]*")


def _tokens(text: str) -> list[str]:
    return [t.strip(".-/") for t in _TOKEN.findall(text.lower()) if t.strip(".-/")]


def _norm(text: str) -> str:
    return " ".join(_tokens(text))


def verify_evidence(quote: str, resume_text: str, threshold: float = 0.8) -> dict:
    """Is `quote` actually present in the resume?

    Exact (normalised) containment passes. Otherwise we slide a window of the
    quote's length over the resume and accept if at least `threshold` of the
    quote's words appear in the best window; this tolerates line breaks and
    small rewording from PDF extraction but rejects invented content.
    """
    q_tokens = _tokens(quote)
    if len(q_tokens) < 2:
        return {"supported": False, "score": 0.0, "reason": "quote too short to verify"}
    if _norm(quote) in _norm(resume_text):
        return {"supported": True, "score": 1.0, "reason": "exact match"}

    r_tokens = _tokens(resume_text)
    window = len(q_tokens) + 3
    best = 0.0
    q_set = q_tokens
    for start in range(0, max(1, len(r_tokens) - window + 1)):
        chunk = set(r_tokens[start : start + window])
        hit = sum(1 for t in q_set if t in chunk) / len(q_set)
        if hit > best:
            best = hit
            if best == 1.0:
                break
    return {
        "supported": best >= threshold,
        "score": round(best, 2),
        "reason": "close match" if best >= threshold else "not found in resume",
    }


def keyword_coverage(keywords: list[str], text: str) -> dict:
    """Which of the job's keywords literally appear in the resume?"""
    haystack = text.lower()
    found, missing = [], []
    for raw in dict.fromkeys(k.strip() for k in keywords if k and k.strip()):
        needle = re.escape(raw.lower())
        # boundaries that work for "C++", "Node.js", "CI/CD" as well as words
        if re.search(rf"(?<![a-z0-9]){needle}(?![a-z0-9])", haystack):
            found.append(raw)
        else:
            missing.append(raw)
    total = len(found) + len(missing)
    return {
        "found": found,
        "missing": missing,
        "coverage": round(len(found) / total, 2) if total else 0.0,
    }
