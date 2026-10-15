"""Execution trace: every plan, decision, tool call and hand-off between agents.

The trace is recorded while the pipeline actually runs and is exported as JSON
(or Markdown) from the UI and the CLI. It is the evidence of what the agents
really did, so nothing in it is ever written by hand.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

EVENT_TYPES = {
    "input",        # what the run started with
    "plan",         # orchestrator's plan or re-plan
    "decision",     # an agent reasoning about what to do next
    "tool_call",    # an agent invoking a tool
    "tool_result",  # what the tool returned (or that it failed)
    "message",      # an agent's output handed to another agent
    "handoff",      # control moving from one agent to another
    "critique",     # critic's verdict on the matcher's work
    "revision",     # matcher revising after critique
    "warning",      # degraded but continuing
    "error",        # an agent failed
    "final",        # the finished report
}


@dataclass
class TraceEvent:
    step: int
    timestamp: str
    elapsed_s: float
    agent: str
    event: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


class Trace:
    def __init__(self, on_event: Callable[[TraceEvent], None] | None = None):
        self.run_id = uuid.uuid4().hex[:12]
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.events: list[TraceEvent] = []
        self._on_event = on_event

    def log(self, agent: str, event: str, summary: str, data: dict[str, Any] | None = None) -> TraceEvent:
        if event not in EVENT_TYPES:
            raise ValueError(f"unknown trace event type: {event}")
        entry = TraceEvent(
            step=len(self.events) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            elapsed_s=round(time.monotonic() - self.started, 2),
            agent=agent,
            event=event,
            summary=summary,
            data=_jsonable(data or {}),
        )
        self.events.append(entry)
        if self._on_event:
            try:
                self._on_event(entry)
            except Exception:  # a UI callback must never break the pipeline
                pass
        return entry

    def handoff(self, src: str, dst: str, reason: str) -> None:
        self.log(src, "handoff", f"{src} → {dst}: {reason}", {"from": src, "to": dst})

    # ---- export -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "event_count": len(self.events),
            "events": [asdict(e) for e in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [f"# HireLens execution trace `{self.run_id}`", "", f"Started: {self.started_at}", ""]
        lines.append("| # | t (s) | Agent | Event | Summary |")
        lines.append("|---|---|---|---|---|")
        for e in self.events:
            summary = e.summary.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {e.step} | {e.elapsed_s} | {e.agent} | {e.event} | {summary} |")
        return "\n".join(lines) + "\n"

    def agents_seen(self) -> list[str]:
        seen: list[str] = []
        for e in self.events:
            if e.agent not in seen:
                seen.append(e.agent)
        return seen


def _jsonable(value: Any) -> Any:
    """Make trace payloads safe to serialise (pydantic models, sets, etc.)."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
