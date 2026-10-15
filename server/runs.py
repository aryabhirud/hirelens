"""In-memory run registry: each analysis runs in a background thread and its
trace events are buffered so the browser can stream them (and reconnect)."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agents.base import AgentContext
from agents.graph import run_pipeline
from lib.config import Settings
from lib.llm import LLMError, SupportsChatJSON
from lib.schemas import HireLensInput
from lib.trace import Trace, TraceEvent

FINISHED = {"completed", "aborted", "failed"}


@dataclass
class Run:
    id: str
    client: str
    created: float
    trace: Trace
    status: str = "running"            # running | completed | aborted | failed
    report: dict[str, Any] | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_event(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(asdict(event))

    def events_since(self, index: int) -> list[dict[str, Any]]:
        with self._lock:
            return self.events[index:]

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class RunStore:
    def __init__(self, max_active: int = 3, max_kept: int = 100, ttl_s: int = 6 * 3600):
        self.max_active = max_active
        self.max_kept = max_kept
        self.ttl_s = ttl_s
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    # ---- queries -------------------------------------------------------
    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def active_for(self, client: str) -> int:
        with self._lock:
            return sum(1 for r in self._runs.values() if r.client == client and not r.finished)

    def active(self) -> int:
        with self._lock:
            return sum(1 for r in self._runs.values() if not r.finished)

    # ---- lifecycle -----------------------------------------------------
    def start(
        self,
        inp: HireLensInput,
        client: str,
        settings: Settings,
        llm_factory: Callable[[Settings], SupportsChatJSON],
        background: bool = True,
    ) -> Run:
        self._cleanup()
        run_holder: dict[str, Run] = {}
        trace = Trace(on_event=lambda e: run_holder["run"].add_event(e))
        run = Run(id=trace.run_id, client=client, created=time.time(), trace=trace)
        run_holder["run"] = run
        with self._lock:
            self._runs[run.id] = run

        def work() -> None:
            try:
                ctx = AgentContext(llm=llm_factory(settings), trace=trace, settings=settings)
                report = run_pipeline(inp, ctx)
                run.report = report.model_dump()
                run.status = report.status
            except (LLMError, ValueError) as exc:
                run.error = str(exc)
                run.status = "failed"
            except Exception as exc:  # never leave a run hanging in "running"
                run.error = f"Unexpected error: {type(exc).__name__}: {exc}"
                run.status = "failed"

        if background:
            threading.Thread(target=work, name=f"run-{run.id}", daemon=True).start()
        else:
            work()
        return run

    def _cleanup(self) -> None:
        now = time.time()
        with self._lock:
            stale = [rid for rid, r in self._runs.items() if r.finished and now - r.created > self.ttl_s]
            for rid in stale:
                del self._runs[rid]
            finished = sorted((r for r in self._runs.values() if r.finished), key=lambda r: r.created)
            for r in finished[: max(0, len(self._runs) - self.max_kept)]:
                del self._runs[r.id]
