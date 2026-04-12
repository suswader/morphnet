"""
trace.py — Structured decision trace recorder for MorphNet.

Deterministic. Zero LLM calls. Zero string matching.

Trace data comes FROM Gemini structured output schema fields (reasoning,
evidence_sources, confidence) — not from post-hoc parsing. Every module
passes model schema output directly to trace.log() or trace.span().

Storage: ./results/{datetime}/trace.jsonl (general use)
         ./results/eval_{benchmark}_{datetime}/{task_id}/trace.jsonl (eval harness)
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

# Default results directory: project_root/results/
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Where a decision input came from.

    For LLM-driven decisions: populated directly from model schema output.
    For deterministic ops: populated by the module itself.
    """
    source: str
    """Source type: dom | axtree | screenshot | traffic | site_profile |
    reflection | task_memory | cookies | meta_tokens | model_output |
    element_map | url | config | page_state"""
    description: str
    """Human-readable: 'Element [5] is the login submit button'"""
    element_id: int | None = None
    """If referencing a specific interactive element by its SoM ID."""
    raw_excerpt: str | None = None
    """First 500 chars of raw data. Keeps trace files manageable."""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"source": self.source, "description": self.description}
        if self.element_id is not None:
            d["element_id"] = self.element_id
        if self.raw_excerpt is not None:
            d["raw_excerpt"] = self.raw_excerpt[:500]
        return d


@dataclass
class TraceEntry:
    """One logged event in the decision trace."""
    timestamp: float
    trace_id: str
    parent_id: str | None
    module: str
    event_type: str
    summary: str
    detail: dict
    reasoning: str | None
    evidence: list[Evidence]
    outcome: str | None
    error: str | None
    duration_ms: float | None
    confidence: float | None

    def to_dict(self) -> dict:
        """Serialise for JSONL output. Evidence objects become dicts."""
        return {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "module": self.module,
            "event_type": self.event_type,
            "summary": self.summary,
            "detail": self.detail,
            "reasoning": self.reasoning,
            "evidence": [e.to_dict() for e in self.evidence],
            "outcome": self.outcome,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# TraceSpan — context manager for timed operations with incremental updates
# ---------------------------------------------------------------------------

class TraceSpan:
    """Accumulates trace data during an operation, logs on exit.

    Usage:
        with trace.span("cu_agent", "action_selected", "Click element [5]") as s:
            s.add_evidence(Evidence("axtree", "Element [5] is login button"))
            s.set_reasoning(model_response.reasoning)   # Direct from schema
            s.set_confidence(model_response.confidence)  # Direct from schema
            s.set_detail("action_type", "click")
            # ... do work ...
            s.set_outcome("success")
    """

    def __init__(
        self,
        trace: TaskTrace,
        module: str,
        event_type: str,
        summary: str,
        *,
        parent_id: str | None = None,
        detail: dict | None = None,
    ):
        self._trace = trace
        self._module = module
        self._event_type = event_type
        self._summary = summary
        self._parent_id = parent_id
        self._detail: dict = detail or {}
        self._reasoning: str | None = None
        self._evidence: list[Evidence] = []
        self._outcome: str | None = None
        self._error: str | None = None
        self._confidence: float | None = None
        self._start_time: float = time.time()
        self.trace_id: str = _short_uuid()

    def add_evidence(self, evidence: Evidence) -> None:
        self._evidence.append(evidence)

    def set_reasoning(self, reasoning: str) -> None:
        self._reasoning = reasoning

    def set_confidence(self, confidence: float) -> None:
        self._confidence = confidence

    def set_outcome(self, outcome: str) -> None:
        self._outcome = outcome

    def set_error(self, error: str) -> None:
        self._error = error

    def set_detail(self, key: str, value: Any) -> None:
        self._detail[key] = value

    def _finalize(self) -> str:
        """Log the accumulated entry. Returns trace_id."""
        duration_ms = (time.time() - self._start_time) * 1000
        entry = TraceEntry(
            timestamp=self._start_time,
            trace_id=self.trace_id,
            parent_id=self._parent_id,
            module=self._module,
            event_type=self._event_type,
            summary=self._summary,
            detail=self._detail,
            reasoning=self._reasoning,
            evidence=self._evidence,
            outcome=self._outcome,
            error=self._error,
            duration_ms=round(duration_ms, 2),
            confidence=self._confidence,
        )
        self._trace._write_entry(entry)
        return self.trace_id


# ---------------------------------------------------------------------------
# TaskTrace — the recorder
# ---------------------------------------------------------------------------

def _short_uuid() -> str:
    return uuid.uuid4().hex[:12]


class TaskTrace:
    """Structured decision trace for a single MorphNet run.

    Writes JSONL to disk (immediate flush, crash-safe) and keeps entries
    in memory for querying during the run.

    For general use:
        trace = TaskTrace(task_prompt="Buy a blue widget")
        # Creates ./results/2026-04-04_143022/trace.jsonl

    For eval harness (run_webarena_evals.py controls the path):
        trace = TaskTrace(
            task_prompt="...",
            output_dir=Path("./results/eval_webarena_2026-04-04/task_123/"),
        )
    """

    def __init__(self, task_prompt: str, output_dir: Path | None = None):
        self.task_prompt = task_prompt
        self.start_time = time.time()

        # Resolve output directory
        if output_dir is None:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
            self.output_dir = RESULTS_DIR / timestamp
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._trace_path = self.output_dir / "trace.jsonl"
        self._entries: list[TraceEntry] = []
        self._file = open(self._trace_path, "a", encoding="utf-8")

        # Log the session start as the first entry
        self.log(
            "trace", "trace_started", f"Trace started: {task_prompt[:100]}",
            detail={"task_prompt": task_prompt, "output_dir": str(self.output_dir)},
            outcome="success",
        )

    def log(
        self,
        module: str,
        event_type: str,
        summary: str,
        *,
        detail: dict | None = None,
        reasoning: str | None = None,
        evidence: list[Evidence] | None = None,
        parent_id: str | None = None,
        outcome: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
        confidence: float | None = None,
    ) -> str:
        """Log a trace entry. Returns the trace_id for parent linking."""
        entry = TraceEntry(
            timestamp=time.time(),
            trace_id=_short_uuid(),
            parent_id=parent_id,
            module=module,
            event_type=event_type,
            summary=summary,
            detail=detail or {},
            reasoning=reasoning,
            evidence=evidence or [],
            outcome=outcome,
            error=error,
            duration_ms=duration_ms,
            confidence=confidence,
        )
        self._write_entry(entry)
        return entry.trace_id

    @contextmanager
    def span(
        self,
        module: str,
        event_type: str,
        summary: str,
        *,
        parent_id: str | None = None,
        detail: dict | None = None,
    ) -> Generator[TraceSpan, None, None]:
        """Context manager for timed operations with incremental evidence."""
        s = TraceSpan(
            self, module, event_type, summary,
            parent_id=parent_id, detail=detail,
        )
        try:
            yield s
        except Exception as exc:
            s.set_outcome("failure")
            s.set_error(str(exc))
            raise
        finally:
            s._finalize()

    # --- Query methods ---------------------------------------------------

    def get_entries(
        self,
        *,
        module: str | None = None,
        event_type: str | None = None,
        parent_id: str | None = None,
        since: float | None = None,
    ) -> list[TraceEntry]:
        """Filter in-memory entries."""
        results = self._entries
        if module is not None:
            results = [e for e in results if e.module == module]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if parent_id is not None:
            results = [e for e in results if e.parent_id == parent_id]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results

    def get_decision_chain(self, trace_id: str) -> list[TraceEntry]:
        """Walk parent_id links from a given entry up to the root."""
        id_to_entry = {e.trace_id: e for e in self._entries}
        chain: list[TraceEntry] = []
        current = id_to_entry.get(trace_id)
        while current:
            chain.append(current)
            if current.parent_id is None:
                break
            current = id_to_entry.get(current.parent_id)
        chain.reverse()  # Root first
        return chain

    def summary(self) -> dict:
        """Stats by module, event_type, outcome. Useful for end-of-run diagnostics."""
        from collections import Counter
        modules = Counter(e.module for e in self._entries)
        events = Counter(e.event_type for e in self._entries)
        outcomes = Counter(e.outcome for e in self._entries if e.outcome)
        total_duration = sum(e.duration_ms or 0 for e in self._entries)
        return {
            "total_entries": len(self._entries),
            "by_module": dict(modules),
            "by_event_type": dict(events),
            "by_outcome": dict(outcomes),
            "total_duration_ms": round(total_duration, 2),
            "wall_time_s": round(time.time() - self.start_time, 2),
            "trace_path": str(self._trace_path),
        }

    # --- Step data (representations) ------------------------------------

    def save_step(self, step_name: str, data: dict) -> Path:
        """Save per-step representation data to steps/ dir.

        step_name: e.g. "plan_001", "action_001_03"
        data: dict with raw inputs + processed representations + prompt + response
        Returns the path to the saved file.
        """
        steps_dir = self.output_dir / "steps"
        steps_dir.mkdir(exist_ok=True)
        path = steps_dir / f"{step_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str, indent=2)
        return path

    def save_screenshot(self, step_name: str, b64_data: str) -> Path:
        """Save a screenshot as JPEG for a given step."""
        import base64
        steps_dir = self.output_dir / "steps"
        steps_dir.mkdir(exist_ok=True)
        path = steps_dir / f"{step_name}.jpg"
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return path

    # --- Internal --------------------------------------------------------

    def _write_entry(self, entry: TraceEntry) -> None:
        """Append to in-memory list and flush to JSONL file."""
        self._entries.append(entry)
        line = json.dumps(entry.to_dict(), ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the trace file. Log final summary."""
        summary_data = self.summary()
        self.log(
            "trace", "trace_closed", f"Trace closed: {summary_data['total_entries']} entries",
            detail=summary_data,
            outcome="success",
        )
        self._file.close()

    def __del__(self) -> None:
        try:
            if self._file and not self._file.closed:
                self._file.close()
        except Exception:
            pass
