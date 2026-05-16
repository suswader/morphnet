"""morphnet_v2/timeline.py — offline step-frame builder for Phase 4 tool synthesis.

Reads a single task's notes dir and emits `step_frames.json` aligning three
streams per CU step:
  1. CU actions (from `actions/{aid}.json` + record.jsonl entries)
  2. HTTP request/response pairs (from `http/index.jsonl` + `http/bodies/`)
  3. JS scripts: parsed (from `scripts/{sid}.js`), causally executed via
     `initiator_stack` on HTTP requests, and precisely executed via per-step
     `Profiler.takePreciseCoverage` deltas captured at step boundaries.

Step windows are bracketed by `step_boundary` events the Orchestrator emits
at branch() and at complete_current()/prune(). Cumulative coverage is stored
in each boundary's payload; per-step deltas come from subtracting adjacent
boundaries for the same scriptId.

CLI: `uv run python -m morphnet_v2.timeline <notes_dir>` writes
`<notes_dir>/step_frames.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimelineEvent:
    ts_ms: int
    kind: str                          # "action" | "http_request" | "http_response" | "script_source"
    record: dict                       # raw record.jsonl row


@dataclass
class LinkedRequest:
    request_id: str
    url: str
    method: str
    status: int
    request_ts_ms: int
    response_ts_ms: int | None
    initiator_type: str | None
    initiator_stack: list[dict]        # raw CDP frames with scriptId/url/lineNumber
    initiator_scripts: list[str]       # unique scriptIds referenced anywhere in the stack
    request_body_path: str | None      # relative path inside notes/
    response_body_path: str | None
    response_mime: str | None
    error: str | None


@dataclass
class ScriptUse:
    script_id: str
    url: str | None
    sha256: str | None
    source_path: str | None            # scripts/{sid}.js relative
    evidence: list[str]                # subset of {"parsed", "initiator_stack", "coverage"}
    coverage_delta_count: int          # sum of call-count deltas across all ranges this step
    executed_functions: list[str]      # function names whose delta_count > 0


@dataclass
class StepFrame:
    step_node_id: str
    start_ts_ms: int
    end_ts_ms: int | None
    start_url: str | None
    end_url: str | None
    actions: list[dict]
    requests: list[LinkedRequest]
    scripts: list[ScriptUse]
    timeline: list[TimelineEvent]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _read_misc_payload(notes_dir: Path, raw_rel: str) -> dict:
    """step_boundary events were stored via _store_misc → misc/{ts}.json."""
    path = notes_dir / raw_rel
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _coverage_delta(start_cov: list[dict], end_cov: list[dict]) -> dict[str, dict]:
    """Per-script execution delta. Returns {scriptId: {url, total, functions: [name, ...]}}."""
    def index(cov: list[dict]) -> dict[str, dict]:
        idx: dict[str, dict] = {}
        for entry in cov:
            sid = entry.get("scriptId")
            if sid is None:
                continue
            idx[sid] = entry
        return idx

    start_idx = index(start_cov)
    end_idx = index(end_cov)
    out: dict[str, dict] = {}
    for sid, end_entry in end_idx.items():
        start_entry = start_idx.get(sid, {"functions": []})
        start_fn_counts: dict[str, int] = {}
        for fn in start_entry.get("functions", []):
            key = fn.get("functionName", "") + "@" + str(fn.get("ranges", [{}])[0].get("startOffset", -1))
            start_fn_counts[key] = sum(r.get("count", 0) for r in fn.get("ranges", []))
        total_delta = 0
        executed: list[str] = []
        for fn in end_entry.get("functions", []):
            key = fn.get("functionName", "") + "@" + str(fn.get("ranges", [{}])[0].get("startOffset", -1))
            end_count = sum(r.get("count", 0) for r in fn.get("ranges", []))
            delta = end_count - start_fn_counts.get(key, 0)
            if delta > 0:
                total_delta += delta
                executed.append(fn.get("functionName", "") or "(anonymous)")
        if total_delta > 0:
            out[sid] = {
                "url": end_entry.get("url"),
                "total": total_delta,
                "functions": executed,
            }
    return out


def build(notes_dir: Path) -> list[StepFrame]:
    record = _load_jsonl(notes_dir / "record.jsonl")
    http_index = _load_jsonl(notes_dir / "http" / "index.jsonl")
    http_by_rid: dict[str, dict] = {}
    for row in http_index:
        rid = row.get("request_id")
        if rid is None:
            continue
        http_by_rid.setdefault(rid, {"request": None, "response": None})
        http_by_rid[rid][row.get("phase", "?")] = row

    boundaries: list[tuple[int, str, str, str | None, dict]] = []
    for row in record:
        if row.get("type") != "step_boundary":
            continue
        raw = row.get("raw")
        payload = _read_misc_payload(notes_dir, raw) if raw else {}
        boundaries.append((
            int(row.get("ts_ms", 0)),
            str(row.get("phase", "?")),
            str(row.get("step_node_id", "?")),
            row.get("url"),
            payload.get("data") or {},
        ))

    starts: dict[str, tuple[int, str | None, dict]] = {}
    ends: dict[str, tuple[int, str | None, dict]] = {}
    for ts_ms, phase, node_id, url, payload in boundaries:
        if phase == "start":
            starts[node_id] = (ts_ms, url, payload)
        elif phase == "end":
            ends[node_id] = (ts_ms, url, payload)

    frames: list[StepFrame] = []
    for node_id, (start_ts, start_url, start_payload) in starts.items():
        end_ts, end_url, end_payload = ends.get(node_id, (None, None, {}))
        upper_bound = end_ts if end_ts is not None else 2**63 - 1
        in_window = [r for r in record if start_ts <= int(r.get("ts_ms", 0)) <= upper_bound]

        actions = [r for r in in_window if r.get("type") == "action"]
        timeline_events: list[TimelineEvent] = []
        seen_rids: set[str] = set()
        for r in in_window:
            t = r.get("type")
            if t in ("action", "http_request", "http_response", "script_source"):
                timeline_events.append(TimelineEvent(
                    ts_ms=int(r.get("ts_ms", 0)),
                    kind=t,
                    record=r,
                ))
            if t in ("http_request", "http_response"):
                rid = r.get("request_id")
                if rid:
                    seen_rids.add(rid)

        requests: list[LinkedRequest] = []
        initiator_script_ids: set[str] = set()
        for rid in seen_rids:
            pair = http_by_rid.get(rid)
            if pair is None:
                continue
            req = pair.get("request") or {}
            resp = pair.get("response") or {}
            stack = req.get("initiator_stack") or []
            stack_sids = sorted({frame.get("scriptId") for frame in stack if frame.get("scriptId")})
            initiator_script_ids.update(stack_sids)
            requests.append(LinkedRequest(
                request_id=rid,
                url=req.get("url", ""),
                method=req.get("method", ""),
                status=int(resp.get("status", 0) or 0),
                request_ts_ms=int(req.get("ts_ms", 0) or 0),
                response_ts_ms=int(resp.get("ts_ms", 0) or 0) if resp else None,
                initiator_type=req.get("initiator_type"),
                initiator_stack=stack,
                initiator_scripts=stack_sids,
                request_body_path=f"http/bodies/{rid}.req",
                response_body_path=f"http/bodies/{rid}.resp",
                response_mime=resp.get("response_mime"),
                error=resp.get("error") or resp.get("body_error"),
            ))

        start_cov = (start_payload or {}).get("coverage") or []
        end_cov = (end_payload or {}).get("coverage") or []
        cov_delta = _coverage_delta(start_cov, end_cov)

        parsed_in_window: set[str] = {
            r.get("script_id") for r in in_window
            if r.get("type") == "script_source" and r.get("script_id")
        }
        all_script_ids = parsed_in_window | initiator_script_ids | set(cov_delta.keys())
        scripts: list[ScriptUse] = []
        for sid in sorted(all_script_ids):
            evidence: list[str] = []
            if sid in parsed_in_window:
                evidence.append("parsed")
            if sid in initiator_script_ids:
                evidence.append("initiator_stack")
            if sid in cov_delta:
                evidence.append("coverage")
            entry = cov_delta.get(sid, {})
            source_path = f"scripts/{sid}.js"
            if not (notes_dir / source_path).exists():
                source_path = None
            scripts.append(ScriptUse(
                script_id=sid,
                url=entry.get("url"),
                sha256=None,
                source_path=source_path,
                evidence=evidence,
                coverage_delta_count=int(entry.get("total", 0)),
                executed_functions=list(entry.get("functions", [])),
            ))

        frames.append(StepFrame(
            step_node_id=node_id,
            start_ts_ms=start_ts,
            end_ts_ms=end_ts,
            start_url=start_url,
            end_url=end_url,
            actions=actions,
            requests=requests,
            scripts=scripts,
            timeline=sorted(timeline_events, key=lambda e: e.ts_ms),
        ))

    frames.sort(key=lambda f: f.start_ts_ms)
    return frames


def write_step_frames(notes_dir: Path) -> Path:
    frames = build(notes_dir)
    out = notes_dir / "step_frames.json"
    out.write_text(
        json.dumps([asdict(f) for f in frames], default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def _main() -> None:
    p = argparse.ArgumentParser(prog="python -m morphnet_v2.timeline")
    p.add_argument("notes_dir", help="Path to a single task's notes directory")
    args = p.parse_args()
    notes_dir = Path(args.notes_dir).resolve()
    if not notes_dir.is_dir():
        print(f"not a directory: {notes_dir}", file=sys.stderr)
        sys.exit(2)
    out = write_step_frames(notes_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
