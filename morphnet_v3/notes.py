"""
notes.py — structured logger for morphnet experiments.

Records every byte we get from the outside world: CDP messages, HTTP traffic,
cookies, page snapshots, JS sources, navigation events, LLM calls, actions.
Lazy: only creates a subdir / file when something is actually recorded into
that channel.

CORE RULE: never truncate. Bodies, cookies, init scripts — always raw and
verbatim. If something looks too large, diagnose; don't blind-truncate.

Two layers:
  1. record.jsonl — the timeline. One line per log() call. Lightweight,
     scannable, holds metadata (ts, caller, type, kwargs) + a 'raw' pointer.
  2. Per-type files — the actual artifact (HTTP body, screenshot, prompt).

Parallel-experiment-safe via contextvars: each asyncio task tree gets its
own active Notes, so two SessionManagers running under asyncio.gather don't
collide on log().

Usage:
    # any module
    from morphnet_v3 import notes
    notes.log(data_type="prompt", data=prompt_str, model="gemini-3-flash-preview")
    notes.log(data_type="screenshot", data=jpeg_bytes)

    # experiment runner
    notes.attach("experiments/results_v2", site_name="swiggy")
    ...run...
    notes.detach()
"""

from __future__ import annotations

import contextvars
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


# ─────────────────────────────────────────────────────────────────
# Time + identity helpers
# ─────────────────────────────────────────────────────────────────

def _filename_ts() -> str:
    """Sortable, filename-safe IST timestamp with millisecond precision.

    e.g. '2026-05-08T14-30-22-123'.
    """
    now = datetime.now(IST)
    return now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond // 1000:03d}"


def _unix_ms() -> int:
    return int(time.time() * 1000)


def _ist_iso() -> str:
    """Human-readable IST timestamp for inline JSONL records."""
    now = datetime.now(IST)
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}+0530"


def _caller_info(skip: int = 0) -> str:
    """'module.function' of whoever invoked the function calling this helper.

    skip=0 is correct when called directly from the user-facing entry point;
    increment for each extra wrapper frame in between.
    """
    frame = sys._getframe(2 + skip)
    module = frame.f_globals.get("__name__", "?")
    func = frame.f_code.co_name
    return f"{module}.{func}"


def site_name_from_url(url: str) -> str:
    """'https://www.swiggy.com/menu/X' → 'swiggy'."""
    host = urlparse(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return (host.split(".")[0] or "unknown").lower()


# ─────────────────────────────────────────────────────────────────
# Notes — one experiment's structured logger
# ─────────────────────────────────────────────────────────────────

class Notes:
    """Holds a base_dir; lazily creates subdirs as types arrive; appends one
    line per log() call to record.jsonl. Each raw artifact lives at a per-
    type path that the record entry references via 'raw'.
    """

    def __init__(self, base_dir: Path | str, site_name: Optional[str] = None):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.site_name = site_name
        self._record_handle = (self.base / "record.jsonl").open("a", encoding="utf-8")
        self._jsonl_handles: dict[Path, Any] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            for h in self._jsonl_handles.values():
                try:
                    h.close()
                except Exception:
                    pass
            self._jsonl_handles.clear()
            try:
                self._record_handle.close()
            except Exception:
                pass

    # ── path / write primitives ─────────────────────────────────

    def _ensure_dir(self, sub: str) -> Path:
        d = self.base / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append_jsonl(self, sub_path: str, record: dict) -> Path:
        full = self.base / sub_path
        with self._lock:
            handle = self._jsonl_handles.get(full)
            if handle is None:
                full.parent.mkdir(parents=True, exist_ok=True)
                handle = full.open("a", encoding="utf-8")
                self._jsonl_handles[full] = handle
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        return full

    def _write_bytes(self, sub_path: str, data: bytes) -> Path:
        full = self.base / sub_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        return full

    def _write_text(self, sub_path: str, data: str) -> Path:
        full = self.base / sub_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(data, encoding="utf-8")
        return full

    def _write_json(self, sub_path: str, data: Any) -> Path:
        full = self.base / sub_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(
            json.dumps(data, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
        return full

    # ── public entry point ──────────────────────────────────────

    def log(self, *, data_type: str, data: Any = None, _caller: Optional[str] = None, **kwargs) -> Optional[str]:
        """Record one outside-world artifact.

        data_type drives storage location and encoding. The raw artifact (full
        bytes / text / dict) goes to a per-type file under base_dir; a
        timeline entry is appended to record.jsonl with kwargs as metadata.

        Returns the relative path of the raw artifact, or None for record-only.
        """
        ts_ms = _unix_ms()
        caller = _caller or _caller_info(skip=0)
        try:
            raw_path = _store(self, data_type, data, **kwargs)
        except Exception as e:
            raw_path = None
            kwargs["_log_error"] = repr(e)

        record: dict[str, Any] = {
            "ts_ms": ts_ms,
            "ts_ist": _ist_iso(),
            "caller": caller,
            "type": data_type,
        }
        if isinstance(raw_path, Path):
            record["raw"] = str(raw_path.relative_to(self.base))
        elif raw_path is not None:
            record["raw"] = str(raw_path)
        for k, v in kwargs.items():
            if k not in record:
                record[k] = v

        with self._lock:
            self._record_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._record_handle.flush()
        return record.get("raw")


# ─────────────────────────────────────────────────────────────────
# Type → storage dispatch. Each handler returns the Path it wrote to
# (or None for record-only types).
# ─────────────────────────────────────────────────────────────────

def _store(notes: Notes, data_type: str, data: Any, **kwargs) -> Optional[Path]:
    handler = _STORE_HANDLERS.get(data_type, _store_misc)
    return handler(notes, data, **kwargs)


def _to_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")


def _store_cdp(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._append_jsonl(
        "cdp/messages.jsonl",
        {"ts_ms": _unix_ms(), **kwargs, "msg": data},
    )


def _store_http_request(notes: Notes, data: Any, *, request_id: Optional[str] = None, **kwargs) -> Path:
    rid = request_id or _filename_ts()
    path = notes._write_bytes(f"http/bodies/{rid}.req", _to_bytes(data))
    notes._append_jsonl(
        "http/index.jsonl",
        {"ts_ms": _unix_ms(), "phase": "request", "request_id": rid, **kwargs},
    )
    return path


def _store_http_response(notes: Notes, data: Any, *, request_id: Optional[str] = None, **kwargs) -> Path:
    rid = request_id or _filename_ts()
    path = notes._write_bytes(f"http/bodies/{rid}.resp", _to_bytes(data))
    notes._append_jsonl(
        "http/index.jsonl",
        {"ts_ms": _unix_ms(), "phase": "response", "request_id": rid, **kwargs},
    )
    return path


def _store_cookies_snapshot(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._write_json(f"http/cookies/all_cookies_{_filename_ts()}.json", data)


def _store_cookie_set(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._append_jsonl(
        "http/cookies/set_events.jsonl",
        {"ts_ms": _unix_ms(), **kwargs, "event": data},
    )


def _store_cookie_sent(notes: Notes, data: Any, **kwargs) -> Path:
    # Mirror of cookie_set for request-side Cookie headers (Chunk 1 expanded A1).
    return notes._append_jsonl(
        "http/cookies/request_events.jsonl",
        {"ts_ms": _unix_ms(), **kwargs, "event": data},
    )


def _store_storage_snapshot(notes: Notes, data: Any, **kwargs) -> Path:
    # localStorage + sessionStorage dumps (Chunk 1 expanded A3).
    return notes._append_jsonl(
        "storage/snapshots.jsonl",
        {"ts_ms": _unix_ms(), **kwargs, "data": data},
    )


def _store_cu_action(notes: Notes, data: Any, **kwargs) -> Path:
    # Rich PageAgent-emitted action records (Chunk 1 expanded B1).
    return notes._append_jsonl(
        "actions/cu_actions.jsonl",
        {"ts_ms": _unix_ms(), **kwargs, "data": data},
    )


def _store_page_html(notes: Notes, data: Any, **kwargs) -> Path:
    text = data if isinstance(data, str) else _to_bytes(data).decode("utf-8", errors="replace")
    return notes._write_text(f"page/{_filename_ts()}_html.html", text)


def _store_page_axtree(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._write_json(f"page/{_filename_ts()}_axtree.json", data)


def _store_planning_tree(notes: Notes, data: Any, **kwargs) -> Path:
    """Mermaid graph text for the planning tree at a given moment."""
    text = data if isinstance(data, str) else _to_bytes(data).decode("utf-8", errors="replace")
    return notes._write_text(f"planning/{_filename_ts()}.mermaid", text)


def _store_pf_raw_payload(notes: Notes, data: Any, *, extraction_id: Optional[str] = None, **kwargs) -> Path:
    eid = extraction_id or _filename_ts()
    return notes._write_json(
        f"pagefilter/{eid}.payload.json",
        {"ts_ms": _unix_ms(), **kwargs, "data": data},
    )


def _store_page_screenshot(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._write_bytes(f"page/{_filename_ts()}_screenshot.jpg", _to_bytes(data))


def _store_page_dom_hash(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._write_text(f"page/{_filename_ts()}_dom_hash.txt", str(data))


def _store_script_source(notes: Notes, data: Any, *, script_id: Optional[str] = None, **kwargs) -> Path:
    sid = script_id or _filename_ts()
    text = data if isinstance(data, str) else _to_bytes(data).decode("utf-8", errors="replace")
    return notes._write_text(f"scripts/{sid}.js", text)


def _store_event_nav(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._append_jsonl("events/navigation.jsonl", {"ts_ms": _unix_ms(), **kwargs, "event": data})


def _store_event_console(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._append_jsonl("events/console.jsonl", {"ts_ms": _unix_ms(), **kwargs, "event": data})


def _store_event_exception(notes: Notes, data: Any, **kwargs) -> Path:
    return notes._append_jsonl("events/exceptions.jsonl", {"ts_ms": _unix_ms(), **kwargs, "event": data})


def _store_llm(notes: Notes, data: Any, *, call_id: Optional[str] = None, **kwargs) -> Path:
    """Persist Gemini prompt OR response. Suffixes are distinct so a shared
    call_id keeps both files (prompt would otherwise be overwritten by response).
    For responses, lift the structured bits (function_call.args + token usage)
    out of the SDK Pydantic objects so we don't lose them to repr-truncation.
    """
    cid = call_id or _filename_ts()
    role = kwargs.pop("_role", None)
    if role is None:
        role = "response" if "success" in kwargs else "prompt"
    serialized: Any = data
    if role == "response":
        serialized = _serialize_gemini_response(data)
    elif role == "prompt":
        serialized = _serialize_gemini_prompt(data)
    return notes._write_json(
        f"llm/{cid}.{role}.json",
        {"ts_ms": _unix_ms(), **kwargs, "data": serialized},
    )


def _serialize_gemini_prompt(contents: Any) -> Any:
    """contents is list[Content|dict|str]. Walk it and pull readable text/parts.

    For Content items (genai_types.Content) we access .role / .parts directly;
    they are always defined on that type. List items that are bare str / dict /
    nested list are passed through (callers occasionally hand-build contents).
    """
    if not isinstance(contents, list):
        return contents
    out: list[Any] = []
    for item in contents:
        if isinstance(item, (str, dict, list)):
            out.append(item)
            continue
        # genai_types.Content has .role and .parts (parts can be None).
        if item.parts is None:
            out.append({"role": item.role, "parts": []})
            continue
        parts_out: list[Any] = []
        for p in item.parts:
            if p.text is not None:
                parts_out.append({"text": p.text})
                continue
            if p.function_call is not None:
                fc = p.function_call
                parts_out.append({
                    "function_call": {"name": fc.name, "args": fc.args, "id": fc.id}
                })
                continue
            if p.function_response is not None:
                fr = p.function_response
                parts_out.append({
                    "function_response": {"name": fr.name, "response": fr.response}
                })
                continue
            parts_out.append(str(p))
        out.append({"role": item.role, "parts": parts_out})
    return out


def _serialize_gemini_response(resp: Any) -> Any:
    """Pull function calls + thoughts + usage out of the SDK response object.

    resp is a genai_types.GenerateContentResponse. Its attributes (candidates,
    usage_metadata, model_version) are always present on the type — but some
    may be None when the model errored or the response is partial.
    """
    out: dict[str, Any] = {}
    cands = resp.candidates or []
    cand_out: list[Any] = []
    for cand in cands:
        content = cand.content
        parts_out: list[Any] = []
        parts = (content.parts if content is not None else None) or []
        for p in parts:
            if p.function_call is not None:
                fc = p.function_call
                parts_out.append({
                    "function_call": {"name": fc.name, "args": fc.args, "id": fc.id}
                })
                continue
            if p.text is not None:
                parts_out.append({"text": p.text, "thought": bool(p.thought)})
        cand_out.append({
            "role": content.role if content is not None else None,
            "parts": parts_out,
            "finish_reason": str(cand.finish_reason) if cand.finish_reason is not None else "",
        })
    out["candidates"] = cand_out
    usage = resp.usage_metadata
    if usage is not None:
        out["usage"] = {
            "prompt_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            "total_tokens": usage.total_token_count,
            "cached_tokens": usage.cached_content_token_count,
        }
    out["model_version"] = resp.model_version
    return out


def _store_action(notes: Notes, data: Any, *, action_id: Optional[str] = None, **kwargs) -> Path:
    aid = action_id or _filename_ts()
    return notes._write_json(f"actions/{aid}.json", {"ts_ms": _unix_ms(), **kwargs, "data": data})


def _store_metadata(notes: Notes, data: Any, **kwargs) -> Path:
    payload = {"ts_ms": _unix_ms()}
    payload.update(kwargs)
    if data is not None:
        payload["data"] = data
    return notes._write_json("metadata.json", payload)


def _store_misc(notes: Notes, data: Any, **kwargs) -> Optional[Path]:
    """Catch-all for unknown types — write to misc/ so we never lose data."""
    if data is None:
        return None
    payload: dict[str, Any] = {"ts_ms": _unix_ms(), **kwargs}
    try:
        json.dumps(data, default=str)
        payload["data"] = data
    except Exception:
        if isinstance(data, (bytes, bytearray)):
            return notes._write_bytes(f"misc/{_filename_ts()}.bin", bytes(data))
        payload["data"] = repr(data)
    return notes._write_json(f"misc/{_filename_ts()}.json", payload)


_STORE_HANDLERS: dict[str, Any] = {
    # CDP
    "cdp_send": _store_cdp,
    "cdp_event": _store_cdp,
    # HTTP
    "http_request": _store_http_request,
    "http_response": _store_http_response,
    "cookies_snapshot": _store_cookies_snapshot,
    "cookie_set": _store_cookie_set,
    "cookie_sent": _store_cookie_sent,
    "storage_snapshot": _store_storage_snapshot,
    "cu_action": _store_cu_action,
    # Page
    "page_html": _store_page_html,
    "page_axtree": _store_page_axtree,
    "page_screenshot": _store_page_screenshot,
    "page_dom_hash": _store_page_dom_hash,
    # PageFilter
    "pf_raw_payload": _store_pf_raw_payload,
    # Planner
    "planning_tree": _store_planning_tree,
    # Scripts
    "script_source": _store_script_source,
    # Events
    "nav_event": _store_event_nav,
    "console": _store_event_console,
    "exception": _store_event_exception,
    # LLM (prompt/response/full call all flow into one file per call_id)
    "llm_call": _store_llm,
    "prompt": _store_llm,
    "response": _store_llm,
    # Actions
    "action": _store_action,
    "action_result": _store_action,
    # Top-level
    "metadata": _store_metadata,
}


# ─────────────────────────────────────────────────────────────────
# Module-level convenience — parallel-safe via ContextVar
# ─────────────────────────────────────────────────────────────────

_active: contextvars.ContextVar[Optional[Notes]] = contextvars.ContextVar(
    "morphnet_v3.notes._active", default=None
)


def attach(results_dir: Path | str, site_name: Optional[str] = None) -> Notes:
    """Open a Notes for one experiment under {results_dir}/{ts}-{site}/.

    Bound to the current asyncio context — parallel SessionManagers spawned
    via asyncio.gather each see their own Notes when calling notes.log().
    """
    if site_name is None:
        site_name = "unknown"
    ts = datetime.now(IST).strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = Path(results_dir) / f"{ts}-{site_name}"
    inst = Notes(exp_dir, site_name=site_name)
    _active.set(inst)
    return inst


def detach() -> None:
    inst = _active.get()
    if inst is not None:
        inst.close()
        _active.set(None)


def log(*, data_type: str, data: Any = None, **kwargs) -> Optional[str]:
    """No-op when no Notes is attached in the current context."""
    inst = _active.get()
    if inst is None:
        return None
    caller = _caller_info(skip=0)
    return inst.log(data_type=data_type, data=data, _caller=caller, **kwargs)


def current() -> Optional[Notes]:
    """Active Notes for the current asyncio context (or None)."""
    return _active.get()
