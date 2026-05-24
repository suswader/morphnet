"""morphnet_v3/temporal.py — single-load, all-inclusive temporal representation.

The contract (locked, per the 8-point thesis):

  tool_builder calls `build_from_capture(capture_dir)` ONCE; it gets a
  `TemporalRepresentation` carrying every event the discovery run produced —
  HTTP nodes, navigations, cookies, storage snapshots, CU actions, step
  boundaries — chronologically ordered. tool_builder never reaches back into
  the capture_dir; this module is the only consumer of on-disk artifacts.

No analysis happens here. No classification, no chaining, no source-trace.
Bodies get parsed only at the SYNTACTIC level (JSON → dict, form-urlencoded
→ list[tuple], multipart → list of parts). Semantic classification
(payload_type, page_class, slot extraction) is Chunk 3's job (tool_builder).

Dataclass consolidation note (vs the original draft): HttpNode absorbs what
would have been HttpReqEvent + HttpRespEvent (they're always paired by
request_id and consumed together — splitting them buys nothing). CookieEvent
absorbs what would have been CookieSetEvent + CookieSentEvent via a
`direction` discriminant (identical payload shape; only the source file +
semantic differ). FrameEvent and the TemporalEvent base are dropped — see
draft.md "Chunk 2 plan" for the rationale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as _email_default_policy
from pathlib import Path
from typing import Any, Literal, Union
from urllib.parse import parse_qsl

logger = logging.getLogger(__name__)


# ── Event dataclasses ─────────────────────────────────────────────────────


@dataclass
class HttpNode:
    """One HTTP request/response pair. The unit of the temporal stream that
    tool_builder chains over. Raw — no payload classification yet.

    Sorting key: `request_ts_ms` (the node 'happens' at request time;
    response_ts_ms is metadata)."""

    request_id: str
    url: str
    method: str
    request_ts_ms: int
    response_ts_ms: int | None
    status: int
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    request_mime: str | None
    response_mime: str | None
    request_body: str | None
    response_body: str | None
    parsed_request_body: Any  # dict for JSON, list[tuple] for form, list for multipart, None otherwise
    parsed_response_body: Any
    initiator_type: str | None
    initiator_stack: list[dict]
    frame_id: str | None
    error: str | None = None


@dataclass
class NavEvent:
    """A navigation: full frame navigation OR within-document SPA route."""

    ts_ms: int
    kind: Literal["frame_navigated", "navigated_within_doc"]
    frame_id: str | None
    from_url: str | None
    to_url: str


@dataclass
class CookieEvent:
    """Cookie sent on a request, OR set on a response. Direction discriminates.

    Payload (`event_data`) is the raw CDP params dict — preserves all
    metadata Chrome carried (blockedCookies, exemptedCookies, partition key)
    so the tracer can dereference cookie sources unambiguously. No truncation."""

    ts_ms: int
    direction: Literal["set", "sent"]
    request_id: str | None
    event_data: dict


@dataclass
class StorageSnapshotEvent:
    """localStorage + sessionStorage snapshot, taken at session start, every
    Page.loadEventFired, and after every PageAgent action."""

    ts_ms: int
    trigger: Literal["session_start", "page_load", "post_action"]
    url: str
    local: dict[str, str]
    session: dict[str, str]


@dataclass
class CuActionEvent:
    """A PageAgent-emitted action record. The Phase-2 source-trace input for
    `user_intent:typed` (via `text`) and `user_intent:click` (via `target_attrs`)."""

    ts_ms: int
    step: int
    kind: str  # click / type_text / scroll / scroll_page / key_press / read_text / copy_paste / drag / draw / hover / probe_drop_zones / drag_batch
    intent: str | None
    target_aid: str | None
    target_attrs: dict | None  # {tag, text, href, value, attrs}
    text: str | None
    success: bool
    navigation_occurred: bool
    post_nav_url: str | None
    dismiss_status: str | None
    reason_code: str | None
    fail_subtype: str | None
    blocker_aid: str | None
    message: str


@dataclass
class StepBoundaryEvent:
    """Orchestrator-emitted bracket around each planning-tree step. Used by
    `step_windows()` to scope chains per CU step."""

    ts_ms: int
    phase: Literal["start", "end"]
    step_node_id: str
    url: str | None


# Tagged union — what the sorted event list contains.
TemporalEvent = Union[
    HttpNode, NavEvent, CookieEvent, StorageSnapshotEvent,
    CuActionEvent, StepBoundaryEvent,
]


# ── Script corpus accessor (lazy file reads) ──────────────────────────────


@dataclass
class ScriptCorpus:
    """Static accessor over `morphnet_v3/sites/<site>/scripts/` (the deduped
    sha256-keyed store). The capture's own `scripts/` dir is keyed by V8
    scriptId (int) and is NOT useful here; the cross-session deduped store is.
    Bytes are read lazily — we don't eat memory holding every bundle."""

    site_dir: Path
    index: dict[str, dict]  # sha256 → {url, length, first_seen_ms, runs}

    @classmethod
    def load_from_site(cls, site_dir: Path) -> ScriptCorpus:
        index_path = site_dir / "scripts" / "index.json"
        index: dict[str, dict] = {}
        if index_path.exists():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    index = raw
            except (json.JSONDecodeError, ValueError):
                index = {}
        return cls(site_dir=site_dir, index=index)

    def url_for(self, sha256: str) -> str | None:
        info = self.index.get(sha256)
        if info is None:
            return None
        return info.get("url")

    def sha_for_url(self, url: str) -> str | None:
        """Reverse lookup — used by tool_builder's link_causality to map
        initiator_stack scriptIds → sha256 via their resolved URL."""
        for sha, info in self.index.items():
            if info.get("url") == url:
                return sha
        return None

    def bytes_for(self, sha256: str) -> str | None:
        """Return the script source for `sha256`, or None if missing."""
        path = self.site_dir / "scripts" / f"{sha256}.js"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="replace")


# ── Top-level container ───────────────────────────────────────────────────


@dataclass
class TemporalRepresentation:
    """The single in-memory artifact tool_builder consumes per task."""

    capture_dir: Path
    events: list[TemporalEvent]   # chronologically sorted by ts_ms (request_ts_ms for HttpNode)
    scripts: ScriptCorpus
    metadata: dict  # start_url, site_name, viewport, etc. — from metadata.json

    # Accessors (declared here; bodies land in increments 3-6 as the loaders
    # for each event family come online).

    def http_nodes(self) -> list[HttpNode]:
        return [e for e in self.events if isinstance(e, HttpNode)]

    def navs(self) -> list[NavEvent]:
        return [e for e in self.events if isinstance(e, NavEvent)]

    def cu_actions(self, before: int | None = None) -> list[CuActionEvent]:
        out = [e for e in self.events if isinstance(e, CuActionEvent)]
        if before is None:
            return out
        return [e for e in out if e.ts_ms < before]

    def prior_responses(self, before: int) -> list[HttpNode]:
        """HttpNodes whose RESPONSE completed before `before`. The chainer
        scans these for whole-value matches of downstream slot values."""
        return [
            n for n in self.http_nodes()
            if n.response_ts_ms is not None and n.response_ts_ms < before
        ]

    def cookies_at(self, ts: int) -> dict[str, str]:
        """Replay set-direction CookieEvents with ts_ms <= ts → {name: value}."""
        return _cookies_at_impl(
            [e for e in self.events if isinstance(e, CookieEvent)], ts,
        )

    def storage_at(self, ts: int) -> dict:
        """Most-recent StorageSnapshotEvent with ts_ms <= ts → {local, session, url}."""
        latest: StorageSnapshotEvent | None = None
        for e in self.events:
            if not isinstance(e, StorageSnapshotEvent):
                continue
            if e.ts_ms > ts:
                continue
            if latest is None or e.ts_ms > latest.ts_ms:
                latest = e
        if latest is None:
            return {"local": {}, "session": {}, "url": None, "trigger": None}
        return {
            "local": dict(latest.local),
            "session": dict(latest.session),
            "url": latest.url,
            "trigger": latest.trigger,
        }

    def step_windows(self) -> list[tuple[str, int, int | None]]:
        """Pair StepBoundaryEvents by step_node_id → [(node_id, start_ts, end_ts), ...].
        Unclosed steps (no end yet) get end_ts=None."""
        starts: dict[str, int] = {}
        ends: dict[str, int] = {}
        for e in self.events:
            if not isinstance(e, StepBoundaryEvent):
                continue
            if e.phase == "start":
                starts[e.step_node_id] = e.ts_ms
            else:
                ends[e.step_node_id] = e.ts_ms
        windows = [(nid, start, ends.get(nid)) for nid, start in starts.items()]
        windows.sort(key=lambda w: w[1])
        return windows


# ── Public entrypoint ─────────────────────────────────────────────────────


# ── Loaders (pure functions, one per artifact) ────────────────────────────


def _jsonl_rows(path: Path) -> list[dict]:
    """Read a JSONL file, skipping malformed lines (per the never-truncate
    rule: keep going on bad lines, don't silently drop bulk on parse error).
    Also filters non-dict rows — some captures contain stray int/str-typed
    JSON lines that would break downstream `.get()` calls."""
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            parsed = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _parse_body(body: str | None, mime: str | None) -> Any:
    """Syntactic body parse only — JSON → dict, form-urlencoded → list[tuple],
    multipart → list of parts (each part: {headers: dict, content: bytes}).
    Returns None for binary/non-text or unparseable bodies. No semantic
    classification (that's Chunk 3's job)."""
    if not body or not mime:
        return None
    m = mime.lower()
    if "json" in m:
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None
    if "x-www-form-urlencoded" in m:
        return list(parse_qsl(body, keep_blank_values=True))
    if "multipart/form-data" in m or m.startswith("multipart/"):
        # email.parser handles multipart structure; we feed it a synthetic
        # message with the content-type header so it finds boundaries.
        try:
            wrapped = f"Content-Type: {mime}\r\n\r\n{body}".encode("utf-8", errors="replace")
            msg = BytesParser(policy=_email_default_policy).parsebytes(wrapped)
            parts: list[dict] = []
            for part in msg.iter_parts():
                parts.append({
                    "headers": dict(part.items()),
                    "content": part.get_content(),  # str for text, bytes for binary
                })
            return parts
        except Exception:
            return None
    return None


def _load_http_pairs(capture_dir: Path) -> list[HttpNode]:
    """Pair `http/index.jsonl` request/response rows by `request_id`, read
    matching body files from `http/bodies/`, build HttpNode list.

    Two filtering rules applied during pairing:
      1. Request rows that arrive twice for the same rid are logged but the
         second overwrites the first. CDP emits multiple `requestWillBeSent`
         for redirect chains with the SAME rid — the final URL wins. The
         redirect history is lost (acceptable per current scope; replay fires
         the final URL directly).
      2. Zombie nodes (rid with `phase="request"` but NO `phase="response"`)
         are DROPPED. They represent failed/cancelled/blocked requests
         (Network.loadingFailed in CDP) — no useful response data, no
         meaningful tool to mint. Skipping them avoids:
           - polluting leaf_index with bodyless URLs
           - wasting an LLM finaliser call on an empty-response tool
           - dispatching the tool at verification only to re-fail."""
    index_path = capture_dir / "http" / "index.jsonl"
    bodies_dir = capture_dir / "http" / "bodies"
    rows = _jsonl_rows(index_path)

    paired: dict[str, dict] = {}
    for r in rows:
        rid = r.get("request_id")
        if rid is None:
            continue
        phase = r.get("phase")
        if phase not in ("request", "response"):
            continue
        bucket = paired.setdefault(rid, {})
        if phase == "request" and "request" in bucket:
            # Redirect chain: CDP shares rid across hops. We log + overwrite.
            logger.debug(
                "http pairing overwrote request row for rid=%s (likely redirect): "
                "prev=%s, new=%s",
                rid,
                (bucket["request"].get("url") or "")[:120],
                (r.get("url") or "")[:120],
            )
        bucket[phase] = r

    nodes: list[HttpNode] = []
    for rid, pair in paired.items():
        if "response" not in pair:
            # Zombie: failed/cancelled/blocked — no response was captured.
            logger.debug("dropping zombie http node rid=%s url=%s",
                         rid, (pair.get("request", {}).get("url") or "")[:120])
            continue
        req = pair.get("request") or {}
        resp = pair.get("response") or {}
        # Request-side body: bodies/{rid}.req
        req_body_path = bodies_dir / f"{rid}.req"
        resp_body_path = bodies_dir / f"{rid}.resp"
        req_body = (
            req_body_path.read_bytes().decode("utf-8", errors="replace")
            if req_body_path.exists() else None
        )
        resp_body = (
            resp_body_path.read_bytes().decode("utf-8", errors="replace")
            if resp_body_path.exists() else None
        )
        req_hdrs = req.get("request_headers") or {}
        resp_hdrs = resp.get("response_headers") or {}
        # Request mime is rarely on the request row; derive from Content-Type header.
        req_mime = None
        for k, v in req_hdrs.items():
            if k.lower() == "content-type":
                req_mime = v.split(";")[0].strip()
                break
        resp_mime = resp.get("response_mime")
        resp_ts_raw = resp.get("ts_ms")

        nodes.append(HttpNode(
            request_id=rid,
            url=req.get("url") or "",
            method=req.get("method") or "",
            request_ts_ms=int(req.get("ts_ms") or 0),
            response_ts_ms=int(resp_ts_raw) if resp_ts_raw is not None else None,
            status=int(resp.get("status") or 0),
            request_headers=dict(req_hdrs),
            response_headers=dict(resp_hdrs),
            request_mime=req_mime,
            response_mime=resp_mime,
            request_body=req_body,
            response_body=resp_body,
            parsed_request_body=_parse_body(req_body, req_mime),
            parsed_response_body=_parse_body(resp_body, resp_mime),
            initiator_type=req.get("initiator_type"),
            initiator_stack=list(req.get("initiator_stack") or []),
            frame_id=req.get("frame_id"),
            error=resp.get("body_error") or resp.get("error"),
        ))
    return nodes


def _load_cookie_events(capture_dir: Path) -> list[CookieEvent]:
    """Load Set-Cookie events (`http/cookies/set_events.jsonl`) AND request-side
    Cookie events (`http/cookies/request_events.jsonl`, new in Chunk 1 A1).
    Both share the same row shape — `event` carries raw CDP params."""
    out: list[CookieEvent] = []
    for direction, fname in (("set", "set_events.jsonl"), ("sent", "request_events.jsonl")):
        for r in _jsonl_rows(capture_dir / "http" / "cookies" / fname):
            out.append(CookieEvent(
                ts_ms=int(r.get("ts_ms") or 0),
                direction=direction,  # type: ignore[arg-type]
                request_id=r.get("request_id"),
                event_data=r.get("event") or {},
            ))
    return out


def _parse_set_cookie(raw: str) -> list[tuple[str, str]]:
    """Split a Set-Cookie header value (possibly multiple cookies separated
    by newlines per CDP's flattening) into [(name, value), ...]. Ignores
    attributes (Path, Domain, Expires, HttpOnly, etc.) — only the name=value
    pair matters for the cookie jar."""
    pairs: list[tuple[str, str]] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        head = line.split(";", 1)[0]
        if "=" not in head:
            continue
        name, value = head.split("=", 1)
        pairs.append((name.strip(), value.strip()))
    return pairs


def _load_cu_actions(capture_dir: Path) -> list[CuActionEvent]:
    """Load PageAgent-emitted cu_action rows from `actions/cu_actions.jsonl`
    (preferred — the Chunk 1 B1 router writes here) or fall back to scanning
    `record.jsonl` for type=cu_action rows that point into misc/ files (for
    captures taken before the notes router landed)."""
    out: list[CuActionEvent] = []
    primary = capture_dir / "actions" / "cu_actions.jsonl"
    if primary.exists():
        for r in _jsonl_rows(primary):
            data = r.get("data") or {}
            out.append(_cu_action_from(int(r.get("ts_ms") or 0), int(r.get("step") or 0), data))
        return out
    # Fallback path — pre-router captures: dereference misc/{ts}.json pointers.
    for r in _jsonl_rows(capture_dir / "record.jsonl"):
        if r.get("type") != "cu_action":
            continue
        raw_rel = r.get("raw")
        data: dict = {}
        if raw_rel:
            payload_path = capture_dir / raw_rel
            if payload_path.exists():
                try:
                    payload = json.loads(payload_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, ValueError):
                    payload = {}
                inner = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(inner, dict):
                    data = inner
        out.append(_cu_action_from(int(r.get("ts_ms") or 0), int(r.get("step") or 0), data))
    return out


def _cu_action_from(ts_ms: int, step: int, data: dict) -> CuActionEvent:
    return CuActionEvent(
        ts_ms=ts_ms,
        step=step,
        kind=data.get("kind") or "",
        intent=data.get("intent"),
        target_aid=data.get("target_aid"),
        target_attrs=data.get("target_attrs"),
        text=data.get("text"),
        success=bool(data.get("success")),
        navigation_occurred=bool(data.get("navigation_occurred")),
        post_nav_url=data.get("post_nav_url"),
        dismiss_status=data.get("dismiss_status"),
        reason_code=data.get("reason_code"),
        fail_subtype=data.get("fail_subtype"),
        blocker_aid=data.get("blocker_aid"),
        message=data.get("message") or "",
    )


def _load_step_boundaries(capture_dir: Path) -> list[StepBoundaryEvent]:
    """Read Orchestrator-emitted step_boundary rows from `record.jsonl`."""
    out: list[StepBoundaryEvent] = []
    for r in _jsonl_rows(capture_dir / "record.jsonl"):
        if r.get("type") != "step_boundary":
            continue
        phase = r.get("phase")
        if phase not in ("start", "end"):
            continue
        out.append(StepBoundaryEvent(
            ts_ms=int(r.get("ts_ms") or 0),
            phase=phase,
            step_node_id=r.get("step_node_id") or "",
            url=r.get("url"),
        ))
    return out


def _load_nav_events(capture_dir: Path) -> list[NavEvent]:
    """Filter `cdp/messages.jsonl` for Page.frameNavigated + Page.navigatedWithinDocument.
    Tracks last URL per frame_id to derive from_url for each event. Note: each
    row wraps the CDP payload as `{ts_ms, event, msg: {method, params}}`."""
    out: list[NavEvent] = []
    last_url_per_frame: dict[str, str] = {}
    for r in _jsonl_rows(capture_dir / "cdp" / "messages.jsonl"):
        msg = r.get("msg") or {}
        method = msg.get("method") or ""
        params = msg.get("params") or {}
        if method == "Page.frameNavigated":
            frame = params.get("frame") or {}
            frame_id = frame.get("id") or ""
            to_url = frame.get("url") or ""
            from_url = last_url_per_frame.get(frame_id)
            last_url_per_frame[frame_id] = to_url
            out.append(NavEvent(
                ts_ms=int(r.get("ts_ms") or 0),
                kind="frame_navigated",
                frame_id=frame_id,
                from_url=from_url,
                to_url=to_url,
            ))
        elif method == "Page.navigatedWithinDocument":
            frame_id = params.get("frameId") or ""
            to_url = params.get("url") or ""
            from_url = last_url_per_frame.get(frame_id)
            last_url_per_frame[frame_id] = to_url
            out.append(NavEvent(
                ts_ms=int(r.get("ts_ms") or 0),
                kind="navigated_within_doc",
                frame_id=frame_id,
                from_url=from_url,
                to_url=to_url,
            ))
    return out


def _load_metadata(capture_dir: Path) -> dict:
    """Read `metadata.json` → the `data` sub-dict (site, start_url, viewport, etc.)."""
    path = capture_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else {}


def _load_storage_snapshots(capture_dir: Path) -> list[StorageSnapshotEvent]:
    """Load localStorage + sessionStorage snapshots. Prefer routed
    `storage/snapshots.jsonl` (post Chunk-1 A3 router); fall back to scanning
    `record.jsonl` for type=storage_snapshot rows that point into misc/."""
    out: list[StorageSnapshotEvent] = []
    primary = capture_dir / "storage" / "snapshots.jsonl"
    if primary.exists():
        for r in _jsonl_rows(primary):
            data = r.get("data") or {}
            out.append(StorageSnapshotEvent(
                ts_ms=int(r.get("ts_ms") or 0),
                trigger=r.get("trigger") or "post_action",  # type: ignore[arg-type]
                url=r.get("url") or "",
                local=dict(data.get("local") or {}),
                session=dict(data.get("session") or {}),
            ))
        return out
    # Fallback — pre-router captures: dereference misc/{ts}.json pointers.
    for r in _jsonl_rows(capture_dir / "record.jsonl"):
        if r.get("type") != "storage_snapshot":
            continue
        raw_rel = r.get("raw")
        data: dict = {}
        if raw_rel:
            payload_path = capture_dir / raw_rel
            if payload_path.exists():
                try:
                    payload = json.loads(payload_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, ValueError):
                    payload = {}
                inner = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(inner, dict):
                    data = inner
        out.append(StorageSnapshotEvent(
            ts_ms=int(r.get("ts_ms") or 0),
            trigger=r.get("trigger") or "post_action",  # type: ignore[arg-type]
            url=r.get("url") or "",
            local=dict(data.get("local") or {}),
            session=dict(data.get("session") or {}),
        ))
    return out


def _cookies_at_impl(events: list[CookieEvent], ts: int) -> dict[str, str]:
    """Replay every Set-Cookie event with ts_ms <= ts and return the
    resulting cookie jar as {name: value}. Set events overwrite earlier
    values for the same name (jar semantics)."""
    jar: dict[str, str] = {}
    for e in events:
        if e.direction != "set" or e.ts_ms > ts:
            continue
        hdrs = e.event_data.get("headers") or {}
        raw = hdrs.get("set-cookie") or hdrs.get("Set-Cookie")
        if not raw:
            continue
        for name, value in _parse_set_cookie(raw):
            jar[name] = value
    return jar


# ── Public entrypoint ─────────────────────────────────────────────────────


def build_from_capture(capture_dir: Path) -> TemporalRepresentation:
    """Single entry point. tool_builder calls this once per task.

    Order: metadata → HTTP nodes → nav → cookies → storage → cu_actions →
    step_boundaries → ScriptCorpus (via metadata["site"] → sites/<site>/scripts).
    Concatenate, sort by request_ts_ms for HttpNode and ts_ms otherwise, wrap.
    """
    metadata = _load_metadata(capture_dir)
    site_name = metadata.get("site") or metadata.get("site_name") or ""
    repo_root = Path(__file__).resolve().parent.parent
    site_dir = repo_root / "morphnet_v3" / "sites" / site_name if site_name else repo_root / "morphnet_v3" / "sites" / "_unknown"
    scripts = ScriptCorpus.load_from_site(site_dir)

    http_nodes = _load_http_pairs(capture_dir)
    nav_events = _load_nav_events(capture_dir)
    cookie_events = _load_cookie_events(capture_dir)
    storage_events = _load_storage_snapshots(capture_dir)
    cu_action_events = _load_cu_actions(capture_dir)
    step_boundary_events = _load_step_boundaries(capture_dir)

    events: list[TemporalEvent] = []
    events.extend(http_nodes)
    events.extend(nav_events)
    events.extend(cookie_events)
    events.extend(storage_events)
    events.extend(cu_action_events)
    events.extend(step_boundary_events)

    def _ts_key(e: TemporalEvent) -> int:
        if isinstance(e, HttpNode):
            return e.request_ts_ms
        return e.ts_ms

    events.sort(key=_ts_key)
    return TemporalRepresentation(
        capture_dir=capture_dir,
        events=events,
        scripts=scripts,
        metadata=metadata,
    )