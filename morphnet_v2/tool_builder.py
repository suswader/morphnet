"""morphnet_v2/tool_builder.py — offline tool synthesis from captured task notes.

Phase 4 of morphnet_v2. Reads a task's `notes/` dir (the artefacts written by
session_manager + notes.py during CU runs) and emits replay-ready tool
candidates. Per-task output: `tool_candidates.json` next to `record.jsonl`.
Cross-task per-site aggregation: `morphnet_v2/sites/{site}/tools.json`.

Operational model: post-task, single-pass, no streaming. CU finishes the task
using its normal flow; this module runs once over the captured representation
with full visibility (every chain has both endpoints visible, every cluster has
its full call count). One batch of LLM finaliser calls runs `asyncio.gather`'d
at end. See `~/.claude/plans/swift-brewing-hare.md` for the full plan.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, unquote_plus, urlsplit

import jinja2

if TYPE_CHECKING:
    from morphnet_v2.page_filter import PageFilterOutput
    from morphnet_v2.session_manager import SessionManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────

SlotKind = Literal["chained", "user_intent_text", "captured"]
DispatchKind = Literal["rest", "graphql", "json_rpc", "form_post", "page_navigate"]
ScriptClass = Literal["framework", "tracker", "ad", "app", "unknown"]


# ─────────────────────────────────────────────────────────────────
# Dataclasses (Chunk 4.1)
# ─────────────────────────────────────────────────────────────────

@dataclass
class ScriptRef:
    script_id: str
    url: str | None
    sha256: str | None


@dataclass
class SlotSource:
    kind: SlotKind
    # chained
    chain_source: str | None = None             # cluster_key OR "live_page"
    response_jmespath: str | None = None
    html_attribute: str | None = None
    container_signature: str | None = None
    list_selector_needed: bool = False
    # user_intent_text
    captured_examples: list[str] = field(default_factory=list)
    # captured
    observed_values: list[str] = field(default_factory=list)
    required: bool = True


@dataclass
class SampleRequest:
    method: str
    url: str
    headers: dict[str, str]
    body_path: str | None
    body_preview: str = ""


@dataclass
class SampleResponse:
    status: int
    mime: str
    body_path: str
    body_size: int
    body_preview: str = ""


@dataclass
class ToolCandidate:
    cluster_key: str
    dispatch_kind: DispatchKind
    sample_request: SampleRequest
    sample_response: SampleResponse
    slots: dict[str, SlotSource] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    call_count: int = 0
    tasks_seen: list[str] = field(default_factory=list)
    initiator_scripts: list[ScriptRef] = field(default_factory=list)
    rule_trace: list[str] = field(default_factory=list)
    tool_id: str = ""
    capability_statement: str = ""
    slot_descriptions: dict[str, str] = field(default_factory=dict)
    verdict: Literal["keep", "discard", "pending"] = "pending"
    verdict_reason: str = ""


# ─────────────────────────────────────────────────────────────────
# Chunk 4.2 — Noise / script / dispatch classifiers + junk gate
# ─────────────────────────────────────────────────────────────────

NOISE_HOSTS = re.compile(r"""
    ^(.*\.)?(
        googletagmanager\.com|google-analytics\.com|analytics\.google\.com|
        googleadservices\.com|googlesyndication\.com|doubleclick\.net|
        scorecardresearch\.com|criteo\.(com|net)|taboola\.com|outbrain\.com|
        clevertap-prod\.com|firebaselogging-pa\.googleapis\.com|
        firebaseinstallations\.googleapis\.com|fcmregistrations\.googleapis\.com|
        firebaseremoteconfig\.googleapis\.com|firebase\.googleapis\.com|
        clarity\.ms|hotjar\.com|segment\.io|snowplowanalytics\.com|
        scout\.services\.lego\.com|cdp\.services\.lego\.com|
        adtech-events\.bookmyshow\.com|
        mixpanel\.com|branch\.io|appsflyer\.com|fbcdn\.net|
        connect\.facebook\.net|graph\.facebook\.com|www\.facebook\.com|facebook\.com|t\.co|
        beacon\.adobedc\.net|demdex\.net|omtrdc\.net|2o7\.net|
        bat\.bing\.com|stats\.g\.doubleclick\.net|nr-data\.net|newrelic\.com|
        sentry\.io|sentry-cdn\.com|bugsnag\.com|datadoghq\.com|datadoghq-browser-logs\.com|
        cdn\.bizibly\.com|bizibly\.com|fundingchoicesmessages\.google\.com|
        adservice\.google\.[a-z]+|cm\.g\.doubleclick\.net|
        analytics\.swiggy\.com|wzrkt\.com|wpadm\.com|prodregistryv2\.org|
        awswaf\.com|wafs\.mfilterit\.net|mfilterit\.net|go-mpulse\.net|akstat\.io|
        bnc\.lt|app\.link|
        wafs_v5_skew_api\.dhiraj7045\.workers\.dev|
        analytics\.s\.bookmyshow\.com|in1\.clevertap-prod\.com|
        bam\.nr-data\.net|bam-cell\.nr-data\.net|edge\.fullstory\.com|
        cdn\.boomtrain\.com|t\.boomtrain\.com|unagi\.amazon\.in|fls-eu\.amazon\.in|
        aax-eu-zaz\.amazon\.in
    )$
""", re.VERBOSE)

NOISE_PATHS = re.compile(r"""
    /(collect|tracking|telemetry|metrics|ping|beacon|pixel|impressions?|
      event|events|log|logs|analytics|tagmanager|gtm|gtag|ga\.js|ga4|
      stats|track|sdk|sentry|raven|hotjar|sentry-debug-id|err|errors|
      crash|nr-rum|jserrors|web-vitals|vitals|rum|
      ads/|adservice|adsystem|cdn-cgi/(rum|trace|zaraz|beacon|insights)|
      googletagmanager|recaptcha/api|gen_204|gen204|csi|jsapi/csi|
      pagead/|com\.snowplowanalytics|dl-event-ingest|v1/rgstr|api/mri/|
      measurement/conversion|tr/|p\.gif|t\.gif|b\.gif|pixel\.gif|
      uedata|com\.amazon\.csm)/?
""", re.VERBOSE)

STATIC_EXTS = (".js", ".mjs", ".css", ".woff", ".woff2", ".ttf", ".otf", ".eot",
               ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp",
               ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg", ".pdf", ".m3u8",
               ".ts", ".map")


def is_noise_host(host: str) -> bool:
    return bool(NOISE_HOSTS.match(host or ""))


def is_noise_path(path: str) -> bool:
    return bool(NOISE_PATHS.search(path or ""))


def is_static(path: str) -> bool:
    p = path.split("?", 1)[0].split("#", 1)[0].lower()
    return p.endswith(STATIC_EXTS)


def classify_script_url(url: str | None) -> ScriptClass:
    """Classify a JS script URL into framework / tracker / ad / app."""
    if not url:
        return "unknown"
    u = url.lower()
    # framework / vendor bundles
    if re.search(r"(react-dom|framework-|polyfill|webpack/runtime|_next/static/chunks/(framework|main|polyfills|webpack|main-app))", u):
        return "framework"
    # trackers (host-based)
    host = urlsplit(url).netloc.lower()
    if is_noise_host(host):
        return "tracker"
    if "googletagmanager.com" in host or "gtag" in u or "gtm.js" in u:
        return "tracker"
    if "scout.services.lego.com" in host or "wzrkt.com" in host:
        return "tracker"
    # ad
    if any(x in host for x in ("doubleclick", "googleads", "criteo", "taboola", "outbrain", "amazon-adsystem", "googletagservices")):
        return "ad"
    # app code — everything else
    return "app"


def is_junk_response(status: int, mime: str | None, body_size: int) -> bool:
    """Rule 0 junk gate. Returns True if the response should be dropped.

    Multi-signal, NO string-key matching on body content (too fragile across
    sites). Just: bad status, OR body too small for its declared MIME to be
    meaningful.
    """
    if status >= 400:
        return True
    if status == 0:
        return True
    m = (mime or "").lower()
    if m.startswith("text/html"):
        return body_size < 500   # 4-byte stubs, etc.
    if "json" in m or m.startswith("application/"):
        return body_size < 30    # {"data":null}=14 chars is borderline; 30 keeps {"success":true}=17 out only if status was bad anyway
    # other (text, octet-stream, image): low size threshold
    return body_size < 30


# ─── Dispatch identity ──────────────────────────────────────────

def cluster_identity(method: str, url: str, request_body: str | None) -> tuple[DispatchKind, str]:
    """Compute the dispatch identity for clustering. Returns (kind, cluster_key)."""
    host = urlsplit(url).netloc.lower()
    path = urlsplit(url).path

    # GraphQL: POST with body containing "operationName"
    if method == "POST" and request_body:
        try:
            body = json.loads(request_body)
            if isinstance(body, dict) and "operationName" in body:
                op = body.get("operationName") or "anonymous"
                return ("graphql", f"POST {host}{path}::{op}")
            if isinstance(body, dict) and "method" in body and "jsonrpc" in body:
                m = body.get("method") or "?"
                return ("json_rpc", f"POST {host}{path}::{m}")
        except (json.JSONDecodeError, ValueError):
            pass

    # page_navigate: GET returning text/html, and the user navigated to it.
    # We mark this later in the pipeline; for now treat any GET as REST.
    # If the response mime is text/html the caller can override.
    return ("rest", f"{method} {host}{path}")


def parse_graphql_operation(body: str | None) -> str | None:
    if not body:
        return None
    try:
        b = json.loads(body)
        if isinstance(b, dict) and "operationName" in b:
            return b["operationName"]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ─────────────────────────────────────────────────────────────────
# Chunk 4.3 — Chain detection: value normalisation, indexing, scoring
# ─────────────────────────────────────────────────────────────────

DATE_RE = re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$|^\d{4}-\d{1,2}-\d{1,2}$")
NONCE_RE_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
NONCE_RE_TS = re.compile(r"^\d{10,13}$")
NONCE_RE_HEX = re.compile(r"^[A-Fa-f0-9]{20,64}$")


def normalise_value(v: Any) -> str:
    """URL-decode + lowercase + datetime-to-ISO if shape matches."""
    if v is None:
        return ""
    s = unquote_plus(str(v))
    if DATE_RE.match(s):
        m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", s)
        if m:
            return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return s.lower().strip()


_LOW_ENTROPY = frozenset((
    "true", "false", "null", "none", "undefined", "yes", "no", "on", "off",
    "in", "en", "us", "uk", "ca", "au", "id", "kr", "jp", "cn", "fr", "de",
    "0", "1", "2", "3", "-1", "",
))


def looks_low_entropy(v: str) -> bool:
    if not v or len(v) < 3:
        return True
    if v.lower() in _LOW_ENTROPY:
        return True
    if v.isdigit() and len(v) < 3:
        return True
    if len(set(v.lower())) <= 2:
        return True
    return False


def looks_like_nonce(v: str) -> bool:
    return bool(NONCE_RE_UUID.match(v) or NONCE_RE_TS.match(v) or NONCE_RE_HEX.match(v))


# ─── Tokenise + score ───────────────────────────────────────────

_TOKEN_SPLIT = re.compile(r"[_\s\-]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NONALNUM = re.compile(r"[^a-z0-9]")

# Common abbreviations we expand for token-overlap matching. Each key gets its
# alias added alongside it, so e.g. "sourceStnCode" tokens become
# {"source","stn","station","code"} — letting it match "stationCode" cleanly.
_TOKEN_ALIASES: dict[str, str] = {
    "stn": "station", "stns": "stations",
    "stntype": "stationtype",
    "cd": "code", "nm": "name", "no": "number", "nbr": "number",
    "tkt": "ticket", "addr": "address", "amt": "amount", "qty": "quantity",
    "dt": "date", "ts": "timestamp", "tm": "time",
    "lat": "latitude", "lng": "longitude", "lon": "longitude",
    "src": "source", "dst": "destination", "dest": "destination",
    "img": "image", "url": "link", "ref": "reference",
}


def tokens(s: str) -> set[str]:
    if not s:
        return set()
    parts = _TOKEN_SPLIT.split(s)
    out: set[str] = set()
    for p in parts:
        p = _NONALNUM.sub("", p.lower())
        if not p:
            continue
        if p in ("id", "cd", "no", "nm", "tp", "pid", "cid", "tid", "sku", "asin"):
            out.add(p)
        elif len(p) >= 2:
            out.add(p)
        # Expand known abbreviations so "stn" matches "station" etc.
        alias = _TOKEN_ALIASES.get(p)
        if alias:
            out.add(alias)
    return out


def token_overlap(param_name: str, field_name: str) -> float:
    tp = tokens(param_name)
    tf = tokens(field_name)
    if not tp or not tf:
        return 0.0
    if tf.issubset(tp):
        return 1.0
    if tp.issubset(tf):
        return 0.9
    inter = tp & tf
    if not inter:
        return 0.0
    return len(inter) / max(len(tp), len(tf))


# ─── Response indexing ──────────────────────────────────────────

@dataclass
class IndexEntry:
    """One leaf value's location in a response."""
    request_id: str          # which response it came from
    cluster_key: str         # which cluster owns that response
    path: str                # JMESPath OR HTML CSS-ish path
    field_name: str          # last path segment / attribute name
    container_signature: str | None  # for HTML sources only


def index_json_body(body_text: str, request_id: str, cluster_key: str) -> dict[str, list[IndexEntry]]:
    """Walk a JSON body and index every leaf string/number that passes the
    entropy filter. Returns {normalised_value → [IndexEntry, ...]}."""
    out: dict[str, list[IndexEntry]] = defaultdict(list)
    try:
        j = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        return out

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:100]):   # cap deep walks
                walk(v, f"{path}[{i}]")
        else:
            if obj is None:
                return
            sv = str(obj)
            if looks_low_entropy(sv):
                return
            nv = normalise_value(sv)
            field = path.rsplit(".", 1)[-1].split("[", 1)[0]
            out[nv].append(IndexEntry(request_id=request_id, cluster_key=cluster_key,
                                       path=path, field_name=field, container_signature=None))

    walk(j)
    return out


async def index_html_body(
    body_text: str,
    base_url: str,
    request_id: str,
    cluster_key: str,
    sm: "SessionManager",
) -> dict[str, list[IndexEntry]]:
    """Walk an HTML body using page_filter(enumerate_mode=True). Index every
    data-* attribute value + leaf textContent that passes entropy filter."""
    out: dict[str, list[IndexEntry]] = defaultdict(list)
    try:
        pf_output = await extract_response_html(body_text, base_url, sm)
    except Exception as e:
        logger.warning("html extract failed for rid=%s: %s", request_id, e)
        return out

    # Walk containers; for each, index every data-attribute value
    for c in pf_output.containers:
        data_attrs = getattr(c, "data_attributes", {}) or {}
        tag = getattr(c, "tag", "?")
        sig = _container_signature(c)
        for attr, val in data_attrs.items():
            if not val or looks_low_entropy(str(val)):
                continue
            if attr in ("cdx-aid",):
                continue
            nv = normalise_value(val)
            out[nv].append(IndexEntry(
                request_id=request_id,
                cluster_key=cluster_key,
                path=f"$.[{tag}].{attr}",
                field_name=attr,
                container_signature=sig,
            ))
    return out


def _container_signature(c: Any) -> str:
    """Compact descriptor of a container — used so the executor can later
    filter live PageFilterOutput.containers to the same signature."""
    tag = getattr(c, "tag", "div")
    data_attrs = getattr(c, "data_attributes", {}) or {}
    parts = [tag]
    # Prefer data-component-type / data-test-id / similar discriminators
    for key in ("data-component-type", "data-test-id", "data-testid", "data-cy", "data-component-id"):
        if key in data_attrs:
            parts.append(f"[{key}={data_attrs[key]}]")
            break
    if "data-asin" in data_attrs:
        parts.append("[data-asin]")
    return "".join(parts)


def detect_chain(
    value: str,
    param_name: str,
    response_index: dict[str, list[IndexEntry]],
    threshold: float = 0.5,
) -> tuple[IndexEntry, float] | None:
    """Find best chain match for (param_name, value). Returns (entry, score)
    if a chain candidate scores above threshold, else None."""
    nv = normalise_value(value)
    candidates = response_index.get(nv)
    if not candidates:
        # Try comma-split for composite values like "B0X,B0Y,B0Z"
        if "," in value:
            for part in value.split(","):
                part_nv = normalise_value(part.strip())
                if part_nv in response_index:
                    candidates = response_index[part_nv]
                    break
    if not candidates:
        return None
    if looks_low_entropy(nv):
        return None
    best: tuple[IndexEntry, float] | None = None
    for entry in candidates:
        score = token_overlap(param_name, entry.field_name)
        if best is None or score > best[1]:
            best = (entry, score)
    if best and best[1] >= threshold:
        return best
    return None


# ─── Causality linking ──────────────────────────────────────────

def link_causality(
    request_initiator_stack: list[dict],
    script_classes: dict[str, ScriptClass],
) -> list[str]:
    """Walk initiator_stack top-down; return the app-code scriptIds (no
    framework / tracker / ad). The first (top-most) app-code scriptId is the
    issuing function; the full list is the dependency set."""
    app_scripts: list[str] = []
    for frame in request_initiator_stack or []:
        sid = frame.get("scriptId")
        if not sid:
            continue
        cls = script_classes.get(str(sid), classify_script_url(frame.get("url")))
        if cls == "app":
            app_scripts.append(str(sid))
    return app_scripts


# ─────────────────────────────────────────────────────────────────
# Chunk 4.4 — User-intent text + captured fallback
# ─────────────────────────────────────────────────────────────────

def detect_user_intent_text(value: str, type_action_texts: list[str]) -> bool:
    """Rule 2. Currently inert until instrumentation 0a lands (type_text
    actions don't capture the typed string yet). The function is defined here
    so once 0a lands the rule fires automatically.

    Returns True iff the normalised value substring-matches any normalised
    typed-text from a recent CU type_text action."""
    if not value or not type_action_texts:
        return False
    nv = normalise_value(value)
    for t in type_action_texts:
        nt = normalise_value(t)
        if not nt:
            continue
        if nv == nt or nt in nv or (len(nv) > 2 and nv in nt):
            return True
    return False


def make_captured_source(observed: list[str]) -> SlotSource:
    """Rule 3 — captured fallback. `required` is False iff any observed value
    is empty/null/undefined-shaped (those are the "empirically the server
    tolerates missing this" signal)."""
    OPTIONAL_VALUES = {"", "null", "undefined", "none"}
    required = not any((v or "").lower() in OPTIONAL_VALUES for v in observed)
    # dedupe preserving order
    seen: set[str] = set()
    dedup: list[str] = []
    for v in observed:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return SlotSource(kind="captured", observed_values=dedup[:10], required=required)


# ─────────────────────────────────────────────────────────────────
# Chunk 4.5 — Graph isolation filter
# ─────────────────────────────────────────────────────────────────

def graph_isolation_filter(candidates: list[ToolCandidate]) -> list[ToolCandidate]:
    """Drop candidates that have zero chain-in, zero chain-out, zero
    user-intent slots. These are isolated nodes — pure telemetry.

    chain-out is computed across the full task: does any other candidate's
    chained slot point at THIS candidate as its source?
    """
    # First pass: collect "is anyone chaining FROM cluster X?"
    chain_targets: set[str] = set()
    for c in candidates:
        for slot in c.slots.values():
            if slot.kind == "chained" and slot.chain_source and slot.chain_source != "live_page":
                chain_targets.add(slot.chain_source)

    survivors: list[ToolCandidate] = []
    for c in candidates:
        chain_in = sum(1 for s in c.slots.values() if s.kind == "chained")
        chain_out = 1 if c.cluster_key in chain_targets else 0
        user_intent = sum(1 for s in c.slots.values() if s.kind == "user_intent_text")
        if chain_in == 0 and chain_out == 0 and user_intent == 0:
            c.rule_trace.append("dropped_by_graph_isolation")
            continue
        c.rule_trace.append(f"survived_isolation: chain_in={chain_in} chain_out={chain_out} user_intent={user_intent}")
        survivors.append(c)
    return survivors


# ─────────────────────────────────────────────────────────────────
# Chunk 4.6 — LLM finaliser
# ─────────────────────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt_template(name: str) -> jinja2.Template:
    text = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    return jinja2.Template(text)


# Note: Gemini's response_schema doesn't support `additionalProperties` on
# objects with dynamic keys. We model `slot_descriptions` as a string instead
# (the LLM emits a JSON-encoded mapping) and parse it client-side.
FINALISER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["keep", "discard"]},
        "reason": {"type": "string"},
        "tool_id": {"type": "string"},
        "capability_statement": {"type": "string"},
        # slot_descriptions: emitted as a JSON-encoded object mapping slot_name → description
        "slot_descriptions_json": {"type": "string"},
    },
    "required": ["verdict", "reason", "tool_id", "capability_statement", "slot_descriptions_json"],
}


async def finalise_cluster(
    candidate: ToolCandidate,
    sm: "SessionManager",
    preceding_actions: list[str] | None = None,
    causal_initiator: str = "",
) -> None:
    """One Gemini Flash call. Mutates `candidate` in place: sets verdict,
    tool_id, capability_statement, slot_descriptions."""
    tpl = _load_prompt_template("tool_finaliser.j2")
    sample_url = urlsplit(candidate.sample_request.url)
    rendered = tpl.render(
        method=candidate.sample_request.method,
        host=sample_url.netloc,
        path=sample_url.path,
        dispatch_kind=candidate.dispatch_kind,
        call_count=candidate.call_count,
        slots=candidate.slots,
        constants=candidate.constants,
        sample_response_preview=candidate.sample_response.body_preview,
        preceding_actions=preceding_actions or [],
        causal_initiator=causal_initiator or "(none)",
    )
    try:
        result = await sm.call_gemini(
            model="gemini-3-flash-preview",
            contents=[rendered],
            response_schema=FINALISER_RESPONSE_SCHEMA,
            temperature=0.3,
            thinking_budget=1024,
            max_output_tokens=2048,
        )
        if isinstance(result, dict):
            candidate.verdict = result.get("verdict", "discard")
            candidate.verdict_reason = result.get("reason", "")
            candidate.tool_id = result.get("tool_id", "")
            candidate.capability_statement = result.get("capability_statement", "")
            # Parse the JSON-encoded slot_descriptions string
            sd_raw = result.get("slot_descriptions_json", "") or "{}"
            try:
                sd = json.loads(sd_raw) if isinstance(sd_raw, str) else sd_raw
                candidate.slot_descriptions = {str(k): str(v) for k, v in (sd or {}).items()}
            except (json.JSONDecodeError, ValueError):
                candidate.slot_descriptions = {}
        else:
            candidate.verdict = "discard"
            candidate.verdict_reason = "LLM returned non-dict"
    except Exception as e:
        logger.warning("finalise_cluster failed for %s: %s", candidate.cluster_key, e)
        candidate.verdict = "discard"
        candidate.verdict_reason = f"finaliser error: {e!r}"


# ─────────────────────────────────────────────────────────────────
# Chunk 4.7 — Build orchestration + CLI
# ─────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def _load_scripts(notes_dir: Path) -> dict[str, ScriptRef]:
    """Walk scripts/ dir; build {script_id → ScriptRef}. The script_id is
    derived from the filename (which is the V8 scriptId for our captures, or
    the sha256 for site-context dedup'd files)."""
    out: dict[str, ScriptRef] = {}
    scripts_dir = notes_dir / "scripts"
    if not scripts_dir.exists():
        return out
    # The record.jsonl entries with type=script_source carry the script_id
    # alongside url + sha256. Walk record.jsonl to get that mapping.
    for row in _read_jsonl(notes_dir / "record.jsonl"):
        if row.get("type") != "script_source":
            continue
        sid = str(row.get("script_id") or "")
        url = row.get("url")
        sha = row.get("sha256")
        if sid and sid not in out:
            out[sid] = ScriptRef(script_id=sid, url=url, sha256=sha)
    return out


def _build_script_classes(scripts: dict[str, ScriptRef]) -> dict[str, ScriptClass]:
    return {sid: classify_script_url(sref.url) for sid, sref in scripts.items()}


def _build_pairs(notes_dir: Path) -> list[tuple[str, dict, dict]]:
    """Walk http/index.jsonl + body files. Return [(rid, req_row, resp_row)]
    for non-OPTIONS pairs with bodies on disk."""
    idx = notes_dir / "http" / "index.jsonl"
    pairs: dict[str, dict] = {}
    for row in _read_jsonl(idx):
        rid = row.get("request_id")
        if not rid:
            continue
        pairs.setdefault(rid, {})[row.get("phase", "?")] = row

    out: list[tuple[str, dict, dict]] = []
    for rid, pair in pairs.items():
        req = pair.get("request") or {}
        resp = pair.get("response") or {}
        if (req.get("method") or "") == "OPTIONS":
            continue
        url = req.get("url") or ""
        if not url or url.startswith(("data:", "blob:")):
            continue
        out.append((rid, req, resp))
    # sort by request ts_ms
    out.sort(key=lambda t: int(t[1].get("ts_ms") or 0))
    return out


def _read_body(notes_dir: Path, body_path_rel: str | None) -> tuple[str, int]:
    if not body_path_rel:
        return "", 0
    p = notes_dir / body_path_rel
    if not p.exists():
        return "", 0
    sz = p.stat().st_size
    if sz == 0:
        return "", 0
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", sz
    return txt, sz


async def build_candidates(notes_dir: Path, sm: "SessionManager") -> list[ToolCandidate]:
    """Single-pass tool synthesis. Returns the surviving candidates after
    graph isolation and LLM finalisation. The caller decides whether to
    `keep` or `discard` based on `candidate.verdict`."""
    notes_dir = Path(notes_dir)
    if not (notes_dir / "record.jsonl").exists():
        raise FileNotFoundError(f"no record.jsonl in {notes_dir}")

    # Step 1 — Static indexes
    scripts = _load_scripts(notes_dir)
    script_classes = _build_script_classes(scripts)

    # Action timeline (typed text unavailable until 0a, but we surface what we have)
    action_records: list[dict] = []
    record = _read_jsonl(notes_dir / "record.jsonl")
    for row in record:
        if row.get("type") == "action":
            action_records.append(row)
    type_action_texts: list[str] = []  # placeholder; populated once instrumentation 0a lands
    for a in action_records:
        # When 0a lands, the typed text will live under a["text"] or a["data"]["text"]
        txt = a.get("text") or (a.get("data") or {}).get("text") or ""
        if txt and a.get("kind") == "type_text" or (a.get("data") or {}).get("kind") == "type_text":
            type_action_texts.append(str(txt))

    # Step 2 — Junk gate + cluster
    pairs = _build_pairs(notes_dir)
    clusters: dict[str, dict[str, Any]] = {}   # cluster_key → {dispatch_kind, requests:[(rid, req, resp, req_body, resp_body, resp_size)]}
    response_index: dict[str, list[IndexEntry]] = defaultdict(list)

    for rid, req, resp in pairs:
        url = req.get("url") or ""
        method = req.get("method") or ""
        status = int(resp.get("status") or 0)
        mime = resp.get("response_mime") or ""
        body_size = 0
        body_path = f"http/bodies/{rid}.resp"
        body_full_path = notes_dir / body_path
        if body_full_path.exists():
            body_size = body_full_path.stat().st_size

        # Junk gate (Rule 0)
        if is_junk_response(status, mime, body_size):
            continue
        # Host / path filters
        host = urlsplit(url).netloc.lower()
        path = urlsplit(url).path
        if is_noise_host(host) or is_noise_path(path) or is_static(path):
            continue

        # Read request body if present
        req_body_path = f"http/bodies/{rid}.req"
        req_body, _ = _read_body(notes_dir, req_body_path)

        # Cluster identity
        kind, cluster_key = cluster_identity(method, url, req_body)
        # Page-navigate override: text/html GET returning a real page
        if method == "GET" and mime.startswith("text/html") and status == 200:
            kind = "page_navigate"

        c = clusters.setdefault(cluster_key, {
            "dispatch_kind": kind,
            "requests": [],
            "issuing_scripts": set(),
        })
        c["requests"].append({
            "rid": rid, "req": req, "resp": resp,
            "req_body": req_body, "body_size": body_size,
            "body_path": body_path,
        })
        # Initiator-stack → app scriptIds
        app_sids = link_causality(req.get("initiator_stack") or [], script_classes)
        for sid in app_sids:
            c["issuing_scripts"].add(sid)

    # Step 3 — Build response_value_index over all surviving responses
    # JSON + HTML pass. HTML uses page_filter offline (Chunk 4.0).
    for cluster_key, c in clusters.items():
        for r in c["requests"]:
            resp = r["resp"]
            mime = (resp.get("response_mime") or "").lower()
            if not mime:
                continue
            body_text, _ = _read_body(notes_dir, r["body_path"])
            if not body_text:
                continue
            if "json" in mime:
                idx = index_json_body(body_text, r["rid"], cluster_key)
                for v, entries in idx.items():
                    response_index[v].extend(entries)
            elif mime.startswith("text/html"):
                idx = await index_html_body(body_text, r["req"].get("url") or "", r["rid"], cluster_key, sm)
                for v, entries in idx.items():
                    response_index[v].extend(entries)

    # Step 4 — Per-cluster slot classification
    candidates: list[ToolCandidate] = []
    for cluster_key, c in clusters.items():
        if not c["requests"]:
            continue
        # Pick canonical sample: largest body with status 200
        sample = max(c["requests"], key=lambda r: (1 if (r["resp"].get("status") == 200) else 0, r["body_size"]))
        # All varying params across the cluster's calls
        param_values: dict[str, list[str]] = defaultdict(list)
        for r in c["requests"]:
            url = r["req"].get("url") or ""
            for k, v in parse_qsl(urlsplit(url).query, keep_blank_values=True):
                param_values[k].append(v)
            # For graphql/json_rpc, also pull body's "variables" if present
            req_body = r.get("req_body") or ""
            if req_body:
                try:
                    bj = json.loads(req_body)
                    if isinstance(bj, dict) and isinstance(bj.get("variables"), dict):
                        for k, v in bj["variables"].items():
                            param_values[f"variables.{k}"].append(json.dumps(v) if not isinstance(v, str) else v)
                except (json.JSONDecodeError, ValueError):
                    pass

        slots: dict[str, SlotSource] = {}
        constants: dict[str, Any] = {}
        rule_trace: list[str] = []
        for pname, values in param_values.items():
            distinct = list(dict.fromkeys(values))
            # Try chain detection on the freshest non-empty value first, even
            # for single-call clusters. With only 1 call, params might still
            # chain to a prior response (e.g. availability_calendar's
            # sourceStationCode came from autocomplete's stationCode field).
            v_repr = next((v for v in reversed(distinct) if v), "")
            chain = detect_chain(v_repr, pname, response_index) if v_repr else None
            if chain is not None:
                entry, score = chain
                src = entry.cluster_key
                slots[pname] = SlotSource(
                    kind="chained",
                    chain_source=src,
                    response_jmespath=entry.path if entry.container_signature is None else None,
                    html_attribute=entry.field_name if entry.container_signature else None,
                    container_signature=entry.container_signature,
                    list_selector_needed=("[" in entry.path),
                    # Record observed values in their ORIGINAL format so the
                    # finaliser LLM can describe the format correctly (e.g.
                    # dateOfJourney = "27-05-2026" not the normalised ISO).
                    observed_values=[v for v in distinct if v][:5],
                )
                rule_trace.append(f"chain {pname}={v_repr!r} → {src} via {entry.field_name} score={score:.2f}")
                continue
            if v_repr and detect_user_intent_text(v_repr, type_action_texts):
                slots[pname] = SlotSource(kind="user_intent_text", captured_examples=distinct[:5])
                rule_trace.append(f"user_intent_text {pname} examples={distinct[:3]}")
                continue
            # No chain, no user-intent. Single value → constant; multiple → captured slot.
            if len(distinct) == 1:
                constants[pname] = distinct[0]
                continue
            slots[pname] = make_captured_source(distinct)
            rule_trace.append(f"captured {pname} observed={distinct[:3]}")

        # Build initiator_scripts list
        init_scripts: list[ScriptRef] = []
        for sid in c["issuing_scripts"]:
            sref = scripts.get(sid)
            if sref:
                init_scripts.append(sref)
            else:
                init_scripts.append(ScriptRef(script_id=sid, url=None, sha256=None))

        # Build SampleRequest / SampleResponse
        sample_url = sample["req"].get("url") or ""
        sample_method = sample["req"].get("method") or ""
        sample_headers = sample["req"].get("request_headers") or {}
        sample_req_body, _ = _read_body(notes_dir, f"http/bodies/{sample['rid']}.req")
        sample_resp = sample["resp"]
        sample_resp_body, sample_resp_size = _read_body(notes_dir, sample["body_path"])

        cand = ToolCandidate(
            cluster_key=cluster_key,
            dispatch_kind=c["dispatch_kind"],
            sample_request=SampleRequest(
                method=sample_method,
                url=sample_url,
                headers=sample_headers,
                body_path=f"http/bodies/{sample['rid']}.req" if sample_req_body else None,
                body_preview=sample_req_body[:500],
            ),
            sample_response=SampleResponse(
                status=int(sample_resp.get("status") or 0),
                mime=sample_resp.get("response_mime") or "",
                body_path=sample["body_path"],
                body_size=sample_resp_size,
                body_preview=sample_resp_body[:800],
            ),
            slots=slots,
            constants=constants,
            call_count=len(c["requests"]),
            tasks_seen=[notes_dir.name],
            initiator_scripts=init_scripts,
            rule_trace=rule_trace,
        )
        candidates.append(cand)

    # Step 5 — Graph isolation
    candidates = graph_isolation_filter(candidates)

    # Step 6 — LLM finalisation (all clusters in parallel)
    if candidates:
        finalise_tasks = [finalise_cluster(c, sm) for c in candidates]
        await asyncio.gather(*finalise_tasks, return_exceptions=False)

    # Step 7 — write tool_candidates.json
    out_path = notes_dir / "tool_candidates.json"
    serialised = [asdict(c) for c in candidates]
    out_path.write_text(json.dumps(serialised, default=str, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d candidates to %s", len(candidates), out_path)
    return candidates


# ─────────────────────────────────────────────────────────────────
# Chunk 4.8 — Per-site cross-task aggregation
# ─────────────────────────────────────────────────────────────────

def _merge_slot_sources(a: SlotSource, b: SlotSource) -> SlotSource:
    """Merge two SlotSource for the same param across tasks. Keep the most
    specific kind (chained > user_intent_text > captured). Merge example sets."""
    rank = {"chained": 3, "user_intent_text": 2, "captured": 1}
    primary, secondary = (a, b) if rank[a.kind] >= rank[b.kind] else (b, a)
    merged = SlotSource(
        kind=primary.kind,
        chain_source=primary.chain_source,
        response_jmespath=primary.response_jmespath,
        html_attribute=primary.html_attribute,
        container_signature=primary.container_signature,
        list_selector_needed=primary.list_selector_needed or secondary.list_selector_needed,
    )
    seen_examples: set[str] = set()
    for ex in primary.captured_examples + secondary.captured_examples:
        if ex not in seen_examples:
            seen_examples.add(ex)
            merged.captured_examples.append(ex)
    seen_obs: set[str] = set()
    for v in primary.observed_values + secondary.observed_values:
        if v not in seen_obs:
            seen_obs.add(v)
            merged.observed_values.append(v)
    merged.required = primary.required and secondary.required
    return merged


def build_site_registry(eval_run_dir: Path, site: str, out_path: Path | None = None) -> Path:
    """Walk all completed tasks in eval_run_dir for `site`, merge their
    tool_candidates.json by cluster_key, and write the site's tools.json.

    Returns the path of the written tools.json."""
    eval_run_dir = Path(eval_run_dir)
    merged: dict[str, ToolCandidate] = {}

    for task_dir in sorted(eval_run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        md_p = task_dir / "metadata.json"
        if not md_p.exists():
            continue
        try:
            md = json.loads(md_p.read_text(encoding="utf-8")).get("data", {})
        except (json.JSONDecodeError, ValueError):
            continue
        if md.get("site") != site:
            continue
        tc_path = task_dir / "tool_candidates.json"
        if not tc_path.exists():
            continue
        try:
            raw_list = json.loads(tc_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            continue
        for raw in raw_list:
            if raw.get("verdict") != "keep":
                continue
            ck = raw.get("cluster_key")
            if not ck:
                continue
            cand = _rehydrate_candidate(raw)
            if ck in merged:
                existing = merged[ck]
                # Merge slots
                for sname, ss_dict in raw.get("slots", {}).items():
                    new_ss = _rehydrate_slot_source(ss_dict)
                    if sname in existing.slots:
                        existing.slots[sname] = _merge_slot_sources(existing.slots[sname], new_ss)
                    else:
                        existing.slots[sname] = new_ss
                # Merge constants — keep latest
                for k, v in raw.get("constants", {}).items():
                    existing.constants.setdefault(k, v)
                existing.call_count += cand.call_count
                existing.tasks_seen.extend(cand.tasks_seen)
                # Pick longest capability_statement
                if len(cand.capability_statement) > len(existing.capability_statement):
                    existing.capability_statement = cand.capability_statement
                    existing.tool_id = cand.tool_id
                    existing.slot_descriptions = cand.slot_descriptions
                # Union initiator_scripts
                existing_sids = {s.script_id for s in existing.initiator_scripts}
                for sref in cand.initiator_scripts:
                    if sref.script_id not in existing_sids:
                        existing.initiator_scripts.append(sref)
                        existing_sids.add(sref.script_id)
            else:
                merged[ck] = cand

    out = out_path or (Path("morphnet_v2/sites") / site / "tools.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in merged.values()]
    out.write_text(json.dumps(payload, default=str, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d tools to %s", len(payload), out)
    return out


def _rehydrate_slot_source(d: dict) -> SlotSource:
    return SlotSource(
        kind=d.get("kind", "captured"),
        chain_source=d.get("chain_source"),
        response_jmespath=d.get("response_jmespath"),
        html_attribute=d.get("html_attribute"),
        container_signature=d.get("container_signature"),
        list_selector_needed=bool(d.get("list_selector_needed")),
        captured_examples=list(d.get("captured_examples") or []),
        observed_values=list(d.get("observed_values") or []),
        required=bool(d.get("required", True)),
    )


def _rehydrate_candidate(d: dict) -> ToolCandidate:
    sr = d.get("sample_request") or {}
    rr = d.get("sample_response") or {}
    return ToolCandidate(
        cluster_key=d.get("cluster_key", ""),
        dispatch_kind=d.get("dispatch_kind", "rest"),
        sample_request=SampleRequest(
            method=sr.get("method", ""), url=sr.get("url", ""),
            headers=sr.get("headers", {}) or {},
            body_path=sr.get("body_path"),
            body_preview=sr.get("body_preview", "") or "",
        ),
        sample_response=SampleResponse(
            status=int(rr.get("status", 0) or 0),
            mime=rr.get("mime", "") or "",
            body_path=rr.get("body_path", "") or "",
            body_size=int(rr.get("body_size", 0) or 0),
            body_preview=rr.get("body_preview", "") or "",
        ),
        slots={n: _rehydrate_slot_source(s) for n, s in (d.get("slots") or {}).items()},
        constants=d.get("constants") or {},
        call_count=int(d.get("call_count", 0) or 0),
        tasks_seen=list(d.get("tasks_seen") or []),
        initiator_scripts=[
            ScriptRef(s.get("script_id", ""), s.get("url"), s.get("sha256"))
            for s in (d.get("initiator_scripts") or [])
        ],
        rule_trace=list(d.get("rule_trace") or []),
        tool_id=d.get("tool_id", "") or "",
        capability_statement=d.get("capability_statement", "") or "",
        slot_descriptions=d.get("slot_descriptions") or {},
        verdict=d.get("verdict", "pending"),
        verdict_reason=d.get("verdict_reason", "") or "",
    )


# ─────────────────────────────────────────────────────────────────
# Offline HTML extraction helper (Chunk 4.1)
# ─────────────────────────────────────────────────────────────────

async def extract_response_html(
    html: str,
    base_url: str,
    sm: "SessionManager",
) -> "PageFilterOutput":
    from morphnet_v2.page_filter import PageFilter, PageSnapshot
    # Use wait_until="commit" + short timeout — we only need the DOM parsed,
    # not network resources loaded (none would resolve from set_content anyway).
    # Captured HTML often contains async inline scripts that never settle if
    # we wait for domcontentloaded.
    try:
        await sm.page.set_content(html, wait_until="commit", timeout=5000)
    except Exception:
        # Even a commit timeout shouldn't block us — the DOM is still parsed
        pass
    pf = PageFilter(sm)
    snapshot = PageSnapshot(url=base_url, title="", html=html)
    return await pf.run(snapshot, enumerate_mode=True)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

async def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m morphnet_v2.tool_builder")
    p.add_argument("notes_dir", help="Path to a task's notes directory")
    p.add_argument("--site", default=None, help="If set, after build also aggregate to morphnet_v2/sites/{site}/tools.json")
    p.add_argument("--start-url", default="https://example.com", help="URL for the offline-extraction SessionManager")
    args = p.parse_args()

    notes_dir = Path(args.notes_dir).resolve()
    if not notes_dir.is_dir():
        print(f"not a directory: {notes_dir}", file=sys.stderr)
        sys.exit(2)

    # Boot a SessionManager (needed for offline page_filter + Gemini calls)
    from morphnet_v2.session_manager import SessionManager
    async with SessionManager(start_url=args.start_url, headless=True) as sm:
        await sm.wait_for_page_ready()
        candidates = await build_candidates(notes_dir, sm)
        kept = [c for c in candidates if c.verdict == "keep"]
        print(f"\n=== tool_builder finished ===")
        print(f"  total candidates: {len(candidates)}")
        print(f"  kept (verdict=keep): {len(kept)}")
        for c in kept:
            print(f"    [{c.tool_id}] {c.capability_statement}")

    if args.site and kept:
        # walk parent dir to find sibling tasks of this site
        out = build_site_registry(notes_dir.parent, args.site)
        print(f"  aggregated to: {out}")


if __name__ == "__main__":
    asyncio.run(_cli())


__all__ = [
    "SlotKind",
    "DispatchKind",
    "ScriptRef",
    "SlotSource",
    "SampleRequest",
    "SampleResponse",
    "ToolCandidate",
    "IndexEntry",
    "build_candidates",
    "build_site_registry",
    "extract_response_html",
    "classify_script_url",
    "is_junk_response",
    "is_noise_host",
    "is_noise_path",
    "is_static",
    "cluster_identity",
    "normalise_value",
    "looks_low_entropy",
    "looks_like_nonce",
    "tokens",
    "token_overlap",
    "detect_chain",
    "graph_isolation_filter",
    "finalise_cluster",
]
