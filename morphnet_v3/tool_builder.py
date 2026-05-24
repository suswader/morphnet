"""morphnet_v3/tool_builder.py — discovery → tool DAG → tools.json.

Consumes a `TemporalRepresentation` from `temporal.py` and emits a tool DAG
with chained slots, user-intent classifications, click recipes, and LLM-authored
names + capability statements. Implements the 8-point thesis end-to-end on the
build side; the executor + verifier live in `tool_executor.py`.

File organization (single file, section banners below):

  §1  Classifiers           — payload_type / page_class / is_noise
  §2  Dataclasses           — StructuredHttpNode + ParamLeaf + SourceBucket
                              + ResponseSummary + ChainEdge + IntentBinding
                              + Tool + ToolDAG
  §3  to_structured()       — per-payload-type normalizer
  §4  Phase-1 fingerprinter — within-fingerprint variance + URL-leaf decomp
  §5  Tracer                — 12-step priority (trivial→generated→typed→
                              click→chained→cookie→state→bundle-grep→
                              bucket_4→unknown)
  §6  Chainer               — value-flow + token_overlap + list_select edges
                              + script-edge attachment
  §7  User-intent + clicks  — typed-text match + click_recipe construction
                              + graph isolation
  §8  Naming + writer       — LLM tool_finaliser + tools.json emitter
  §8b Verifier              — live attached-mode replay against open browser

The cross-cutting invariant: **no values are ever stored in tools.json.**
Every leaf has a recipe; tools.json holds structure + descriptions + history.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlsplit

import jinja2
from pydantic import BaseModel

if TYPE_CHECKING:
    from morphnet_v3.session_manager import SessionManager  # type: ignore[import-not-found]
    from morphnet_v3.temporal import HttpNode  # type: ignore[import-not-found]


# ═════════════════════════════════════════════════════════════════════════
# §1 — Classifiers
# ═════════════════════════════════════════════════════════════════════════
#
# Three deterministic classifiers run on every captured HTTP node before it
# is considered for tool minting:
#
#   1. classify_payload  → discovers the structure (graphql / rest_json / ...)
#   2. classify_page     → discovers the page-class for HTML document responses
#   3. is_noise          → drops analytics / WAF / static-asset endpoints
#
# Combined, these reject ~70-90% of captured HTTP nodes BEFORE the expensive
# tracer + chainer ever sees them.

# Lifted verbatim from morphnet_v2/tool_builder.py:117-156. The list grew
# across the 5-site audit; keep adding to it as new sites surface telemetry
# hosts the existing entries miss.
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
               ".map")

# Anti-bot challenge endpoint signatures (separate from generic telemetry).
ANTI_BOT_HOSTS = re.compile(r"^(.*\.)?(awswaf\.com|challenges\.cloudflare\.com|"
                            r"(.*\.)?captcha-delivery\.com|hcaptcha\.com|"
                            r"recaptcha\.net|(.*\.)?perimeterx\.net)$")
ANTI_BOT_PATHS = re.compile(r"/(cdn-cgi/challenge-platform|_px/|px/api/|"
                            r"mp_verify|captcha|anti-bot)/?")


PayloadType = Literal[
    "graphql", "graphql_apq", "json_rpc",
    "rest_json", "rest_form_urlencoded", "rest_multipart", "rest_xml",
    "sse", "websocket", "grpc_web",
    "static_asset", "telemetry", "anti_bot_waf", "other",
]

PageClass = Literal["spa_shell", "ssr_hybrid", "mpa", "amp", "static_html", "unknown"]


_STATIC_PATH_PREFIXES = (
    "/static/", "/assets/", "/build/", "/media/", "/bundles/", "/cdn/", "/dist/",
)
_STATIC_CONTENT_TYPES = (
    "application/javascript", "text/javascript", "application/x-javascript",
    "text/css",
)
_STATIC_CONTENT_TYPE_PREFIXES = ("image/", "font/", "video/", "audio/")


def classify_payload(
    req_headers: dict, resp_headers: dict, url: str, body: str | None,
) -> PayloadType:
    """Per JS-learnings.md table — deterministic, no LLM.

    Detection order matters: static-asset and anti-bot are cheap checks that
    short-circuit; specific JSON shapes (graphql/json_rpc) before generic
    rest_json; multipart before form_urlencoded (both use POST with body)."""
    p = urlsplit(url)
    path_lower = (p.path or "").lower()
    host_lower = (p.netloc or "").lower()
    scheme_lower = (p.scheme or "").lower()
    # 5.2 — data: URI scheme. Drops base64-inlined images that pollute
    # ParamLeaves with hundreds of bogus path segments.
    if scheme_lower == "data":
        return "static_asset"
    if path_lower.endswith(STATIC_EXTS):
        return "static_asset"
    # 5.3 — path-prefix match. Catches Magento /static/version*/...html
    # templates and locale .json files that extension-only filtering misses.
    if any(path_lower.startswith(prefix) for prefix in _STATIC_PATH_PREFIXES):
        return "static_asset"
    if ANTI_BOT_HOSTS.match(host_lower) or ANTI_BOT_PATHS.search(path_lower):
        return "anti_bot_waf"
    if NOISE_HOSTS.match(host_lower) or NOISE_PATHS.search(path_lower):
        return "telemetry"
    ctype = ""
    cache_control_lower = ""
    for k, v in (resp_headers or {}).items():
        kl = k.lower()
        if kl == "content-type":
            ctype = (v or "").lower().split(";", 1)[0].strip()
        elif kl == "cache-control":
            cache_control_lower = (v or "").lower()
    # 5.5 — Content-Type-based asset signal (catches JS/CSS/image/font/etc.
    # responses whose URL didn't have a static extension or prefix). Script
    # bytes are still captured via the existing Debugger.scriptParsed path.
    if ctype in _STATIC_CONTENT_TYPES:
        return "static_asset"
    if ctype.startswith(_STATIC_CONTENT_TYPE_PREFIXES):
        return "static_asset"
    # 5.6 — Cache-Control max-age >= 86400 (1 day) AND not no-cache/no-store
    # indicates immutable asset. The sanity check showed 476/487 responses
    # on the live capture had max-age=31536000.
    if cache_control_lower and "no-cache" not in cache_control_lower and "no-store" not in cache_control_lower:
        m = re.search(r"max-age\s*=\s*(\d+)", cache_control_lower)
        if m and int(m.group(1)) >= 86400:
            return "static_asset"
    # SSE / WebSocket / gRPC are content-type-driven.
    if ctype == "text/event-stream":
        return "sse"
    if ctype.startswith("application/grpc-web"):
        return "grpc_web"
    upg = ""
    for k, v in (req_headers or {}).items():
        if k.lower() == "upgrade":
            upg = (v or "").lower()
            break
    if upg == "websocket":
        return "websocket"
    # GraphQL signatures
    if "graphql" in path_lower:
        qs = (p.query or "").lower()
        # APQ encodes variables+extensions in URL query and the body is
        # effectively empty. Critically, Apollo sometimes serializes the body
        # as the literal string "null" (not Python None / empty string) — that
        # still counts as "no body" for APQ purposes.
        body_stripped = (body or "").strip().lower()
        body_effectively_empty = body_stripped in ("", "null", "{}")
        if body_effectively_empty and (
            "variables=" in qs or "extensions=" in qs or "operationname=" in qs
        ):
            return "graphql_apq"
        if body and isinstance(body, str):
            bl = body.lstrip()
            if bl.startswith("{") and ("\"query\"" in bl or "\"operationName\"" in bl):
                return "graphql"
    # JSON shapes — JSON-RPC vs plain REST-JSON
    if body and isinstance(body, str):
        bl = body.lstrip()
        if bl.startswith("{") and "\"jsonrpc\"" in bl:
            return "json_rpc"
    # Content-type-driven REST classes
    if "multipart" in ctype:
        return "rest_multipart"
    if "x-www-form-urlencoded" in ctype:
        return "rest_form_urlencoded"
    if "xml" in ctype:
        return "rest_xml"
    if "json" in ctype:
        return "rest_json"
    # Some servers return text/html for JSON or skip content-type. If the body
    # looks like JSON, call it rest_json.
    if body and isinstance(body, str):
        bl = body.lstrip()
        if bl.startswith("{") or bl.startswith("["):
            return "rest_json"
    return "other"


def classify_page(html_body: str | None, url: str) -> PageClass:
    """Per JS-learnings.md page_class table. Cheap heuristics on the doc body."""
    if not html_body:
        return "unknown"
    body = html_body[:8000]  # heuristics only need the head — not data, no truncation rule applies
    low = body.lower()
    if "<html" in low and ("⚡" in body or "<html amp" in low or " amp " in low[:200]):
        return "amp"
    if "__next_data__" in low or "__nuxt__" in low or "__remix_context__" in low or "__sveltekit_data__" in low:
        return "ssr_hybrid"
    if len(html_body) < 15_000 and 'id="root"' in low and "<script" in low:
        return "spa_shell"
    if len(html_body) > 30_000 and low.count("<script") < 6:
        return "mpa"
    if "<script" not in low:
        return "static_html"
    return "unknown"


def is_noise(url: str, payload_type: PayloadType) -> bool:
    """Combined noise gate. Drops static assets, telemetry, anti-bot, plus
    payload_types that are never useful for tool minting (sse/websocket are
    valid but excluded from the v1 tool builder — separate replay path)."""
    if payload_type in ("static_asset", "telemetry", "anti_bot_waf"):
        return True
    p = urlsplit(url)
    host_lower = (p.netloc or "").lower()
    path_lower = (p.path or "").lower()
    return bool(NOISE_HOSTS.match(host_lower) or NOISE_PATHS.search(path_lower))


# ═════════════════════════════════════════════════════════════════════════
# §2 — Dataclasses (the universal API contract)
# ═════════════════════════════════════════════════════════════════════════
#
# Every downstream stage (tracer, chainer, user-intent linker, naming LLM,
# writer, executor) reads these shapes. Treat them as the schema; getting
# them right once means everything else stays simple.
#
# Key choice: `ParamLeaf.parent_path` is a list of explicit (op, key) tuples
# rather than a jsonpath string. The executor walks this directly to mutate
# the request template at replay — no parsing, no surprises.

Location = Literal["url_query", "url_path_segment", "header", "body"]
ParentKind = Literal[
    "scalar", "list_elem", "dict_value",
    "url_segment",  # for URL-leaf decomposition (Option D) sub-leaves
    "header_value",
]
PathOpKind = Literal[
    "key",            # dict key access — ("key", "variables")
    "index",          # list index — ("index", 0)
    "header_name",    # HTTP header name — ("header_name", "authorization")
    "url_path",       # extract Nth URL path segment from a URL-valued parent
    "url_query",      # extract named query param from a URL-valued parent
]
SourceBucketKind = Literal[
    "slot",             # placeholder for un-classified slots before tracer runs
    "user_intent",      # typed text or click target — IntentBinding carries the detail
    "chained_resp",     # value flows from a prior tool's response — ChainEdge carries the detail
    "cookie",           # value lives in the live cookie jar at request time
    "session_state",    # value lives in localStorage / sessionStorage
    "bundle",           # value originates from the JS bundle (literal OR runtime output)
                        # — sub_type narrows: apq_hash / literal / enum_upper / enum_snake /
                        # enum_kebab / semver / geohash / hex_default / vendor_config_literal /
                        # ref_tag / event_name / platform_literal / pipe_composite /
                        # csv_composite / bool_literal / numeric_literal
    "generated",        # re-generated per call (traceparent / timestamp / uuid / viewport / user_agent)
    "adapter_injected", # header value set by the adapter via Network.setExtraHTTPHeaders
                        # (e.g. webarena auto-login). Replay materializer reads the adapter
                        # registry by header_name.
    "fixed_by_capture", # value identical across every observation of this cluster — empirical
                        # stability claim. Executor sends the captured value at replay; verifier
                        # catches drift. NOT a "constant" — it's an evidence-based origin.
    "trivial",          # empty value only — 1-char values flow to bundle:numeric_literal et al
    "unknown",
]


@dataclass
class SourceBucket:
    """The classified provenance of one ParamLeaf. The `recipe` field is the
    payload tools.json carries; the bucket+sub_type drive the executor's
    materializer dispatch."""

    bucket: SourceBucketKind
    sub_type: str = ""           # e.g. "traceparent", "apq_hash", "enum_upper"
    value: Any = None            # captured value — stored ONLY for bundle sub_types where the recipe IS "use this captured literal verbatim" (literal/enum_*/composite/etc.)
    recipe: dict = field(default_factory=dict)
    # For bucket=="chained_resp": recipe = {source_node_id, source_tool_id, jsonpath, extract}
    # For bucket=="cookie": recipe = {name}
    # For bucket=="session_state": recipe = {storage: "localStorage"|"sessionStorage", key}
    # For bucket=="bundle" + sub_type=="apq_hash": recipe = {operation_name, query_text_locator}
    #   → executor grep-finds the operation's query text in the live bundle + hashes it
    # For bucket=="bundle" + sub_type=="literal"/"enum_*"/"composite"/"semver"/etc.:
    #   recipe is empty; the materializer uses SourceBucket.value verbatim
    # For bucket=="generated": recipe = {} (handler is sub_type-keyed:
    #   traceparent / timestamp_ms / timestamp_s / uuid / viewport / viewport_wxh /
    #   user_agent / user_agent_data / network_type / fingerprint_hash)


@dataclass
class ParamLeaf:
    """One scalar leaf addressable for chain detection / replay-time mutation.

    `parent_path` is a list of explicit ops the executor walks to set the
    value at replay. Example: a body leaf `$.variables.slug` is
    `[("key","variables"), ("key","slug")]`. A URL path-segment leaf is
    `[("index", 2)]`. A header is `[("header_name", "authorization")]`."""

    location: Location
    key: str                          # human-readable key (last segment of parent_path)
    raw_value: str
    parent_path: list[tuple[PathOpKind, Any]]
    parent_kind: ParentKind
    source: SourceBucket | None = None    # populated by the tracer (§5)


@dataclass
class ResponseSummary:
    """The chain-detection surface on the RESPONSE side. Carries:
      - `slots`: flat list of ParamLeaf over the parsed response body
                  (post URL-leaf decomposition from §4).
      - `repeating_jsonpath`: the deepest array whose items share a key set
                              (e.g. `$.data.contentPage.products`). Used by
                              the chainer to detect list_select edges.
      - `item_key_fields`: the fields the LLM list_selector renders per item
                              (e.g. `["id","name","slug"]`)."""

    slots: list[ParamLeaf]
    repeating_jsonpath: str | None = None
    item_key_fields: list[str] = field(default_factory=list)


@dataclass
class StructuredHttpNode:
    """The normalized HTTP unit. Carries the raw HttpNode handle so the
    executor can re-issue if needed, plus classifier outputs and the slot
    surfaces every downstream stage reads."""

    node_id: str                      # we use HttpNode.request_id
    ts_ms: int
    host: str
    path: str
    method: str
    status: int
    request_mime: str | None
    response_mime: str | None
    payload_type: PayloadType
    page_class: PageClass | None      # set only for HTML doc responses
    is_noise: bool
    slots: list[ParamLeaf]            # request-side leaves
    response_summary: ResponseSummary
    raw_node: Any                     # the HttpNode it was derived from (for executor re-issue)


# ── Chain + tool shapes ──

ChainEdgeKind = Literal["direct", "list_select"]
# Note: http precursors are IMPLICIT — derived at write time from
# `set(slot.chain_edge.source_tool_id for slot in tool.slots.values() if slot.chain_edge)`.
# js_render is moved off ChainEdge onto `Tool.requires_js_render: bool` since
# every tool's js_render precursor has the same shape (navigate to entry_url + hydrate).


@dataclass
class ChainEdge:
    """An edge from a source tool's response to a downstream tool's input slot.
    Always slot-level — tool-level precursors are derived, not stored."""

    kind: ChainEdgeKind
    source_tool_id: str               # the upstream tool whose response carries the value
    list_jsonpath: str | None = None  # the array path on the source response (None for direct)
    per_item_extract: dict = field(default_factory=dict)   # {field, regex|path_segment_at}
    selector_recipe: dict = field(default_factory=dict)    # {kind, candidate_fields, trivial_if_n_eq_1}
    confidence: float = 0.0           # token_overlap-derived confidence (0..1)


IntentBindingKind = Literal["user_intent:typed", "user_intent:click", "user_intent:selection"]


@dataclass
class IntentBinding:
    """A slot bound to a CU action (typed text or click). Replaces a
    ChainEdge for user-driven values."""

    kind: IntentBindingKind
    source_action_ts: int             # the CuActionEvent.ts_ms that originated this binding
    click_recipe: dict = field(default_factory=dict)       # {selector, attribute, extract_regex|extract}
    examples: list[str] = field(default_factory=list)


@dataclass
class Slot:
    """One parameter of a Tool. Either chained (ChainEdge), user-intent
    (IntentBinding), or sourced from cookie/state/bucket_4/generated."""

    name: str
    type: str = "string"
    required: bool = True
    location: Location = "body"
    parent_path: list[tuple[PathOpKind, Any]] = field(default_factory=list)
    source: SourceBucket | None = None
    chain_edge: ChainEdge | None = None
    intent_binding: IntentBinding | None = None
    description: str = ""             # filled by LLM in §8
    examples: list[str] = field(default_factory=list)


@dataclass
class Tool:
    """One tool in the DAG. Maps 1:1 to an emitted tools.json entry."""

    tool_id: str                      # e.g. "lego_open_product" (filled by LLM in §8)
    cluster_key: str                  # (host, path_template, op_name) — internal stable key
    capability_statement: str = ""    # filled by LLM in §8
    payload_type: PayloadType = "other"
    page_class: PageClass | None = None
    endpoint_template: dict = field(default_factory=dict)  # method, url_template, headers_template, body_template
    slots: dict[str, Slot] = field(default_factory=dict)
    requires_js_render: bool = True   # every tool needs the entry_url hydration once per session
    # NOTE: HTTP precursors are derived at write time from slot chain edges —
    # not stored explicitly. See ChainEdgeKind comment.
    script_dependencies: list[str] = field(default_factory=list)  # sha256 list from link_causality
    history: dict = field(default_factory=lambda: {
        "total_runs": 0, "success_rate": 0.0, "lifecycle": "pending",
    })
    verdict: Literal["keep", "discard", "pending"] = "pending"    # LLM verdict in §8
    # Fraction of slot leaves with non-unknown origin (0.0..1.0). Set by the
    # isolation filter. FORENSICS-ONLY: not surfaced to the planner prompt
    # (same treatment as history.lifecycle). Used to correlate with verifier
    # outcomes in tools_verification.md to diagnose whether chain LOGIC is
    # broken (high-confidence FAIL) vs chain COVERAGE is partial (many
    # low-confidence tools). Not a runtime gate.
    classification_confidence: float = 0.0


@dataclass
class ToolDAG:
    """The output of §6/§7 — what §8 writes to tools.json."""

    site: str
    page_class: PageClass | None      # the site's primary entry page_class
    entry_url: str                    # the JS-render precursor target
    tools: list[Tool] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════
# §3 — to_structured() per-payload-type normalizer
# ═════════════════════════════════════════════════════════════════════════
#
# Takes a raw HttpNode and emits a StructuredHttpNode with a ParamLeaf for
# every scalar address-able in the request (URL tokens + query params +
# selected headers + body fields) and in the response body.
#
# Per-payload-type body extractors handle:
#   rest_json            → walk parsed_request_body
#   rest_form_urlencoded → iterate list[(k,v)]
#   rest_multipart       → iterate parts
#   graphql              → walk body.variables (the slot-bearing part)
#   graphql_apq          → walk url_query.variables + url_query.extensions
#   json_rpc             → walk body.params
#   rest_xml / other     → no body leaves (v1 scope)
#
# Sub-leaves for URL-shaped response leaves (Option D) are added in §4.

# Headers we never want as slot candidates — browser/transport-controlled,
# regenerated per request, never slot-driven.
_NOISE_HEADER_NAMES = frozenset({
    "accept", "accept-encoding", "accept-language", "accept-charset",
    "cache-control", "connection", "host", "content-length", "content-type",
    "cookie", "dnt", "origin", "pragma", "referer",
    "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform", "sec-ch-ua-arch",
    "sec-ch-ua-bitness", "sec-ch-ua-full-version", "sec-ch-ua-full-version-list",
    "sec-ch-ua-model", "sec-ch-ua-platform-version",
    "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "sec-fetch-user",
    "upgrade-insecure-requests",
})


def _walk_json_leaves(
    obj: Any, location: Location, parent_path: list[tuple[PathOpKind, Any]],
) -> list[ParamLeaf]:
    """Depth-first walk of a parsed JSON value (dict / list / scalar). Emits a
    ParamLeaf per scalar with `parent_path` carrying the navigation breadcrumbs
    the executor uses to re-set the value at replay. Null values are skipped
    (no value to slot)."""
    out: list[ParamLeaf] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = parent_path + [("key", k)]
            if isinstance(v, (dict, list)):
                out.extend(_walk_json_leaves(v, location, new_path))
            elif v is None:
                continue
            else:
                out.append(ParamLeaf(
                    location=location, key=k, raw_value=str(v),
                    parent_path=new_path, parent_kind="dict_value",
                ))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = parent_path + [("index", i)]
            if isinstance(item, (dict, list)):
                out.extend(_walk_json_leaves(item, location, new_path))
            elif item is None:
                continue
            else:
                out.append(ParamLeaf(
                    location=location, key=f"[{i}]", raw_value=str(item),
                    parent_path=new_path, parent_kind="list_elem",
                ))
    elif obj is None:
        return out
    else:
        # Top-level scalar (rare — most bodies are dict/list). Address via empty
        # parent_path; caller decides key.
        out.append(ParamLeaf(
            location=location, key="$", raw_value=str(obj),
            parent_path=parent_path, parent_kind="scalar",
        ))
    return out


def _extract_url_leaves(url: str) -> list[ParamLeaf]:
    """URL path segments + query params as ParamLeaf."""
    out: list[ParamLeaf] = []
    p = urlsplit(url)
    segments = [s for s in (p.path or "/").split("/") if s]
    for i, seg in enumerate(segments):
        out.append(ParamLeaf(
            location="url_path_segment",
            key=f"path[{i}]",
            raw_value=seg,
            parent_path=[("index", i)],
            parent_kind="scalar",
        ))
    if p.query:
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            out.append(ParamLeaf(
                location="url_query",
                key=k,
                raw_value=v,
                parent_path=[("key", k)],
                parent_kind="dict_value",
            ))
    return out


def _extract_apq_url_extras(url: str) -> list[ParamLeaf]:
    """For graphql_apq: url query carries `variables=<json>` and
    `extensions=<json>`. Parse those JSON blobs and emit leaves over their
    inner structure (each variable becomes its own slot candidate)."""
    out: list[ParamLeaf] = []
    p = urlsplit(url)
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        if k not in ("variables", "extensions"):
            continue
        try:
            obj = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            continue
        out.extend(_walk_json_leaves(obj, "url_query", [("key", k)]))
    return out


def _extract_header_leaves(headers: dict[str, str]) -> list[ParamLeaf]:
    """Headers that may carry slot values — drops the noise list (browser/
    transport-controlled headers regenerated per request)."""
    out: list[ParamLeaf] = []
    for k, v in (headers or {}).items():
        if not v:
            continue
        if k.lower() in _NOISE_HEADER_NAMES:
            continue
        out.append(ParamLeaf(
            location="header",
            key=k,
            raw_value=str(v),
            parent_path=[("header_name", k)],
            parent_kind="header_value",
        ))
    return out


def _extract_body_leaves(
    node: HttpNode, payload_type: PayloadType,
) -> list[ParamLeaf]:
    """Per-payload-type body walker. Returns [] for payload types whose body
    isn't slot-bearing (graphql_apq has body=null; static_asset / sse / etc.
    are filtered upstream)."""
    body = node.parsed_request_body
    if body is None:
        return []
    if payload_type in ("graphql", "graphql_apq"):
        # Only `variables` carries slots; `query` is the query text (literal).
        # graphql_apq body is usually null anyway — variables move to URL.
        if isinstance(body, dict):
            variables = body.get("variables")
            if isinstance(variables, (dict, list)):
                return _walk_json_leaves(variables, "body", [("key", "variables")])
        return []
    if payload_type == "json_rpc":
        if isinstance(body, dict):
            params = body.get("params")
            if isinstance(params, (dict, list)):
                return _walk_json_leaves(params, "body", [("key", "params")])
        return []
    if payload_type == "rest_json":
        return _walk_json_leaves(body, "body", [])
    if payload_type == "rest_form_urlencoded":
        out: list[ParamLeaf] = []
        if isinstance(body, list):
            for k, v in body:
                out.append(ParamLeaf(
                    location="body", key=k, raw_value=v,
                    parent_path=[("key", k)], parent_kind="dict_value",
                ))
        return out
    if payload_type == "rest_multipart":
        out: list[ParamLeaf] = []
        if isinstance(body, list):
            for i, part in enumerate(body):
                if not isinstance(part, dict):
                    continue
                content = part.get("content")
                if isinstance(content, str):
                    out.append(ParamLeaf(
                        location="body", key=f"part[{i}]", raw_value=content,
                        parent_path=[("index", i), ("key", "content")],
                        parent_kind="list_elem",
                    ))
        return out
    return []


def _extract_response_leaves(node: HttpNode) -> list[ParamLeaf]:
    """Walk parsed_response_body → ParamLeaf list (for ResponseSummary.slots)."""
    body = node.parsed_response_body
    if body is None:
        return []
    return _walk_json_leaves(body, "body", [])


def _detect_repeating(obj: Any) -> tuple[str | None, list[str]]:
    """Find the DEEPEST list-of-dicts in `obj` whose items share a non-empty
    key set. Returns (`jsonpath_to_list`, `sorted(common_keys)`) — these power
    the chainer's list_select edge detection. If no such list exists, returns
    (None, [])."""
    best_path: str | None = None
    best_fields: list[str] = []
    best_depth = -1

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal best_path, best_fields, best_depth
        if isinstance(node, list):
            if len(node) >= 2 and all(isinstance(item, dict) for item in node):
                key_sets = [set(item.keys()) for item in node]
                common = set(key_sets[0])
                for ks in key_sets[1:]:
                    common &= ks
                if common and depth > best_depth:
                    best_depth = depth
                    best_path = path
                    best_fields = sorted(common)
            for item in node:
                walk(item, f"{path}[*]", depth + 1)
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}", depth + 1)

    walk(obj, "$", 0)
    return best_path, best_fields


def to_structured(node: HttpNode) -> StructuredHttpNode:
    """Normalize one HttpNode → StructuredHttpNode. The entry point for §4-§7."""
    payload_type = classify_payload(
        node.request_headers, node.response_headers, node.url, node.request_body,
    )
    # 5.7 — empty response body (e.g. 204 No Content, tracking beacons). Skip
    # classifying these as candidate tools. graphql/json_rpc/sse/etc. keep
    # their classification because some valid endpoints legitimately return
    # empty bodies (mutations, fire-and-forget calls); we only reclassify
    # generic `other` / `rest_json` whose body is empty.
    if payload_type in ("other", "rest_json") and not (node.response_body or node.parsed_response_body):
        payload_type = "static_asset"
    page_class: PageClass | None = None
    if node.response_mime and "html" in node.response_mime.lower():
        page_class = classify_page(node.response_body, node.url)
    noise = is_noise(node.url, payload_type)

    # Request-side leaves
    slots = list(_extract_url_leaves(node.url))
    if payload_type == "graphql_apq":
        slots.extend(_extract_apq_url_extras(node.url))
    slots.extend(_extract_header_leaves(node.request_headers))
    slots.extend(_extract_body_leaves(node, payload_type))

    # Response-side
    resp_leaves = _extract_response_leaves(node)
    repeating_jsonpath, item_key_fields = _detect_repeating(
        node.parsed_response_body if node.parsed_response_body is not None else {},
    )
    response_summary = ResponseSummary(
        slots=resp_leaves,
        repeating_jsonpath=repeating_jsonpath,
        item_key_fields=item_key_fields,
    )

    p = urlsplit(node.url)
    return StructuredHttpNode(
        node_id=node.request_id,
        ts_ms=node.request_ts_ms,
        host=p.netloc,
        path=p.path,
        method=node.method,
        status=node.status,
        request_mime=node.request_mime,
        response_mime=node.response_mime,
        payload_type=payload_type,
        page_class=page_class,
        is_noise=noise,
        slots=slots,
        response_summary=response_summary,
        raw_node=node,
    )


# ═════════════════════════════════════════════════════════════════════════
# §4 — URL-leaf decomposition (Option D)
# ═════════════════════════════════════════════════════════════════════════
#
# After §3 emits the base leaves, any response-side leaf whose value is
# URL-shaped (starts with "http"/"/"/"//") gets EXPLODED into structural
# sub-leaves: one per URL path segment + one per URL query param.
#
# This is what makes the chainer detect "the slug came from a URL inside the
# response" via whole-value match. Without this, a downstream slot like
# `mclaren-p1-42172` wouldn't match anything in the prior response's leaves —
# because the value lives INSIDE a URL leaf like `/product/mclaren-p1-42172`.

_URL_SHAPED = re.compile(r"^(https?:)?//|^/[^/]")


def _is_url_shaped(value: str) -> bool:
    """Conservative URL-shape detector. Accepts http(s)://, //, and root-path /x.
    Rejects plain strings like 'product' or 'mclaren-p1-42172'."""
    if not value or "/" not in value:
        return False
    return bool(_URL_SHAPED.match(value))


def _decompose_url_leaf(leaf: ParamLeaf) -> list[ParamLeaf]:
    """Explode one URL-shaped response leaf into sub-leaves. Returns the
    sub-leaves to ADD to response_summary.slots (the parent leaf stays)."""
    if not _is_url_shaped(leaf.raw_value):
        return []
    try:
        parsed = urlsplit(leaf.raw_value)
    except ValueError:
        return []
    out: list[ParamLeaf] = []
    segments = [s for s in (parsed.path or "/").split("/") if s]
    for i, seg in enumerate(segments):
        out.append(ParamLeaf(
            location=leaf.location,
            key=f"{leaf.key}#path[{i}]",
            raw_value=seg,
            parent_path=leaf.parent_path + [("url_path", i)],
            parent_kind="url_segment",
        ))
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            out.append(ParamLeaf(
                location=leaf.location,
                key=f"{leaf.key}#query.{k}",
                raw_value=v,
                parent_path=leaf.parent_path + [("url_query", k)],
                parent_kind="url_segment",
            ))
    return out


def decompose_url_leaves(snode: StructuredHttpNode) -> None:
    """Apply URL-leaf decomposition to the response_summary.slots (in-place)."""
    extras: list[ParamLeaf] = []
    for leaf in snode.response_summary.slots:
        extras.extend(_decompose_url_leaf(leaf))
    snode.response_summary.slots.extend(extras)


# ═════════════════════════════════════════════════════════════════════════
# §5 — Tracer (12-step priority pipeline)
# ═════════════════════════════════════════════════════════════════════════
#
# For each ParamLeaf in a request, walk the priority list and return the
# FIRST matching SourceBucket. Order matters: cheap-and-unambiguous first
# (regex patterns), then context-aware (typed/click), then expensive
# (bundle grep), then last-resort shape recognizers.

# Pattern constants lifted from experiments/click_cu_extraction_study.py.
_PAT_TRACEPARENT = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")
# UUID matching deferred — cookie/state/chained branches catch most UUIDs
# (session/device IDs live there). Add a regex step when we see a truly
# generated-fresh UUID falling to unknown.
# 13-digit numbers = epoch milliseconds (covers 2001-09 through 2286-11).
# 10-digit = epoch seconds (covers 2001 through 2286). Gating by key-name
# below for the seconds case dodges false matches against generic numeric IDs.
_PAT_TIMESTAMP_MS = re.compile(r"^\d{13}$")
_PAT_TIMESTAMP_S = re.compile(r"^\d{10}$")
_PAT_JWT = re.compile(r"^eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+$")
_PAT_APQ_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAT_USER_AGENT = re.compile(r"^Mozilla/\d")
_PAT_SCREEN_RES = re.compile(r"^\d{3,4}x\d{3,4}$")
_PAT_SEMVER = re.compile(r"^\d+\.\d+\.\d+([._\-+]\w+)?$")
_PAT_VENDOR_PRODUCT = re.compile(r"^([a-z]+\.)+[a-z]+/\d+\.\d+\.\d+([._\-+]\w+)?$")
_PAT_ENUM_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PAT_ENUM_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
_PAT_ENUM_KEBAB = re.compile(r"^[a-z][a-z0-9-]*$")
_PAT_HEX_COLOR = re.compile(r"^[0-9a-f]{6}$|^[0-9a-f]{8}$")

_BOOL_LITERALS = frozenset({"true", "false"})
_NETWORK_LITERALS = frozenset({"2g", "3g", "4g", "5g", "wifi", "ethernet", "none", "slow-2g", "unknown"})
_OS_LITERALS = frozenset({"macOS", "iOS", "Android", "Windows", "Linux", "ChromeOS"})
_BROWSER_LITERALS = frozenset({"Chrome", "Firefox", "Safari", "Edge", "Opera", "Chromium"})


@dataclass
class TraceContext:
    """Pre-computed indexes the tracer consults. Built once per task, used
    for every leaf classification call. Heavy lift here = fast trace_param()."""

    temporal: Any                                                    # TemporalRepresentation (lazy import)
    leaf_index: dict[str, list[tuple[str, ParamLeaf]]] = field(default_factory=dict)
    # value → [(source_node_id, the_leaf_it_came_from), ...]
    node_response_ts: dict[str, int] = field(default_factory=dict)
    # source_node_id → response_ts_ms (for "prior response" gating)
    # Adapter-injected headers — keyed by lowercase header NAME mapping to the
    # exact value the adapter sent via CDP setExtraHTTPHeaders. Populated from
    # the SessionManager's `injected_headers` registry (e.g. webarena's
    # X-M2-Admin-Auto-Login = admin:admin1234). The tracer matches header
    # leaves against this set and classifies as `adapter_injected` origin.
    adapter_injected: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls, temporal: Any, structured_nodes: list[StructuredHttpNode],
        adapter_injected: dict[str, str] | None = None,
    ) -> TraceContext:
        ctx = cls(temporal=temporal)
        # Normalise header names to lowercase for case-insensitive matching;
        # HTTP header names are case-insensitive per RFC 7230.
        if adapter_injected:
            ctx.adapter_injected = {k.lower(): v for k, v in adapter_injected.items()}
        for snode in structured_nodes:
            if snode.is_noise:
                continue
            resp_ts = snode.raw_node.response_ts_ms or snode.ts_ms
            ctx.node_response_ts[snode.node_id] = resp_ts
            for leaf in snode.response_summary.slots:
                v = leaf.raw_value
                if not v:
                    continue
                ctx.leaf_index.setdefault(v, []).append((snode.node_id, leaf))
        return ctx


def _is_apq_hash_leaf(leaf: ParamLeaf) -> bool:
    """A 64-hex value is APQ-hash ONLY if its leaf key path ends in
    persistedQuery.sha256Hash (else it's just a generic SHA256)."""
    if not _PAT_APQ_SHA256.match(leaf.raw_value):
        return False
    key_lower = leaf.key.lower()
    path_str = ".".join(str(op[1]) for op in leaf.parent_path).lower()
    return "sha256hash" in key_lower or "sha256hash" in path_str


def _click_target_value(action: Any, value: str) -> str | None:
    """Search a click CuActionEvent.target_attrs for `value`. Returns the
    matched attribute name (e.g., 'href', 'data-id', 'text') so the click
    recipe can be anchored on that field, or None if no match."""
    attrs = action.target_attrs
    if not isinstance(attrs, dict):
        return None
    for field_name in ("href", "value", "id"):
        v = attrs.get(field_name)
        if v and value in str(v):
            return field_name
    text = attrs.get("text")
    if text and value == str(text).strip():
        return "text"
    inner_attrs = attrs.get("attrs")
    if isinstance(inner_attrs, dict):
        for k, v in inner_attrs.items():
            if v and value in str(v):
                return k
    return None


def trace_param(
    leaf: ParamLeaf, ctx: TraceContext, request_ts_ms: int,
) -> SourceBucket:
    """The 12-step priority pipeline. Returns the FIRST matching SourceBucket."""
    v = leaf.raw_value

    # 1. Trivial (empty only — 1-char numerics flow through to bundle:numeric)
    if v == "":
        return SourceBucket(bucket="trivial", sub_type="empty")

    # 2. Generated patterns (cheapest, zero false positives)
    if _PAT_TRACEPARENT.match(v):
        return SourceBucket(bucket="generated", sub_type="traceparent")
    if _PAT_JWT.match(v):
        return SourceBucket(
            bucket="session_state", sub_type="jwt",
            recipe={"storage": "localStorage", "key_hint": "auth_token"},
        )
    # Timestamp classification: require BOTH the 13-/10-digit shape AND a
    # key-name hint. Without the key gate, a 13-digit numeric ID (snowflake-
    # adjacent, primary keys) would be misclassified as a timestamp, then
    # the executor would generate a fresh "timestamp" at replay → wrong ID
    # → server rejection.
    _ts_keys = ("time", "ts", "stamp", "epoch", "_at", "millis", "_ms")
    _kl_ts = leaf.key.lower()
    _has_ts_hint = any(k in _kl_ts for k in _ts_keys)
    if _has_ts_hint and _PAT_TIMESTAMP_MS.match(v):
        return SourceBucket(bucket="generated", sub_type="timestamp_ms")
    if _has_ts_hint and _PAT_TIMESTAMP_S.match(v):
        return SourceBucket(bucket="generated", sub_type="timestamp_s")
    if _PAT_USER_AGENT.match(v):
        return SourceBucket(bucket="generated", sub_type="user_agent")
    if _PAT_SCREEN_RES.match(v):
        return SourceBucket(bucket="generated", sub_type="viewport_wxh")
    if _is_apq_hash_leaf(leaf):
        return SourceBucket(
            bucket="bundle", sub_type="apq_hash",
            recipe={"hash_algo": "sha256", "query_text_locator": "live_bundle_grep_by_operation_name"},
        )

    # 2b. Adapter-injected headers — when the WebArena adapter (or any
    # caller) registers headers via SessionManager(injected_headers={...}),
    # match the leaf's header name + value against that registry. Catches
    # X-M2-Admin-Auto-Login, X-Postmill-Auto-Login etc. that would otherwise
    # fall to unknown. Only applies to header-location leaves.
    if leaf.location == "header" and ctx.adapter_injected:
        expected = ctx.adapter_injected.get(leaf.key.lower())
        if expected is not None and expected == v:
            return SourceBucket(
                bucket="adapter_injected",
                sub_type=leaf.key.lower(),
                recipe={"header_name": leaf.key},
            )

    # 3. Typed-text from CU actions
    for action in ctx.temporal.cu_actions(before=request_ts_ms):
        if action.kind != "type_text":
            continue
        if action.text and action.text.strip().lower() == v.strip().lower():
            return SourceBucket(
                bucket="user_intent", sub_type="typed",
                recipe={"action_ts_ms": action.ts_ms, "match": "exact"},
            )

    # 4. Click-target match
    for action in ctx.temporal.cu_actions(before=request_ts_ms):
        if action.kind != "click":
            continue
        matched_field = _click_target_value(action, v)
        if matched_field is not None:
            return SourceBucket(
                bucket="user_intent", sub_type="click",
                recipe={
                    "action_ts_ms": action.ts_ms,
                    "target_aid": action.target_aid,
                    "matched_attr": matched_field,
                },
            )

    # 5. Whole-value chained_resp (incl. URL sub-leaves from §4)
    if v in ctx.leaf_index:
        prior = [
            (nid, src_leaf) for nid, src_leaf in ctx.leaf_index[v]
            if ctx.node_response_ts.get(nid, 0) < request_ts_ms
        ]
        if prior:
            src_node_id, src_leaf = prior[0]
            return SourceBucket(
                bucket="chained_resp", sub_type="whole_value",
                recipe={
                    "source_node_id": src_node_id,
                    "source_parent_path": src_leaf.parent_path,
                    "source_parent_kind": src_leaf.parent_kind,
                    "extract": "whole",
                },
            )

    # 6. Cookie whole-value
    jar = ctx.temporal.cookies_at(request_ts_ms)
    for cname, cval in jar.items():
        if cval == v:
            return SourceBucket(bucket="cookie", sub_type=cname, recipe={"name": cname})

    # 7. Cookie substring (≥10 chars to dodge spurious tiny matches)
    if len(v) >= 10:
        for cname, cval in jar.items():
            if v in cval:
                return SourceBucket(
                    bucket="cookie", sub_type=f"{cname}:substring",
                    recipe={"name": cname, "extract": "substring"},
                )

    # 8. localStorage / sessionStorage
    state = ctx.temporal.storage_at(request_ts_ms)
    for k, sval in (state.get("local") or {}).items():
        if sval == v:
            return SourceBucket(
                bucket="session_state", sub_type=k,
                recipe={"storage": "localStorage", "key": k},
            )
    for k, sval in (state.get("session") or {}).items():
        if sval == v:
            return SourceBucket(
                bucket="session_state", sub_type=k,
                recipe={"storage": "sessionStorage", "key": k},
            )

    # 9. Bundle source grep (lazy — only for ≥6-char, non-pure-digit values)
    if len(v) >= 6 and not v.isdigit():
        for sha in list(ctx.temporal.scripts.index.keys()):
            src = ctx.temporal.scripts.bytes_for(sha)
            if src and v in src:
                return SourceBucket(
                    bucket="bundle", sub_type="literal",
                    value=v, recipe={"first_seen_in_script": sha},
                )

    # 10. Bundle sub-pattern shape recognizers (last-resort safety net)
    if v.lower() in _BOOL_LITERALS:
        return SourceBucket(bucket="bundle", sub_type="bool_literal", value=v)
    if v.isdigit() and len(v) <= 4:
        return SourceBucket(bucket="bundle", sub_type="numeric_literal", value=v)
    if v in _BROWSER_LITERALS:
        return SourceBucket(bucket="bundle", sub_type="browser_literal", value=v)
    if v in _OS_LITERALS:
        return SourceBucket(bucket="bundle", sub_type="os_literal", value=v)
    if v in _NETWORK_LITERALS:
        return SourceBucket(bucket="bundle", sub_type="network_literal", value=v)
    if _PAT_VENDOR_PRODUCT.match(v):
        return SourceBucket(bucket="bundle", sub_type="vendor_product", value=v)
    if _PAT_SEMVER.match(v):
        return SourceBucket(bucket="bundle", sub_type="semver_literal", value=v)
    if _PAT_HEX_COLOR.match(v):
        return SourceBucket(bucket="bundle", sub_type="hex_color", value=v)
    if "|" in v and len(v) <= 200:
        parts = [p.strip() for p in v.split("|")]
        if all(parts) and all(len(p) <= 50 for p in parts):
            return SourceBucket(bucket="bundle", sub_type="pipe_composite", value=v)
    if "," in v and len(v) <= 300:
        parts = [p.strip() for p in v.split(",")]
        if len(parts) >= 2 and all(parts) and not all(p.isdigit() for p in parts):
            return SourceBucket(bucket="bundle", sub_type="csv_composite", value=v)
    if _PAT_ENUM_UPPER.match(v) and len(v) >= 2:
        return SourceBucket(bucket="bundle", sub_type="enum_upper", value=v)
    if "_" in v and _PAT_ENUM_SNAKE.match(v) and 2 <= len(v) <= 60:
        return SourceBucket(bucket="bundle", sub_type="enum_snake", value=v)
    if "-" in v and _PAT_ENUM_KEBAB.match(v) and 2 <= len(v) <= 60:
        return SourceBucket(bucket="bundle", sub_type="enum_kebab", value=v)

    return SourceBucket(bucket="unknown")


def populate_sources(snode: StructuredHttpNode, ctx: TraceContext) -> None:
    """Run trace_param on every request-side leaf of `snode`, mutating each
    leaf's `source` field in place. Skips noise nodes."""
    if snode.is_noise:
        return
    for leaf in snode.slots:
        leaf.source = trace_param(leaf, ctx, snode.ts_ms)


# ═════════════════════════════════════════════════════════════════════════
# §6 — Chainer (value-flow + token_overlap + list_select edges)
# ═════════════════════════════════════════════════════════════════════════
#
# Groups structured nodes into clusters (one Tool per cluster), builds
# Slot objects from request-side ParamLeafs, attaches typed ChainEdges /
# IntentBindings derived from the tracer's SourceBuckets.

# Token splitter for confidence scoring (lifted shape from v2's `tokens()`).
_TOKEN_SPLIT = re.compile(r"[_\-\.\s]+")


def _tokens(s: str) -> set[str]:
    """camelCase + snake_case + kebab-case → lowercase token set."""
    if not s:
        return set()
    # camelCase split
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    parts = _TOKEN_SPLIT.split(s.lower())
    return {p for p in parts if p}


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-ish overlap on token sets. 1.0 = identical token sets,
    0.0 = no overlap. Used for chain-confidence (param-name vs source-field-name)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _cluster_key(snode: StructuredHttpNode) -> str:
    """Stable internal key per Tool. (host, path_template, op_name)."""
    op_name = ""
    if snode.payload_type in ("graphql", "graphql_apq"):
        # op_name = last path segment (e.g., "ProductDetails" from /api/graphql/ProductDetails)
        segs = [s for s in snode.path.split("/") if s]
        if segs:
            op_name = segs[-1]
    return f"{snode.host}|{snode.path}|{op_name}|{snode.method}"


def _source_node_to_cluster(
    structured_nodes: list[StructuredHttpNode],
) -> dict[str, str]:
    """node_id → cluster_key. Used by the chainer to translate
    SourceBucket.recipe.source_node_id (a node) into a tool_id (a cluster)."""
    return {s.node_id: _cluster_key(s) for s in structured_nodes}


def _slot_name_from_leaf(leaf: ParamLeaf) -> str:
    """A stable slot name from a leaf — uses the leaf's `key`, stripping
    list-index suffixes like `[0]` and structural prefixes like `path[N]`."""
    name = leaf.key
    if "#" in name:
        # url sub-leaf style: `url#path[1]` → take the part after #
        name = name.split("#")[-1]
        name = name.replace("[", "_").replace("]", "")
    return name


def build_tool_dag(
    temporal: Any, structured_nodes: list[StructuredHttpNode],
) -> ToolDAG:
    """Build the ToolDAG: cluster non-noise nodes, materialize one Tool per
    cluster with slots derived from request-side leaves. Slot.chain_edge and
    Slot.intent_binding are populated where the leaf's SourceBucket warrants."""
    # 1. Filter and cluster
    keep = [s for s in structured_nodes if not s.is_noise]
    by_cluster: dict[str, list[StructuredHttpNode]] = {}
    for snode in keep:
        by_cluster.setdefault(_cluster_key(snode), []).append(snode)
    node_to_cluster = _source_node_to_cluster(keep)

    # 2. Per-cluster Tool
    tools: list[Tool] = []
    for cluster_key, snodes in by_cluster.items():
        # Canonical exemplar = highest-status non-empty response, else first.
        snodes_sorted = sorted(
            snodes,
            key=lambda s: (1 if s.status == 200 else 0, s.ts_ms),
            reverse=True,
        )
        exemplar = snodes_sorted[0]

        slots: dict[str, Slot] = {}
        for leaf in exemplar.slots:
            src = leaf.source
            if src is None:
                continue
            # Drop trivial-empty leaves (no slot value to manage)
            if src.bucket == "trivial":
                continue
            name = _slot_name_from_leaf(leaf)
            # If multiple leaves resolve to the same slot name (e.g., url_query
            # variables.slug AND extensions.slug somehow), keep the first.
            if name in slots:
                continue
            slot = Slot(
                name=name, type="string",
                location=leaf.location,
                parent_path=list(leaf.parent_path),
                source=src,
                examples=[leaf.raw_value] if leaf.raw_value else [],
            )

            # Derived typed views — chain_edge / intent_binding
            if src.bucket == "chained_resp":
                src_node_id = src.recipe.get("source_node_id")
                source_cluster = node_to_cluster.get(src_node_id) if src_node_id else None
                # Determine direct vs list_select
                source_node = next(
                    (s for s in keep if s.node_id == src_node_id), None,
                )
                kind: ChainEdgeKind = "direct"
                list_jp = None
                selector = {}
                per_item = {"extract": "whole"}
                if source_node is not None and source_node.response_summary.repeating_jsonpath:
                    # Check if the source leaf is INSIDE the repeating list
                    src_path = src.recipe.get("source_parent_path") or []
                    src_path_str = ".".join(str(op[1]) for op in src_path)
                    rep_jp = source_node.response_summary.repeating_jsonpath
                    rep_jp_norm = rep_jp.replace("$", "").replace("[*]", "")
                    if rep_jp_norm.strip(".") in src_path_str:
                        kind = "list_select"
                        list_jp = rep_jp
                        selector = {
                            "kind": "list_select",
                            "candidate_fields": source_node.response_summary.item_key_fields[:6],
                            "trivial_if_n_eq_1": True,
                        }
                # Per-item extract: if the source leaf is a URL sub-leaf, encode the path index
                src_kind = src.recipe.get("source_parent_kind")
                if src_kind == "url_segment":
                    # Find the final url_path / url_query op
                    src_path = src.recipe.get("source_parent_path") or []
                    tail = src_path[-1] if src_path else None
                    if tail and tail[0] == "url_path":
                        per_item = {"field": _last_field_name(src_path), "path_segment_at": tail[1]}
                    elif tail and tail[0] == "url_query":
                        per_item = {"field": _last_field_name(src_path), "url_query": tail[1]}
                confidence = _token_overlap(name, _last_field_name(src.recipe.get("source_parent_path") or []) or "")
                slot.chain_edge = ChainEdge(
                    kind=kind, source_tool_id=source_cluster or "unknown_source",
                    list_jsonpath=list_jp, per_item_extract=per_item,
                    selector_recipe=selector, confidence=confidence,
                )
            elif src.bucket == "user_intent":
                kind_in: IntentBindingKind = (
                    "user_intent:typed" if src.sub_type == "typed" else "user_intent:click"
                )
                slot.intent_binding = IntentBinding(
                    kind=kind_in,
                    source_action_ts=src.recipe.get("action_ts_ms") or 0,
                    examples=[leaf.raw_value] if leaf.raw_value else [],
                )
                # Click recipe construction is finalized in §7.

            slots[name] = slot

        # Build endpoint template (raw for now; §8 finalizes). Drop Cookie
        # from captured headers — the executor's cookie jar supplies the
        # session value at replay time. Capturing it verbatim would re-send
        # a stale jar entry and break long-lived sessions where the cookie
        # rotates server-side.
        raw_headers = dict(exemplar.raw_node.request_headers or {})
        captured_headers = {
            k: v for k, v in raw_headers.items()
            if k.lower() != "cookie"
        }
        endpoint_template = {
            "method": exemplar.method,
            "url_template": exemplar.raw_node.url,
            "headers_template": captured_headers,
            "body_template": exemplar.raw_node.request_body,
        }
        tool = Tool(
            tool_id=cluster_key,                   # placeholder; §8 renames via LLM
            cluster_key=cluster_key,
            payload_type=exemplar.payload_type,
            page_class=exemplar.page_class,
            endpoint_template=endpoint_template,
            slots=slots,
        )
        tools.append(tool)

    # 3. Script-edge attachment via initiator_stack
    _attach_script_dependencies(tools, temporal, keep)

    # 4. Pick site + entry_url + page_class from metadata
    site = temporal.metadata.get("site") or ""
    entry_url = temporal.metadata.get("start_url") or ""
    primary_page: PageClass | None = None
    for snode in keep:
        if snode.page_class is not None:
            primary_page = snode.page_class
            break
    return ToolDAG(site=site, page_class=primary_page, entry_url=entry_url, tools=tools)


def _last_field_name(parent_path: list) -> str | None:
    """Return the last 'key' op's value in a parent_path (e.g., for
    `[("key","suggestions"),("index",0),("key","url"),("url_path",1)]` → "url")."""
    for op, val in reversed(parent_path):
        if op == "key":
            return str(val)
    return None


def _attach_script_dependencies(
    tools: list[Tool], temporal: Any, snodes: list[StructuredHttpNode],
) -> None:
    """For each tool, walk the exemplar request's initiator_stack to find the
    app-code script(s) that originated the call. Map scriptId → sha256 via
    ScriptCorpus url-keyed reverse lookup. Set Tool.script_dependencies."""
    by_cluster = {t.cluster_key: t for t in tools}
    for snode in snodes:
        tool = by_cluster.get(_cluster_key(snode))
        if tool is None or tool.script_dependencies:
            continue
        stack = snode.raw_node.initiator_stack or []
        seen_shas: list[str] = []
        for frame in stack:
            url = frame.get("url")
            if not url:
                continue
            sha = temporal.scripts.sha_for_url(url)
            if sha and sha not in seen_shas:
                seen_shas.append(sha)
        tool.script_dependencies = seen_shas[:5]   # cap


# ═════════════════════════════════════════════════════════════════════════
# §7 — User-intent linker + click_recipe construction + graph isolation
# ═════════════════════════════════════════════════════════════════════════

def _build_click_recipe(
    binding: IntentBinding, slot_value: str, temporal: Any,
) -> dict:
    """Build the executor-facing click_recipe from a click IntentBinding.
    Prefers regex anchored on a STRONG_TEMPLATE neighbour (the structural
    constant before the slot value) over positional path_segment_at."""
    action = next(
        (a for a in temporal.cu_actions() if a.ts_ms == binding.source_action_ts),
        None,
    )
    if action is None or not isinstance(action.target_attrs, dict):
        return {"selector": None, "attribute": "text", "extract": "whole"}

    href = action.target_attrs.get("href")
    if href and slot_value in str(href):
        # Find what precedes the slot value in the href
        href_str = str(href)
        idx = href_str.find(slot_value)
        if idx > 0:
            # Walk back to the previous '/' or '?' or '=' — that's our anchor
            prefix = href_str[:idx].rstrip("/")
            anchor = prefix.split("/")[-1] if "/" in prefix else prefix.split("=")[-1]
            if anchor:
                return {
                    "selector": f"a[href*='/{anchor}/']",
                    "attribute": "href",
                    "extract_regex": f"/{re.escape(anchor)}/([^/?]+)",
                }
        return {
            "selector": "a[href]",
            "attribute": "href",
            "extract": "whole",
        }
    # No href anchor — fall back to text match
    return {
        "selector": None,
        "attribute": "text",
        "extract": "whole",
    }


def finalize_intent_bindings(dag: ToolDAG, temporal: Any) -> None:
    """For each user_intent:click slot, build the click_recipe. (Typed bindings
    don't need a recipe — they're filled at planner-call time.)"""
    for tool in dag.tools:
        for slot in tool.slots.values():
            ib = slot.intent_binding
            if ib is None:
                continue
            if ib.kind != "user_intent:click":
                continue
            value = slot.examples[0] if slot.examples else ""
            ib.click_recipe = _build_click_recipe(ib, value, temporal)


def graph_isolation_filter(dag: ToolDAG) -> ToolDAG:
    """KEEP a tool iff at least one of:
      (a) it has a user_intent slot (typed/click/selection) — the planner has
          something meaningful to supply.
      (b) chain_in > 0 — it consumes another tool's response (downstream node
          in a chain).
      (c) chain_out > 0 — its response is consumed by another tool (upstream
          node in a chain).
    Otherwise DROP. This kills:
      - analytics / consent / experiment leakers (no user input, no chain)
      - useless page-load GETs (open_homepage etc.) that nobody chains from
      - cluster duplicates that aren't load-bearing

    Tools with `unknown` leaves are KEPT — the executor uses the captured
    value verbatim for those slots at replay. Each surviving tool's
    `classification_confidence` is set to the fraction of leaves with
    non-unknown origin; the verifier correlates confidence with PASS/FAIL
    to diagnose whether chain LOGIC is broken (high-confidence FAIL) vs
    chain COVERAGE is partial (many low-confidence tools).

    `ChainEdge.source_tool_id` carries the `cluster_key` at this stage (LLM
    `tool_id` is assigned later in §8), so the lookup is keyed on cluster_key."""

    chain_out_count: dict[str, int] = {t.cluster_key: 0 for t in dag.tools}
    for t in dag.tools:
        for slot in t.slots.values():
            ce = slot.chain_edge
            if ce and ce.source_tool_id and ce.source_tool_id in chain_out_count:
                chain_out_count[ce.source_tool_id] += 1

    def _has_user_intent(t: Tool) -> bool:
        return any(s.intent_binding is not None for s in t.slots.values())

    def _has_chain_in(t: Tool) -> bool:
        return any(s.chain_edge is not None for s in t.slots.values())

    def _compute_confidence(t: Tool) -> float:
        if not t.slots:
            return 0.0
        n = len(t.slots)
        n_known = sum(
            1 for s in t.slots.values()
            if s.source is not None and s.source.bucket != "unknown"
        )
        return n_known / n

    survivors: list[Tool] = []
    for t in dag.tools:
        if _has_user_intent(t) or _has_chain_in(t) or chain_out_count[t.cluster_key] > 0:
            t.classification_confidence = _compute_confidence(t)
            survivors.append(t)
    dag.tools = survivors
    return dag


# ═════════════════════════════════════════════════════════════════════════
# §8 — LLM naming (Gemini) + tools.json writer
# ═════════════════════════════════════════════════════════════════════════
#
# One Gemini call per kept tool authors:
#   - verdict (keep | discard) — final filter pass
#   - tool_id  (snake_case, <site>_<verb>_<noun>)
#   - capability_statement (1 sentence, user-facing)
#   - slot_descriptions (1 sentence per slot)
#
# The writer then emits tools.json carrying ONLY recipes (per the 8-point
# thesis: no values ever stored). The chained-slot shape carries source_tool,
# list_jsonpath, per_item_extract, selector_recipe.


class SlotDescription(BaseModel):
    """One slot's name + LLM-authored description. Listed (not dict-keyed)
    because Gemini's structured-output API rejects dict-with-arbitrary-keys
    (it serializes to `additionalProperties` which Gemini doesn't support)."""

    slot_name: str
    description: str


class ToolFinaliserResponse(BaseModel):
    """Strict Gemini structured-output schema. The LLM is an author + verifier,
    NOT a classifier — source buckets are deterministic facts from the tracer.
    The LLM writes three things: tool_id, capability_statement (action + shape),
    slot_descriptions (one per slot, as a list). No keep/discard verdict —
    graph isolation + is_noise handle filtering; the LLM flags suspected
    analytics inline in the capability_statement instead."""

    tool_id: str
    capability_statement: str
    slot_descriptions: list[SlotDescription]


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_TEMPLATE_CACHE: dict[str, jinja2.Template] = {}


def _load_template(name: str) -> jinja2.Template:
    if name not in _TEMPLATE_CACHE:
        text = (_PROMPT_DIR / f"{name}.j2").read_text(encoding="utf-8")
        _TEMPLATE_CACHE[name] = jinja2.Template(text, trim_blocks=True, lstrip_blocks=True)
    return _TEMPLATE_CACHE[name]


def _summarize_cu_action(a: Any) -> str:
    """One-line human-readable summary of a CuActionEvent."""
    if a.kind == "type_text" and a.text:
        return f"typed `{a.text[:80]}`"
    if a.kind == "click" and a.target_attrs:
        text = a.target_attrs.get("text") or ""
        href = a.target_attrs.get("href") or ""
        if text:
            return f"clicked text `{str(text)[:80]}`"
        if href:
            return f"clicked link `{str(href)[:80]}`"
        return f"clicked aid={a.target_aid}"
    if a.kind in ("scroll", "scroll_page"):
        return f"scrolled (target={a.target_aid or 'page'})"
    if a.kind == "key_press":
        return f"pressed keys"
    return f"{a.kind} target={a.target_aid}"


def _scoped_cu_actions(tool: Tool, temporal: Any) -> list[dict]:
    """Find the CU actions that fired in the SAME step window as this tool's
    exemplar HTTP node, BEFORE the HTTP request fired. This is the right
    causal scope — pre-tool actions in the same step are the ones that led to
    this call. (Earlier-step actions or later actions confuse the LLM.)"""
    # The exemplar's request_ts_ms is on the tool's endpoint_template only
    # implicitly; find it via the cluster_key match against the temporal HTTP
    # nodes (best-effort — we don't carry the raw HttpNode handle into Tool).
    target_url = tool.endpoint_template.get("url_template") or ""
    target_method = tool.endpoint_template.get("method") or ""
    exemplar_ts = 0
    for n in temporal.http_nodes():
        if n.url == target_url and n.method == target_method:
            exemplar_ts = n.request_ts_ms
            break
    if exemplar_ts == 0:
        # Fallback: last 5 CU actions (less precise but never empty).
        return [
            {"kind": a.kind, "summary": _summarize_cu_action(a)}
            for a in temporal.cu_actions()[-5:]
        ]
    # Find the containing step window.
    step_start = 0
    for _step_id, start, end in temporal.step_windows():
        end_ts = end if end is not None else (exemplar_ts + 1)
        if start <= exemplar_ts <= end_ts:
            step_start = start
            break
    actions_in_scope = [
        a for a in temporal.cu_actions(before=exemplar_ts)
        if a.ts_ms >= step_start
    ]
    if not actions_in_scope:
        # No actions in the step before this HTTP fired — page-load triggered.
        return []
    return [{"kind": a.kind, "summary": _summarize_cu_action(a)} for a in actions_in_scope[-8:]]


def _render_tool_finaliser_prompt(
    tool: Tool, temporal: Any, original_task: str,
) -> str:
    """Build the inputs the tool_finaliser.j2 expects."""
    p = urlsplit(tool.endpoint_template.get("url_template") or "")
    # Sample response preview — find the exemplar HttpNode by URL+method match.
    sample_response = "(no response body captured)"
    target_url = tool.endpoint_template.get("url_template") or ""
    target_method = tool.endpoint_template.get("method") or ""
    for n in temporal.http_nodes():
        if n.url == target_url and n.method == target_method and n.response_body:
            sample_response = n.response_body[:800]
            break
    # Scoped CU actions — only those in the same step window before this HTTP fired.
    scoped_actions = _scoped_cu_actions(tool, temporal)
    # Script summary — app scripts from initiator_stack.
    script_summary: list[str] = []
    for sha in tool.script_dependencies[:3]:
        url = temporal.scripts.url_for(sha) or "(unresolved)"
        script_summary.append(f"{sha[:12]}… {url}")
    if not script_summary:
        script_summary = ["(no app-code initiator captured)"]

    tmpl = _load_template("tool_finaliser")
    return tmpl.render(
        original_task=original_task,
        method=tool.endpoint_template.get("method") or "",
        host=p.netloc or "",
        path=p.path or "",
        payload_type=tool.payload_type,
        status=200,
        slots=tool.slots,
        sample_response_preview=sample_response,
        scoped_cu_actions=scoped_actions,
        script_summary=script_summary,
    )


async def _finalise_one(
    tool: Tool, temporal: Any, sm: SessionManager, original_task: str,
) -> None:
    """Single Gemini call to finalise one Tool. Mutates the tool in place.
    The LLM is constrained by `response_schema=ToolFinaliserResponse` (Gemini
    structured output) — no free-form JSON parsing; the parsed object is
    enforced by the API itself."""
    prompt = _render_tool_finaliser_prompt(tool, temporal, original_task)
    response = await sm.call_gemini(
        model="gemini-3-flash-preview",
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        response_schema=ToolFinaliserResponse,
        temperature=0.0,
    )
    parsed = ToolFinaliserResponse.model_validate(response)
    if parsed.tool_id:
        tool.tool_id = parsed.tool_id
    tool.capability_statement = parsed.capability_statement
    for entry in parsed.slot_descriptions:
        if entry.slot_name in tool.slots:
            tool.slots[entry.slot_name].description = entry.description
    # `[ANALYTICS-LIKE]` / `[HOUSEKEEPING]` prefix in the capability_statement
    # is the LLM's leak-flag — planner naturally avoids these. Tool stays in
    # the DAG for debug visibility (per the user's directive: don't silently
    # drop, let the planner ignore via description signal).


async def finalise_tools(
    dag: ToolDAG, temporal: Any, sm: SessionManager, original_task: str,
) -> None:
    """One Gemini call per tool, in parallel. Mutates `dag.tools` in place
    with `tool_id`, `capability_statement`, slot descriptions. The LLM is
    AUTHOR + VERIFIER, NOT classifier — source buckets came from the
    deterministic tracer and are not revised here. No keep/discard verdict;
    suspected analytics get flagged inline via `[ANALYTICS-LIKE]` prefix in
    the capability statement, so the planner naturally avoids them while the
    tool stays in the DAG for debug.

    Post-call: deduplicate any colliding tool_ids — the LLM occasionally
    picks the same name for two cluster_keys (e.g. /en-in and /en-sg both
    became `open_homepage`). The dedup keeps the first occurrence and
    appends `_2`, `_3`, ... to later collisions, so each tool_id stays
    unique across tools.json.

    After dedup, rewrite every chain edge's `source_tool_id` from the
    chain-time `cluster_key` to the final LLM-assigned `tool_id`. The
    executor's per-task response cache is keyed on tool_id; without this
    rewrite, chained-slot lookups would always miss. Asserts that every
    rewritten source_tool_id exists in the DAG so a stray chain edge
    fails loud at write time, not silently at replay."""
    await asyncio.gather(*[_finalise_one(t, temporal, sm, original_task) for t in dag.tools])
    seen: dict[str, int] = {}
    for tool in dag.tools:
        base = tool.tool_id
        if not base:
            continue
        if base not in seen:
            seen[base] = 1
            continue
        seen[base] += 1
        tool.tool_id = f"{base}_{seen[base]}"

    cluster_to_id: dict[str, str] = {t.cluster_key: t.tool_id for t in dag.tools if t.tool_id}
    known_ids = set(cluster_to_id.values())
    for tool in dag.tools:
        for slot in tool.slots.values():
            ce = slot.chain_edge
            if ce is None or not ce.source_tool_id:
                continue
            if ce.source_tool_id in cluster_to_id:
                ce.source_tool_id = cluster_to_id[ce.source_tool_id]
            elif ce.source_tool_id not in known_ids:
                raise RuntimeError(
                    f"chain edge in tool {tool.tool_id!r} points at "
                    f"{ce.source_tool_id!r} which is neither a known cluster_key "
                    f"nor a tool_id — DAG corruption."
                )


# ── tools.json writer ────────────────────────────────────────────────────


def _slot_to_json(slot: Slot) -> dict:
    """Serialize one Slot to tools.json shape. RECIPES ONLY — no captured
    values escape into the persisted artifact. The one exception is
    `bundle:<sub_type>` where the recipe IS "use this captured literal" — that
    value lives under `source.value`, not `examples`."""
    out: dict[str, Any] = {
        "type": slot.type,
        "required": slot.required,
        "location": slot.location,
        "parent_path": slot.parent_path,
        "description": slot.description,
    }
    src = slot.source
    if src is not None:
        source_obj: dict[str, Any] = {"bucket": src.bucket}
        if src.sub_type:
            source_obj["sub_type"] = src.sub_type
        if src.recipe:
            source_obj["recipe"] = src.recipe
        if src.bucket == "bundle" and src.value is not None:
            # bundle sub_types where the recipe IS "use the captured literal"
            source_obj["value"] = src.value
        out["source"] = source_obj
    if slot.chain_edge:
        ce = slot.chain_edge
        out["chain"] = {
            "kind": ce.kind,
            "source_tool_id": ce.source_tool_id,
            "list_jsonpath": ce.list_jsonpath,
            "per_item_extract": ce.per_item_extract,
            "selector_recipe": ce.selector_recipe,
            "confidence": ce.confidence,
        }
    if slot.intent_binding:
        ib = slot.intent_binding
        out["intent"] = {
            "kind": ib.kind,
            "source_action_ts": ib.source_action_ts,
            "click_recipe": ib.click_recipe,
        }
    # Examples — keep ONE captured example for the LLM planner's reference.
    # NOT a recipe; just helpful for the planner LLM at the next layer.
    if slot.examples:
        out["examples"] = [slot.examples[0]]
    return out


def _tool_to_json(tool: Tool) -> dict:
    """Serialize one Tool to tools.json shape. Derives http_precursors from
    slot chain edges (per the 'precursors are implicit' decision)."""
    http_precursors = sorted({
        slot.chain_edge.source_tool_id
        for slot in tool.slots.values()
        if slot.chain_edge is not None
    })
    return {
        "tool_id": tool.tool_id,
        "cluster_key": tool.cluster_key,
        "capability_statement": tool.capability_statement,
        "payload_type": tool.payload_type,
        "page_class": tool.page_class,
        "requires_js_render": tool.requires_js_render,
        "http_precursors": http_precursors,
        "script_dependencies": tool.script_dependencies,
        "endpoint": {
            "method": tool.endpoint_template.get("method"),
            "url_template": tool.endpoint_template.get("url_template"),
            # Header + body templates: stored as-captured so the executor can
            # substitute materialized values. The materializer walks each
            # slot's parent_path to apply values; the template provides the
            # surrounding bytes.
            "headers_template": tool.endpoint_template.get("headers_template"),
            "body_template": tool.endpoint_template.get("body_template"),
        },
        "slots": {name: _slot_to_json(slot) for name, slot in tool.slots.items()},
        "history": tool.history,
        # Forensics — NOT surfaced to planner prompt. Used by verifier output
        # + cross-task analysis to diagnose chaining logic vs coverage gaps.
        "classification_confidence": round(tool.classification_confidence, 3),
    }


def write_tools_json(dag: ToolDAG, out_path: Path) -> Path:
    """Emit tools.json. Includes only `verdict=keep` tools; `discard` tools
    are dropped here (the verdict was the LLM's final filter pass).

    Schema (canonical reference — what tool_executor.py reads):
    ```jsonc
    {
      "schema_version": 1,
      "site": "<host>",
      "page_class": "<spa_shell|ssr_hybrid|mpa|amp|static_html|unknown>",
      "entry_url": "<JS-render precursor target>",
      "tools": [
        {
          "tool_id": "<site>_<verb>_<noun>",
          "cluster_key": "<host>|<path>|<op>|<method>",
          "capability_statement": "...",
          "payload_type": "...",
          "page_class": null | "...",
          "requires_js_render": true,
          "http_precursors": ["<other_tool_id>", ...],
          "script_dependencies": ["<sha256>", ...],
          "endpoint": {method, url_template, headers_template, body_template},
          "slots": {
            "<slot_name>": {
              "type": "string",
              "required": true,
              "location": "url_query|url_path_segment|header|body",
              "parent_path": [["key","..."], ["index", N], ...],
              "description": "...",
              "source": {"bucket": "...", "sub_type": "...", "recipe": {...}, "value": ...?},
              "chain": {kind, source_tool_id, list_jsonpath, per_item_extract, selector_recipe, confidence},
              "intent": {kind, source_action_ts, click_recipe},
              "examples": ["<one captured example for planner reference>"]
            }
          },
          "history": {"total_runs": 0, "success_rate": 0.0, "lifecycle": "pending"}
        }
      ]
    }
    ```

    Invariants:
      - NO captured values escape except: `bundle:*` `source.value` (the recipe IS
        the literal), `examples[0]` (one reference value for the planner).
      - All other slot resolution at replay goes through `source.recipe` +
        materializer dispatch.
    """
    # Read existing tools.json (if present) and merge by cluster_key. Tools
    # accumulate across mints so prior tasks' discoveries survive into the
    # current run's registry. On cluster_key match:
    #   - preserve old tool_id + capability_statement (LLM-authored, stable
    #     identifiers the planner has already seen)
    #   - preserve old history block (lifecycle / total_runs / success_rate)
    #     — the verifier owns that timeline, not the miner
    #   - update endpoint_template + slots + script_dependencies from new mint
    #     (new chain edges or slot recipes may have surfaced)
    # Existing tools whose cluster_key did NOT re-appear in this mint stay
    # as-is — we don't delete them (they may have been minted by a different
    # task on the same site and remain valid).
    existing_by_cluster: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing_doc = json.loads(out_path.read_text(encoding="utf-8"))
            for t in (existing_doc.get("tools") or []):
                key = t.get("cluster_key")
                if key:
                    existing_by_cluster[key] = t
        except Exception:
            # corrupt / older schema — fall through, overwrite cleanly
            existing_by_cluster = {}

    merged_tools: list[dict] = []
    new_cluster_keys: set[str] = set()
    for tool in dag.tools:
        new_t = _tool_to_json(tool)
        ck = new_t.get("cluster_key") or ""
        new_cluster_keys.add(ck)
        existing = existing_by_cluster.get(ck)
        if existing is not None:
            # Preserve stable / time-sensitive fields from existing entry.
            if existing.get("tool_id"):
                new_t["tool_id"] = existing["tool_id"]
            if existing.get("capability_statement"):
                new_t["capability_statement"] = existing["capability_statement"]
            if existing.get("history"):
                new_t["history"] = existing["history"]
            # Preserve LLM-authored slot descriptions on slots that still exist
            for slot_name, slot_obj in (new_t.get("slots") or {}).items():
                old_slot = (existing.get("slots") or {}).get(slot_name) or {}
                if old_slot.get("description") and not slot_obj.get("description"):
                    slot_obj["description"] = old_slot["description"]
        merged_tools.append(new_t)

    # Carry over existing tools whose cluster_key did not re-appear.
    for ck, t in existing_by_cluster.items():
        if ck in new_cluster_keys:
            continue
        merged_tools.append(t)

    doc = {
        "schema_version": 1,
        "site": dag.site,
        "page_class": dag.page_class,
        "entry_url": dag.entry_url,
        "tools": merged_tools,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return out_path


# ═════════════════════════════════════════════════════════════════════════
# §8b — Verifier (live attached-mode replay against the still-open browser)
# ═════════════════════════════════════════════════════════════════════════
#
# Per swift-brewing-hare plan Chunk 7: immediately after tool_builder mints
# the DAG, hand it to the verifier. The verifier walks the DAG topologically
# and, FOR EACH TOOL, asks the planner LLM to generate user-intent slot
# values from the original task, then invokes the tool via ToolExecutor in
# attached mode against the still-open discovery browser.
#
# PASS = 2xx + non-empty body + not anti-bot HTML.
# FAIL → structured `history.last_failure` + lifecycle = "failing".
# Never silently dropped — diagnostic visibility is the goal.


@dataclass
class VerificationVerdict:
    tool_id: str
    passed: bool
    http_status: int = 0
    planner_args: dict = field(default_factory=dict)
    error: str | None = None
    response_summary: str = ""


@dataclass
class VerificationReport:
    site: str
    task: str
    verdicts: list[VerificationVerdict] = field(default_factory=list)

    def summary(self) -> str:
        passed = sum(1 for v in self.verdicts if v.passed)
        total = len(self.verdicts)
        return f"{self.site}: {passed}/{total} verified"


def _topological_sort(dag: ToolDAG) -> list[Tool]:
    """Topological sort by chain edges: a tool comes after all tools whose
    responses feed its slots. Tools with cycles or missing dependencies are
    placed at the end (deterministic fallback)."""
    by_id: dict[str, Tool] = {t.tool_id or t.cluster_key: t for t in dag.tools}
    deps: dict[str, set[str]] = {
        key: set() for key in by_id.keys()
    }
    for key, tool in by_id.items():
        for slot in tool.slots.values():
            if slot.chain_edge and slot.chain_edge.source_tool_id in by_id:
                deps[key].add(slot.chain_edge.source_tool_id)
    out: list[Tool] = []
    visited: set[str] = set()

    def visit(node_key: str) -> None:
        if node_key in visited:
            return
        visited.add(node_key)
        for d in deps.get(node_key, set()):
            visit(d)
        if node_key in by_id:
            out.append(by_id[node_key])

    for key in by_id.keys():
        visit(key)
    return out


def _build_verification_fingerprint(
    tool: Tool, sm: SessionManager, planner_args: dict,
) -> dict:
    """Snapshot which environmental things this tool depended on at verify
    time. Production-time invoke() can sanity-check these (e.g. cookie names
    still present, bundle SHAs still loaded) before firing."""
    cookie_names: list[str] = []
    state_keys: list[str] = []
    for slot in tool.slots.values():
        src = slot.source
        if src is None:
            continue
        if src.bucket == "cookie" and src.recipe.get("name"):
            cookie_names.append(src.recipe["name"])
        elif src.bucket == "session_state" and src.recipe.get("key"):
            state_keys.append(src.recipe["key"])
    return {
        "ran_at_ms": int(time.time() * 1000) if (time := __import__("time")) else 0,
        "cookie_names_used": sorted(set(cookie_names)),
        "localStorage_keys_used": sorted(set(state_keys)),
        "bundle_shas_used": list(tool.script_dependencies),
        "page_url_at_verify": sm.page.url if sm and sm.page else "",
        "planner_args_used": planner_args,
    }


def _is_pass(result: Any) -> bool:
    """ToolResult → PASS verdict. 2xx + non-empty body + not anti-bot HTML.
    `result` is always a ToolResult dataclass; the fields below are explicit."""
    if result is None:
        return False
    if not (200 <= result.http_status < 300):
        return False
    if result.fall_back_to_cu:
        return False
    body = result.body
    if not body or len(body.strip()) < 4:
        return False
    return True


async def _planner_args_for_one_tool(
    sm: SessionManager, tool: Tool, original_task: str,
) -> dict:
    """One planner turn: build a function declaration for just THIS tool,
    pass it + the original task, capture the function_call args. Reuses the
    production planner.j2 / build_planner_function_declarations machinery —
    no new prompt path."""
    from morphnet_v3.planner import (
        PlanningTree, ToolRegistry, ToolEntry, SlotDef,
        build_planner_function_declarations, render_planner_prompt,
        _parse_planner_response,
    )
    from google.genai import types as genai_types

    # Build a single-tool registry so the planner only sees `invoke_<tool>`
    # alongside the static actions (continue_cu / complete_task / give_up).
    registry = ToolRegistry()
    _slot_type_allowed = {"string", "number", "integer", "boolean", "array", "object"}
    registry.register(ToolEntry(
        tool_id=tool.tool_id or tool.cluster_key,
        capability_statement=tool.capability_statement or "(no description)",
        slots=[
            SlotDef(
                name=name,
                type=slot.type if slot.type in _slot_type_allowed else "string",  # type: ignore[arg-type]
                required=slot.required,
                description=slot.description,
                examples=slot.examples[:1],
            )
            for name, slot in tool.slots.items()
        ],
    ))
    tree = PlanningTree()
    tree.create_root(original_task)
    browser_state = {"url": sm.page.url if sm.page else "", "v5": ""}
    prompt = render_planner_prompt(
        task=original_task, tree=tree, registry=registry,
        trigger="task_start", browser_state=browser_state,
    )
    declarations = build_planner_function_declarations(registry)
    tools_arg = [genai_types.Tool(function_declarations=declarations)]
    contents = [genai_types.Content(
        role="user", parts=[genai_types.Part.from_text(text=prompt)],
    )]
    resp = await sm.call_gemini(
        model="gemini-3-flash-preview",
        contents=contents,
        tools=tools_arg,
        temperature=0.0,
    )
    decision = _parse_planner_response(resp)
    return decision.tool_user_intent or {}


async def verify_dag(
    dag: ToolDAG, original_task: str, sm: SessionManager,
) -> VerificationReport:
    """End-to-end: walk DAG in topo order, planner generates slots per tool,
    executor invokes against live server, classify pass/fail, stamp lifecycle
    + verification_fingerprint, return report."""
    from morphnet_v3.tool_executor import ToolExecutor

    # Construct executor in attached mode — discovery already populated the browser.
    executor = ToolExecutor(sm=sm, site=dag.site, attached=True)
    # Inject the DAG's tools without going through tools.json (the writer
    # hasn't been called yet at verification time — verify-then-write order).
    for tool in dag.tools:
        tid = tool.tool_id or tool.cluster_key
        # Re-serialize this single tool via _tool_to_json so the executor
        # sees the same shape it would post-write.
        executor._tools_by_id[tid] = _tool_to_json(tool)

    sorted_tools = _topological_sort(dag)
    report = VerificationReport(site=dag.site, task=original_task)

    for tool in sorted_tools:
        tid = tool.tool_id or tool.cluster_key

        # 1. Planner generates user-intent args.
        try:
            planner_args = await _planner_args_for_one_tool(sm, tool, original_task)
        except Exception as exc:
            tool.history["lifecycle"] = "failing"
            tool.history["last_failure"] = {
                "stage": "planner",
                "reason": f"planner generation failed: {exc!r}",
            }
            report.verdicts.append(VerificationVerdict(
                tool_id=tid, passed=False, error=f"planner: {exc!r}",
            ))
            continue

        # 2. Executor invokes.
        try:
            result = await executor.invoke(tid, planner_args, task_text=original_task)
        except Exception as exc:
            tool.history["lifecycle"] = "failing"
            tool.history["last_failure"] = {
                "stage": "executor",
                "reason": f"invoke raised: {exc!r}",
                "planner_args": planner_args,
            }
            report.verdicts.append(VerificationVerdict(
                tool_id=tid, passed=False, planner_args=planner_args,
                error=f"executor: {exc!r}",
            ))
            continue

        # 3. Classify + stamp.
        passed = _is_pass(result)
        tool.history["last_planner_args"] = planner_args
        tool.history["verification_fingerprint"] = _build_verification_fingerprint(
            tool, sm, planner_args,
        )
        if passed:
            tool.history["lifecycle"] = "verified"
            tool.history.pop("last_failure", None)
        else:
            tool.history["lifecycle"] = "failing"
            tool.history["last_failure"] = {
                "stage": "response",
                "http_status": result.http_status,
                "reason": result.error or "non-2xx or anti-replay HTML",
                "response_summary": result.response_summary[:300],
                "planner_args": planner_args,
            }
        report.verdicts.append(VerificationVerdict(
            tool_id=tid, passed=passed,
            http_status=result.http_status,
            planner_args=planner_args,
            error=result.error,
            response_summary=result.response_summary[:300],
        ))

    return report


def write_verification_report(report: VerificationReport, out_path: Path) -> Path:
    """Markdown verdict table → `morphnet_v3/sites/<site>/tools_verification.md`."""
    lines = [
        f"# Tools verification — {report.site}",
        "",
        f"Task: `{report.task}`",
        f"Result: {report.summary()}",
        "",
        "| tool_id | verdict | status | planner_args | notes |",
        "|---|---|---|---|---|",
    ]
    for v in report.verdicts:
        verdict = "✅ verified" if v.passed else "❌ failing"
        notes = (v.error or v.response_summary or "")[:80].replace("|", "\\|")
        args = json.dumps(v.planner_args, separators=(",", ":"))[:80].replace("|", "\\|")
        lines.append(
            f"| `{v.tool_id}` | {verdict} | {v.http_status} | `{args}` | {notes} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path