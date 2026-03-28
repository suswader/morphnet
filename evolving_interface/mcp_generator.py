"""Analyze HTTP traffic from computer-use runs and generate reusable
MCP tool definitions stored per-site under mcps/{site}/tools.json."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, parse_qs

import httpx
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from playwright.async_api import Page

from . import config
from .computer_use import CapturedRequest, get_browser_cookies

# ── Gemini client ────────────────────────────────────────────────
_gemini = genai.Client(api_key=config.GEMINI_API_KEY)
_NAMING_MODEL = config.GEMINI_FAST_MODEL

# ── shared constants ─────────────────────────────────────────────
STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

EXCLUDED_PATH_FRAGMENTS = {
    "/static/", "/media/", "/pub/", "/theme/",
    "/customer/section/load", "/directory/currency/switch",
    "/translation/ajax/index", "/wishlist/index/",
    "/searchtermslog/", "/favicon",
}

_ID_PATTERNS = [
    re.compile(r"^\d{2,}$"),
    re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE),
    re.compile(r"^[A-Za-z0-9_-]{20,}$"),
    re.compile(r"^\d+\.\d{2}$"),
]

_PATH_STOP_WORDS = {
    "cart", "products", "product", "search", "checkout", "order",
    "orders", "account", "customer", "admin", "catalog", "rest",
    "ajax", "api", "index", "view", "list", "page", "save",
    "load", "update", "delete", "add", "remove", "edit", "create",
    "v1", "v2", "default",
}

_TASK_STOP_WORDS = {
    "the", "this", "that", "with", "from", "have", "been", "were",
    "will", "would", "could", "should", "about", "after", "before",
    "first", "last", "each", "every", "many", "much", "some", "more",
    "also", "just", "only", "very", "well", "back", "down", "here",
    "into", "over", "such", "than", "them", "then", "when", "what",
    "make", "like", "tell", "find", "give", "take", "want", "does",
}

# Session/CSRF key detection rules
_SESSION_KEY_RULES: dict[str, dict[str, str]] = {
    "form_key": {
        "method": "css_selector",
        "pattern": 'input[name="form_key"]',
        "attr": "value",
    },
    "_token": {
        "method": "css_selector",
        "pattern": 'input[name="_token"]',
        "attr": "value",
    },
    "csrf_token": {
        "method": "css_selector",
        "pattern": 'input[name="csrf_token"]',
        "attr": "value",
    },
    "csrf": {
        "method": "css_selector",
        "pattern": 'input[name="csrf"]',
        "attr": "value",
    },
    "authenticity_token": {
        "method": "css_selector",
        "pattern": 'input[name="authenticity_token"]',
        "attr": "value",
    },
    "X-CSRF-Token": {
        "method": "meta_tag",
        "pattern": 'meta[name="csrf-token"]',
        "attr": "content",
    },
}

_BASE64_KEY_NAMES = {
    "uenc", "redirect", "return_url", "referer_url",
    "login_redirect", "success_url", "error_url",
}

_SESSION_ID_KEY_NAMES = {
    "session_id", "sid", "phpsessid", "jsessionid",
    "session", "token",
}

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/_-]{16,}={0,3}~{0,2}$")
_CSRF_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{8,32}$")

# Lifecycle thresholds
TRUST_THRESHOLD = 3
STALE_THRESHOLD = 2


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

class SessionParam(BaseModel):
    """How to fetch a fresh session-managed value at execution time."""
    name: str
    extraction_method: str          # "regex" | "css_selector" | "meta_tag"
    extraction_pattern: str         # the regex or CSS selector
    source_page: str                # URL path to fetch for extraction


class HttpExecution(BaseModel):
    """Wire-level details for executing the tool as an HTTP request."""
    method: str
    url_template: str               # full URL with {param} placeholders
    query_template: dict[str, str] | None = None
    body_template: dict[str, Any] | list | str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    session_params: dict[str, SessionParam] = Field(
        default_factory=dict,
    )


class ToolMeta(BaseModel):
    """Lifecycle and provenance tracking for a tool."""
    discovered_from_task: str = ""
    task_patterns: list[str] = Field(default_factory=list)
    status: str = "new"             # new|trusted|untrusted|stale|deprecated
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0


class MCPToolDef(BaseModel):
    """A complete MCP tool definition: agent-facing schema + hidden
    execution details + lifecycle metadata."""
    name: str                       # snake_case verb_noun
    description: str                # one-line capability description
    inputSchema: dict               # JSON Schema for agent-facing params
    http: HttpExecution = Field(alias="_http")
    meta: ToolMeta = Field(alias="_meta")

    model_config = {"populate_by_name": True}


class ToolInvocation(BaseModel):
    """Record of a single tool execution for parameter analysis."""
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    timestamp: str = ""


class WorkflowChain(BaseModel):
    """A discovered sequence of tools that are commonly used together."""
    name: str
    steps: list[str]
    param_flow: dict[str, str] = Field(default_factory=dict)
    occurrences: int = 0


# ═══════════════════════════════════════════════════════════════════
# STEP A — Identify Causal Requests
# ═══════════════════════════════════════════════════════════════════

def find_causal_requests(
    captured_requests: list[CapturedRequest],
) -> list[CapturedRequest]:
    """Filter captured traffic down to requests that caused meaningful
    server-side effects or returned useful data."""
    seen_urls: set[str] = set()
    causal: list[CapturedRequest] = []

    for req in captured_requests:
        # Must have a successful response
        if req.response_status == 0:
            continue
        if not (200 <= req.response_status < 300):
            continue

        # Exclude static/infra paths
        path_lower = req.path.lower()
        if any(frag in path_lower for frag in EXCLUDED_PATH_FRAGMENTS):
            continue

        # State-changing methods are always causal
        is_causal = req.method in STATE_CHANGING_METHODS

        # GET with JSON body = data retrieval
        if (
            not is_causal
            and req.method == "GET"
            and isinstance(req.response_body, (dict, list))
        ):
            is_causal = True

        # GET with query params = search/filter page (even if HTML
        # body wasn't captured, the URL pattern is still valuable)
        if (
            not is_causal
            and req.method == "GET"
            and req.query_params
        ):
            is_causal = True

        # GET with HTML content-type — only causal if it has query params
        # (indicating a search/filter). Pure navigation GETs to HTML pages
        # don't make useful reusable tools.
        if (
            not is_causal
            and req.method == "GET"
            and req.query_params
        ):
            content_type = req.response_headers.get(
                "content-type", "",
            )
            if "text/html" in content_type:
                is_causal = True

        if not is_causal:
            continue

        # Deduplicate: same method+path combo keeps first only
        dedup_key = f"{req.method}:{req.path}"
        if dedup_key in seen_urls:
            continue
        seen_urls.add(dedup_key)

        causal.append(req)

    return causal


# ═══════════════════════════════════════════════════════════════════
# STEP B — Templatize Each Causal Request
# ═══════════════════════════════════════════════════════════════════

def _looks_like_id(value: str) -> bool:
    return any(pat.match(value) for pat in _ID_PATTERNS)


def _extract_task_values(task: str) -> set[str]:
    """Pull quoted strings and numbers from the task description."""
    values: set[str] = set()
    for m in re.finditer(r'["\']([^"\']+)["\']', task):
        values.add(m.group(1))
        values.add(m.group(1).lower())
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", task):
        values.add(m.group(1))
    # Also add individual words (lowercased, > 3 chars)
    for word in re.findall(r"\b[A-Za-z]{4,}\b", task):
        low = word.lower()
        if low not in _TASK_STOP_WORDS:
            values.add(low)
    return values


def _is_task_value(value: str, task_values: set[str]) -> bool:
    val_lower = value.lower().strip()
    if val_lower in _TASK_STOP_WORDS:
        return False
    return val_lower in task_values or value in task_values


# ── session detection ────────────────────────────────────────────

def _is_base64(value: str) -> bool:
    import base64 as b64
    if not _BASE64_RE.match(value):
        return False
    try:
        clean = value.rstrip("~")
        clean += "=" * (-len(clean) % 4)
        decoded = b64.urlsafe_b64decode(clean)
        printable = sum(1 for b in decoded if 32 <= b < 127)
        return printable > len(decoded) * 0.6
    except Exception:
        return False


def _classify_session_value(
    key: str,
    value: str,
    source_page: str,
) -> SessionParam | None:
    """If key/value is a session-dependent field, return a SessionParam
    describing how to extract a fresh value. Otherwise None."""
    key_lower = key.lower()
    lookup_key = key_lower if key_lower in _SESSION_KEY_RULES else key

    # 1) Known CSRF / token field names
    if lookup_key in _SESSION_KEY_RULES:
        rule = _SESSION_KEY_RULES[lookup_key]
        return SessionParam(
            name=key,
            extraction_method=rule["method"],
            extraction_pattern=rule["pattern"],
            source_page=source_page,
        )

    # 2) Base64-encoded redirect / context values
    if key_lower in _BASE64_KEY_NAMES:
        return SessionParam(
            name=key,
            extraction_method="regex",
            extraction_pattern=(
                rf'name="{re.escape(key)}"[^>]*value="([^"]+)"'
            ),
            source_page=source_page,
        )

    # 3) Known session ID key names
    if key_lower in _SESSION_ID_KEY_NAMES:
        return SessionParam(
            name=key,
            extraction_method="regex",
            extraction_pattern=(
                rf'name="{re.escape(key)}"[^>]*value="([^"]+)"'
            ),
            source_page=source_page,
        )

    # 4) Value looks like a CSRF token (mixed alphanumeric)
    if (
        isinstance(value, str)
        and _CSRF_TOKEN_RE.match(value)
        and key_lower not in _TASK_STOP_WORDS
        and not value.isdigit()
        and not value.isalpha()
        and not value.islower()
    ):
        return SessionParam(
            name=key,
            extraction_method="css_selector",
            extraction_pattern=f'input[name="{key}"]',
            source_page=source_page,
        )

    # 5) Value is base64-encoded regardless of key name
    if isinstance(value, str) and len(value) > 20 and _is_base64(value):
        return SessionParam(
            name=key,
            extraction_method="regex",
            extraction_pattern=(
                rf'name="{re.escape(key)}"[^>]*value="([^"]+)"'
            ),
            source_page=source_page,
        )

    return None


# ── path templatization ──────────────────────────────────────────

def _templatize_path(
    path: str,
    task_values: set[str],
) -> tuple[str, dict[str, str]]:
    """Replace dynamic path segments with {param} placeholders.
    Returns (templatized_path, {param_name: example_value})."""
    segments = path.split("/")
    result: list[str] = []
    path_params: dict[str, str] = {}

    for seg in segments:
        if not seg:
            result.append(seg)
            continue
        if seg.lower() in _PATH_STOP_WORDS:
            result.append(seg)
            continue
        if seg.isdigit() or _looks_like_id(seg):
            pname = f"path_id_{len(path_params)}"
            path_params[pname] = seg
            result.append(f"{{{pname}}}")
        elif _is_task_value(seg, task_values):
            pname = f"path_{seg.lower().replace('-', '_')}"
            path_params[pname] = seg
            result.append(f"{{{pname}}}")
        else:
            result.append(seg)

    return "/".join(result), path_params


# ── query templatization ─────────────────────────────────────────

def _templatize_query(
    query_params: dict[str, Any],
    task_values: set[str],
    source_page: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, SessionParam]]:
    """Returns (query_template, user_params, session_params)."""
    query_template: dict[str, str] = {}
    user_params: dict[str, str] = {}
    session_params: dict[str, SessionParam] = {}

    for key, vals in query_params.items():
        val = vals[0] if isinstance(vals, list) and vals else str(vals)

        # Check if it's a session value
        if isinstance(val, str):
            sp = _classify_session_value(key, val, source_page)
            if sp:
                session_params[key] = sp
                query_template[key] = f"{{{key}}}"
                continue

        # Check if it's a task-provided value
        if isinstance(val, str) and _is_task_value(val, task_values):
            user_params[key] = val
            query_template[key] = f"{{{key}}}"
        else:
            query_template[key] = str(val)

    return query_template, user_params, session_params


# ── body templatization ──────────────────────────────────────────

def _templatize_body(
    body: Any,
    task_values: set[str],
    source_page: str,
) -> tuple[
    dict[str, Any] | list | str | None,
    dict[str, Any],
    dict[str, SessionParam],
]:
    """Returns (body_template, user_params, session_params)."""
    if body is None:
        return None, {}, {}

    if isinstance(body, str):
        # Try to parse form data
        form = _parse_form_data(body)
        if form:
            return _templatize_body(form, task_values, source_page)
        return body, {}, {}

    # Handle list bodies (e.g. GraphQL batch requests)
    if isinstance(body, list):
        if len(body) == 1 and isinstance(body[0], dict):
            # Single-element list → unwrap and templatize the dict
            return _templatize_body(body[0], task_values, source_page)
        # Multi-element list → keep as-is (not parameterizable)
        return body, {}, {}

    if not isinstance(body, dict):
        return body, {}, {}

    template: dict[str, Any] = {}
    user_params: dict[str, Any] = {}
    session_params: dict[str, SessionParam] = {}

    for key, value in body.items():
        if isinstance(value, dict):
            sub_t, sub_u, sub_s = _templatize_body(
                value, task_values, source_page,
            )
            template[key] = sub_t
            user_params.update(sub_u)
            session_params.update(sub_s)
            continue

        str_val = str(value)

        # Session value?
        if isinstance(value, str):
            sp = _classify_session_value(key, value, source_page)
            if sp:
                session_params[key] = sp
                template[key] = f"{{{key}}}"
                continue

        # Task value?
        if _is_task_value(str_val, task_values):
            user_params[key] = value
            template[key] = f"{{{key}}}"
        elif isinstance(value, (int, float)) and key not in (
            "store_id", "website_id",
        ):
            user_params[key] = value
            template[key] = f"{{{key}}}"
        else:
            template[key] = value

    return template, user_params, session_params


def _parse_form_data(body: str) -> dict[str, str] | None:
    if "Content-Disposition: form-data" not in body:
        return None
    fields: dict[str, str] = {}
    for m in re.finditer(
        r'Content-Disposition: form-data; name="([^"]+)"'
        r"\r?\n\r?\n([^\r\n-]+)",
        body,
    ):
        fields[m.group(1)] = m.group(2).strip()
    return fields or None


# ═══════════════════════════════════════════════════════════════════
# STEP C — Generate Semantic Name + Description via Gemini
# ═══════════════════════════════════════════════════════════════════

def _generate_name_fallback(method: str, path: str) -> str:
    """Mechanical fallback: method + last 2 meaningful path segments."""
    segments = [
        seg for seg in path.strip("/").split("/")
        if seg and not seg.startswith("{") and not seg.isdigit()
    ]
    slug = "_".join(segments[-2:]).replace("-", "_").lower()
    slug = re.sub(r"_+", "_", slug).strip("_")
    name = f"{method.lower()}_{slug}" if slug else method.lower()
    return name[:60]


def _generate_tool_definition_via_gemini(
    method: str,
    path: str,
    query_param_names: list[str],
    body_field_names: list[str],
    response_status: int,
    response_body_preview: str,
    task_description: str,
) -> dict[str, Any] | None:
    """Call Gemini Flash to generate name, description, inputSchema."""
    # Tell Gemini the method matters for accurate naming
    method_guidance = ""
    if method == "GET":
        method_guidance = (
            "This is a GET request — it READS data. The tool name "
            "MUST use a read verb: get_, list_, search_, fetch_, "
            "find_, count_. Do NOT use create/add/update/delete/post "
            "verbs for GET requests.\n"
        )
    elif method in ("POST", "PUT", "PATCH"):
        method_guidance = (
            f"This is a {method} request — it WRITES/MODIFIES data. "
            "Use an appropriate write verb: create_, add_, update_, "
            "post_, submit_, set_, change_.\n"
        )
    elif method == "DELETE":
        method_guidance = (
            "This is a DELETE request. Use: delete_, remove_.\n"
        )

    prompt = (
        "You are generating an MCP tool definition from an HTTP "
        "request observed during browser automation.\n\n"
        f"HTTP request:\n"
        f"Method: {method}\n"
        f"Path: {path}\n"
        f"Query params: {query_param_names}\n"
        f"Body fields: {body_field_names}\n"
        f"Response: {response_status} "
        f"{response_body_preview[:200]}\n"
        f"Task context: '{task_description}'\n\n"
        f"{method_guidance}\n"
        "Generate JSON with these fields:\n"
        "- name: snake_case verb_noun "
        "(e.g. search_products, add_to_cart)\n"
        "- description: One sentence describing what this tool does. "
        "Include the HTTP method in the description.\n"
        "- inputSchema: JSON Schema with only USER-facing parameters "
        "(search terms, product names, quantities, text content). "
        "Exclude internal IDs, tokens, session values.\n\n"
        "Return ONLY valid JSON. No markdown fences."
    )

    try:
        resp = _gemini.models.generate_content(
            model=_NAMING_MODEL,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        if not resp.candidates:
            return None

        text = resp.text.strip()
        # Strip markdown fences if present despite instructions
        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*\n?", "", text,
            )
            text = re.sub(r"\n?```\s*$", "", text)

        return json.loads(text)
    except (APIError, json.JSONDecodeError, Exception) as exc:
        print(f"  Gemini naming failed ({exc}), using fallback")
        return None


def _build_input_schema(
    user_params: dict[str, Any],
    path_params: dict[str, str],
) -> dict[str, Any]:
    """Build a JSON Schema from discovered user-facing parameters."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, example in path_params.items():
        properties[name] = {
            "type": "string",
            "description": f"Path parameter (example: {example})",
        }
        required.append(name)

    for name, example in user_params.items():
        if isinstance(example, int):
            prop = {"type": "integer"}
        elif isinstance(example, float):
            prop = {"type": "number"}
        else:
            prop = {"type": "string"}
        prop["description"] = f"Example: {example}"
        properties[name] = prop
        required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ═══════════════════════════════════════════════════════════════════
# STEP D — Deduplication
# ═══════════════════════════════════════════════════════════════════

def _path_pattern(url_template: str) -> str:
    """Normalize a URL template to a comparable pattern by stripping
    param placeholders from path segments."""
    parsed = urlparse(url_template)
    segments = [
        seg for seg in parsed.path.split("/")
        if seg and not seg.startswith("{")
    ]
    return "/".join(segments)


def _find_duplicate(
    new_tool: MCPToolDef,
    existing: dict[str, MCPToolDef],
) -> str | None:
    """If an existing tool matches the same method + path pattern,
    return its name. Otherwise None."""
    new_pattern = _path_pattern(new_tool.http.url_template)
    new_method = new_tool.http.method

    for name, tool in existing.items():
        if tool.http.method != new_method:
            continue
        if _path_pattern(tool.http.url_template) == new_pattern:
            return name
    return None


# ═══════════════════════════════════════════════════════════════════
# STEP D2 — Validation at Discovery Time
# ═══════════════════════════════════════════════════════════════════


def _validate_tool(
    tool: MCPToolDef,
    original_req: CapturedRequest,
) -> bool:
    """Replay the original HTTP request to validate the tool endpoint.
    Uses the original request's headers (including cookies)."""
    try:
        headers = {
            h: v for h, v in original_req.headers.items()
            if h.lower() not in {
                "host", "content-length", "transfer-encoding",
            }
        }
        with httpx.Client(
            headers=headers, follow_redirects=True,
        ) as client:
            content = None
            if original_req.body is not None:
                if isinstance(original_req.body, (dict, list)):
                    content = json.dumps(original_req.body)
                else:
                    content = str(original_req.body)

            resp = client.request(
                method=original_req.method,
                url=original_req.url,
                content=content,
                timeout=15.0,
            )
            valid = 200 <= resp.status_code < 400
            if valid:
                print(
                    f"    Validated: {original_req.method} "
                    f"{original_req.path} -> {resp.status_code}"
                )
            else:
                print(
                    f"    Validation failed: {original_req.method} "
                    f"{original_req.path} -> {resp.status_code}"
                )
            return valid
    except Exception as exc:
        print(f"    Validation error for {tool.name}: {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════
# STEP E — Tool Execution
# ═══════════════════════════════════════════════════════════════════

def _extract_via_css_selector(
    html: str,
    selector: str,
    attr: str = "value",
) -> str | None:
    """Lightweight CSS-selector extraction using regex.
    Supports patterns like: input[name="form_key"], meta[name="x"]."""
    sel_match = re.match(r'(\w+)\[(\w+)="([^"]+)"\]', selector)
    if not sel_match:
        return None

    tag, sel_attr, sel_val = (
        sel_match.group(1),
        sel_match.group(2),
        sel_match.group(3),
    )

    # Try both attribute orderings
    for pattern in [
        (
            rf"<{tag}\s[^>]*?"
            rf'{sel_attr}="{re.escape(sel_val)}"'
            rf'[^>]*?{attr}="([^"]*)"'
        ),
        (
            rf"<{tag}\s[^>]*?"
            rf'{attr}="([^"]*)"'
            rf'[^>]*?{sel_attr}="{re.escape(sel_val)}"'
        ),
    ]:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            return m.group(1)

    return None


def _fetch_session_value(
    sp: SessionParam,
    http_client: httpx.Client,
) -> str | None:
    """GET the source page and extract a fresh session value."""
    try:
        resp = http_client.get(sp.source_page, timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None

    html = resp.text

    if sp.extraction_method in ("css_selector", "meta_tag"):
        val = _extract_via_css_selector(html, sp.extraction_pattern)
        if val:
            return val
        # Fall through to regex on the raw selector pattern
        fallback = re.search(
            rf'name="{re.escape(sp.name)}"[^>]*value="([^"]+)"',
            html,
        )
        if fallback:
            return fallback.group(1)

    if sp.extraction_method == "regex":
        m = re.search(sp.extraction_pattern, html)
        if m:
            return m.group(1)

    return None


async def execute_tool(
    site_name: str,
    tool_name: str,
    params: dict[str, Any],
    page: Page | None = None,
) -> dict[str, Any]:
    """Execute an MCP tool by making the underlying HTTP request.

    1. Load tool from mcps/{site}/tools.json
    2. Attach browser cookies if page is available
    3. Resolve session params (CSRF tokens etc.)
    4. Substitute all params and make the request
    5. Return {status, body, success}
    """
    tools = load_tools(site_name)
    if tool_name not in tools:
        return {
            "status": 0,
            "body": f"Unknown tool: {tool_name}",
            "success": False,
        }

    tool = tools[tool_name]
    http = tool.http
    merged = dict(params)

    # Attach browser cookies
    extra_headers: dict[str, str] = {}
    if page is not None:
        extra_headers = await get_browser_cookies(page)

    # Resolve session params
    base_headers = {**http.headers, **extra_headers}
    client = httpx.Client(
        headers=base_headers,
        follow_redirects=True,
    )
    try:
        for sp_name, sp in http.session_params.items():
            if sp_name in merged:
                continue
            fresh = _fetch_session_value(sp, client)
            if fresh:
                merged[sp_name] = fresh
            else:
                print(f"  Session param {sp_name}: could not fetch")

        # Substitute into URL template
        url = http.url_template
        for key, val in merged.items():
            url = url.replace(f"{{{key}}}", str(val))

        # Substitute into query
        query: dict[str, str] | None = None
        if http.query_template:
            query = {}
            for k, v in http.query_template.items():
                if v.startswith("{") and v.endswith("}"):
                    pname = v[1:-1]
                    query[k] = str(merged.get(pname, v))
                else:
                    query[k] = v

        # Substitute into body
        content: str | None = None
        if http.body_template is not None:
            body_str = json.dumps(http.body_template)
            for key, val in merged.items():
                body_str = body_str.replace(
                    f"{{{key}}}", str(val),
                )
            content = body_str
            base_headers.setdefault(
                "content-type", "application/json",
            )

        resp = client.request(
            method=http.method,
            url=url,
            params=query,
            content=content,
            timeout=30.0,
        )

        # Parse response
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:2000] if resp.text else None

        success = 200 <= resp.status_code < 300
        log_invocation(site_name, tool_name, merged, success)

        # Build CapturedRequest so NetworkEventEvaluator can see MCP traffic
        actual_parsed = urlparse(str(resp.url))
        captured = CapturedRequest(
            timestamp=time.time(),
            method=http.method,
            url=str(resp.url),
            path=actual_parsed.path,
            query_params=dict(parse_qs(actual_parsed.query, keep_blank_values=True)),
            headers={k.lower(): v for k, v in resp.request.headers.items()},
            body=json.loads(content) if content else None,
            response_status=resp.status_code,
            response_body=body,
            response_headers={k.lower(): v for k, v in resp.headers.items()},
        )

        return {
            "status": resp.status_code,
            "body": body,
            "success": success,
            "captured_request": captured,
        }
    finally:
        client.close()


# ═══════════════════════════════════════════════════════════════════
# STEP F — Visual Verification
# ═══════════════════════════════════════════════════════════════════

async def verify_tool_result(
    page: Page,
    tool_name: str,
    tool_description: str,
    params: dict[str, Any],
    http_result: dict[str, Any],
) -> bool:
    """Take a screenshot and ask Gemini Flash whether the action
    succeeded based on the visible page state."""
    if not http_result.get("success"):
        return False

    from .computer_use import _png_to_jpeg, MEDIA_RES_LOW
    screenshot_png = await page.screenshot(type="png")
    screenshot_bytes = _png_to_jpeg(screenshot_png, quality=50)

    prompt = (
        f"An MCP tool '{tool_name}' was executed: "
        f"{tool_description}\n"
        f"Parameters: {params}\n"
        f"HTTP response status: {http_result['status']}\n\n"
        "Look at this screenshot of the current browser page. "
        "Does the page state confirm the action succeeded? "
        "Reply ONLY 'yes' or 'no'."
    )

    try:
        resp = _gemini.models.generate_content(
            model=_NAMING_MODEL,
            contents=[genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(text=prompt),
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            mime_type="image/jpeg",
                            data=screenshot_bytes,
                        ),
                        media_resolution=MEDIA_RES_LOW,
                    ),
                ],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8,
            ),
        )
        answer = resp.text.strip().lower()
        return answer.startswith("yes")
    except Exception as exc:
        print(f"  Verification failed ({exc}), assuming success")
        return True


# ═══════════════════════════════════════════════════════════════════
# STEP G — Tool Lifecycle Management
# ═══════════════════════════════════════════════════════════════════

def update_tool_status(
    site_name: str,
    tool_name: str,
    success: bool,
) -> None:
    """Update a tool's lifecycle status based on execution outcome."""
    tools = load_tools(site_name)
    if tool_name not in tools:
        return

    tool = tools[tool_name]
    meta = tool.meta

    if success:
        meta.success_count += 1
        meta.consecutive_failures = 0
        if (
            meta.status in ("new", "validated")
            and meta.success_count >= TRUST_THRESHOLD
        ):
            meta.status = "trusted"
        elif meta.status == "untrusted":
            if meta.success_count >= TRUST_THRESHOLD:
                meta.status = "trusted"
    else:
        meta.failure_count += 1
        meta.consecutive_failures += 1
        if meta.consecutive_failures >= STALE_THRESHOLD:
            meta.status = "stale"

    save_tools(site_name, tools)


# ═══════════════════════════════════════════════════════════════════
# Storage — per-site tools.json
# ═══════════════════════════════════════════════════════════════════

def _tools_path(site_name: str) -> Path:
    d = config.MCPS_DIR / site_name
    d.mkdir(parents=True, exist_ok=True)
    return d / "tools.json"


def load_tools(site_name: str) -> dict[str, MCPToolDef]:
    """Load all tools for a site from mcps/{site}/tools.json."""
    path = _tools_path(site_name)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    result: dict[str, MCPToolDef] = {}
    for name, data in raw.items():
        result[name] = MCPToolDef(**data)
    return result


def save_tools(
    site_name: str,
    tools: dict[str, MCPToolDef],
) -> None:
    """Persist all tools for a site to mcps/{site}/tools.json."""
    path = _tools_path(site_name)
    data = {
        name: tool.model_dump(by_alias=True)
        for name, tool in tools.items()
    }
    path.write_text(json.dumps(data, indent=2, default=str))


# ── Invocation log ────────────────────────────────────────────────

def _invocations_path(site_name: str) -> Path:
    d = config.MCPS_DIR / site_name
    d.mkdir(parents=True, exist_ok=True)
    return d / "invocations.json"


def load_invocations(site_name: str) -> list[ToolInvocation]:
    path = _invocations_path(site_name)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [ToolInvocation(**r) for r in raw]


def save_invocations(
    site_name: str,
    invocations: list[ToolInvocation],
) -> None:
    path = _invocations_path(site_name)
    path.write_text(json.dumps(
        [inv.model_dump() for inv in invocations],
        indent=2, default=str,
    ))


def log_invocation(
    site_name: str,
    tool_name: str,
    params: dict[str, Any],
    success: bool,
) -> None:
    """Append an invocation record for later optimization analysis."""
    invocations = load_invocations(site_name)
    invocations.append(ToolInvocation(
        tool_name=tool_name,
        params=params,
        success=success,
        timestamp=datetime.now(timezone.utc).isoformat(),
    ))
    save_invocations(site_name, invocations)


# ── Workflow storage ──────────────────────────────────────────────

def _workflows_path(site_name: str) -> Path:
    d = config.MCPS_DIR / site_name
    d.mkdir(parents=True, exist_ok=True)
    return d / "workflows.json"


def load_workflows(site_name: str) -> list[WorkflowChain]:
    path = _workflows_path(site_name)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [WorkflowChain(**w) for w in raw]


def save_workflows(
    site_name: str,
    workflows: list[WorkflowChain],
) -> None:
    path = _workflows_path(site_name)
    path.write_text(json.dumps(
        [w.model_dump() for w in workflows],
        indent=2, default=str,
    ))


def get_tools_for_agent(site_name: str) -> list[dict[str, Any]]:
    """Return only agent-facing fields (name, description, inputSchema)
    for non-stale/non-deprecated tools. Sorted: trusted > validated > new."""
    tools = load_tools(site_name)
    _STATUS_PRIORITY = {
        "trusted": 0, "validated": 1, "new": 2, "untrusted": 3,
    }
    result: list[tuple[int, dict[str, Any]]] = []
    for tool in tools.values():
        if tool.meta.status in ("stale", "deprecated"):
            continue
        priority = _STATUS_PRIORITY.get(tool.meta.status, 9)
        result.append((priority, {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
            "http_method": tool.http.method,
        }))
    result.sort(key=lambda t: t[0])
    return [t[1] for t in result]


# ═══════════════════════════════════════════════════════════════════
# Main entry point — discover_tools
# ═══════════════════════════════════════════════════════════════════

def discover_tools(
    captured_requests: list[CapturedRequest],
    _action_log: list,
    task_description: str,
    site_name: str,
) -> list[MCPToolDef]:
    """Full pipeline: filter causal requests → templatize → name via
    Gemini → deduplicate → persist. Returns newly added tools."""
    existing = load_tools(site_name)
    task_values = _extract_task_values(task_description)
    causal = find_causal_requests(captured_requests)
    new_tools: list[MCPToolDef] = []

    for req in causal:
        parsed = urlparse(req.url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        referer = req.headers.get("referer", base_url)

        # B1: Path params
        templ_path, path_params = _templatize_path(
            req.path, task_values,
        )

        # B2: Query params
        qt, query_user, query_session = _templatize_query(
            req.query_params, task_values, referer,
        )

        # B3: Body params
        bt, body_user, body_session = _templatize_body(
            req.body, task_values, referer,
        )

        # Merge all session params
        all_session: dict[str, SessionParam] = {
            **query_session,
            **body_session,
        }

        # Merge all user params
        all_user: dict[str, Any] = {
            **path_params,
            **query_user,
            **body_user,
        }

        # Build URL template
        url_template = f"{base_url}{templ_path}"

        # Build HttpExecution
        http = HttpExecution(
            method=req.method,
            url_template=url_template,
            query_template=qt if qt else None,
            body_template=bt,
            headers={
                h: v for h, v in req.headers.items()
                if h.lower() in {
                    "authorization", "cookie",
                    "x-csrf-token", "x-requested-with",
                    "x-magento-csrf-token",
                }
            },
            session_params=all_session,
        )

        # Build inputSchema
        input_schema = _build_input_schema(all_user, path_params)

        # Generate response body preview for Gemini
        resp_preview = ""
        if req.response_body is not None:
            if isinstance(req.response_body, (dict, list)):
                resp_preview = json.dumps(
                    req.response_body, default=str,
                )[:200]
            else:
                resp_preview = str(req.response_body)[:200]

        # C: Semantic naming via Gemini
        gemini_def = _generate_tool_definition_via_gemini(
            method=req.method,
            path=req.path,
            query_param_names=list(
                (qt or {}).keys(),
            ),
            body_field_names=list(
                (bt if isinstance(bt, dict) else {}).keys(),
            ),
            response_status=req.response_status,
            response_body_preview=resp_preview,
            task_description=task_description,
        )

        if gemini_def and isinstance(gemini_def, dict):
            name = gemini_def.get(
                "name",
                _generate_name_fallback(req.method, req.path),
            )
            description = gemini_def.get(
                "description",
                f"{req.method} {req.path}",
            )
            # Use Gemini's inputSchema if it looks valid
            gemini_schema = gemini_def.get("inputSchema")
            if (
                isinstance(gemini_schema, dict)
                and "properties" in gemini_schema
            ):
                input_schema = gemini_schema
        else:
            name = _generate_name_fallback(req.method, req.path)
            description = f"{req.method} {req.path}"

        meta = ToolMeta(
            discovered_from_task=task_description,
            task_patterns=[task_description],
        )

        tool = MCPToolDef(
            name=name,
            description=description,
            inputSchema=input_schema,
            **{"_http": http, "_meta": meta},
        )

        # Validate by replaying the original request
        if _validate_tool(tool, req):
            meta.status = "validated"
        else:
            print(f"  Tool '{name}' failed validation, skipping")
            continue

        # D: Deduplication
        dup_name = _find_duplicate(tool, existing)
        if dup_name:
            dup = existing[dup_name]
            if dup.meta.status == "trusted":
                # Keep the trusted one, just record the new task
                if (
                    task_description
                    not in dup.meta.task_patterns
                ):
                    dup.meta.task_patterns.append(
                        task_description,
                    )
                continue
            if dup.meta.status == "stale":
                # Replace stale tool
                dup.meta.status = "deprecated"
                existing[dup_name] = dup

        existing[name] = tool
        new_tools.append(tool)

    save_tools(site_name, existing)
    return new_tools


# ═══════════════════════════════════════════════════════════════════
# Library Optimization — optimize_library()
# ═══════════════════════════════════════════════════════════════════

def _merge_duplicates(tools: dict[str, MCPToolDef]) -> int:
    """Merge tools with same method + base path (ignoring params).
    Keep the one with higher success_count, merge task_patterns."""
    groups: dict[str, list[str]] = {}
    for name, tool in tools.items():
        if tool.meta.status == "deprecated":
            continue
        key = f"{tool.http.method}:{_path_pattern(tool.http.url_template)}"
        groups.setdefault(key, []).append(name)

    merged_count = 0
    for names in groups.values():
        if len(names) < 2:
            continue
        best_name = max(
            names, key=lambda n: tools[n].meta.success_count,
        )
        best = tools[best_name]
        for name in names:
            if name == best_name:
                continue
            loser = tools[name]
            for tp in loser.meta.task_patterns:
                if tp not in best.meta.task_patterns:
                    best.meta.task_patterns.append(tp)
            loser.meta.status = "deprecated"
            merged_count += 1
            print(f"  Merged '{name}' into '{best_name}'")

    return merged_count


def _generalize_parameters(
    site_name: str,
    tools: dict[str, MCPToolDef],
) -> int:
    """Analyze invocation logs. If a param always had the same value
    across 2+ successful calls, hardcode it in the HTTP template and
    remove from inputSchema."""
    invocations = load_invocations(site_name)
    if not invocations:
        return 0

    by_tool: dict[str, list[dict[str, Any]]] = {}
    for inv in invocations:
        if inv.success:
            by_tool.setdefault(inv.tool_name, []).append(inv.params)

    generalized = 0
    for tool_name, param_sets in by_tool.items():
        if tool_name not in tools or len(param_sets) < 2:
            continue
        tool = tools[tool_name]
        if tool.meta.status == "deprecated":
            continue
        props = tool.inputSchema.get("properties", {})

        for param_name in list(props.keys()):
            values = [
                ps[param_name]
                for ps in param_sets
                if param_name in ps
            ]
            if len(values) < 2:
                continue
            unique = set(str(v) for v in values)
            if len(unique) != 1:
                continue  # varied — stays parameterized

            constant = values[0]
            # Hardcode in the appropriate HTTP template
            if tool.http.query_template and param_name in tool.http.query_template:
                tool.http.query_template[param_name] = str(constant)
            elif isinstance(tool.http.body_template, dict) and param_name in tool.http.body_template:
                tool.http.body_template[param_name] = constant
            # For path params, substitute directly in url_template
            elif f"{{{param_name}}}" in tool.http.url_template:
                tool.http.url_template = tool.http.url_template.replace(
                    f"{{{param_name}}}", str(constant),
                )

            # Remove from inputSchema
            del props[param_name]
            req = tool.inputSchema.get("required", [])
            if param_name in req:
                req.remove(param_name)

            generalized += 1
            print(
                f"  Hardcoded {param_name}={constant!r} "
                f"in '{tool_name}'"
            )

    return generalized


def _build_workflow_chains(
    site_name: str,
    tools: dict[str, MCPToolDef],
) -> int:
    """Analyze results/*.jsonl for sequential MCP tool usage.
    If tool A precedes tool B, record as a workflow chain.
    Save to mcps/{site}/workflows.json."""
    result_files = sorted(config.RESULTS_DIR.glob("run_*.jsonl"))

    sequences: list[list[str]] = []
    for rf in result_files:
        for line in rf.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                task_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if task_data.get("site") != site_name:
                continue
            mcp_tools = [
                s["tool_name"]
                for s in task_data.get("steps", [])
                if s.get("method") == "mcp" and s.get("tool_name")
            ]
            if len(mcp_tools) >= 2:
                sequences.append(mcp_tools)

    # Count adjacent pairs
    pair_counts: dict[tuple[str, str], int] = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    existing = load_workflows(site_name)
    existing_names = {w.name for w in existing}
    new_count = 0

    for (a, b), count in pair_counts.items():
        if a not in tools or b not in tools:
            continue
        if (
            tools[a].meta.status == "deprecated"
            or tools[b].meta.status == "deprecated"
        ):
            continue

        # Concise name from verb parts, fall back to full names
        a_verb = a.split("_")[0]
        b_verb = b.split("_")[0]
        name = f"{a_verb}_and_{b_verb}"
        if name in existing_names:
            # Check if it's the same pair — update count
            for w in existing:
                if w.steps == [a, b]:
                    w.occurrences = count
            continue
            # Different pair, same name — use full names
            name = f"{a}_then_{b}"  # pragma: no cover
        if name in existing_names:
            continue

        # Infer param_flow from B's input schema
        param_flow: dict[str, str] = {}
        for pname in tools[b].inputSchema.get("properties", {}):
            param_flow[f"{a}.result.{pname}"] = f"{b}.{pname}"

        existing.append(WorkflowChain(
            name=name,
            steps=[a, b],
            param_flow=param_flow,
            occurrences=count,
        ))
        existing_names.add(name)
        new_count += 1
        print(f"  Workflow: {a} -> {b} ({count}x)")

    save_workflows(site_name, existing)
    return new_count


def _prune_dead_tools(tools: dict[str, MCPToolDef]) -> int:
    """Remove: status=='deprecated', or
    (success_count==0 and failure_count>=3)."""
    to_remove: list[str] = []
    for name, tool in tools.items():
        if tool.meta.status == "deprecated":
            to_remove.append(name)
        elif (
            tool.meta.success_count == 0
            and tool.meta.failure_count >= 3
        ):
            to_remove.append(name)

    for name in to_remove:
        t = tools[name]
        print(
            f"  Pruned '{name}' "
            f"(status={t.meta.status}, "
            f"ok={t.meta.success_count}, "
            f"fail={t.meta.failure_count})"
        )
        del tools[name]

    return len(to_remove)


def optimize_library(site_name: str) -> dict[str, int]:
    """Optimize the MCP tool library for a site.

    1. Merge duplicates (same method + base path)
    2. Generalize parameters (hardcode constants)
    3. Build workflow chains from result history
    4. Prune dead tools

    Call after every 10 tasks or manually.
    """
    tools = load_tools(site_name)
    if not tools:
        print(f"Opzed {site_name}: no tools to optimize")
        return {
            "merged": 0, "pruned": 0,
            "workflows": 0, "total": 0,
        }

    merged = _merge_duplicates(tools)
    generalized = _generalize_parameters(site_name, tools)
    workflows = _build_workflow_chains(site_name, tools)
    pruned = _prune_dead_tools(tools)

    save_tools(site_name, tools)

    active = sum(
        1 for t in tools.values()
        if t.meta.status != "deprecated"
    )

    print(
        f"Opzed {site_name}: {merged} merged, {pruned} pruned, "
        f"{workflows} workflows found. {active} active tools."
    )

    return {
        "merged": merged,
        "generalized": generalized,
        "pruned": pruned,
        "workflows": workflows,
        "total": active,
    }
