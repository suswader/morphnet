"""
mcp_manager.py — API Tool Discovery, Execution, and Lifecycle Management.

Three jobs:
1. DISCOVER: After successful CU subtask, analyze captured HTTP traffic
   for replayable API patterns. Trace parameter values to browser state sources.
2. EXECUTE: Generate parameters from current browser state using source hints,
   execute via curl_cffi.
3. MANAGE: Store tool definitions, evolve schemas across observations,
   validate and lifecycle-manage tools.

Discovery is triggered by the orchestrator after a successful CU subtask.
Execution is triggered when the orchestrator routes a subtask to an MCP tool.
No validation replay at discovery — some APIs are not idempotent.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Any

from morphnet.session_manager import (
    SessionManager, CapturedRequest, call_gemini,
)
from morphnet.reflector import Reflector
from morphnet.trace import TaskTrace, Evidence

logger = logging.getLogger(__name__)
SITES_DIR = Path(__file__).parent / "sites"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParameterSource:
    """Where a parameter value was found in the browser state.

    Stored as dicts inside MCPToolDefinition.parameter_sources for JSON
    serialization. This dataclass documents the expected shape.
    """
    source_type: str    # cookie | dom_element | dom_attribute | local_storage |
                        # session_storage | prior_api_response | url_segment |
                        # url_query_param | meta_tag | form_hidden_field
    source_key: str     # Cookie name, CSS selector, JSON path, storage key
    source_url: str     # Which page/API response it came from
    confidence: float   # 1.0 = exact match found in registry
    example_value: str  # The value we observed (truncated to 100 chars)


@dataclass
class MCPToolDefinition:
    """A discovered API tool, ready for execution.

    Serialized to/from sites/{site_name}/tools.json via asdict/constructor.
    All fields must be JSON-serializable primitives (no custom objects).
    """
    name: str
    description: str
    endpoint_identity: str

    method: str                 # GET, POST, PUT, DELETE, PATCH
    url_template: str           # Concrete on first observation, parameterized after 2+
    protocol: str               # rest | graphql | jsonrpc | form | multipart

    parameter_schema: dict      # JSON Schema inferred from request bodies
    header_template: dict       # Headers to replay (excludes auto-managed)
    parameter_sources: dict     # JSON path → list[ParameterSource-shaped dicts]

    examples: list[dict]        # Recent request/response pairs (max 5)

    status: str                 # verified | trusted | degraded | discarded
    observation_count: int
    is_safe_to_replay: bool     # Gemini-determined: idempotent or not
    created_at: float
    last_used_at: float

    # All observed URLs — for frequency-based URL template evolution
    observed_urls: list[str] = field(default_factory=list)

    # Per-tool extraction recipe: list of ExtractionStep-shaped dicts
    # Built from traced sources at discovery. Executed by representation.py.
    extraction_recipe: list[dict] = field(default_factory=list)

    # Response structure template — learned from successful responses.
    # Used by reflector for deterministic structural checks.
    response_template: dict = field(default_factory=dict)

    # GraphQL-specific
    graphql_query: str | None = None
    graphql_variables_schema: dict | None = None


@dataclass
class ExtractionStep:
    """HOW to extract one parameter from the browser/session state at execution time.

    Built at discovery from traced ParameterSources. Used at execution by
    build_mcp_parameter_context() in representation.py.
    Serialized as dicts inside MCPToolDefinition.extraction_recipe.

    source_type determines the extraction mechanism:
    - "cookie": config={key} → extract cookie value by name
    - "dom_field": config={selector} → extract one element's value
    - "dom_list": config={selector, value_attr, label_attr} → ALL matching
      elements as options the parameter generator picks from
    - "storage": config={storage_type, key} → localStorage/sessionStorage
    - "meta_tag": config={selector} → meta tag content attribute
    - "url_component": config={component, index/key} → URL path/query
    - "prior_api_response": config={endpoint_identity, json_path} → navigates
      JSON in the most recent response from that endpoint. Checks MCP response
      cache first, then browser captured traffic.
    - "task_description": config={} → nothing to extract, LLM derives from task

    classification tells the parameter generator what kind of value this is:
    - "user_intent": free-form from task (search queries, addresses)
    - "ephemeral": changes every session, extract fresh (CSRF, auth tokens)
    - "chained": from a prior API response (IDs from upstream calls)
    - "page_context": from current page DOM (product IDs, data attributes)
    - "static": same every time (API version, content type flags)
    """
    param_path: str         # JSON path in request body, e.g. "$.place_id"
    source_type: str        # cookie | dom_field | dom_list | storage | meta_tag |
                            # url_component | prior_api_response | task_description
    source_config: dict     # Source-type-specific extraction config
    classification: str     # user_intent | ephemeral | chained | page_context | static
    description: str        # Human-readable, e.g. "CSRF token from cookie"


@dataclass
class SourceRecord:
    """Internal: one entry in the ValueSourceRegistry."""
    value: str
    source_type: str
    source_key: str
    source_url: str = ""
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Gemini Schemas
# ---------------------------------------------------------------------------

TOOL_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Short descriptive function name in snake_case. "
                "Examples: 'add_to_cart', 'search_restaurants', "
                "'update_delivery_address'. Describes WHAT the tool does."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "3-4 sentences: what this tool does, when to use it, "
                "what it returns, important constraints."
            ),
        },
        "is_safe_to_replay": {
            "type": "boolean",
            "description": (
                "Can this be called multiple times safely? True for idempotent "
                "operations (search, get). False for state-creating operations "
                "(add to cart, place order, make payment)."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "Why you chose this name and description.",
        },
    },
    "required": ["name", "description", "is_safe_to_replay", "reasoning"],
}


def build_param_generation_schema(tool: MCPToolDefinition) -> dict:
    """Build a Gemini response schema specific to THIS tool's parameters.

    Every parameter field is typed and described based on observations
    and source hints — not a generic {"parameters": "object"}.
    """
    param_properties: dict[str, dict] = {}
    param_required: list[str] = []

    for field_name, field_schema in tool.parameter_schema.get("properties", {}).items():
        prop = dict(field_schema)
        # Strip x-examples from schema sent to Gemini (internal tracking only)
        prop.pop("x-examples", None)

        # Enrich with source hints
        sources_key = f"$.{field_name}"
        source_list = tool.parameter_sources.get(sources_key, [])
        if source_list and isinstance(source_list[0], dict):
            src = source_list[0]
            prop["description"] = (
                f"{prop.get('description', field_name)}. "
                f"Previously found in {src['source_type']} at "
                f"'{src['source_key']}'. "
                f"Example: '{str(src.get('example_value', ''))[:50]}'"
            )
        else:
            # Use examples for hints
            example_values: set[str] = set()
            for ex in tool.examples[-3:]:
                body = ex.get("request_body") if isinstance(ex, dict) else None
                if body and field_name in body:
                    example_values.add(str(body[field_name])[:50])
            if example_values:
                prop["description"] = (
                    f"{prop.get('description', field_name)}. "
                    f"Examples: {', '.join(list(example_values)[:3])}"
                )

        param_properties[field_name] = prop

    if tool.parameter_schema.get("required"):
        param_required = tool.parameter_schema["required"]

    return {
        "type": "object",
        "properties": {
            "parameters": {
                "type": "object",
                "description": (
                    f"Request body for {tool.name} "
                    f"({tool.method} {tool.url_template})"
                ),
                "properties": param_properties,
                "required": param_required,
            },
            "url_params": {
                "type": "object",
                "description": (
                    "Path parameters for URL template "
                    "(e.g., restaurant_id). Empty if no placeholders."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Which browser state values you used and why. "
                    "Reference specific source locations."
                ),
            },
            "confidence": {
                "type": "number",
                "description": (
                    "0.0-1.0 confidence that these parameters will "
                    "produce the intended result."
                ),
            },
            "evidence_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Which parameter context entries informed each value.",
            },
        },
        "required": ["parameters", "reasoning", "confidence", "evidence_sources"],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _iter_json_leaves(obj: Any, path: str = "$"):
    """Recursively yield (json_path, leaf_value) from a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_json_leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:50]):  # Cap list traversal
            yield from _iter_json_leaves(item, f"{path}[{i}]")
    else:
        yield (path, obj)


def _format_json_compact(obj: Any, max_depth: int = 3, _depth: int = 0) -> str:
    """Format JSON compactly — show structure, truncate values."""
    indent = "  " * (_depth + 1)
    if _depth >= max_depth:
        return f"{indent}..."

    if isinstance(obj, dict):
        if not obj:
            return f"{indent}{{}}"
        lines = [f"{indent}{{"]
        for key, value in list(obj.items())[:15]:
            val_str = _format_value_compact(value, max_depth, _depth)
            lines.append(f'{indent}  "{key}": {val_str}')
        if len(obj) > 15:
            lines.append(f"{indent}  ... ({len(obj) - 15} more fields)")
        lines.append(f"{indent}}}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return f"{indent}[]"
        first = _format_value_compact(obj[0], max_depth, _depth)
        return f"{indent}[{first}, ... ({len(obj)} items)]"
    else:
        return f"{indent}{_format_value_compact(obj, max_depth, _depth)}"


def _collect_response_paths(
    obj: Any, prefix: str = "$",
) -> dict[str, set[str]]:
    """Collect all JSON paths in a response, categorized as present and non-null."""
    result: dict[str, set[str]] = {"present": set(), "non_null": set()}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}"
            result["present"].add(path)
            if v is not None:
                result["non_null"].add(path)
            sub = _collect_response_paths(v, path)
            result["present"] |= sub["present"]
            result["non_null"] |= sub["non_null"]
    elif isinstance(obj, list) and obj:
        # Sample first element for structural template
        path = f"{prefix}[0]"
        result["present"].add(path)
        if obj[0] is not None:
            result["non_null"].add(path)
        sub = _collect_response_paths(obj[0], path)
        result["present"] |= sub["present"]
        result["non_null"] |= sub["non_null"]
    return result


def _format_value_compact(value: Any, max_depth: int, depth: int) -> str:
    """Format a single value compactly."""
    if isinstance(value, str):
        return f'"{value[:40]}..."' if len(value) > 40 else f'"{value}"'
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, dict):
        return f"{{...{len(value)} fields}}" if depth >= max_depth - 1 else "{...}"
    elif isinstance(value, list):
        return f"[...{len(value)} items]"
    elif value is None:
        return "null"
    return str(value)[:40]


# ---------------------------------------------------------------------------
# ValueSourceRegistry
# ---------------------------------------------------------------------------

class ValueSourceRegistry:
    """Index of all string values observed during the session.

    Ingests from: cookies, localStorage, sessionStorage, DOM attributes,
    form fields, prior API response bodies, URL segments/query params.

    Provides O(1) lookup: "where did this parameter value come from?"
    """

    def __init__(self, max_entries: int = 50_000):
        self._registry: dict[str, list[SourceRecord]] = defaultdict(list)
        self._max = max_entries
        self._total = 0

    def register(self, value: str, source: SourceRecord) -> None:
        """Register a value with its source. Dedup by source_type+source_key."""
        if len(value) < 2 or len(value) > 500:
            return
        if self._total >= self._max:
            return
        existing = self._registry[value]
        if not any(
            r.source_type == source.source_type and r.source_key == source.source_key
            for r in existing
        ):
            existing.append(source)
            self._total += 1

    def lookup(self, value: str) -> list[SourceRecord]:
        """Find all known sources for a given value."""
        return self._registry.get(value, [])

    # --- Ingestion methods ------------------------------------------------

    async def ingest_cookies(self, context) -> None:
        """Ingest all browser cookies."""
        try:
            cookies = await context.cookies()
            now = time.time()
            for c in cookies:
                self.register(c["value"], SourceRecord(
                    value=c["value"],
                    source_type="cookie",
                    source_key=c["name"],
                    source_url=c.get("domain", ""),
                    timestamp=now,
                ))
        except Exception:
            pass

    async def ingest_storage(self, page) -> None:
        """Ingest localStorage and sessionStorage."""
        now = time.time()
        for storage_type in ("localStorage", "sessionStorage"):
            try:
                data = await page.evaluate(f"""() => {{
                    const s = window.{storage_type};
                    const result = {{}};
                    for (let i = 0; i < s.length; i++) {{
                        result[s.key(i)] = s.getItem(s.key(i));
                    }}
                    return result;
                }}""")
                mapped = (
                    "local_storage" if storage_type == "localStorage"
                    else "session_storage"
                )
                for key, value in (data or {}).items():
                    if isinstance(value, str):
                        self.register(value, SourceRecord(
                            value=value,
                            source_type=mapped,
                            source_key=key,
                            timestamp=now,
                        ))
            except Exception:
                pass

    async def ingest_dom_fields(self, page) -> None:
        """Ingest form field values, data attributes, hidden inputs, meta tags."""
        try:
            fields = await page.evaluate("""() => {
                const results = [];
                for (const el of document.querySelectorAll(
                    'input, select, textarea, [data-testid], [data-id]'
                )) {
                    const selector = el.tagName.toLowerCase() +
                        (el.id ? '#' + el.id : '') +
                        (el.name ? '[name="' + el.name + '"]' : '');
                    if (el.value)
                        results.push({value: el.value, key: selector, type: 'dom_element'});
                    for (const attr of el.attributes) {
                        if (attr.name.startsWith('data-') && attr.value) {
                            results.push({
                                value: attr.value,
                                key: selector + '@' + attr.name,
                                type: 'dom_attribute',
                            });
                        }
                    }
                }
                for (const el of document.querySelectorAll('input[type="hidden"]')) {
                    if (el.value && el.name) {
                        results.push({
                            value: el.value,
                            key: 'input[name="' + el.name + '"]',
                            type: 'form_hidden_field',
                        });
                    }
                }
                for (const el of document.querySelectorAll('meta[name], meta[property]')) {
                    const content = el.getAttribute('content');
                    const name = el.getAttribute('name') || el.getAttribute('property');
                    if (content && name) {
                        results.push({
                            value: content,
                            key: 'meta[name="' + name + '"]',
                            type: 'meta_tag',
                        });
                    }
                }
                return results;
            }""")
            page_url = page.url
            now = time.time()
            for item in (fields or []):
                self.register(item["value"], SourceRecord(
                    value=item["value"],
                    source_type=item["type"],
                    source_key=item["key"],
                    source_url=page_url,
                    timestamp=now,
                ))
        except Exception:
            pass

    def ingest_api_response(self, url: str, body: dict | list | None) -> None:
        """Ingest all leaf string values from an API response body."""
        if not body:
            return
        body_str = json.dumps(body, default=str)
        if len(body_str) > 100_000:
            return
        now = time.time()
        for path, val in _iter_json_leaves(body):
            if isinstance(val, str) and 2 <= len(val) <= 500:
                self.register(val, SourceRecord(
                    value=val,
                    source_type="prior_api_response",
                    source_key=path,
                    source_url=url,
                    timestamp=now,
                ))

    def ingest_url(self, url: str) -> None:
        """Ingest URL path segments and query parameters."""
        parsed = urlparse(url)
        now = time.time()
        for i, segment in enumerate(parsed.path.strip("/").split("/")):
            if segment and len(segment) >= 2:
                self.register(segment, SourceRecord(
                    value=segment,
                    source_type="url_segment",
                    source_key=f"path[{i}]",
                    source_url=url,
                    timestamp=now,
                ))
        for key, values in parse_qs(parsed.query).items():
            for val in values:
                self.register(val, SourceRecord(
                    value=val,
                    source_type="url_query_param",
                    source_key=key,
                    source_url=url,
                    timestamp=now,
                ))


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------

class MCPManager:
    """API tool discovery, execution, and lifecycle management.

    - discover_tools_from_subtask: called by orchestrator after CU success
    - execute_tool: called when orchestrator routes to an MCP tool
    - get_available_tools: called by orchestrator for planning context
    """

    def __init__(
        self,
        session: SessionManager,
        reflector: Reflector,
        trace: TaskTrace,
    ):
        self.session = session
        self.reflector = reflector
        self.trace = trace
        self.registry = ValueSourceRegistry()
        self._tools: dict[str, MCPToolDefinition] = {}
        # endpoint_identity → tool_name for dedup across observations
        self._endpoint_map: dict[str, str] = {}
        # Learned noise endpoints — updated over time, not hardcoded
        self._known_noise_endpoints: set[str] = set()
        # MCP response cache: endpoint_identity → most recent response body.
        # Enables chaining: Tool B's recipe can reference Tool A's response
        # regardless of whether Tool A ran via MCP or CU.
        self._response_cache: dict[str, dict | list | None] = {}
        self._load_tools()

    def _log(self, event_type: str, summary: str, **kwargs) -> str | None:
        return self.trace.log("mcp_manager", event_type, summary, **kwargs)

    # ===================================================================
    # Discovery
    # ===================================================================

    async def discover_tools_from_subtask(
        self,
        traffic_since: float,
        subtask_description: str,
    ) -> list[MCPToolDefinition]:
        """Analyze traffic from a successful CU subtask for API tool candidates.

        Called by orchestrator AFTER CU subtask succeeds.
        Only state-changing requests with successful responses are candidates.
        """
        requests = self.session.get_captured_traffic(since_timestamp=traffic_since)
        if not requests:
            return []

        # Index current browser state into the registry
        await self._refresh_registry()

        # Ingest API response bodies for source tracing
        for req in requests:
            if req.response_body_parsed:
                self.registry.ingest_api_response(req.url, req.response_body_parsed)

        # Filter to actionable candidates
        candidates = [r for r in requests if not self._is_noise_endpoint(r)]
        if not candidates:
            return []

        discovered: list[MCPToolDefinition] = []
        for req in candidates:
            tool = self._analyze_request(req, subtask_description)
            if tool:
                discovered.append(tool)
                self._log("tool_discovered", f"Discovered: {tool.name}", detail={
                    "tool_name": tool.name,
                    "endpoint": tool.endpoint_identity,
                    "method": tool.method,
                    "protocol": tool.protocol,
                })

        if discovered:
            self._save_tools()

        return discovered

    _NOISE_SUBDOMAINS = frozenset({
        "analytics", "tracking", "telemetry", "metrics",
        "events", "collect", "pixel", "beacon", "log", "stats",
    })

    def _is_noise_endpoint(self, req: CapturedRequest) -> bool:
        """Data-driven noise detection — no hardcoded path keywords.

        session_manager already filters known noise domains via _is_noise_url().
        Here we filter further: non-state-changing, errors, empty beacons,
        cross-domain tracking, noise subdomains, and previously-learned noise endpoints.
        """
        if not req.is_state_changing:
            return True
        if req.status_code >= 400:
            return True
        # Subdomain noise: analytics.swiggy.com, tracking.lego.com, etc.
        hostname = urlparse(req.url).hostname or ""
        if hostname.count(".") >= 2:
            subdomain = hostname.split(".")[0]
            if subdomain in self._NOISE_SUBDOMAINS:
                return True
        # Empty POST with no query params = tracking beacon
        if (
            not req.request_body_parsed
            and not urlparse(req.url).query
            and req.method.upper() == "POST"
        ):
            return True
        # Cross-domain requests are almost always analytics/tracking.
        # A tool for lego.com shouldn't point to google.com or facebook.com.
        req_host = (urlparse(req.url).hostname or "").lower()
        site_host = (urlparse(self.session.start_url).hostname or "").lower()
        if req_host and site_host:
            # Extract registrable domain (last two parts, e.g. "lego.com")
            req_domain = ".".join(req_host.rsplit(".", 2)[-2:])
            site_domain = ".".join(site_host.rsplit(".", 2)[-2:])
            if req_domain != site_domain:
                return True
        return (req.endpoint_identity or "") in self._known_noise_endpoints

    def _analyze_request(
        self, req: CapturedRequest, subtask_description: str,
    ) -> MCPToolDefinition | None:
        """Analyze a single captured request and build/update a tool definition."""
        endpoint_id = req.endpoint_identity or ""

        # Existing tool for this endpoint? Update it.
        if endpoint_id and endpoint_id in self._endpoint_map:
            existing_name = self._endpoint_map[endpoint_id]
            if existing_name in self._tools:
                self._update_existing_tool(existing_name, req)
                return self._tools[existing_name]

        # Build parameter schema from request body
        param_schema: dict = {}
        if req.request_body_parsed and isinstance(req.request_body_parsed, dict):
            param_schema = self._build_schema_from_example(req.request_body_parsed)

        # Trace parameter values to their browser state sources
        param_sources: dict[str, list[dict]] = {}
        if req.request_body_parsed and isinstance(req.request_body_parsed, dict):
            raw_sources = self._trace_parameter_sources(req.request_body_parsed)
            for path, src_list in raw_sources.items():
                param_sources[path] = [asdict(s) for s in src_list]

        # Build extraction recipe from traced sources
        extraction_recipe: list[dict] = []
        if req.request_body_parsed and isinstance(req.request_body_parsed, dict):
            extraction_recipe = self._build_recipe_from_sources(
                req.request_body_parsed, param_sources,
            )

        # Build response template from this first observation
        response_template: dict = {}
        if req.response_body_parsed:
            response_template = self._build_response_template(req.response_body_parsed)

        # Generate tool name and description via Gemini
        metadata = self._generate_tool_metadata(req, subtask_description)
        tool_name = metadata.get("name", self._fallback_tool_name(req))

        # Deduplicate name on collision
        if tool_name in self._tools:
            tool_name = f"{tool_name}_{int(time.time()) % 10000}"

        # Build example
        example = {
            "request_body": (
                req.request_body_parsed
                if isinstance(req.request_body_parsed, dict) else None
            ),
            "query_params": dict(parse_qs(urlparse(req.url).query)),
            "response_status": req.status_code,
            "response_body_excerpt": (
                json.dumps(req.response_body_parsed, default=str)[:500]
                if req.response_body_parsed else ""
            ),
            "timestamp": req.timestamp,
        }

        tool = MCPToolDefinition(
            name=tool_name,
            description=metadata.get("description", f"API call from: {subtask_description}"),
            endpoint_identity=endpoint_id,
            method=req.method,
            url_template=req.url,  # First observation: concrete URL
            protocol=req.protocol or "rest",
            parameter_schema=param_schema,
            header_template=self._extract_header_template(req),
            parameter_sources=param_sources,
            examples=[example],
            status="verified",
            observation_count=1,
            is_safe_to_replay=metadata.get("is_safe_to_replay", False),
            created_at=req.timestamp,
            last_used_at=req.timestamp,
            observed_urls=[req.url],
            extraction_recipe=extraction_recipe,
            response_template=response_template,
            graphql_query=(
                req.request_body_parsed.get("query")
                if req.protocol == "graphql"
                and isinstance(req.request_body_parsed, dict)
                else None
            ),
        )

        self._tools[tool_name] = tool
        if endpoint_id:
            self._endpoint_map[endpoint_id] = tool_name

        return tool

    def _update_existing_tool(self, tool_name: str, req: CapturedRequest) -> None:
        """Update an existing tool with a new observation."""
        tool = self._tools[tool_name]
        tool.observation_count += 1
        tool.last_used_at = req.timestamp

        # Evolve URL template from frequency analysis
        if req.url not in tool.observed_urls:
            tool.observed_urls.append(req.url)
            tool.url_template = self._parameterize_url(
                req.url, tool.observed_urls[:-1],
            )

        # Merge schema observation
        if req.request_body_parsed and isinstance(req.request_body_parsed, dict):
            tool.parameter_schema = self._merge_schema_observation(
                tool.parameter_schema, req.request_body_parsed,
            )

        # Merge new parameter sources (keep existing — already validated)
        if req.request_body_parsed and isinstance(req.request_body_parsed, dict):
            new_sources = self._trace_parameter_sources(req.request_body_parsed)
            for path, src_list in new_sources.items():
                if path not in tool.parameter_sources:
                    tool.parameter_sources[path] = [asdict(s) for s in src_list]

        # Merge response template
        if req.response_body_parsed:
            tool.response_template = self._merge_response_template(
                tool.response_template, req.response_body_parsed,
            )

        # Update extraction recipe — merge new sources into existing steps
        if req.request_body_parsed and isinstance(req.request_body_parsed, dict):
            new_sources = self._trace_parameter_sources(req.request_body_parsed)
            new_param_sources = {
                path: [asdict(s) for s in src_list]
                for path, src_list in new_sources.items()
            }
            existing_paths = {
                step["param_path"] for step in tool.extraction_recipe
            }
            for path, value in _iter_json_leaves(req.request_body_parsed):
                if path not in existing_paths:
                    sources = new_param_sources.get(path, [])
                    step = self._build_extraction_step(path, sources, value)
                    tool.extraction_recipe.append(step)

        # Add example (cap at 5)
        example = {
            "request_body": (
                req.request_body_parsed
                if isinstance(req.request_body_parsed, dict) else None
            ),
            "query_params": dict(parse_qs(urlparse(req.url).query)),
            "response_status": req.status_code,
            "response_body_excerpt": (
                json.dumps(req.response_body_parsed, default=str)[:500]
                if req.response_body_parsed else ""
            ),
            "timestamp": req.timestamp,
        }
        tool.examples.append(example)
        if len(tool.examples) > 5:
            tool.examples = tool.examples[-5:]

    def _trace_parameter_sources(
        self, body: dict, prefix: str = "",
    ) -> dict[str, list[ParameterSource]]:
        """For each leaf value in request body, find its source in browser state.

        This is the core innovation — source hints enable targeted parameter
        generation on future runs. Instead of scanning the entire DOM for a
        CSRF token, the parameter generator knows where to look.
        """
        sources: dict[str, list[ParameterSource]] = {}
        for path, value in _iter_json_leaves(body, prefix or "$"):
            if not isinstance(value, str) or len(value) < 2:
                continue
            records = self.registry.lookup(value)
            if records:
                sources[path] = [
                    ParameterSource(
                        source_type=r.source_type,
                        source_key=r.source_key,
                        source_url=r.source_url,
                        confidence=1.0,
                        example_value=value[:100],
                    )
                    for r in records[:3]  # Top 3 sources per param
                ]
        return sources

    # ===================================================================
    # Extraction Recipe Building (discovery-time → execution-time)
    # ===================================================================

    @staticmethod
    def _classify_parameter(
        param_path: str,
        sources: list[dict],
        value: str,
    ) -> str:
        """Classify a parameter for the extraction recipe.

        Determines how the parameter should be handled at execution time.
        """
        if not sources:
            return "user_intent"
        source_types = {s.get("source_type", "") for s in sources}
        if source_types & {"cookie", "meta_tag", "form_hidden_field"}:
            return "ephemeral"
        if "prior_api_response" in source_types:
            return "chained"
        if source_types & {"dom_attribute", "dom_element"}:
            return "page_context"
        if source_types & {"url_segment", "url_query_param"}:
            return "static"
        if source_types & {"local_storage", "session_storage"}:
            return "ephemeral"
        return "user_intent"

    def _build_extraction_step(
        self,
        param_path: str,
        sources: list[dict],
        value: Any,
    ) -> dict:
        """Convert a traced ParameterSource into an ExtractionStep dict.

        Maps each source type to the corresponding extraction mechanism.
        """
        classification = self._classify_parameter(param_path, sources, str(value))

        if not sources:
            return asdict(ExtractionStep(
                param_path=param_path,
                source_type="task_description",
                source_config={},
                classification="user_intent",
                description=f"Derive from task description (no source found)",
            ))

        src = sources[0]  # Primary source
        src_type = src.get("source_type", "")
        src_key = src.get("source_key", "")
        src_url = src.get("source_url", "")

        match src_type:
            case "cookie":
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="cookie",
                    source_config={"key": src_key},
                    classification=classification,
                    description=f"Cookie '{src_key}'",
                ))
            case "dom_element" | "form_hidden_field":
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="dom_field",
                    source_config={"selector": src_key},
                    classification=classification,
                    description=f"DOM field at '{src_key}'",
                ))
            case "dom_attribute":
                # DOM attributes often represent repeated items (product cards)
                parts = src_key.split("@")
                selector = parts[0] if parts else src_key
                attr = parts[1] if len(parts) > 1 else "value"
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="dom_list",
                    source_config={
                        "selector": f"[{attr}]" if selector == attr else selector,
                        "value_attr": attr,
                        "label_attr": "textContent",
                    },
                    classification="page_context",
                    description=f"DOM attribute '{attr}' on elements matching '{selector}'",
                ))
            case "local_storage" | "session_storage":
                storage_type = (
                    "localStorage" if src_type == "local_storage"
                    else "sessionStorage"
                )
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="storage",
                    source_config={"storage_type": storage_type, "key": src_key},
                    classification=classification,
                    description=f"{storage_type} key '{src_key}'",
                ))
            case "meta_tag":
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="meta_tag",
                    source_config={"selector": src_key},
                    classification=classification,
                    description=f"Meta tag '{src_key}'",
                ))
            case "prior_api_response":
                # Find endpoint_identity from the source URL
                endpoint_id = self._url_to_endpoint_identity(src_url)
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="prior_api_response",
                    source_config={
                        "endpoint_identity": endpoint_id,
                        "json_path": src_key,  # Already a JSON path like $.data[0].place_id
                    },
                    classification="chained",
                    description=f"From {endpoint_id} response at {src_key}",
                ))
            case "url_segment":
                idx_match = re.search(r"\[(\d+)\]", src_key)
                index = int(idx_match.group(1)) if idx_match else 0
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="url_component",
                    source_config={"component": "path_segment", "index": index},
                    classification=classification,
                    description=f"URL path segment [{index}]",
                ))
            case "url_query_param":
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="url_component",
                    source_config={"component": "query_param", "key": src_key},
                    classification=classification,
                    description=f"URL query param '{src_key}'",
                ))
            case _:
                return asdict(ExtractionStep(
                    param_path=param_path,
                    source_type="task_description",
                    source_config={},
                    classification="user_intent",
                    description=f"Unknown source type '{src_type}', derive from task",
                ))

    def _url_to_endpoint_identity(self, url: str) -> str:
        """Derive an endpoint_identity from a URL by checking captured traffic."""
        for req in reversed(self.session.get_captured_traffic()):
            if req.url == url and req.endpoint_identity:
                return req.endpoint_identity
        # Fallback: construct from URL path
        parsed = urlparse(url)
        return f"GET {parsed.path}" if parsed.path else url

    def _build_recipe_from_sources(
        self,
        body: dict,
        param_sources: dict[str, list[dict]],
    ) -> list[dict]:
        """Build extraction recipe for all parameters in request body."""
        recipe: list[dict] = []
        for path, value in _iter_json_leaves(body):
            sources = param_sources.get(path, [])
            step = self._build_extraction_step(path, sources, value)
            recipe.append(step)
        return recipe

    # ===================================================================
    # Response Template Building (learned from successful responses)
    # ===================================================================

    @staticmethod
    def _build_response_template(response_body: Any) -> dict:
        """Build a structural template from a response for deterministic checking.

        Tracks which paths are always present and which are always non-null.
        """
        if not isinstance(response_body, (dict, list)):
            return {}
        paths = _collect_response_paths(response_body)
        return {
            "always_present_paths": sorted(paths["present"]),
            "always_non_null_paths": sorted(paths["non_null"]),
            "observation_count": 1,
        }

    @staticmethod
    def _merge_response_template(
        existing: dict, response_body: Any,
    ) -> dict:
        """Merge a new response observation into the template.

        always_present_paths = intersection (present in ALL observations).
        always_non_null_paths = intersection (non-null in ALL observations).
        """
        if not existing:
            return MCPManager._build_response_template(response_body)
        if not isinstance(response_body, (dict, list)):
            return existing

        new_paths = _collect_response_paths(response_body)
        existing["always_present_paths"] = sorted(
            set(existing.get("always_present_paths", []))
            & set(new_paths["present"])
        )
        existing["always_non_null_paths"] = sorted(
            set(existing.get("always_non_null_paths", []))
            & set(new_paths["non_null"])
        )
        existing["observation_count"] = existing.get("observation_count", 0) + 1
        return existing

    async def _refresh_registry(self) -> None:
        """Ingest current browser state into the value source registry."""
        if self.session._context:
            await self.registry.ingest_cookies(self.session._context)
        if self.session.page:
            await self.registry.ingest_storage(self.session.page)
            await self.registry.ingest_dom_fields(self.session.page)
            self.registry.ingest_url(self.session.page.url)

    # ===================================================================
    # Execution
    # ===================================================================

    async def execute_tool(
        self,
        tool_name: str,
        subtask_description: str,
    ) -> dict:
        """Execute an MCP tool for a subtask.

        1. Build focused parameter context (recipe-based or legacy)
        2. Call Gemini to generate parameters (schema-driven)
        3. Execute HTTP request via curl_cffi
        4. Cache response for chaining, return result for reflector
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        with self.trace.span(
            "mcp_manager", "tool_execution", f"Execute: {tool_name}",
        ) as span:
            # 1. Build parameter context — recipe-based if available
            if tool.extraction_recipe:
                from morphnet.representation import build_mcp_parameter_context
                param_context = await build_mcp_parameter_context(
                    recipe=tool.extraction_recipe,
                    tool_name=tool.name,
                    tool_description=tool.description,
                    tool_method=tool.method,
                    tool_url_template=tool.url_template,
                    tool_protocol=tool.protocol,
                    tool_examples=tool.examples[-3:],
                    subtask=subtask_description,
                    session=self.session,
                    mcp_response_cache=self._response_cache,
                )
            else:
                # Legacy fallback for tools discovered before recipe system
                param_context = await self._build_parameter_context(
                    tool, subtask_description,
                )

            # 2. Generate parameters via Gemini
            gen_result = self._generate_parameters(
                tool, subtask_description, param_context,
            )
            if not gen_result:
                span.set_outcome("failure")
                span.set_reasoning("Parameter generation failed")
                return {"success": False, "error": "Parameter generation failed"}

            params = gen_result.get("parameters", {})
            url_params = gen_result.get("url_params", {})

            # 3. Transfer cookies for replay
            await self.session.sync_cookies_to_http_session()

            # 4. Resolve URL template placeholders
            url = tool.url_template
            for key, val in url_params.items():
                url = url.replace(f"{{{key}}}", str(val))

            # 5. Execute HTTP request
            result = self._execute_http(tool, url, params)

            # 6. Cache response for chaining + update tool
            result["sent_params"] = params
            tool.observation_count += 1
            tool.last_used_at = time.time()
            if result.get("success"):
                response_body = result.get("response_body")
                # Cache by endpoint_identity for chaining
                if tool.endpoint_identity:
                    self._response_cache[tool.endpoint_identity] = response_body
                # Also ingest into value registry for downstream tools
                if isinstance(response_body, (dict, list)):
                    self.registry.ingest_api_response(url, response_body)
                # Merge response template
                if response_body:
                    tool.response_template = self._merge_response_template(
                        tool.response_template, response_body,
                    )
                tool.examples.append({
                    "request_body": params,
                    "query_params": None,
                    "response_status": result.get("status_code", 0),
                    "response_body_excerpt": str(
                        result.get("response_body", "")
                    )[:500],
                    "timestamp": time.time(),
                })
                if len(tool.examples) > 5:
                    tool.examples = tool.examples[-5:]

            self._save_tools()

            span.set_outcome("success" if result.get("success") else "failure")
            span.set_reasoning(
                f"HTTP {result.get('status_code', '?')} — "
                f"{gen_result.get('reasoning', '')[:100]}"
            )
            span.set_confidence(gen_result.get("confidence", 0.5))
            for src in gen_result.get("evidence_sources", []):
                span.add_evidence(Evidence(source="model_output", description=src))

            return result

    async def _build_parameter_context(
        self, tool: MCPToolDefinition, subtask: str,
    ) -> str:
        """Build a focused parameter context from browser state.

        Like representation.py transforms AXTree for CU, this transforms
        browser state into an actionable context for parameter generation.

        Three sections: task + template + current values at source locations.
        """
        lines: list[str] = []

        # Section 1: Task context
        lines.append(f"TASK: {subtask}")
        lines.append(f"TOOL: {tool.name} — {tool.description}")
        lines.append(f"ENDPOINT: {tool.method} {tool.url_template}")
        lines.append(f"PROTOCOL: {tool.protocol}")
        lines.append("")

        # Section 2: Request template from most recent example
        if tool.examples:
            latest = tool.examples[-1]
            body = latest.get("request_body") if isinstance(latest, dict) else None
            if body:
                lines.append("LAST SUCCESSFUL REQUEST BODY:")
                lines.append(_format_json_compact(body, max_depth=3))
                lines.append(
                    f"→ Response: {latest.get('response_status', '?')}"
                )
                lines.append("")

        # Section 3: GraphQL query signature
        if tool.protocol == "graphql" and tool.graphql_query:
            lines.append("GRAPHQL QUERY:")
            query_lines = tool.graphql_query.strip().split("\n")
            for ql in query_lines[:10]:
                lines.append(f"  {ql.rstrip()}")
            if len(query_lines) > 10:
                lines.append(f"  ... ({len(query_lines) - 10} more lines)")
            lines.append("")

        # Section 4: Current browser state at each parameter's source location
        lines.append("CURRENT BROWSER STATE (at parameter source locations):")

        for param_path, sources in tool.parameter_sources.items():
            param_name = param_path.split(".")[-1]
            lines.append(f"  {param_name}:")

            source_dicts = sources if isinstance(sources, list) else []
            for src_dict in source_dicts:
                if not isinstance(src_dict, dict):
                    continue
                src = ParameterSource(**src_dict)
                current_value = await self._extract_current_value(src)
                if current_value is not None:
                    lines.append(
                        f"    → {src.source_type} @ {src.source_key}: "
                        f'"{current_value[:80]}"'
                    )
                else:
                    lines.append(
                        f"    → {src.source_type} @ {src.source_key}: NOT FOUND "
                        f'(was: "{src.example_value[:40]}")'
                    )

        # Section 5: Current URL and page context
        if self.session.page:
            lines.append("")
            lines.append(f"CURRENT URL: {self.session.page.url}")
            try:
                title = await self.session.page.title()
                lines.append(f"PAGE: {title}")
            except Exception:
                pass

        return "\n".join(lines)

    async def _extract_current_value(
        self, source: ParameterSource,
    ) -> str | None:
        """Extract the current value from a specific browser state location.

        Targeted extraction — not scanning the entire DOM, just looking
        at the exact location where we previously found this parameter.
        """
        try:
            match source.source_type:
                case "cookie":
                    cookies = await self.session._context.cookies()
                    for c in cookies:
                        if c["name"] == source.source_key:
                            return c["value"]

                case "dom_element" | "form_hidden_field":
                    selector = source.source_key.replace("'", "\\'")
                    return await self.session.page.evaluate(
                        f"() => {{ const el = document.querySelector('{selector}'); "
                        f"return el ? (el.value || el.textContent || '') : null; }}"
                    )

                case "dom_attribute":
                    parts = source.source_key.split("@")
                    if len(parts) == 2:
                        selector = parts[0].replace("'", "\\'")
                        attr = parts[1].replace("'", "\\'")
                        return await self.session.page.evaluate(
                            f"() => {{ const el = document.querySelector('{selector}'); "
                            f"return el ? el.getAttribute('{attr}') : null; }}"
                        )

                case "local_storage":
                    key = source.source_key.replace("'", "\\'")
                    return await self.session.page.evaluate(
                        f"() => localStorage.getItem('{key}')"
                    )

                case "session_storage":
                    key = source.source_key.replace("'", "\\'")
                    return await self.session.page.evaluate(
                        f"() => sessionStorage.getItem('{key}')"
                    )

                case "meta_tag":
                    selector = source.source_key.replace("'", "\\'")
                    return await self.session.page.evaluate(
                        f"() => {{ const el = document.querySelector('{selector}'); "
                        f"return el ? el.getAttribute('content') : null; }}"
                    )

                case "url_query_param":
                    parsed = urlparse(self.session.page.url)
                    params = parse_qs(parsed.query)
                    vals = params.get(source.source_key, [])
                    return vals[0] if vals else None

                case "url_segment":
                    idx_match = re.search(r"\[(\d+)\]", source.source_key)
                    if idx_match:
                        idx = int(idx_match.group(1))
                        segments = (
                            urlparse(self.session.page.url)
                            .path.strip("/")
                            .split("/")
                        )
                        if idx < len(segments):
                            return segments[idx]
        except Exception:
            pass
        return None

    def _generate_parameters(
        self,
        tool: MCPToolDefinition,
        subtask: str,
        param_context: str,
    ) -> dict | None:
        """Call Gemini to generate parameters for this tool execution.

        The prompt includes: tool schema, source hints, examples, and the
        focused parameter context from current browser state.
        """
        schema = build_param_generation_schema(tool)

        prompt = (
            f'Generate parameters for the MCP tool "{tool.name}".\n\n'
            f"{param_context}\n\n"
            f"SOURCE HINTS:\n{self._format_source_hints(tool.parameter_sources)}\n\n"
            f"PREVIOUS EXAMPLES:\n{self._format_examples(tool.examples[-3:])}\n\n"
            "Generate the request body as a JSON object matching the schema. "
            "Use current values from the browser state where available. "
            "For parameters not found in current state, use the pattern from examples."
        )

        try:
            result = call_gemini(
                model="gemini-3-flash-preview",
                contents=[prompt],
                response_schema=schema,
                generation_config={"temperature": 0.1, "max_output_tokens": 8192},
                prompt_log_dir=self.trace.output_dir / "prompt_mcp",
            )
        except Exception as exc:
            logger.warning("Gemini param generation failed for %s: %s", tool.name, exc)
            return None

        if result and isinstance(result, dict) and result.get("parameters"):
            self._log(
                "param_generation",
                f"Params for {tool.name}",
                detail={"tool_name": tool.name},
                reasoning=result.get("reasoning"),
                confidence=result.get("confidence"),
                evidence=[
                    Evidence(source="model_output", description=src)
                    for src in result.get("evidence_sources", [])
                ],
            )
            return result
        return None

    def _execute_http(
        self, tool: MCPToolDefinition, url: str, params: dict,
    ) -> dict:
        """Execute the HTTP request via curl_cffi."""
        try:
            http = self.session.get_http_session()
            headers = dict(tool.header_template)
            headers.pop("host", None)
            headers.pop("Host", None)

            match tool.protocol:
                case "graphql":
                    body = {"query": tool.graphql_query, "variables": params}
                    resp = http.post(url, json=body, headers=headers)
                case "form" | "multipart":
                    resp = http.post(url, data=params, headers=headers)
                case _:  # rest, jsonrpc
                    method = tool.method.upper()
                    if method == "GET":
                        resp = http.get(url, params=params, headers=headers)
                    elif method == "POST":
                        resp = http.post(url, json=params, headers=headers)
                    elif method == "PUT":
                        resp = http.put(url, json=params, headers=headers)
                    elif method == "DELETE":
                        resp = http.delete(url, json=params, headers=headers)
                    elif method == "PATCH":
                        resp = http.patch(url, json=params, headers=headers)
                    else:
                        resp = http.request(
                            method, url, json=params, headers=headers,
                        )

            response_body = None
            try:
                response_body = resp.json()
            except Exception:
                response_body = resp.text[:2000] if resp.text else None

            return {
                "success": resp.status_code < 400,
                "status_code": resp.status_code,
                "response_body": response_body,
                "response_headers": dict(resp.headers),
            }

        except Exception as exc:
            return {
                "success": False,
                "status_code": 0,
                "error": str(exc),
                "response_body": None,
            }

    # ===================================================================
    # Schema Building & Evolution
    # ===================================================================

    def _build_schema_from_example(self, body: dict) -> dict:
        """Build a JSON schema from a single example request body."""
        schema: dict = {
            "type": "object",
            "properties": {},
            "required": list(body.keys()),
        }
        for key, value in body.items():
            schema["properties"][key] = self._infer_field_schema(value)
        return schema

    def _merge_schema_observation(
        self, existing_schema: dict, new_body: dict,
    ) -> dict:
        """Merge a new observation into the existing schema.

        After each observation:
        - Required = intersection (fields present in ALL observations)
        - New fields → optional (not in required)
        - Type conflicts → anyOf union
        - Low-cardinality values → enum candidates
        """
        if not existing_schema.get("properties"):
            return self._build_schema_from_example(new_body)

        existing_props = existing_schema.get("properties", {})
        existing_required = set(existing_schema.get("required", []))

        new_keys = set(new_body.keys())

        # Required = intersection only
        updated_required = existing_required & new_keys

        # Add new properties not seen before
        for key in new_keys - set(existing_props.keys()):
            existing_props[key] = self._infer_field_schema(new_body[key])

        # Update existing with new type observations
        for key in new_keys & set(existing_props.keys()):
            new_value = new_body[key]
            new_type = self._infer_type(new_value)
            existing_type = existing_props[key].get("type")

            # Type conflict → anyOf
            if (
                existing_type
                and new_type != existing_type
                and "anyOf" not in existing_props[key]
            ):
                existing_props[key] = {
                    "anyOf": [
                        {"type": existing_type},
                        {"type": new_type},
                    ]
                }
            elif "anyOf" in existing_props[key]:
                types = {
                    item["type"] for item in existing_props[key]["anyOf"]
                }
                if new_type not in types:
                    existing_props[key]["anyOf"].append({"type": new_type})

            # Track example values for enum detection
            examples = existing_props[key].setdefault("x-examples", [])
            examples.append(str(new_value)[:100])
            if len(examples) > 20:
                existing_props[key]["x-examples"] = examples[-20:]

            # Enum detection removed — catastrophic for dynamic fields.
            # User-intent fields (search queries), session tokens (CSRF),
            # and chained outputs (place_id from prior API) all get locked
            # to a small set of observed values, causing param generation
            # to fail on any new input. The x-examples list provides the
            # LLM with sufficient guidance without constraining the schema.
            # Remove any enum that was previously added.
            existing_props[key].pop("enum", None)

        existing_schema["properties"] = existing_props
        existing_schema["required"] = sorted(updated_required)
        return existing_schema

    @staticmethod
    def _infer_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    def _infer_field_schema(self, value: Any) -> dict:
        """Infer a JSON schema for a single field from one observation."""
        schema: dict = {"type": self._infer_type(value)}
        if isinstance(value, str):
            schema["x-examples"] = [value[:100]]
            if re.match(r"^\d{4}-\d{2}-\d{2}", value):
                schema["format"] = "date-time"
            elif re.match(r"^[0-9a-f]{8}-", value, re.I):
                schema["format"] = "uuid"
            elif re.match(r"^https?://", value):
                schema["format"] = "uri"
        elif isinstance(value, dict):
            schema = self._build_schema_from_example(value)
        elif isinstance(value, list):
            if value:
                schema["items"] = self._infer_field_schema(value[0])
            else:
                # Empty array — default to string items so Gemini schema is valid
                schema["items"] = {"type": "string"}
        return schema

    # ===================================================================
    # URL Parameterization (data-driven, not pattern-based)
    # ===================================================================

    @staticmethod
    def _parameterize_url(
        url: str, existing_urls: list[str] | None = None,
    ) -> str:
        """Derive URL template by comparing multiple observed URLs.

        First observation: concrete URL as-is.
        2+ observations: segments that differ across observations become
        template parameters. No hardcoded regex patterns.
        """
        if not existing_urls:
            return url

        parsed = urlparse(url)
        current_segments = parsed.path.strip("/").split("/")

        # Collect comparable URLs (same path length)
        comparable = [current_segments]
        for prev_url in existing_urls:
            prev_segments = urlparse(prev_url).path.strip("/").split("/")
            if len(prev_segments) == len(current_segments):
                comparable.append(prev_segments)

        if len(comparable) < 2:
            return url  # Different path lengths — can't compare

        template_segments: list[str] = []
        for i, seg in enumerate(current_segments):
            values = {segs[i] for segs in comparable if i < len(segs)}
            if len(values) == 1:
                template_segments.append(seg)
            else:
                # Name parameter from preceding static segment
                if (
                    i > 0
                    and template_segments
                    and not template_segments[-1].startswith("{")
                ):
                    param_name = template_segments[-1].rstrip("s") + "_id"
                else:
                    param_name = f"path_param_{i}"
                template_segments.append(f"{{{param_name}}}")

        new_path = (
            "/" + "/".join(template_segments)
            if template_segments
            else parsed.path
        )
        return f"{parsed.scheme}://{parsed.netloc}{new_path}"

    # ===================================================================
    # Header Template
    # ===================================================================

    @staticmethod
    def _extract_header_template(req: CapturedRequest) -> dict:
        """Extract replay-relevant headers.

        curl_cffi with impersonate="chrome" sends standard browser headers.
        We only keep headers specific to this API: Authorization, CSRF,
        custom app headers, Content-Type. Browser security headers
        (sec-ch-*, sec-fetch-*) are generated by curl_cffi impersonation.
        """
        AUTO_MANAGED = frozenset({
            "host", "content-length", "connection", "accept-encoding",
            "cookie",  # Transferred via sync_cookies_to_http_session()
        })

        template: dict[str, str] = {}
        for k, v in req.request_headers.items():
            k_lower = k.lower()
            if k_lower in AUTO_MANAGED:
                continue
            # W3C browser security headers — curl_cffi provides via impersonate
            if k_lower.startswith("sec-ch-") or k_lower.startswith("sec-fetch-"):
                continue
            template[k] = v
        return template

    # ===================================================================
    # Tool Naming (Gemini-generated)
    # ===================================================================

    def _generate_tool_metadata(
        self, req: CapturedRequest, subtask_description: str,
    ) -> dict:
        """Generate descriptive tool name and description using Gemini.

        The model sees endpoint, request/response shapes, and subtask
        context to generate a name like 'add_to_cart' rather than
        path-derived 'post_dapi_cart'.
        """
        request_summary = (
            json.dumps(req.request_body_parsed, default=str)[:800]
            if req.request_body_parsed else "(no body)"
        )
        response_summary = (
            json.dumps(req.response_body_parsed, default=str)[:800]
            if req.response_body_parsed else "(no body)"
        )

        prompt = (
            "An API call was made during this browser automation task:\n"
            f"TASK: {subtask_description}\n\n"
            f"ENDPOINT: {req.method} {urlparse(req.url).path}\n"
            f"PROTOCOL: {req.protocol}\n"
            f"REQUEST BODY:\n{request_summary}\n\n"
            f"RESPONSE ({req.status_code}):\n{response_summary}\n\n"
            "Generate a descriptive tool name and description for this API endpoint."
        )

        try:
            result = call_gemini(
                model="gemini-3-flash-preview",
                contents=[prompt],
                response_schema=TOOL_DESCRIPTION_SCHEMA,
                generation_config={"temperature": 0.1, "max_output_tokens": 2048},
                prompt_log_dir=self.trace.output_dir / "prompt_mcp",
            )
            if result and isinstance(result, dict) and result.get("name"):
                return result
        except Exception as exc:
            logger.warning("Tool metadata generation failed: %s", exc)

        return {
            "name": self._fallback_tool_name(req),
            "description": f"API call from: {subtask_description}",
            "is_safe_to_replay": False,
        }

    @staticmethod
    def _fallback_tool_name(req: CapturedRequest) -> str:
        """Derive name from endpoint identity if Gemini fails."""
        identity = (req.endpoint_identity or "unknown").lower()
        name = re.sub(
            r"[^a-z0-9_]", "_",
            identity.replace("/", "_").replace(" ", "_"),
        )
        return re.sub(r"_+", "_", name).strip("_") or "unknown_tool"

    # ===================================================================
    # Formatting Helpers
    # ===================================================================

    @staticmethod
    def _format_examples(examples: list[dict]) -> str:
        lines: list[str] = []
        for i, ex in enumerate(examples, 1):
            if not isinstance(ex, dict):
                continue
            lines.append(f"Example {i}:")
            body = ex.get("request_body")
            if body:
                lines.append(
                    f"  Request: {json.dumps(body, default=str)[:400]}"
                )
            status = ex.get("response_status", "?")
            excerpt = ex.get("response_body_excerpt", "")[:200]
            lines.append(f"  Response: {status} — {excerpt}")
        return "\n".join(lines) if lines else "(no examples)"

    @staticmethod
    def _format_source_hints(sources: dict) -> str:
        lines: list[str] = []
        for param, src_list in sources.items():
            if not isinstance(src_list, list):
                continue
            for src in src_list:
                if isinstance(src, dict):
                    lines.append(
                        f"  {param}: found in {src['source_type']} "
                        f"@ {src['source_key']} "
                        f"(e.g. \"{str(src.get('example_value', ''))[:40]}\")"
                    )
        return "\n".join(lines) if lines else "(no source hints)"

    # ===================================================================
    # Persistence
    # ===================================================================

    def _load_tools(self) -> None:
        """Load tool definitions from sites/{site_name}/tools.json."""
        site_name = self.session.site_name
        if not site_name:
            return
        tools_path = SITES_DIR / site_name / "tools.json"
        if not tools_path.exists():
            return
        try:
            data = json.loads(tools_path.read_text())
            valid_fields = {f.name for f in dataclass_fields(MCPToolDefinition)}
            for tool_data in data:
                # Filter to known fields for forward compatibility
                filtered = {
                    k: v for k, v in tool_data.items() if k in valid_fields
                }
                tool = MCPToolDefinition(**filtered)
                self._tools[tool.name] = tool
                if tool.endpoint_identity:
                    self._endpoint_map[tool.endpoint_identity] = tool.name
        except Exception as exc:
            logger.warning("Failed to load MCP tools: %s", exc)

    def _save_tools(self) -> None:
        """Persist tool definitions to sites/{site_name}/tools.json."""
        site_name = self.session.site_name
        if not site_name:
            return
        tools_path = SITES_DIR / site_name / "tools.json"
        tools_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(tool) for tool in self._tools.values()]
        tools_path.write_text(json.dumps(data, indent=2, default=str))

    def get_tool(self, name: str) -> MCPToolDefinition | None:
        return self._tools.get(name)

    def get_available_tools(self) -> list[MCPToolDefinition]:
        return [t for t in self._tools.values() if t.status != "discarded"]

    # ===================================================================
    # A/B Learning (MCP failure + CU success → recipe correction)
    # ===================================================================

    async def learn_from_cu_fallback(
        self,
        tool_name: str,
        failed_mcp_result: dict,
        cu_traffic_since: float,
        subtask: str,
    ) -> None:
        """Compare MCP failure params vs CU success traffic to correct the recipe.

        Called by orchestrator after CU fallback succeeds for an MCP-routed subtask.
        1. Find matching CU request by endpoint_identity
        2. For each parameter where values differ: trace the correct value, rebuild step
        3. Merge the correct request body into the schema
        4. Merge the correct response into the response template
        5. Store corrected example tagged source="cu_fallback"
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return

        # Refresh registry with current browser state
        await self._refresh_registry()

        # Find matching CU request by endpoint_identity
        cu_traffic = self.session.get_captured_traffic(since_timestamp=cu_traffic_since)
        matching_req: CapturedRequest | None = None
        for req in cu_traffic:
            if req.endpoint_identity == tool.endpoint_identity and req.status_code < 400:
                matching_req = req
                break

        if not matching_req or not matching_req.request_body_parsed:
            self._log(
                "ab_learning_skip",
                f"No matching CU traffic for {tool_name}",
                detail={"tool_name": tool_name, "subtask": subtask[:100]},
            )
            return

        cu_body = matching_req.request_body_parsed
        if not isinstance(cu_body, dict):
            return

        # Ingest CU response for downstream chaining
        if matching_req.response_body_parsed:
            self.registry.ingest_api_response(
                matching_req.url, matching_req.response_body_parsed,
            )
            # Cache for chaining
            if tool.endpoint_identity:
                self._response_cache[tool.endpoint_identity] = (
                    matching_req.response_body_parsed
                )

        # Compare MCP failed params vs CU correct params
        mcp_params = failed_mcp_result.get("sent_params", {})
        corrections: list[str] = []

        # Re-trace sources for the correct CU body
        cu_sources = self._trace_parameter_sources(cu_body)
        cu_param_sources = {
            path: [asdict(s) for s in src_list]
            for path, src_list in cu_sources.items()
        }

        for path, correct_value in _iter_json_leaves(cu_body):
            param_name = path.split(".")[-1]
            mcp_value = mcp_params.get(param_name)
            if str(correct_value) != str(mcp_value):
                corrections.append(
                    f"{param_name}: MCP='{str(mcp_value)[:40]}' → "
                    f"CU='{str(correct_value)[:40]}'"
                )
                # Rebuild extraction step for this parameter
                sources = cu_param_sources.get(path, [])
                new_step = self._build_extraction_step(path, sources, correct_value)
                # Replace existing step in recipe
                tool.extraction_recipe = [
                    new_step if s.get("param_path") == path else s
                    for s in tool.extraction_recipe
                ]
                # Add if not already present
                if not any(s.get("param_path") == path for s in tool.extraction_recipe):
                    tool.extraction_recipe.append(new_step)

        # Merge correct body into schema
        tool.parameter_schema = self._merge_schema_observation(
            tool.parameter_schema, cu_body,
        )

        # Merge correct response into response template
        if matching_req.response_body_parsed:
            tool.response_template = self._merge_response_template(
                tool.response_template, matching_req.response_body_parsed,
            )

        # Merge parameter sources
        for path, src_list in cu_param_sources.items():
            if path not in tool.parameter_sources:
                tool.parameter_sources[path] = src_list

        # Store corrected example
        tool.examples.append({
            "request_body": cu_body,
            "query_params": dict(parse_qs(urlparse(matching_req.url).query)),
            "response_status": matching_req.status_code,
            "response_body_excerpt": (
                json.dumps(matching_req.response_body_parsed, default=str)[:500]
                if matching_req.response_body_parsed else ""
            ),
            "timestamp": matching_req.timestamp,
            "source": "cu_fallback",
        })
        if len(tool.examples) > 5:
            tool.examples = tool.examples[-5:]

        self._save_tools()

        self._log(
            "ab_learning_complete",
            f"A/B learning for {tool_name}: {len(corrections)} corrections",
            detail={
                "tool_name": tool_name,
                "corrections": corrections,
                "subtask": subtask[:100],
            },
        )
