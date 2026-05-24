"""morphnet_v3/tool_executor.py — replay orchestrator (Chunk 8).

Reads `morphnet_v3/sites/<site>/tools.json` (emitted by tool_builder.py:§G3)
and dispatches HTTP calls per the recipes therein. Two entry modes:

  - **Production replay**: open a fresh browser session, navigate to entry_url,
    let the SPA hydrate (JS-render precursor), then per-tool: materialize slots
    → assemble request → fire via curl_cffi → return ToolResult.

  - **Attached mode** (`attached=True`): used by Chunk-7 verification. Skips
    `open_session()` and trusts the current browser state (cookies + localStorage
    + V8 already populated from the discovery session that just ended).

Source-bucket materializers are the heart of this module — one handler per
sub_type. They translate `tools.json` recipes into live values at replay:

  cookie:<name>            → read from live jar
  session_state:<key>      → read from live localStorage / sessionStorage
  bundle:apq_hash          → re-derive via sha256(live_bundle_query_text)
  bundle:literal/enum/...  → use captured value (recipe IS the literal)
  generated:traceparent    → fresh per call
  generated:timestamp_ms   → int(time.time()*1000)
  generated:viewport_wxh   → constant matching the captured Chrome viewport
  generated:user_agent     → curl_cffi's impersonation UA
  chained:resp_of(t, ...)  → from `self._per_task_responses[t]` cache
  user_intent:typed        → from `planner_slots` (planner generated it)
  user_intent:click        → from `planner_slots` OR derived live via click_recipe

Per the 8-point thesis: tools.json holds RECIPES only. The executor never
reads "captured values" except `bundle:*` (where the captured literal IS the
recipe) and the optional planner-reference `examples[0]`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if TYPE_CHECKING:
    from morphnet_v3.session_manager import SessionManager  # type: ignore[import-not-found]


# ── ToolResult dataclass ─────────────────────────────────────────────────


@dataclass
class ToolResult:
    """The unit returned by ToolExecutor.invoke. Surfaced to the planner."""

    tool_id: str
    http_status: int
    body: str = ""
    parsed: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    fall_back_to_cu: bool = False
    response_summary: str = ""
    resolved_slots: dict[str, Any] = field(default_factory=dict)


# ── Anti-replay signatures ──────────────────────────────────────────────
#
# When a server detects a bot-replay it usually serves a challenge page (HTML)
# or a stub JSON. These signatures route the result into `fall_back_to_cu=True`
# so the planner re-routes the step through CU.

_ANTI_BOT_SIGNATURES = (
    "Just a moment...",           # Cloudflare challenge title
    "Checking if you are human",  # Cloudflare interstitial
    "Please enable JavaScript",   # Generic JS gate
    "aws-waf-token",              # AWS WAF cookie reference
    "captcha-delivery.com",       # DataDome
    "px-captcha",                 # PerimeterX
    "<title>Access Denied</title>",
)


def _detect_anti_replay(body: str | None, status: int) -> bool:
    if status in (403, 429):
        return True
    if not body:
        return False
    head = body[:4000].lower()
    for sig in _ANTI_BOT_SIGNATURES:
        if sig.lower() in head:
            return True
    return False


# ── ToolExecutor ─────────────────────────────────────────────────────────


class ToolExecutor:
    """Per-task replay orchestrator. One instance per SessionManager."""

    def __init__(
        self,
        sm: SessionManager,
        site: str,
        attached: bool = False,
        tools_dir: Path | None = None,
    ) -> None:
        self._sm = sm
        self._site = site
        self._attached = attached
        # Per-task chain cache: tool_id → parsed response body. Drives
        # `chained_resp` materialization for downstream tools in the same task.
        self._per_task_responses: dict[str, Any] = {}
        # list_selector decision cache (Q4 in plan): keyed by
        # (task_text, tool_id, slot_name, candidate_fingerprint).
        self._list_select_cache: dict[tuple[str, str, str, str], int] = {}
        # Loaded tools — raw dicts (no rehydrate into Tool dataclass; this
        # avoids importing tool_builder at executor load time).
        if tools_dir is None:
            tools_dir = Path(__file__).resolve().parent / "sites" / site
        self._tools_path = tools_dir / "tools.json"
        self._site_dir = tools_dir
        self._tools_by_id: dict[str, dict] = {}
        self._entry_url: str = ""
        self._page_class: str | None = None
        self._load_tools()

    def _load_tools(self) -> None:
        if not self._tools_path.exists():
            return
        try:
            doc = json.loads(self._tools_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return
        self._entry_url = doc.get("entry_url") or ""
        self._page_class = doc.get("page_class")
        for tool in doc.get("tools", []):
            tid = tool.get("tool_id")
            if tid:
                self._tools_by_id[tid] = tool

    # ── Public API ───────────────────────────────────────────────────────

    async def open_session(self) -> None:
        """Production-mode entry: navigate to entry_url + wait for the
        bundle to hydrate. Skipped on attached=True (verification reuses the
        discovery browser's already-warmed state)."""
        if self._attached or not self._entry_url:
            return
        await self._sm.page.goto(self._entry_url)
        await self._sm.wait_for_page_ready()

    def reset_per_task(self) -> None:
        """Clear chain + list-selector caches at task boundary."""
        self._per_task_responses.clear()
        self._list_select_cache.clear()

    def list_ids(self) -> list[str]:
        return list(self._tools_by_id.keys())

    def get(self, tool_id: str) -> dict | None:
        return self._tools_by_id.get(tool_id)

    async def invoke(
        self, tool_id: str, planner_slots: dict[str, Any], task_text: str = "",
    ) -> ToolResult:
        """End-to-end: materialize → build → dispatch → parse → return."""
        tool = self._tools_by_id.get(tool_id)
        if tool is None:
            return ToolResult(tool_id=tool_id, http_status=0, error=f"tool_id {tool_id!r} not in registry",
                              fall_back_to_cu=True)

        try:
            resolved = await self._materialize_slots(tool, planner_slots, task_text)
        except Exception as exc:
            return ToolResult(tool_id=tool_id, http_status=0,
                              error=f"slot materialization failed: {exc!r}", fall_back_to_cu=True)

        try:
            method, url, headers, body = self._build_request(tool, resolved)
        except Exception as exc:
            return ToolResult(tool_id=tool_id, http_status=0, error=f"request build failed: {exc!r}",
                              fall_back_to_cu=True, resolved_slots=resolved)

        result = await self._dispatch(method, url, headers, body)
        result.tool_id = tool_id
        result.resolved_slots = resolved

        # Cache the parsed response for downstream chains in this task.
        if result.http_status == 200 and result.parsed is not None:
            self._per_task_responses[tool_id] = result.parsed

        # Anti-replay detection — flip fall_back_to_cu if we got a challenge page.
        if _detect_anti_replay(result.body, result.http_status):
            result.fall_back_to_cu = True
            result.error = result.error or "anti-replay HTML detected"

        return result

    # ── Materializer ─────────────────────────────────────────────────────

    async def _materialize_slots(
        self, tool: dict, planner_slots: dict[str, Any], task_text: str,
    ) -> dict[str, Any]:
        """Walk each slot in `tool["slots"]` and produce a {slot_name: value}
        dict the request builder can substitute. Dispatches per source bucket."""
        out: dict[str, Any] = {}
        slots = tool.get("slots") or {}
        for name, slot in slots.items():
            # Planner-pre-resolved (typed / explicit override) wins.
            if name in planner_slots:
                out[name] = planner_slots[name]
                continue
            value = await self._materialize_one(name, slot, tool, planner_slots, task_text)
            if value is not None:
                out[name] = value
        return out

    async def _materialize_one(
        self, name: str, slot: dict, tool: dict, planner_slots: dict[str, Any], task_text: str,
    ) -> Any:
        """Per-slot bucket dispatch."""
        # 1. User-intent (typed) — must come from planner_slots
        intent = slot.get("intent")
        if intent and intent.get("kind") == "user_intent:typed":
            return planner_slots.get(name) or (slot.get("examples", [None])[0])
        # 2. User-intent (click) — derive live from the page via describe_target / DOM query
        if intent and intent.get("kind") == "user_intent:click":
            return await self._resolve_click_recipe(intent.get("click_recipe") or {}, planner_slots, name, slot)
        # 3. Chained from a prior tool's response
        chain = slot.get("chain")
        if chain:
            return await self._resolve_chained(chain, planner_slots, name, slot, tool, task_text)
        # 4. source-bucket dispatch
        src = slot.get("source") or {}
        bucket = src.get("bucket") or ""
        sub_type = src.get("sub_type") or ""
        recipe = src.get("recipe") or {}
        if bucket == "cookie":
            return await self._mat_cookie(recipe, sub_type)
        if bucket == "session_state":
            return await self._mat_session_state(recipe, sub_type)
        if bucket == "generated":
            return self._mat_generated(sub_type)
        if bucket == "bundle":
            return await self._mat_bundle(src, sub_type, tool)
        if bucket == "adapter_injected":
            return self._mat_adapter_injected(recipe, sub_type)
        # Unknown / trivial / fixed_by_capture — fall back to the captured
        # example. For fixed_by_capture this IS the recipe (we observed the
        # value never varies across the cluster). For unknown it's the best
        # we have until the verifier surfaces drift.
        if src.get("value") is not None:
            return src.get("value")
        examples = slot.get("examples") or []
        return examples[0] if examples else None

    # — cookie / state —

    async def _mat_cookie(self, recipe: dict, sub_type: str) -> str | None:
        """Read from the live Playwright cookie jar."""
        name = recipe.get("name") or sub_type.split(":")[0]
        if not name:
            return None
        cookies = await self._sm.cookies_snapshot()
        for c in cookies:
            if c.get("name") == name:
                return c.get("value")
        return None

    async def _mat_session_state(self, recipe: dict, sub_type: str) -> str | None:
        """Read from live localStorage / sessionStorage via page.evaluate."""
        storage = recipe.get("storage") or "localStorage"
        key_value = recipe.get("key") or recipe.get("key_hint") or sub_type
        if not key_value:
            return None
        key = str(key_value)
        try:
            val = await self._sm.page.evaluate(
                f"(k) => window.{storage}.getItem(k)", key,
            )
        except Exception:
            return None
        return val

    # — adapter-injected (e.g. webarena auto-login headers) —

    def _mat_adapter_injected(self, recipe: dict, sub_type: str) -> str | None:
        """Look up the header's expected value in the adapter's registry on
        the SessionManager. `sub_type` is the lowercase header name set by
        the tracer; `recipe.header_name` carries the original case."""
        header_name = recipe.get("header_name") or sub_type
        if not header_name:
            return None
        registry = self._sm.injected_headers or {}
        for k, v in registry.items():
            if k.lower() == str(header_name).lower():
                return v
        return None

    # — generated —

    def _mat_generated(self, sub_type: str) -> str | None:
        if sub_type == "traceparent":
            trace_id = "".join(random.choices("0123456789abcdef", k=32))
            span_id = "".join(random.choices("0123456789abcdef", k=16))
            return f"00-{trace_id}-{span_id}-01"
        if sub_type == "timestamp_ms":
            return str(int(time.time() * 1000))
        if sub_type == "timestamp_s":
            return str(int(time.time()))
        if sub_type == "viewport_wxh":
            # SessionManager's viewport is set at start; use that.
            w, h = self._sm.viewport
            return f"{w}x{h}"
        if sub_type == "user_agent":
            # Return None — curl_cffi's impersonate="chrome" sets a coherent
            # User-Agent for the impersonated browser version. The previous
            # implementation incorrectly returned `.__await__()` (a generator
            # object) which str()ed into garbage. Letting impersonation handle
            # this is the canonical fix; if a tool needs a SPECIFIC UA, it
            # belongs as a user_intent slot, not as generated:user_agent.
            return None
        return None

    # — bundle —

    async def _mat_bundle(self, src: dict, sub_type: str, tool: dict) -> Any:
        """Bundle sub_types where the recipe IS the captured literal: use
        `source.value` verbatim. Special case: bundle:apq_hash recomputes
        sha256 of the live bundle's query text for the operation."""
        if sub_type == "apq_hash":
            # The recipe says: re-derive sha256(bundle_query_text_for_operation).
            # Extract operation name from the tool's cluster_key (the last
            # segment of path is the GraphQL op for graphql_apq tools).
            op_name = self._extract_op_name(tool)
            query_text = await self._find_query_text_in_live_bundle(op_name)
            if query_text:
                return hashlib.sha256(query_text.encode("utf-8")).hexdigest()
            # Fallback: return the captured hash if we couldn't re-derive.
            return src.get("value")
        # All other bundle:* — use the captured literal directly.
        return src.get("value")

    def _extract_op_name(self, tool: dict) -> str:
        """For graphql/graphql_apq tools, the operation name is the last URL
        path segment (e.g. /api/graphql/ProductDetails → 'ProductDetails')."""
        url = (tool.get("endpoint") or {}).get("url_template") or ""
        path = urlsplit(url).path or ""
        segs = [s for s in path.split("/") if s]
        return segs[-1] if segs else ""

    async def _find_query_text_in_live_bundle(self, op_name: str) -> str | None:
        """Grep loaded scripts in the live page for the GraphQL query text of
        `op_name`. Looks for `query OpName(...)` or `mutation OpName(...)`
        followed by braces, then collects the balanced-brace body."""
        if not op_name:
            return None
        try:
            sources: list[str] = await self._sm.page.evaluate(
                """() => {
                    const out = [];
                    for (const sc of document.querySelectorAll('script')) {
                        const t = sc.textContent;
                        if (t) out.push(t);
                    }
                    return out;
                }""",
            )
        except Exception:
            return None
        for src in sources:
            text = _find_graphql_op_text(src, op_name)
            if text:
                return text
        return None

    # — chained —

    async def _resolve_chained(
        self, chain: dict, planner_slots: dict, name: str, slot: dict,
        tool: dict, task_text: str,
    ) -> Any:
        """Resolve from upstream tool's cached response. Direct edge → single
        extraction. list_select edge → LLM picks 1-based index then extract."""
        source_tool_id = chain.get("source_tool_id") or ""
        source_resp = self._per_task_responses.get(source_tool_id) if source_tool_id else None
        if source_resp is None:
            return None  # Upstream not run yet; caller decides if this is fatal.
        kind = chain.get("kind") or "direct"
        per_item = chain.get("per_item_extract") or {}
        if kind == "direct":
            # Single candidate — walk per_item_extract directly against the response.
            return _extract_from_value(source_resp, per_item)
        # list_select — enumerate candidates, ask LLM (or trivial-skip if 1)
        list_jp = chain.get("list_jsonpath") or "$"
        candidates = _resolve_jsonpath_list(source_resp, list_jp)
        if not candidates:
            return None
        selector = chain.get("selector_recipe") or {}
        if len(candidates) == 1 and selector.get("trivial_if_n_eq_1"):
            return _extract_from_value(candidates[0], per_item)
        # LLM list-selector. Pass full upstream/downstream context so the LLM
        # can reason about the chain, not just the candidates in isolation.
        fingerprint = self._fingerprint_candidates(candidates, selector.get("candidate_fields") or [])
        cache_key = (task_text, tool.get("tool_id", ""), name, fingerprint)
        if cache_key in self._list_select_cache:
            idx = self._list_select_cache[cache_key]
        else:
            upstream_tool = self._tools_by_id.get(source_tool_id) or {}
            idx = await self._call_list_selector(
                task_text=task_text,
                slot_name=name,
                slot_description=slot.get("description", ""),
                candidates=candidates,
                candidate_fields=selector.get("candidate_fields") or [],
                downstream_tool_id=tool.get("tool_id", ""),
                downstream_tool_capability=tool.get("capability_statement", ""),
                upstream_tool_id=source_tool_id,
                upstream_tool_capability=upstream_tool.get("capability_statement", ""),
            )
            self._list_select_cache[cache_key] = idx
        if 1 <= idx <= len(candidates):
            return _extract_from_value(candidates[idx - 1], per_item)
        return None

    def _fingerprint_candidates(self, candidates: list, fields: list[str]) -> str:
        """Stable hash over the candidate set's distinguishing fields. Same
        fingerprint → cached selector index reused."""
        items = []
        for c in candidates[:50]:
            if isinstance(c, dict):
                items.append({f: str(c.get(f, ""))[:80] for f in fields})
            else:
                items.append(str(c)[:80])
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    async def _call_list_selector(
        self,
        task_text: str,
        slot_name: str,
        slot_description: str,
        candidates: list,
        candidate_fields: list[str],
        downstream_tool_id: str = "",
        downstream_tool_capability: str = "",
        upstream_tool_id: str = "",
        upstream_tool_capability: str = "",
    ) -> int:
        """LLM call via list_selector.j2 → 1-based chosen index.

        Uses Gemini's FUNCTION-CALLING API (not free-form structured output) —
        the LLM MUST invoke `pick_candidate(chosen_index, reason)` exactly once.
        Richer context passed: upstream/downstream tool ids + capabilities so
        the LLM can reason about the chain, not just the candidates in isolation.

        Format-correctness is NOT this LLM's job — the deterministic
        `per_item_extract` recipe handles that downstream. The LLM only picks
        the index."""
        import jinja2
        from google.genai import types as genai_types

        tmpl_path = Path(__file__).resolve().parent / "prompts" / "list_selector.j2"
        tmpl = jinja2.Template(
            tmpl_path.read_text(encoding="utf-8"), trim_blocks=True, lstrip_blocks=True,
        )
        rendered_candidates: list[dict] = []
        for c in candidates:
            if isinstance(c, dict):
                fields_to_show = candidate_fields or list(c.keys())[:6]
                rendered_candidates.append({f: c.get(f) for f in fields_to_show})
            else:
                rendered_candidates.append({"value": c})
        prompt = tmpl.render(
            user_intent=task_text,
            slot_name=slot_name,
            slot_description=slot_description,
            candidates=rendered_candidates,
            downstream_tool_id=downstream_tool_id,
            downstream_tool_capability=downstream_tool_capability,
            upstream_tool_id=upstream_tool_id,
            upstream_tool_capability=upstream_tool_capability,
        )

        # Function-calling — Gemini MUST invoke this function with valid args.
        T = genai_types.Type
        pick_candidate_decl = genai_types.FunctionDeclaration(
            name="pick_candidate",
            description=(
                "Pick exactly one candidate from the Candidates list to satisfy "
                "the user's intent for this slot. Format-correctness is handled "
                "deterministically by the runtime — you only choose the index."
            ),
            parameters=genai_types.Schema(
                type=T.OBJECT,
                properties={
                    "chosen_index": genai_types.Schema(
                        type=T.INTEGER,
                        description=f"1-based index into the Candidates list (1 to {len(rendered_candidates)}).",
                    ),
                    "reason": genai_types.Schema(
                        type=T.STRING,
                        description="One short sentence explaining the choice.",
                    ),
                },
                required=["chosen_index", "reason"],
            ),
        )
        tools_arg = [genai_types.Tool(function_declarations=[pick_candidate_decl])]
        contents = [genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=prompt)],
        )]

        resp = await self._sm.call_gemini(
            model="gemini-3-flash-preview",
            contents=contents,
            tools=tools_arg,
            temperature=0.0,
        )

        # Parse the function_call from the response. The google.genai
        # GenerateContentResponse protocol always defines `candidates` and
        # Part.function_call (None when absent) — explicit attribute access
        # is correct without hasattr guards.
        cands = resp.candidates or []
        if not cands:
            return 1
        content = cands[0].content
        parts = (content.parts if content is not None else None) or []
        for p in parts:
            fn = p.function_call
            if fn is None:
                continue
            args = dict(fn.args or {})
            raw_idx = args.get("chosen_index")
            try:
                return int(raw_idx) if raw_idx is not None else 1
            except (TypeError, ValueError):
                return 1
        return 1

    # — click recipe (replay-time, uses describe_target on the live DOM) —

    async def _resolve_click_recipe(
        self, click_recipe: dict, planner_slots: dict, name: str, slot: dict,
    ) -> Any:
        """Find an element matching the recipe on the live page; extract value."""
        selector = click_recipe.get("selector")
        attribute = click_recipe.get("attribute") or "text"
        extract_regex = click_recipe.get("extract_regex")
        if not selector:
            # No selector — planner must provide the value directly.
            return planner_slots.get(name)
        try:
            attr_val = await self._sm.page.evaluate(
                f"""(args) => {{
                    const [sel, attr] = args;
                    const el = document.querySelector(sel);
                    if (!el) return null;
                    if (attr === 'text') return (el.textContent || '').trim();
                    return el.getAttribute(attr);
                }}""",
                [selector, attribute],
            )
        except Exception:
            return None
        if attr_val is None:
            return None
        if extract_regex:
            m = re.search(extract_regex, str(attr_val))
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return attr_val

    # ── Request builder ─────────────────────────────────────────────────

    def _build_request(
        self, tool: dict, slots: dict[str, Any],
    ) -> tuple[str, str, dict, str | None]:
        """Apply materialized slot values to the endpoint template, producing
        the final (method, url, headers, body) tuple curl_cffi will dispatch."""
        endpoint = tool.get("endpoint") or {}
        method = (endpoint.get("method") or "GET").upper()
        url = endpoint.get("url_template") or ""
        headers = dict(endpoint.get("headers_template") or {})
        body = endpoint.get("body_template")

        # Substitute slot values per parent_path.
        for slot_name, value in slots.items():
            slot_def = (tool.get("slots") or {}).get(slot_name) or {}
            location = slot_def.get("location")
            parent_path = slot_def.get("parent_path") or []
            if location == "header":
                # parent_path = [["header_name", "<name>"]]
                if parent_path and parent_path[0][0] == "header_name":
                    headers[parent_path[0][1]] = str(value)
            elif location == "url_path_segment":
                url = _set_url_path_segment(url, parent_path, str(value))
            elif location == "url_query":
                url = _set_url_query(url, parent_path, str(value))
            elif location == "body":
                body = _set_body(body, parent_path, value)

        # Drop noise headers — curl_cffi adds them via impersonation. Cookie
        # is dropped here because the captured Cookie header would carry
        # discovery-time cookies; live cookies are injected per-request at
        # dispatch via sm.cookies_snapshot() → curl_cffi `cookies=` kwarg.
        for noise in (
            "host", "content-length", "accept", "accept-encoding", "accept-language",
            "connection", "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
            "sec-ch-ua-arch", "sec-ch-ua-bitness", "sec-ch-ua-full-version",
            "sec-ch-ua-full-version-list", "sec-ch-ua-model", "sec-ch-ua-platform-version",
            "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "sec-fetch-user",
            "upgrade-insecure-requests", "user-agent", "cookie", "priority",
        ):
            for k in list(headers.keys()):
                if k.lower() == noise:
                    del headers[k]

        return method, url, headers, body

    # ── Dispatcher (curl_cffi via SessionManager) ────────────────────────

    async def _dispatch(
        self, method: str, url: str, headers: dict, body: str | None,
    ) -> ToolResult:
        """Fire via curl_cffi with Chrome impersonation. The Chrome-default
        headers (UA, Sec-CH-UA*, Accept-*, Sec-Fetch-*) are added by the
        impersonation profile. Cookies are injected here from the LIVE
        Playwright jar — the captured Cookie header was dropped at
        _build_request because it would carry stale discovery-time cookies."""
        try:
            session = await self._sm.make_http_session(impersonate="chrome")
        except Exception as exc:
            return ToolResult(tool_id="", http_status=0, error=f"http session error: {exc!r}",
                              fall_back_to_cu=True)
        try:
            cookies_list = await self._sm.cookies_snapshot()
        except Exception:
            cookies_list = []
        cookies_dict: dict[str, str] = {}
        for c in cookies_list:
            name = c.get("name")
            value = c.get("value")
            if name and value is not None:
                cookies_dict[str(name)] = str(value)
        method_u = method.upper()
        kwargs: dict[str, Any] = {"headers": headers, "cookies": cookies_dict}
        if body is not None and method_u in ("POST", "PUT", "PATCH"):
            kwargs["data"] = body if isinstance(body, str) else json.dumps(body)
        # Explicit method dispatch — preferred over getattr(session, method).
        if method_u == "GET":
            dispatch = session.get
        elif method_u == "POST":
            dispatch = session.post
        elif method_u == "PUT":
            dispatch = session.put
        elif method_u == "PATCH":
            dispatch = session.patch
        elif method_u == "DELETE":
            dispatch = session.delete
        elif method_u == "HEAD":
            dispatch = session.head
        elif method_u == "OPTIONS":
            dispatch = session.options
        else:
            return ToolResult(tool_id="", http_status=0,
                              error=f"unsupported method: {method!r}", fall_back_to_cu=True)
        try:
            # curl_cffi is synchronous; offload to a thread.
            resp = await asyncio.to_thread(dispatch, url, **kwargs)
        except Exception as exc:
            return ToolResult(tool_id="", http_status=0, error=f"dispatch failed: {exc!r}",
                              fall_back_to_cu=True)
        body_text = resp.text or ""
        parsed: Any = None
        try:
            parsed = resp.json()
        except (ValueError, json.JSONDecodeError):
            parsed = None
        return ToolResult(
            tool_id="",
            http_status=resp.status_code,
            body=body_text,
            parsed=parsed,
            headers={k: v for k, v in (resp.headers or {}).items() if v is not None},
            response_summary=body_text[:600],
        )


# ── Adapter for planner.py:_seed_registry_from_site ──────────────────────


def load_tools_for_site(site: str) -> dict[str, dict]:
    """Read morphnet_v3/sites/<site>/tools.json and return {tool_id: tool_dict}.
    Used by planner._seed_registry_from_site; the new shape is dict-based
    (NOT v2 ToolCandidate). Planner.py:_seed_registry_from_site (the adapter)
    walks the slots dict to build ToolEntry+SlotDef."""
    path = Path(__file__).resolve().parent / "sites" / site / "tools.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}
    return {t.get("tool_id"): t for t in doc.get("tools", []) if t.get("tool_id")}


# ═════════════════════════════════════════════════════════════════════════
# Internal helpers — JSONPath walker + URL mutation + GraphQL grep
# ═════════════════════════════════════════════════════════════════════════


def _set_url_path_segment(url: str, parent_path: list, value: str) -> str:
    """parent_path[0] = ["index", N] — replace the Nth path segment."""
    if not parent_path or parent_path[0][0] != "index":
        return url
    idx = int(parent_path[0][1])
    p = urlsplit(url)
    segs = [s for s in (p.path or "/").split("/") if s]
    if 0 <= idx < len(segs):
        segs[idx] = value
        new_path = "/" + "/".join(segs)
        return urlunsplit((p.scheme, p.netloc, new_path, p.query, p.fragment))
    return url


def _set_url_query(url: str, parent_path: list, value: str) -> str:
    """parent_path[0] = ["key", "<name>"] — set/replace the named query param.
    Handles APQ-style nested JSON params: if parent_path[0].value is "variables"
    or "extensions" and there's a deeper path, we re-serialize the JSON."""
    if not parent_path:
        return url
    op, key = parent_path[0]
    if op != "key":
        return url
    p = urlsplit(url)
    qs = parse_qsl(p.query, keep_blank_values=True)
    qs_dict: dict[str, str] = dict(qs)
    if len(parent_path) == 1:
        # Top-level query param
        qs_dict[key] = value
    else:
        # Nested into a JSON-encoded URL param (graphql_apq variables/extensions).
        nested = qs_dict.get(key, "{}")
        try:
            obj = json.loads(nested)
        except (json.JSONDecodeError, ValueError):
            obj = {}
        _set_nested(obj, parent_path[1:], value)
        qs_dict[key] = json.dumps(obj, separators=(",", ":"))
    new_query = urlencode(qs_dict)
    return urlunsplit((p.scheme, p.netloc, p.path, new_query, p.fragment))


def _set_body(body: Any, parent_path: list, value: Any) -> Any:
    """Walk parent_path into a JSON body (str or dict) and set the leaf."""
    if isinstance(body, str):
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return body
        _set_nested(obj, parent_path, value)
        return json.dumps(obj, separators=(",", ":"))
    if isinstance(body, dict):
        _set_nested(body, parent_path, value)
        return body
    return body


def _set_nested(obj: Any, path: list, value: Any) -> None:
    """In-place set: navigate `obj` per path, set the final element."""
    if not path:
        return
    cur = obj
    for op, key in path[:-1]:
        if op == "key" and isinstance(cur, dict):
            cur = cur.setdefault(key, {})
        elif op == "index" and isinstance(cur, list):
            while len(cur) <= int(key):
                cur.append(None)
            if cur[int(key)] is None:
                cur[int(key)] = {}
            cur = cur[int(key)]
        else:
            return
    last_op, last_key = path[-1]
    if last_op == "key" and isinstance(cur, dict):
        cur[last_key] = value
    elif last_op == "index" and isinstance(cur, list):
        while len(cur) <= int(last_key):
            cur.append(None)
        cur[int(last_key)] = value


def _resolve_jsonpath_list(obj: Any, jsonpath: str) -> list:
    """Resolve a `$.a.b[*].c` style jsonpath → list of items. Limited
    implementation: handles `$`, `.<key>`, `[*]`."""
    if not jsonpath or jsonpath == "$":
        return [obj] if not isinstance(obj, list) else obj
    s = jsonpath.lstrip("$.").rstrip(".")
    parts: list[Any] = []
    buf = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == ".":
            if buf:
                parts.append(buf)
                buf = ""
        elif c == "[":
            if buf:
                parts.append(buf)
                buf = ""
            end = s.find("]", i)
            if end == -1:
                return []
            inner = s[i + 1 : end]
            parts.append("[*]" if inner == "*" else int(inner))
            i = end
        else:
            buf += c
        i += 1
    if buf:
        parts.append(buf)
    # Walk
    cur: Any = obj
    for p in parts:
        if p == "[*]":
            if not isinstance(cur, list):
                return []
            # Flatten one level
            nxt: list = []
            for item in cur:
                nxt.append(item)
            cur = nxt
        elif isinstance(p, int):
            if isinstance(cur, list) and 0 <= p < len(cur):
                cur = cur[p]
            else:
                return []
        else:
            if isinstance(cur, list):
                cur = [item.get(p) if isinstance(item, dict) else None for item in cur]
            elif isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return []
    return cur if isinstance(cur, list) else [cur]


def _extract_from_value(value: Any, per_item: dict) -> Any:
    """Apply per_item_extract recipe to a single value (dict or scalar).
    Supports: {extract: 'whole'}, {field: 'url', regex: '...'},
    {field: 'url', path_segment_at: N}, {field: 'url', url_query: 'k'}."""
    if not per_item or per_item.get("extract") == "whole":
        return value
    field_name = per_item.get("field")
    if field_name and isinstance(value, dict):
        value = value.get(field_name)
    if value is None:
        return None
    s = str(value)
    if "regex" in per_item:
        m = re.search(per_item["regex"], s)
        if m:
            return m.group(1) if m.groups() else m.group(0)
        return None
    if "path_segment_at" in per_item and per_item["path_segment_at"] is not None:
        try:
            n = int(per_item["path_segment_at"])
        except (TypeError, ValueError):
            return s
        p = urlsplit(s)
        segs = [seg for seg in (p.path or "/").split("/") if seg]
        return segs[n] if 0 <= n < len(segs) else None
    if "url_query" in per_item:
        p = urlsplit(s)
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            if k == per_item["url_query"]:
                return v
        return None
    return s


_GQL_OP_RE = re.compile(r"(query|mutation|subscription)\s+([A-Z][A-Za-z0-9_]+)")


def _find_graphql_op_text(script_text: str, op_name: str) -> str | None:
    """Scan a script body for `query <op_name>(...)` or `mutation <op_name>(...)`,
    return the full balanced-brace block (for APQ-hash recomputation)."""
    if op_name not in script_text:
        return None
    for m in _GQL_OP_RE.finditer(script_text):
        if m.group(2) != op_name:
            continue
        # Walk forward from match end, find the first `{` then balance braces.
        start = m.start()
        brace_idx = script_text.find("{", m.end())
        if brace_idx == -1:
            continue
        depth = 1
        i = brace_idx + 1
        while i < len(script_text) and depth > 0:
            ch = script_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return script_text[start:i].strip()
    return None