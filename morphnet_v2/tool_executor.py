"""morphnet_v2/tool_executor.py — replay tools at execution time.

Phase 5 of morphnet_v2. Loads a `ToolCandidate` from the site's `tools.json`,
takes slot values from the planner, resolves chained slots (from prior tool
responses in the same task OR from the live page), dispatches the HTTP call
via the existing curl_cffi session (which already carries captured cookies),
returns a ReplayResult.

Design principles (per plan):
- Reuse the captured request template; only varying slot values are
  substituted at call time.
- Cookies / headers come from the live SessionManager — no need to bake
  per-request cookies into the tool.
- List-selector LLM picks one when chain source is list-shaped.
- Lifecycle update on success/failure feeds back into ToolRegistry's
  deterministic lifecycle (verified / failing / discarded).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import jinja2

from morphnet_v2.tool_builder import (
    ToolCandidate,
    SlotSource,
    _container_signature,
)

if TYPE_CHECKING:
    from morphnet_v2.session_manager import SessionManager
    from morphnet_v2.page_filter import PageFilterOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# ReplayResult — what the executor returns to the orchestrator
# ─────────────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    http_status: int
    body: str
    body_path: str | None
    error: str | None = None
    resolved_slot_values: dict[str, Any] = field(default_factory=dict)


_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> jinja2.Template:
    return jinja2.Template((_PROMPT_DIR / name).read_text(encoding="utf-8"))


LIST_SELECTOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chosen_index": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["chosen_index"],
}


# ─────────────────────────────────────────────────────────────────
# ToolExecutor
# ─────────────────────────────────────────────────────────────────

class ToolExecutor:
    """Replays tools at execution time. One instance per SessionManager.

    Owns:
    - `_registry_lookup`: cluster_key → ToolCandidate (loaded from tools.json)
    - `_per_task_responses`: cluster_key → most recent ReplayResult body (JSON)
      for chained lookups within this task
    """

    def __init__(self, sm: "SessionManager", tools_by_id: dict[str, ToolCandidate]) -> None:
        self._sm = sm
        self._tools_by_id = tools_by_id
        # Lookup by cluster_key for chain resolution
        self._tools_by_cluster: dict[str, ToolCandidate] = {
            tool.cluster_key: tool for tool in tools_by_id.values()
        }
        self._per_task_responses: dict[str, dict] = {}     # cluster_key → parsed JSON or {"_html": text}

    # ── public API ─────────────────────────────────────────────

    async def replay(self, tool_id: str, planner_slot_values: dict[str, Any]) -> ReplayResult:
        """Resolve slots, dispatch HTTP, return ReplayResult.

        `planner_slot_values` carries the planner's choices for user_intent
        and any chained-slot pre-resolved values. The executor fills in the
        rest from chain lookups + captured constants.
        """
        tool = self._tools_by_id.get(tool_id)
        if tool is None:
            return ReplayResult(http_status=0, body="", body_path=None,
                                 error=f"tool_id {tool_id!r} not in registry")

        # Resolve every slot
        resolved: dict[str, Any] = {}
        try:
            for name, slot in tool.slots.items():
                resolved[name] = await self._resolve_slot(name, slot, planner_slot_values, tool)
        except Exception as e:
            return ReplayResult(http_status=0, body="", body_path=None,
                                 error=f"slot resolution failed: {e!r}",
                                 resolved_slot_values=resolved)

        # Build request URL / body using captured template + resolved slots
        try:
            method, url, headers, body = self._build_request(tool, resolved)
        except Exception as e:
            return ReplayResult(http_status=0, body="", body_path=None,
                                 error=f"request build failed: {e!r}",
                                 resolved_slot_values=resolved)

        # Dispatch via the existing curl_cffi session (cookies attached)
        result = await self._dispatch(method, url, headers, body)
        result.resolved_slot_values = resolved
        # Cache response for downstream chain resolution
        if result.http_status == 200 and result.body:
            try:
                self._per_task_responses[tool.cluster_key] = json.loads(result.body)
            except (json.JSONDecodeError, ValueError):
                self._per_task_responses[tool.cluster_key] = {"_html": result.body}
        return result

    # ── slot resolution ───────────────────────────────────────

    async def _resolve_slot(
        self,
        slot_name: str,
        slot: SlotSource,
        planner_values: dict[str, Any],
        tool: ToolCandidate,
    ) -> Any:
        """Return the slot value for replay-time dispatch."""
        # Planner may pre-resolve any slot — if so, use that verbatim
        if slot_name in planner_values:
            return planner_values[slot_name]

        if slot.kind == "captured":
            # Use first observed value
            if slot.observed_values:
                return slot.observed_values[0]
            return ""

        if slot.kind == "user_intent_text":
            # Planner should have supplied; fall back to first captured example
            if slot.captured_examples:
                logger.warning("user_intent_text slot %s not supplied; falling back to first example", slot_name)
                return slot.captured_examples[0]
            return ""

        if slot.kind == "chained":
            return await self._resolve_chained(slot_name, slot, planner_values, tool)

        return ""

    async def _resolve_chained(
        self,
        slot_name: str,
        slot: SlotSource,
        planner_values: dict[str, Any],
        tool: ToolCandidate,
    ) -> Any:
        """Resolve a chained slot. Source can be:
        - "live_page": run page_filter(enumerate_mode=True), list-selector picks one
        - <cluster_key>: look up that upstream tool's response on this task
        """
        src = slot.chain_source
        if src == "live_page":
            return await self._resolve_live_page(slot_name, slot, planner_values)
        if src and src in self._per_task_responses:
            return self._resolve_from_response(slot_name, slot, self._per_task_responses[src], planner_values)
        # Upstream not yet called — invoke it first if it's in the registry
        if src and src in self._tools_by_cluster:
            upstream = self._tools_by_cluster[src]
            logger.info("recursively invoking upstream tool %s for slot %s", upstream.tool_id, slot_name)
            up_result = await self.replay(upstream.tool_id, planner_values)
            if up_result.http_status == 200:
                return self._resolve_from_response(slot_name, slot, self._per_task_responses.get(src, {}), planner_values)
        raise RuntimeError(f"chained slot {slot_name!r} from {src!r} could not be resolved")

    async def _resolve_live_page(
        self,
        slot_name: str,
        slot: SlotSource,
        planner_values: dict[str, Any],
    ) -> Any:
        """Run page_filter(enumerate_mode=True) on the current live page, filter
        containers by signature, list_selector picks one, extract attribute."""
        from morphnet_v2.page_filter import PageFilter, PageSnapshot
        title = await self._sm.page.title()
        html = await self._sm.page.content()
        snapshot = PageSnapshot(url=self._sm.page.url, title=title, html=html)
        pf = PageFilter(self._sm)
        out = await pf.run(snapshot, enumerate_mode=True)
        # Filter containers matching the signature
        target_sig = slot.container_signature or ""
        candidates = []
        for c in out.containers:
            sig = _container_signature(c)
            if not target_sig or _signature_matches(sig, target_sig):
                if (c.data_attributes or {}).get(slot.html_attribute or ""):
                    candidates.append(c)
        if not candidates:
            raise RuntimeError(f"live_page chain: no containers match signature {target_sig!r}")
        if len(candidates) == 1:
            return (candidates[0].data_attributes or {}).get(slot.html_attribute or "")
        # Multiple candidates → list-selector LLM
        idx = await self._list_select(slot_name, slot, candidates, planner_values)
        chosen = candidates[idx]
        return (chosen.data_attributes or {}).get(slot.html_attribute or "")

    def _resolve_from_response(
        self,
        slot_name: str,
        slot: SlotSource,
        response_data: dict,
        planner_values: dict[str, Any],
    ) -> Any:
        """Look up a value inside a previously cached response. The jmespath
        may end with a list element; if so we may need list_selector."""
        if isinstance(response_data, dict) and "_html" in response_data:
            # HTML response cached — can't navigate it offline here; rely on
            # html_attribute + container_signature. Fall through to live_page
            # semantics if the slot has those.
            raise RuntimeError(f"chained slot {slot_name!r} needs HTML re-extraction; not yet implemented")
        path = slot.response_jmespath or ""
        if not path:
            raise RuntimeError(f"chained slot {slot_name!r} has no response_jmespath")
        # Simple JMESPath-ish traversal: ".data.stationList[0].stationCode"
        # We support: dot segments + [N] (numeric) + [*] (any/all → flatten)
        return _eval_simple_path(response_data, path)

    async def _list_select(
        self,
        slot_name: str,
        slot: SlotSource,
        candidates: list[Any],
        planner_values: dict[str, Any],
    ) -> int:
        """One Gemini Flash call. Returns 0-based index into candidates."""
        # Build a summary per candidate from its data_attributes + nearest text
        cands_repr: list[dict] = []
        for c in candidates[:30]:    # cap at 30 for prompt size
            attrs = c.data_attributes or {}
            summary_bits = [f"{k}={v[:60]}" for k, v in list(attrs.items())[:6]]
            cands_repr.append({"summary": " ".join(summary_bits) or "(no attrs)"})
        tpl = _load_prompt("list_selector.j2")
        prompt = tpl.render(
            user_intent=planner_values.get("_user_task") or planner_values.get("task") or "(planner intent unknown)",
            tool_id=planner_values.get("_tool_id", ""),
            slot_name=slot_name,
            slot_description=planner_values.get(f"_slot_desc_{slot_name}") or slot.html_attribute or "",
            candidates=cands_repr,
        )
        try:
            result = await self._sm.call_gemini(
                model="gemini-3-flash-preview",
                contents=[prompt],
                response_schema=LIST_SELECTOR_RESPONSE_SCHEMA,
                temperature=0.0,
                thinking_budget=512,
                max_output_tokens=512,
            )
            if isinstance(result, dict):
                idx_1 = int(result.get("chosen_index", 1) or 1)
                # 1-based in prompt → 0-based in array
                return max(0, min(len(candidates) - 1, idx_1 - 1))
        except Exception as e:
            logger.warning("list_selector failed: %s", e)
        return 0

    # ── request build + dispatch ──────────────────────────────

    def _build_request(
        self,
        tool: ToolCandidate,
        resolved: dict[str, Any],
    ) -> tuple[str, str, dict[str, str], str | None]:
        """Build (method, url, headers, body) from the captured template
        substituting in resolved slot values."""
        sr = tool.sample_request
        method = sr.method
        url = sr.url
        headers = {k: v for k, v in (sr.headers or {}).items()
                   if k.lower() not in ("host", "content-length", "connection", "cookie")}

        # For REST: rebuild query string from constants + slot values
        sp = urlsplit(url)
        query_params = dict(parse_qsl(sp.query, keep_blank_values=True))
        # Overlay constants first (these stay the same)
        for k, v in tool.constants.items():
            if "." in k:    # body field — skip for now
                continue
            query_params[k] = str(v)
        # Overlay resolved slot values
        for k, v in resolved.items():
            if k.startswith("variables."):    # GraphQL body slot
                continue
            if v is None:
                continue
            query_params[k] = str(v)
        new_query = urlencode(query_params)
        url = urlunsplit((sp.scheme, sp.netloc, sp.path, new_query, sp.fragment))

        body: str | None = None
        if tool.dispatch_kind in ("graphql", "json_rpc", "form_post") and sr.body_preview:
            # Reconstruct body from preview by substituting variables.* slots
            try:
                b = json.loads(sr.body_preview)
            except (json.JSONDecodeError, ValueError):
                b = None
            if isinstance(b, dict):
                # Update variables.* keys
                vars_dict = b.get("variables") if isinstance(b.get("variables"), dict) else {}
                for k, v in resolved.items():
                    if k.startswith("variables."):
                        leaf = k.split(".", 1)[1]
                        vars_dict[leaf] = v
                if vars_dict:
                    b["variables"] = vars_dict
                body = json.dumps(b)
            else:
                body = sr.body_preview

        return method, url, headers, body

    async def _dispatch(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
    ) -> ReplayResult:
        """Fire the HTTP request via curl_cffi session (cookies attached)."""
        try:
            sess = await self._sm.make_http_session()
        except Exception as e:
            return ReplayResult(http_status=0, body="", body_path=None,
                                 error=f"make_http_session failed: {e!r}")
        try:
            kwargs: dict[str, Any] = {"headers": headers, "timeout": 30}
            if body is not None:
                kwargs["data"] = body
            resp = await asyncio_to_thread(sess.request, method, url, **kwargs)
            status = int(resp.status_code)
            text = resp.text or ""
            return ReplayResult(http_status=status, body=text, body_path=None)
        except Exception as e:
            return ReplayResult(http_status=0, body="", body_path=None,
                                 error=f"dispatch failed: {e!r}")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _signature_matches(actual: str, target: str) -> bool:
    """Check if container signatures align. Both are produced by
    `_container_signature` — they should be identical for matched cards."""
    return actual == target


_PATH_SEG = re.compile(r"\[(\d+|\*)\]|\.([A-Za-z_][A-Za-z0-9_]*)")


def _eval_simple_path(obj: Any, path: str) -> Any:
    """Evaluate a simple dotted/bracketed path like `.data.stationList[0].stationCode`.
    Returns the first match if path has [N]; the full list if [*]."""
    segments = _PATH_SEG.findall(path)
    cur: Any = obj
    for idx, key in segments:
        if key:    # .name segment
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
        elif idx == "*":
            if not isinstance(cur, list):
                return cur
            # Flatten the rest of the path across all items — for now just take first
            return cur[0] if cur else None
        else:
            i = int(idx)
            if isinstance(cur, list) and 0 <= i < len(cur):
                cur = cur[i]
            else:
                return None
    return cur


async def asyncio_to_thread(fn, *args, **kwargs):
    """asyncio.to_thread shim (Python 3.9+ has it; we just call it)."""
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)


# ─────────────────────────────────────────────────────────────────
# Loading utilities — read tools.json from disk
# ─────────────────────────────────────────────────────────────────

def load_tools_for_site(site: str) -> dict[str, ToolCandidate]:
    """Read morphnet_v2/sites/{site}/tools.json into {tool_id: ToolCandidate}.
    Returns empty dict if file doesn't exist."""
    from morphnet_v2.tool_builder import _rehydrate_candidate
    path = Path(__file__).parent / "sites" / site / "tools.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}
    out: dict[str, ToolCandidate] = {}
    for d in raw:
        cand = _rehydrate_candidate(d)
        if cand.tool_id:
            out[cand.tool_id] = cand
    return out


__all__ = [
    "ReplayResult",
    "ToolExecutor",
    "load_tools_for_site",
]
