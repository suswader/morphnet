"""
reflector.py — Three-stage verification pipeline for MorphNet.

Stage 1: Deterministic signal harvest (every action, zero LLM cost)
Stage 2: Focused AXTree diff (ambiguous cases only)
Stage 3: LLM semantic evaluation (only when stages 1-2 can't resolve, ~2-3 per subtask)

Separate verification paths for CU actions, MCP calls, and full subtasks.
Every stage logged via trace. Schemas enforce reasoning + evidence + confidence.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from morphnet.session_manager import (
    SessionManager, InteractiveElement, CapturedRequest,
    ActionResult, call_gemini_async,
)
from morphnet.trace import TaskTrace, Evidence

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Sentinel for _extract_json_path — distinguishes "path not found" from "value is None"
_JSON_MISSING = object()


def _extract_json_path(obj: Any, path: str) -> Any:
    """Extract a value from a nested dict/list by JSON path.

    Returns _JSON_MISSING if the path doesn't exist.
    Supports: $.key, $.key1.key2, $.arr[0], $.arr[0].field
    """
    import re as _re
    if not path or obj is None:
        return _JSON_MISSING

    parts = path.lstrip("$").lstrip(".").split(".")
    current = obj

    for part in parts:
        if not part:
            continue
        arr_match = _re.match(r"^(.+?)\[(\d+)\]$", part)
        if arr_match:
            key = arr_match.group(1)
            idx = int(arr_match.group(2))
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return _JSON_MISSING
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return _JSON_MISSING
        elif part.startswith("[") and part.endswith("]"):
            idx = int(part[1:-1])
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return _JSON_MISSING
        else:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return _JSON_MISSING

    return current


# ---------------------------------------------------------------------------
# Gemini Schemas — every schema includes reasoning, confidence, evidence_sources
# ---------------------------------------------------------------------------

ACTION_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {
            "type": "boolean",
            "description": "Did the action achieve its intended effect?",
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0",
        },
        "what_changed": {
            "type": "string",
            "description": "One sentence: what observable change occurred?",
        },
        "failure_type": {
            "type": "string",
            "enum": [
                "none", "wrong_element", "element_not_found", "value_not_set",
                "navigation_unexpected", "no_visible_change", "error_message_appeared",
                "form_validation_failed", "server_error", "claimed_but_not_executed",
                "page_not_ready", "unknown",
            ],
            "description": "Failure category. 'none' if succeeded.",
        },
        "should_retry": {
            "type": "boolean",
            "description": "Should the CU agent retry with a different approach?",
        },
        "correction_hint": {
            "type": "string",
            "description": "If retry: what to do differently. Empty string if success.",
        },
        "reasoning": {
            "type": "string",
            "description": "Chain of thought. MUST cite specific evidence signals.",
        },
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which signals did you check? e.g. ['aria_alerts: Invalid email', 'http_status: 422']",
        },
    },
    "required": [
        "success", "confidence", "what_changed", "failure_type",
        "should_retry", "correction_hint", "reasoning", "evidence_sources",
    ],
}

MCP_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "confidence": {"type": "number"},
        "failure_reason": {
            "type": "string",
            "description": "If failed: HTTP error, response mismatch, or page state contradiction. Empty if success.",
        },
        "response_matches_intent": {
            "type": "boolean",
            "description": "Does the API response body confirm the intended action was done correctly?",
        },
        "intent_mismatch_detail": {
            "type": "string",
            "description": "If response_matches_intent=false: what specifically was wrong?",
        },
        "page_state_verified": {
            "type": "boolean",
            "description": "Was the expected page state change confirmed via AXTree diff?",
        },
        "recommendation": {
            "type": "string",
            "enum": ["proceed", "retry_mcp", "fallback_to_cu", "mark_mcp_degraded"],
        },
        "response_summary": {
            "type": "string",
            "description": "Brief summary of what the API returned.",
        },
        "reasoning": {"type": "string"},
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "success", "confidence", "failure_reason", "response_matches_intent",
        "intent_mismatch_detail", "page_state_verified", "recommendation",
        "response_summary", "reasoning", "evidence_sources",
    ],
}

SUBTASK_REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "subtask_achieved": {"type": "boolean"},
        "confidence": {"type": "number"},
        "outcome_summary": {
            "type": "string",
            "description": "2-3 sentences: what happened and what the page shows now.",
        },
        "failure_analysis": {
            "type": "string",
            "description": "Root cause if failed. Empty if success.",
        },
        "recommendation": {
            "type": "string",
            "enum": [
                "proceed_to_next_subtask", "retry_same_subtask",
                "retry_different_approach", "prune_current_branch",
                "task_impossible", "complete",
            ],
        },
        "page_state_summary": {"type": "string"},
        "extracted_data": {
            "type": "string",
            "description": "Data found for retrieval tasks. Empty if N/A.",
        },
        "mcp_failure_reason": {
            "type": "string",
            "description": "If MCP was used and failed: why? Empty otherwise.",
        },
        "false_positive_check": {
            "type": "string",
            "description": "Did the CU agent claim success without performing the key action? Report any 'claimed but not executed' patterns.",
        },
        "reasoning": {"type": "string"},
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "subtask_achieved", "confidence", "outcome_summary", "failure_analysis",
        "recommendation", "page_state_summary", "extracted_data",
        "mcp_failure_reason", "false_positive_check", "reasoning",
        "evidence_sources",
    ],
}

# Schema for lightweight MCP response-vs-intent check (Stage B)
MCP_INTENT_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "response_matches_intent": {
            "type": "boolean",
            "description": "Does the response confirm the intended action was completed correctly?",
        },
        "mismatch_detail": {
            "type": "string",
            "description": "If mismatch: what specifically was wrong? e.g. 'Requested qty=3 but response shows qty=1'",
        },
        "response_summary": {
            "type": "string",
            "description": "Brief summary of what the API returned.",
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "response_matches_intent", "mismatch_detail", "response_summary",
        "reasoning", "confidence", "evidence_sources",
    ],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DeterministicSignals:
    """Collected automatically after every action — no LLM involved."""
    url_before: str
    url_after: str
    url_changed: bool
    # Element-specific (for type/select/click on a specific element)
    target_element_value_before: str | None
    target_element_value_after: str | None
    expected_value: str | None  # From action["text"] or action["value"]
    value_matches_expected: bool | None
    # HTTP signals from traffic captured during the action
    http_status_codes: list[int]
    has_error_status: bool       # Any 4xx or 5xx
    has_success_status: bool     # Any 2xx state-changing request
    # ARIA signals from after AXTree
    aria_alerts: list[str]       # role="alert" or role="alertdialog" text
    aria_status_messages: list[str]  # role="status" or aria-live="polite" text
    aria_invalid_fields: list[dict]  # [{element_id, field_name}]
    new_dialogs: list[str]       # New role="dialog" text
    # Structural changes
    elements_added: int
    elements_removed: int
    # Verdict
    deterministic_verdict: str   # "success" | "failure" | "ambiguous"
    verdict_reason: str


@dataclass
class AXTreeNode:
    """Flattened representation of one AXTree node for diffing."""
    role: str
    name: str
    value: str | None
    path: str           # Parent chain: "navigation > link"
    properties: dict    # checked, expanded, disabled, invalid, etc.


# ---------------------------------------------------------------------------
# AXTree Walking Utilities
# ---------------------------------------------------------------------------

_SKIP_ROLES = frozenset(("none", "generic", "GenericContainer"))
_ARIA_SIGNAL_ROLES = frozenset(("alert", "alertdialog", "status", "dialog", "log"))
_PROPERTY_KEYS = (
    "checked", "expanded", "disabled", "required", "selected",
    "level", "invalid", "readonly", "pressed", "multiselectable",
)


def _walk_axtree_for_signals(
    node: dict,
    *,
    role_filter: frozenset[str] | None = None,
    _depth: int = 0,
) -> list[dict]:
    """Recursively walk raw AXTree, collect nodes matching role_filter.

    Returns list of {role, name, value, children_roles} dicts.
    If role_filter is None, collects all non-skip nodes.
    """
    if _depth > 30:
        return []
    results: list[dict] = []
    role = node.get("role", "none")
    name = node.get("name", "")

    if role_filter is None or role.lower() in role_filter:
        if name or role.lower() in (role_filter or set()):
            results.append({
                "role": role,
                "name": name,
                "value": node.get("value"),
            })

    for child in node.get("children", []):
        results.extend(_walk_axtree_for_signals(child, role_filter=role_filter, _depth=_depth + 1))
    return results


def _collect_aria_invalid(node: dict, elements: list[InteractiveElement], *, _depth: int = 0) -> list[dict]:
    """Find nodes where invalid=True or invalid property exists in AXTree."""
    if _depth > 30:
        return []
    results: list[dict] = []
    role = node.get("role", "none")
    name = node.get("name", "")

    if node.get("invalid"):
        # Try to match to an InteractiveElement by (role, name)
        matched_id = None
        name_lower = name.lower().strip()
        for el in elements:
            if el.role.lower() == role.lower() and (el.name or "").lower().strip() == name_lower:
                matched_id = el.element_id
                break
        results.append({"element_id": matched_id, "field_name": name or role})

    for child in node.get("children", []):
        results.extend(_collect_aria_invalid(child, elements, _depth=_depth + 1))
    return results


# ---------------------------------------------------------------------------
# AXTree Flattening & Diffing
# ---------------------------------------------------------------------------

def _flatten_axtree(node: dict, path: str = "", *, _depth: int = 0) -> list[AXTreeNode]:
    """Recursively flatten AXTree dict into a list of comparable nodes.

    Walks depth-first. Skips WebArea root, noise roles without names.
    Records: role, name, value, structural path, properties.
    """
    if _depth > 30:
        return []
    results: list[AXTreeNode] = []
    role = node.get("role", "none")
    name = node.get("name", "")

    # Skip root WebArea — process children directly
    if role == "WebArea":
        for child in node.get("children", []):
            results.extend(_flatten_axtree(child, path, _depth=_depth + 1))
        return results

    # Skip noise roles unless they have a name
    if role.lower() in _SKIP_ROLES and not name:
        for child in node.get("children", []):
            results.extend(_flatten_axtree(child, path, _depth=_depth + 1))
        return results

    current_path = f"{path} > {role}" if path else role

    # Extract meaningful properties
    props: dict[str, Any] = {}
    for key in _PROPERTY_KEYS:
        if key in node:
            props[key] = node[key]

    results.append(AXTreeNode(
        role=role,
        name=name,
        value=node.get("value"),
        path=current_path,
        properties=props,
    ))

    for child in node.get("children", []):
        results.extend(_flatten_axtree(child, current_path, _depth=_depth + 1))
    return results


def compute_axtree_diff(before: dict | None, after: dict | None) -> str:
    """Compute focused diff between before/after AXTree snapshots.

    Flattens both trees, compares node signatures, reports meaningful changes.
    Prioritizes ARIA signal nodes and structural changes.
    Output capped at ~500 tokens for LLM consumption.
    """
    if before is None and after is None:
        return "NO AXTREE DATA AVAILABLE"
    if before is None:
        return "NO BEFORE STATE — first action"
    if after is None:
        return "AFTER STATE UNAVAILABLE"

    before_nodes = _flatten_axtree(before)
    after_nodes = _flatten_axtree(after)

    # Signature: (role, name, path) — identifies structural position
    def sig(n: AXTreeNode) -> tuple[str, str, str]:
        return (n.role, n.name, n.path)

    # Content key: (role, path) — for detecting text content changes at same position
    def ck(n: AXTreeNode) -> tuple[str, str]:
        return (n.role, n.path)

    before_sigs = {sig(n) for n in before_nodes}
    after_sigs = {sig(n) for n in after_nodes}
    before_by_sig = {sig(n): n for n in before_nodes}
    after_by_sig = {sig(n): n for n in after_nodes}

    # Group by content key for text-change detection
    before_by_ck: dict[tuple[str, str], list[AXTreeNode]] = {}
    after_by_ck: dict[tuple[str, str], list[AXTreeNode]] = {}
    for n in before_nodes:
        before_by_ck.setdefault(ck(n), []).append(n)
    for n in after_nodes:
        after_by_ck.setdefault(ck(n), []).append(n)

    # (priority, description) — priority: 0=HIGH, 1=MEDIUM, 2=LOW
    changes: list[tuple[int, str]] = []

    def _path_label(path: str) -> str:
        return path.rsplit(" > ", 1)[-1] if " > " in path else path

    # --- ADDED nodes ---
    for s in after_sigs - before_sigs:
        node = after_by_sig[s]
        r = node.role.lower()
        label = f'{node.role} "{node.name}"' if node.name else node.role
        ctx = f" (in {_path_label(node.path)})" if " > " in node.path else ""

        if r in _ARIA_SIGNAL_ROLES or r in ("heading", "form"):
            changes.append((0, f"ADDED: {label}{ctx}"))
        elif r in ("button", "link", "textbox", "combobox"):
            changes.append((1, f"ADDED: {label}{ctx}"))
        else:
            changes.append((2, f"ADDED: {label}{ctx}"))

    # --- REMOVED nodes ---
    for s in before_sigs - after_sigs:
        node = before_by_sig[s]
        r = node.role.lower()
        label = f'{node.role} "{node.name}"' if node.name else node.role
        ctx = f" (from {_path_label(node.path)})" if " > " in node.path else ""

        if r in _ARIA_SIGNAL_ROLES or r in ("heading", "form"):
            changes.append((0, f"REMOVED: {label}{ctx}"))
        elif r in ("button", "link", "textbox"):
            changes.append((1, f"REMOVED: {label}{ctx}"))
        else:
            changes.append((2, f"REMOVED: {label}{ctx}"))

    # --- CHANGED — same role+path, different name (text content update) ---
    for key in set(before_by_ck) & set(after_by_ck):
        b_names = [n.name for n in before_by_ck[key]]
        a_names = [n.name for n in after_by_ck[key]]
        if b_names != a_names and b_names and a_names:
            role = key[0]
            ctx = _path_label(key[1])
            for bn, an in zip(b_names, a_names):
                if bn != an:
                    desc = f'CHANGED: {role} "{bn}" → "{an}" (in {ctx})'
                    pri = 0 if role.lower() in ("heading", "status", "alert") else 1
                    changes.append((pri, desc))

    # --- PROPERTY CHANGED — same signature, different properties ---
    for s in before_sigs & after_sigs:
        b = before_by_sig[s]
        a = after_by_sig[s]
        if b.properties != a.properties:
            for key in set(b.properties) | set(a.properties):
                bv = b.properties.get(key)
                av = a.properties.get(key)
                if bv != av:
                    desc = f'PROPERTY CHANGED: {a.role} "{a.name}" {key}: {bv} → {av}'
                    pri = 0 if key == "invalid" else 1
                    changes.append((pri, desc))

    if not changes:
        return "NO MEANINGFUL CHANGES DETECTED"

    # Sort by priority, emit up to 50 lines
    changes.sort(key=lambda c: c[0])
    budget = 50
    lines = ["=== Page Changes ==="]
    for _, desc in changes[:budget]:
        lines.append(desc)
    if len(changes) > budget:
        lines.append(f"... and {len(changes) - budget} more changes omitted")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------

class Reflector:
    """Three-stage verification pipeline.

    Public API:
        reflect_on_action()  — per-action, 3 stages
        reflect_on_mcp_call() — MCP verification, 3 stages (HTTP + intent + page state)
        reflect_on_subtask() — full journey, deep evaluation
    """

    def __init__(self, session: SessionManager, trace: TaskTrace):
        self.session = session
        self.trace = trace
        self._action_prompt = self._load_prompt("reflect_action.txt")
        self._subtask_prompt = self._load_prompt("reflect_subtask.txt")

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = PROMPTS_DIR / filename
        if not path.exists():
            logger.warning("Prompt file not found: %s", path)
            return ""
        return path.read_text(encoding="utf-8")

    def _log(self, event_type: str, summary: str, **kwargs) -> str | None:
        return self.trace.log("reflector", event_type, summary, **kwargs)

    # ===================================================================
    # Stage 1 — Deterministic Signal Harvest
    # ===================================================================

    def _harvest_deterministic_signals(
        self,
        action: dict,
        result: ActionResult,
        before_elements: list[InteractiveElement],
        after_elements: list[InteractiveElement],
        before_url: str,
        after_url: str,
        after_axtree: dict | None = None,
        traffic_since: float = 0,
        before_axtree: dict | None = None,
    ) -> DeterministicSignals:
        """Collect all deterministic signals. Runs after EVERY action, zero LLM cost."""
        action_type = action.get("action_type", "")
        element_id = action.get("element_id")

        # 1. URL comparison
        url_before = before_url
        url_after = after_url
        url_changed = url_before != url_after

        # 2. Element value comparison (for type/select — compare before/after element values)
        val_before: str | None = None
        val_after: str | None = None
        expected: str | None = None
        val_matches: bool | None = None

        if action_type in ("type", "select") and element_id is not None:
            expected = action.get("text") if action_type == "type" else action.get("value")
            # Find element in before/after element lists by ID
            for el in before_elements:
                if el.element_id == element_id:
                    val_before = el.value
                    break
            for el in after_elements:
                if el.element_id == element_id:
                    val_after = el.value
                    break
            if expected is not None and val_after is not None:
                val_matches = expected.strip() == val_after.strip()

        # 3. HTTP status codes from traffic during the action
        traffic = self.session.get_captured_traffic(since_timestamp=traffic_since)
        status_codes = [r.status_code for r in traffic]
        has_error = any(s >= 400 for s in status_codes)
        has_success = any(
            200 <= r.status_code < 300 and r.is_state_changing
            for r in traffic
        )

        # 4. ARIA signals from after AXTree
        axtree = after_axtree
        aria_alerts: list[str] = []
        aria_statuses: list[str] = []
        new_dialogs: list[str] = []
        aria_invalid: list[dict] = []

        if axtree:
            # Alerts and alertdialogs
            for node in _walk_axtree_for_signals(axtree, role_filter=frozenset(("alert", "alertdialog"))):
                if node["name"]:
                    aria_alerts.append(node["name"])
            # Status messages
            for node in _walk_axtree_for_signals(axtree, role_filter=frozenset(("status",))):
                if node["name"]:
                    aria_statuses.append(node["name"])
            # Dialogs — compare with before to find NEW ones
            after_dialogs = {
                n["name"] for n in _walk_axtree_for_signals(axtree, role_filter=frozenset(("dialog",)))
                if n["name"]
            }
            before_dialogs: set[str] = set()
            new_dialogs = list(after_dialogs - before_dialogs)
            # aria-invalid fields — only count NEW invalid fields (not pre-existing)
            after_invalid = _collect_aria_invalid(axtree, after_elements)
            if before_axtree:
                before_invalid = _collect_aria_invalid(before_axtree, before_elements)
                before_field_names = {d["field_name"] for d in before_invalid}
                aria_invalid = [d for d in after_invalid if d["field_name"] not in before_field_names]
            else:
                aria_invalid = after_invalid

        # 5. Element count diff via fingerprints
        before_fps = {el.fingerprint for el in before_elements}
        after_fps = {el.fingerprint for el in after_elements}
        elements_added = len(after_fps - before_fps)
        elements_removed = len(before_fps - after_fps)

        # --- Determine verdict ---
        verdict = "ambiguous"
        reason = ""

        match action_type:
            case "type" | "select":
                if val_matches is True:
                    verdict, reason = "success", f"Value set to '{val_after}' as expected"
                elif val_matches is False:
                    verdict, reason = "failure", f"Expected '{expected}' but got '{val_after}'"
                elif has_error:
                    verdict, reason = "failure", f"HTTP error during action: {status_codes}"
                # else: ambiguous (couldn't verify value)

            case "navigate" | "go_back":
                if url_changed:
                    verdict, reason = "success", f"URL changed to {url_after}"
                else:
                    verdict, reason = "failure", "URL did not change after navigation"

            case "scroll":
                if elements_added > 0:
                    verdict, reason = "success", f"{elements_added} new elements revealed"
                else:
                    # Deterministic failure — 0 new elements means scroll had no effect.
                    # Do NOT send to Stage 3 LLM; it overrides to "success" ~96% of the time,
                    # causing infinite scroll loops (observed on LEGO, BMS, ConfirmTkt).
                    verdict, reason = "failure", "Scroll produced no new elements"

            case "note":
                verdict, reason = "success", "Note recorded"

            case "wait":
                if elements_added > 0 or elements_removed > 0:
                    verdict, reason = "success", f"Wait completed — {elements_added} new elements appeared"
                else:
                    verdict, reason = "ambiguous", "Wait completed but no new elements appeared"

            case "click":
                if aria_alerts:
                    verdict, reason = "failure", f"Alert: {aria_alerts[0]}"
                elif has_error:
                    verdict, reason = "failure", f"HTTP error: {[s for s in status_codes if s >= 400]}"
                elif url_changed:
                    verdict, reason = "success", f"Navigated to {url_after}"
                elif has_success:
                    verdict, reason = "success", "Successful state-changing request detected"
                elif elements_added > 0 or elements_removed > 0:
                    verdict, reason = "success", f"Page changed: +{elements_added}/-{elements_removed} elements"
                # aria_invalid not checked for clicks — form validation
                # is only meaningful for type/select actions, not clicks
                # else: ambiguous

            case "press_key":
                key = action.get("text", "").lower()
                if key == "enter":
                    if has_success and url_changed:
                        verdict, reason = "success", "Form submitted — URL changed with success response"
                    elif url_changed:
                        verdict, reason = "success", f"Enter pressed — navigated to {url_after}"
                    elif has_success:
                        verdict, reason = "success", "Enter pressed — successful response detected"
                    elif has_error:
                        verdict, reason = "failure", f"HTTP error after Enter: {[s for s in status_codes if s >= 400]}"
                    elif aria_alerts:
                        verdict, reason = "failure", f"Alert after Enter: {aria_alerts[0]}"
                    elif elements_added > 0 or elements_removed > 0:
                        verdict, reason = "success", f"Enter pressed — page changed: +{elements_added}/-{elements_removed} elements"
                    # aria_invalid not checked for press_key — form validation
                    # is only meaningful for type/select actions
                    # else: ambiguous — Enter pressed but unclear outcome
                elif key in ("tab", "escape", "arrowdown", "arrowup", "space"):
                    verdict, reason = "success", f"Key '{key}' pressed"
                # else: ambiguous

            case "hover":
                # Hover effects are hard to verify deterministically
                if not result.success:
                    verdict, reason = "failure", result.error or "Hover failed"
                else:
                    verdict, reason = "success", "Hover executed"

        # Override: if the action itself failed at the Playwright level
        if not result.success:
            verdict = "failure"
            reason = result.error or "Action execution failed"

        signals = DeterministicSignals(
            url_before=url_before,
            url_after=url_after,
            url_changed=url_changed,
            target_element_value_before=val_before,
            target_element_value_after=val_after,
            expected_value=expected,
            value_matches_expected=val_matches,
            http_status_codes=status_codes,
            has_error_status=has_error,
            has_success_status=has_success,
            aria_alerts=aria_alerts,
            aria_status_messages=aria_statuses,
            aria_invalid_fields=aria_invalid,
            new_dialogs=new_dialogs,
            elements_added=elements_added,
            elements_removed=elements_removed,
            deterministic_verdict=verdict,
            verdict_reason=reason,
        )

        self._log("deterministic_signals", f"Stage 1: {verdict} — {reason}", detail={
            "action_type": action_type,
            "verdict": verdict,
            "reason": reason,
            "url_changed": url_changed,
            "aria_alerts": aria_alerts,
            "http_status_codes": status_codes[:10],
            "value_matches": val_matches,
            "elements_added": elements_added,
            "elements_removed": elements_removed,
        })
        return signals

    # ===================================================================
    # Stage 2 — AXTree Diff (ambiguous cases only)
    # ===================================================================

    def _compute_axtree_diff_stage(
        self,
        action: dict,
        signals: DeterministicSignals,
        before_axtree: dict | None,
        after_axtree: dict | None,
        before_elements: list[InteractiveElement] | None = None,
    ) -> str:
        """Stage 2: compute AXTree diff and attempt to resolve ambiguity.

        Detects silent failures: submit action + no changes + no alerts + no errors.
        """
        diff = compute_axtree_diff(
            before_axtree,
            after_axtree,
        )

        # Silent failure detection for submit-like actions
        action_type = action.get("action_type", "")
        is_submit = (
            (action_type == "click" and before_elements is not None and self._is_submit_element(action, before_elements))
            or (action_type == "press_key" and action.get("text", "").lower() == "enter")
        )
        if (
            is_submit
            and diff == "NO MEANINGFUL CHANGES DETECTED"
            and not signals.aria_alerts
            and not signals.has_error_status
        ):
            diff += (
                "\n\nWARNING: A submit action was performed but no page changes were detected. "
                "This may indicate a silent failure — the form may not have actually submitted."
            )

        self._log("axtree_diff", f"Stage 2: {len(diff)} chars of diff", detail={
            "action_type": action_type,
            "diff_preview": diff[:300],
            "is_submit": is_submit,
        })
        return diff

    @staticmethod
    def _is_submit_element(action: dict, elements: list[InteractiveElement]) -> bool:
        """Check if the clicked element is a submit control.

        Uses structural signals only:
        1. type="submit" (HTML spec: this IS a submit button)
        2. role="button" inside a form (element's selector contains 'form')
        3. Element has form-related attributes (action, method, formaction)
        """
        element_id = action.get("element_id")
        if element_id is None:
            return False
        for el in elements:
            if el.element_id == element_id:
                type_lower = (el.element_type or "").lower()
                # Definitive: input[type=submit] or button[type=submit]
                if type_lower == "submit":
                    return True
                # Element has formaction attribute (HTML spec submit override)
                attrs = el.attributes or {}
                if attrs.get("formaction") or attrs.get("formmethod"):
                    return True
                # Button/link inside a form — check selector for form ancestry
                role_lower = el.role.lower()
                if role_lower == "button" and "form" in (el.selector or "").lower():
                    return True
                return False
        return False

    # ===================================================================
    # Stage 3 — LLM Semantic Evaluation (ambiguous only)
    # ===================================================================

    async def _llm_evaluate_action(
        self,
        action: dict,
        signals: DeterministicSignals,
        axtree_diff: str,
        subtask_description: str,
    ) -> dict:
        """LLM evaluation for genuinely ambiguous action outcomes.

        Only called when stages 1-2 couldn't resolve. Uses Gemini Flash.
        """
        # Build evidence summary for the LLM
        evidence_text = self._format_signals_for_llm(signals, axtree_diff)

        prompt = (
            f"Subtask: {subtask_description}\n\n"
            f"Action attempted: {json.dumps(action)}\n\n"
            f"Evidence:\n{evidence_text}"
        )

        with self.trace.span("reflector", "llm_action_eval", f"Stage 3 LLM: {action.get('action_type')}") as span:
            result = await call_gemini_async(
                model="gemini-3-flash-preview",
                contents=[prompt],
                response_schema=ACTION_REFLECTION_SCHEMA,
                system_instruction=self._action_prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 8192},
                prompt_log_dir=self.trace.output_dir / "prompt_made",
            )
            span.set_reasoning(result.get("reasoning", ""))
            span.set_confidence(result.get("confidence", 0.0))
            for src in result.get("evidence_sources", []):
                span.add_evidence(Evidence("model_output", src))
            span.set_outcome("success" if result.get("success") else "failure")

        self._log("llm_action_verdict", f"Stage 3: {'success' if result.get('success') else 'failure'}", detail={
            "action_type": action.get("action_type"),
            "llm_verdict": result,
        })
        return result

    @staticmethod
    def _format_signals_for_llm(signals: DeterministicSignals, axtree_diff: str) -> str:
        """Format deterministic signals as a readable evidence block for the LLM."""
        lines = [
            f"URL: {signals.url_before} → {signals.url_after} (changed: {signals.url_changed})",
        ]
        if signals.expected_value is not None:
            lines.append(
                f"Element value: '{signals.target_element_value_before}' → '{signals.target_element_value_after}' "
                f"(expected: '{signals.expected_value}', matches: {signals.value_matches_expected})"
            )
        if signals.http_status_codes:
            lines.append(f"HTTP status codes: {signals.http_status_codes}")
        if signals.aria_alerts:
            lines.append(f"ARIA alerts: {signals.aria_alerts}")
        if signals.aria_status_messages:
            lines.append(f"ARIA status messages: {signals.aria_status_messages}")
        if signals.aria_invalid_fields:
            lines.append(f"Fields with aria-invalid: {signals.aria_invalid_fields}")
        if signals.new_dialogs:
            lines.append(f"New dialogs: {signals.new_dialogs}")
        lines.append(f"Elements added: {signals.elements_added}, removed: {signals.elements_removed}")
        lines.append(f"\n{axtree_diff}")
        return "\n".join(lines)

    # ===================================================================
    # Public: reflect_on_action (three-stage pipeline)
    # ===================================================================

    async def reflect_on_action(
        self,
        action: dict,
        result: ActionResult,
        before_elements: list[InteractiveElement],
        after_elements: list[InteractiveElement],
        before_url: str,
        after_url: str,
        subtask_description: str,
        after_axtree: dict | None = None,
        traffic_since: float = 0,
        before_axtree: dict | None = None,
    ) -> dict:
        """Three-stage action reflection.

        1. Harvest deterministic signals (always)
        2. If ambiguous: compute AXTree diff
        3. If still ambiguous: invoke LLM

        Returns {deterministic_signals, axtree_diff, llm_used, verdict}.
        """
        # Stage 1
        signals = self._harvest_deterministic_signals(
            action, result, before_elements, after_elements,
            before_url, after_url, after_axtree, traffic_since,
            before_axtree=before_axtree,
        )
        axtree_diff: str | None = None
        llm_used = False

        if signals.deterministic_verdict != "ambiguous":
            # Resolved at Stage 1 — build verdict from signals
            verdict = {
                "success": signals.deterministic_verdict == "success",
                "confidence": 0.95 if signals.deterministic_verdict == "success" else 0.9,
                "what_changed": signals.verdict_reason,
                "failure_type": "none" if signals.deterministic_verdict == "success" else self._infer_failure_type(signals),
                "should_retry": signals.deterministic_verdict == "failure",
                "correction_hint": "",
                "reasoning": f"Deterministic Stage 1: {signals.verdict_reason}",
                "evidence_sources": self._collect_signal_evidence(signals),
            }
        else:
            # Stage 2 — AXTree diff (no before_axtree stored; pass None)
            axtree_diff = self._compute_axtree_diff_stage(
                action, signals, None, after_axtree, before_elements,
            )

            # Can Stage 2 resolve it?
            if "WARNING: A submit action was performed but no page changes" in axtree_diff:
                # Silent failure detected
                verdict = {
                    "success": False,
                    "confidence": 0.75,
                    "what_changed": "No observable changes after submit action",
                    "failure_type": "no_visible_change",
                    "should_retry": True,
                    "correction_hint": "The form may not have submitted. Check if all required fields are filled.",
                    "reasoning": "Silent failure: submit action performed but AXTree shows no changes.",
                    "evidence_sources": ["axtree_diff: NO MEANINGFUL CHANGES after submit"],
                }
            elif axtree_diff != "NO MEANINGFUL CHANGES DETECTED" and "ADDED:" in axtree_diff:
                # Stage 2 found changes — probably success for click actions
                has_alert = any("alert" in line.lower() for line in axtree_diff.split("\n"))
                if has_alert:
                    verdict = {
                        "success": False,
                        "confidence": 0.85,
                        "what_changed": "Alert appeared in AXTree diff",
                        "failure_type": "error_message_appeared",
                        "should_retry": True,
                        "correction_hint": "An error or alert appeared. Check the alert message.",
                        "reasoning": f"AXTree diff shows alert: {axtree_diff[:200]}",
                        "evidence_sources": ["axtree_diff: alert node appeared"],
                    }
                else:
                    verdict = {
                        "success": True,
                        "confidence": 0.8,
                        "what_changed": "Page content changed after action",
                        "failure_type": "none",
                        "should_retry": False,
                        "correction_hint": "",
                        "reasoning": f"AXTree diff shows meaningful changes: {axtree_diff[:200]}",
                        "evidence_sources": ["axtree_diff: content changes detected"],
                    }
            else:
                # Stage 3 — LLM needed
                llm_used = True
                verdict = await self._llm_evaluate_action(
                    action, signals, axtree_diff, subtask_description,
                )

        return {
            "deterministic_signals": asdict(signals),
            "axtree_diff": axtree_diff,
            "llm_used": llm_used,
            "verdict": verdict,
        }

    @staticmethod
    def _infer_failure_type(signals: DeterministicSignals) -> str:
        """Infer failure_type from deterministic signals."""
        if signals.aria_alerts:
            return "error_message_appeared"
        if signals.aria_invalid_fields:
            return "form_validation_failed"
        if signals.has_error_status:
            return "server_error"
        if signals.value_matches_expected is False:
            return "value_not_set"
        return "unknown"

    @staticmethod
    def _collect_signal_evidence(signals: DeterministicSignals) -> list[str]:
        """Build evidence_sources list from deterministic signals."""
        evidence: list[str] = []
        if signals.url_changed:
            evidence.append(f"url_changed: {signals.url_before} → {signals.url_after}")
        if signals.value_matches_expected is not None:
            evidence.append(f"value_match: expected='{signals.expected_value}' got='{signals.target_element_value_after}'")
        if signals.http_status_codes:
            evidence.append(f"http_status: {signals.http_status_codes}")
        if signals.aria_alerts:
            evidence.append(f"aria_alerts: {signals.aria_alerts}")
        if signals.aria_invalid_fields:
            evidence.append(f"aria_invalid: {len(signals.aria_invalid_fields)} fields")
        if signals.elements_added or signals.elements_removed:
            evidence.append(f"elements: +{signals.elements_added} -{signals.elements_removed}")
        return evidence

    # ===================================================================
    # Public: reflect_on_mcp_call (three-stage: HTTP + intent + page state)
    # ===================================================================

    async def reflect_on_mcp_call(
        self,
        tool_name: str,
        tool_intent: str,
        http_status: int,
        response_body: dict | str | None,
        before_axtree: dict | None = None,
        after_axtree: dict | None = None,
        response_template: dict | None = None,
    ) -> dict:
        """Deterministic-only MCP verification. No LLM calls.

        Stage A: HTTP status + GraphQL error check.
        Stage B: Response structure check against learned template.
        Stage C: Page state AXTree diff (for mutations).

        Semantic verification (does the response match intent?) is the
        orchestrator's job via fresh page state on the next planning step.
        """
        with self.trace.span("reflector", "mcp_reflection", f"MCP reflect: {tool_name}") as span:
            evidence: list[str] = []

            # --- Stage A: Deterministic HTTP check ---
            if http_status >= 400:
                error_msg = self._extract_error_from_body(response_body)
                verdict = {
                    "success": False,
                    "confidence": 0.95,
                    "failure_reason": f"HTTP {http_status}: {error_msg}",
                    "response_matches_intent": False,
                    "intent_mismatch_detail": f"Server returned error status {http_status}",
                    "page_state_verified": False,
                    "recommendation": "fallback_to_cu",
                    "response_summary": error_msg[:200],
                    "reasoning": f"HTTP {http_status} is a definitive failure.",
                    "evidence_sources": [f"http_status: {http_status}", f"error: {error_msg[:100]}"],
                }
                span.set_outcome("failure")
                span.set_reasoning(verdict["reasoning"])
                return verdict

            # Check for GraphQL errors inside 200 response
            if isinstance(response_body, dict) and "errors" in response_body:
                errors = response_body["errors"]
                if errors:  # Empty list = success in GraphQL spec
                    error_msgs = [e.get("message", str(e)) for e in errors] if isinstance(errors, list) else [str(errors)]
                    if not error_msgs:
                        error_msgs = ["(unknown GraphQL error)"]
                    verdict = {
                        "success": False,
                        "confidence": 0.9,
                        "failure_reason": f"GraphQL errors: {error_msgs}",
                        "response_matches_intent": False,
                        "intent_mismatch_detail": f"GraphQL error in response: {error_msgs[0]}",
                        "page_state_verified": False,
                        "recommendation": "retry_mcp",
                        "response_summary": f"GraphQL errors: {error_msgs}",
                        "reasoning": "GraphQL response contains errors array despite 200 status.",
                        "evidence_sources": [f"graphql_errors: {error_msgs}"],
                    }
                    span.set_outcome("failure")
                    span.set_reasoning(verdict["reasoning"])
                    return verdict

            evidence.append(f"http_status: {http_status}")

            # --- Stage B: Response structure check against template ---
            structure_issues: list[str] = []
            if response_template and isinstance(response_body, (dict, list)):
                structure_issues = self._check_response_structure(
                    response_body, response_template,
                )
                if structure_issues:
                    evidence.extend(
                        f"structure_issue: {issue}" for issue in structure_issues
                    )

            # --- Stage C: Page state AXTree diff (for mutations) ---
            page_verified = False
            if before_axtree and after_axtree:
                page_diff = compute_axtree_diff(before_axtree, after_axtree)
                page_verified = page_diff != "NO MEANINGFUL CHANGES DETECTED"
                evidence.append(
                    f"page_diff: {'changes detected' if page_verified else 'no changes'}"
                )

            # --- Combine verdict ---
            response_summary = self._summarize_response(response_body)
            has_structural_failure = len(structure_issues) > 0

            if has_structural_failure:
                # Template says expected paths are missing or empty
                success = False
                recommendation = "retry_mcp"
                reasoning = (
                    f"Response structure mismatch: {'; '.join(structure_issues)}"
                )
            elif page_verified or before_axtree is None:
                # HTTP 2xx, no structural issues, page changed (or no page check)
                success = True
                recommendation = "proceed"
                reasoning = f"HTTP {http_status}, response structure OK"
            else:
                # HTTP 2xx, structure OK, but page didn't change — suspicious but not fatal.
                # Orchestrator will verify semantically on next planning step.
                success = True
                recommendation = "proceed"
                reasoning = (
                    f"HTTP {http_status}, response structure OK. "
                    "Page state unchanged — orchestrator will verify."
                )

            verdict = {
                "success": success,
                "confidence": 0.85 if success else 0.75,
                "failure_reason": "; ".join(structure_issues) if not success else "",
                "response_matches_intent": success,
                "intent_mismatch_detail": "; ".join(structure_issues),
                "page_state_verified": page_verified,
                "recommendation": recommendation,
                "response_summary": response_summary,
                "reasoning": reasoning,
                "evidence_sources": evidence,
            }

            span.set_outcome("success" if success else "failure")
            span.set_reasoning(reasoning)
            span.set_confidence(verdict["confidence"])
            for src in evidence:
                span.add_evidence(Evidence("model_output", src))

        return verdict

    @staticmethod
    def _check_response_structure(
        response_body: dict | list,
        template: dict,
    ) -> list[str]:
        """Check response against learned structural template.

        Only flags when:
        - A path in always_present_paths is missing
        - A path in always_non_null_paths is null/empty (where it never was before)
        Does NOT flag different array lengths or missing optional fields.
        """
        issues: list[str] = []

        always_present = template.get("always_present_paths", [])
        always_non_null = template.get("always_non_null_paths", [])
        obs_count = template.get("observation_count", 0)

        # Only enforce structure after 2+ observations (single observation
        # isn't enough to establish what's "always" present)
        if obs_count < 2:
            return issues

        for path in always_present:
            val = _extract_json_path(response_body, path)
            if val is _JSON_MISSING:
                issues.append(f"Expected path '{path}' is missing")

        for path in always_non_null:
            val = _extract_json_path(response_body, path)
            if val is _JSON_MISSING:
                continue  # Already flagged as missing above if in always_present
            if val is None:
                issues.append(f"Path '{path}' is null (was never null before)")
            elif isinstance(val, (list, dict, str)) and not val:
                issues.append(f"Path '{path}' is empty (was never empty before)")

        return issues

    @staticmethod
    def _summarize_response(body: dict | str | None) -> str:
        """Compact one-line summary of a response body for logging."""
        if body is None:
            return "(no body)"
        if isinstance(body, str):
            return body[:200]
        if isinstance(body, dict):
            # Extract the most informative fields
            status = body.get("statusCode", body.get("status", ""))
            data = body.get("data", body.get("result", ""))
            msg = body.get("message", body.get("msg", ""))
            parts: list[str] = []
            if status != "":
                parts.append(f"status={status}")
            if msg:
                parts.append(f"msg={str(msg)[:60]}")
            if isinstance(data, list):
                parts.append(f"data=[{len(data)} items]")
            elif isinstance(data, dict):
                parts.append(f"data={{{len(data)} fields}}")
            elif data:
                parts.append(f"data={str(data)[:60]}")
            if parts:
                return ", ".join(parts)
            return json.dumps(body, default=str)[:200]
        return str(body)[:200]

    @staticmethod
    def _extract_error_from_body(body: dict | str | None) -> str:
        """Extract a human-readable error message from a response body."""
        if body is None:
            return "(no response body)"
        if isinstance(body, str):
            return body[:200]
        if isinstance(body, dict):
            # Common error fields
            for key in ("message", "error", "detail", "errors", "error_message"):
                if key in body:
                    val = body[key]
                    if isinstance(val, str):
                        return val[:200]
                    return json.dumps(val, default=str)[:200]
        return json.dumps(body, default=str)[:200]

    # ===================================================================
    # Public: reflect_on_subtask (deep journey evaluation)
    # ===================================================================

    async def reflect_on_subtask(
        self,
        subtask_description: str,
        actions_taken: list[dict],
        initial_url: str,
        final_url: str,
        final_elements: list[InteractiveElement],
        final_axtree: dict | None,
        notes: list[str] | None = None,
        screenshot_base64: str | None = None,
    ) -> dict:
        """Deep evaluation of the entire subtask journey.

        Uses Gemini Pro Preview with high thinking budget.
        Checks for false positives, silent failures, partial success.
        """
        # Build condensed action log
        action_log_lines: list[str] = []
        for i, record in enumerate(actions_taken, 1):
            action = record.get("action", {})
            verdict = record.get("verdict", {})
            at = action.get("action_type", "?")
            eid = action.get("element_id", "")
            text = action.get("text", "")[:40]
            success_str = "success" if verdict.get("success") else "failure" if verdict.get("success") is False else "ambiguous"
            what = verdict.get("what_changed", "")[:60]
            eid_str = f" [{eid}]" if eid else ""
            text_str = f' "{text}"' if text else ""
            action_log_lines.append(f"Step {i}: {at}{eid_str}{text_str} — {success_str} ({what})")

        action_log = "\n".join(action_log_lines)

        # Build content-focused representation for outcome verification.
        # Purpose-built pipeline: no element IDs, chrome compressed,
        # card-aware formatting, larger budget than old _build_simple_axtree.
        from morphnet.representation import build_reflector_representation
        axtree_text = build_reflector_representation(final_axtree) if final_axtree else "(no AXTree)"

        # Get focused DOM excerpt for subtask reflection
        try:
            dom_tree = await self.session.get_dom_tree(max_length=50_000)
            dom_excerpt = self._extract_focused_dom(dom_tree)
        except Exception:
            dom_excerpt = "(DOM unavailable)"

        # Collect all ARIA signals across all actions
        all_alerts: list[str] = []
        for record in actions_taken:
            signals = record.get("deterministic_signals", {})
            all_alerts.extend(signals.get("aria_alerts", []))
            all_alerts.extend(signals.get("aria_status_messages", []))
        all_alerts = list(dict.fromkeys(all_alerts))  # Deduplicate preserving order

        notes_text = "\n".join(f"- {n}" for n in (notes or []))

        prompt = (
            f"Subtask: {subtask_description}\n\n"
            f"Action Log:\n{action_log}\n\n"
            f"Current Page:\n{axtree_text}\n\n"
            f"Focused DOM Excerpt:\n{dom_excerpt[:2000]}\n\n"
            f"ARIA Signals Across All Actions: {all_alerts if all_alerts else 'none'}\n\n"
            f"Agent Notes:\n{notes_text if notes_text else 'none'}\n"
        )

        contents: list[Any] = [prompt]
        if screenshot_base64:
            contents.append({"mime_type": "image/jpeg", "data": screenshot_base64})

        with self.trace.span("reflector", "subtask_reflection", f"Subtask reflect: {subtask_description[:60]}") as span:
            result = await call_gemini_async(
                model="gemini-3.1-pro-preview",
                contents=contents,
                response_schema=SUBTASK_REFLECTION_SCHEMA,
                system_instruction=self._subtask_prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 4096},
                prompt_log_dir=self.trace.output_dir / "prompt_made",
            )
            span.set_reasoning(result.get("reasoning", ""))
            span.set_confidence(result.get("confidence", 0.0))
            for src in result.get("evidence_sources", []):
                span.add_evidence(Evidence("model_output", src))
            span.set_outcome("success" if result.get("subtask_achieved") else "failure")

        return result

    @staticmethod
    def _build_simple_axtree(node: dict, depth: int = 0) -> str:
        """Minimal text-only AXTree for subtask reflection context."""
        role = node.get("role", "none")
        name = node.get("name", "")
        lines: list[str] = []

        if role == "WebArea":
            for child in node.get("children", []):
                lines.append(Reflector._build_simple_axtree(child, depth))
            return "\n".join(lines)

        if role.lower() in ("none", "generic", "genericcontainer") and not name:
            for child in node.get("children", []):
                lines.append(Reflector._build_simple_axtree(child, depth))
            return "\n".join(lines)

        indent = "  " * depth
        line = f"{indent}{role}"
        if name:
            line += f' "{name}"'
        level = node.get("level")
        if level:
            line += f" — level {level}"
        value = node.get("value")
        if value and value != name:
            line += f' — "{value}"'
        lines.append(line)

        for child in node.get("children", []):
            lines.append(Reflector._build_simple_axtree(child, depth + 1))

        return "\n".join(lines)

    @staticmethod
    def _extract_focused_dom(dom_tree: str) -> str:
        """Extract DOM sections most relevant to verification.

        Looks for: forms, alert/status containers, cart indicators,
        flash messages, data attributes with counts/status.
        Deterministic HTML text scanning — not model output parsing.
        """
        if not dom_tree:
            return "(no DOM)"

        # Collect lines near key structural signals
        lines = dom_tree.split("\n")
        relevant: list[str] = []
        context_window = 3  # Lines before/after a match

        keywords = (
            "form", "alert", "status", "error", "success", "warning",
            "cart", "basket", "total", "flash", "notification", "message",
            "data-count", "data-cart", "data-qty", "aria-live",
        )
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)
                for j in range(start, end):
                    if lines[j] not in relevant:
                        relevant.append(lines[j])

        if not relevant:
            return "(no relevant DOM sections found)"
        return "\n".join(relevant[:100])  # Cap at 100 lines
