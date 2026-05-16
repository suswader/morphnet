"""
computer_use.py — Browser action agent for MorphNet.

Receives a subtask description. Has 10 actions to complete it.
Uses representation.py for page representation. Owns SoM annotation.
Uses reflector after each action.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from morphnet.session_manager import (
    SessionManager, InteractiveElement, CapturedRequest, ActionResult,
    call_gemini_async,
)
from morphnet.reflector import Reflector
from morphnet.trace import TaskTrace, Evidence
from morphnet.representation import (
    build_cu_representation,
    format_element,
    format_elements_summary,
    summarize_raw_axtree,
    render_raw_axtree_text,
    run_enrichments,
    apply_enrichments,
    analyze_page_context,
)

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Gemini Schema — CU action selection
# ---------------------------------------------------------------------------

CU_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [
                "click", "type", "select", "scroll", "press_key",
                "navigate", "hover", "go_back", "wait", "note", "stop",
            ],
            "description": (
                "The action to perform. 'note' records an observation without "
                "browser interaction. 'stop' signals subtask complete or impossible."
            ),
        },
        "element_id": {
            "type": "integer",
            "description": (
                "The [N] ID from the page. Required for click, type, select, hover. "
                "Not used for scroll, press_key, navigate, go_back, wait, note, stop."
            ),
        },
        "text": {
            "type": "string",
            "description": (
                "For type: text to enter. For navigate: URL. For press_key: key name "
                "(Enter, Tab, Escape, Backspace, ArrowDown). For note: the observation. "
                "For stop: reason for stopping."
            ),
        },
        "value": {
            "type": "string",
            "description": "For select: the option value or label to select.",
        },
        "direction": {
            "type": "string",
            "enum": ["up", "down"],
            "description": "For scroll: direction.",
        },
        "scroll_amount": {
            "type": "integer",
            "description": "For scroll: number of wheel clicks (default 3).",
        },
        "clear_first": {
            "type": "boolean",
            "description": "For type: clear existing field content before typing (default true).",
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Why this action? What do you expect to happen? "
                "Reference specific [N] element IDs from the page."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0 — how confident this is the right action?",
        },
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What page information informed this? e.g. ['[5] is the submit button']",
        },
    },
    "required": ["action_type", "reasoning", "confidence", "evidence_sources"],
}


# ---------------------------------------------------------------------------
# Batch Action Schemas — Plan-Then-Execute (2 LLM calls instead of N)
# ---------------------------------------------------------------------------

CU_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["fill_form", "search_and_select", "click_single",
                     "scroll_and_read", "navigate", "stop"],
            "description": "What type of action to take next.",
        },
        "target_description": {
            "type": "string",
            "description": "What you intend to do (e.g., 'Fill the login form', 'Search for Koregaon Park').",
        },
        "reasoning": {
            "type": "string",
            "description": "Why this action type. Reference [N] IDs.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence.",
        },
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Page elements that informed this decision.",
        },
    },
    "required": ["action_type", "target_description", "reasoning", "confidence", "evidence_sources"],
}

FILL_FORM_SCHEMA = {
    "type": "object",
    "properties": {
        "field_actions": {
            "type": "array",
            "description": "Sequence of fills. Each fills one field.",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "integer", "description": "The [N] ID of the field"},
                    "action": {"type": "string", "enum": ["type", "select", "click"],
                               "description": "type for text fields, select for dropdowns, click for checkboxes"},
                    "value": {"type": "string", "description": "Value to type/select, or empty for click"},
                    "clear_first": {"type": "boolean", "description": "Clear field before typing (default true)"},
                },
                "required": ["element_id", "action", "value"],
            },
        },
        "submit_action": {
            "type": "object",
            "description": "How to submit after filling. Omit if no submission needed yet.",
            "properties": {
                "element_id": {"type": "integer", "description": "Submit button [N] ID"},
                "action": {"type": "string", "enum": ["click", "press_key"]},
                "value": {"type": "string", "description": "Key name for press_key, empty for click"},
            },
            "required": ["element_id", "action"],
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["field_actions", "reasoning"],
}

SEARCH_AND_SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "search_field_id": {"type": "integer", "description": "The [N] ID of the search input"},
        "query": {"type": "string", "description": "Text to type into the search field"},
        "suggestion_preference": {
            "type": "string",
            "description": "Which suggestion to select: 'first', or a keyword to match (e.g., 'Koregaon Park')",
        },
        "submit_after_select": {
            "type": "boolean",
            "description": "Press Enter after selecting? Default false.",
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["search_field_id", "query", "suggestion_preference", "reasoning"],
}


# ---------------------------------------------------------------------------
# SoM Screenshot Annotation (owned by CU, not session_manager)
# ---------------------------------------------------------------------------

async def generate_som_screenshot(
    session: SessionManager,
    elements: list[InteractiveElement],
) -> str:
    """Inject SoM labels on page, screenshot, remove labels. Returns base64 JPEG."""
    assert session.page is not None

    elem_data = [
        {
            "id": el.element_id,
            "x": el.bounding_box.get("x", 0),
            "y": el.bounding_box.get("y", 0),
            "w": el.bounding_box.get("width", 0),
            "h": el.bounding_box.get("height", 0),
        }
        for el in elements
        if el.is_visible
    ]

    # Inject temporary annotation overlay
    await session.page.evaluate("""(elems) => {
        const container = document.createElement('div');
        container.id = '__morphnet_som_overlay__';
        container.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:999999;';
        document.body.appendChild(container);

        for (const el of elems) {
            // Thin border around element
            const border = document.createElement('div');
            border.style.cssText = 'position:absolute;box-sizing:border-box;pointer-events:none;'
                + 'border:1px solid rgba(30,80,220,0.45);'
                + 'top:' + el.y + 'px;left:' + el.x + 'px;'
                + 'width:' + el.w + 'px;height:' + el.h + 'px;';
            container.appendChild(border);

            // Small ID pill at top-left
            const label = document.createElement('div');
            label.textContent = el.id;
            label.style.cssText = 'position:absolute;pointer-events:none;'
                + 'background:rgba(30,80,220,0.78);color:#fff;'
                + 'font:bold 10px/1.2 monospace;padding:1px 4px;border-radius:3px;'
                + 'top:' + Math.max(el.y - 14, 0) + 'px;left:' + el.x + 'px;'
                + 'white-space:nowrap;';
            container.appendChild(label);
        }
    }""", elem_data)

    # Tall pages produce images that exceed Gemini's processing limits.
    # Use viewport-only for pages taller than 5x viewport.
    page_height = await session.page.evaluate("document.documentElement.scrollHeight")
    use_full_page = page_height <= session.viewport_height * 5
    annotated = await session.page.screenshot(full_page=use_full_page, type="jpeg", quality=85)

    # Remove overlay
    await session.page.evaluate("""() => {
        const el = document.getElementById('__morphnet_som_overlay__');
        if (el) el.remove();
    }""")

    return base64.b64encode(annotated).decode()


# ---------------------------------------------------------------------------
# AXTree Helpers
# ---------------------------------------------------------------------------

def _count_axtree_nodes(node: dict) -> int:
    """Count total nodes in AXTree dict. Used to detect empty trees during page transitions."""
    count = 1
    for child in node.get("children", []):
        count += _count_axtree_nodes(child)
    return count


# ---------------------------------------------------------------------------
# Viewport-Aware Element Filtering
# ---------------------------------------------------------------------------

def filter_viewport_elements(
    elements: list[InteractiveElement],
    viewport_height: int,
    scroll_y: int = 0,
) -> list[InteractiveElement]:
    """Keep elements that are visible, in viewport range, and non-zero size.

    Three filters applied in order:
    1. is_visible=True (from JS enumerator's computed-style check)
    2. Within current viewport + one viewport below
    3. Non-zero width and height (skip collapsed/hidden elements)
    """
    max_y = scroll_y + (viewport_height * 2)
    filtered = []
    for el in elements:
        # Filter 1: JS-computed visibility (top-level field on InteractiveElement)
        if not el.is_visible:
            continue
        # Filter 2: Viewport range
        y = el.bounding_box.get("y", 0)
        if y >= max_y:
            continue
        # Filter 3: Non-zero dimensions (collapsed/display:none elements)
        w = el.bounding_box.get("width", 0)
        h = el.bounding_box.get("height", 0)
        if w <= 0 or h <= 0:
            continue
        filtered.append(el)
    return filtered


# Role priority for dedup: higher-priority roles win when two elements overlap
_ROLE_PRIORITY = {
    "button": 10, "link": 10, "textbox": 10, "combobox": 10,
    "checkbox": 9, "radio": 9, "switch": 9, "slider": 9, "spinbutton": 9,
    "menuitem": 8, "menuitemcheckbox": 8, "menuitemradio": 8, "tab": 8,
    "option": 7, "treeitem": 7, "gridcell": 7,
    "search": 6, "navigation": 5, "region": 4,
}


def deduplicate_by_relevance(
    elements: list[InteractiveElement],
) -> list[InteractiveElement]:
    """Remove near-duplicate elements sharing the same accessible name.

    When multiple elements have identical name + role, keep the one with:
    1. Higher role priority (interactive roles beat structural)
    2. Larger bounding box area (more prominent)
    3. Closer to page top (earlier in visual flow)
    """
    if len(elements) <= 1:
        return elements

    # Group by (lowercase name, role) — only dedup within same name+role
    groups: dict[tuple[str, str], list[InteractiveElement]] = {}
    for el in elements:
        key = (el.name.strip().lower(), el.role.lower())
        if not key[0]:  # Don't dedup unnamed elements
            continue
        groups.setdefault(key, []).append(el)

    # Find which elements to remove (keep best from each group)
    to_remove: set[int] = set()
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        # Sort: highest role priority, then largest area, then closest to top
        group.sort(key=lambda e: (
            -_ROLE_PRIORITY.get(e.role.lower(), 0),
            -(e.bounding_box.get("width", 0) * e.bounding_box.get("height", 0)),
            e.bounding_box.get("y", 0),
        ))
        # Keep the first (best), mark the rest for removal
        for el in group[1:]:
            to_remove.add(el.element_id)

    if not to_remove:
        return elements
    return [el for el in elements if el.element_id not in to_remove]


# ---------------------------------------------------------------------------
# Screenshot Decision Logic
# ---------------------------------------------------------------------------

def should_include_screenshot(
    step: int,
    last_action_failed: bool,
    consecutive_failures: int,
) -> bool:
    """SoM screenshot on first action, after failure, or after 2+ consecutive failures."""
    return step == 1 or last_action_failed or consecutive_failures >= 2


# ---------------------------------------------------------------------------
# Action History (flat, recency-weighted)
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    """One step in a subtask execution."""
    step: int
    action: dict
    reflection: dict           # Full reflection result from Reflector
    one_line_summary: str      # "Step 3: Typed 'admin' into [4] — success"

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action,
            "verdict": self.reflection.get("verdict", {}),
            "deterministic_signals": self.reflection.get("deterministic_signals", {}),
        }


def build_action_history(records: list[ActionRecord], current_step: int) -> str:
    """Last 2 steps: full detail. Earlier: one-line summaries."""
    if not records:
        return "(no actions taken yet)"

    lines: list[str] = []
    for record in records:
        if record.step >= current_step - 2:
            # Full detail for recent steps
            action = record.action
            verdict = record.reflection.get("verdict", {})
            at = action.get("action_type", "?")
            eid = action.get("element_id", "")
            text = action.get("text", "")[:40]
            reasoning = action.get("reasoning", "")[:80]
            what = verdict.get("what_changed", "")[:60]
            success = "success" if verdict.get("success") else "failure"
            hint = verdict.get("correction_hint", "")

            line = f"Step {record.step}: {at}"
            if eid:
                line += f" [{eid}]"
            if text:
                line += f' "{text}"'
            line += f" → {success}: {what}"
            if hint:
                line += f" (hint: {hint})"
            if reasoning:
                line += f"\n  Agent reasoning: {reasoning}"
            lines.append(line)
        else:
            # One-line summary for older steps
            lines.append(record.one_line_summary)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SubtaskResult
# ---------------------------------------------------------------------------

@dataclass
class SubtaskResult:
    success: bool
    actions_taken: list[ActionRecord]
    subtask_reflection: dict          # From reflector.reflect_on_subtask()
    final_url: str
    final_elements: list[InteractiveElement]
    extracted_data: str | None        # For retrieval tasks
    notes: list[str]                  # Collected from "note" actions
    traffic_during_subtask: list[CapturedRequest]
    steps_used: int


# ---------------------------------------------------------------------------
# ComputerUseAgent
# ---------------------------------------------------------------------------

class ComputerUseAgent:
    """Browser action agent. 10 actions per subtask.

    Builds AgentOccam-pruned AXTree. Uses reflector after each action.
    SoM screenshot on first action and after failures.
    """

    def __init__(
        self,
        session: SessionManager,
        reflector: Reflector,
        trace: TaskTrace,
        observer: Any = None,
    ):
        self.session = session
        self.reflector = reflector
        self.trace = trace
        self._observer = observer  # Observer instance for recording CU actions (optional)
        self._cu_prompt = self._load_prompt("cu_action.txt")
        # Contextual prompt components (Phase 3: AXI-inspired injection)
        self._cu_core = self._load_prompt("cu_core.txt")
        self._cu_plan_prompt = self._load_prompt("cu_plan.txt")
        self._context_prompts = {
            "form": self._load_prompt("cu_context_form.txt"),
            "search": self._load_prompt("cu_context_search.txt"),
            "listing": self._load_prompt("cu_context_listing.txt"),
            "recovery": self._load_prompt("cu_context_recovery.txt"),
        }
        self._subtask_counter = 0  # Incremented per execute_subtask call

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = PROMPTS_DIR / filename
        if not path.exists():
            # Not a warning — optional prompt files may not exist yet
            return ""
        return path.read_text(encoding="utf-8")

    def _build_contextual_prompt(
        self,
        elements: list[InteractiveElement],
        last_action_failed: bool = False,
    ) -> str:
        """Build a contextual system prompt by injecting only relevant sections.

        Uses cu_core.txt (always, ~600 tokens) + context-specific files
        based on page analysis. Falls back to full cu_action.txt if
        cu_core.txt is not available.
        """
        if not self._cu_core:
            return self._cu_prompt  # Fallback to full prompt

        ctx = analyze_page_context(elements)

        parts = [self._cu_core]
        if ctx.get("has_form") and self._context_prompts.get("form"):
            parts.append(self._context_prompts["form"])
        if ctx.get("has_search") and self._context_prompts.get("search"):
            parts.append(self._context_prompts["search"])
        if ctx.get("has_listing") and self._context_prompts.get("listing"):
            parts.append(self._context_prompts["listing"])
        if last_action_failed and self._context_prompts.get("recovery"):
            parts.append(self._context_prompts["recovery"])

        return "\n\n".join(parts)

    def _log(self, event_type: str, summary: str, **kwargs) -> str | None:
        return self.trace.log("cu_agent", event_type, summary, **kwargs)

    async def _recover_from_crash(self, fallback_url: str) -> bool:
        """Attempt to recover from a Chromium render process crash.

        Opens a new page in the existing context, navigates to fallback_url,
        and re-registers traffic capture. Returns True if recovery succeeded.
        """
        try:
            if self.session._context is None:
                return False
            # Try creating a fresh page
            new_page = await self.session._context.new_page()
            self.session.page = new_page
            await self.session._setup_traffic_capture(new_page)
            await new_page.goto(fallback_url, wait_until="domcontentloaded", timeout=15_000)
            await self.session.wait_for_page_ready()
            self._log("crash_recovered", f"Recovered from crash — navigated to {fallback_url[:60]}", detail={
                "fallback_url": fallback_url,
            })
            logger.warning("Recovered from Chromium crash — new page at %s", fallback_url[:60])
            return True
        except Exception as exc:
            logger.error("Crash recovery failed: %s", exc)
            return False

    async def execute_subtask(
        self,
        subtask_description: str,
        max_actions: int = 10,
    ) -> SubtaskResult:
        """Execute subtask within action budget. Uses n+1 extraction pattern."""
        self._subtask_counter += 1
        self._log("subtask_started", f"CU subtask: {subtask_description[:80]}", detail={
            "subtask": subtask_description,
            "max_actions": max_actions,
        })

        traffic_start = time.time()
        records: list[ActionRecord] = []
        notes: list[str] = []
        consecutive_failures = 0
        last_action_failed = False
        # Track failed element+action pairs to warn against repetition
        failed_actions: dict[str, int] = {}  # "element_id:action_type" → failure count
        # Track action signatures to detect loops (same action on same element)
        action_signatures: list[str] = []  # sequence of "action_type:element_id"
        # Track batch failures — disable batch after repeated failures to stop wasting actions
        consecutive_batch_failures = 0
        batch_disabled = False

        # Initial state extraction (once, with navigation retry)
        await self.session.wait_for_page_ready()
        # Dismiss popups that may have appeared during navigation
        try:
            await self.session.dismiss_popups(max_rounds=2)
        except Exception:
            pass
        for _retry in range(3):
            try:
                elements, axtree = await asyncio.gather(
                    self.session.get_interactive_elements(),
                    self.session.get_raw_accessibility_tree(),
                )
                current_url = self.session.page.url
                break
            except Exception as e:
                if "context was destroyed" in str(e).lower() and _retry < 2:
                    logger.warning("Initial extraction: context destroyed (retry %d/2)", _retry + 1)
                    try:
                        await self.session.page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        await asyncio.sleep(1)
                    continue
                raise
        initial_url = current_url

        for step in range(1, max_actions + 1):
            # Human-like pause between actions (bot detection mitigation)
            await asyncio.sleep(0.8 + random.random() * 1.2)

            # 0.5: Run enrichments + scroll position in parallel
            enriched_context: dict[int, str] = {}

            async def _run_enrichments():
                try:
                    enrichments = await run_enrichments(elements, self.session.page, axtree)
                    return apply_enrichments(elements, enrichments)
                except Exception as exc:
                    logger.debug("Enrichments failed (non-fatal): %s", exc)
                    return {}

            enriched_context, scroll_y = await asyncio.gather(
                _run_enrichments(), self._get_scroll_y(),
            )

            # 1. Build pruned AXTree from current state (already in memory)
            viewport_h = self.session.viewport_height
            filtered_elements = filter_viewport_elements(elements, viewport_h, scroll_y)
            filtered_elements = deduplicate_by_relevance(filtered_elements)
            axtree_text = build_cu_representation(
                axtree or {}, filtered_elements, enriched_context=enriched_context,
            )

            # 2. Maybe generate SoM screenshot
            screenshot_b64: str | None = None
            if should_include_screenshot(step, last_action_failed, consecutive_failures):
                screenshot_b64 = await generate_som_screenshot(self.session, filtered_elements)

            # 3. Build action history
            history = build_action_history(records, step)

            # 3.5: Build contextual system prompt (AXI: inject only relevant sections)
            system_prompt = self._build_contextual_prompt(filtered_elements, last_action_failed)

            # 3.7: Try batch action if page has forms/search (Plan-Then-Execute)
            batch_result = None
            if not batch_disabled:
                batch_result = await self._try_batch_action(
                    subtask_description, axtree_text, history,
                    step, max_actions, screenshot_b64, filtered_elements,
                )
            if batch_result is not None:
                # Record batch action in observer
                if self._observer:
                    try:
                        await self._observer.record_cu_action(
                            action_type=f"batch_{batch_result.get('type', 'unknown')}",
                            target={"batch": True},
                            value=json.dumps(batch_result.get("completed", []))[:500],
                            reasoning=batch_result.get("reasoning", "batch action"),
                        )
                    except Exception:
                        pass
                batch_success = batch_result.get("success", False)
                summary = f"Step {step}: batch ({batch_result.get('type', '?')}) — {'success' if batch_success else 'failure'}"
                records.append(ActionRecord(
                    step=step,
                    action={"action_type": "batch", "batch_detail": batch_result},
                    reflection={"verdict": {"success": batch_success, "what_changed": summary}},
                    one_line_summary=summary,
                ))
                if batch_success:
                    last_action_failed = False
                    consecutive_failures = 0
                    consecutive_batch_failures = 0
                else:
                    last_action_failed = True
                    consecutive_failures += 1
                    consecutive_batch_failures += 1
                    if consecutive_batch_failures >= 2:
                        batch_disabled = True
                        logger.info("Batch disabled after %d consecutive failures", consecutive_batch_failures)
                # Re-extract state after batch
                try:
                    await self.session.wait_for_page_ready()
                    elements, axtree = await asyncio.gather(
                        self.session.get_interactive_elements(),
                        self.session.get_raw_accessibility_tree(),
                    )
                    current_url = self.session.page.url
                except Exception:
                    pass
                continue

            # 4. Call Gemini for next action (single-action path)
            action_dict = await self._select_action(
                subtask_description, axtree_text, history,
                step, max_actions, screenshot_b64,
                failed_actions=failed_actions,
                action_signatures=action_signatures,
                system_prompt=system_prompt,
            )

            # 4b. Save step representations for debugging
            # Also take a raw (un-annotated) screenshot for comparison
            raw_screenshot_b64: str | None = None
            if screenshot_b64:
                try:
                    raw_ss = await self.session.take_screenshot()
                    raw_screenshot_b64 = raw_ss.image_base64
                except Exception:
                    pass
            self._save_action_step(
                subtask_num=self._subtask_counter,
                step=step,
                subtask=subtask_description,
                raw_elements=elements,
                filtered_elements=filtered_elements,
                raw_axtree=axtree,
                axtree_text=axtree_text,
                history=history,
                screenshot_b64=screenshot_b64,
                raw_screenshot_b64=raw_screenshot_b64,
                failed_actions=failed_actions,
                action=action_dict,
                system_prompt=system_prompt,
            )

            action_type = action_dict.get("action_type", "")

            # 5. Handle "stop"
            if action_type == "stop":
                self._log("subtask_stop", f"CU stop at step {step}: {action_dict.get('text', '')[:80]}", detail={
                    "step": step,
                    "reason": action_dict.get("text", ""),
                    "reasoning": action_dict.get("reasoning", ""),
                })
                records.append(ActionRecord(
                    step=step,
                    action=action_dict,
                    reflection={"verdict": {"success": True, "what_changed": "Agent stopped"}},
                    one_line_summary=f"Step {step}: stop — {action_dict.get('text', '')[:60]}",
                ))
                break

            # 6. Handle "note"
            if action_type == "note":
                note_text = action_dict.get("text", "") or action_dict.get("reasoning", "")
                notes.append(note_text)
                self._log("note_recorded", f"Note: {note_text[:60]}", detail={
                    "step": step,
                    "note": note_text,
                })
                records.append(ActionRecord(
                    step=step,
                    action=action_dict,
                    reflection={"verdict": {"success": True, "what_changed": f"Noted: {note_text[:40]}"}},
                    one_line_summary=f"Step {step}: note — {note_text[:60]}",
                ))
                last_action_failed = False
                consecutive_failures = 0
                continue

            # 7. Snapshot before-state references (already in memory — zero extraction)
            before_elements = elements
            before_url = current_url
            before_axtree = axtree

            # 7b. Record action in observer (before execution)
            if self._observer:
                try:
                    # Resolve target element info for observer
                    target_info = {}
                    eid = action_dict.get("element_id")
                    if eid is not None:
                        el = self.session._resolve_element(eid)
                        if el:
                            target_info = {
                                "element_id": eid,
                                "selector": el.selector if hasattr(el, "selector") else "",
                                "text": (el.name or "")[:200],
                                "ax_node_id": getattr(el, "ax_node_id", None),
                                "attributes": {
                                    "role": el.role,
                                    "tag": getattr(el, "tag", ""),
                                },
                            }
                    await self._observer.record_cu_action(
                        action_type=action_type,
                        target=target_info,
                        value=action_dict.get("text"),
                        reasoning=action_dict.get("reasoning", ""),
                    )
                except Exception as obs_exc:
                    logger.debug("Observer record_cu_action failed (non-fatal): %s", obs_exc)

            # 8. Execute action
            result = await self.session.execute_action(action_dict)

            # 9. Get after-state (becomes next iteration's current state)
            #    Retry on page/context loss (TargetClosedError, context destroyed)
            #    — actions can open new tabs, cause SPA navigations, or close pages.
            #    Also retry when AXTree is essentially empty (< 5 nodes) —
            #    page transitions can leave the accessibility tree unpopulated
            #    even after page_ready passes.
            try:
                await self.session.wait_for_page_ready()
            except Exception as e:
                err_low = str(e).lower()
                if "target" in err_low or "closed" in err_low or "crashed" in err_low:
                    logger.warning("Page lost during wait_for_page_ready, attempting reattach: %s", str(e)[:80])
                    reattached = await self.session.reattach_page()
                    if reattached:
                        await self.session.wait_for_page_ready()
                    else:
                        # Chromium render process crashed — try navigating back
                        recovered = await self._recover_from_crash(before_url)
                        if not recovered:
                            self._log("crash_unrecoverable", f"Browser crashed at step {step}, cannot recover", detail={
                                "error": str(e)[:200], "step": step,
                            }, outcome="failure")
                            break
            for _retry in range(3):
                try:
                    elements, axtree = await asyncio.gather(
                        self.session.get_interactive_elements(),
                        self.session.get_raw_accessibility_tree(),
                    )
                    current_url = self.session.page.url
                    # Check for empty AXTree (page still transitioning)
                    node_count = _count_axtree_nodes(axtree) if axtree else 0
                    if node_count < 5 and _retry < 2:
                        logger.warning("AXTree nearly empty (%d nodes, retry %d/2) — page likely transitioning", node_count, _retry + 1)
                        await asyncio.sleep(1)
                        await self.session.wait_for_page_ready()
                        continue
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    is_recoverable = (
                        "context was destroyed" in err_msg
                        or "target page" in err_msg
                        or "target closed" in err_msg
                        or "has been closed" in err_msg
                        or "target crashed" in err_msg
                    )
                    if is_recoverable and _retry < 2:
                        logger.warning("Page/context lost (retry %d/2): %s", _retry + 1, str(e)[:80])
                        # Try to reattach to a surviving page
                        reattached = await self.session.reattach_page()
                        if reattached:
                            await self.session.wait_for_page_ready()
                            continue
                        # Chromium process may have crashed — try full recovery
                        recovered = await self._recover_from_crash(before_url)
                        if recovered:
                            continue
                        # Fallback: wait for navigation to settle
                        try:
                            await self.session.page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except Exception:
                            await asyncio.sleep(1)
                        continue
                    raise

            # 9b. Dismiss popups after navigation (common on e-commerce sites)
            if current_url != before_url:
                try:
                    dismissed = await self.session.dismiss_popups(max_rounds=2)
                    if dismissed:
                        # Re-extract state after popup dismissal
                        elements, axtree = await asyncio.gather(
                            self.session.get_interactive_elements(),
                            self.session.get_raw_accessibility_tree(),
                        )
                except Exception:
                    pass

            # 10. Reflect on action — pass element lists, URLs, and both AXTrees
            reflection = await self.reflector.reflect_on_action(
                action_dict, result,
                before_elements, elements,
                before_url, current_url,
                subtask_description,
                after_axtree=axtree,
                before_axtree=before_axtree,
            )

            # 11. Record
            verdict = reflection.get("verdict", {})
            action_success = verdict.get("success", False)
            what_changed = verdict.get("what_changed", "")[:60]

            eid = action_dict.get("element_id", "")
            text_preview = action_dict.get("text", "")[:30]
            eid_str = f" [{eid}]" if eid else ""
            text_str = f' "{text_preview}"' if text_preview else ""
            status = "success" if action_success else "failure"

            record = ActionRecord(
                step=step,
                action=action_dict,
                reflection=reflection,
                one_line_summary=f"Step {step}: {action_type}{eid_str}{text_str} — {status} ({what_changed})",
            )
            records.append(record)

            # Track action signature for loop detection
            sig_eid = action_dict.get("element_id", "?")
            action_signatures.append(f"{action_type}:{sig_eid}")

            if action_success:
                last_action_failed = False
                consecutive_failures = 0
            else:
                last_action_failed = True
                consecutive_failures += 1
                # Track which element+action pairs have failed
                fail_eid = action_dict.get("element_id", "")
                fail_key = f"{fail_eid}:{action_type}"
                failed_actions[fail_key] = failed_actions.get(fail_key, 0) + 1

        # --- After loop: subtask reflection ---
        # Get optional screenshot for subtask reflection
        subtask_screenshot: str | None = None
        if records and not records[-1].reflection.get("verdict", {}).get("success", False):
            screenshot = await self.session.take_screenshot()
            subtask_screenshot = screenshot.image_base64

        subtask_reflection = await self.reflector.reflect_on_subtask(
            subtask_description,
            [r.to_dict() for r in records],
            initial_url,
            current_url,
            elements,
            axtree,
            notes=notes,
            screenshot_base64=subtask_screenshot,
        )

        traffic = self.session.get_captured_traffic(since_timestamp=traffic_start)

        subtask_result = SubtaskResult(
            success=subtask_reflection.get("subtask_achieved", False),
            actions_taken=records,
            subtask_reflection=subtask_reflection,
            final_url=current_url,
            final_elements=elements,
            extracted_data=subtask_reflection.get("extracted_data"),
            notes=notes,
            traffic_during_subtask=traffic,
            steps_used=len(records),
        )

        self._log("subtask_completed", f"CU subtask {'succeeded' if subtask_result.success else 'failed'}", detail={
            "subtask": subtask_description[:80],
            "steps_used": subtask_result.steps_used,
            "success": subtask_result.success,
            "recommendation": subtask_reflection.get("recommendation", ""),
            "notes_count": len(notes),
        }, outcome="success" if subtask_result.success else "failure")

        return subtask_result

    def _save_action_step(
        self, *, subtask_num: int, step: int, subtask: str,
        raw_elements: list, filtered_elements: list,
        raw_axtree: dict | None, axtree_text: str, history: str,
        screenshot_b64: str | None, raw_screenshot_b64: str | None = None,
        failed_actions: dict | None, action: dict,
        system_prompt: str | None = None,
    ) -> None:
        """Save CU action step data for post-run analysis.

        Saves both raw inputs and processed outputs so we can compare:
        - Raw AXTree (compact summary) vs pruned AXTree
        - Raw elements vs filtered elements
        - Full prompt exactly as sent to LLM
        - System instruction (cu_action.txt)
        - SoM screenshot + raw screenshot side by side
        """
        # Build element summaries (compact: id, name, role, tag, bbox, visible)
        def _el_summary(el) -> dict:
            return {
                "id": el.element_id,
                "name": (el.name or "")[:80],
                "role": el.role,
                "tag": getattr(el, "tag", None),
                "bbox": el.bounding_box,
                "visible": el.is_visible,
            }

        # Compact raw AXTree summary: count nodes by role, total depth
        raw_axtree_summary = summarize_raw_axtree(raw_axtree) if raw_axtree else {}

        # Reconstruct the prompt (same logic as _select_action)
        remaining = 10 - step + 1
        prompt = (
            f"Subtask: {subtask}\n\n"
            f"Step {step} of 10 ({remaining} remaining)\n\n"
            f"Current Page:\n{axtree_text[:12000]}\n\n"
            f"Action History:\n{history}\n"
        )
        if failed_actions:
            warnings = []
            for key, count in failed_actions.items():
                eid, at = key.split(":", 1)
                warnings.append(f"  - {at} on element [{eid}] failed {count} time(s)")
            prompt += (
                "\nWARNING — Previously failed actions (do NOT repeat these):\n"
                + "\n".join(warnings)
                + "\nTry a different element or approach instead.\n"
            )

        data = {
            "type": "action_step",
            "subtask_num": subtask_num,
            "step": step,
            "subtask": subtask,
            "raw": {
                "elements_count": len(raw_elements),
                "elements": [_el_summary(el) for el in raw_elements[:200]],
                "axtree_summary": raw_axtree_summary,
                "axtree_text_preview": render_raw_axtree_text(raw_axtree, max_lines=150) if raw_axtree else "",
            },
            "processed": {
                "filtered_elements_count": len(filtered_elements),
                "filtered_elements": [_el_summary(el) for el in filtered_elements[:100]],
                "element_summary": format_elements_summary(filtered_elements),
                "pruned_axtree": axtree_text,
                "pruned_axtree_chars": len(axtree_text),
                "pruned_axtree_words": len(axtree_text.split()),
            },
            "llm_input": {
                "system_instruction": system_prompt or self._cu_prompt,
                "user_prompt": prompt,
                "has_screenshot": screenshot_b64 is not None,
                "total_prompt_chars": len(system_prompt or self._cu_prompt) + len(prompt),
            },
            "response": action,
        }

        step_name = f"action_{subtask_num:03d}_{step:02d}"
        self.trace.save_step(step_name, data)

        # Save SoM-annotated screenshot
        if screenshot_b64:
            self.trace.save_screenshot(f"{step_name}_som", screenshot_b64)
        # Save raw (un-annotated) screenshot for comparison
        if raw_screenshot_b64:
            self.trace.save_screenshot(f"{step_name}_raw", raw_screenshot_b64)

    async def _select_action(
        self,
        subtask: str,
        axtree: str,
        history: str,
        step: int,
        max_actions: int,
        screenshot_b64: str | None,
        *,
        failed_actions: dict[str, int] | None = None,
        action_signatures: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        """Call Gemini Flash Preview for next action, wrapped in trace.span()."""
        remaining = max_actions - step + 1

        prompt = (
            f"Subtask: {subtask}\n\n"
            f"Step {step} of {max_actions} ({remaining} remaining)\n\n"
            f"Current Page:\n{axtree[:12000]}\n\n"
            f"Action History:\n{history}\n"
        )

        # Inject failure warnings so agent avoids repeating failed actions
        if failed_actions:
            warnings = []
            for key, count in failed_actions.items():
                eid, at = key.split(":", 1)
                warnings.append(f"  - {at} on element [{eid}] failed {count} time(s)")
            prompt += (
                "\nWARNING — Previously failed actions (do NOT repeat these):\n"
                + "\n".join(warnings)
                + "\nTry a different element or approach instead.\n"
            )

        # Detect action loops — same action on same element repeated 3+ times
        sigs = action_signatures or []
        if len(sigs) >= 3:
            from collections import Counter
            sig_counts = Counter(sigs[-6:])  # last 6 actions
            repeated = [(sig, cnt) for sig, cnt in sig_counts.items() if cnt >= 3]
            if repeated:
                loop_items = [f"  - {sig} repeated {cnt} times" for sig, cnt in repeated]
                prompt += (
                    "\nLOOP DETECTED — You are repeating the same actions:\n"
                    + "\n".join(loop_items) + "\n"
                )

        # Inject user credentials for form-filling (booking, registration, etc.)
        creds = self.session.get_credentials()
        if creds:
            user = creds.get("user", {})
            if user:
                prompt += "\nUser Info (use for forms/booking):\n"
                for k, v in user.items():
                    prompt += f"  {k}: {v}\n"

        contents: list[Any] = [prompt]
        if screenshot_b64:
            contents.append({"mime_type": "image/jpeg", "data": screenshot_b64})

        effective_prompt = system_prompt or self._cu_prompt
        with self.trace.span("cu_agent", "action_selected", f"Step {step}: selecting action") as span:
            action = await call_gemini_async(
                model="gemini-3-flash-preview",
                contents=contents,
                response_schema=CU_ACTION_SCHEMA,
                system_instruction=effective_prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 8192},
                prompt_log_dir=self.trace.output_dir / "prompt_made",
            )
            span.set_reasoning(action.get("reasoning", ""))
            span.set_confidence(action.get("confidence", 0.0))
            for src in action.get("evidence_sources", []):
                span.add_evidence(Evidence("model_output", src))
            span.set_detail("action_type", action.get("action_type", ""))
            span.set_detail("element_id", action.get("element_id"))
            span.set_detail("action", action)
            span.set_outcome("success")

        return action

    async def _try_batch_action(
        self,
        subtask: str,
        axtree_text: str,
        history: str,
        step: int,
        max_actions: int,
        screenshot_b64: str | None,
        elements: list[InteractiveElement],
    ) -> dict | None:
        """Try batch action (Plan-Then-Execute) if page has forms/search.

        Returns batch result dict if a batch was executed, None if not appropriate.
        The caller handles recording and state re-extraction.
        """
        if not self._cu_plan_prompt:
            return None

        ctx = analyze_page_context(elements)
        if not (ctx.get("has_form") or ctx.get("has_search")):
            return None

        try:
            plan = await self._plan_batch_action(
                subtask, axtree_text, history, step, max_actions, screenshot_b64,
            )
        except Exception as exc:
            logger.debug("Batch plan failed: %s", exc)
            return None

        plan_type = plan.get("action_type", "")
        if plan_type not in ("fill_form", "search_and_select"):
            return None  # Single-action types — fall through to _select_action

        # 2nd LLM call: get the execution details
        target = plan.get("target_description", subtask)
        exec_prompt = f"Subtask: {target}\n\nPage:\n{axtree_text[:10000]}"
        exec_contents: list[Any] = [exec_prompt]
        if screenshot_b64:
            exec_contents.append({"mime_type": "image/jpeg", "data": screenshot_b64})

        try:
            if plan_type == "fill_form":
                exec_result = await call_gemini_async(
                    model="gemini-3-flash-preview",
                    contents=exec_contents,
                    response_schema=FILL_FORM_SCHEMA,
                    system_instruction=self._cu_plan_prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 4096},
                    prompt_log_dir=self.trace.output_dir / "prompt_made",
                )
                result = await self._execute_fill_form(exec_result)
                result["type"] = "fill_form"
                return result

            elif plan_type == "search_and_select":
                exec_result = await call_gemini_async(
                    model="gemini-3-flash-preview",
                    contents=exec_contents,
                    response_schema=SEARCH_AND_SELECT_SCHEMA,
                    system_instruction=self._cu_plan_prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 4096},
                    prompt_log_dir=self.trace.output_dir / "prompt_made",
                )
                result = await self._execute_search_and_select(exec_result)
                result["type"] = "search_and_select"
                return result
        except Exception as exc:
            logger.warning("Batch execution failed: %s", exc)
            return None

        return None

    async def _get_scroll_y(self) -> int:
        """Get current vertical scroll position."""
        if self.session.page is None:
            return 0
        try:
            return await self.session.page.evaluate("() => window.scrollY")
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Batch Actions — Plan-Then-Execute (Phase 2)
    # ------------------------------------------------------------------

    async def _plan_batch_action(
        self,
        subtask: str,
        axtree_text: str,
        history: str,
        step: int,
        max_actions: int,
        screenshot_b64: str | None,
    ) -> dict:
        """Call 1: decide WHAT type of action to take (plan step)."""
        if not self._cu_plan_prompt:
            return {}  # Batch actions not available — fall back to single action

        prompt = (
            f"Subtask: {subtask}\n\n"
            f"Step {step}/{max_actions}\n\n"
            f"Page:\n{axtree_text}\n\n"
            f"History:\n{history}"
        )
        contents: list = [prompt]
        if screenshot_b64:
            contents.append({"mime_type": "image/jpeg", "data": screenshot_b64})

        result = await call_gemini_async(
            model="gemini-3-flash-preview",
            contents=contents,
            response_schema=CU_PLAN_SCHEMA,
            system_instruction=self._cu_plan_prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 4096},
            prompt_log_dir=self.trace.output_dir / "prompt_made",
        )
        return result

    async def _execute_fill_form(
        self,
        exec_result: dict,
    ) -> dict:
        """Execute a fill_form batch: fill each field, then optionally submit.

        Returns {"success": bool, "completed": list, "actions_count": int}.
        """
        # Pre-check: abort if any type-target is a non-native form element.
        # React/Vue/Svelte render custom widgets as <div>/<span>/<p> — typing
        # into these via JS textContent won't update framework state, causing
        # silent failures where the DOM text changes but form submission uses
        # the original framework state values.
        _NATIVE_FORM_TAGS = {"input", "textarea", "select"}
        for fa in exec_result.get("field_actions", []):
            if fa["action"] == "type":
                el = self.session._resolve_element(fa["element_id"])
                if el and el.tag not in _NATIVE_FORM_TAGS:
                    logger.debug(
                        "fill_form abort: element [%d] is <%s>, not a native form field",
                        fa["element_id"], el.tag,
                    )
                    return {
                        "success": False, "completed": [],
                        "reason": f"Element [{fa['element_id']}] is <{el.tag}>, not a native form field",
                        "actions_count": 0,
                    }

        executed: list[dict] = []
        for field_action in exec_result.get("field_actions", []):
            eid = field_action["element_id"]
            action = field_action["action"]
            value = field_action.get("value", "")

            action_dict = {
                "action_type": action,
                "element_id": eid,
                "text": value,
                "clear_first": field_action.get("clear_first", True),
            }
            result = await self.session.execute_action(action_dict)
            executed.append({"element_id": eid, "action": action, "result_success": result.success})

            if not result.success:
                return {
                    "success": False, "completed": executed,
                    "failed_at": eid, "reason": result.error,
                    "actions_count": len(executed),
                }

        # Submit if specified
        submit = exec_result.get("submit_action")
        if submit:
            submit_dict = {
                "action_type": submit["action"],
                "element_id": submit.get("element_id"),
                "text": submit.get("value", ""),
            }
            result = await self.session.execute_action(submit_dict)
            executed.append({
                "element_id": submit.get("element_id"),
                "action": submit["action"],
                "result_success": result.success,
                "is_submit": True,
            })

        return {"success": True, "completed": executed, "actions_count": len(executed)}

    async def _execute_search_and_select(
        self,
        exec_result: dict,
    ) -> dict:
        """Execute a search_and_select batch: type query, wait, select suggestion."""
        search_id = exec_result["search_field_id"]
        query = exec_result["query"]

        # Snapshot element IDs before typing so we can detect new suggestions
        try:
            pre_elements = await self.session.get_interactive_elements()
            pre_ids = {el.element_id for el in pre_elements}
        except Exception:
            pre_ids = set()

        # Type the query
        type_result = await self.session.execute_action({
            "action_type": "type",
            "element_id": search_id,
            "text": query,
            "clear_first": True,
        })
        if not type_result.success:
            return {"success": False, "reason": type_result.error, "actions_count": 1}

        # Wait for suggestions to appear
        await asyncio.sleep(1.0)

        # Find suggestion matching preference
        preference = exec_result.get("suggestion_preference", "first")
        suggestion = await self._find_suggestion(preference, pre_element_ids=pre_ids)
        if suggestion:
            click_result = await self.session.execute_action({
                "action_type": "click",
                "element_id": suggestion["element_id"],
            })
            actions_count = 3  # type + wait + click
            if exec_result.get("submit_after_select"):
                await self.session.execute_action({
                    "action_type": "press_key",
                    "text": "Enter",
                })
                actions_count += 1
            return {"success": click_result.success, "actions_count": actions_count}

        return {"success": False, "reason": "No matching suggestion found", "actions_count": 2}

    async def _find_suggestion(
        self,
        preference: str,
        *,
        pre_element_ids: set[int] | None = None,
    ) -> dict | None:
        """Find an autocomplete suggestion matching the preference.

        Reloads elements to find newly appeared suggestions.
        Uses two strategies:
          1. Standard suggestion roles (option, menuitem, listitem, link).
          2. Fallback: any NEW visible named element that appeared after typing
             (detected via pre_element_ids diff), excluding form controls.
        """
        # Roles that are never suggestions (form controls, structural)
        _NON_SUGGESTION_ROLES = frozenset({
            "textbox", "searchbox", "button", "combobox", "checkbox", "radio",
            "slider", "spinbutton", "switch", "tab", "heading", "navigation",
            "banner", "main", "contentinfo", "img", "image", "separator",
        })
        try:
            elements = await self.session.get_interactive_elements()

            # Strategy 1: standard suggestion roles
            suggestions = [
                el for el in elements
                if el.role in ("option", "menuitem", "listitem", "link")
                and el.is_visible
                and el.name
            ]

            # Strategy 2: if no standard suggestions, find NEW elements
            # that appeared after typing (role-agnostic — covers div, cell, etc.)
            if not suggestions and pre_element_ids is not None:
                suggestions = [
                    el for el in elements
                    if el.element_id not in pre_element_ids
                    and el.is_visible
                    and el.name
                    and el.role not in _NON_SUGGESTION_ROLES
                ]

            if not suggestions:
                return None

            if preference.lower() == "first":
                return {"element_id": suggestions[0].element_id}

            # Match by keyword
            pref_lower = preference.lower()
            for el in suggestions:
                if pref_lower in (el.name or "").lower():
                    return {"element_id": el.element_id}

            # No keyword match — return first suggestion
            return {"element_id": suggestions[0].element_id}
        except Exception:
            return None
