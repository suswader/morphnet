"""
computer_use.py — Browser action agent for MorphNet.

Receives a subtask description. Has 10 actions to complete it.
Uses representation.py for page representation. Owns SoM annotation.
Uses reflector after each action.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from morphnet.session_manager import (
    SessionManager, InteractiveElement, CapturedRequest, ActionResult,
    call_gemini,
)
from morphnet.reflector import Reflector
from morphnet.trace import TaskTrace, Evidence
from morphnet.representation import (
    build_cu_representation,
    format_element,
    format_elements_summary,
    summarize_raw_axtree,
    render_raw_axtree_text,
)

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


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

    annotated = await session.page.screenshot(full_page=True, type="jpeg", quality=85)

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
    ):
        self.session = session
        self.reflector = reflector
        self.trace = trace
        self._cu_prompt = self._load_prompt("cu_action.txt")
        self._subtask_counter = 0  # Incremented per execute_subtask call

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = PROMPTS_DIR / filename
        if not path.exists():
            logger.warning("Prompt file not found: %s", path)
            return ""
        return path.read_text(encoding="utf-8")

    def _log(self, event_type: str, summary: str, **kwargs) -> str | None:
        return self.trace.log("cu_agent", event_type, summary, **kwargs)

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

        # Initial state extraction (once, with navigation retry)
        await self.session.wait_for_page_ready()
        for _retry in range(3):
            try:
                elements = await self.session.get_interactive_elements()
                axtree = await self.session.get_raw_accessibility_tree()
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
            # 1. Build pruned AXTree from current state (already in memory)
            viewport_h = self.session.viewport_height
            scroll_y = await self._get_scroll_y()
            filtered_elements = filter_viewport_elements(elements, viewport_h, scroll_y)
            filtered_elements = deduplicate_by_relevance(filtered_elements)
            axtree_text = build_cu_representation(axtree or {}, filtered_elements)

            # 2. Maybe generate SoM screenshot
            screenshot_b64: str | None = None
            if should_include_screenshot(step, last_action_failed, consecutive_failures):
                screenshot_b64 = await generate_som_screenshot(self.session, filtered_elements)

            # 3. Build action history
            history = build_action_history(records, step)

            # 4. Call Gemini for next action
            action_dict = await self._select_action(
                subtask_description, axtree_text, history,
                step, max_actions, screenshot_b64,
                failed_actions=failed_actions,
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
                if "target" in str(e).lower() or "closed" in str(e).lower():
                    logger.warning("Page lost during wait_for_page_ready, attempting reattach")
                    await self.session.reattach_page()
                    await self.session.wait_for_page_ready()
            for _retry in range(3):
                try:
                    elements = await self.session.get_interactive_elements()
                    axtree = await self.session.get_raw_accessibility_tree()
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
                    )
                    if is_recoverable and _retry < 2:
                        logger.warning("Page/context lost (retry %d/2): %s", _retry + 1, str(e)[:80])
                        # Try to reattach to a surviving page
                        reattached = await self.session.reattach_page()
                        if reattached:
                            await self.session.wait_for_page_ready()
                            continue
                        # Fallback: wait for navigation to settle
                        try:
                            await self.session.page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except Exception:
                            await asyncio.sleep(1)
                        continue
                    raise

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
                "system_instruction": self._cu_prompt,
                "user_prompt": prompt,
                "has_screenshot": screenshot_b64 is not None,
                "total_prompt_chars": len(self._cu_prompt) + len(prompt),
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

        contents: list[Any] = [prompt]
        if screenshot_b64:
            contents.append({"mime_type": "image/jpeg", "data": screenshot_b64})

        with self.trace.span("cu_agent", "action_selected", f"Step {step}: selecting action") as span:
            action = call_gemini(
                model="gemini-3-flash-preview",
                contents=contents,
                response_schema=CU_ACTION_SCHEMA,
                system_instruction=self._cu_prompt,
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

    async def _get_scroll_y(self) -> int:
        """Get current vertical scroll position."""
        if self.session.page is None:
            return 0
        try:
            return await self.session.page.evaluate("() => window.scrollY")
        except Exception:
            return 0
