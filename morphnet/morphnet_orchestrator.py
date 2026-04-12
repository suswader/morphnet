"""
morphnet_orchestrator.py — Task planner and router for MorphNet.

Receives task + URL. Decomposes into subtasks. Routes each to CU or MCP.
Manages AgentOccam branch/prune planning tree and MCP lifecycle.

Representation: text-only AXTree distillation + lightweight DOM summary.
No screenshots — no actionable planning information beyond text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from morphnet.session_manager import (
    SessionManager, InteractiveElement, CapturedRequest,
    ActionResult, call_gemini,
)
from morphnet.reflector import Reflector
from morphnet.computer_use import ComputerUseAgent, SubtaskResult, ActionRecord
from morphnet.trace import TaskTrace, Evidence
from morphnet.representation import build_orchestrator_representation
from morphnet.mcp_manager import MCPManager, MCPToolDefinition

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SITES_DIR = Path(__file__).parent / "sites"


# ---------------------------------------------------------------------------
# Gemini Schema — orchestrator planning
# ---------------------------------------------------------------------------

ORCHESTRATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "description": "2-3 sentences: current state, what is accomplished, what remains.",
        },
        "planning_action": {
            "type": "string",
            "enum": ["continue", "branch", "prune", "complete_task"],
            "description": (
                "continue=next subtask. branch=try new approach. "
                "prune=abandon current. complete_task=done."
            ),
        },
        "branch_intent": {
            "type": "string",
            "description": "Intent of new branch (if planning_action='branch'). Empty otherwise.",
        },
        "prune_reason": {
            "type": "string",
            "description": "Why abandon this approach (if planning_action='prune'). Empty otherwise.",
        },
        "next_subtask": {
            "type": "string",
            "description": (
                "Natural language subtask for CU agent. Must be completable in 10 actions. "
                "e.g. 'Fill login form with username admin and password admin123, then submit'"
            ),
        },
        "routing": {
            "type": "string",
            "enum": ["computer_use", "mcp"],
            "description": "Route to CU (discovery) or existing MCP tool.",
        },
        "mcp_tool_name": {
            "type": "string",
            "description": "Which MCP tool (if routing='mcp'). Empty for CU.",
        },
        "urgency": {
            "type": "string",
            "enum": ["normal", "low_budget", "final_action"],
            "description": "normal=plenty of steps. low_budget=≤3 left. final_action=last step.",
        },
        "website_insights": {
            "type": "string",
            "description": "New insights about this website for profile. Empty if none.",
        },
        "final_answer": {
            "type": "string",
            "description": (
                "If complete_task: the answer. For retrieval: extracted info. "
                "For mutation: confirmation. Empty if not complete."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "Detailed reasoning: why this subtask, why this routing, what evidence.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0",
        },
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What informed this decision?",
        },
    },
    "required": [
        "assessment", "planning_action", "next_subtask", "routing",
        "urgency", "final_answer", "reasoning", "confidence", "evidence_sources",
    ],
}


# ---------------------------------------------------------------------------
# AXTree Utilities
# ---------------------------------------------------------------------------

def _summarize_axtree(node: dict | None, max_depth: int = 2, _depth: int = 0) -> list[dict]:
    """Compact summary of raw AXTree top levels for debugging."""
    if not node or _depth > max_depth:
        return []
    role = node.get("role", {}).get("value", "") if isinstance(node.get("role"), dict) else str(node.get("role", ""))
    name = node.get("name", {}).get("value", "") if isinstance(node.get("name"), dict) else str(node.get("name", ""))
    children = node.get("children", [])
    entry = {"role": role, "name": name[:80], "children_count": len(children)}
    if _depth < max_depth and children:
        entry["children"] = []
        for child in children[:10]:  # Cap at 10 children per level
            entry["children"].extend(_summarize_axtree(child, max_depth, _depth + 1))
    return [entry]




# ---------------------------------------------------------------------------
# Lightweight DOM Summary (orchestrator's view)
# ---------------------------------------------------------------------------

def build_dom_summary(dom_tree: str) -> str:
    """Lightweight DOM structural summary for orchestrator.

    Extracts via deterministic HTML text scanning:
    1. Page landmarks: header, nav, main, footer, aside
    2. Form structures: action URL, method, field names
    3. OpenGraph/Schema.org metadata
    4. Semantic data attributes
    """
    if not dom_tree:
        return "(no DOM available)"

    lines = dom_tree.split("\n")
    landmarks: list[str] = []
    forms: list[str] = []
    metadata: list[str] = []
    data_attrs: list[str] = []

    for line in lines:
        stripped = line.strip().lower()

        # Landmarks
        for tag in ("header", "nav", "main", "footer", "aside"):
            if stripped.startswith(f"<{tag}") or f'role="{tag}"' in stripped or f"role=\"navigation\"" in stripped:
                # Extract brief content hint
                landmarks.append(tag)
                break

        # Forms
        if stripped.startswith("<form"):
            action = _extract_attr(line, "action")
            method = _extract_attr(line, "method") or "GET"
            form_desc = f"{method.upper()} {action}" if action else "form (no action)"
            forms.append(form_desc)

        # OpenGraph metadata
        if "og:" in stripped and "content=" in stripped:
            prop = _extract_attr(line, "property") or _extract_attr(line, "name")
            content = _extract_attr(line, "content")
            if prop and content:
                metadata.append(f"{prop}={content[:60]}")

        # Semantic data attributes with counts or status info
        for attr in ("data-count", "data-cart", "data-qty", "data-page", "data-category", "data-total"):
            if attr in stripped:
                val = _extract_attr(line, attr)
                if val:
                    data_attrs.append(f"{attr}={val}")

    parts: list[str] = []
    if landmarks:
        unique = list(dict.fromkeys(landmarks))
        parts.append(f"Structure: {' | '.join(unique)}")
    if forms:
        parts.append(f"Forms: {'; '.join(forms[:5])}")
    if metadata:
        parts.append(f"Metadata: {', '.join(metadata[:5])}")
    if data_attrs:
        parts.append(f"Data attrs: {', '.join(data_attrs[:5])}")

    return "\n".join(parts) if parts else "(no structural signals found)"


def _extract_attr(line: str, attr_name: str) -> str | None:
    """Extract an HTML attribute value from a line. Simple deterministic parsing."""
    patterns = [f'{attr_name}="', f"{attr_name}='"]
    for pat in patterns:
        idx = line.lower().find(pat.lower())
        if idx >= 0:
            start = idx + len(pat)
            quote = pat[-1]
            end = line.find(quote, start)
            if end > start:
                return line[start:end]
    return None


# ---------------------------------------------------------------------------
# Planning Tree — AgentOccam Branch/Prune
# ---------------------------------------------------------------------------

@dataclass
class BranchSummary:
    """Structured summary of a completed or pruned branch.

    Preserves key context for future planning decisions —
    not a one-liner but a pointed digest.
    """
    what_was_attempted: str     # "Searched for red hoodies using the search bar"
    key_actions: list[str]      # ["Typed 'red hoodie' in search", "Clicked Search"]
    outcome: str                # "Found 20 results but no price sort available"
    reasoning: str              # "The site uses category-based filtering"
    insights_gained: str        # "Left sidebar has category filters, no sort dropdowns"
    data_collected: str | None  # "Cheapest: Basic Red Hoodie at $29.99"


@dataclass
class PlanNode:
    """One node in the planning tree."""
    node_id: str                # "plan_0", "plan_0_1", "plan_0_1_2"
    parent_id: str | None
    intent: str                 # What this branch tries to accomplish
    status: str                 # "active" | "completed" | "pruned"
    prune_reason: str | None
    summary: BranchSummary | None
    children: list[str] = field(default_factory=list)


class PlanningTree:
    """Branch/prune planning tree.

    Manages context growth: completed/pruned branches → structured summaries.
    Only the current active branch retains full context.
    """

    def __init__(self):
        self._nodes: dict[str, PlanNode] = {}
        self._current_id: str | None = None
        self._next_child_idx: dict[str, int] = {}  # parent_id → next child index

    def create_root(self, intent: str) -> str:
        """Create the root planning node. Returns node_id."""
        root_id = "plan_0"
        self._nodes[root_id] = PlanNode(
            node_id=root_id,
            parent_id=None,
            intent=intent,
            status="active",
            prune_reason=None,
            summary=None,
        )
        self._current_id = root_id
        self._next_child_idx[root_id] = 1
        return root_id

    def branch(self, intent: str) -> str:
        """Create a new child branch under the current node. Returns new node_id."""
        assert self._current_id is not None, "No active node — call create_root first"
        parent = self._nodes[self._current_id]

        idx = self._next_child_idx.get(self._current_id, 1)
        new_id = f"{self._current_id}_{idx}"
        self._next_child_idx[self._current_id] = idx + 1

        self._nodes[new_id] = PlanNode(
            node_id=new_id,
            parent_id=self._current_id,
            intent=intent,
            status="active",
            prune_reason=None,
            summary=None,
        )
        parent.children.append(new_id)
        self._next_child_idx[new_id] = 1
        self._current_id = new_id
        return new_id

    def prune(self, reason: str, summary: BranchSummary | None = None) -> str | None:
        """Prune the current branch and return to parent. Returns parent node_id.

        Safety: if pruning would leave _current_id=None (root was pruned),
        auto-branches so callers always have an active node.
        """
        assert self._current_id is not None
        node = self._nodes[self._current_id]
        node.status = "pruned"
        node.prune_reason = reason
        node.summary = summary
        self._current_id = node.parent_id
        if self._current_id is None:
            # Root was pruned — shouldn't happen, but recover
            self._current_id = "plan_0"
            self.branch("Recovery after root prune")
        return self._current_id

    def complete_current(self, summary: BranchSummary) -> str | None:
        """Mark current branch as completed and return to parent. Returns parent node_id.

        Safety: if completing would leave _current_id=None (root was completed),
        auto-branches so callers always have an active node.
        """
        assert self._current_id is not None
        node = self._nodes[self._current_id]
        node.status = "completed"
        node.summary = summary
        self._current_id = node.parent_id
        if self._current_id is None:
            # Root was completed — shouldn't happen, but recover
            self._current_id = "plan_0"
            self.branch("Recovery after root completion")
        return self._current_id

    @property
    def current_node(self) -> PlanNode | None:
        if self._current_id is None:
            return None
        return self._nodes.get(self._current_id)

    def get_context_for_planning(self) -> str:
        """Full tree as text for orchestrator prompt.

        Completed/pruned: BranchSummary fields.
        Active: marked as current focus.
        """
        if not self._nodes:
            return "(empty planning tree)"
        root_id = "plan_0"
        if root_id not in self._nodes:
            return "(empty planning tree)"
        lines: list[str] = []
        self._render_node(root_id, 0, lines)
        return "\n".join(lines)

    def _render_node(self, node_id: str, depth: int, lines: list[str]) -> None:
        node = self._nodes[node_id]
        indent = "│   " * depth
        prefix = "├── " if depth > 0 else ""

        status_marker = {
            "active": "← CURRENT FOCUS" if node_id == self._current_id else "[active]",
            "completed": "[completed]",
            "pruned": "[pruned]",
        }.get(node.status, f"[{node.status}]")

        lines.append(f"{indent}{prefix}{node.node_id}: \"{node.intent}\" {status_marker}")

        if node.summary:
            s = node.summary
            pad = indent + "│   "
            lines.append(f"{pad}Attempted: {s.what_was_attempted}")
            if s.key_actions:
                lines.append(f"{pad}Actions: {', '.join(s.key_actions[:5])}")
            lines.append(f"{pad}Outcome: {s.outcome}")
            if s.reasoning:
                lines.append(f"{pad}Reasoning: {s.reasoning}")
            if s.insights_gained:
                lines.append(f"{pad}Insights: {s.insights_gained}")
            if s.data_collected:
                lines.append(f"{pad}Data: {s.data_collected}")
        elif node.status == "pruned" and node.prune_reason:
            pad = indent + "│   "
            lines.append(f"{pad}Pruned: {node.prune_reason}")

        for child_id in node.children:
            self._render_node(child_id, depth + 1, lines)


# ---------------------------------------------------------------------------
# MCP Lifecycle Tracking
# ---------------------------------------------------------------------------

@dataclass
class MCPToolStatus:
    tool_name: str
    status: str = "verified"   # "verified" | "trusted" | "degraded" | "discarded"
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_reason: str | None = None


def update_mcp_status(
    tool: MCPToolStatus,
    success: bool,
    failure_reason: str | None = None,
) -> None:
    """Update MCP lifecycle status based on outcome.

    Transitions:
    - verified + 3 consecutive successes → trusted
    - trusted + 1 failure → degraded
    - degraded + 2 more consecutive failures → discarded
    - any state + 3 consecutive failures → discarded
    - discarded tools are never routed to again
    """
    if success:
        tool.consecutive_successes += 1
        tool.consecutive_failures = 0
        tool.last_failure_reason = None

        if tool.status == "verified" and tool.consecutive_successes >= 3:
            tool.status = "trusted"
        elif tool.status == "degraded" and tool.consecutive_successes >= 3:
            tool.status = "trusted"
    else:
        tool.consecutive_failures += 1
        tool.consecutive_successes = 0
        tool.last_failure_reason = failure_reason

        match tool.status:
            case "trusted":
                tool.status = "degraded"
            case "degraded":
                if tool.consecutive_failures >= 3:
                    tool.status = "discarded"
            case "verified":
                if tool.consecutive_failures >= 3:
                    tool.status = "discarded"
            case "discarded":
                pass  # Already discarded


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    success: bool
    final_answer: str | None
    subtasks_completed: int
    planning_tree_summary: str
    website_insights: list[str]
    total_actions: int
    total_mcp_calls: int


# ---------------------------------------------------------------------------
# MorphNetOrchestrator
# ---------------------------------------------------------------------------

class MorphNetOrchestrator:
    """Task planner and router.

    Receives a natural language task + start URL. Decomposes into subtasks.
    Routes each to CU or MCP. Manages planning tree and MCP lifecycle.
    """

    def __init__(self, session: SessionManager, trace: TaskTrace):
        self.session = session
        self.trace = trace
        self.reflector = Reflector(session, trace)
        self.cu_agent = ComputerUseAgent(session, self.reflector, trace)
        self.mcp_manager = MCPManager(session, self.reflector, trace)
        self.planning_tree = PlanningTree()
        self._mcp_statuses: dict[str, MCPToolStatus] = {}
        self._website_insights: list[str] = []
        self._orchestrator_prompt = self._load_prompt("orchestrator_plan.txt")
        self._load_mcp_tools()

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = PROMPTS_DIR / filename
        if not path.exists():
            logger.warning("Prompt file not found: %s", path)
            return ""
        return path.read_text(encoding="utf-8")

    def _log(self, event_type: str, summary: str, **kwargs) -> str | None:
        return self.trace.log("orchestrator", event_type, summary, **kwargs)

    def _load_mcp_tools(self) -> None:
        """Load MCP tool definitions from sites/{site_name}/tools.json."""
        site_name = self.session.site_name
        if not site_name:
            return
        tools_path = SITES_DIR / site_name / "tools.json"
        if not tools_path.exists():
            return
        try:
            tools = json.loads(tools_path.read_text())
            for tool in tools:
                name = tool.get("name", "")
                if name:
                    self._mcp_statuses[name] = MCPToolStatus(
                        tool_name=name,
                        status=tool.get("status", "verified"),
                    )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load MCP tools: %s", exc)

    def _get_available_mcp_summary(self) -> str:
        """Build MCP tools summary for orchestrator prompt."""
        if not self._mcp_statuses:
            return "(no MCP tools available)"

        lines: list[str] = []
        for name, status in self._mcp_statuses.items():
            if status.status == "discarded":
                continue
            marker = {
                "trusted": "[TRUSTED]",
                "verified": "[verified]",
                "degraded": "[DEGRADED — use with caution]",
            }.get(status.status, f"[{status.status}]")
            lines.append(f"  {name} {marker}")
            # Include tool description so planner knows when to route
            tool_def = self.mcp_manager.get_tool(name)
            if tool_def:
                lines.append(f"    {tool_def.description[:120]}")
                lines.append(f"    Endpoint: {tool_def.method} {tool_def.protocol}")
            if status.last_failure_reason:
                lines.append(f"    Last failure: {status.last_failure_reason}")

        return "\n".join(lines) if lines else "(no available MCP tools)"

    # ===================================================================
    # Main Loop
    # ===================================================================

    async def run_task(
        self,
        task_prompt: str,
        max_subtasks: int = 15,
    ) -> TaskResult:
        """Main task execution loop.

        1. Get page state on-demand → build text-only AXTree + DOM summary
        2. Build planning context (tree + website profile + MCP tools)
        3. Call Gemini Pro Preview for plan
        4. Handle planning_action (complete/branch/prune/continue)
        5. Route subtask (CU or MCP)
        6. Subtask reflection → update planning tree
        7. Loop until done or budget exhausted
        """
        self._task_prompt = task_prompt
        self._log("task_started", f"Task: {task_prompt[:80]}", detail={
            "task_prompt": task_prompt,
            "max_subtasks": max_subtasks,
            "mcp_tools": list(self._mcp_statuses.keys()),
        })

        self.planning_tree.create_root(task_prompt)
        # Branch immediately so the root is never completed/pruned directly
        # (completing root sets current_id=None since root has no parent).
        self.planning_tree.branch("Initial approach")
        total_actions = 0
        total_mcp_calls = 0
        subtasks_completed = 0

        for step in range(1, max_subtasks + 1):
            # 1. Get current page state (on-demand, with navigation retry)
            await self.session.wait_for_page_ready()
            for _retry in range(3):
                try:
                    axtree_raw = await self.session.get_raw_accessibility_tree()
                    current_url = self.session.page.url
                    current_title = await self.session.page.title()
                    break
                except Exception as e:
                    if "context was destroyed" in str(e).lower() and _retry < 2:
                        logger.warning("Orchestrator extraction: context destroyed (retry %d/2)", _retry + 1)
                        try:
                            await self.session.page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except Exception:
                            await asyncio.sleep(1)
                        continue
                    raise

            # 1b. Get interactive elements for visibility cross-check
            #     Orchestrator uses these to mark invisible elements in its AXTree
            #     so it doesn't plan subtasks around unreachable elements.
            visible_elements = await self.session.get_interactive_elements()
            # Apply same visibility filter as CU agent
            from morphnet.computer_use import filter_viewport_elements
            viewport_h = self.session.viewport_height
            visible_elements = filter_viewport_elements(visible_elements, viewport_h)

            # 2. Build orchestrator views
            axtree_view = build_orchestrator_representation(axtree_raw, visible_elements=visible_elements)
            dom_tree = await self.session.get_dom_tree(max_length=100_000)
            dom_summary = build_dom_summary(dom_tree)
            tree_context = self.planning_tree.get_context_for_planning()
            mcp_summary = self._get_available_mcp_summary()
            profile_summary = self._get_website_profile_summary()

            # 3. Determine urgency
            remaining = max_subtasks - step
            if remaining <= 0:
                urgency_hint = "FINAL ACTION — this is your last subtask."
            elif remaining <= 3:
                urgency_hint = f"LOW BUDGET — only {remaining} subtasks remaining."
            else:
                urgency_hint = ""

            # 4. Call planner
            plan = await self._call_planner(
                task_prompt, axtree_view, dom_summary,
                tree_context, mcp_summary, profile_summary,
                current_url, step, max_subtasks, urgency_hint,
            )

            # 4b. Save step representations for debugging
            self._save_plan_step(
                step=step,
                raw_axtree=axtree_raw,
                raw_elements_count=len(visible_elements) if visible_elements else 0,
                axtree_view=axtree_view,
                dom_summary=dom_summary,
                tree_context=tree_context,
                current_url=current_url,
                urgency_hint=urgency_hint,
                plan=plan,
            )

            # 5. Handle planning actions
            planning_action = plan.get("planning_action", "continue")

            match planning_action:
                case "complete_task":
                    final_answer = plan.get("final_answer", "")
                    self._log("task_completed", f"Task complete: {final_answer[:80]}", detail={
                        "final_answer": final_answer,
                        "steps_used": step,
                    }, outcome="success")

                    # Save insights
                    if plan.get("website_insights"):
                        self._website_insights.append(plan["website_insights"])

                    if self._website_insights:
                        await self._save_website_insights()

                    return TaskResult(
                        success=True,
                        final_answer=final_answer,
                        subtasks_completed=subtasks_completed,
                        planning_tree_summary=self.planning_tree.get_context_for_planning(),
                        website_insights=self._website_insights,
                        total_actions=total_actions,
                        total_mcp_calls=total_mcp_calls,
                    )

                case "branch":
                    intent = plan.get("branch_intent", "new approach")
                    self.planning_tree.branch(intent)
                    self._log("tree_branch", f"Branch: {intent[:60]}", detail={
                        "intent": intent,
                        "step": step,
                    })
                    continue

                case "prune":
                    reason = plan.get("prune_reason", "approach failed")
                    self.planning_tree.prune(reason)
                    self._log("tree_prune", f"Prune: {reason[:60]}", detail={
                        "reason": reason,
                        "step": step,
                    })
                    continue

            # 6. Execute subtask (planning_action == "continue")
            subtask = plan.get("next_subtask", "")
            routing = plan.get("routing", "computer_use")
            subtask_start_time = time.time()

            match routing:
                case "computer_use":
                    result = await self.cu_agent.execute_subtask(subtask)
                    total_actions += result.steps_used

                case "mcp":
                    tool_name = plan.get("mcp_tool_name", "")
                    mcp_ok = False
                    mcp_result = None  # Track for A/B learning
                    if tool_name and tool_name in self._mcp_statuses:
                        status = self._mcp_statuses[tool_name]
                        if status.status != "discarded":
                            try:
                                mcp_result = await self.mcp_manager.execute_tool(
                                    tool_name, subtask,
                                )
                                total_mcp_calls += 1
                            except Exception as exc:
                                logger.warning("MCP execute_tool crashed for '%s': %s", tool_name, exc)
                                mcp_result = {"success": False, "error": str(exc), "status_code": 0}

                            if mcp_result.get("success"):
                                # Reflect on MCP result — pass response template for structural check
                                tool_def = self.mcp_manager.get_tool(tool_name)
                                mcp_verdict = await self.reflector.reflect_on_mcp_call(
                                    tool_name=tool_name,
                                    tool_intent=subtask,
                                    http_status=mcp_result.get("status_code", 0),
                                    response_body=mcp_result.get("response_body"),
                                    response_template=(
                                        tool_def.response_template
                                        if tool_def else None
                                    ),
                                )

                                # Update lifecycle
                                update_mcp_status(
                                    status,
                                    mcp_verdict.get("success", False),
                                    mcp_verdict.get("failure_reason"),
                                )

                                # Build SubtaskResult for planning tree
                                mcp_action = ActionRecord(
                                    step=1,
                                    action={"action_type": "mcp_call", "tool": tool_name},
                                    reflection=mcp_verdict,
                                    one_line_summary=(
                                        f"Step 1: MCP {tool_name} → "
                                        f"{'success' if mcp_verdict.get('success') else 'failure'}: "
                                        f"{mcp_verdict.get('response_summary', '')[:60]}"
                                    ),
                                )
                                result = SubtaskResult(
                                    success=mcp_verdict.get("success", False),
                                    actions_taken=[mcp_action],
                                    subtask_reflection={
                                        "subtask_achieved": mcp_verdict.get("success", False),
                                        "outcome_summary": mcp_verdict.get("response_summary", ""),
                                        "reasoning": mcp_verdict.get("reasoning", ""),
                                        "recommendation": mcp_verdict.get("recommendation", "proceed_to_next_subtask"),
                                        "failure_analysis": mcp_verdict.get("failure_reason", ""),
                                        "extracted_data": mcp_verdict.get("response_summary"),
                                    },
                                    final_url=self.session.page.url if self.session.page else "",
                                    final_elements=[],
                                    extracted_data=mcp_verdict.get("response_summary"),
                                    notes=[],
                                    traffic_during_subtask=[],
                                    steps_used=1,
                                )
                                mcp_ok = True
                            else:
                                # MCP execution failed — degrade and fall back to CU
                                error_msg = mcp_result.get("error", "unknown error")
                                logger.warning("MCP '%s' failed: %s. Falling back to CU.", tool_name, error_msg[:100])
                                self._log("mcp_fallback_to_cu", f"MCP {tool_name} failed, falling back to CU", detail={
                                    "tool_name": tool_name,
                                    "error": error_msg[:200],
                                    "subtask": subtask[:100],
                                })
                                update_mcp_status(status, False, error_msg[:200])

                    if not mcp_ok:
                        # Fall back to CU: tool not found, discarded, or execution failed
                        subtask_start_for_cu = time.time()
                        result = await self.cu_agent.execute_subtask(subtask)
                        total_actions += result.steps_used

                        # A/B learning: compare MCP failure vs CU success
                        if result.success and tool_name and mcp_result:
                            try:
                                await self.mcp_manager.learn_from_cu_fallback(
                                    tool_name=tool_name,
                                    failed_mcp_result=mcp_result,
                                    cu_traffic_since=subtask_start_for_cu,
                                    subtask=subtask,
                                )
                            except Exception as exc:
                                logger.warning("A/B learning failed: %s", exc)

                case _:
                    logger.warning("Unknown routing '%s'. Using CU.", routing)
                    result = await self.cu_agent.execute_subtask(subtask)
                    total_actions += result.steps_used

            # 6b. Discover MCP tools from successful CU traffic
            if result.success and routing == "computer_use":
                try:
                    discovered = await self.mcp_manager.discover_tools_from_subtask(
                        traffic_since=subtask_start_time,
                        subtask_description=subtask,
                    )
                    for tool in discovered:
                        if tool.name not in self._mcp_statuses:
                            self._mcp_statuses[tool.name] = MCPToolStatus(
                                tool_name=tool.name,
                            )
                            self._log(
                                "tool_discovered",
                                f"MCP tool discovered: {tool.name}",
                                detail={
                                    "tool_name": tool.name,
                                    "endpoint": tool.endpoint_identity,
                                },
                            )
                except Exception as exc:
                    logger.warning("MCP discovery failed: %s", exc)

            # 7. Update planning tree based on reflection
            reflection = result.subtask_reflection
            recommendation = reflection.get("recommendation", "proceed_to_next_subtask")

            summary = BranchSummary(
                what_was_attempted=subtask,
                key_actions=[r.one_line_summary for r in result.actions_taken],
                outcome=reflection.get("outcome_summary", ""),
                reasoning=reflection.get("reasoning", ""),
                insights_gained=plan.get("website_insights", ""),
                data_collected=reflection.get("extracted_data"),
            )

            if reflection.get("subtask_achieved", False):
                self.planning_tree.complete_current(summary)
                # Branch to continue — next subtask will be planned
                self.planning_tree.branch(f"After: {subtask[:40]}")
                subtasks_completed += 1
            else:
                # Failed subtask — always prune and branch so the planner
                # sees the failure in tree context and plans differently.
                # AgentOccam: failed approaches must be pruned, not silently retried.
                failure_reason = reflection.get("failure_analysis", "subtask failed")
                match recommendation:
                    case "retry_same_subtask":
                        # Even "retry same" gets pruned — the planner sees the
                        # prune summary and can decide to retry or change approach
                        self.planning_tree.prune(
                            f"Failed (retry suggested): {failure_reason}",
                            summary,
                        )
                    case _:
                        self.planning_tree.prune(failure_reason, summary)

            # 8. Update MCP lifecycle if MCP was used
            if routing == "mcp" and plan.get("mcp_tool_name"):
                tool_name = plan["mcp_tool_name"]
                if tool_name in self._mcp_statuses:
                    update_mcp_status(
                        self._mcp_statuses[tool_name],
                        reflection.get("subtask_achieved", False),
                        reflection.get("mcp_failure_reason"),
                    )

            # 9. Save website insights
            if plan.get("website_insights"):
                self._website_insights.append(plan["website_insights"])
                await self._save_website_insights()

        # Budget exhausted
        self._log("task_budget_exhausted", f"Budget exhausted after {max_subtasks} subtasks", detail={
            "subtasks_completed": subtasks_completed,
            "total_actions": total_actions,
        }, outcome="failure")

        if self._website_insights:
            await self._save_website_insights()

        return TaskResult(
            success=False,
            final_answer=None,
            subtasks_completed=subtasks_completed,
            planning_tree_summary=self.planning_tree.get_context_for_planning(),
            website_insights=self._website_insights,
            total_actions=total_actions,
            total_mcp_calls=total_mcp_calls,
        )

    # ===================================================================
    # Planner Call
    # ===================================================================

    async def _call_planner(
        self,
        task_prompt: str,
        axtree_view: str,
        dom_summary: str,
        tree_context: str,
        mcp_summary: str,
        profile_summary: str,
        current_url: str,
        step: int,
        max_subtasks: int,
        urgency_hint: str,
    ) -> dict:
        """Call Gemini Pro Preview for planning decision, wrapped in trace.span()."""
        prompt = (
            f"Task: {task_prompt}\n\n"
            f"Current URL: {current_url}\n"
            f"Planning step {step} of {max_subtasks}\n"
            f"{urgency_hint}\n\n"
            f"Current Page (AXTree):\n{axtree_view[:6000]}\n\n"
            f"DOM Summary:\n{dom_summary[:2000]}\n\n"
            f"Planning Tree:\n{tree_context[:3000]}\n\n"
            f"MCP Tools:\n{mcp_summary}\n\n"
            f"Website Profile:\n{profile_summary}\n"
        )

        with self.trace.span("orchestrator", "plan_decision", f"Plan step {step}") as span:
            plan = call_gemini(
                model="gemini-3-flash-preview",
                contents=[prompt],
                response_schema=ORCHESTRATOR_SCHEMA,
                system_instruction=self._orchestrator_prompt,
                generation_config={"temperature": 0.4, "max_output_tokens": 4096},
                prompt_log_dir=self.trace.output_dir / "prompt_made",
            )
            span.set_reasoning(plan.get("reasoning", ""))
            span.set_confidence(plan.get("confidence", 0.0))
            for src in plan.get("evidence_sources", []):
                span.add_evidence(Evidence("model_output", src))
            span.set_detail("planning_action", plan.get("planning_action", ""))
            span.set_detail("routing", plan.get("routing", ""))
            span.set_detail("next_subtask", plan.get("next_subtask", "")[:80])
            span.set_outcome("success")

        self._log("plan_made", f"Step {step}: {plan.get('planning_action')} → {plan.get('next_subtask', '')[:50]}", detail={
            "step": step,
            "plan": plan,
        })
        return plan

    # ===================================================================
    # Step Data Logging
    # ===================================================================

    def _save_plan_step(
        self, *, step: int, raw_axtree: dict | None,
        raw_elements_count: int, axtree_view: str, dom_summary: str,
        tree_context: str, current_url: str, urgency_hint: str, plan: dict,
    ) -> None:
        """Save orchestrator planning step data for post-run analysis."""
        # Build the exact prompt that was sent to Gemini
        prompt = (
            f"Task: {self._task_prompt}\n\n"
            f"Current URL: {current_url}\n"
            f"Planning step {step}\n"
            f"{urgency_hint}\n\n"
            f"Current Page (AXTree):\n{axtree_view[:6000]}\n\n"
            f"DOM Summary:\n{dom_summary[:2000]}\n\n"
            f"Planning Tree:\n{tree_context[:3000]}\n"
        )
        # Count raw AXTree nodes
        def _count_nodes(node: dict | None) -> int:
            if not node:
                return 0
            count = 1
            for child in node.get("children", []):
                count += _count_nodes(child)
            return count

        data = {
            "type": "plan_step",
            "step": step,
            "url": current_url,
            "raw": {
                "axtree_node_count": _count_nodes(raw_axtree),
                "axtree_top_level": _summarize_axtree(raw_axtree, max_depth=2),
                "visible_elements_count": raw_elements_count,
            },
            "processed": {
                "text_only_axtree": axtree_view,
                "dom_summary": dom_summary,
                "tree_context": tree_context,
            },
            "prompt": prompt,
            "system_instruction_length": len(self._orchestrator_prompt),
            "response": plan,
        }
        self.trace.save_step(f"plan_{step:03d}", data)

    # ===================================================================
    # Website Profile
    # ===================================================================

    def _get_website_profile_summary(self) -> str:
        """Build website profile summary for orchestrator prompt."""
        profile = self.session.get_site_profile()
        if not profile and not self._website_insights:
            return "(no website profile)"

        parts: list[str] = []
        if profile:
            for key, value in profile.items():
                parts.append(f"  {key}: {value}")
        if self._website_insights:
            parts.append("  Recent insights:")
            for insight in self._website_insights[-5:]:
                parts.append(f"    - {insight}")
        return "\n".join(parts)

    async def _save_website_insights(self) -> None:
        """Persist accumulated website insights to sites/{site_name}/profile.json."""
        site_name = self.session.site_name
        if not site_name or not self._website_insights:
            return

        site_dir = SITES_DIR / site_name
        site_dir.mkdir(parents=True, exist_ok=True)

        profile_path = site_dir / "profile.json"
        profile: dict = {}
        if profile_path.exists():
            try:
                profile = json.loads(profile_path.read_text())
            except json.JSONDecodeError:
                pass

        existing_insights = profile.get("insights", [])
        existing_insights.extend(self._website_insights)
        profile["insights"] = existing_insights[-20:]
        profile["last_updated"] = time.time()
        profile["url"] = self.session.start_url

        profile_path.write_text(json.dumps(profile, indent=2))
        self._website_insights.clear()
