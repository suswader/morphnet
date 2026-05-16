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
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from morphnet.session_manager import (
    SessionManager, InteractiveElement, CapturedRequest,
    ActionResult, call_gemini_async,
)
from morphnet.reflector import Reflector
from morphnet.computer_use import ComputerUseAgent, SubtaskResult, ActionRecord
from morphnet.trace import TaskTrace, Evidence
from morphnet.representation import build_orchestrator_representation
from morphnet.manifest import (
    Graph, ExecutionResult, SubtaskObservation,
    find_candidates, list_graphs, purge_unverified_graphs,
    promote_graph, discard_graph,
)
from morphnet.observer import Observer
from morphnet.learner import Learner
from morphnet.executor import Executor

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).parent / "prompts"
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
            "enum": ["computer_use", "executor"],
            "description": "Route to CU (discovery) or existing learned graph tool.",
        },
        "graph_name": {
            "type": "string",
            "description": "Which learned graph tool (if routing='executor'). Empty for CU.",
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
        "task_success": {
            "type": "boolean",
            "description": (
                "If complete_task: did the task ACTUALLY succeed? "
                "False if you couldn't find the information, the page errored, "
                "or the result is incomplete/uncertain. True only if you are confident "
                "the answer fully addresses what was asked."
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

    def detect_repeated_approaches(self) -> str | None:
        """Scan pruned branches for repeated patterns.

        If 2+ pruned branches attempted similar things (based on summary text),
        return a warning string that gets injected into the planning prompt.
        This prevents the LLM from endlessly retrying the same failing strategy.
        """
        pruned_attempts: list[str] = []
        for node in self._nodes.values():
            if node.status == "pruned" and node.summary:
                pruned_attempts.append(node.summary.what_was_attempted.lower().strip())

        if len(pruned_attempts) < 2:
            return None

        # Find clusters of similar attempts via simple word overlap
        from collections import Counter
        clusters: list[list[str]] = []
        used = set()
        for i, a in enumerate(pruned_attempts):
            if i in used:
                continue
            words_a = set(a.split())
            cluster = [a]
            used.add(i)
            for j, b in enumerate(pruned_attempts):
                if j in used:
                    continue
                words_b = set(b.split())
                overlap = len(words_a & words_b)
                # >40% word overlap = same approach
                if overlap > 0.4 * max(len(words_a), len(words_b), 1):
                    cluster.append(b)
                    used.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        if not clusters:
            return None

        warnings = []
        for cluster in clusters:
            warnings.append(
                f"  - Tried {len(cluster)} times: \"{cluster[0][:80]}\" (and similar)"
            )
        return (
            "LOOP DETECTED — The following approaches have been tried multiple times and FAILED:\n"
            + "\n".join(warnings)
            + "\n\nYou MUST try a FUNDAMENTALLY DIFFERENT approach. Do NOT retry these strategies."
        )

    def to_mermaid(self) -> str:
        """Generate Mermaid graph visualization of the planning tree.

        Nodes colored: green=completed, red=pruned, blue=active.
        Edges labeled with transition type.
        """
        if not self._nodes or "plan_0" not in self._nodes:
            return "graph TD\n    empty[No planning tree]"

        lines = ["graph TD"]
        self._mermaid_walk("plan_0", lines)
        lines.append("    classDef success fill:#90EE90,stroke:#333")
        lines.append("    classDef failure fill:#FFB6C1,stroke:#333")
        lines.append("    classDef active fill:#87CEEB,stroke:#333")
        return "\n".join(lines)

    def _mermaid_walk(self, node_id: str, lines: list[str]) -> None:
        node = self._nodes[node_id]
        # Sanitize label for Mermaid (escape quotes)
        label = node.intent[:40].replace('"', "'").replace("\n", " ")
        safe_id = node_id.replace(".", "_")

        style = {
            "completed": ":::success",
            "pruned": ":::failure",
            "active": ":::active",
        }.get(node.status, "")
        lines.append(f'    {safe_id}["{label}"]{style}')

        for child_id in node.children:
            child = self._nodes.get(child_id)
            if not child:
                continue
            safe_child = child_id.replace(".", "_")
            edge_label = "prune" if child.status == "pruned" else "branch"
            lines.append(f'    {safe_id} -->|{edge_label}| {safe_child}')
            self._mermaid_walk(child_id, lines)

    def save_visualization(self, output_dir) -> None:
        """Save Mermaid visualization to trace output directory."""
        from pathlib import Path
        path = Path(output_dir) / "planning_tree.mermaid"
        path.write_text(self.to_mermaid())


# ---------------------------------------------------------------------------
# User Intent Extraction Schema (called by orchestrator for executor)
# ---------------------------------------------------------------------------

def _build_intent_schema(param_descriptors: list[dict]) -> dict:
    """Build a Gemini-compatible schema for per-node user_intent extraction.

    Each property is keyed by '{node_id}_{param_name}' to avoid deduplication
    when multiple nodes share the same param name (e.g., searchString on both
    source and destination auto-suggest).
    """
    properties = {}
    for pd in param_descriptors:
        key = pd["key"]
        desc = pd["node_description"] or ""
        example = pd.get("example_value", "")
        properties[key] = {
            "type": "string",
            "description": (
                f"Value for '{pd['param_name']}' in step: {desc}. "
                f"Example: {example}. Match the example format."
            ),
            "nullable": True,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }


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
    total_executor_calls: int


# ---------------------------------------------------------------------------
# MorphNetOrchestrator
# ---------------------------------------------------------------------------

class MorphNetOrchestrator:
    """Task planner and router.

    Receives a natural language task + start URL. Decomposes into subtasks.
    Routes each to CU or executor. Manages planning tree and graph lifecycle.

    Three-layer architecture:
    - Observer: always-on capture during CU sessions
    - Learner: post-subtask graph builder (background task)
    - Executor: deterministic graph runner (no LLM)
    """

    def __init__(self, session: SessionManager, trace: TaskTrace):
        self.session = session
        self.trace = trace
        self.reflector = Reflector(session, trace)

        # Observer captures CU actions + HTTP traffic
        self.observer = Observer(session)

        # CU agent with observer attached
        self.cu_agent = ComputerUseAgent(session, self.reflector, trace, observer=self.observer)

        # Learner builds graphs from observations (background)
        self.learner = Learner(session)

        # Executor runs learned graphs deterministically
        self.executor = Executor(session)

        # Background learner tasks (awaited before session cleanup)

        self.planning_tree = PlanningTree()
        self._website_insights: list[str] = []
        self._orchestrator_prompt = self._load_prompt("orchestrator_plan.txt")

        # State for response-aware orchestration
        self._last_executor_response: str | None = None

    @staticmethod
    def _load_prompt(filename: str) -> str:
        path = PROMPTS_DIR / filename
        if not path.exists():
            logger.warning("Prompt file not found: %s", path)
            return ""
        return path.read_text(encoding="utf-8")

    def _log(self, event_type: str, summary: str, **kwargs) -> str | None:
        return self.trace.log("orchestrator", event_type, summary, **kwargs)

    def _get_available_graphs_summary(self) -> str:
        """Build learned graph tools summary for orchestrator prompt.

        Shows graph name, capability, verified status, and per-node user_intent params
        so the planner can see that same-named params serve different roles.
        """
        site = self.session.site_name
        if not site:
            return "(no learned graph tools available)"

        graphs = list_graphs(site)
        if not graphs:
            return "(no learned graph tools available)"

        lines: list[str] = []
        for graph in graphs:
            stats = graph.execution_stats
            runs = stats.get("runs", 0)
            successes = stats.get("successes", 0)
            if graph.verified:
                marker = "[verified]"
            elif not graph.verification_only_read:
                marker = "[probationary]"
            else:
                marker = "[unverified]"
            lines.append(f"  {graph.name} {marker}")
            lines.append(f"    {graph.capability_statement}")
            if runs > 0:
                lines.append(f"    Stats: {successes}/{runs} successful executions")
            # Show per-node user_intent params with node descriptions
            intent_lines: list[str] = []
            for node in graph.nodes:
                node_intents = [p for p in node.core_parameters if p.role == "user_intent"]
                if node_intents:
                    desc = node.node_description or node.endpoint_fingerprint
                    param_names = ", ".join(p.name for p in node_intents)
                    intent_lines.append(f"      {node.id} ({desc}): {param_names}")
            if intent_lines:
                lines.append("    Steps requiring input:")
                lines.extend(intent_lines)

        return "\n".join(lines) if lines else "(no learned graph tools available)"

    def _find_graph(self, graph_name: str) -> Graph | None:
        """Look up a graph by name from the current site's graph store."""
        site = self.session.site_name
        if not site:
            return None
        graphs = list_graphs(site)
        return next((g for g in graphs if g.name == graph_name), None)

    def _build_response_summary(self, exec_result: ExecutionResult, graph: Graph) -> str:
        """Build text summary from the last node executed in the graph.

        The last node is the terminal result of the workflow (e.g., train
        search results). Intermediate and parallel nodes are irrelevant to
        the planner — only the final output matters.
        """
        if not exec_result.node_outputs:
            return ""

        # Find the last node that produced output (by graph execution order)
        last_node_id = None
        last_data = None
        for node_id, data in exec_result.node_outputs.items():
            if node_id.startswith("__"):
                continue
            last_node_id = node_id
            last_data = data

        if last_node_id is None or last_data is None:
            return ""

        node_lookup = {n.id: n for n in graph.nodes}
        node = node_lookup.get(last_node_id)
        desc = node.node_description or node.endpoint_fingerprint if node else last_node_id

        truncated = self._truncate_for_summary(last_data, max_items=20)
        data_str = json.dumps(truncated, indent=2, default=str)
        if len(data_str) > 5000:
            data_str = data_str[:5000] + "\n... (truncated)"

        return f"[{last_node_id}] {desc}:\n{data_str}"

    @staticmethod
    def _truncate_for_summary(data: Any, max_items: int = 20) -> Any:
        """Recursively truncate arrays and cap depth for context efficiency."""
        if isinstance(data, list):
            truncated = [MorphNetOrchestrator._truncate_for_summary(item, max_items) for item in data[:max_items]]
            if len(data) > max_items:
                truncated.append(f"... ({len(data) - max_items} more items)")
            return truncated
        if isinstance(data, dict):
            return {k: MorphNetOrchestrator._truncate_for_summary(v, max_items) for k, v in data.items()}
        return data

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
        })

        # Start observer for the entire task (accumulates across subtasks)
        site = self.session.site_name or "unknown"
        try:
            await self.observer.start_task(site, task_prompt)
        except Exception as exc:
            logger.warning("Observer start_task failed (non-fatal): %s", exc)

        self.planning_tree.create_root(task_prompt)
        # Branch immediately so the root is never completed/pruned directly
        # (completing root sets current_id=None since root has no parent).
        self.planning_tree.branch("Initial approach")

        total_actions = 0
        total_executor_calls = 0
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
                    err_msg = str(e).lower()
                    if ("context was destroyed" in err_msg
                            or "has been closed" in err_msg) and _retry < 2:
                        logger.warning("Orchestrator extraction: page error (retry %d/2): %s", _retry + 1, str(e)[:100])
                        try:
                            # Try to recover by creating a fresh page
                            self.session.page = await self.session._context.new_page()
                            await self.session.page.goto(
                                self.session.start_url,
                                wait_until="domcontentloaded",
                                timeout=15_000,
                            )
                            await self.session.wait_for_page_ready()
                        except Exception:
                            await asyncio.sleep(2)
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
            loop_warning = self.planning_tree.detect_repeated_approaches()
            if loop_warning:
                tree_context += "\n\n" + loop_warning
            mcp_summary = self._get_available_graphs_summary()
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

                    # Use the model's structured task_success field directly.
                    # The Gemini schema enforces this boolean — no text matching needed.
                    actual_success = plan.get("task_success", False)

                    self._log("task_completed", f"Task complete (success={actual_success}): {final_answer[:80]}", detail={
                        "final_answer": final_answer,
                        "steps_used": step,
                        "task_success": actual_success,
                        "confidence": plan.get("confidence", 0),
                    }, outcome="success" if actual_success else "failure")

                    # Save insights
                    if plan.get("website_insights"):
                        self._website_insights.append(plan["website_insights"])

                    if self._website_insights:
                        await self._save_website_insights()

                    # Save planning tree visualization
                    self.planning_tree.save_visualization(self.trace.output_dir)

                    # End observer → learner processes full task traffic
                    await self._end_task_and_learn()

                    return TaskResult(
                        success=actual_success,
                        final_answer=final_answer,
                        subtasks_completed=subtasks_completed,
                        planning_tree_summary=self.planning_tree.get_context_for_planning(),
                        website_insights=self._website_insights,
                        total_actions=total_actions,
                        total_executor_calls=total_executor_calls,
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
            subtask_id = f"subtask_{step}_{int(time.time())}"

            # 6a. Try executor first if routing='executor' or candidates exist
            executor_succeeded = False
            if routing == "executor":
                graph_name = plan.get("graph_name", "")
                exec_result = await self._try_executor(subtask, graph_name, subtask_id)
                graph = self._find_graph(graph_name)
                is_probationary = graph and not graph.verified and not graph.verification_only_read

                if exec_result and exec_result.status == "success":
                    executor_succeeded = True
                    total_executor_calls += 1

                    # Probationary write-op graph succeeded — promote to verified
                    if is_probationary and graph:
                        promote_graph(graph.site, graph.id)
                        self._log("graph_promoted", f"Probationary graph promoted: {graph_name}", detail={
                            "graph_id": graph.id[:12], "graph_name": graph_name,
                        })

                    # Build response summary from node_outputs
                    response_summary = self._build_response_summary(exec_result, graph) if graph else ""

                    # Store for next planning loop
                    self._last_executor_response = response_summary or None

                    result = SubtaskResult(
                        success=True,
                        actions_taken=[ActionRecord(
                            step=1,
                            action={"action_type": "executor_call", "graph": graph_name},
                            reflection={"verdict": {"success": True, "what_changed": "Graph executed successfully"}},
                            one_line_summary=f"Step 1: Executor {graph_name} → success",
                        )],
                        subtask_reflection={
                            "subtask_achieved": True,
                            "outcome_summary": f"Executor completed: {graph_name}",
                            "reasoning": "Graph executed deterministically",
                            "recommendation": "proceed_to_next_subtask",
                            "extracted_data": response_summary[:2000] if response_summary else None,
                        },
                        final_url=self.session.page.url if self.session.page else "",
                        final_elements=[],
                        extracted_data=response_summary[:2000] if response_summary else None,
                        notes=[],
                        traffic_during_subtask=[],
                        steps_used=1,
                    )
                    self._log("executor_success", f"Executor succeeded: {graph_name}", detail={
                        "graph_name": graph_name, "subtask": subtask[:100],
                        "response_summary_length": len(response_summary),
                        "response_summary_preview": response_summary[:200] if response_summary else None,
                    })
                elif exec_result:
                    # Executor failed
                    if is_probationary and graph:
                        # Probationary write-op graph failed — discard it
                        discard_graph(graph.site, graph.id)
                        self._log("graph_discarded", f"Probationary graph discarded: {graph_name}", detail={
                            "graph_id": graph.id[:12], "graph_name": graph_name,
                            "reason": exec_result.reason or exec_result.status,
                        })
                    self._log("executor_fallback_to_cu", f"Executor failed ({exec_result.status}), falling back to CU", detail={
                        "graph_name": graph_name, "status": exec_result.status,
                        "reason": exec_result.reason or "", "subtask": subtask[:100],
                    })

            # 6b. Fall back to CU if executor didn't succeed
            if not executor_succeeded:
                # Mark subtask boundary for observer (no reset — traffic accumulates)
                try:
                    await self.observer.start_subtask(subtask_id, self.session.site_name or "unknown", subtask)
                except Exception as exc:
                    logger.warning("Observer start_subtask failed (non-fatal): %s", exc)

                try:
                    result = await self.cu_agent.execute_subtask(subtask)
                except Exception as exc:
                    logger.error("CU execute_subtask crashed: %s", exc)
                    self._log("cu_crash", f"CU crashed: {exc}", detail={"subtask": subtask[:200], "error": str(exc)[:500]}, outcome="failure")
                    result = SubtaskResult(
                        success=False, actions_taken=[], subtask_reflection={
                            "subtask_achieved": False, "outcome_summary": f"CU agent crashed: {exc}",
                            "recommendation": "retry_different_approach", "failure_analysis": str(exc)[:300],
                        },
                        final_url=self.session.page.url if self.session.page else "",
                        final_elements=[], extracted_data=None, notes=[], traffic_during_subtask=[], steps_used=0,
                    )
                total_actions += result.steps_used

                # Mark subtask end (observer keeps accumulating)
                try:
                    verdict = "success" if result.success else "failure"
                    await self.observer.end_subtask(subtask_id, verdict)
                except Exception as exc:
                    logger.warning("Observer end_subtask failed (non-fatal): %s", exc)

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

            # 8. Save website insights
            if plan.get("website_insights"):
                self._website_insights.append(plan["website_insights"])
                await self._save_website_insights()

            # 9. Memory cleanup between subtasks
            try:
                await self.session.cleanup_between_subtasks()
            except Exception:
                pass

            # 10. Human-like pause between subtasks (bot detection mitigation)
            await asyncio.sleep(2.0 + random.random() * 3.0)

        # Budget exhausted
        self._log("task_budget_exhausted", f"Budget exhausted after {max_subtasks} subtasks", detail={
            "subtasks_completed": subtasks_completed,
            "total_actions": total_actions,
        }, outcome="failure")

        if self._website_insights:
            await self._save_website_insights()

        # Save planning tree visualization
        self.planning_tree.save_visualization(self.trace.output_dir)

        # End observer → learner processes full task traffic
        await self._end_task_and_learn()

        return TaskResult(
            success=False,
            final_answer=None,
            subtasks_completed=subtasks_completed,
            planning_tree_summary=self.planning_tree.get_context_for_planning(),
            website_insights=self._website_insights,
            total_actions=total_actions,
            total_executor_calls=total_executor_calls,
        )

    # ===================================================================
    # Executor + Learner Helpers
    # ===================================================================

    async def _try_executor(
        self,
        subtask: str,
        graph_name: str,
        subtask_id: str,
    ) -> ExecutionResult | None:
        """Try to execute a subtask using a learned graph.

        1. Find candidate graphs matching the subtask and current page.
        2. If graph_name specified, prefer that; otherwise take best candidate.
        3. Extract user_intent params via LLM.
        4. Call executor.execute().
        5. Return result or None if no candidates found.
        """
        site = self.session.site_name
        if not site:
            return None

        current_url = self.session.page.url if self.session.page else ""

        # Find candidate graphs
        candidates = find_candidates(site, subtask, current_url)
        if not candidates:
            logger.info("No candidate graphs for subtask: %s", subtask[:80])
            return None

        # Prefer the named graph if it exists in candidates
        graph = None
        if graph_name:
            graph = next((g for g in candidates if g.name == graph_name), None)
        if graph is None:
            # Take the first candidate (find_candidates returns them ranked)
            graph = candidates[0]

        # Extract user_intent parameters via LLM
        try:
            user_intent = await self._extract_user_intent(subtask, graph)
        except Exception as exc:
            logger.warning("Intent extraction failed: %s", exc)
            return None

        # Execute the graph
        try:
            result = await self.executor.execute(graph, user_intent, subtask_context=subtask)
            self._log("executor_attempt", f"Graph {graph.name}: {result.status}", detail={
                "graph_id": graph.id[:12],
                "graph_name": graph.name,
                "status": result.status,
                "subtask": subtask[:100],
            })
            return result
        except Exception as exc:
            logger.error("Executor crashed on graph %s: %s", graph.name, exc)
            return ExecutionResult(
                status="execution_error",
                reason=f"Executor crash: {str(exc)[:200]}",
            )

    async def _end_task_and_learn(self) -> None:
        """End observer, feed full task traffic to learner, purge unverified noise."""
        try:
            observation = await self.observer.end_task()
            if observation.http_requests:
                logger.info("Task ended: %d HTTP requests captured — running learner",
                            len(observation.http_requests))
                await self._run_learner_safe(observation)
            else:
                logger.info("Task ended: no HTTP traffic captured — skipping learner")
        except Exception as exc:
            logger.warning("end_task / learner failed: %s", exc)

        # Clean up: remove any unverified graphs (noisy CU retries, duplicates)
        site = self.session.site_name
        if site:
            removed = purge_unverified_graphs(site)
            if removed:
                logger.info("Purged %d unverified graph(s) for %s", removed, site)

    async def _run_learner_safe(self, observation: SubtaskObservation) -> None:
        """Background wrapper for learner — exceptions logged, never propagated."""
        try:
            graph = await self.learner.learn_from_subtask(observation)
            if graph:
                logger.info("Learner produced graph: %s (%s)", graph.name, graph.id[:12])
        except Exception:
            logger.exception("Learner failed for subtask %s", observation.subtask_id)

    async def _extract_user_intent(
        self,
        subtask: str,
        graph: Graph,
    ) -> dict[str, dict[str, Any]]:
        """Per-node user_intent extraction.

        Returns {node_id: {param_name: value}} so same-named params on different
        nodes (e.g., searchString on source vs destination auto-suggest) get
        distinct values.
        """
        # Build per-node descriptors
        param_descriptors: list[dict] = []
        for node in graph.nodes:
            for param in node.core_parameters:
                if param.role == "user_intent":
                    param_descriptors.append({
                        "key": f"{node.id}_{param.name}",
                        "node_id": node.id,
                        "param_name": param.name,
                        "node_description": node.node_description or node.cu_reasoning_sample[:100],
                        "example_value": str(param.value_example) if param.value_example else "",
                    })

        if not param_descriptors:
            return {}

        extraction_prompt = (
            f"Subtask: {subtask}\n\n"
            f"Graph capability: {graph.capability_statement}\n\n"
            f"Parameters to extract (each belongs to a specific workflow step):\n"
            + "\n".join(
                f"  - {pd['key']}: {pd['node_description']} → param '{pd['param_name']}' (example: {pd['example_value']})"
                for pd in param_descriptors
            )
        )

        schema = _build_intent_schema(param_descriptors)
        raw = await call_gemini_async(
            model="gemini-3-flash-preview",
            contents=[extraction_prompt],
            response_schema=schema,
            system_instruction=self._load_prompt("intent_extraction.txt"),
            generation_config={"temperature": 0.1, "max_output_tokens": 1024},
        )

        # Reshape flat {node_id_param: value} → {node_id: {param: value}}
        result: dict[str, dict[str, Any]] = {}
        for pd in param_descriptors:
            value = raw.get(pd["key"])
            if value is not None:
                result.setdefault(pd["node_id"], {})[pd["param_name"]] = value
        return result

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
            f"Learned Graph Tools:\n{mcp_summary}\n\n"
            f"Website Profile:\n{profile_summary}\n"
        )

        # Inject user credentials if available (for booking, form-filling, etc.)
        creds = self.session.get_credentials()
        if creds:
            user = creds.get("user", {})
            if user:
                prompt += f"\nUser Info (for forms/booking):\n"
                for k, v in user.items():
                    prompt += f"  {k}: {v}\n"

        # Inject executor response data if available from previous step
        if self._last_executor_response:
            prompt += (
                f"\n\nExecutor API Response Data:\n{self._last_executor_response[:5000]}\n\n"
                "The above data was returned by API calls the executor just made. "
                "If the task's answer is in this data, use complete_task and answer directly. "
                "Do NOT fall back to CU to 'read the page' when API data already has the answer."
            )
            self._last_executor_response = None

        with self.trace.span("orchestrator", "plan_decision", f"Plan step {step}") as span:
            plan = await call_gemini_async(
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
