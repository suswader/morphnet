"""
morphnet_v2/planner.py — Phase 3 brain (Chunk 3.1: data structures only).

Owns four pure-stdlib data structures:

1. PlanNode — one step in the planning tree (CU step or tool step).
2. PlanningTree — branch/prune memory across the lifetime of one task.
3. ToolEntry — lifecycle + success rate for one registered tool.
4. ToolRegistry — per-task registry the orchestrator hands to the planner LLM.

Chunk 3.1 contains NO LLM call, NO orchestrator loop. Both land in 3.2 and
3.3 in this same file.

Design notes (from May-12 review):

- PlanNode has no `intent` field. CU steps don't need one (CU reads the task
  + the tree text directly). Tool steps are identified by tool_id +
  tool_user_intent slots — the tool's name IS the descriptor.
- `status` is derived, not stored: a node is "active" iff it equals the
  tree's current_id, else "completed"/"pruned" based on outcome.
- Summaries are single strings emitted by the planner LLM in iteration N+1
  for the node that ran in iteration N. Action-level detail stays in
  StepResult on disk; the planner distills it down to one line for the tree.
- Tool lifecycle has FOUR internal states (verified / trusted / failing /
  discarded) but the planner only sees three — `discarded` tools are
  filtered out by `available_for_planner()`.
- Lifecycle transitions are deterministic — thresholds on success_rate +
  min-runs gate. No LLM judgment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from google.genai import types as genai_types
from jinja2 import Template

from morphnet_v2 import notes
from morphnet_v2.computer_use.page_agent import PageAgent
from morphnet_v2.computer_use.schemas import SessionExit, StepResult
from morphnet_v2.page_filter import PageFilter

if TYPE_CHECKING:
    from morphnet_v2.session_manager import SessionManager


# ─────────────────────────────────────────────────────────────────
# Planning tree — per-task branch/prune memory
# ─────────────────────────────────────────────────────────────────

Outcome = Literal["success", "failure"]
NodeKind = Literal["cu", "tool"]


@dataclass
class PlanNode:
    """One step in the planning tree.

    For CU steps: kind="cu", tool_id/tool_user_intent both None.
    For tool steps: kind="tool", tool_id + tool_user_intent populated by the
    orchestrator at branch time (from the planner's `invoke_tool` decision).
    `summary` and `outcome` get written by the orchestrator in the NEXT
    iteration, from the planner's `tree_update` for this step.
    """
    
    node_id: str
    parent_id: str | None
    kind: NodeKind
    tool_id: str | None = None
    tool_user_intent: dict | None = None
    summary: str | None = None
    outcome: Outcome | None = None
    children: list[str] = field(default_factory=list)


class PlanningTree:
    """Branch/prune memory for one task. Standalone — orchestrator mutates
    it via `create_root`, `branch`, `complete_current`, `prune`; planner LLM
    reads via `get_context_for_planning()`.

    Root invariant: there is always exactly one root node ("plan_0") created
    by `create_root(task)`. The root never carries kind/tool_id — it is a
    sentinel parent for the first real step. The user task is stored on the
    tree itself (rendered at the top of `get_context_for_planning`).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, PlanNode] = {}
        self._current_id: str | None = None
        self._next_child_idx: dict[str, int] = {}
        self._task: str | None = None
        # Result fields populated by the orchestrator at task termination.
        # The tree is the single source of truth for what happened on this
        # task — there is no separate TaskResult dataclass. Caller reads
        # task_exit, final_answer, totals + the derived properties below.
        self.task_exit: "TaskExit | None" = None
        self.final_answer: str | None = None
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._final_url: str | None = None

    # ---- mutators -----------------------------------------------------

    def create_root(self, task: str) -> str:
        """Create the root node. Stores the user task for rendering. Idempotent
        only on first call — calling twice on the same tree raises."""
        if self._nodes:
            raise RuntimeError("PlanningTree already has a root — create a fresh tree per task")
        root_id = "plan_0"
        self._nodes[root_id] = PlanNode(
            node_id=root_id, parent_id=None, kind="cu",  # kind unused for root
        )
        self._current_id = root_id
        self._next_child_idx[root_id] = 1
        self._task = task
        return root_id

    def branch(
        self,
        *,
        kind: NodeKind,
        tool_id: str | None = None,
        tool_user_intent: dict | None = None,
    ) -> str:
        """Create a child under the current node and make it current. Returns
        new node_id. kind="tool" requires tool_id."""
        assert self._current_id is not None, "Call create_root before branch"
        if kind == "tool" and tool_id is None:
            raise ValueError("kind='tool' requires tool_id")
        parent_id = self._current_id
        idx = self._next_child_idx.get(parent_id, 1)
        new_id = f"{parent_id}_{idx}"
        self._next_child_idx[parent_id] = idx + 1

        self._nodes[new_id] = PlanNode(
            node_id=new_id,
            parent_id=parent_id,
            kind=kind,
            tool_id=tool_id,
            tool_user_intent=tool_user_intent,
        )
        self._nodes[parent_id].children.append(new_id)
        self._next_child_idx[new_id] = 1
        self._current_id = new_id
        return new_id

    def complete_current(self, summary: str) -> str | None:
        """Mark current node outcome=success, write summary, move current to
        parent. Returns the new current_id (parent), or None if root completed."""
        return self._close_current("success", summary)

    def prune(self, summary: str) -> str | None:
        """Mark current node outcome=failure, write summary, move current to
        parent. Returns the new current_id (parent), or None if root pruned."""
        return self._close_current("failure", summary)

    def _close_current(self, outcome: Outcome, summary: str) -> str | None:
        assert self._current_id is not None, "No active node to close"
        node = self._nodes[self._current_id]
        node.outcome = outcome
        node.summary = summary
        self._current_id = node.parent_id
        return self._current_id

    # ---- accessors ----------------------------------------------------

    @property
    def current_node(self) -> PlanNode | None:
        if self._current_id is None:
            return None
        return self._nodes.get(self._current_id)

    @property
    def task(self) -> str | None:
        return self._task

    # ---- task-result accessors (populated by orchestrator at termination) ----

    @property
    def success(self) -> bool:
        """True iff task_exit == 'completed'. Derived — the orchestrator sets
        task_exit; success follows from it."""
        return self.task_exit == "completed"

    @property
    def step_count(self) -> int:
        """Number of branched steps under root. Derived from tree shape."""
        root = self._nodes.get("plan_0")
        return len(root.children) if root else 0

    @property
    def final_url(self) -> str | None:
        """Where execution ended up. Stored by orchestrator at termination."""
        return self._final_url

    def set_final_url(self, url: str) -> None:
        self._final_url = url

    def journey(self) -> list[str]:
        """One human-readable line per step under root, in order. Used by
        callers (run_eval, the CLI) for the post-task summary. Each entry:
            '<kind>(<status>): <planner-emitted summary>'
        """
        lines: list[str] = []
        root = self._nodes.get("plan_0")
        if root is None:
            return lines
        for child_id in root.children:
            n = self._nodes[child_id]
            kind_tag = "CU" if n.kind == "cu" else f"Tool({n.tool_id})"
            outcome_tag = n.outcome or "active"
            line = f"{kind_tag} ({outcome_tag}): {n.summary or '(no summary)'}"
            lines.append(line)
        return lines

    def status_of(self, node_id: str) -> str:
        node = self._nodes.get(node_id)
        if node is None:
            return "unknown"
        if node_id == self._current_id:
            return "active"
        if node.outcome == "success":
            return "completed"
        if node.outcome == "failure":
            return "pruned"
        return "active"  # root before any close

    # ---- rendering ----------------------------------------------------

    def get_context_for_planning(self) -> str:
        """Renders task + tree as text for the planner LLM. The current node
        is marked `← CURRENT FOCUS`. Per-node displays: kind, tool_id (if any),
        outcome, summary."""
        if not self._nodes:
            return "(empty planning tree)"
        lines: list[str] = []
        if self._task:
            lines.append(f"# Task: {self._task}")
            lines.append("")
        lines.append("# Planning Tree:")
        self._render_node("plan_0", 0, lines)
        return "\n".join(lines)

    def _render_node(self, node_id: str, depth: int, lines: list[str]) -> None:
        node = self._nodes[node_id]
        indent = "│   " * depth
        prefix = "├── " if depth > 0 else ""
        status = self.status_of(node_id)
        marker = "← CURRENT FOCUS" if status == "active" and node_id == self._current_id else f"[{status}]"

        # Identify the node — root has no kind label, children show kind + tool_id.
        if node.parent_id is None:
            head = f"{indent}{prefix}{node_id} (root) {marker}"
        elif node.kind == "tool":
            tool_part = f" tool={node.tool_id}"
            intent_part = f" intent={node.tool_user_intent}" if node.tool_user_intent else ""
            head = f"{indent}{prefix}{node_id} kind=tool{tool_part}{intent_part} {marker}"
        else:
            head = f"{indent}{prefix}{node_id} kind=cu {marker}"
        lines.append(head)

        if node.summary:
            outcome_label = "✓" if node.outcome == "success" else "✗" if node.outcome == "failure" else "·"
            pad = indent + "│   "
            lines.append(f"{pad}{outcome_label} {node.summary}")

        for child_id in node.children:
            self._render_node(child_id, depth + 1, lines)

    def detect_repeated_approaches(self) -> str | None:
        """Scan PRUNED nodes for similar summaries (>40% word overlap clusters
        of 2+). Lifted from morphnet — used as a planner-prompt warning when
        the LLM keeps trying the same failing approach.
        """
        pruned: list[str] = [
            n.summary.lower().strip()
            for n in self._nodes.values()
            if n.outcome == "failure" and n.summary
        ]
        if len(pruned) < 2:
            return None

        clusters: list[list[str]] = []
        used: set[int] = set()
        for i, a in enumerate(pruned):
            if i in used:
                continue
            words_a = set(a.split())
            cluster = [a]
            used.add(i)
            for j, b in enumerate(pruned):
                if j in used:
                    continue
                words_b = set(b.split())
                if not (words_a or words_b):
                    continue
                overlap = len(words_a & words_b)
                if overlap > 0.4 * max(len(words_a), len(words_b), 1):
                    cluster.append(b)
                    used.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        if not clusters:
            return None

        warnings = [
            f"  - Tried {len(c)} times: \"{c[0][:80]}\" (and similar)"
            for c in clusters
        ]
        return (
            "LOOP DETECTED — The following approaches have been tried multiple "
            "times and FAILED:\n"
            + "\n".join(warnings)
            + "\n\nYou MUST try a FUNDAMENTALLY DIFFERENT approach. Do NOT retry "
            "these strategies."
        )

    def to_mermaid(self) -> str:
        """Mermaid graph. Nodes colored: green=success, red=failure, blue=active."""
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
        safe_id = node_id.replace(".", "_")
        if node.parent_id is None:
            label = "root"
        elif node.kind == "tool":
            label = f"tool: {node.tool_id}"
        else:
            label = "cu"
        if node.summary:
            label += f"\\n{node.summary[:50]}"
        status = self.status_of(node_id)
        style = {"completed": ":::success", "pruned": ":::failure", "active": ":::active"}.get(status, "")
        safe_label = label.replace('"', "'")
        lines.append(f'    {safe_id}["{safe_label}"]{style}')
        for child_id in node.children:
            safe_child = child_id.replace(".", "_")
            child_status = self.status_of(child_id)
            edge_label = "prune" if child_status == "pruned" else "branch"
            lines.append(f'    {safe_id} -->|{edge_label}| {safe_child}')
            self._mermaid_walk(child_id, lines)

    def save_visualization(self) -> None:
        """Persist Mermaid graph via notes (data_type='planning_tree' →
        planning/{ts}.mermaid). No-op if notes isn't attached to a results dir."""
        notes.log(data_type="planning_tree", data=self.to_mermaid())


# ─────────────────────────────────────────────────────────────────
# Tool registry — per-task tool lifecycle
# ─────────────────────────────────────────────────────────────────

ToolLifecycle = Literal["verified", "trusted", "failing", "discarded"]
SlotType = Literal["string", "number", "integer", "boolean", "array", "object"]


@dataclass
class SlotDef:
    """One parameter of a tool — what the planner LLM must produce when it
    invokes this tool. Phase 4's tool_builder populates these from the
    captured request/response graph; Phase 3 leaves them empty.
    """

    name: str
    type: SlotType
    required: bool = True
    description: str = ""
    examples: list[str] = field(default_factory=list)  # values from past successful runs


@dataclass
class ToolEntry:
    """One registered tool. tool_id doubles as the human-readable name —
    capability_statement is the 1-sentence description the planner reads to
    decide whether to invoke this tool. slots describe what params the
    planner must produce; they get translated into Gemini function-call
    parameter schemas at planner-call time."""

    tool_id: str
    capability_statement: str
    slots: list[SlotDef] = field(default_factory=list)
    lifecycle: ToolLifecycle = "verified"
    success_rate: float = 0.0    # 0.0–1.0
    total_runs: int = 0          # divisor + min-runs gate

    @property
    def success_count(self) -> int:
        return round(self.success_rate * self.total_runs)


class ToolRegistry:
    """Per-task tool lifecycle. Phase 3 stub — the orchestrator builds one
    empty registry per task; Phase 4 tool_builder will populate it; Phase 5
    tool_executor will read from it. Lifecycle transitions are deterministic
    threshold rules (no LLM judgment).
    """

    # Tunable thresholds — keep them in one place.
    _MIN_RUNS_BEFORE_DOWNGRADE = 3
    _PROMOTE_THRESHOLD = 0.8      # success_rate ≥ → "trusted"
    _FAILING_THRESHOLD = 0.5      # success_rate < → "failing"
    _DISCARD_THRESHOLD = 0.2      # success_rate < → "discarded"

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool: ToolEntry) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"tool_id {tool.tool_id!r} already registered")
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> ToolEntry | None:
        return self._tools.get(tool_id)

    def record_success(self, tool_id: str) -> None:
        tool = self._require(tool_id)
        new_count = tool.success_count + 1
        tool.total_runs += 1
        tool.success_rate = new_count / tool.total_runs
        self._maybe_transition(tool)

    def record_failure(self, tool_id: str, reason: str) -> None:
        # reason is not stored here — it's already on the corresponding PlanNode
        # in the tree. We just bump counters.
        tool = self._require(tool_id)
        new_count = tool.success_count
        tool.total_runs += 1
        tool.success_rate = new_count / tool.total_runs
        self._maybe_transition(tool)

    def _require(self, tool_id: str) -> ToolEntry:
        tool = self._tools.get(tool_id)
        if tool is None:
            raise KeyError(f"tool_id {tool_id!r} not registered")
        return tool

    def _maybe_transition(self, tool: ToolEntry) -> None:
        """Apply deterministic lifecycle rules after every run."""
        if tool.total_runs < self._MIN_RUNS_BEFORE_DOWNGRADE:
            return
        if tool.success_rate < self._DISCARD_THRESHOLD:
            tool.lifecycle = "discarded"
        elif tool.success_rate < self._FAILING_THRESHOLD:
            tool.lifecycle = "failing"
        elif tool.success_rate >= self._PROMOTE_THRESHOLD:
            tool.lifecycle = "trusted"
        # else keep current lifecycle (could be "verified" still on first OK run)

    def available_for_planner(self) -> list[ToolEntry]:
        """All tools EXCEPT `discarded`. Order: trusted → verified → failing
        (so the planner sees its best options first)."""
        order = {"trusted": 0, "verified": 1, "failing": 2}
        return sorted(
            (t for t in self._tools.values() if t.lifecycle != "discarded"),
            key=lambda t: (order.get(t.lifecycle, 99), t.tool_id),
        )

    def format_for_planner(self) -> str:
        """One line per tool: id, capability, lifecycle, success_rate. Empty
        string when nothing's registered or everything's discarded."""
        tools = self.available_for_planner()
        if not tools:
            return "(no tools available)"
        lines = ["# Available tools:"]
        for t in tools:
            rate_pct = round(t.success_rate * 100)
            lines.append(
                f"- {t.tool_id} [{t.lifecycle}, success={rate_pct}% over "
                f"{t.total_runs} runs] — {t.capability_statement}"
            )
        return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════
# Chunk 3.2 — Planner LLM call + function-declaration generator
# ═════════════════════════════════════════════════════════════════

Trigger = Literal["task_start", "cu_returned", "tool_returned"]
PlanningAction = Literal["continue_cu", "invoke_tool", "complete_task", "give_up"]

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "planner.j2"
_template_cache: Template | None = None


def _load_template() -> Template:
    """Lazy-load the planner Jinja template once per process."""
    global _template_cache
    if _template_cache is None:
        _template_cache = Template(
            _PROMPT_PATH.read_text(encoding="utf-8"),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _template_cache


# Field names every planner function carries. Used by `_parse_planner_response`
# to separate common-fields-vs-tool-slot-fields when an invoke_<tool> fires.
_COMMON_FIELDS = frozenset({
    "tree_update_outcome",
    "tree_update_summary",
    "reasoning",
    "confidence",
    "evidence_sources",
    "final_answer",  # complete_task only
})

_SLOT_TYPE_TO_GENAI: dict[str, "genai_types.Type"] = {
    "string":  genai_types.Type.STRING,
    "number":  genai_types.Type.NUMBER,
    "integer": genai_types.Type.INTEGER,
    "boolean": genai_types.Type.BOOLEAN,
    "array":   genai_types.Type.ARRAY,
    "object":  genai_types.Type.OBJECT,
}


@dataclass
class PlannerDecision:
    """Typed extract from the planner's function_call response. Orchestrator
    consumes this — applies tree_update first, then dispatches per planning_action.
    """

    planning_action: PlanningAction
    tool_id: str | None = None
    tool_user_intent: dict | None = None      # the chosen tool's slot values
    final_answer: str | None = None
    tree_update_outcome: Outcome | None = None
    tree_update_summary: str | None = None
    reasoning: str = ""
    confidence: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    # Token usage for this planner call — orchestrator accumulates into tree.
    input_tokens: int = 0
    output_tokens: int = 0


def _common_planner_props() -> dict[str, genai_types.Schema]:
    """The 5 fields every planner function carries: tree_update_outcome,
    tree_update_summary, reasoning, confidence, evidence_sources. Extracted as
    a helper because every function declaration (continue_cu, complete_task,
    give_up, plus one per tool) embeds this same block. The 'required' list
    is returned separately by the caller — these properties are nullable for
    the tree_update ones (which can be null on task_start).
    """
    T = genai_types.Type
    return {
        "tree_update_outcome": genai_types.Schema(
            type=T.STRING,
            enum=["success", "failure"],
            nullable=True,
            description=(
                "Outcome of the just-ended step. Null only on task_start trigger; "
                "otherwise must be set. Start from the mechanical signals shown in "
                "the prompt; override only if context contradicts them."
            ),
        ),
        "tree_update_summary": genai_types.Schema(
            type=T.STRING,
            nullable=True,
            description=(
                "1–3 sentence distillation of what the just-ended step accomplished "
                "or why it failed. INCLUDE every task-relevant data point that "
                "surfaced (items, prices, IDs, error reasons, page transitions). "
                "This is the only persistent memory across planner turns. "
                "Null only on task_start."
            ),
        ),
        "reasoning": genai_types.Schema(
            type=T.STRING,
            description="Terse explanation of why this planning_action was chosen.",
        ),
        "confidence": genai_types.Schema(
            type=T.NUMBER,
            description="Self-rated confidence 0.0–1.0 in this decision.",
        ),
        "evidence_sources": genai_types.Schema(
            type=T.ARRAY,
            items=genai_types.Schema(type=T.STRING),
            description=(
                "Concrete evidence supporting the decision — URLs, element IDs, "
                "tool response fields. One short string per source."
            ),
        ),
    }


_COMMON_REQUIRED = ["reasoning", "confidence", "evidence_sources"]


def build_planner_function_declarations(
    registry: ToolRegistry,
) -> list[genai_types.FunctionDeclaration]:
    """Build the function declarations Gemini sees this turn.

    Always includes the three static actions (continue_cu, complete_task,
    give_up). For each tool in the registry that's not `discarded`, adds one
    `invoke_<tool_id>` function whose params = the tool's slots + common
    planner fields. The planner picks exactly ONE function — that choice IS
    the planning_action.
    """
    T = genai_types.Type
    common_props = _common_planner_props()

    decls: list[genai_types.FunctionDeclaration] = []

    # Static action: continue_cu
    decls.append(genai_types.FunctionDeclaration(
        name="continue_cu",
        description=(
            "Hand the current page state back to the CU (browser) agent so it "
            "can continue interacting. Use when no registered tool fits the "
            "next required action, or when CU needs to gather more info."
        ),
        parameters=genai_types.Schema(
            type=T.OBJECT,
            properties=dict(common_props),
            required=list(_COMMON_REQUIRED),
        ),
    ))

    # Static action: complete_task
    complete_props = dict(common_props)
    complete_props["final_answer"] = genai_types.Schema(
        type=T.STRING,
        description=(
            "The final answer to the original task, synthesized from prior "
            "step summaries. Self-contained — the user reads only this."
        ),
    )
    decls.append(genai_types.FunctionDeclaration(
        name="complete_task",
        description=(
            "Prior step summaries contain enough information to answer the "
            "task. Provide final_answer."
        ),
        parameters=genai_types.Schema(
            type=T.OBJECT,
            properties=complete_props,
            required=list(_COMMON_REQUIRED) + ["final_answer"],
        ),
    ))

    # Static action: give_up
    decls.append(genai_types.FunctionDeclaration(
        name="give_up",
        description=(
            "No remaining viable approach. The reasoning field must explain "
            "why and what was tried."
        ),
        parameters=genai_types.Schema(
            type=T.OBJECT,
            properties=dict(common_props),
            required=list(_COMMON_REQUIRED),
        ),
    ))

    # Dynamic per-tool functions
    for tool in registry.available_for_planner():
        tool_props = dict(common_props)
        tool_required: list[str] = list(_COMMON_REQUIRED)
        for slot in tool.slots:
            ex_part = ""
            if slot.examples:
                ex_part = " Past values that worked: " + ", ".join(
                    f"'{e}'" for e in slot.examples[:5]
                )
            tool_props[slot.name] = genai_types.Schema(
                type=_SLOT_TYPE_TO_GENAI.get(slot.type, T.STRING),
                description=(slot.description or "") + ex_part,
            )
            if slot.required:
                tool_required.append(slot.name)
        decls.append(genai_types.FunctionDeclaration(
            name=f"invoke_{tool.tool_id}",
            description=tool.capability_statement,
            parameters=genai_types.Schema(
                type=T.OBJECT,
                properties=tool_props,
                required=tool_required,
            ),
        ))

    return decls


def _parse_planner_response(resp: Any) -> PlannerDecision:
    """Extract the single function_call from Gemini's response, map it to a
    PlannerDecision. Raises RuntimeError if no function_call was emitted (a
    text-only response is a planner protocol violation — orchestrator will
    surface this as an error)."""
    cands = resp.candidates or []
    if not cands:
        raise RuntimeError("Planner response had no candidates")
    content = cands[0].content
    parts = (content.parts if content is not None else None) or []
    fn_call = None
    for p in parts:
        if p.function_call is not None:
            fn_call = p.function_call
            break
    if fn_call is None:
        raise RuntimeError("Planner emitted no function_call (text-only response)")

    name = fn_call.name or ""
    args: dict = dict(fn_call.args or {})

    # Common fields (every function declares them)
    tu_outcome = args.get("tree_update_outcome")
    tu_summary = args.get("tree_update_summary")
    reasoning = args.get("reasoning", "")
    confidence = float(args.get("confidence", 0.0))
    evidence_sources = list(args.get("evidence_sources") or [])

    # Action-specific dispatch
    if name == "continue_cu":
        action: PlanningAction = "continue_cu"
        tool_id, tool_user_intent, final_answer = None, None, None
    elif name == "complete_task":
        action = "complete_task"
        tool_id, tool_user_intent = None, None
        final_answer = args.get("final_answer")
    elif name == "give_up":
        action = "give_up"
        tool_id, tool_user_intent, final_answer = None, None, None
    elif name.startswith("invoke_"):
        action = "invoke_tool"
        tool_id = name[len("invoke_"):]
        tool_user_intent = {k: v for k, v in args.items() if k not in _COMMON_FIELDS}
        final_answer = None
    else:
        raise RuntimeError(f"Planner emitted unknown function: {name!r}")

    # Token usage — Gemini's GenerateContentResponse always carries
    # usage_metadata, but the field can be None on some error paths.
    usage = resp.usage_metadata
    in_tok = int(usage.prompt_token_count or 0) if usage is not None else 0
    out_tok = int(usage.candidates_token_count or 0) if usage is not None else 0

    return PlannerDecision(
        planning_action=action,
        tool_id=tool_id,
        tool_user_intent=tool_user_intent,
        final_answer=final_answer,
        tree_update_outcome=tu_outcome,
        tree_update_summary=tu_summary,
        reasoning=reasoning,
        confidence=confidence,
        evidence_sources=evidence_sources,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def render_planner_prompt(
    *,
    task: str,
    tree: PlanningTree,
    registry: ToolRegistry,
    trigger: Trigger,
    browser_state: dict[str, Any],
) -> str:
    """Render the planner Jinja template. browser_state shape depends on trigger:

    task_start:    {"url": str, "v5": str}
    cu_returned:   {"url": str, "v5": str, "cu_success": bool, "cu_exit_reason": str,
                    "cu_total_actions": int, "cu_failed_actions": int,
                    "cu_report_message": str | None, "cu_action_history": list[str]}
    tool_returned: {"url": str, "tool_id": str, "tool_user_intent": dict,
                    "tool_http_status": int, "tool_success": bool,
                    "tool_error": str | None, "tool_response_digest": str}
    """
    template = _load_template()
    # Pre-format steps list for the template — keeps Jinja simple.
    steps: list[dict[str, Any]] = []
    root_id = "plan_0"
    root = tree._nodes.get(root_id)
    if root is not None:
        for i, child_id in enumerate(root.children, 1):
            node = tree._nodes[child_id]
            status_label = {
                "success": "✓ completed",
                "failure": "✗ pruned",
            }.get(node.outcome or "", "in progress — awaiting summary")
            steps.append({
                "number": i,
                "kind_label": "CU" if node.kind == "cu" else "Tool",
                "tool_id": node.tool_id,
                "tool_user_intent": node.tool_user_intent,
                "status_label": status_label,
                "summary": node.summary,
            })
    return template.render(
        task=task,
        steps=steps,
        tools=registry.available_for_planner(),
        trigger=trigger,
        **browser_state,
    )


async def call_planner(
    sm: "SessionManager",
    *,
    task: str,
    tree: PlanningTree,
    registry: ToolRegistry,
    trigger: Trigger,
    browser_state: dict[str, Any],
    model: str = "gemini-3-flash-preview",
    thinking_budget: int = 2048,
    max_output_tokens: int = 8192,
    temperature: float = 0.4,
) -> PlannerDecision:
    """Single planner turn. Builds the prompt + function declarations,
    calls Gemini in tools-mode, parses the function_call into a typed
    PlannerDecision.

    The orchestrator (Chunk 3.3) consumes the decision: apply tree_update
    to the just-ended PlanNode first, then dispatch per planning_action.
    """
    prompt = render_planner_prompt(
        task=task,
        tree=tree,
        registry=registry,
        trigger=trigger,
        browser_state=browser_state,
    )
    declarations = build_planner_function_declarations(registry)
    tools = [genai_types.Tool(function_declarations=declarations)]

    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=prompt)],
        ),
    ]
    resp = await sm.call_gemini(
        model=model,
        contents=contents,
        tools=tools,
        thinking_budget=thinking_budget,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return _parse_planner_response(resp)


# ═════════════════════════════════════════════════════════════════
# Chunk 3.3 — Orchestrator (trigger-driven routing loop)
# ═════════════════════════════════════════════════════════════════

# Where the task ended up. The orchestrator sets `tree.task_exit` to one of
# these values before returning. `PlanningTree.success` is derived from this.
TaskExit = Literal["completed", "give_up", "max_steps", "infrastructure_error"]


class Orchestrator:
    """Trigger-driven routing loop. The production planner for Phase 3+
    (the Phase-2 pass-through stub has been retired).

    `SessionManager.run_task(task)` lazily builds one Orchestrator instance
    and forwards the call. Returns the `PlanningTree` (which carries
    task_exit, final_answer, totals, plus the full per-step journey).
    No separate TaskResult dataclass.

    Phase 3: `tool_executor` is None; if the planner picks `invoke_tool` (it
    shouldn't, since no `invoke_<tool>` declarations are emitted when the
    registry is empty), we synthesize an HTTP 503 failure so the next planner
    turn can recover. Phase 5 replaces that defensive path with a real call
    to `tool_executor.replay(...)`.

    Reflector is intentionally absent (architecture rule 11): step outcome
    comes from the planner LLM's `tree_update.outcome`, which the planner
    derives from deterministic mechanical signals (`StepResult.success` and
    HTTP status) shown in the prompt. No second-guessing in code.
    """

    def __init__(
        self,
        *,
        sm: "SessionManager",
        max_steps: int = 10,
        max_turns_per_step: int = 60,
    ) -> None:
        self._sm = sm
        self._max_steps = max_steps
        self._max_turns_per_step = max_turns_per_step
        # Sub-components built lazily on first run_task call. The Orchestrator
        # OWNS these — SessionManager doesn't construct or pass them in.
        self._page_agent: PageAgent | None = None
        self._tool_executor: Any | None = None
        # Phase 5: seed the ToolRegistry from sites/{site}/tools.json if present.
        # The planner's existing build_planner_function_declarations() will then
        # surface every kept tool as an invoke_<tool_id> function declaration.
        self._registry = ToolRegistry()
        self._seed_registry_from_site()

    def _seed_registry_from_site(self) -> None:
        """If `morphnet_v2/sites/{site}/tools.json` exists, load its tools
        into the registry. Builds the lazy ToolExecutor on first invoke."""
        try:
            from morphnet_v2.tool_executor import load_tools_for_site
        except ImportError:
            return
        site = getattr(self._sm, "site_name", None)
        if not site:
            return
        try:
            tools = load_tools_for_site(site)
        except Exception:
            return
        for tool_id, cand in tools.items():
            slot_defs: list[SlotDef] = []
            for name, slot in cand.slots.items():
                # Per-slot description (from LLM) and observed examples
                desc = cand.slot_descriptions.get(name, "")
                examples = []
                if slot.captured_examples:
                    examples = list(slot.captured_examples[:5])
                elif slot.observed_values:
                    examples = list(slot.observed_values[:5])
                slot_defs.append(SlotDef(
                    name=name,
                    type="string",
                    required=getattr(slot, "required", True),
                    description=desc or f"{name} (auto)",
                    examples=examples,
                ))
            entry = ToolEntry(
                tool_id=tool_id,
                capability_statement=cand.capability_statement or "(no description)",
                slots=slot_defs,
            )
            try:
                self._registry.register(entry)
            except Exception:
                pass

    def _ensure_page_agent(self) -> PageAgent:
        """Build PageAgent + PageFilter on first use. Cached for the lifetime
        of the Orchestrator instance."""
        if self._page_agent is None:
            page_filter = PageFilter(self._sm)
            self._page_agent = PageAgent(
                sm=self._sm,
                page_filter=page_filter,
                model="gemini-3-flash-preview",   # CU model — hardcoded
                max_turns=self._max_turns_per_step,
            )
        return self._page_agent

    async def run_task(self, task: str) -> PlanningTree:
        """Drive the trigger loop end-to-end. Returns the tree (which now
        carries task_exit, final_answer, totals, journey)."""
        page_agent = self._ensure_page_agent()
        tree = PlanningTree()
        tree.create_root(task)
        # Use the seeded registry (loaded from sites/{site}/tools.json at __init__).
        # Falls back to empty registry if no tools.json exists.
        registry = self._registry

        # Capture initial fresh V5 via the same pipeline CU uses.
        initial_v5 = await page_agent.run_re_extract()
        trigger: Trigger = "task_start"
        browser_state: dict[str, Any] = {
            "url": self._sm.page.url,
            "v5": initial_v5,
        }

        # The loop runs at most max_steps+1 iterations: max_steps branchings,
        # plus one final iteration where the planner can emit complete_task
        # after the last step's tree_update.
        for iteration in range(self._max_steps + 1):
            decision = await call_planner(
                self._sm,
                task=task,
                tree=tree,
                registry=registry,
                trigger=trigger,
                browser_state=browser_state,
            )
            tree.total_input_tokens += decision.input_tokens
            tree.total_output_tokens += decision.output_tokens

            # Apply tree_update FIRST — closes the just-ended in-flight node.
            # On task_start the planner returns null tree_update and we skip
            # (root has no outcome to set). On every other trigger the
            # in-flight node MUST be closed before we either branch a new
            # sibling or terminate — otherwise a follow-up branch() would
            # nest under the just-ended step instead of beside it. If the
            # planner forgot to emit tree_update, fall back to closing as
            # success with a placeholder summary so the tree stays valid.
            if trigger != "task_start":
                outcome = decision.tree_update_outcome or "success"
                summary = decision.tree_update_summary or "(planner emitted no summary)"
                closing_node_id = tree._current_id
                if outcome == "success":
                    tree.complete_current(summary)
                else:
                    tree.prune(summary)
                await self._log_step_boundary("end", closing_node_id)

            # Termination paths.
            if decision.planning_action == "complete_task":
                tree.task_exit = "completed"
                tree.final_answer = decision.final_answer
                tree.set_final_url(self._sm.page.url)
                tree.save_visualization()
                return tree
            if decision.planning_action == "give_up":
                tree.task_exit = "give_up"
                tree.set_final_url(self._sm.page.url)
                tree.save_visualization()
                return tree

            # Step budget — at the final allowed iteration we can no longer
            # branch a new step; terminate with max_steps.
            if iteration == self._max_steps:
                tree.task_exit = "max_steps"
                tree.set_final_url(self._sm.page.url)
                tree.save_visualization()
                return tree

            # Dispatch the new step.
            if decision.planning_action == "continue_cu":
                tree.branch(kind="cu")
                await self._log_step_boundary("start", tree._current_id)
                step_result = await page_agent.run_step(task)
                tree.total_input_tokens += step_result.total_input_tokens
                tree.total_output_tokens += step_result.total_output_tokens
                # On a NAVIGATED exit, `step_result.last_v5` is the pre-nav
                # snapshot — the new page hasn't been extracted yet. Re-extract
                # so the next planner turn sees V5 that matches the URL.
                fresh_v5: str | None = None
                if step_result.exit_reason == SessionExit.NAVIGATED:
                    fresh_v5 = await page_agent.run_re_extract()
                trigger = "cu_returned"
                browser_state = self._build_cu_browser_state(step_result, fresh_v5)
                continue

            if decision.planning_action == "invoke_tool":
                tool_id = decision.tool_id or "?"
                tool_user_intent = decision.tool_user_intent or {}
                tree.branch(
                    kind="tool",
                    tool_id=tool_id,
                    tool_user_intent=tool_user_intent,
                )
                await self._log_step_boundary("start", tree._current_id)
                # Lazy-construct the executor on first invoke
                if self._tool_executor is None and self._registry._tools:
                    from morphnet_v2.tool_executor import ToolExecutor, load_tools_for_site
                    site = getattr(self._sm, "site_name", None) or ""
                    self._tool_executor = ToolExecutor(self._sm, load_tools_for_site(site))

                if self._tool_executor is None:
                    # No tools registered — synthesise a 503 so the planner can route differently.
                    trigger = "tool_returned"
                    browser_state = self._build_tool_browser_state(
                        tool_id=tool_id,
                        tool_user_intent=tool_user_intent,
                        http_status=503,
                        success=False,
                        error="no tools registered for site",
                        response_digest="(empty registry)",
                    )
                else:
                    # Pass the user task description + tool_id into planner-values
                    # so the list_selector LLM has context.
                    pv = dict(tool_user_intent)
                    pv.setdefault("_user_task", task)
                    pv.setdefault("_tool_id", tool_id)
                    replay_result = await self._tool_executor.replay(tool_id, pv)
                    success = replay_result.http_status == 200
                    if success:
                        registry.record_success(tool_id)
                    else:
                        registry.record_failure(
                            tool_id, f"HTTP {replay_result.http_status}: {replay_result.error or ''}",
                        )
                    trigger = "tool_returned"
                    browser_state = self._build_tool_browser_state(
                        tool_id=tool_id,
                        tool_user_intent=tool_user_intent,
                        http_status=replay_result.http_status,
                        success=success,
                        error=replay_result.error,
                        response_digest=str(replay_result.body)[:600],
                    )
                continue

            # Unknown action — should never happen since the schema is enum-bounded.
            tree.task_exit = "infrastructure_error"
            tree.set_final_url(self._sm.page.url)
            tree.save_visualization()
            return tree

        # Fallthrough — unreachable in practice (loop bounded by max_steps+1).
        tree.task_exit = "max_steps"
        tree.set_final_url(self._sm.page.url)
        tree.save_visualization()
        return tree

    async def _log_step_boundary(self, phase: str, step_node_id: str | None) -> None:
        coverage = await self._sm.take_coverage_snapshot()
        notes.log(
            data_type="step_boundary",
            data={"coverage": coverage},
            phase=phase,
            step_node_id=step_node_id,
            url=self._sm.page.url,
        )

    # ---- browser_state builders ---------------------------------------

    def _build_cu_browser_state(
        self, step_result: "StepResult", fresh_v5: str | None = None,
    ) -> dict[str, Any]:
        """Build the dict for a `cu_returned` trigger. action_log is passed
        in FULL — no truncation. Token cost is acceptable; the planner needs
        the full temporal history to identify loops + extract data points.

        `fresh_v5` is set by the caller on NAVIGATED exits (where
        `step_result.last_v5` is pre-nav and would mismatch the current URL).
        """
        return {
            "url": self._sm.page.url,
            "v5": fresh_v5 if fresh_v5 is not None else step_result.last_v5,
            "cu_success": step_result.success,
            "cu_exit_reason": step_result.exit_reason.value,
            "cu_total_actions": step_result.total_actions_attempted,
            "cu_failed_actions": step_result.total_actions_failed,
            "cu_report_message": step_result.report_message,
            "cu_action_history": list(step_result.action_log),
        }

    def _build_tool_browser_state(
        self,
        *,
        tool_id: str,
        tool_user_intent: dict,
        http_status: int,
        success: bool,
        error: str | None,
        response_digest: str,
    ) -> dict[str, Any]:
        """Build the dict for a `tool_returned` trigger. No V5 — browser
        state is stale from before the tool ran. Orchestrator re-renders
        for the next CU dispatch via `page_agent.run_re_extract()`."""
        return {
            "url": self._sm.page.url,
            "tool_id": tool_id,
            "tool_user_intent": tool_user_intent,
            "tool_http_status": http_status,
            "tool_success": success,
            "tool_error": error,
            "tool_response_digest": response_digest,
        }
