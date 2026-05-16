"""
executor.py — Graph runner for MorphNet.

Executes learned graphs via CDP. Nearly zero LLM calls — only for array
selection when chained responses have multiple candidates.
Takes a graph and per-node user_intent values, runs nodes in topological
order, chains outputs via JSONPath extraction or LLM selection,
and returns all node responses to the orchestrator.

States returned to orchestrator:
  - success(result): graph completed, terminal response available
  - not_applicable: preconditions failed
  - degraded: canary failed, graph may need rediscovery
  - execution_error(node_id, reason): mid-execution failure
  - completion_timeout: graph ran but success indicator didn't appear
"""

from __future__ import annotations

import json
import logging
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

from morphnet.manifest import (
    Graph,
    GraphNode,
    GraphEdge,
    ExecutionResult,
    update_graph_stats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def _topological_sort(nodes: list[GraphNode], edges: list[GraphEdge]) -> list[GraphNode]:
    """Sort nodes in topological order based on edges.

    Returns nodes ordered so that for every edge A→B, A appears before B.
    Falls back to original order if the graph has no edges or has cycles.
    """
    if not edges:
        return list(nodes)

    node_map = {n.id: n for n in nodes}
    in_degree = {n.id: 0 for n in nodes}
    adjacency: dict[str, list[str]] = {n.id: [] for n in nodes}

    for edge in edges:
        if edge.to_node_id in in_degree:
            in_degree[edge.to_node_id] += 1
        if edge.from_node_id in adjacency:
            adjacency[edge.from_node_id].append(edge.to_node_id)

    # Kahn's algorithm
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        nid = queue.pop(0)
        if nid in node_map:
            result.append(node_map[nid])
        for neighbor in adjacency.get(nid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all nodes were sorted (cycle), append remaining
    sorted_ids = {n.id for n in result}
    for node in nodes:
        if node.id not in sorted_ids:
            result.append(node)

    return result


# ---------------------------------------------------------------------------
# JSONPath extraction from response
# ---------------------------------------------------------------------------

def _extract_jsonpath(data: Any, path: str) -> Any:
    """Extract a value from nested data using a JSONPath-like expression.

    Supports: $.key, $.key.subkey, $.key[0], $.key[0].subkey
    """
    if not path or path == "$":
        return data

    # Strip leading $. or $
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]

    current = data
    for part in _split_jsonpath(path):
        if current is None:
            return None
        if part.startswith("[") and part.endswith("]"):
            # Array index
            try:
                idx = int(part[1:-1])
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            except (ValueError, IndexError):
                return None
        else:
            # Object key
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

    return current


def _split_jsonpath(path: str) -> list[str]:
    """Split a JSONPath into parts: 'data.stationList[0].stationCode' →
    ['data', 'stationList', '[0]', 'stationCode']
    """
    parts = []
    current = ""
    for char in path:
        if char == ".":
            if current:
                parts.append(current)
                current = ""
        elif char == "[":
            if current:
                parts.append(current)
                current = ""
            current = "["
        elif char == "]":
            current += "]"
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------------------
# Parameter substitution in expressions
# ---------------------------------------------------------------------------

def _substitute_params(expression: str, params: dict[str, Any]) -> str:
    """Replace ${params.name} and ${name} placeholders with actual values.

    Placeholders appear inside JS string literals (URL templates, body templates)
    so string values are injected RAW (no added quotes). Non-string values are
    serialized: booleans to true/false, numbers to digits, null to null.
    """
    result = expression
    for key, value in params.items():
        # Handle ${params.key} pattern
        placeholder1 = f"${{params.{key}}}"
        # Handle ${key} pattern
        placeholder2 = f"${{{key}}}"

        if isinstance(value, str):
            safe_val = value  # Raw — placeholder is inside a JS string literal
        elif isinstance(value, bool):
            safe_val = "true" if value else "false"
        elif isinstance(value, (int, float)):
            safe_val = str(value)
        elif value is None:
            safe_val = "null"
        else:
            safe_val = json.dumps(value)

        result = result.replace(placeholder1, safe_val)
        result = result.replace(placeholder2, safe_val)

    return result


# ---------------------------------------------------------------------------
# Response structure comparison
# ---------------------------------------------------------------------------

def _compare_structure(expected: Any, actual: Any) -> bool:
    """Compare JSON response structures (key names and types, not values).

    Returns True if the actual response has the same top-level key structure.
    Allows extra keys in actual (superset is fine).
    """
    if expected is None or actual is None:
        return actual is not None  # As long as we got a response

    if isinstance(expected, dict) and isinstance(actual, dict):
        # Check all expected keys exist in actual
        for key in expected:
            if key not in actual:
                return False
        return True

    if isinstance(expected, list) and isinstance(actual, list):
        return True  # Both are arrays, structure matches

    # Scalar types: both exist, that's enough
    return True


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Deterministic graph runner via CDP.

    Executes learned graphs with user_intent values substituted.
    No LLM calls — purely mechanical execution.
    """

    def __init__(self, session_manager: Any):
        """Initialize executor.

        Args:
            session_manager: SessionManager instance with page, evaluate_js(), and
                             wait_for_dom_stable() methods.
        """
        self._session = session_manager

    async def execute(
        self,
        graph: Graph,
        user_intent: dict[str, dict[str, Any]],
        subtask_context: str = "",
    ) -> ExecutionResult:
        """Execute a graph with per-node user_intent parameter values.

        Pipeline:
        1. Precondition check (URL pattern, required globals, bundle hash)
        2. Canary test (simplest node with example values)
        3. Resolve user_intent parameters (per-node)
        4. Topological execution (CDP evaluate each node)
        5. Return all node outputs to orchestrator
        6. Stats update

        Args:
            graph: The graph to execute.
            user_intent: Per-node params: {node_id: {param_name: value}}.
            subtask_context: Natural language task description for array selection.

        Returns:
            ExecutionResult with status and node_outputs.
        """
        self._subtask_context = subtask_context
        t0 = time.time()
        logger.info("Executor: starting graph %s (%s)", graph.id[:12], graph.name)

        # Step 1: Precondition check
        result = await self._check_preconditions(graph)
        if result:
            self._update_stats(graph, success=False)
            return result

        # Step 2: Canary test
        canary_ok = await self._canary_test(graph)
        if not canary_ok:
            self._update_stats(graph, success=False)
            return ExecutionResult(
                status="degraded",
                reason="Canary test failed — graph may need rediscovery",
            )

        # Step 3: Resolve parameters
        resolved_params = self._resolve_params(graph, user_intent)

        # Step 4: Topological execution
        sorted_nodes = _topological_sort(graph.nodes, graph.edges)
        node_outputs: dict[str, Any] = {}  # node_id -> parsed response

        for node in sorted_nodes:
            node_result = await self._execute_node(node, graph, resolved_params, node_outputs)
            if node_result is None:
                self._update_stats(graph, success=False)
                return ExecutionResult(
                    status="execution_error",
                    failed_node_id=node.id,
                    reason=f"Node {node.id} ({node.endpoint_fingerprint}) failed",
                )
            node_outputs[node.id] = node_result

        # Step 5: Completion
        completion_result = await self._complete(graph, resolved_params, node_outputs)

        elapsed = time.time() - t0
        logger.info("Executor: graph %s completed in %.1fs — %s",
                     graph.id[:12], elapsed, completion_result.status)

        # Step 6: Stats update
        self._update_stats(graph, success=completion_result.status == "success")

        return completion_result

    # ------------------------------------------------------------------
    # Step 1: Precondition check
    # ------------------------------------------------------------------

    async def _check_preconditions(self, graph: Graph) -> Optional[ExecutionResult]:
        """Check if the graph's preconditions are met.

        Returns None if all preconditions pass, or an ExecutionResult if they fail.
        """
        page = self._session.page
        if not page or page.is_closed():
            return ExecutionResult(
                status="not_applicable",
                reason="No active browser page",
            )

        # URL pattern check
        url_pattern = graph.preconditions.get("url_pattern", "*")
        current_url = page.url
        if not fnmatch(current_url, url_pattern):
            return ExecutionResult(
                status="not_applicable",
                reason=f"URL {current_url} does not match pattern {url_pattern}",
            )

        # Required globals check
        required_globals = graph.preconditions.get("required_globals", [])
        missing_globals = []
        for global_path in required_globals[:5]:  # Check first 5 only for speed
            try:
                exists = await page.evaluate(f"typeof window.{global_path} !== 'undefined'")
                if not exists:
                    missing_globals.append(global_path)
            except Exception:
                missing_globals.append(global_path)

        if missing_globals:
            return ExecutionResult(
                status="not_applicable",
                reason=f"Required globals not found: {', '.join(missing_globals)}",
            )

        # Bundle hash check: compare loaded script URLs against expected
        expected_hash = graph.preconditions.get("bundle_hash", "")
        if expected_hash:
            try:
                current_scripts = await page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('script[src]'))
                        .map(s => s.src)
                        .sort();
                }""")
                import hashlib as _hl
                current_hash = _hl.sha256("|".join(current_scripts or []).encode()).hexdigest()
                if current_hash != expected_hash:
                    logger.info("Bundle hash mismatch: expected %s, got %s — escalating to canary",
                                expected_hash[:12], current_hash[:12])
                    # Don't fail here — escalate to canary test which will catch real breakage
            except Exception as exc:
                logger.debug("Bundle hash check failed: %s", exc)

        return None  # All preconditions passed

    # ------------------------------------------------------------------
    # Step 2: Canary test
    # ------------------------------------------------------------------

    async def _canary_test(self, graph: Graph) -> bool:
        """Run the simplest node with example values to verify the graph still works.

        Pick the node with fewest dependencies (first in topological order).
        Execute with example values. Compare response structure.
        """
        sorted_nodes = _topological_sort(graph.nodes, graph.edges)
        if not sorted_nodes:
            return False

        canary_node = sorted_nodes[0]

        # Build example params from the node's stored examples
        example_params = {}
        for param in canary_node.core_parameters:
            if param.value_example is not None:
                example_params[param.name] = param.value_example

        # Build invocation with example values
        invocation = canary_node.invocation
        if not invocation.expression:
            # No expression to evaluate — skip canary (DOM replay nodes)
            return True

        expression = _substitute_params(invocation.expression, example_params)
        logger.info("Canary: node %s, params=%s, expr=%s",
                     canary_node.id, list(example_params.keys()), expression[:200])

        try:
            page = self._session.page
            result = await page.evaluate(f"""async () => {{
                try {{
                    const result = {expression};
                    return {{ success: true, data: result }};
                }} catch(e) {{
                    return {{ success: false, error: e.message }};
                }}
            }}""")

            if not result.get("success"):
                logger.info("Canary test failed for node %s: %s", canary_node.id,
                            result.get("error", "unknown"))
                return False

            # Compare response structure if we have an example
            if canary_node.example_response_body:
                try:
                    expected = json.loads(canary_node.example_response_body)
                    actual = result.get("data")
                    if not _compare_structure(expected, actual):
                        logger.info("Canary test: response structure mismatch for node %s",
                                    canary_node.id)
                        return False
                except (json.JSONDecodeError, TypeError):
                    pass

            logger.info("Canary test passed for graph %s", graph.id[:12])
            return True
        except Exception as exc:
            logger.info("Canary test error for node %s: %s", canary_node.id, exc)
            return False

    # ------------------------------------------------------------------
    # Step 3: Resolve parameters
    # ------------------------------------------------------------------

    def _resolve_params(
        self,
        graph: Graph,
        user_intent: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Resolve per-node parameters from user_intent.

        Returns {node_id: {param_name: resolved_value}}.
        """
        resolved: dict[str, dict[str, Any]] = {}
        for node in graph.nodes:
            node_params: dict[str, Any] = {}
            node_intent = user_intent.get(node.id, {})
            for param in node.core_parameters:
                if param.role == "user_intent":
                    value = node_intent.get(param.name)
                    if value is not None:
                        for transform in param.transformations:
                            value = self._apply_transform(value, transform)
                        node_params[param.name] = value
                    elif param.value_example is not None:
                        node_params[param.name] = param.value_example
                elif param.role == "website_generated":
                    if param.value_example is not None:
                        node_params[param.name] = param.value_example
            resolved[node.id] = node_params
        return resolved

    def _apply_transform(self, value: Any, transform: str) -> Any:
        """Apply a named transformation to a parameter value."""
        if transform == "uppercase":
            return str(value).upper()
        if transform == "lowercase":
            return str(value).lower()
        if transform == "date_iso_to_dmy":
            # Convert YYYY-MM-DD to DD-MM-YYYY
            try:
                parts = str(value).split("-")
                if len(parts) == 3:
                    return f"{parts[2]}-{parts[1]}-{parts[0]}"
            except Exception:
                pass
        if transform == "url_encode":
            from urllib.parse import quote
            return quote(str(value))
        return value

    # ------------------------------------------------------------------
    # Step 4: Execute a single node
    # ------------------------------------------------------------------

    def _find_edge(
        self, graph: Graph, from_id: str, to_id: str, to_param: str,
    ) -> Optional[GraphEdge]:
        """Find the edge connecting two nodes for a specific parameter."""
        return next(
            (e for e in graph.edges
             if e.from_node_id == from_id and e.to_node_id == to_id
             and e.to_parameter == to_param),
            None,
        )

    async def _select_from_array(
        self,
        array_data: list,
        subtask_context: str,
        node_description: str,
        target_param: str,
        item_field: str,
    ) -> Any:
        """LLM selection: pick the correct item from an array based on user intent.

        If array has exactly 1 element, returns it immediately (no LLM call).
        Otherwise, calls Flash model to select.
        """
        if len(array_data) == 1:
            return array_data[0]

        from morphnet.session_manager import call_gemini_async

        # Format items for the LLM — show key fields, cap at 20 items
        items_display = []
        for i, item in enumerate(array_data[:20]):
            if isinstance(item, dict):
                # Show compact key-value pairs
                fields = ", ".join(f"{k}: {v}" for k, v in list(item.items())[:8])
                items_display.append(f"  [{i}] {fields}")
            else:
                items_display.append(f"  [{i}] {item}")

        prompt = (
            f"User's task: {subtask_context}\n"
            f"Current workflow step: {node_description}\n"
            f"Target parameter: {target_param}\n"
            f"Field to extract from selected item: {item_field}\n\n"
            f"Array items ({len(array_data)} total):\n"
            + "\n".join(items_display)
            + "\n\nSelect the item that best matches the user's intent for this step."
        )

        selection_schema = {
            "type": "object",
            "properties": {
                "selected_index": {
                    "type": "integer",
                    "description": "0-based index of the correct item",
                },
                "reasoning": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["selected_index", "reasoning", "confidence"],
        }

        try:
            system_prompt = (Path(__file__).parent / "prompts" / "array_selection.txt").read_text(encoding="utf-8")
        except FileNotFoundError:
            system_prompt = "Select the array item that best matches the user's intent."

        try:
            result = await call_gemini_async(
                model="gemini-3-flash-preview",
                contents=[prompt],
                response_schema=selection_schema,
                system_instruction=system_prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 512},
            )
            idx = result.get("selected_index", 0)
            if 0 <= idx < len(array_data):
                logger.info("Array selection: picked [%d] for %s (confidence=%.2f): %s",
                            idx, target_param, result.get("confidence", 0), result.get("reasoning", "")[:100])
                return array_data[idx]
        except Exception as exc:
            logger.warning("Array selection LLM failed: %s — defaulting to [0]", exc)

        return array_data[0]

    async def _execute_node(
        self,
        node: GraphNode,
        graph: Graph,
        resolved_params: dict[str, dict[str, Any]],
        node_outputs: dict[str, Any],
    ) -> Optional[Any]:
        """Execute a single graph node via CDP.

        Resolves all inputs (per-node user_intent/website_generated, chained from
        upstream outputs), substitutes into the invocation expression, and CDP-evaluates.
        For chained params with requires_selection edges, uses LLM to pick from array.
        Returns parsed response data, or None on failure.
        """
        # Start with this node's resolved params (user_intent + website_generated)
        node_params = dict(resolved_params.get(node.id, {}))

        # Resolve chained parameters from upstream node outputs
        for param in node.core_parameters:
            if param.role == "chained" and param.chained_from:
                parts = param.chained_from.split(".", 1)
                if len(parts) == 2:
                    source_node_id, extract_path = parts
                    source_data = node_outputs.get(source_node_id)
                    if source_data is None:
                        continue

                    # Check if this edge requires LLM selection from array
                    edge = self._find_edge(graph, source_node_id, node.id, param.name)
                    if edge and edge.requires_selection:
                        jsonpath = edge.selection_array_path
                        if not jsonpath.startswith("$"):
                            jsonpath = f"$.{jsonpath}"
                        array_data = _extract_jsonpath(source_data, jsonpath)
                        if isinstance(array_data, list) and array_data:
                            selected_item = await self._select_from_array(
                                array_data,
                                self._subtask_context,
                                node.node_description,
                                param.name,
                                edge.selection_item_field,
                            )
                            if edge.selection_item_field and isinstance(selected_item, dict):
                                node_params[param.name] = selected_item.get(edge.selection_item_field)
                            else:
                                node_params[param.name] = selected_item
                    else:
                        # Direct extraction
                        jsonpath = extract_path if extract_path.startswith("$") else f"$.{extract_path}"
                        extracted = _extract_jsonpath(source_data, jsonpath)
                        if extracted is not None:
                            node_params[param.name] = extracted

        invocation = node.invocation

        if not invocation.expression:
            logger.warning("Node %s has no invocation expression", node.id)
            return None

        # Substitute parameters into expression
        expression = _substitute_params(invocation.expression, node_params)

        logger.debug("Executing node %s: %s", node.id, expression[:200])

        try:
            page = self._session.page

            # Wrap in async IIFE for CDP evaluation
            wrapper = f"""async () => {{
                try {{
                    const result = {expression};
                    return {{ success: true, data: result }};
                }} catch(e) {{
                    return {{ success: false, error: e.message, stack: e.stack }};
                }}
            }}"""

            result = await page.evaluate(f"({wrapper})()")

            if not result or not result.get("success"):
                error = result.get("error", "unknown") if result else "null result"
                logger.warning("Node %s execution failed: %s", node.id, error)
                return None

            data = result.get("data")
            logger.info("Node %s executed successfully", node.id)

            return data

        except Exception as exc:
            logger.warning("Node %s CDP evaluation error: %s", node.id, exc)
            return None

    # ------------------------------------------------------------------
    # Step 5: Completion
    # ------------------------------------------------------------------

    async def _complete(
        self,
        graph: Graph,
        resolved_params: dict[str, dict[str, Any]],
        node_outputs: dict[str, Any],
    ) -> ExecutionResult:
        """Return success with all node outputs."""
        return ExecutionResult(status="success", node_outputs=node_outputs)

    # ------------------------------------------------------------------
    # Step 6: Stats update
    # ------------------------------------------------------------------

    def _update_stats(self, graph: Graph, success: bool) -> None:
        """Update execution_stats on the graph."""
        try:
            update_graph_stats(graph.site, graph.id, success)
        except Exception as exc:
            logger.debug("Stats update failed: %s", exc)
