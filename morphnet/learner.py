"""
learner.py — Post-subtask graph builder for MorphNet.

Runs after each CU subtask completes. Examines the observer's recordings
and produces an execution graph (a DAG of CDP invocations). Many subtasks
produce no graph; that is expected.

Pipeline:
  1. Filter relevant HTTP traffic (noise filter + content-type + response size)
  2. Build candidate nodes (CU action windows → prefix-chain collapsing → representatives)
  3. Build chain candidates (exact value matching across prior node responses)
  4. Classify parameter roles via single LLM call (chained / user_intent / website_generated)
  5. Identify entry points (5-strategy fallback)
  6a. Verify invocations via CDP re-invocation (HTTP fingerprint match)
  6b. Verify full pipeline (synthetic task → intent extraction → executor → validate)
  7. Identify terminal nodes
  8. Build completion signals
  9. Compute graph identity
  10. Check against existing registry
  11. Name and describe (ONE LLM call)
  12. Save + update param profile
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs, urlencode

from morphnet.manifest import (
    SubtaskObservation,
    HTTPRequest,
    CUAction,
    DOMSnapshot,
    ParameterSpec,
    NodeInvocation,
    GraphNode,
    GraphEdge,
    Graph,
    ExecutionResult,
    compute_graph_id,
    is_subset,
    graphs_equivalent,
    list_graphs,
    save_graph,
    load_graph,
    save_embeddings,
    load_embeddings,
)
from google.genai import types as genai_types
from morphnet.session_manager import call_gemini
from morphnet.noise_filter import is_noise_url as _is_noise_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM prompt for graph naming (domain-general, no site-specific examples)
# ---------------------------------------------------------------------------

_NAMING_SYSTEM_PROMPT = """You are analyzing a workflow that an AI agent discovered on a website. \
The workflow is a graph of HTTP operations. Your job is to give it a clear, general name and \
description so it can be matched to future tasks semantically.

Examples of good names across domains:
- "Search products by category" (e-commerce search)
- "Find flights between two cities on a date" (travel booking)
- "Add item to cart after verification" (e-commerce mutation)
- "Log in with email and password" (authentication)
- "Get detailed information for a specific item" (detail lookup)
- "Search then filter then get detail" (composite workflow)
- "Post a new message in a thread" (social media mutation)
- "Narrow result list by price range" (filtering)

Be specific about the user's visible outcome. Avoid jargon about HTTP or APIs."""

_NAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Short, capability-focused name (e.g., 'Search products by category')",
        },
        "description": {
            "type": "string",
            "description": "2-3 sentences: what the workflow does, inputs needed, outputs produced",
        },
        "capability_statement": {
            "type": "string",
            "description": "ONE natural language sentence for semantic search",
        },
        "reason_for_version": {
            "type": "string",
            "description": "What this extends over parent workflows (empty if no parents)",
        },
        "node_descriptions": {
            "type": "array",
            "description": "One entry per node: its human-readable role in the workflow",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID (e.g., 'n0', 'n1')"},
                    "description": {
                        "type": "string",
                        "description": (
                            "Brief role description distinguishing this node from others. "
                            "E.g., 'Source station auto-suggest', 'Destination station auto-suggest', "
                            "'Train search with availability'. Same-endpoint nodes MUST have distinct descriptions."
                        ),
                    },
                },
                "required": ["node_id", "description"],
            },
        },
    },
    "required": ["name", "description", "capability_statement", "reason_for_version", "node_descriptions"],
}


# ---------------------------------------------------------------------------
# LLM prompt for parameter classification (chained / user_intent / website_generated)
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / "classify_params.txt"
).read_text(encoding="utf-8")


def _build_classify_schema(params_to_classify: list[dict]) -> dict:
    """Build a Gemini-compatible output schema for param classification.

    One classification entry per param. Gemini doesn't support additionalProperties,
    so we build explicit properties for each param.
    """
    item_schema = {
        "type": "object",
        "properties": {
            "node_id": {"type": "string"},
            "param_name": {"type": "string"},
            "role": {
                "type": "string",
                "enum": ["chained", "user_intent", "website_generated"],
            },
            "chained_from_node_id": {
                "type": "string",
                "description": "Source node ID if role=chained, empty otherwise",
                "nullable": True,
            },
            "chained_from_field": {
                "type": "string",
                "description": "JSONPath in source node response if role=chained, empty otherwise",
                "nullable": True,
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for this classification",
            },
        },
        "required": ["node_id", "param_name", "role", "chained_from_node_id", "chained_from_field", "reasoning"],
    }

    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": item_schema,
            },
        },
        "required": ["classifications"],
    }


# ---------------------------------------------------------------------------
# Helper: JSON value finder
# ---------------------------------------------------------------------------

def _find_value_jsonpath(data: Any, target_value: str, prefix: str = "$") -> Optional[str]:
    """Find the JSONPath of a value in a nested JSON structure.

    Returns the first JSONPath where the target string appears as a leaf value.
    Only matches string values (case-sensitive exact match).
    """
    if isinstance(data, str) and data == target_value:
        return prefix
    if isinstance(data, dict):
        for key, val in data.items():
            path = _find_value_jsonpath(val, target_value, f"{prefix}.{key}")
            if path:
                return path
    if isinstance(data, list):
        for i, val in enumerate(data):
            path = _find_value_jsonpath(val, target_value, f"{prefix}[{i}]")
            if path:
                return path
    return None


def _find_all_values_jsonpath(data: Any, target_value: str, prefix: str = "$") -> list[str]:
    """Find ALL JSONPaths where a value appears in nested JSON."""
    results = []
    if isinstance(data, str) and data == target_value:
        results.append(prefix)
    if isinstance(data, dict):
        for key, val in data.items():
            results.extend(_find_all_values_jsonpath(val, target_value, f"{prefix}.{key}"))
    if isinstance(data, list):
        for i, val in enumerate(data):
            results.extend(_find_all_values_jsonpath(val, target_value, f"{prefix}[{i}]"))
    return results


def _extract_all_leaf_values(data: Any) -> list[tuple[str, str]]:
    """Extract all (jsonpath, string_value) pairs from nested JSON."""
    results = []
    _extract_leaves_recursive(data, "$", results)
    return results


def _extract_leaves_recursive(data: Any, prefix: str, results: list[tuple[str, str]]) -> None:
    if isinstance(data, str):
        results.append((prefix, data))
    elif isinstance(data, (int, float, bool)):
        results.append((prefix, str(data)))
    elif isinstance(data, dict):
        for key, val in data.items():
            _extract_leaves_recursive(val, f"{prefix}.{key}", results)
    elif isinstance(data, list):
        for i, val in enumerate(data):
            _extract_leaves_recursive(val, f"{prefix}[{i}]", results)


def _json_schema_from_value(data: Any, max_depth: int = 5) -> dict:
    """Infer a JSON schema from a value (types of each field, recursively)."""
    if max_depth <= 0:
        return {"type": "any"}
    if data is None:
        return {"type": "null"}
    if isinstance(data, bool):
        return {"type": "boolean"}
    if isinstance(data, int):
        return {"type": "integer"}
    if isinstance(data, float):
        return {"type": "number"}
    if isinstance(data, str):
        return {"type": "string"}
    if isinstance(data, list):
        if not data:
            return {"type": "array", "items": {}}
        return {"type": "array", "items": _json_schema_from_value(data[0], max_depth - 1)}
    if isinstance(data, dict):
        properties = {}
        for key, val in data.items():
            properties[key] = _json_schema_from_value(val, max_depth - 1)
        return {"type": "object", "properties": properties}
    return {"type": "any"}


# ---------------------------------------------------------------------------
# Helper: endpoint fingerprint computation
# ---------------------------------------------------------------------------

# Patterns that look like IDs in URL path segments
_ID_PATTERNS = re.compile(
    r"^("
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # UUID
    r"|[0-9a-f]{24}"  # MongoDB ObjectId
    r"|\d{6,}"  # Long numeric ID
    r"|[A-Za-z0-9_-]{20,}"  # Base64-ish tokens
    r")$"
)


def _compute_endpoint_fingerprint(req: HTTPRequest) -> str:
    """Compute canonical endpoint fingerprint for a request.

    REST: '{method} {url_path_template} [query_param_names_sorted]'
    GraphQL: '{operation_name}#{query_hash[:12]}'
    JSON-RPC: '{jsonrpc_method}'
    """
    if req.request_type == "graphql":
        op = req.graphql_operation_name or "anonymous"
        qhash = req.graphql_query_hash or "nohash"
        return f"{op}#{qhash}"

    if req.request_type == "json_rpc":
        return req.jsonrpc_method or "unknown_rpc"

    # REST: method + path template + sorted query param names
    parsed = urlparse(req.url)
    path_parts = [p for p in parsed.path.split("/") if p]

    # Replace ID-like trailing segments with {id}
    templated_parts = []
    for part in path_parts:
        if _ID_PATTERNS.match(part):
            templated_parts.append("{id}")
        else:
            templated_parts.append(part)
    path_template = "/" + "/".join(templated_parts)

    query_params = sorted(parse_qs(parsed.query).keys())
    param_str = f" [{','.join(query_params)}]" if query_params else ""

    return f"{req.method} {path_template}{param_str}"


def _compute_url_template(req: HTTPRequest) -> str:
    """Compute URL template with {param} placeholders for query parameters."""
    parsed = urlparse(req.url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    params = parse_qs(parsed.query)
    if params:
        template_params = []
        for key in sorted(params.keys()):
            template_params.append(f"{key}={{{key}}}")
        return f"{base}?{'&'.join(template_params)}"
    return base


# ---------------------------------------------------------------------------
# Helper: extract request parameters
# ---------------------------------------------------------------------------

def _extract_request_params(req: HTTPRequest) -> list[tuple[str, str]]:
    """Extract all parameter (name, value) pairs from a request.

    Combines query string params + JSON body fields + form fields.
    Returns flat list of (name, value_as_string).
    """
    params = []

    # Query string parameters
    parsed = urlparse(req.url)
    for key, values in parse_qs(parsed.query).items():
        for v in values:
            params.append((key, v))

    # Body parameters
    if req.body:
        try:
            body_parsed = json.loads(req.body)
            if isinstance(body_parsed, dict):
                for key, val in body_parsed.items():
                    if isinstance(val, (str, int, float, bool)):
                        params.append((key, str(val)))
                    elif isinstance(val, dict):
                        # Flatten one level for GraphQL variables etc.
                        for subkey, subval in val.items():
                            if isinstance(subval, (str, int, float, bool)):
                                params.append((f"{key}.{subkey}", str(subval)))
        except (json.JSONDecodeError, TypeError):
            # Try form-encoded
            if req.body and "=" in req.body:
                for pair in req.body.split("&"):
                    if "=" in pair:
                        key, _, val = pair.partition("=")
                        params.append((key, val))

    return params


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class Learner:
    """Post-subtask graph builder.

    Examines observer recordings and produces execution graphs.
    Uses LLM for param classification (one call) and naming (one call).
    """

    def __init__(
        self,
        session_manager: Any,
        embedding_client: Any = None,
    ):
        """Initialize learner.

        Args:
            session_manager: SessionManager instance (for CDP evaluation during verification).
            embedding_client: Optional embedding client with embed(text) -> list[float] method.
        """
        self._session = session_manager
        self._embedding_client = embedding_client
        self._node_to_request: dict[str, Any] = {}  # populated by _build_candidate_nodes

    # ------------------------------------------------------------------
    # Profile storage: per-endpoint param value history across tasks
    # ------------------------------------------------------------------

    @staticmethod
    def _load_profile(site: str) -> dict[str, dict[str, list[str]]]:
        """Load param profile: {endpoint_fingerprint: {param_name: [values_seen]}}."""
        from morphnet.manifest import SITES_DIR
        path = SITES_DIR / site / "profiles.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _save_profile(site: str, profile: dict[str, dict[str, list[str]]]) -> None:
        """Save param profile."""
        from morphnet.manifest import SITES_DIR
        site_dir = SITES_DIR / site
        site_dir.mkdir(parents=True, exist_ok=True)
        path = site_dir / "profiles.json"
        path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    @staticmethod
    def _update_profile(site: str, nodes: list[GraphNode]) -> None:
        """Update profile with param values from this graph build."""
        profile = Learner._load_profile(site)
        for node in nodes:
            fp = node.endpoint_fingerprint
            if fp not in profile:
                profile[fp] = {}
            for param in node.core_parameters:
                val = str(param.value_example)
                if param.name not in profile[fp]:
                    profile[fp][param.name] = []
                history = profile[fp][param.name]
                if val not in history:
                    history.append(val)
                # Cap history at 50 values
                profile[fp][param.name] = history[-50:]
        Learner._save_profile(site, profile)

    async def learn_from_subtask(self, observation: SubtaskObservation) -> Optional[Graph]:
        """Main pipeline: observation → graph.

        Returns the graph that was created or updated, or None if learning produced nothing.
        """
        logger.info("Learner processing subtask %s (%d HTTP requests, %d CU actions)",
                     observation.subtask_id, len(observation.http_requests), len(observation.cu_actions))

        # Step 1: Filter relevant HTTP traffic
        relevant_requests = self._filter_traffic(observation)
        if not relevant_requests:
            logger.info("No relevant HTTP requests after filtering — skipping graph building")
            return None

        logger.info("Step 1: %d/%d requests survived filtering",
                     len(relevant_requests), len(observation.http_requests))

        # Step 2: Build candidate nodes (CU action windows + prefix-chain collapsing)
        nodes = self._build_candidate_nodes(relevant_requests, observation)
        if not nodes:
            logger.info("No candidate nodes built — skipping")
            return None

        logger.info("Step 2: Built %d candidate nodes (from %d requests)", len(nodes), len(relevant_requests))

        # Step 3: Build chain candidates (exact value matching across prior responses)
        chain_candidates = self._build_chain_candidates(nodes)
        total_candidates = sum(len(v) for v in chain_candidates.values())
        logger.info("Step 3: Found %d chain candidates across %d params", total_candidates, len(chain_candidates))

        # Step 4: Classify parameter roles via single bundled LLM call
        # Returns edges (from chained params) and updated nodes
        edges, nodes = self._classify_parameters(nodes, chain_candidates, observation)
        logger.info("Step 4: Classified params — %d edges from chained params", len(edges))

        # pushState/replaceState are side effects of executing API nodes — the SPA's
        # own JS fires them when it processes responses. We observe the URL change but
        # don't build separate graph nodes for them.

        # Step 5: Identify entry points
        nodes_with_invocation = await self._identify_entry_points(nodes, observation)
        if not nodes_with_invocation:
            logger.info("No nodes survived entry point identification — skipping")
            return None

        logger.info("Step 5: %d/%d nodes have entry points", len(nodes_with_invocation), len(nodes))

        # Check if any critical node was lost (terminal or has dependents)
        surviving_ids = {n.id for n in nodes_with_invocation}
        lost_ids = {n.id for n in nodes} - surviving_ids
        if lost_ids:
            # A lost node is critical if: (a) it's a terminal candidate, or
            # (b) other surviving nodes depend on it via edges
            has_dependents = any(
                e.from_node_id in lost_ids and e.to_node_id in surviving_ids
                for e in edges
            )
            # Check if lost node would be terminal (last node in original list)
            is_terminal_lost = nodes[-1].id in lost_ids if nodes else False
            if has_dependents or is_terminal_lost:
                logger.warning(
                    "Critical node(s) lost entry points (%s) — discarding graph",
                    ", ".join(sorted(lost_ids)),
                )
                return None

        # Filter edges to only reference surviving nodes
        edges = [e for e in edges if e.from_node_id in surviving_ids and e.to_node_id in surviving_ids]

        # Step 6: Verify — two stages
        # 6a: HTTP re-invocation (quick sanity check — invocations fire correct requests)
        # 6b: Full pipeline (synthetic task → intent extraction → executor → validate responses)
        is_read_only = all(n.http_method in ("GET", "OPTIONS", "HEAD", "NAVIGATE") for n in nodes_with_invocation)
        verified = False
        if is_read_only:
            verified_invocations = await self._verify_invocations(nodes_with_invocation, observation)
            logger.info("Step 6a: Invocation verification %s", "passed" if verified_invocations else "failed")
            if not verified_invocations:
                logger.warning("Read-only graph failed invocation verification — discarding")
                return None

            verified_pipeline = await self._verify_pipeline(nodes_with_invocation, edges, observation)
            logger.info("Step 6b: Pipeline verification %s", "passed" if verified_pipeline else "failed")
            verified = verified_invocations and verified_pipeline
        else:
            logger.info("Step 6: Skipping verification (contains write operations)")

        # If verification failed for read-only graph, skip saving
        if is_read_only and not verified:
            logger.warning("Read-only graph failed verification — discarding")
            return None

        # Step 7+8: Terminal detection and completion are not needed.
        # pushState fires as a side effect of API node execution.
        terminal_ids: list[str] = []
        completion: dict = {}

        # Step 9: Compute graph identity
        graph_id = compute_graph_id(nodes_with_invocation, edges)
        logger.info("Step 9: Graph ID = %s", graph_id[:12])

        # Step 10: Check against existing registry
        existing_graphs = list_graphs(observation.site)
        action, parent_ids = self._check_registry(
            graph_id, nodes_with_invocation, edges, existing_graphs,
        )
        logger.info("Step 10: Registry check → %s", action)

        if action == "already_exists":
            # Update observation history on existing graph
            existing = load_graph(observation.site, graph_id)
            if existing:
                for node in existing.nodes:
                    if observation.subtask_id not in node.observed_in_subtasks:
                        node.observed_in_subtasks.append(observation.subtask_id)
                save_graph(existing)
            return existing

        if action == "subsumed":
            # New graph is already a subset of an existing one — just update parent
            return None

        # Step 11: Name and describe (ONE LLM call)
        naming = await self._name_graph(
            observation, nodes_with_invocation, edges, parent_ids, existing_graphs,
        )
        logger.info("Step 11: Named graph: %s", naming.get("name", "unnamed"))

        # Apply per-node descriptions from naming LLM
        node_lookup = {n.id: n for n in nodes_with_invocation}
        for nd in naming.get("node_descriptions", []):
            node = node_lookup.get(nd.get("node_id", ""))
            if node:
                node.node_description = nd.get("description", "")

        # Determine detected framework
        frameworks = observation.framework_fingerprint.get("frameworks", [])
        if "nextjs" in frameworks:
            framework = "nextjs"
        elif "react" in frameworks:
            framework = "react"
        elif "vue" in frameworks or "nuxt" in frameworks:
            framework = "vue"
        elif "angular" in frameworks:
            framework = "angular"
        else:
            framework = "unknown"

        # Build preconditions
        start_host = urlparse(observation.start_url).hostname or ""
        preconditions = {
            "url_pattern": f"*{start_host}*",
            "required_globals": list(observation.framework_fingerprint.get("globals", {}).keys())[:10],
            "bundle_hash": observation.bundle_hash,
        }

        # Build the Graph
        graph = Graph(
            id=graph_id,
            site=observation.site,
            name=naming.get("name", "Unnamed workflow"),
            description=naming.get("description", ""),
            capability_statement=naming.get("capability_statement", ""),
            reason_for_version=naming.get("reason_for_version", ""),
            parent_graph_ids=parent_ids,
            nodes=nodes_with_invocation,
            edges=edges,
            terminal_node_ids=terminal_ids,
            completion=completion,
            preconditions=preconditions,
            verified=verified,
            verification_only_read=is_read_only,
            framework_detected=framework,
        )

        # Step 12: Save
        save_graph(graph)
        logger.info("Step 12: Saved graph %s (%s)", graph_id[:12], graph.name)

        # Update param profile with values from this build
        self._update_profile(observation.site, nodes_with_invocation)

        # Embed capability statement if client available
        if self._embedding_client:
            try:
                emb = self._embedding_client.embed(graph.capability_statement)
                existing_emb = load_embeddings(observation.site)
                existing_emb[graph.id] = emb
                save_embeddings(observation.site, existing_emb)
            except Exception as exc:
                logger.debug("Embedding failed: %s", exc)

        return graph

    # ------------------------------------------------------------------
    # Step 1: Filter relevant HTTP traffic
    # ------------------------------------------------------------------

    def _filter_traffic(self, observation: SubtaskObservation) -> list[HTTPRequest]:
        """Filter out noise, analytics, empty responses, and prefetches."""
        candidates = []

        for req in observation.http_requests:
            # Skip noise domains
            if _is_noise_url(req.url):
                continue

            # Skip preflight/prefetch
            if req.initiator_type in ("preflight", "preload", "prefetch"):
                continue
            if req.method == "OPTIONS":
                continue

            # Skip empty or trivial responses
            if req.response_status == 0:
                continue
            body = req.response_body or ""
            if len(body) < 100:
                # Allow small responses that might be consumed downstream
                # (we'll check this in the second pass)
                try:
                    parsed = json.loads(body) if body else {}
                    if isinstance(parsed, dict):
                        # Simple status-only responses
                        if set(parsed.keys()) <= {"status", "ok", "success", "message"}:
                            continue
                except (json.JSONDecodeError, TypeError):
                    continue

            # Skip non-JSON/text responses
            resp_ct = req.response_headers.get("content-type", "")
            if not any(t in resp_ct for t in ("json", "text", "graphql", "xml", "javascript")):
                if body and not body.strip().startswith(("{", "[")):
                    continue

            candidates.append(req)

        if not candidates:
            return []

        # Second pass: keep requests whose response values flow downstream,
        # appear in URL changes, or appear in DOM content hash changes.
        all_downstream_params: set[str] = set()
        for req in candidates:
            for name, val in _extract_request_params(req):
                if len(val) >= 2:  # Skip single-char values
                    all_downstream_params.add(val)

        # Collect URL change values for relevance check
        url_values: set[str] = set()
        snapshot_urls = [s.url for s in observation.dom_snapshots]
        for i in range(1, len(snapshot_urls)):
            if snapshot_urls[i] != snapshot_urls[i - 1]:
                # Extract path segments and query values from the new URL
                parsed = urlparse(snapshot_urls[i])
                for seg in parsed.path.split("/"):
                    if len(seg) >= 2:
                        url_values.add(seg)
                for vals in parse_qs(parsed.query).values():
                    for v in vals:
                        if len(v) >= 2:
                            url_values.add(v)

        # Collect DOM hash transitions for checking response-driven DOM changes
        dom_transitions: list[tuple[int, int]] = []  # (response_ts_range_start, range_end)
        for i in range(1, len(observation.dom_snapshots)):
            prev = observation.dom_snapshots[i - 1]
            curr = observation.dom_snapshots[i]
            if curr.dom_content_hash != prev.dom_content_hash:
                dom_transitions.append((prev.timestamp_ms, curr.timestamp_ms))

        final = []
        for req in candidates:
            # Always keep the request if it's substantial
            if len(req.response_body or "") >= 100:
                final.append(req)
                continue

            kept = False
            if req.response_body:
                try:
                    resp_data = json.loads(req.response_body)
                    leaves = _extract_all_leaf_values(resp_data)
                    for _, val in leaves:
                        # Keep if response value appears in downstream request params
                        if val in all_downstream_params:
                            kept = True
                            break
                        # Keep if response value appears in a URL change
                        if val in url_values:
                            kept = True
                            break
                except (json.JSONDecodeError, TypeError):
                    pass

            # Keep if the request's response timestamp falls within a DOM change window
            if not kept:
                resp_ts = req.timestamp_ms
                for dom_start, dom_end in dom_transitions:
                    if dom_start <= resp_ts <= dom_end:
                        kept = True
                        break

            if kept:
                final.append(req)

        return final

    # ------------------------------------------------------------------
    # Step 2: Build candidate nodes
    # ------------------------------------------------------------------

    def _build_candidate_nodes(
        self,
        requests: list[HTTPRequest],
        observation: SubtaskObservation,
    ) -> list[GraphNode]:
        """Build candidate GraphNodes from filtered HTTP requests.

        Uses CU action windows to group repeated endpoint calls by user intent,
        then collapses prefix chains (keystroke sequences) to the last/most-complete
        instance per chain per window.

        This correctly separates e.g. source auto-suggest (typed "Pune") from
        destination auto-suggest (typed "Bangalore") even though they hit the
        same endpoint.
        """
        from collections import defaultdict

        # --- Phase 1: Assign each request to a CU action window ---
        # A CU action window spans from one action's timestamp to the next.
        # Requests between two consecutive actions belong to the earlier action.
        # Requests before the first action or after the last get their own windows.
        actions = sorted(observation.cu_actions, key=lambda a: a.timestamp_ms)
        action_boundaries = [a.timestamp_ms for a in actions]

        def _get_action_window(req_ts: int) -> int:
            """Return the index of the CU action window this request belongs to.
            -1 means before any action (page-load side effects)."""
            if not action_boundaries:
                return -1
            for i in range(len(action_boundaries) - 1, -1, -1):
                if req_ts >= action_boundaries[i]:
                    return i
            return -1  # Before first action

        # --- Phase 2: Group requests by (action_window, fingerprint) ---
        # Each group represents one semantic use of an endpoint within one user action.
        groups: dict[tuple[int, str], list[tuple[int, HTTPRequest]]] = defaultdict(list)
        for i, req in enumerate(requests):
            fp = _compute_endpoint_fingerprint(req)
            window = _get_action_window(req.timestamp_ms)
            groups[(window, fp)].append((i, req))

        # --- Phase 3: Collapse prefix chains within each group ---
        # For keystroke auto-suggest (P, Pu, Pun, Pune), keep the last/most-complete.
        # For non-prefix groups, keep the request with the most non-empty params.
        representative_requests: list[tuple[int, HTTPRequest]] = []

        for (window, fp), group_reqs in sorted(groups.items()):
            if len(group_reqs) == 1:
                representative_requests.append(group_reqs[0])
                continue

            # Detect prefix chains: find the param that varies across requests
            # (e.g., searchString: P → Pu → Pun → Pune)
            varying_param = self._find_varying_param(group_reqs)

            if varying_param:
                # Collapse prefix chain — split into sub-chains and keep last of each
                reps = self._collapse_prefix_chains(group_reqs, varying_param)
                representative_requests.extend(reps)
            else:
                # No prefix chain — keep the request with most non-empty params (most complete)
                best = max(group_reqs, key=lambda ir: sum(
                    1 for _, v in _extract_request_params(ir[1]) if v
                ))
                representative_requests.append(best)

        # Sort by original request order
        representative_requests.sort(key=lambda ir: ir[0])

        # --- Phase 4: Build nodes from representative requests ---
        # Compute core vs optional params across ALL instances of each fingerprint
        all_groups: dict[str, list[HTTPRequest]] = defaultdict(list)
        for _, req in representative_requests:
            fp = _compute_endpoint_fingerprint(req)
            all_groups[fp].append(req)

        nodes = []
        node_to_request: dict[str, HTTPRequest] = {}  # for later use in edge building

        for idx, (orig_i, req) in enumerate(representative_requests):
            fingerprint = _compute_endpoint_fingerprint(req)
            url_template = _compute_url_template(req)

            # Core vs optional across all representatives with this fingerprint
            fp_reqs = all_groups[fingerprint]
            all_param_sets = []
            for greq in fp_reqs:
                param_names = {name for name, _ in _extract_request_params(greq)}
                all_param_sets.append(param_names)

            if all_param_sets:
                core_names = set.intersection(*all_param_sets)
                all_names = set.union(*all_param_sets)
                optional_names = all_names - core_names
            else:
                core_names = set()
                optional_names = set()

            params = _extract_request_params(req)
            param_map = {name: val for name, val in params}

            core_params = [
                ParameterSpec(name=name, role="website_generated", value_example=param_map.get(name, ""))
                for name in sorted(core_names) if name in param_map
            ]
            optional_params = [
                ParameterSpec(name=name, role="website_generated", value_example=param_map.get(name, ""))
                for name in sorted(optional_names) if name in param_map
            ]

            # Response schema
            response_schema = {}
            if req.response_body:
                try:
                    resp_data = json.loads(req.response_body)
                    response_schema = _json_schema_from_value(resp_data)
                except (json.JSONDecodeError, TypeError):
                    pass

            # CU reasoning — from the owning CU action window
            cu_reasoning = self._find_closest_cu_reasoning(req.timestamp_ms, observation.cu_actions)

            node_id = f"n{idx}"
            node = GraphNode(
                id=node_id,
                endpoint_fingerprint=fingerprint,
                http_method=req.method,
                url_template=url_template,
                request_type=req.request_type,
                core_parameters=core_params,
                optional_parameters=optional_params,
                response_schema=response_schema,
                response_extract_paths={},  # filled after edge building
                invocation=NodeInvocation(type="pending"),
                cu_reasoning_sample=cu_reasoning,
                observed_in_subtasks=[observation.subtask_id],
                example_request_body=req.body,
                example_response_body=(req.response_body or "")[:5000],
            )
            nodes.append(node)
            node_to_request[node_id] = req

        # Store mapping for edge building
        self._node_to_request = node_to_request

        return nodes

    def _find_varying_param(
        self, group_reqs: list[tuple[int, HTTPRequest]],
    ) -> Optional[str]:
        """Find the parameter that varies across requests in a same-fingerprint group.

        Returns the param name if exactly one param varies and the values look
        like keystroke progression (short strings that change each request).
        Returns None if no clear varying param.
        """
        if len(group_reqs) < 2:
            return None

        # Extract params for each request
        all_params: list[dict[str, str]] = []
        for _, req in group_reqs:
            param_map = {name: val for name, val in _extract_request_params(req)}
            all_params.append(param_map)

        # Find params that have different values across requests
        common_keys = set.intersection(*(set(pm.keys()) for pm in all_params)) if all_params else set()
        varying = []
        for key in common_keys:
            values = [pm[key] for pm in all_params]
            if len(set(values)) > 1:
                varying.append(key)

        # We expect exactly one varying param for keystroke sequences
        if len(varying) == 1:
            return varying[0]

        # If multiple vary, pick the one with the most distinct values
        # (likely the search term, not a timestamp or request ID)
        if varying:
            return max(varying, key=lambda k: len(set(pm.get(k, "") for pm in all_params)))

        return None

    def _collapse_prefix_chains(
        self,
        group_reqs: list[tuple[int, HTTPRequest]],
        varying_param: str,
    ) -> list[tuple[int, HTTPRequest]]:
        """Collapse keystroke prefix chains, keeping the last request per chain.

        A prefix chain is a sequence where each value is a prefix of the next
        (P → Pu → Pun → Pune). When the chain breaks (Pune → B), a new chain starts.
        Returns the last request from each chain.
        """
        if not group_reqs:
            return []

        # Sort by original order (request index)
        sorted_reqs = sorted(group_reqs, key=lambda ir: ir[0])

        chains: list[list[tuple[int, HTTPRequest]]] = [[sorted_reqs[0]]]

        for ir in sorted_reqs[1:]:
            _, req = ir
            params = {name: val for name, val in _extract_request_params(req)}
            curr_val = params.get(varying_param, "").lower()

            _, prev_req = chains[-1][-1]
            prev_params = {name: val for name, val in _extract_request_params(prev_req)}
            prev_val = prev_params.get(varying_param, "").lower()

            # Check if current is a prefix extension of previous
            if curr_val.startswith(prev_val) or prev_val.startswith(curr_val):
                chains[-1].append(ir)
            else:
                # New chain
                chains.append([ir])

        # Take the last (most complete) request from each chain
        return [chain[-1] for chain in chains]

    def _find_closest_cu_reasoning(self, request_ts_ms: int, actions: list[CUAction]) -> str:
        """Find CU action reasoning closest in time before a request (within 2 seconds)."""
        best = ""
        best_delta = float("inf")
        for action in actions:
            delta = request_ts_ms - action.timestamp_ms
            if 0 <= delta <= 2000 and delta < best_delta:
                best = action.cu_reasoning
                best_delta = delta
        return best

    # ------------------------------------------------------------------
    # Step 3: Build edges (value tracing)
    # ------------------------------------------------------------------

    # Values that appear everywhere and should never be chained
    _TRIVIAL_VALUES = frozenset({
        "true", "false", "null", "none", "undefined",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "yes", "no", "ok", "error", "success",
        "GET", "POST", "PUT", "DELETE",
        "asc", "desc", "ASC", "DESC",
        "DEFAULT",
    })

    def _build_chain_candidates(
        self,
        nodes: list[GraphNode],
    ) -> dict[tuple[str, str], list[dict]]:
        """Find chain candidates by exact value matching across prior node responses.

        For each (node_id, param_name), returns a list of candidate sources:
          [{"source_node_id": str, "source_field": str, "source_value": str}, ...]

        Only looks at earlier nodes (strict temporal ordering).
        Skips trivial values (booleans, short strings, common constants).
        """
        # Parse responses for all nodes
        node_responses: dict[str, Any] = {}
        for node in nodes:
            req = self._node_to_request.get(node.id)
            if req and req.response_body:
                try:
                    node_responses[node.id] = json.loads(req.response_body)
                except (json.JSONDecodeError, TypeError):
                    pass

        candidates: dict[tuple[str, str], list[dict]] = {}

        for j, target_node in enumerate(nodes):
            for param in target_node.core_parameters:
                val = str(param.value_example)
                key = (target_node.id, param.name)
                candidates[key] = []

                # Skip trivial values that would create false edges
                if len(val) < 2:
                    continue
                if val.lower() in self._TRIVIAL_VALUES:
                    continue

                # Find all exact matches in earlier nodes' responses
                for i, source_node in enumerate(nodes):
                    if i >= j:
                        break
                    resp_data = node_responses.get(source_node.id)
                    if resp_data is None:
                        continue

                    # Find ALL matching paths (not just first)
                    paths = _find_all_values_jsonpath(resp_data, val)
                    for jp in paths:
                        # Extract the field name from the jsonpath for name comparison
                        candidates[key].append({
                            "source_node_id": source_node.id,
                            "source_field": jp,
                            "source_value": val,
                            "source_endpoint": source_node.endpoint_fingerprint,
                        })

        return candidates

    # ------------------------------------------------------------------
    # Step 4: Classify parameter roles via single bundled LLM call
    # ------------------------------------------------------------------

    def _classify_parameters(
        self,
        nodes: list[GraphNode],
        chain_candidates: dict[tuple[str, str], list[dict]],
        observation: SubtaskObservation,
    ) -> tuple[list[GraphEdge], list[GraphNode]]:
        """Classify all params via one LLM call. Returns (edges, updated_nodes).

        The LLM receives chain candidates (from exact value matching),
        task/action context, existing graph context, and profile history.
        It classifies each param as chained, user_intent, or website_generated.
        Edges are built from chained classifications.
        """
        # Build CU action summary for context
        action_summaries = []
        for action in observation.cu_actions:
            parts = [action.action_type]
            if action.typed_value:
                parts.append(f'"{action.typed_value}"')
            if action.target_text:
                parts.append(f"on {action.target_text}")
            action_summaries.append(" ".join(parts))
        actions_text = "; ".join(action_summaries) if action_summaries else "no actions recorded"

        # Build existing graph context (prior graph names/descriptions from registry)
        existing_graphs = list_graphs(observation.site)
        prior_graph_context = ""
        if existing_graphs:
            lines = []
            for g in existing_graphs[:5]:  # Cap at 5 to keep prompt size manageable
                lines.append(f"  - {g.name}: {g.capability_statement}")
            prior_graph_context = "Prior known workflows on this site:\n" + "\n".join(lines)

        # Load profile history for param values across tasks
        profile = self._load_profile(observation.site)

        # Build the params list for the LLM
        params_to_classify = []
        param_index: list[tuple[str, str]] = []  # maps LLM output index → (node_id, param_name)

        for node in nodes:
            # Determine CU action context for this node
            cu_ctx = node.cu_reasoning_sample or ""
            if not cu_ctx:
                req = self._node_to_request.get(node.id)
                if req:
                    cu_ctx = self._find_closest_cu_reasoning(req.timestamp_ms, observation.cu_actions)

            for param in node.core_parameters:
                val = str(param.value_example)
                key = (node.id, param.name)
                candidates = chain_candidates.get(key, [])

                # Infer data type from value
                data_type = "string"
                if val.lower() in ("true", "false"):
                    data_type = "boolean"
                elif val.replace("-", "").replace("/", "").isdigit() and len(val) >= 8:
                    data_type = "date"
                elif val.isdigit():
                    data_type = "integer"

                # Fetch profile history for this endpoint+param
                fp = node.endpoint_fingerprint
                history = profile.get(fp, {}).get(param.name, [])

                # Format chain candidates for the LLM (cap at 5 per param)
                formatted_candidates = []
                for c in candidates[:5]:
                    formatted_candidates.append({
                        "source_node_id": c["source_node_id"],
                        "source_endpoint": c["source_endpoint"],
                        "source_field": c["source_field"],
                        "source_value": c["source_value"],
                    })

                params_to_classify.append({
                    "node_id": node.id,
                    "node_endpoint": node.endpoint_fingerprint,
                    "param_name": param.name,
                    "param_value": val,
                    "data_type": data_type,
                    "is_required": True,  # core params are always required
                    "chain_candidates": formatted_candidates,
                    "cu_action_context": cu_ctx,
                    "profile_history": history[-10:],  # last 10 values
                })
                param_index.append(key)

        if not params_to_classify:
            return [], nodes

        # Build LLM prompt
        user_prompt = json.dumps({
            "task": observation.subtask_description,
            "actions_performed": actions_text,
            "prior_graph_context": prior_graph_context,
            "nodes_summary": [
                {"id": n.id, "endpoint": n.endpoint_fingerprint, "method": n.http_method}
                for n in nodes
            ],
            "params_to_classify": params_to_classify,
        }, indent=2)

        schema = _build_classify_schema(params_to_classify)

        try:
            result = call_gemini(
                model="gemini-3-flash-preview",
                contents=[user_prompt],
                response_schema=schema,
                system_instruction=_CLASSIFY_SYSTEM_PROMPT,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 8192,
                    "thinking_config": genai_types.ThinkingConfig(thinking_budget=4096),
                },
            )
        except Exception as exc:
            logger.error("Parameter classification LLM call failed: %s", exc)
            # Fallback: leave everything as website_generated (the default)
            return [], nodes

        # Parse classifications and apply to nodes
        classifications = result.get("classifications", [])
        edges = []

        # Build lookup for fast access
        node_lookup = {n.id: n for n in nodes}

        for cls in classifications:
            node_id = cls.get("node_id", "")
            param_name = cls.get("param_name", "")
            role = cls.get("role", "website_generated")

            node = node_lookup.get(node_id)
            if not node:
                continue

            # Find the param on this node
            param = None
            for p in node.core_parameters:
                if p.name == param_name:
                    param = p
                    break
            if not param:
                continue

            if role == "chained":
                source_node_id = cls.get("chained_from_node_id") or ""
                source_field = cls.get("chained_from_field") or ""

                if source_node_id and source_field:
                    # Validate that the source actually exists in our candidates
                    key = (node_id, param_name)
                    valid = any(
                        c["source_node_id"] == source_node_id and c["source_field"] == source_field
                        for c in chain_candidates.get(key, [])
                    )
                    if valid:
                        param.role = "chained"
                        param.chained_from = f"{source_node_id}.{source_field}"

                        # Detect array edges: path contains [N] index
                        array_match = re.search(r'\[(\d+)\]', source_field)
                        if array_match:
                            array_path = source_field[:array_match.start()]
                            item_field = source_field[array_match.end():].lstrip('.')
                            edges.append(GraphEdge(
                                from_node_id=source_node_id,
                                to_node_id=node_id,
                                from_extract=source_field,
                                to_parameter=param_name,
                                requires_selection=True,
                                selection_array_path=array_path,
                                selection_item_field=item_field,
                            ))
                        else:
                            edges.append(GraphEdge(
                                from_node_id=source_node_id,
                                to_node_id=node_id,
                                from_extract=source_field,
                                to_parameter=param_name,
                            ))

                        # Update response_extract_paths on the source node
                        source_node = node_lookup.get(source_node_id)
                        if source_node:
                            extract_name = param_name.replace(".", "_")
                            source_node.response_extract_paths[extract_name] = source_field
                    else:
                        logger.warning(
                            "LLM hallucinated chain: %s.%s from %s.%s — not in candidates, defaulting to website_generated",
                            node_id, param_name, source_node_id, source_field,
                        )
                        param.role = "website_generated"
                else:
                    param.role = "website_generated"

            elif role == "user_intent":
                param.role = "user_intent"
                param.cu_action_binding = {"source": "llm_classification", "reasoning": cls.get("reasoning", "")}

            else:
                param.role = "website_generated"

            logger.debug("Classified %s.%s → %s (%s)", node_id, param_name, param.role, cls.get("reasoning", "")[:80])

        return edges, nodes

    # ------------------------------------------------------------------
    # Step 4b: Build pushState/replaceState navigation nodes
    # ------------------------------------------------------------------

    def _build_navigation_nodes(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        observation: SubtaskObservation,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Create graph nodes from captured pushState/replaceState events.

        For each navigation event, templatize the URL by replacing known
        chained and user_intent values with their chain references. Build
        edges from the source nodes that produced those values.
        """
        if not observation.navigation_events:
            return [], []

        # Build a lookup of all known values → their sources
        # From chained params: value → (node_id, chain_source_ref)
        # From user_intent params: value → param_name
        chained_values: dict[str, tuple[str, str]] = {}  # value → (source_node_id, extract_path)
        user_intent_values: dict[str, str] = {}  # value → param_name

        for edge in edges:
            # Find the target param's example value
            for node in nodes:
                for param in node.core_parameters:
                    if param.name == edge.to_parameter and param.chained_from:
                        val = str(param.value_example)
                        if len(val) >= 2:
                            chained_values[val] = (edge.from_node_id, edge.from_extract)
                        break

        for node in nodes:
            for param in node.core_parameters:
                if param.role == "user_intent":
                    val = str(param.value_example)
                    if len(val) >= 2:
                        user_intent_values[val] = param.name

        nav_nodes: list[GraphNode] = []
        nav_edges: list[GraphEdge] = []
        next_id = len(nodes)

        for nav_evt in observation.navigation_events:
            url = nav_evt.url
            url_template = url
            nav_params: list[ParameterSpec] = []
            evt_edges: list[GraphEdge] = []
            node_id = f"n{next_id}"

            # Replace chained values in the URL — longest match first to avoid partial replacement
            for val in sorted(chained_values.keys(), key=len, reverse=True):
                if val in url_template:
                    source_node_id, extract_path = chained_values[val]
                    # Derive param name from the extract path (e.g., $.data.stationList[0].stationCode → stationCode)
                    param_name = extract_path.rsplit(".", 1)[-1] if "." in extract_path else extract_path
                    # Disambiguate if same param name used multiple times
                    existing_names = {p.name for p in nav_params}
                    if param_name in existing_names:
                        # Find which edge this is — source vs destination
                        for enode in nodes:
                            for ep in enode.core_parameters:
                                if ep.role == "chained" and str(ep.value_example) == val and ep.name != param_name:
                                    param_name = ep.name
                                    break
                    if param_name in existing_names:
                        param_name = f"{param_name}_{source_node_id}"

                    url_template = url_template.replace(val, f"${{{param_name}}}")
                    nav_params.append(ParameterSpec(
                        name=param_name,
                        role="chained",
                        value_example=val,
                        chained_from=f"{source_node_id}.{extract_path}",
                    ))
                    # Detect array edges
                    array_match = re.search(r'\[(\d+)\]', extract_path)
                    if array_match:
                        evt_edges.append(GraphEdge(
                            from_node_id=source_node_id,
                            to_node_id=node_id,
                            from_extract=extract_path,
                            to_parameter=param_name,
                            requires_selection=True,
                            selection_array_path=extract_path[:array_match.start()],
                            selection_item_field=extract_path[array_match.end():].lstrip('.'),
                        ))
                    else:
                        evt_edges.append(GraphEdge(
                            from_node_id=source_node_id,
                            to_node_id=node_id,
                            from_extract=extract_path,
                            to_parameter=param_name,
                        ))

            # Replace user_intent values in the URL — but as CHAINED from the
            # upstream node that has that user_intent param, not as new user_intent.
            # pushState URLs are pure derivatives of API params/responses — the SPA
            # builds them, we just observe. No param generation needed for pushState.
            for val in sorted(user_intent_values.keys(), key=len, reverse=True):
                if val in url_template:
                    param_name = user_intent_values[val]
                    url_template = url_template.replace(val, f"${{{param_name}}}")
                    # Find which upstream node has this as user_intent
                    source_node_id = None
                    for src_node in nodes:
                        for sp in src_node.core_parameters:
                            if sp.role == "user_intent" and sp.name == param_name and str(sp.value_example) == val:
                                source_node_id = src_node.id
                                break
                        if source_node_id:
                            break
                    if source_node_id:
                        nav_params.append(ParameterSpec(
                            name=param_name,
                            role="chained",
                            value_example=val,
                            chained_from=f"{source_node_id}.params.{param_name}",
                        ))
                        evt_edges.append(GraphEdge(
                            from_node_id=source_node_id,
                            to_node_id=node_id,
                            from_extract=f"params.{param_name}",
                            to_parameter=param_name,
                        ))

            # Only create a node if the URL had dynamic values (otherwise it's a static navigation)
            if not nav_params:
                continue

            # Build the pushState expression
            expression = f"history.pushState({{}}, '', \"{url_template}\")"

            nav_node = GraphNode(
                id=node_id,
                endpoint_fingerprint=f"pushState {url_template}",
                http_method="NAVIGATE",
                url_template=url_template,
                request_type="pushstate",
                core_parameters=nav_params,
                optional_parameters=[],
                response_schema={},
                response_extract_paths={},
                invocation=NodeInvocation(type="pushstate", expression=expression),
                cu_reasoning_sample="SPA client-side navigation captured from pushState",
                observed_in_subtasks=[observation.subtask_id],
            )
            nav_nodes.append(nav_node)
            nav_edges.extend(evt_edges)
            next_id += 1

        return nav_nodes, nav_edges

    # ------------------------------------------------------------------
    # Step 5: Identify entry points (5-strategy fallback)
    # ------------------------------------------------------------------

    async def _identify_entry_points(
        self,
        nodes: list[GraphNode],
        observation: SubtaskObservation,
    ) -> list[GraphNode]:
        """For each node, determine how to invoke it via CDP.

        Tries 5 strategies in order:
        A. Reachable global function
        B. Framework action dispatch
        C. Extracted function IIFE
        D. DOM action replay
        E. Give up (discard node)

        Returns only nodes that got a valid entry point.
        """
        surviving = []

        for node in nodes:
            # pushState nodes already have their invocation set — pass through
            if node.invocation and node.invocation.expression:
                surviving.append(node)
                continue

            invocation = await self._find_entry_point(node, observation)
            if invocation:
                node.invocation = invocation
                surviving.append(node)
            else:
                logger.debug("No entry point found for node %s (%s) — discarding",
                             node.id, node.endpoint_fingerprint)

        return surviving

    async def _find_entry_point(
        self,
        node: GraphNode,
        observation: SubtaskObservation,
    ) -> Optional[NodeInvocation]:
        """Try each entry point strategy in order for a single node."""
        # Get the HTTP request that corresponds to this node
        matching_req = self._node_to_request.get(node.id)
        if not matching_req:
            for req in observation.http_requests:
                if _compute_endpoint_fingerprint(req) == node.endpoint_fingerprint:
                    matching_req = req
                    break

        if not matching_req:
            return None

        # Strategy A: Reachable global function
        invocation = await self._try_reachable_global(node, matching_req, observation)
        if invocation:
            logger.debug("Node %s: Strategy A (reachable global) succeeded", node.id)
            return invocation

        # Strategy B: Framework action dispatch
        invocation = await self._try_framework_dispatch(node, matching_req, observation)
        if invocation:
            logger.debug("Node %s: Strategy B (framework dispatch) succeeded", node.id)
            return invocation

        # Strategy C: Raw fetch replay — use captured URL template + headers directly.
        # Preferred over extracted functions for REST APIs because minified bundle
        # functions depend on module-scoped closures that break when evaluated standalone.
        invocation = self._try_fetch_replay(node, matching_req)
        if invocation:
            logger.debug("Node %s: Strategy C (fetch replay) succeeded", node.id)
            return invocation

        # Strategy D: Extracted function IIFE (fallback for non-REST protocols)
        invocation = await self._try_extracted_function(node, matching_req, observation)
        if invocation:
            logger.debug("Node %s: Strategy D (extracted function) succeeded", node.id)
            return invocation

        # Strategy E: DOM action replay
        invocation = self._try_dom_replay(node, matching_req, observation)
        if invocation:
            logger.debug("Node %s: Strategy E (DOM replay) succeeded", node.id)
            return invocation

        # Strategy F: Give up
        logger.debug("Node %s: All strategies failed", node.id)
        return None

    async def _try_reachable_global(
        self,
        node: GraphNode,
        req: HTTPRequest,
        observation: SubtaskObservation,
    ) -> Optional[NodeInvocation]:
        """Strategy A: Find a reachable window.* function that produces this request.

        Walk the initiator stack to find function names, then search window.*
        up to depth 4 for a matching function.
        """
        if not req.initiator_stack:
            return None

        page = self._session.page
        if not page or page.is_closed():
            return None

        # Extract function names from the stack
        stack_functions = []
        for frame in req.initiator_stack:
            fn_name = frame.get("functionName", "")
            if fn_name and fn_name not in ("", "anonymous", "(anonymous)"):
                stack_functions.append(fn_name)

        if not stack_functions:
            return None

        # Search window.* up to depth 4 for a function matching any stack function
        try:
            result = await page.evaluate("""(targetNames) => {
                const found = [];
                const visited = new WeakSet();

                function walk(obj, path, depth) {
                    if (depth > 4 || !obj || visited.has(obj)) return;
                    try { visited.add(obj); } catch(e) { return; }

                    for (const key of Object.getOwnPropertyNames(obj)) {
                        if (key.startsWith('__') || key === 'constructor') continue;
                        try {
                            const val = obj[key];
                            if (typeof val === 'function') {
                                const fullPath = path + '.' + key;
                                if (targetNames.includes(key) || targetNames.includes(val.name)) {
                                    found.push({
                                        path: fullPath,
                                        name: val.name || key,
                                        argCount: val.length,
                                    });
                                }
                            } else if (typeof val === 'object' && val !== null) {
                                walk(val, path + '.' + key, depth + 1);
                            }
                        } catch(e) {}
                    }
                }

                walk(window, 'window', 0);
                return found;
            }""", stack_functions)

            if result and len(result) > 0:
                # Use the first match
                match = result[0]
                global_path = match["path"]

                # Build invocation expression with parameter placeholders
                param_args = {}
                for param in node.core_parameters:
                    param_args[param.name] = f"${{params.{param.name}}}"

                # Construct the call expression
                args_json = json.dumps(param_args)
                expression = f"await {global_path}({args_json})"

                return NodeInvocation(
                    type="cdp_eval_global",
                    expression=expression,
                    await_promise=True,
                )
        except Exception as exc:
            logger.debug("Strategy A failed for node %s: %s", node.id, exc)

        return None

    async def _try_framework_dispatch(
        self,
        node: GraphNode,
        req: HTTPRequest,
        observation: SubtaskObservation,
    ) -> Optional[NodeInvocation]:
        """Strategy B: Framework-specific action dispatch.

        Tries Redux store.dispatch, Apollo client.query/mutate, React Query.
        """
        frameworks = observation.framework_fingerprint.get("frameworks", [])
        page = self._session.page
        if not page or page.is_closed():
            return None

        # Redux dispatch
        if any(f in frameworks for f in ("redux", "redux-store")):
            invocation = await self._try_redux_dispatch(node, req, page)
            if invocation:
                return invocation

        # Apollo Client
        if "apollo" in frameworks:
            invocation = await self._try_apollo_dispatch(node, req, page)
            if invocation:
                return invocation

        return None

    async def _try_redux_dispatch(
        self,
        node: GraphNode,
        req: HTTPRequest,
        page: Any,
    ) -> Optional[NodeInvocation]:
        """Try to find a Redux action creator that triggers this HTTP request."""
        try:
            # Check if store.dispatch is available
            has_store = await page.evaluate("typeof window.store === 'object' && typeof window.store.dispatch === 'function'")
            if not has_store:
                return None

            # Look for thunk patterns in the stack
            for frame in req.initiator_stack:
                fn_name = frame.get("functionName", "")
                if not fn_name:
                    continue
                # Common Redux thunk patterns: fetchX, loadX, getX, searchX
                if any(fn_name.startswith(prefix) for prefix in ("fetch", "load", "get", "search", "post", "create", "update", "delete")):
                    # Try to find this function in window scope
                    try:
                        exists = await page.evaluate(f"typeof window.{fn_name} === 'function'")
                        if exists:
                            param_args = {p.name: f"${{params.{p.name}}}" for p in node.core_parameters}
                            args_str = json.dumps(param_args)
                            expression = f"await window.store.dispatch(window.{fn_name}({args_str}))"
                            return NodeInvocation(
                                type="cdp_eval_redux",
                                expression=expression,
                                await_promise=True,
                            )
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("Redux dispatch failed: %s", exc)
        return None

    async def _try_apollo_dispatch(
        self,
        node: GraphNode,
        req: HTTPRequest,
        page: Any,
    ) -> Optional[NodeInvocation]:
        """Try to invoke via Apollo Client for GraphQL operations."""
        if req.request_type != "graphql":
            return None

        try:
            has_apollo = await page.evaluate("typeof window.__APOLLO_CLIENT__ === 'object'")
            if not has_apollo:
                return None

            op_name = req.graphql_operation_name or ""
            if not op_name:
                return None

            # Determine query vs mutation
            is_mutation = False
            if req.body:
                try:
                    body = json.loads(req.body)
                    query_str = body.get("query", "") or body.get("mutation", "")
                    is_mutation = query_str.strip().startswith("mutation")
                except Exception:
                    pass

            # Build variables from parameters
            variables = {p.name: f"${{params.{p.name}}}" for p in node.core_parameters}

            if is_mutation:
                expression = f"""await window.__APOLLO_CLIENT__.mutate({{
                    mutation: gql`{req.body and json.loads(req.body).get('query', '')}`,
                    variables: {json.dumps(variables)}
                }})"""
            else:
                expression = f"""await window.__APOLLO_CLIENT__.query({{
                    query: gql`{req.body and json.loads(req.body).get('query', '')}`,
                    variables: {json.dumps(variables)}
                }})"""

            return NodeInvocation(
                type="cdp_eval_apollo" if not is_mutation else "cdp_eval_apollo",
                expression=expression,
                await_promise=True,
            )
        except Exception as exc:
            logger.debug("Apollo dispatch failed: %s", exc)
        return None

    async def _try_extracted_function(
        self,
        node: GraphNode,
        req: HTTPRequest,
        observation: SubtaskObservation,
    ) -> Optional[NodeInvocation]:
        """Strategy C: Extract the non-library function closest to fetch and wrap as IIFE.

        Find the frame in the initiator stack that's closest to the actual fetch call
        but isn't from a library. Extract its source, check self-containment, wrap as IIFE.
        """
        if not req.initiator_stack:
            return None

        # Find non-library frames (skip common library paths)
        _LIBRARY_PATTERNS = ("node_modules", "webpack", "chunk-", "vendor", "polyfill", "runtime")

        for frame in req.initiator_stack:
            url = frame.get("url", "")
            fn_name = frame.get("functionName", "")

            # Skip library frames
            if any(pat in url.lower() for pat in _LIBRARY_PATTERNS):
                continue
            if not fn_name or fn_name in ("anonymous", "(anonymous)"):
                continue

            script_id = frame.get("scriptId", "")
            if not script_id:
                continue

            # Try to get the script source
            script = observation.scripts.get(script_id)
            if not script:
                continue

            line_num = frame.get("lineNumber", 0)
            col_num = frame.get("columnNumber", 0)

            # Extract function around the given location
            func_source = self._extract_function_at_location(
                script.source, line_num, col_num,
            )
            if not func_source:
                continue

            # Check self-containment: no free variables beyond arguments, window, globals
            if not self._is_self_contained(func_source):
                continue

            # Build IIFE invocation
            param_args = {p.name: f"${{params.{p.name}}}" for p in node.core_parameters}
            args_json = json.dumps(param_args)
            expression = f"await (({func_source})({args_json}))"

            return NodeInvocation(
                type="cdp_eval_extracted",
                expression=expression,
                extracted_function_source=func_source,
                await_promise=True,
            )

        return None

    def _extract_function_at_location(self, source: str, line: int, col: int) -> Optional[str]:
        """Extract a function body from source code at a specific location.

        Naive approach: find the nearest function keyword and extract its body
        by brace matching. This handles most common patterns.
        """
        lines = source.split("\n")
        if line >= len(lines):
            return None

        # Search backward from the line for function declaration
        search_start = max(0, line - 5)
        search_end = min(len(lines), line + 50)
        region = "\n".join(lines[search_start:search_end])

        # Find function patterns
        func_patterns = [
            r'((?:async\s+)?function\s+\w+\s*\([^)]*\)\s*\{)',
            r'((?:async\s+)?\([^)]*\)\s*=>\s*\{)',
            r'((?:async\s+)?\w+\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)\s*[({])',
        ]

        for pattern in func_patterns:
            match = re.search(pattern, region)
            if match:
                start_pos = match.start()
                # Find matching closing brace
                brace_count = 0
                func_start = None
                for i in range(start_pos, len(region)):
                    if region[i] == "{":
                        if func_start is None:
                            func_start = start_pos
                        brace_count += 1
                    elif region[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return region[start_pos:i + 1]

        return None

    def _is_self_contained(self, func_source: str) -> bool:
        """Check if a function body only references its arguments, window, and primitive globals.

        Conservative check: if the function references common closure patterns
        (this., _private, imported symbols), consider it NOT self-contained.
        """
        # Indicators of closure dependency
        closure_patterns = [
            r'\bthis\.',          # References `this` context
            r'\b_[a-z]',          # Private-style variables
            r'\bimport\b',        # Import statements
            r'\brequire\b',       # CommonJS requires
            r'\bmodule\.',        # Module references
            r'\bexports\.',       # Export references
        ]
        for pattern in closure_patterns:
            if re.search(pattern, func_source):
                return False

        # Allow: window, document, fetch, console, JSON, Promise, URL, etc.
        return True

    def _try_dom_replay(
        self,
        node: GraphNode,
        req: HTTPRequest,
        observation: SubtaskObservation,
    ) -> Optional[NodeInvocation]:
        """Strategy D: Find the CU action that triggered this request and build a DOM replay."""
        # Find the closest CU action before this request
        closest_action = None
        closest_delta = float("inf")

        for action in observation.cu_actions:
            delta = req.timestamp_ms - action.timestamp_ms
            if 0 <= delta <= 5000 and delta < closest_delta:
                closest_action = action
                closest_delta = delta

        if not closest_action:
            return None
        if not closest_action.target_selector:
            return None

        # Map action type to DOM event
        event_map = {
            "click": "click",
            "type": "input",
            "select": "change",
            "scroll": "scroll",
        }
        event_type = event_map.get(closest_action.action_type, "click")

        # Build a DOM replay expression
        selector = closest_action.target_selector
        if closest_action.action_type == "type" and closest_action.typed_value:
            # For type actions: set value then dispatch input event
            expression = f"""(() => {{
                const el = document.querySelector('{selector}');
                if (!el) throw new Error('Element not found: {selector}');
                el.value = '${{params.{node.core_parameters[0].name if node.core_parameters else "value"}}}';
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }})()"""
        else:
            expression = f"""(() => {{
                const el = document.querySelector('{selector}');
                if (!el) throw new Error('Element not found: {selector}');
                el.dispatchEvent(new MouseEvent('{event_type}', {{ bubbles: true, cancelable: true }}));
            }})()"""

        return NodeInvocation(
            type="dom_replay",
            expression=expression,
            dom_target_selector=selector,
            dom_event_type=event_type,
            await_promise=False,
        )

    def _try_fetch_replay(
        self,
        node: GraphNode,
        req: HTTPRequest,
    ) -> Optional[NodeInvocation]:
        """Strategy E: Raw fetch replay using captured URL template + headers.

        Most reliable for REST/GraphQL APIs where the original JS uses
        module-scoped fetch wrappers that can't be extracted or reached as globals.
        Builds a self-contained fetch() call from the node's URL template and
        the request's captured headers.
        """
        if not node.url_template:
            return None

        # Build header object from captured request (exclude pseudo-headers)
        headers = {
            k: v for k, v in (req.headers or {}).items()
            if not k.startswith(":") and k.lower() not in ("host", "content-length")
        }
        headers_json = json.dumps(headers)

        # Build URL with parameter placeholders
        # url_template already has {param_name} placeholders from node building.
        # Convert {param} → ${params.param} for executor's _substitute_params.
        url_expr = json.dumps(node.url_template)
        for param in node.core_parameters:
            url_expr = url_expr.replace(
                f"{{{param.name}}}",
                f"${{params.{param.name}}}",
            )

        if req.method == "GET":
            expression = f"""await fetch({url_expr}, {{
                method: 'GET',
                headers: {headers_json},
            }}).then(r => r.json())"""
        else:
            body_json = json.dumps(req.body or "{}")
            # Convert {param} → ${params.param} in body
            for param in node.core_parameters:
                body_json = body_json.replace(
                    f"{{{param.name}}}",
                    f"${{params.{param.name}}}",
                )
            expression = f"""await fetch({url_expr}, {{
                method: '{req.method}',
                headers: {headers_json},
                body: {body_json},
            }}).then(r => r.json())"""

        return NodeInvocation(
            type="fetch_replay",
            expression=expression,
            await_promise=True,
        )

    # ------------------------------------------------------------------
    # Step 6: Verify read-only graphs
    # ------------------------------------------------------------------

    async def _verify_invocations(
        self,
        nodes: list[GraphNode],
        observation: SubtaskObservation,
    ) -> bool:
        """Verify a read-only graph by re-invoking entry points on the CURRENT page.

        Entry points were discovered on the live SPA page. Verification must
        run on the same page state — navigating to start_url would blow away
        the SPA (many sites load AMP/SSR first, then hydrate to SPA after
        interaction). We verify in-place: re-invoke each node's entry point
        with captured values, intercept the resulting HTTP request, and compare
        endpoint fingerprint + body key structure against the original.
        """
        page = self._session.page
        if not page or page.is_closed():
            return False

        # Do NOT navigate away — verify on the current (SPA) page state
        # where the entry points were discovered.

        for node in nodes:
            inv = node.invocation
            if not inv.expression:
                continue

            logger.info("Verify node %s: type=%s, expression=%s", node.id, inv.type, inv.expression[:120])

            # Step A: Check global path existence for CDP eval types
            if inv.type in ("cdp_eval_global", "cdp_eval_redux", "cdp_eval_apollo"):
                match = re.search(r'(window(?:\.\w+)+)', inv.expression)
                if match:
                    path = match.group(1)
                    try:
                        exists = await page.evaluate(f"typeof {path} !== 'undefined'")
                        if not exists:
                            logger.info("Verification FAIL Step A: %s does not exist for node %s", path, node.id)
                            return False
                    except Exception as exc:
                        logger.info("Verification FAIL Step A: exception for node %s: %s", node.id, exc)
                        return False

            # Step B: Re-invoke with captured example values and intercept HTTP
            example_params = {p.name: p.value_example for p in node.core_parameters if p.value_example is not None}
            expression = inv.expression
            for pname, pval in example_params.items():
                # Replace ${params.X} placeholders with actual values
                expression = expression.replace(f"${{params.{pname}}}", json.dumps(pval) if not isinstance(pval, str) else pval)

            captured_requests: list[dict] = []

            async def _intercept_request(request):
                captured_requests.append({
                    "url": request.url,
                    "method": request.method,
                    "post_data": request.post_data,
                })

            try:
                page.on("request", _intercept_request)
                result = await page.evaluate(f"""async () => {{
                    try {{ return await (async () => {{ return {expression}; }})(); }}
                    catch(e) {{ return {{ __error: e.message }}; }}
                }}""")
                if isinstance(result, dict) and "__error" in result:
                    logger.info("Verification FAIL Step B: JS error for node %s: %s", node.id, result["__error"])
                # Brief wait for async requests to fire
                await page.wait_for_timeout(1500)
            except Exception as exc:
                logger.info("Verification FAIL Step B: exception for node %s: %s", node.id, str(exc)[:200])
                page.remove_listener("request", _intercept_request)
                return False
            finally:
                try:
                    page.remove_listener("request", _intercept_request)
                except Exception:
                    pass

            # Step C: Compare intercepted request to original
            logger.info("Verify node %s Step C: captured %d requests", node.id, len(captured_requests))
            if not captured_requests:
                logger.info("Verification FAIL Step C: no HTTP requests for node %s (type=%s)", node.id, inv.type)
                # DOM replay nodes may not produce HTTP — that's acceptable
                if inv.type == "dom_replay":
                    continue
                return False

            # Check that at least one captured request matches the expected fingerprint
            expected_fp = node.endpoint_fingerprint
            matched = False
            for creq in captured_requests:
                mock_req = HTTPRequest(
                    timestamp_ms=0, subtask_id="", request_id="", url=creq["url"],
                    method=creq["method"], headers={}, body=creq.get("post_data"),
                    request_type=node.request_type, graphql_operation_name=None,
                    graphql_query_hash=None, jsonrpc_method=None, response_status=0,
                    response_headers={}, response_body=None, response_time_ms=0,
                    initiator_stack=[], initiator_type="",
                )
                captured_fp = _compute_endpoint_fingerprint(mock_req)
                if captured_fp == expected_fp:
                    # Fingerprint matches — now compare body key structure
                    if creq.get("post_data") and node.example_request_body:
                        try:
                            captured_keys = sorted(json.loads(creq["post_data"]).keys())
                            expected_keys = sorted(json.loads(node.example_request_body).keys())
                            if captured_keys != expected_keys:
                                logger.debug("Verification: body key mismatch for node %s: %s vs %s",
                                             node.id, captured_keys, expected_keys)
                                return False
                        except (json.JSONDecodeError, TypeError, AttributeError):
                            pass  # Non-JSON bodies — fingerprint match is sufficient
                    matched = True
                    break

            if not matched:
                logger.debug("Verification: no captured request matches fingerprint %s for node %s",
                             expected_fp, node.id)
                return False

        logger.info("Graph verification passed (HTTP re-invocation validated)")
        return True

    async def _verify_pipeline(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        observation: SubtaskObservation,
    ) -> bool:
        """Full pipeline verification: real task → intent extraction → execute → validate.

        Replays the actual task that produced this graph through the full
        executor pipeline: param extraction, chaining, array selection, execution.
        If it can't replay the original task successfully, the graph is broken.
        """
        from morphnet.executor import Executor

        PROMPTS_DIR = Path(__file__).parent / "prompts"

        task = observation.subtask_description
        if not task:
            logger.info("Pipeline verification skipped: no task description")
            return True

        # Build a temporary Graph object
        graph_id = compute_graph_id(nodes, edges)
        temp_graph = Graph(
            id=graph_id,
            site=observation.site,
            name="verification_temp",
            description="Temporary graph for pipeline verification",
            capability_statement=task,
            reason_for_version="verification",
            parent_graph_ids=[],
            nodes=list(nodes),
            edges=list(edges),
            terminal_node_ids=[],
            completion={},
            preconditions={},
            verified=False,
        )

        # Step 1: Per-node intent extraction from the real task
        param_descriptors: list[dict] = []
        for node in nodes:
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
            logger.info("Pipeline verification skipped: no user_intent parameters")
            return True

        properties = {}
        for pd in param_descriptors:
            properties[pd["key"]] = {
                "type": "string",
                "description": (
                    f"Value for '{pd['param_name']}' in step: {pd['node_description']}. "
                    f"Example: {pd['example_value']}"
                ),
            }

        extraction_prompt = (
            f"Subtask: {task}\n\n"
            f"Graph capability: {temp_graph.capability_statement}\n\n"
            f"Parameters to extract (each belongs to a specific workflow step):\n"
            + "\n".join(
                f"  - {pd['key']}: {pd['node_description']} → param '{pd['param_name']}' (example: {pd['example_value']})"
                for pd in param_descriptors
            )
        )

        intent_prompt_path = PROMPTS_DIR / "intent_extraction.txt"
        intent_system = intent_prompt_path.read_text(encoding="utf-8") if intent_prompt_path.exists() else ""

        try:
            raw_intent = call_gemini(
                model="gemini-3-flash-preview",
                contents=[extraction_prompt],
                response_schema={"type": "object", "properties": properties, "required": list(properties.keys())},
                system_instruction=intent_system,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            )
        except Exception as exc:
            logger.warning("Pipeline verification: intent extraction failed: %s", exc)
            return True  # Don't block on LLM failure

        # Reshape to per-node format
        user_intent: dict[str, dict[str, Any]] = {}
        for pd in param_descriptors:
            value = raw_intent.get(pd["key"])
            if value is not None:
                user_intent.setdefault(pd["node_id"], {})[pd["param_name"]] = value

        # Validate: every user_intent param got a value
        for pd in param_descriptors:
            node_vals = user_intent.get(pd["node_id"], {})
            if pd["param_name"] not in node_vals:
                logger.warning(
                    "Pipeline verification FAIL: param %s.%s not extracted",
                    pd["node_id"], pd["param_name"],
                )
                return False

        logger.info("Pipeline verification: intent extraction OK — %s", user_intent)

        # Step 2: Execute graph with extracted params
        executor = Executor(self._session)
        try:
            result = await executor.execute(temp_graph, user_intent, subtask_context=task)
        except Exception as exc:
            logger.warning("Pipeline verification: executor crashed: %s", exc)
            return False

        if result.status != "success":
            logger.warning("Pipeline verification FAIL: executor status=%s reason=%s",
                           result.status, result.reason)
            return False

        # Step 3: Validate — every node returned data
        for node in nodes:
            data = result.node_outputs.get(node.id)
            if data is None:
                logger.warning("Pipeline verification FAIL: node %s returned no data", node.id)
                return False

        # Step 4: Selection edges resolved
        for edge in edges:
            if edge.requires_selection:
                downstream_data = result.node_outputs.get(edge.to_node_id)
                if downstream_data is None:
                    logger.warning(
                        "Pipeline verification FAIL: selection edge %s→%s — no downstream data",
                        edge.from_node_id, edge.to_node_id,
                    )
                    return False

        logger.info("Pipeline verification passed (real task → intent → execute → validate)")
        return True

    # ------------------------------------------------------------------
    # Step 7: Identify terminal nodes
    # ------------------------------------------------------------------

    def _identify_terminals(
        self,
        nodes: list[GraphNode],
        requests: list[HTTPRequest],
        observation: SubtaskObservation,
    ) -> list[str]:
        """Identify terminal nodes whose response precedes observable state changes.

        A node is terminal if within 1-2 seconds after its response:
        - URL navigated
        - DOM content hash changed
        - AXTree node count changed by >20%
        """
        terminal_ids = []

        # Build timestamp-sorted snapshots for comparison
        snapshots = sorted(observation.dom_snapshots, key=lambda s: s.timestamp_ms)

        for node in nodes:
            # Find the matching request — prefer _node_to_request, fall back to fingerprint search
            matching_req = self._node_to_request.get(node.id)
            if not matching_req:
                for req in requests:
                    if _compute_endpoint_fingerprint(req) == node.endpoint_fingerprint:
                        matching_req = req
                        break
            if not matching_req:
                continue

            resp_ts = matching_req.timestamp_ms

            # Check URL change within 1 second after response
            for snap in snapshots:
                if 0 < snap.timestamp_ms - resp_ts <= 1000:
                    # Find the previous snapshot
                    prev_snap = None
                    for s in snapshots:
                        if s.timestamp_ms < resp_ts:
                            prev_snap = s
                    if prev_snap and snap.url != prev_snap.url:
                        terminal_ids.append(node.id)
                        break

            if node.id in terminal_ids:
                continue

            # Check DOM content hash change within 1 second
            for snap in snapshots:
                if 0 < snap.timestamp_ms - resp_ts <= 1000:
                    prev_snap = None
                    for s in snapshots:
                        if s.timestamp_ms < resp_ts:
                            prev_snap = s
                    if prev_snap and snap.dom_content_hash != prev_snap.dom_content_hash:
                        terminal_ids.append(node.id)
                        break

            if node.id in terminal_ids:
                continue

            # Check AXTree node count change >20% within 2 seconds
            for snap in snapshots:
                if 0 < snap.timestamp_ms - resp_ts <= 2000:
                    prev_snap = None
                    for s in snapshots:
                        if s.timestamp_ms < resp_ts:
                            prev_snap = s
                    if prev_snap:
                        prev_count = self._count_ax_nodes(prev_snap.ax_tree)
                        curr_count = self._count_ax_nodes(snap.ax_tree)
                        if prev_count > 0:
                            change_pct = abs(curr_count - prev_count) / prev_count
                            if change_pct > 0.2:
                                terminal_ids.append(node.id)
                                break

        # If no terminals found, use the last node
        if not terminal_ids and nodes:
            terminal_ids = [nodes[-1].id]

        return terminal_ids

    def _count_ax_nodes(self, ax_tree: dict) -> int:
        """Count nodes in an AXTree dict."""
        if not ax_tree:
            return 0
        count = 1
        for child in ax_tree.get("children", []):
            count += self._count_ax_nodes(child)
        return count

    # ------------------------------------------------------------------
    # Step 8: Build completion signals
    # ------------------------------------------------------------------

    def _build_completion(
        self,
        terminal_ids: list[str],
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        observation: SubtaskObservation,
    ) -> dict:
        """Build completion signals: navigate URL template + success indicator."""
        if not terminal_ids:
            return {}

        # Find the URL after the last terminal node's response
        navigate_url = observation.end_url

        # Try to templatize the URL: replace chained/user_intent values with placeholders
        url_template = navigate_url
        for node in nodes:
            for param in node.core_parameters:
                val = str(param.value_example)
                if len(val) >= 2 and val in url_template:
                    if param.role == "chained" and param.chained_from:
                        placeholder = f"${{{param.chained_from}}}"
                    elif param.role == "user_intent":
                        placeholder = f"${{{param.name}}}"
                    else:
                        continue
                    url_template = url_template.replace(val, placeholder)

        # Success indicator: find a stable selector in the final DOM snapshot
        # Priority: data-test/data-testid attributes > stable class patterns > structural selectors
        success_indicator = {}
        if observation.dom_snapshots:
            last_snap = observation.dom_snapshots[-1]
            selector = self._find_success_indicator_selector(last_snap)
            success_indicator = {
                "type": "dom_query",
                "selector": selector,
                "timeout_ms": 5000,
            }

        return {
            "navigate_url_template": url_template,
            "success_indicator": success_indicator,
        }

    def _find_success_indicator_selector(self, snapshot: DOMSnapshot) -> str:
        """Analyze a DOM snapshot to find a stable success indicator selector.

        Priority order:
        1. data-testid / data-test attributes on prominent elements
        2. Stable semantic selectors (main, [role="main"], article, .results, .content)
        3. Structural selectors based on AXTree landmark roles
        4. Fallback to "body"
        """
        ax = snapshot.ax_tree
        if not ax:
            return "body"

        # Walk AXTree looking for landmark roles that indicate meaningful content
        candidates: list[tuple[int, str]] = []  # (priority, selector)

        def _walk_ax(node: dict, depth: int) -> None:
            if depth > 6:
                return
            role = ""
            name = ""
            if isinstance(node.get("role"), dict):
                role = node["role"].get("value", "")
            elif isinstance(node.get("role"), str):
                role = node["role"]
            if isinstance(node.get("name"), dict):
                name = node["name"].get("value", "")
            elif isinstance(node.get("name"), str):
                name = node["name"]

            # Priority 1: data-testid references (from name heuristic)
            if name and "test" in name.lower():
                candidates.append((1, f"[data-testid='{name}']"))

            # Priority 2: Semantic landmarks that indicate results/content areas
            if role == "main":
                candidates.append((2, "main, [role='main']"))
            elif role == "region" and name:
                safe_name = name.replace("'", "\\'")
                candidates.append((3, f"[aria-label='{safe_name}']"))
            elif role == "article":
                candidates.append((3, "article"))

            for child in node.get("children", []):
                _walk_ax(child, depth + 1)

        _walk_ax(ax, 0)

        # Sort by priority (lower = better)
        candidates.sort(key=lambda x: x[0])
        if candidates:
            return candidates[0][1]

        # Priority 3: Common CSS class selectors for results pages
        for sel in (".results", ".search-results", "[class*='result']", ".content", ".main-content"):
            return sel  # Return first — these are tried in order by the executor

        return "body"

    # ------------------------------------------------------------------
    # Step 10: Check against existing registry
    # ------------------------------------------------------------------

    def _check_registry(
        self,
        new_graph_id: str,
        new_nodes: list[GraphNode],
        new_edges: list[GraphEdge],
        existing_graphs: list[Graph],
    ) -> tuple[str, list[str]]:
        """Check new graph against existing registry.

        Returns: (action, parent_graph_ids)
        - "already_exists" if identical graph exists
        - "subsumed" if new graph is subset of existing
        - "new" if it's a new independent graph or a supergraph
        """
        parent_ids = []

        # Build a temporary Graph for comparison
        temp_graph = Graph(
            id=new_graph_id,
            site="",
            name="",
            description="",
            capability_statement="",
            reason_for_version="",
            parent_graph_ids=[],
            nodes=new_nodes,
            edges=new_edges,
            terminal_node_ids=[],
            completion={},
            preconditions={},
        )

        for existing in existing_graphs:
            if graphs_equivalent(temp_graph, existing):
                return ("already_exists", [])

            if is_subset(temp_graph, existing):
                return ("subsumed", [])

            if is_subset(existing, temp_graph):
                parent_ids.append(existing.id)

        return ("new", parent_ids)

    # ------------------------------------------------------------------
    # Step 11: Name and describe (ONE LLM call)
    # ------------------------------------------------------------------

    async def _name_graph(
        self,
        observation: SubtaskObservation,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        parent_ids: list[str],
        existing_graphs: list[Graph],
    ) -> dict:
        """Generate name, description, and capability_statement via ONE LLM call."""
        # Build user prompt
        node_descriptions = []
        for node in nodes:
            params_str = ", ".join(
                f"{p.name} ({p.role})" for p in node.core_parameters
            )
            extracts_str = ", ".join(
                f"{k}: {v}" for k, v in node.response_extract_paths.items()
            ) or "none"
            node_descriptions.append(
                f"  - [{node.id}] Operation at endpoint {node.endpoint_fingerprint}\n"
                f"    CU reasoning at this step: {node.cu_reasoning_sample[:200]}\n"
                f"    Parameters: {params_str}\n"
                f"    Response contains: {extracts_str}"
            )

        edge_descriptions = []
        for edge in edges:
            edge_descriptions.append(
                f"  - Output {edge.from_extract} of {edge.from_node_id} "
                f"feeds into parameter {edge.to_parameter} of {edge.to_node_id}"
            )

        parent_descriptions = []
        for pid in parent_ids:
            for g in existing_graphs:
                if g.id == pid:
                    parent_descriptions.append(f"  - {g.name}: {g.capability_statement}")
                    break

        existing_descriptions = []
        for g in existing_graphs:
            if g.id not in parent_ids:
                existing_descriptions.append(f"  - {g.name}: {g.capability_statement}")

        user_prompt = f"""Site: {observation.site}
Subtask that led to this discovery: {observation.subtask_description}

This workflow has these operations in sequence (topologically ordered):
{chr(10).join(node_descriptions)}

Connections between operations:
{chr(10).join(edge_descriptions) if edge_descriptions else "  (no connections — single operation)"}
"""

        if parent_descriptions:
            user_prompt += f"""
This workflow extends these prior workflows:
{chr(10).join(parent_descriptions)}

What does this workflow add over them?
"""

        if existing_descriptions:
            user_prompt += f"""
Other existing workflows on this site:
{chr(10).join(existing_descriptions)}
"""

        user_prompt += """
Produce JSON with keys:
  name: short, capability-focused (e.g., "Search products by category", "Add item to cart after verification")
  description: 2-3 sentences describing what the workflow does, inputs needed, outputs produced
  capability_statement: ONE natural language sentence for semantic search (e.g., "Given a search query and category, returns a list of matching products with their prices and availability")
  reason_for_version: what this extends over parents (empty string if no parents)
  node_descriptions: array of {node_id, description} for EVERY node listed above. Each description must be brief and distinguish nodes that share the same endpoint (e.g., "Source station auto-suggest" vs "Destination station auto-suggest").

Be specific about the user's visible outcome. Avoid jargon about HTTP or APIs."""

        try:
            result = call_gemini(
                model="gemini-3-flash-preview",
                contents=[user_prompt],
                response_schema=_NAMING_SCHEMA,
                system_instruction=_NAMING_SYSTEM_PROMPT,
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
            )
            return result
        except Exception as exc:
            logger.warning("Graph naming LLM call failed: %s", exc)
            return {
                "name": f"Workflow on {observation.site}",
                "description": observation.subtask_description,
                "capability_statement": observation.subtask_description,
                "reason_for_version": "",
            }
