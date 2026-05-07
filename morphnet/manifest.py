"""
manifest.py — Graph schema, storage, identity, and retrieval for MorphNet.

Owns all data models used by observer, learner, and executor.
Provides persistence operations for observations, graphs, scripts, and embeddings.
Computes graph identity via structural hashing for deduplication and subsumption.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SITES_DIR = Path(__file__).parent / "sites"


# ---------------------------------------------------------------------------
# Observation data models (captured by observer)
# ---------------------------------------------------------------------------

@dataclass
class CUAction:
    """A single CU browser action with its reasoning."""
    timestamp_ms: int
    subtask_id: str
    action_type: str  # "click", "type", "select", "scroll", "wait", "navigate"
    target_selector: str  # xpath or CSS selector
    target_attributes: dict  # all attributes on the target element at action time
    target_text: str  # text content of target element
    target_ax_node_id: Optional[str]
    typed_value: Optional[str]  # for type actions
    cu_reasoning: str  # CU's thinking/rationale for this action


@dataclass
class HTTPRequest:
    """A captured HTTP request/response pair with initiator stack trace."""
    timestamp_ms: int
    subtask_id: str
    request_id: str  # CDP request ID
    url: str
    method: str
    headers: dict
    body: Optional[str]
    request_type: str  # "rest", "graphql", "json_rpc", "form", "other"
    graphql_operation_name: Optional[str]
    graphql_query_hash: Optional[str]
    jsonrpc_method: Optional[str]
    response_status: int
    response_headers: dict
    response_body: Optional[str]
    response_time_ms: int
    initiator_stack: list  # CDP stack frames: [{scriptId, lineNumber, columnNumber, functionName}]
    initiator_type: str  # "script", "parser", "preflight", etc.


@dataclass
class ScriptSource:
    """A captured JavaScript source file, deduplicated by content hash."""
    script_id: str
    url: str
    content_hash: str  # sha256
    source: str
    is_module: bool


@dataclass
class DOMSnapshot:
    """A point-in-time DOM/AXTree snapshot."""
    timestamp_ms: int
    subtask_id: str
    url: str
    ax_tree: dict  # full AXTree at this point
    dom_content_hash: str  # hash of body.innerHTML for quick comparison
    storage_keys: dict  # {"localStorage": [...], "sessionStorage": [...], "cookies": [...]}


@dataclass
class NavigationEvent:
    """A captured history.pushState or replaceState call."""
    timestamp_ms: int
    nav_type: str  # "pushState" or "replaceState"
    url: str


@dataclass
class SubtaskObservation:
    """Complete observation record for one CU subtask execution."""
    subtask_id: str
    site: str
    start_url: str
    end_url: str
    subtask_description: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    cu_actions: list[CUAction]
    http_requests: list[HTTPRequest]
    scripts: dict[str, ScriptSource]  # script_id -> source
    dom_snapshots: list[DOMSnapshot]
    navigation_events: list[NavigationEvent]
    framework_fingerprint: dict  # detected frameworks, reachable globals
    bundle_hash: str  # hash of all script content hashes combined
    reflector_verdict: str  # success / failure / uncertain


# ---------------------------------------------------------------------------
# Graph data models (built by learner, consumed by executor)
# ---------------------------------------------------------------------------

@dataclass
class ParameterSpec:
    """Specification for a single parameter in a graph node."""
    name: str
    role: str  # "user_intent", "chained", "website_generated"
    value_example: Any
    chained_from: Optional[str] = None  # "node_id.extract_path" for chained
    cu_action_binding: Optional[dict] = None  # for user_intent: which CU action produced it
    transformations: list[str] = field(default_factory=list)
    profile_history: list[str] = field(default_factory=list)  # values seen across prior tasks


@dataclass
class NodeInvocation:
    """How to invoke a graph node via CDP."""
    type: str  # "cdp_eval_global", "cdp_eval_redux", "cdp_eval_extracted", "dom_replay"
    expression: Optional[str] = None  # JS expression for cdp_eval_* types
    extracted_function_source: Optional[str] = None  # for cdp_eval_extracted
    dom_target_selector: Optional[str] = None  # for dom_replay
    dom_event_type: Optional[str] = None  # for dom_replay
    await_promise: bool = True


@dataclass
class GraphNode:
    """A single HTTP endpoint node in the execution graph."""
    id: str  # stable within the graph
    endpoint_fingerprint: str  # canonical form
    http_method: str
    url_template: str  # with {param} placeholders
    request_type: str  # "rest", "graphql", "json_rpc", etc.
    core_parameters: list[ParameterSpec]
    optional_parameters: list[ParameterSpec]
    response_schema: dict  # JSON schema of response
    response_extract_paths: dict[str, str]  # name -> JSONPath, for chained use
    invocation: NodeInvocation
    cu_reasoning_sample: str
    observed_in_subtasks: list[str]
    example_request_body: Optional[str] = None
    example_response_body: Optional[str] = None
    node_description: str = ""  # human-readable role (e.g., "Source station auto-suggest")


@dataclass
class GraphEdge:
    """A data-flow edge connecting two graph nodes."""
    from_node_id: str
    to_node_id: str
    from_extract: str  # extract path from source node's response
    to_parameter: str  # parameter name in target node
    requires_selection: bool = False  # array source needs LLM pick during execution
    selection_array_path: str = ""  # JSONPath to array (without index)
    selection_item_field: str = ""  # field to extract from selected item


@dataclass
class Graph:
    """A complete execution graph for a website capability."""
    id: str  # stable hash of node set + edge set
    site: str
    name: str  # LLM-generated, capability-focused
    description: str  # LLM-generated, detailed
    capability_statement: str  # LLM-generated, embedding-friendly
    reason_for_version: str
    parent_graph_ids: list[str]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    terminal_node_ids: list[str]
    completion: dict  # {"navigate_url_template": str, "success_indicator": {...}}
    preconditions: dict  # {"url_pattern": str, "required_globals": list[str], "bundle_hash": str}
    execution_stats: dict = field(default_factory=lambda: {"runs": 0, "successes": 0, "last_run_at": ""})
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verified: bool = False
    verification_only_read: bool = True
    framework_detected: str = "unknown"


@dataclass
class ToolRegistry:
    """Registry of all graphs for a site."""
    site: str
    graphs: list[Graph] = field(default_factory=list)
    capability_embeddings: Optional[dict] = None  # graph_id -> embedding vector


# ---------------------------------------------------------------------------
# Executor result types
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing a graph via the executor."""
    status: str  # "success", "not_applicable", "degraded", "execution_error", "completion_timeout"
    result: Optional[dict] = None  # terminal node response data on success
    failed_node_id: Optional[str] = None  # for execution_error
    reason: Optional[str] = None  # human-readable reason for non-success
    node_outputs: dict = field(default_factory=dict)  # node_id → full response data


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _dict_to_cu_action(d: dict) -> CUAction:
    return CUAction(**d)


def _dict_to_http_request(d: dict) -> HTTPRequest:
    return HTTPRequest(**d)


def _dict_to_script_source(d: dict) -> ScriptSource:
    return ScriptSource(**d)


def _dict_to_dom_snapshot(d: dict) -> DOMSnapshot:
    return DOMSnapshot(**d)


def _dict_to_navigation_event(d: dict) -> NavigationEvent:
    return NavigationEvent(
        timestamp_ms=d["timestamp_ms"],
        nav_type=d["nav_type"],
        url=d["url"],
    )


def _dict_to_observation(d: dict) -> SubtaskObservation:
    return SubtaskObservation(
        subtask_id=d["subtask_id"],
        site=d["site"],
        start_url=d["start_url"],
        end_url=d["end_url"],
        subtask_description=d["subtask_description"],
        start_timestamp_ms=d["start_timestamp_ms"],
        end_timestamp_ms=d["end_timestamp_ms"],
        cu_actions=[_dict_to_cu_action(a) for a in d["cu_actions"]],
        http_requests=[_dict_to_http_request(r) for r in d["http_requests"]],
        scripts={k: _dict_to_script_source(v) for k, v in d["scripts"].items()},
        dom_snapshots=[_dict_to_dom_snapshot(s) for s in d["dom_snapshots"]],
        navigation_events=[_dict_to_navigation_event(n) for n in d.get("navigation_events", [])],
        framework_fingerprint=d["framework_fingerprint"],
        bundle_hash=d["bundle_hash"],
        reflector_verdict=d["reflector_verdict"],
    )


def _dict_to_parameter_spec(d: dict) -> ParameterSpec:
    return ParameterSpec(
        name=d["name"],
        role=d["role"],
        value_example=d.get("value_example"),
        chained_from=d.get("chained_from"),
        cu_action_binding=d.get("cu_action_binding"),
        transformations=d.get("transformations", []),
        profile_history=d.get("profile_history", []),
    )


def _dict_to_invocation(d: dict) -> NodeInvocation:
    return NodeInvocation(
        type=d["type"],
        expression=d.get("expression"),
        extracted_function_source=d.get("extracted_function_source"),
        dom_target_selector=d.get("dom_target_selector"),
        dom_event_type=d.get("dom_event_type"),
        await_promise=d.get("await_promise", True),
    )


def _dict_to_graph_node(d: dict) -> GraphNode:
    return GraphNode(
        id=d["id"],
        endpoint_fingerprint=d["endpoint_fingerprint"],
        http_method=d["http_method"],
        url_template=d["url_template"],
        request_type=d["request_type"],
        core_parameters=[_dict_to_parameter_spec(p) for p in d["core_parameters"]],
        optional_parameters=[_dict_to_parameter_spec(p) for p in d.get("optional_parameters", [])],
        response_schema=d.get("response_schema", {}),
        response_extract_paths=d.get("response_extract_paths", {}),
        invocation=_dict_to_invocation(d["invocation"]),
        cu_reasoning_sample=d.get("cu_reasoning_sample", ""),
        observed_in_subtasks=d.get("observed_in_subtasks", []),
        example_request_body=d.get("example_request_body"),
        example_response_body=d.get("example_response_body"),
        node_description=d.get("node_description", ""),
    )


def _dict_to_graph_edge(d: dict) -> GraphEdge:
    return GraphEdge(
        from_node_id=d["from_node_id"],
        to_node_id=d["to_node_id"],
        from_extract=d["from_extract"],
        to_parameter=d["to_parameter"],
        requires_selection=d.get("requires_selection", False),
        selection_array_path=d.get("selection_array_path", ""),
        selection_item_field=d.get("selection_item_field", ""),
    )


def _dict_to_graph(d: dict) -> Graph:
    return Graph(
        id=d["id"],
        site=d["site"],
        name=d["name"],
        description=d["description"],
        capability_statement=d.get("capability_statement", ""),
        reason_for_version=d.get("reason_for_version", ""),
        parent_graph_ids=d.get("parent_graph_ids", []),
        nodes=[_dict_to_graph_node(n) for n in d["nodes"]],
        edges=[_dict_to_graph_edge(e) for e in d["edges"]],
        terminal_node_ids=d.get("terminal_node_ids", []),
        completion=d.get("completion", {}),
        preconditions=d.get("preconditions", {}),
        execution_stats=d.get("execution_stats", {"runs": 0, "successes": 0, "last_run_at": ""}),
        created_at=d.get("created_at", ""),
        verified=d.get("verified", False),
        verification_only_read=d.get("verification_only_read", True),
        framework_detected=d.get("framework_detected", "unknown"),
    )


# ---------------------------------------------------------------------------
# Graph identity
# ---------------------------------------------------------------------------

def compute_node_fingerprint(node: GraphNode) -> str:
    """Canonical fingerprint for a graph node.

    REST: '{method} {url_path_template} [query_param_names_sorted]'
    GraphQL: '{operation_name}#{query_hash[:12]}' at the GraphQL endpoint URL
    JSON-RPC: '{jsonrpc_method}' at the JSON-RPC endpoint URL

    Uses the pre-computed endpoint_fingerprint stored on the node.
    """
    return node.endpoint_fingerprint


def compute_graph_id(nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
    """Compute stable graph identity from its structure.

    sha256 of sorted (node_endpoint_fingerprints, edge_tuples).
    Two graphs with the same node set and edge set always get the same ID.
    """
    node_fps = sorted(n.endpoint_fingerprint for n in nodes)
    edge_tuples = sorted(
        (e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter)
        for e in edges
    )
    identity_str = json.dumps({"nodes": node_fps, "edges": edge_tuples}, sort_keys=True)
    return hashlib.sha256(identity_str.encode()).hexdigest()


def is_subset(g_small: Graph, g_large: Graph) -> bool:
    """Check if g_small is a strict subgraph of g_large.

    g_small ⊑ g_large ⟺ N_small ⊆ N_large ∧ E_small ⊆ E_large
    Strict: excludes equality (g_small must be properly smaller).
    Uses endpoint_fingerprints for node comparison.
    """
    small_fps = {n.endpoint_fingerprint for n in g_small.nodes}
    large_fps = {n.endpoint_fingerprint for n in g_large.nodes}

    if not small_fps < large_fps:
        # Not a strict subset of nodes (either equal or not a subset)
        if small_fps == large_fps:
            # Nodes equal — check if edges are strictly smaller
            small_edges = {(e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter) for e in g_small.edges}
            large_edges = {(e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter) for e in g_large.edges}
            return small_edges < large_edges
        return False

    # Nodes are strict subset — check edges are a subset (not necessarily strict)
    small_edges = {(e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter) for e in g_small.edges}
    large_edges = {(e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter) for e in g_large.edges}
    return small_edges <= large_edges


def graphs_equivalent(g1: Graph, g2: Graph) -> bool:
    """Check structural equivalence: same node fingerprints and same edge set."""
    fps1 = sorted(n.endpoint_fingerprint for n in g1.nodes)
    fps2 = sorted(n.endpoint_fingerprint for n in g2.nodes)
    if fps1 != fps2:
        return False

    edges1 = sorted((e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter) for e in g1.edges)
    edges2 = sorted((e.from_node_id, e.to_node_id, e.from_extract, e.to_parameter) for e in g2.edges)
    return edges1 == edges2


# ---------------------------------------------------------------------------
# Storage operations
# ---------------------------------------------------------------------------

def _site_dir(site: str) -> Path:
    return SITES_DIR / site


def save_observation(observation: SubtaskObservation) -> Path:
    """Write SubtaskObservation to sites/{site}/captures/{subtask_id}.json.

    Scripts are stored separately in the bundle directory to deduplicate
    across subtasks sharing the same JS bundle.
    """
    site_dir = _site_dir(observation.site)
    captures_dir = site_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    # Save scripts to bundle directory
    if observation.scripts:
        for script in observation.scripts.values():
            save_script(observation.site, observation.bundle_hash, script)

    # Save observation (without full script sources — just references)
    obs_dict = _dataclass_to_dict(observation)
    # Replace script sources with references to avoid duplication
    obs_dict["scripts"] = {
        sid: {"script_id": s["script_id"], "url": s["url"], "content_hash": s["content_hash"], "is_module": s["is_module"]}
        for sid, s in obs_dict["scripts"].items()
    }

    path = captures_dir / f"{observation.subtask_id}.json"
    path.write_text(json.dumps(obs_dict, indent=2, default=str), encoding="utf-8")
    logger.info("Saved observation %s to %s", observation.subtask_id, path)
    return path


def save_script(site: str, bundle_hash: str, script: ScriptSource) -> Path:
    """Write a script source to sites/{site}/bundle/{bundle_hash}/scripts/{script_id}.js."""
    bundle_dir = _site_dir(site) / "bundle" / bundle_hash / "scripts"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / f"{script.script_id}.js"
    if not path.exists():  # Deduplicate — don't overwrite identical scripts
        path.write_text(script.source, encoding="utf-8")
    return path


def load_script(site: str, bundle_hash: str, script_id: str) -> Optional[ScriptSource]:
    """Load a script source from the bundle directory."""
    path = _site_dir(site) / "bundle" / bundle_hash / "scripts" / f"{script_id}.js"
    if not path.exists():
        return None
    source = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(source.encode()).hexdigest()
    return ScriptSource(
        script_id=script_id,
        url="",  # URL not stored in script file — use observation reference
        content_hash=content_hash,
        source=source,
        is_module=False,
    )


def save_bundle_metadata(site: str, bundle_hash: str, framework_fingerprint: dict) -> Path:
    """Save framework detection metadata for a bundle version."""
    bundle_dir = _site_dir(site) / "bundle" / bundle_hash
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / "metadata.json"
    path.write_text(json.dumps(framework_fingerprint, indent=2), encoding="utf-8")
    return path


def save_graph(graph: Graph) -> Path:
    """Write a graph to sites/{site}/graphs/{graph_id}.json and update tools.json."""
    site_dir = _site_dir(graph.site)
    graphs_dir = site_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Save full graph
    graph_dict = _dataclass_to_dict(graph)
    path = graphs_dir / f"{graph.id}.json"
    path.write_text(json.dumps(graph_dict, indent=2, default=str), encoding="utf-8")

    # Update tools.json registry
    _update_registry(graph)

    logger.info("Saved graph %s (%s) to %s", graph.id[:12], graph.name, path)
    return path


def _update_registry(graph: Graph) -> None:
    """Update tools.json with graph metadata."""
    site_dir = _site_dir(graph.site)
    registry_path = site_dir / "tools.json"

    if registry_path.exists():
        registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry_data = {"site": graph.site, "graphs": []}

    # Find existing entry or create new
    existing_idx = None
    for i, entry in enumerate(registry_data["graphs"]):
        if entry["id"] == graph.id:
            existing_idx = i
            break

    entry = {
        "id": graph.id,
        "name": graph.name,
        "description": graph.description,
        "capability_statement": graph.capability_statement,
        "parent_graph_ids": graph.parent_graph_ids,
        "execution_stats": graph.execution_stats,
        "verified": graph.verified,
        "verification_only_read": graph.verification_only_read,
        "framework_detected": graph.framework_detected,
        "created_at": graph.created_at,
        "file_path": f"graphs/{graph.id}.json",
    }

    if existing_idx is not None:
        registry_data["graphs"][existing_idx] = entry
    else:
        registry_data["graphs"].append(entry)

    registry_path.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")


def load_graph(site: str, graph_id: str) -> Optional[Graph]:
    """Load a graph from sites/{site}/graphs/{graph_id}.json."""
    path = _site_dir(site) / "graphs" / f"{graph_id}.json"
    if not path.exists():
        logger.warning("Graph %s not found at %s", graph_id[:12], path)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _dict_to_graph(data)


def list_graphs(site: str) -> list[Graph]:
    """Load all graphs for a site from the registry."""
    registry_path = _site_dir(site) / "tools.json"
    if not registry_path.exists():
        return []

    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    graphs = []
    for entry in registry_data.get("graphs", []):
        graph = load_graph(site, entry["id"])
        if graph:
            graphs.append(graph)
    return graphs


def update_graph_stats(site: str, graph_id: str, success: bool) -> None:
    """Update execution_stats on a graph after execution."""
    graph = load_graph(site, graph_id)
    if not graph:
        return

    graph.execution_stats["runs"] = graph.execution_stats.get("runs", 0) + 1
    if success:
        graph.execution_stats["successes"] = graph.execution_stats.get("successes", 0) + 1
    graph.execution_stats["last_run_at"] = datetime.now(timezone.utc).isoformat()

    save_graph(graph)


def purge_unverified_graphs(site: str) -> int:
    """Delete unverified read-only graphs (failed verification noise).

    Write-operation graphs (verification_only_read=False) are kept as
    probationary — the executor tries them on the next matching task.
    If they succeed, they get promoted to verified. If they fail, the
    executor discards them via discard_graph().
    """
    site_dir = _site_dir(site)
    graphs_dir = site_dir / "graphs"
    registry_path = site_dir / "tools.json"

    if not registry_path.exists():
        return 0

    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
    kept = []
    removed = 0

    for entry in registry_data.get("graphs", []):
        if entry.get("verified"):
            kept.append(entry)
        elif not entry.get("verification_only_read", True):
            # Write-op graph — probationary, executor will try it
            kept.append(entry)
            logger.info("Kept probationary write-op graph %s (%s)",
                        entry["id"][:12], entry.get("name", ""))
        else:
            # Read-only graph that failed verification — noise, delete
            graph_file = graphs_dir / f"{entry['id']}.json"
            if graph_file.exists():
                graph_file.unlink()
            logger.info("Purged unverified graph %s (%s)", entry["id"][:12], entry.get("name", ""))
            removed += 1

    if removed:
        registry_data["graphs"] = kept
        registry_path.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    return removed


def promote_graph(site: str, graph_id: str) -> bool:
    """Promote a probationary graph to verified after successful executor run."""
    graph = load_graph(site, graph_id)
    if not graph:
        return False
    graph.verified = True
    save_graph(graph)
    logger.info("Promoted graph %s (%s) to verified", graph_id[:12], graph.name)
    return True


def discard_graph(site: str, graph_id: str) -> bool:
    """Discard a probationary graph after failed executor run."""
    site_dir = _site_dir(site)
    graph_file = site_dir / "graphs" / f"{graph_id}.json"
    if graph_file.exists():
        graph_file.unlink()

    # Remove from registry
    registry_path = site_dir / "tools.json"
    if registry_path.exists():
        registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
        registry_data["graphs"] = [e for e in registry_data.get("graphs", []) if e["id"] != graph_id]
        registry_path.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    logger.info("Discarded probationary graph %s", graph_id[:12])
    return True


# ---------------------------------------------------------------------------
# Embedding storage
# ---------------------------------------------------------------------------

def save_embeddings(site: str, embeddings: dict[str, list[float]]) -> Path:
    """Save graph_id -> embedding vector mapping."""
    path = _site_dir(site) / "embeddings.json"
    path.write_text(json.dumps(embeddings), encoding="utf-8")
    return path


def load_embeddings(site: str) -> dict[str, list[float]]:
    """Load graph_id -> embedding vector mapping."""
    path = _site_dir(site) / "embeddings.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def find_candidates(
    site: str,
    subtask_description: str,
    page_url: str,
    embedding_client: Any = None,
) -> list[Graph]:
    """Find candidate graphs that could execute a subtask.

    1. Load all graphs for site
    2. Filter by precondition (URL pattern matches page_url)
    3. If embedding client available, rank by cosine similarity
    4. Otherwise, return in insertion order (most recently created last)
    """
    all_graphs = list_graphs(site)
    if not all_graphs:
        return []

    # Filter by URL precondition
    candidates = []
    for graph in all_graphs:
        url_pattern = graph.preconditions.get("url_pattern", "*")
        if fnmatch(page_url, url_pattern):
            candidates.append(graph)

    if not candidates:
        return []

    # Try semantic ranking via embeddings
    if embedding_client is not None:
        try:
            embeddings = load_embeddings(site)
            if embeddings:
                candidates = _rank_by_similarity(candidates, subtask_description, embeddings, embedding_client)
        except Exception as e:
            logger.debug("Embedding-based ranking failed, using insertion order: %s", e)

    return candidates


def _rank_by_similarity(
    candidates: list[Graph],
    query: str,
    embeddings: dict[str, list[float]],
    embedding_client: Any,
) -> list[Graph]:
    """Rank candidate graphs by cosine similarity of query to capability_statement embeddings."""
    try:
        query_embedding = embedding_client.embed(query)
    except Exception:
        return candidates

    scored = []
    for graph in candidates:
        graph_emb = embeddings.get(graph.id)
        if graph_emb is None:
            scored.append((graph, 0.0))
            continue
        sim = _cosine_similarity(query_embedding, graph_emb)
        scored.append((graph, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in scored]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
