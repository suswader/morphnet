# MorphNet

MorphNet transforms browser automation (computer use) into reusable API tools. CU is discovery infrastructure -- observe successful browser interactions, capture HTTP traffic with initiator stack traces, and crystallize patterns into deterministic execution graphs that invoke the website's own JavaScript via CDP. Over time, CU gets replaced by fast API-level execution.

**Thesis:** The website's own JavaScript code builds every HTTP request correctly. We keep that code running in a persistent browser session and invoke it via CDP rather than reconstructing its logic in Python.

## Architecture

```
User Query + URL
       |
       v
+--------------------+
| session_manager    |  Persistent Chrome via CDP. Raw data server.
|                    |  Shared Gemini inference utility.
+--------+-----------+
         |
         v
+--------------------+
| morphnet_          |  Branch/prune planning tree (AgentOccam)
| orchestrator       |  Routes subtasks to CU or executor
+---+----------+-----+
    |          |
    v          v
+--------+ +----------+
|computer| | executor  |  Deterministic graph runner via CDP
|_use    | | (0 LLM)   |  Topological sort, param chaining, JSONPath
| 10 acts| +----------+
+---+----+
    |
    v (always-on)
+--------------------+
| observer           |  CDP capture: HTTP traffic + stack traces,
|                    |  CU actions + reasoning, DOM snapshots,
|                    |  framework fingerprinting
+--------+-----------+
         |
         v (background, post-subtask)
+--------------------+
| learner            |  12-step pipeline: traffic -> nodes -> edges ->
|                    |  entry points -> graph -> verify -> name -> store
+--------------------+
    |
    v
+--------------------+
| reflector          |  Three-stage: deterministic -> AXTree diff -> LLM
+--------------------+
    |
    v
+--------------------+
| trace.py           |  Every decision: reasoning, evidence, confidence -> JSONL
+--------------------+
```

### Three-Layer Tool Discovery Pipeline

1. **Observer** (always-on during CU) -- Captures raw material without blocking CU execution or making decisions. HTTP traffic with initiator stack traces, CU actions with reasoning, DOM snapshots, framework fingerprinting (React/Redux/Vue/Angular/Next.js).

2. **Learner** (background, post-subtask) -- Builds execution graphs from observations. 12-step pipeline: filter traffic, build nodes, classify endpoints (REST/GraphQL/JSON-RPC), trace edges from stack frames, map parameter roles (`user_intent`/`chained`/`captured_constant`/`session_derived`), identify entry points (5-strategy fallback), verify via CDP re-invocation, name via LLM. ONE LLM call total (naming).

3. **Executor** (deterministic, zero LLM) -- Runs learned graphs via CDP. Topological sort of nodes, JSONPath extraction for parameter chaining, precondition checks, canary tests, completion detection.

## Graph Model

### Formal Definitions

A **node** is a tuple `(endpoint_fingerprint, core_params, response_schema_fingerprint)`. Two nodes are equivalent iff all three fields match.

A **edge** is a triple `(from_node_id, to_node_id, via_extract_path)`. Two edges are equivalent iff all three fields match.

A **graph** G = (N, E) where N is a set of nodes and E is a set of directed edges.

- **Equivalence**: `G1 = G2` iff `N1 = N2` and `E1 = E2`
- **Subgraph**: `G1 < G2` iff `N1 ⊆ N2` and `E1 ⊆ E2` (strict subgraph excludes equality)
- **Terminal set**: `T(G) = {n in N : response(n) precedes an observable state change}` -- URL navigation, DOM hash change, or AXTree count change >20%
- **Identity**: `id(G) = sha256(sorted(node_fingerprints) + sorted(edge_tuples))` -- structural, not nominal

**Endpoint fingerprinting** varies by protocol:
- REST: `method + path_template + sorted(param_names)`
- GraphQL: `operation_name + query_hash`
- JSON-RPC: `method`

**Lifecycle**: Graphs start unverified. Verification via CDP re-invocation at discovery (read-only graphs). Execution stats tracked (runs, successes). Degraded/discarded on repeated failure.

### Graph Operations

**Creation**: observation -> filter traffic -> build nodes -> build edges -> map parameter roles -> identify entry points -> verify (read-only) -> identify terminals -> build completion -> compute identity -> check registry -> name (1 LLM call) -> save.

**Merge (union)**: Given G1 and G2 observed together in a single subtask where their nodes share some identity, produce `G3 = (N1 ∪ N2, E1 ∪ E2 ∪ E_cross)` where `E_cross` includes any newly observed edges between N1 and N2. G1 and G2 remain as parents of G3.

**Example**: X = ({a,b,c}, {a->b, b->c}), Y = ({d,e,c}, {d->e, c->e}). Observed together, a new cross-edge c->e connects X and Y. Result: Z = ({a,b,c,d,e}, {a->b, b->c, c->e, d->e}). Z is a supergraph of both X and Y.

**Subsumption**: When a new graph Z is produced, `parent_graph_ids` = all existing graphs G where `G < Z`.

**Retrieval**: `find_candidates(site, subtask_description, page_url)` filters by precondition (URL pattern match), then ranks by cosine similarity of capability_statement embeddings (if available) or insertion order.

### Significance Score (deferred)

Currently returns 1.0 for all candidates. Future work: rank by `semantic_match * execution_success_rate * terminality_score * recency`.

### Drift Detection (future work)

Bundle hash (`sha256(sorted(script_content_hashes))`) captures the JS bundle identity at graph creation time. When `executor._check_preconditions` detects a bundle hash mismatch, it escalates to a canary test. If the canary fails, the graph is marked degraded. Full re-discovery (triggering the learner to rebuild the graph from fresh CU observations) is planned but not yet implemented.

### Entry Point Identification (5-Strategy Fallback)

When a graph node needs to be invoked, the learner identifies how to trigger the website's JS:

1. **Reachable global** -- Function is directly accessible on `window.*`
2. **Framework dispatch** -- React/Redux/Vue store dispatch or action creator
3. **Extracted function IIFE** -- Extract the function body, wrap in IIFE, evaluate via CDP
4. **DOM replay** -- Simulate the original DOM interaction that triggered the request
5. **Give up** -- Mark node as non-executable (graph still stored for future attempts)

## Directory Structure

```
morphnet/
  session_manager.py         # Browser session + raw data + Gemini utility + stealth
  morphnet_orchestrator.py   # Planning, routing (CU or executor), website profiling
  computer_use.py            # CU agent (10 actions per subtask, batch Plan-Then-Execute)
  observer.py                # Always-on CDP capture (HTTP, actions, DOM, frameworks)
  learner.py                 # Post-subtask graph builder (12-step pipeline)
  executor.py                # Deterministic graph runner via CDP (zero LLM)
  manifest.py                # Data models, storage, identity, retrieval
  representation.py          # TOON notation pipeline (CLEAN->COLLECT->ENRICH->STRUCTURE->FORMAT)
  reflector.py               # Three-stage verification pipeline
  trace.py                   # Decision trace recorder (deterministic)
  prompts/                   # LLM prompts (.txt files, not hardcoded)
    cu_core.txt              # CU base system prompt (~600 tokens)
    cu_action.txt            # CU action generation prompt
    cu_plan.txt              # CU batch planning prompt
    cu_context_*.txt         # Context-specific injections (form, search, listing, recovery)
    orchestrator_plan.txt    # Orchestrator planning prompt
    reflect_action.txt       # Per-action reflection prompt
    reflect_subtask.txt      # Per-subtask reflection prompt
    tool_naming.txt          # Graph naming LLM prompt
    param_generation.txt     # Parameter extraction prompt
  sites/                     # Per-website persistent state (auto-created from URL hostname)
    {site_name}/
      profile.json           # Website insights, navigation patterns
      credentials.json       # Login credentials (optional, untracked)
      graphs/                # Learned execution graphs
        {graph_id}.json      # One graph per file (structural hash as ID)
      captures/              # Raw subtask observations
        {subtask_id}.json    # Complete observation record
      bundle/                # JS bundle snapshots
        {hash}/scripts/      # Captured script sources
      tools.json             # Graph registry (name -> ID mapping)
      embeddings.json        # Capability statement embeddings for retrieval
experiments/
  eval_140_tasks.json        # 7 sites x 20 tasks = 140 eval tasks
  run_eval.sh                # Parallel eval runner (1 Chrome per site)
  analyze_eval.py            # Result analysis: per-site metrics, graph stats
  real_world_tasks.json      # Original 50-task subset
  graph_formation_test.json  # Graph builder integration tests
  test_graph_in_browser.py   # End-to-end graph execution test
  test_perfect_graph_direct.py  # Direct graph execution test
  plot_tool_graph.py         # Graph visualization
results/                     # Auto-created by runs
  {YYYY-MM-DD_HHMMSS}/      # Single run output
    result.json              # Success, answer, executor stats
    trace.jsonl              # Decision trace
    steps/                   # Per-step representations + screenshots
    planning_tree.mermaid    # Planning tree visualization
  eval_{datetime}/           # Eval run output
    {label}/                 # Per-task directory
    eval_summary.json        # Aggregate metrics
```

## Module Specifications

### session_manager.py -- Raw Data Server

Owns the browser. Every other module operates through it.

**Serves (on-demand):** `get_raw_accessibility_tree()`, `get_dom_tree()`, `take_screenshot()`, `get_interactive_elements()`, `get_cookies()`, `get_storage()`, meta tokens, captured network traffic. Each consumer calls only what it needs.

**CDP access for observer/executor:** `get_cdp_session()` returns a fresh CDP session. `evaluate_js(expression, await_promise)` is a thin `Runtime.evaluate` wrapper. `wait_for_dom_stable(timeout_ms)` uses MutationObserver-based stability detection.

**Does not do:** LLM-oriented formatting, task interpretation, graph logic.

**Shared Gemini utility:** `call_gemini()` at module level. Each consumer provides its own model, schema, prompt, config. Defaults: `max_output_tokens=8192`, `ThinkingConfig(thinking_budget=4096)`.

**Chrome via CDP** for real browser fingerprint. **Bot detection hardening:** rebrowser-playwright, playwright-stealth v2, custom stealth scripts, behavioral humanization, TLS fingerprint alignment.

### morphnet_orchestrator.py -- Task Planner + Router

Receives a natural language task + start URL. Decomposes into subtasks. Routes each to CU or executor.

**Routing:** Executor-first when a matching graph exists (via `find_candidates()`). Extracts `user_intent` parameters via LLM. Falls back to CU when executor fails or no graph matches. Observer runs during all CU subtasks; learner fires as a background `asyncio.Task` after successful CU execution.

**Planning:** AgentOccam's branch/prune tree. Each completed/pruned branch is condensed into a `BranchSummary` (what attempted, key actions, outcome, reasoning, insights, data). Loop detection via word overlap on pruned branches.

**Representation:** Text-only AXTree distillation + lightweight DOM summary. No screenshots.

**Model:** `gemini-3.1-pro-preview`, thinking enabled, ~0.4 temperature.

### observer.py -- Always-On CDP Capture

Attaches during CU subtasks. Captures without blocking or making decisions.

**Captures:** HTTP traffic via `Network.requestWillBeSent` / `responseReceived` / `loadingFinished` (with full initiator stack traces), CU actions with reasoning (from `action_dict["reasoning"]` -- a required field in CU's Gemini schema), periodic DOM snapshots (5s timer), framework fingerprinting via `page.evaluate` probes.

**Noise filtering:** Analytics/telemetry domains filtered (google-analytics, segment, sentry, hotjar, etc.). Only substantive API traffic retained.

**Output:** `SubtaskObservation` dataclass containing all captured data + `bundle_hash` (sha256 of sorted script content hashes for drift detection).

### learner.py -- Post-Subtask Graph Builder

Runs in background after successful CU subtasks. ONE LLM call (naming).

**12-step pipeline:**
1. Filter relevant HTTP traffic (remove noise, static assets)
2. Build graph nodes from filtered requests
3. Classify endpoint protocol (REST/GraphQL/JSON-RPC/form)
4. Compute endpoint fingerprints
5. Trace edges from initiator stack frames (shared script -> data flow)
6. Map parameter roles from CU action bindings
7. Identify entry points (5-strategy fallback)
8. Build completion spec (terminal nodes, success indicators)
9. Compute graph identity (structural hash)
10. Check registry for duplicates/subsumption
11. Verify graph via CDP re-invocation (navigate to start_url first)
12. Name via LLM (domain-general prompt, one call)

**Naming LLM (step 12):** The learner's single LLM call is site-agnostic and domain-general. It receives: site name, subtask description, CU reasoning for each node, existing graph names/capability statements, parent graph info, and the new graph's endpoint fingerprints + extract paths. It produces: `name` (short, capability-focused), `description` (2-3 sentences), `capability_statement` (one sentence for semantic retrieval), and `reason_for_version` (what this extends over parents). The prompt uses abstract examples across domains (search, filter, detail, auth, mutation, composite) -- no hardcoded site-specific vocabulary.

**Deduplication:** If new graph's ID matches existing, skip. If new graph is a supergraph of existing, replace. If subset, skip.

### executor.py -- Deterministic Graph Runner

Zero LLM calls. Invokes the website's own JS via CDP.

**Pipeline:**
1. Precondition check (URL pattern, required globals, bundle hash)
2. Canary test (simplest node with example values)
3. Resolve `user_intent` parameters from orchestrator
4. Topological execution (CDP `Runtime.evaluate` each node)
5. JSONPath extraction for parameter chaining between nodes
6. Completion (navigate + success indicator detection)
7. Stats update

**Returns:** `success`, `not_applicable`, `degraded`, `execution_error`, `completion_timeout`

### manifest.py -- Data Models + Storage + Identity

Foundation module. All data models used by observer, learner, and executor.

**Key types:** `CUAction`, `HTTPRequest`, `ScriptSource`, `DOMSnapshot`, `SubtaskObservation`, `ParameterSpec`, `NodeInvocation`, `GraphNode`, `GraphEdge`, `Graph`, `ExecutionResult`

**Storage:** `save_graph()`, `load_graph()`, `list_graphs()`, `find_candidates()`, `save_observation()`, `save_script()`. All under `sites/{site_name}/`.

**Identity:** `compute_graph_id()` = `sha256(sorted(node_fingerprints) + sorted(edge_tuples))`. `is_subset()` and `graphs_equivalent()` for deduplication.

### representation.py -- Page Representation Pipeline

Owns ALL AXTree-to-text transformations. Four views:

- `build_cu_representation()` -- Section-based, inline elements with context, footer excluded
- `build_orchestrator_representation()` -- Text-only, full page, no element IDs
- `build_reflector_representation()` -- Content-focused, card-aware, chrome-compressed
- `build_tool_param_context()` -- DOM-focused extraction for parameter generation

**TOON notation:** Compact abbreviations (~32% char savings, ~48% token savings). `[5] btn"ADD" |near:Margherita Pizza` instead of `[5] button "ADD" -- near: Margherita Pizza`.

**Context tracking:** Depth-keyed `_ContextStack` during AXTree walk disambiguates generic buttons by nearest significant text.

### reflector.py -- Three-Stage Verification

**Stage 1 (deterministic, every action, zero LLM):** Element value before/after, URL change, HTTP status, ARIA alert/status/dialog nodes, element count diff. Most actions resolve here.

**Stage 2 (AXTree diff, ambiguous cases):** Flatten before/after AXTrees, compare node signatures. Submit + no changes + no alerts = silent failure (flagged).

**Stage 3 (LLM, ~2-3 per subtask):** Receives action, deterministic signals, AXTree diff. Must cite specific evidence. Binary verdict.

### trace.py -- Decision Trace

Deterministic recorder. Zero LLM calls. Every Gemini call wraps in `trace.span()`. Schema fields (`reasoning`, `confidence`, `evidence_sources`) flow directly from model output to trace entries.

## Model Assignments

| Role | Model | Thinking |
|---|---|---|
| Orchestrator planning | `gemini-3.1-pro-preview` | Enabled, budget 4096 |
| Subtask reflection | `gemini-3.1-pro-preview` | Enabled, budget 4096 |
| CU actions | `gemini-3-flash-preview` | Enabled, budget 4096 |
| Action reflection (Stage 3) | `gemini-3-flash-preview` | Enabled, budget 4096 |
| Intent extraction (executor) | `gemini-3-flash-preview` | Disabled |
| Graph naming (learner) | `gemini-3-flash-preview` | Enabled, budget 4096 |

Flash Lite is not used. Every call is on a critical path.

## Design Principles

1. **Schemas over string matching.** Every Gemini call uses structured output. Every schema includes `reasoning`, `confidence`, `evidence_sources`. Never regex on model outputs.
2. **Each module owns its representation.** session_manager serves raw data. Each consumer distills via representation.py.
3. **Justify every field.** What consumes it? What breaks if removed?
4. **No unnecessary files.** Don't split into utils/helpers unless genuinely reused.
5. **Trace through schema fields.** trace.py is deterministic -- Gemini schema fields flow directly to trace entries.
6. **Prompts in ./morphnet/prompts/.** Not hardcoded in Python.
7. **On-demand extraction.** session_manager never bundles all extractions -- each consumer calls what it needs.
8. **Observer never blocks CU.** All observer calls wrapped in try/except (non-fatal).
9. **Learner never blocks orchestrator.** Runs as background asyncio.Task, awaited only at task end.
10. **Executor is deterministic.** Zero LLM calls. Invokes the website's own JS. Fails fast on precondition mismatch.

## Commands

```bash
# Run a single task
uv run python -m morphnet.session_manager --url "https://example.com" --task "Find X" --headless true --port 9222

# Run with site credentials and custom subtask limit
uv run python -m morphnet.session_manager --url "https://swiggy.com" --task "Order food" --site swiggy_com --max-subtasks 12

# Run full 140-task eval (7 sites x 20 tasks, parallel)
./experiments/run_eval.sh

# Run eval for specific sites
./experiments/run_eval.sh --site reddit --site youtube --per-site 5

# Resume a failed eval run
./experiments/run_eval.sh --resume results/eval_20260416_143000

# Analyze eval results
uv run python experiments/analyze_eval.py results/eval_20260416_143000/ --verbose

# Install dependencies
uv sync
```

Always use `uv run python` -- never bare `python`.
