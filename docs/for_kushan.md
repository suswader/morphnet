# MorphNet — Onboarding Doc (for Claude Code)

> Hi Claude. This is a primer Kushan can paste into a fresh Claude Code session so you have enough context to be useful immediately. Read this end-to-end before touching code. The repo also has a `CLAUDE.md` at the root with the canonical rules — treat that as authoritative; this file is the *narrative* version that explains the *why*.

---

## 1. The one-line pitch

MorphNet turns slow browser automation (computer use) into fast, deterministic API tool calls. Computer use is the **discovery layer**, not the execution layer. Once we've watched a successful interaction once, we crystallize it into an executable graph that calls the website's own JS via CDP — no more screenshots, no more LLM-in-the-loop for replays.

**Thesis:** the website's own JavaScript already builds every HTTP request correctly. We keep that JS resident in a persistent Chrome session and invoke it through CDP rather than reverse-engineering it in Python.

## 2. The mental model

There are three loops, and you should always know which one you're working in:

```
┌──────────────────────────────────────────────────────────────────┐
│  Loop 1 — DISCOVERY  (slow, expensive, LLM-heavy)                │
│  computer_use.py drives the browser. observer.py records         │
│  everything (HTTP traffic + initiator stacks, DOM, CU actions).  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (after a successful subtask)
┌──────────────────────────────────────────────────────────────────┐
│  Loop 2 — LEARNING   (background, ONE LLM call: naming)          │
│  learner.py walks the 12-step pipeline and produces a Graph.     │
│  Graphs live under morphnet/sites/{site}/graphs/{id}.json        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (next time the orchestrator routes)
┌──────────────────────────────────────────────────────────────────┐
│  Loop 3 — EXECUTION  (deterministic, ZERO LLM)                   │
│  executor.py topologically runs the graph, chains params via    │
│  JSONPath, fires nodes via CDP Runtime.evaluate.                 │
└──────────────────────────────────────────────────────────────────┘
```

If you're ever confused about where a piece of logic belongs, ask: *which loop is this for?* Discovery code goes in `computer_use.py` / `observer.py`. Learning code goes in `learner.py`. Execution code goes in `executor.py`. The orchestrator is the *router* between loops, not a participant.

## 3. Module-by-module map

| File | Responsibility | LLM calls | Notes |
|---|---|---|---|
| `session_manager.py` | Owns Chrome (CDP). Raw-data server. Hosts shared `call_gemini()`. | 0 | No formatting, no task logic. Every other module talks to the browser through this. |
| `morphnet_orchestrator.py` | Plans + routes. AgentOccam branch/prune tree. | many | `gemini-3.1-pro-preview`. Routes each subtask to **executor** (if a graph matches) or **CU** (otherwise). |
| `computer_use.py` | The CU agent. 10 actions per subtask. | many | `gemini-3-flash-preview`. Section-based AXTree, inline elements. |
| `observer.py` | Always-on CDP capture during CU. | 0 | HTTP + initiator stacks + DOM snapshots + framework fingerprints. **Must never block CU** (everything wrapped in try/except). |
| `learner.py` | Post-subtask graph builder. 12 steps. | 1 (naming) | Background `asyncio.Task`. Awaited only at task end. |
| `executor.py` | Deterministic graph runner. | 0 | Topological sort, JSONPath param chaining, preconditions, canary tests. |
| `manifest.py` | Data models, storage, identity. | 0 | All dataclasses live here. `compute_graph_id()` = sha256 over node fingerprints + edges. |
| `representation.py` | All AXTree → text transforms. | 0 | Four views: CU, orchestrator, reflector, MCP/param-extraction. TOON notation, depth-keyed context stack. |
| `reflector.py` | Three-stage verification. | 2-3 per subtask | Stage 1 deterministic, Stage 2 AXTree diff, Stage 3 LLM (only if needed). |
| `noise_filter.py` | Filters analytics/telemetry traffic. | 0 | Used by observer + learner. |
| `trace.py` | JSONL decision recorder. | 0 | **Deterministic only** — no parsing, no inference. Schema fields flow straight in. |

Prompts live in `morphnet/prompts/*.txt`. Never hardcode prompt text in Python.

## 4. The Graph data model (read this twice)

A **node** = `(endpoint_fingerprint, core_params, response_schema_fingerprint)`. Two nodes are equal iff all three match.

An **edge** = `(from_node_id, to_node_id, via_extract_path)`. The `via_extract_path` is a JSONPath telling the executor which field of the upstream response feeds the downstream node.

A **graph** `G = (N, E)` is identified by `sha256(sorted(node_fingerprints) + sorted(edge_tuples))`. Identity is **structural**, not nominal — two graphs with the same shape are the same graph even if learned on different days.

Endpoint fingerprints vary by protocol:
- **REST:** `method + path_template + sorted(param_names)`
- **GraphQL:** `operation_name + query_hash`
- **JSON-RPC:** `method`

### Lifecycle

`unverified → verified → trusted → degraded → discarded`. Verification happens at discovery via CDP re-invocation (read-only graphs only). Repeated execution failures degrade. Drift detection via `bundle_hash` (sha256 of sorted script content hashes) is wired up but full re-discovery on drift is not implemented yet — flag as future work, don't pretend it's done.

### Parameter roles

Every parameter on every node is tagged with one of: `user_intent`, `chained`, `captured_constant`, `session_derived`. The executor resolves them in this order:
1. `user_intent` — pulled from the orchestrator's intent extraction.
2. `chained` — JSONPath-extracted from an upstream node's response.
3. `captured_constant` — hardcoded from the original observation.
4. `session_derived` — read from cookies / storage / page globals.

If you're adding a new parameter source, you're probably adding a new role. Don't shoehorn it into an existing one.

## 5. Hard rules (these will get you yelled at if violated)

1. **Schemas over string matching.** Every Gemini call uses a structured output schema defined at module level. Every schema includes `reasoning`, `confidence`, `evidence_sources`. **Never** regex/substring on model outputs.
2. **Each module owns its representation.** `session_manager` serves raw bytes. Each consumer (CU, orchestrator, MCP, reflector) calls `representation.py` to get its own view. Do not bundle.
3. **Justify every field.** Before adding a field to a dataclass or schema, answer: *what consumes it? what breaks if I remove it?* If you can't answer, don't add it.
4. **No unnecessary files.** Don't split into utils/helpers/constants unless the code is genuinely reused across multiple call sites.
5. **Trace through schema fields, not parsing.** `trace.py` is deterministic. Gemini schema fields (`reasoning`, `confidence`, `evidence_sources`) flow directly to trace entries with zero interpretation.
6. **Never hardcode dates.** Use `datetime` to compute dynamically.
7. **Prompts in `./morphnet/prompts/` as `.txt` files.** Not in Python.
8. **On-demand extraction.** `session_manager` never bundles all extractions. Each consumer asks for exactly what it needs.
9. **Always use `uv run python ...`** — never bare `python`.
10. **Update `README.md`** when you make structural changes (new modules, directories, architecture shifts).
11. **Don't read `reference/` without asking.** It's old code — build fresh from guidance, not by copying patterns.
12. **Observer never blocks CU.** Every observer call is wrapped in try/except. A capture failure is non-fatal.
13. **Learner never blocks orchestrator.** It runs as a background task, awaited only at task completion.
14. **Executor is deterministic.** Zero LLM calls. Fails fast on precondition mismatch.

## 6. Models (and why)

| Role | Model | Thinking |
|---|---|---|
| Orchestrator planning | `gemini-3.1-pro-preview` | budget 4096 |
| Subtask reflection | `gemini-3.1-pro-preview` | budget 4096 |
| CU actions | `gemini-3-flash-preview` | budget 4096 |
| Action reflection (Stage 3) | `gemini-3-flash-preview` | budget 4096 |
| Intent extraction (executor) | `gemini-3-flash-preview` | disabled |
| Graph naming (learner) | `gemini-3-flash-preview` | budget 4096 |

Flash Lite is **not** used. Every call is on a critical path.

## 7. Per-site state

Auto-created from URL hostname. Lives at `morphnet/sites/{site_name}/`:

```
{site_name}/
  profile.json         # learned navigation patterns, UI quirks (gitignored)
  credentials.json     # login creds, optional (gitignored)
  tools.json           # graph registry: name → ID (gitignored)
  graphs/{id}.json     # one graph per file, structural hash as filename
  captures/{id}.json   # raw subtask observations
  bundle/{hash}/scripts/  # captured JS bundles for drift detection
  embeddings.json      # capability statement embeddings for retrieval
```

Anything site-specific that's *learned* is gitignored. Don't commit it.

## 8. Where the bodies are buried

- **`representation.py` is large** (~93k). Don't try to refactor it casually. The TOON notation pipeline (CLEAN → COLLECT → ENRICH → STRUCTURE → FORMAT) is load-bearing — touching one stage breaks the others. Read the docstrings before editing.
- **`session_manager.py` is the largest file** (~138k) because it owns CDP, stealth, raw extraction, and the shared `call_gemini()` utility. Resist the urge to split it — splitting it has been tried and the seams hurt more than the size.
- **`learner.py` is ~108k** but it's twelve discrete steps. Each step is independently testable. If you're modifying it, identify the step number first.
- **The `cu_*.txt` prompts are short on purpose.** CU is supposed to be a fast, focused agent. If you find yourself wanting to add more context to the CU prompt, the answer is almost always *no — distill it in `representation.py` instead*.
- **There's no MCP module anymore.** `mcp_manager.py` was removed in the rewrite. The "MCP" terminology in older docs/comments refers to what is now the Graph + Executor pair. If you see references to `mcp_manager`, they're stale.

## 9. Common commands

```bash
# Single task
uv run python -m morphnet.session_manager --url "https://example.com" --task "Find X" --headless true --port 9222

# With site credentials + custom subtask cap
uv run python -m morphnet.session_manager --url "https://swiggy.com" --task "Order food" --site swiggy_com --max-subtasks 12

# Full eval (7 sites × 20 tasks, parallel; one Chrome per site)
./experiments/run_eval.sh

# Specific sites only
./experiments/run_eval.sh --site reddit --site youtube --per-site 5

# Resume a failed run
./experiments/run_eval.sh --resume results/eval_20260416_143000

# Analyze results
uv run python experiments/analyze_eval.py results/eval_20260416_143000/ --verbose

# Dependencies
uv sync
```

## 10. How to be productive in this repo as Claude

- **Start with the task, not the code.** Ask: which loop am I in (discovery / learning / execution)? Which module owns the responsibility? Then read just that file.
- **Use `Grep` and `Glob` aggressively.** The files are big. Don't read 90k-line files cover-to-cover when you can grep for the symbol.
- **Prefer `Edit` over `Write`.** Almost everything you'll do is a surgical change. Full rewrites are rarely the right call.
- **When unsure, ask before doing.** Especially around: deleting graphs, modifying `manifest.py` data classes, changing prompt text, restructuring the AXTree pipeline. These have downstream effects that aren't obvious from the local diff.
- **Run the trace.** If you're debugging a behavior, the JSONL trace under `results/{datetime}/trace.jsonl` has the model's reasoning and evidence for every decision. Read it before guessing.
- **Don't add backwards-compat shims.** This is a research codebase. If something's removed, delete it cleanly — don't leave deprecated stubs.

## 11. Glossary

- **CU** — Computer Use. The browser-driving LLM agent.
- **AXTree** — Accessibility tree. The structured representation of the page CU operates on.
- **TOON** — The compact AXTree text notation defined in `representation.py`. Saves ~32% chars / ~48% tokens vs verbose.
- **CDP** — Chrome DevTools Protocol. How `session_manager` and `executor` talk to the browser.
- **Graph** — A learned execution plan (nodes + edges). The output of the learner, the input to the executor.
- **Subtask** — One unit of work the orchestrator decomposes the user task into. CU gets up to 10 actions per subtask.
- **AgentOccam** — The branch-and-prune planning tree the orchestrator uses.

## 12. If you only remember three things

1. **Discovery slow, execution fast.** CU is observation; graphs are replay.
2. **Schemas, not strings.** Every Gemini call has a structured output. Never parse free text.
3. **Each module owns its view.** Raw data lives in `session_manager`; representations live in `representation.py`.

That's the whole shape of the system. Welcome aboard, Kushan.
