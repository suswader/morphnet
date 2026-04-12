# MorphNet

MorphNet transforms volatile, expensive computer use (CU) into stable, fast, affordable MCP tool calls. Use CU as a **discovery mechanism** — observe successful browser interactions, capture HTTP traffic, identify deterministic request patterns, and crystallize these into reusable MCP tools. Over time, the system shifts from unreliable browser automation to deterministic API-level execution. **CU is discovery infrastructure, not the end state.**

## Architecture

```
User Query + URL
       │
       ▼
┌──────────────────┐
│ session_manager   │  Persistent Chrome via CDP · Raw data server
│                   │  Shared Gemini inference utility
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ morphnet_         │  Branch/prune planning tree (AgentOccam)
│ orchestrator      │  Routes subtasks to CU or MCP
└───┬──────────┬───┘
    │          │
    ▼          ▼
┌────────┐ ┌────────────┐
│computer│ │mcp_manager  │  All protocols: REST, GraphQL, JSON-RPC, form, multipart
│_use    │ │             │  Lifecycle: verified → trusted → degraded → discarded
│ 10 acts│ │             │
└───┬────┘ └──────┬─────┘
    │              │
    ▼              ▼
┌──────────────────────┐
│ reflector             │  Three-stage pipeline: deterministic → AXTree diff → LLM
│                       │  Separate paths for CU actions vs MCP calls vs subtasks
└───────────────────────┘
    │
    ▼
┌──────────────────────┐
│ trace.py              │  Every decision: reasoning, evidence, confidence → JSONL
└───────────────────────┘
```

## Directory Structure

```
morphnet/
├── session_manager.py         # Browser session + raw data + Gemini utility
├── morphnet_orchestrator.py   # Planning, routing, website profiling
├── computer_use.py            # CU agent (10 actions per subtask)
├── representation.py          # Page representation pipeline (CLEAN→COLLECT→STRUCTURE→FORMAT)
├── mcp_manager.py             # MCP creation, execution, lifecycle
├── reflector.py               # Three-stage verification pipeline
├── trace.py                   # Decision trace recorder (deterministic)
├── run_webarena_evals.py      # Eval harness (deterministic, no LLM calls)
├── prompts/                   # All LLM prompts as .txt files
└── sites/                     # Per-website persistent state
    ├── noise_domains.txt
    └── {site_name}/
        ├── profile.json       # Website insights, auth patterns
        ├── credentials.json   # Login credentials
        └── tools.json         # MCP tools + lifecycle status

results/                          # Trace output (auto-created)
├── YYYY-MM-DD_HHMMSS/           # Single run
│   └── trace.jsonl
└── eval_{benchmark}_{datetime}/ # Eval batch
    ├── task_{id}/
    │   └── trace.jsonl
    └── eval_summary.json
```

---

## Core Design Decision: Each Module Owns Its Representation

**session_manager.py serves raw data.** It extracts DOM, AXTree, screenshots, cookies, tokens, and traffic — hands them unprocessed to consumer modules. Basic structural cleaning only (strip `<script>`, `<style>`, `<noscript>`; filter noise domains).

**Each consumer module distills this raw data into the representation its LLM needs.** The raw toolkit includes: AXTree (semantic structure), DOM (parameter sources, form structure, hidden fields), screenshots (visual layout), Set-of-Marks annotation (element grounding for VLMs), cookies/storage (session state), meta tokens (CSRF/auth), and captured traffic (API patterns). No single module uses all of these — each selects and processes the subset relevant to its task, following AgentOccam (ICLR 2025) and Agent-E's principle of task-adaptive distillation. The raw data sources at each module's disposal include: AXTree (semantic roles, states, accessible names), DOM (structure, hidden fields, data attributes, form layout, parameter sources), screenshots (visual layout — annotated with SoM bounding boxes by CU when needed), cookies/storage (session state, auth tokens), meta tokens (CSRF, form keys with source annotations), and captured network traffic (request/response pairs with protocol classification). Each module selects and distills only the sources relevant to its task.

---

## Module Specifications

### session_manager.py — Raw Data Server

Owns the browser. Every other module operates through it.

**Serves (on-demand):** Each consumer calls only what it needs — no bundled extraction. Available: `get_raw_accessibility_tree()`, `get_dom_tree()` (fast regex-cleaned `page.content()`), `take_screenshot()`, `get_interactive_elements()` (with hierarchical filtering at 200+ elements), `get_cookies()`, `get_storage()`, meta tokens with source annotations, captured network traffic with protocol classification.

**Does not do:** LLM-oriented formatting, SoM annotation, DOM distillation, task interpretation, MCP logic.

**Shared Gemini utility:** `call_gemini()` at module level handles API mechanics. Each consumer provides its own model, schema, prompt, and config. Defaults: `max_output_tokens=8192`, `ThinkingConfig(thinking_budget=4096)`. Retries once with doubled thinking budget on truncated JSON.

**Action execution:** Receives structured action dicts from CU agent, resolves element IDs to Playwright selectors, executes, returns structured results. Never decides what action to take.

**Chrome via CDP** for real browser fingerprint. **curl_cffi** for TLS-matched MCP HTTP replay.

---

### morphnet_orchestrator.py — Task Planner

Receives a natural language task + start URL. Decomposes into subtasks. Routes each to CU or MCP.

**Representation:** Text-only AXTree distillation (strip element-level details, keep headings/landmarks/text/structure) + lightweight DOM summary (page landmarks, form structures, metadata). No screenshots — no actionable planning information beyond what text provides.

**Planning model:** AgentOccam's branch/prune tree. Each node is a sub-plan. The orchestrator can `branch` (try new approach), `prune` (abandon failed approach), or `continue`. When a branch completes or is pruned, its observations are condensed into a **structured summary** — not a one-liner but a pointed digest capturing: what was attempted, key actions taken, outcome, reasoning for the outcome, and any insights gained. Only the current active branch retains full context. This manages context growth while preserving enough history for informed planning.

**MCP lifecycle management:** Tracks tool status. Routes to trusted/verified MCPs when available. Falls back to CU when MCPs fail. Does not interpret MCP HTTP responses — reads the reflector's structured verdict. If reflector said success but the page state contradicts it on the next planning step, the orchestrator notices naturally (it loads fresh page state for planning anyway) and degrades the MCP.

**Model:** `gemini-3.1-pro-preview`, thinking enabled, ~0.4 temperature, 8192 max tokens.

---

### representation.py — Page Representation Pipeline

Owns ALL AXTree-to-text transformations. Both CU agent and orchestrator import from it.

**Pipeline:** CLEAN (whitespace normalization, CSS-name filtering, text compression) → COLLECT (element matching, functional role inference) → STRUCTURE (depth-keyed context tracking, text dedup, footer exclusion) → FORMAT (section-based output with inline elements).

**Context tracking:** A `_ContextStack` records the most recent significant text at each AXTree depth during the walk. When a generic button like "ADD" is encountered, the stack provides the nearest product name — regardless of whether it's a heading, StaticText, or paragraph. This solves the "which ADD button?" disambiguation problem on food delivery menus, e-commerce product lists, etc.

**Four views:** `build_cu_representation()` (section-based, inline elements with context, footer excluded), `build_orchestrator_representation()` (text-only, full page, no element IDs), `build_reflector_representation()` (content-focused, card-aware, chrome-compressed — for subtask outcome verification), and `build_mcp_parameter_context()` (recipe-based extraction from browser state for MCP parameter generation).

---

### computer_use.py — Browser Action Agent

Receives a subtask description. Has 10 actions to complete it.

**Representation:** Uses `representation.py` for AXTree-to-text transformation. Interactive elements appear inline with their context text. Generic buttons get nearby-text disambiguation. Footer excluded. Pruning rules: merge redundant StaticText, convert tables/lists to Markdown, strip rendering artifacts, collapse repetitive siblings, exclude CSS-class names.

**Viewport-aware:** Loads visible + one viewport below. Scroll remains a valid action for revealing more content. Unlike AgentOccam's "load full page" approach, this handles real-world infinite-scroll sites.

**Screenshots:** SoM-annotated screenshot only on first action and after failed actions. AXTree with element IDs is the primary representation.

**Action space:** `click`, `type`, `select`, `scroll`, `press_key`, `navigate`, `hover`, `go_back`, `wait`, `note`, `stop`. The `note` action records observations without browser interaction (critical for multi-step retrieval). The `stop` action signals subtask completion.

**History:** Flat within subtask (no branching for 10 actions). Last 2-3 actions: full detail. Earlier: one-line summaries. Current state dominates context.

**Extraction pattern (n+1):** Initial extraction once before the action loop. After each action, the after-state becomes the next iteration's before-state. For n actions, this requires n+1 extractions instead of the naive 2n.

**On success:** Signals mcp_manager to analyze captured traffic for MCP discovery.

**Model:** `gemini-3-flash-preview`, thinking enabled, 8192 max tokens.

---

### mcp_manager.py — API Tool Manager

Creates, validates, executes, and lifecycle-manages MCP tools. Built after the three core modules.

**Representation:** Raw DOM focused on parameter sources (hidden fields, data attributes, form structure), meta tokens with source annotations, cookies, storage dumps, and captured traffic. Does not receive AXTree or screenshots.

**Protocol support:** REST, GraphQL (operationName-based identity, mutation detection), JSON-RPC (method-based identity), URL-encoded form, multipart form.

**Evolving parameter schema:** Each MCP tool maintains an inferred schema that grows with every observation. Per parameter, the schema tracks: data type, required vs optional (presence frequency across observations), example values, value ranges for numerics, format hints (UUID, ISO date, JWT, etc.), and — critically — **source hints** noting where this parameter value was found in the browser state (which DOM element, which cookie, which prior API response field). These source hints mean that when the MCP is used in an entirely new scenario, the parameter generator knows exactly where to look first. Early observations produce a draft schema; after 10+ observations it stabilizes with confident required/optional classification. No enum detection — enums catastrophically constrain user-intent fields, session tokens, and chained outputs. Example values (`x-examples`) guide the LLM without constraining it.

**Extraction recipe:** Each tool has a per-parameter `extraction_recipe` — a list of `ExtractionStep` dicts that tell representation.py HOW to extract each parameter at execution time. Steps are typed (cookie, dom_field, dom_list, storage, meta_tag, url_component, prior_api_response, task_description) and classified (user_intent, ephemeral, chained, page_context, static). Built automatically from traced parameter sources at discovery time. The recipe executor in representation.py (`build_mcp_parameter_context`) runs each step deterministically against the browser state to produce structured context for the parameter generation LLM.

**Response chaining:** MCP response bodies are cached by `endpoint_identity`. Tool B's extraction recipe can reference Tool A's response via `prior_api_response` steps — works regardless of whether Tool A ran via MCP or CU (checks cache first, then browser captured traffic). This enables multi-step workflows like "search for location → use place_id to set delivery address."

**Response template:** Each tool learns a structural response template from successful responses. Tracks `always_present_paths` and `always_non_null_paths` (intersection across observations). The reflector uses this for deterministic structural checks — if a path that was always present is suddenly missing, or always-non-null data becomes empty, it's flagged as a failure without needing an LLM.

**A/B learning:** When an MCP tool fails and CU fallback succeeds, `learn_from_cu_fallback` compares the failed parameters against the correct CU traffic. For each differing parameter: traces the correct value in the browser state registry, rebuilds the extraction step, and replaces the old recipe step. Also merges the correct request/response into the schema and template.

**Validation at discovery:** Immediate replay via curl_cffi + independent param generation test + reflector confirms state change. Tool only marked "validated" if all three pass.

**MCP Lifecycle:**

| State | Entry Condition | Orchestrator Behavior |
|---|---|---|
| **Verified** | Passes validation at discovery | Available for routing |
| **Trusted** | 3 consecutive successes from verified | Preferred over CU |
| **Degraded** | Trusted tool fails once | Available with warning; 2 more consecutive failures → discarded |
| **Discarded** | 3 consecutive failures from any state | Removed from routing. Failure reason logged for future reference |

---

### reflector.py — Three-Stage Verification

Determines whether actions and subtasks succeeded. Most actions verified without LLM calls.

**Stage 1 — Deterministic Signals (every action, zero LLM cost):**
Element value before/after, URL change, HTTP status codes from captured traffic, ARIA alert/status/dialog nodes in AXTree (W3C standard — framework-agnostic), `aria-invalid` field changes, element count diff.

Most actions resolve here: type (value match), select (value match), scroll (new elements), navigate (URL change), click-with-navigation (URL change + no alerts).

**Stage 2 — AXTree Diff (ambiguous cases only):**
Flatten before/after AXTrees, compare node signatures, report additions/removals/changes. Prioritize ARIA signal nodes and structural changes. Inherently excludes cosmetic noise (CSS, animations, decorations aren't in AXTree).

Key detection: submit action + no meaningful changes + no ARIA alerts + no HTTP errors = silent failure (flagged as suspicious, never auto-classified as success).

**Stage 3 — LLM Evaluation (only when Stages 1-2 can't resolve, ~2-3 per subtask):**
Receives: the action attempted, deterministic signals as facts, compact AXTree diff, ARIA alert/status text. Must cite specific evidence for its verdict — cannot claim success without pointing to concrete signals. Binary verdict (not rubric-based — research shows 87% human agreement vs ambiguous rubric scores).

**MCP verification — deterministic-only:**
- **Reflector (immediate):** Deterministic HTTP status check → response structure check against learned template (always_present_paths, always_non_null_paths) → page state AXTree diff for mutations. No LLM calls. Returns structured verdict to orchestrator.
- **Orchestrator (natural):** Loads fresh page state on next planning step. If reflector said success but page contradicts, orchestrator notices and degrades the MCP. Semantic verification is the orchestrator's job, not the reflector's.

**Subtask reflection (deep, after entire subtask):**
Full journey evaluation: condensed action log with per-action verdicts, current page AXTree, focused DOM excerpt around expected change region, notes from CU agent. Specifically checks for "claimed but not executed" (agent said stop/success but never performed the key submit/click action). Uses Gemini Pro Preview with high thinking budget.

---

### trace.py — Decision Trace

Already built. Deterministic recorder. Zero LLM calls.

Every Gemini call wraps in `trace.span()`. Every schema includes `reasoning`, `confidence`, `evidence_sources` — these flow directly from model output to trace entries. Every browser action, traffic capture, and reflection assessment is logged.

Output: `./results/{datetime}/trace.jsonl`. Eval harness controls path for benchmark runs.

---

### run_webarena_evals.py — Eval Harness

Deterministic scoring. Zero LLM calls. Wraps MorphNet for WebArena Verified benchmarks.

---

## Model Assignments

| Role | Model | Thinking | Max Tokens |
|---|---|---|---|
| Orchestrator planning | `gemini-3.1-pro-preview` | Enabled, budget 4096 | 8192 |
| CU action generation | `gemini-3-flash-preview` | Enabled, budget 4096 | 8192 |
| Per-action reflection (Stage 3) | `gemini-3-flash-preview` | Enabled, budget 4096 | 8192 |
| Per-subtask reflection | `gemini-3.1-pro-preview` | Enabled, budget 4096 | 8192 |
| MCP parameter generation | `gemini-3-flash-preview` | Enabled, budget 4096 | 8192 |
| MCP response-vs-intent check | `gemini-3-flash-preview` | Enabled, budget 4096 | 8192 |

**Flash Lite is not used anywhere.** Every call sits on a critical path.

---

## Development Principles

1. **Gemini structured output schemas are typed function contracts.** Maximally descriptive field names, types, enums, descriptions. Every schema includes `reasoning`, `confidence`, `evidence_sources`. These flow directly to trace entries.

2. **Never string match on unstructured natural language.** Parsing structured material (HTML, JSON) is fine. Never regex/substring on model outputs.

3. **Centralized representation pipeline in `representation.py`.** session_manager serves raw data. `representation.py` owns all AXTree-to-text transformations — CU gets section-based inline elements with context tracking, orchestrator gets text-only distillation. Additional views: Set-of-Marks annotation (CU on failure), task-adaptive DOM distillation (MCP), adaptive evidence selection (reflector).

4. **Justify every field in every data structure.** What consumes it? What breaks if removed?

5. **No unnecessary files.** Consolidated. Whatever is used in a module, keep it closeby.

6. **Comments explain why, not what.** Related logic stays together.

7. **Prompts live in ./prompts/ as .txt files.** Not hardcoded.

8. **Every decision is traced.** Gemini calls wrap in `trace.span()`. Schema fields → JSONL.

9. **On-demand extraction, not bundled.** session_manager never bundles all extractions into one call. Each consumer calls exactly what it needs. This prevents 26+ second bottlenecks on complex pages.

10. **Auto site profiling.** `site_name` is derived from the URL hostname automatically. Site directories and configs are created on first access, not manually.

---

## Architectural Rules

1. **session_manager owns the browser.** No other module creates contexts, pages, or HTTP clients.
2. **session_manager serves raw data.** Each consumer builds its own view.
3. **Chrome via CDP + curl_cffi.** Real fingerprint for browsing and API replay.
4. **Orchestrator is benchmark-agnostic.** Eval logic in run_webarena_evals.py only.
5. **CU is stateless per subtask.** Orchestrator manages memory via planning tree.
6. **MCP lifecycle: verified → trusted → degraded → discarded.** Orchestrator checks status before routing.
7. **Reflector uses three stages.** Deterministic first, AXTree diff second, LLM third. Most actions need no LLM.
8. **All LLM outputs use structured schemas.** No free-form parsing.
9. **Website state in ./sites/.** Tools, profiles, credentials per-website.
10. **AgentOccam principles throughout.** Align to LLM pretraining. Simplify action/observation spaces. Branch/prune for context.
11. **Python 3.12.** Modern features throughout.
12. **Hierarchical element filtering.** When pages have >200 interactive elements, structural/navigational elements are preserved and the rest are sampled with section summaries for collapsed groups.
13. **Every decision traced via schema fields.** Gemini schemas include reasoning + evidence_sources + confidence. `./results/` stores all trace output, organized by datetime.