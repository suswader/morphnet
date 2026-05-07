# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MorphNet transforms browser automation (computer use) into reusable API tools (MCP). CU is discovery infrastructure — observe successful browser interactions, capture HTTP traffic, and crystallize patterns into deterministic MCP tool calls. Over time, CU gets replaced by fast API-level execution.

## Commands

```bash
# Run a single task
uv run python -m morphnet.session_manager --url "https://example.com" --task "Find X" --headless true --port 9222

# Run with site credentials and custom subtask limit
uv run python -m morphnet.session_manager --url "https://swiggy.com" --task "Order food" --site swiggy_com --max-subtasks 12

# Run full 140-task eval (7 sites × 20 tasks, parallel)
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

Always use `uv run python` — never bare `python`.

## Architecture

The system is a pipeline: **session_manager → orchestrator → (CU | MCP) → reflector → trace**.

- **session_manager.py** — Owns the browser (Chrome via CDP). Serves raw data (AXTree, DOM, screenshots, cookies, traffic) on-demand. Executes actions. Provides shared `call_gemini()` utility. No formatting, no task logic.
- **morphnet_orchestrator.py** — Decomposes tasks into subtasks via AgentOccam branch/prune tree. Routes each subtask to CU or MCP based on tool availability and lifecycle state. Model: `gemini-3.1-pro-preview`.
- **computer_use.py** — Browser action agent. 10 actions per subtask. Section-based AXTree with inline elements. On success, triggers MCP discovery from captured traffic. Model: `gemini-3-flash-preview`.
- **representation.py** — Owns ALL AXTree-to-text transformations. Four views: CU (section-based, inline elements), orchestrator (text-only, no element IDs), reflector (content-focused, card-aware), MCP (DOM-focused extraction). Uses TOON compact notation and depth-keyed context stack for disambiguation.
- **mcp_manager.py** — Full MCP lifecycle: discovery from traffic → noise filtering → protocol classification (REST, GraphQL, JSON-RPC, form, multipart) → parameter schema building → extraction recipe creation → execution with preflight → graph-based compound tools → A/B learning from CU fallback. Lifecycle: verified → trusted → degraded → discarded.
- **reflector.py** — Three-stage verification: deterministic signals (zero LLM) → AXTree diff → LLM (only ~2-3 per subtask). Separate paths for CU actions, MCP calls, and subtask completion.
- **trace.py** — Deterministic JSONL recorder. Zero LLM calls. Trace data comes FROM Gemini schema fields (reasoning, evidence_sources, confidence), never from post-hoc parsing.

## Key Design Rules

1. **Schemas over string matching.** Every Gemini call uses a structured output schema defined at module level. Every schema includes `reasoning`, `confidence`, `evidence_sources`. Never regex/substring on model outputs.
2. **Each module owns its representation.** session_manager serves raw data. Each consumer (CU, orchestrator, MCP, reflector) distills to its own view via representation.py.
3. **Justify every field.** What consumes it? What breaks if removed? No noise in data structures.
4. **No unnecessary files.** Don't split into utils/helpers/constants unless genuinely reused across multiple files.
5. **Trace through schema fields, not parsing.** trace.py is deterministic — Gemini schema fields flow directly to trace entries with zero interpretation.
6. **Never hardcode dates.** Use `datetime` to compute dynamically.
7. **Never read `reference/` without asking.** Build fresh from guidance, not by copying old patterns.
8. **Prompts live in `./morphnet/prompts/` as .txt files.** Not hardcoded in Python.
9. **On-demand extraction.** session_manager never bundles all extractions — each consumer calls exactly what it needs.
10. **Update README.md when making structural changes** (new modules, directories, architecture shifts).

## Per-Site State

`morphnet/sites/{site_name}/` is auto-created from URL hostname. Contains:
- `profile.json` — Learned website insights (navigation patterns, UI quirks)
- `tools.json` — Discovered MCP tools with lifecycle status and extraction recipes
- `credentials.json` — Login credentials (optional, untracked)

## Results

All output goes to `./results/`. Single runs auto-create `results/YYYY-MM-DD_HHMMSS/`. Eval runs create `results/eval_{datetime}/` with per-task subdirectories containing `result.json`, `trace.jsonl`, and `steps/` (representations + screenshots).

## Models

- **Orchestrator + subtask reflection:** `gemini-3.1-pro-preview` (thinking enabled, budget 4096)
- **CU actions + action reflection + MCP param generation:** `gemini-3-flash-preview` (thinking enabled, budget 4096)
- Flash Lite is not used — every call is on a critical path.
