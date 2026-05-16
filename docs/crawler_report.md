# Crawler vs MorphNet — Side-by-Side System Report

> Comparison of `browser-challenge/crawler/` (Kushan's agent) and `morphnet/` (this repo's agent). Both bet that **the representation, not the reasoning, is the bottleneck for browser agents.** They diverge sharply on *how* to build that representation and *how much* of the task to delegate to the LLM.
>
> Source files read: crawler `master.py` (3043 LOC), `page_filter.py` (3066 LOC), `master_markdown.py` (771 LOC), `templates/page_agent.j2`, `config.py`, `executor.py`, `schemas.py`; morphnet `morphnet_orchestrator.py`, `representation.py`, `computer_use.py`, `session_manager.py`, `reflector.py`, `trace.py`, plus `docs/morphnet_onboarding.md`.

---

## 1. The one-paragraph thesis of each system

**Crawler.** A page is a structural object. Stamp every interesting DOM element with a stable identity (`data-cdx-aid`), compute geometric facts (occlusion, blocking, container hierarchy) that no LLM needs to re-derive, render the page as **markdown** with explicit overlay/blocking annotations, and let one LLM call emit a **batch of 10+ actions**. Page-by-page, no task tree. The extractor is an honest reporter — it never classifies which button is "the submit." If the markdown fails the model, the model re-plans from a fresh extraction on the next turn. Recovery is just "the next turn after a failure."

**MorphNet.** A task has a structure. Decompose it with an **AgentOccam branch/prune planning tree**, route each subtask to either learned API tools (MCP graphs) or computer use, and treat CU as **discovery infrastructure** — every CU run captures HTTP traffic that the learner crystallizes into reusable, deterministic API calls. Over time, the LLM should be doing less browsing and more API calling. The representation is the **AXTree** (Chrome's accessibility tree) projected into four different views (CU, orchestrator, reflector, MCP), each tuned to its consumer. CU emits one action at a time by default with adaptive batching for forms.

Both teams arrived at the same design rules independently: schemas over string matching, hierarchy over classification, on-demand extraction, and "trace data comes from model schema fields, never post-hoc parsing."

---

## 2. Side-by-side at a glance

| Concern | Crawler (V5) | MorphNet |
|---|---|---|
| **Top-level driver** | `master.py` — page-by-page FIFO, no task tree | `morphnet_orchestrator.py` — AgentOccam branch/prune tree, max 15 subtasks |
| **Decomposition** | None. User goal goes to page agent verbatim. | Planner LLM → `next_subtask` (max 10 actions) per turn |
| **Page representation** | Markdown V5, `## Overlays` + `## Content` sections | TOON (`[5] btn"ADD" \|near:Vietnamese Cold Brew`), 4 distinct views |
| **Element identity** | `data-cdx-aid` stamped on the DOM, persists across re-extractions | Per-extraction integer IDs from interactive-element enumeration |
| **Extraction primary source** | One large `page.evaluate()` JS payload + Chrome AX merged on top | Chrome AXTree (CDP `Accessibility.getFullAXTree`) primary; DOM secondary |
| **Per-page extraction stages** | Stamp aids → collect containers/blocks/controls → AX merge → 5-point hit-test occlusion → 5-point hit-test blocking → container hierarchy → form bundling | Visibility filter → dedup → enrichment (4 enrichers w/ circuit breakers) → TOON walk w/ depth-keyed context stack → repetitive-group collapse → Fields/Actions summary |
| **Action emission** | One tool call (`execute_actions`) with N-action batch — 10+ actions per LLM turn is normal | One action per turn (default); batched only for forms / search via Plan-Then-Execute (still ≤2 LLM calls per step) |
| **Re-extraction** | Automatic post-batch; **never exposed as a tool** (model hallucinated AIDs when it could request it) | Automatic before every action and between subtasks |
| **Recovery** | Same path. After a failed batch, model receives the failure tool-result + fresh V5 → re-plans. | Same path. Failed subtask → reflector → planning tree decides `prune` / `branch` / `retry`. |
| **Verification** | Mutation observer + post-batch re-extraction. No reflector LLM. | Three-stage reflector: deterministic signals → AXTree diff → LLM (only ~2-3 LLM calls per subtask) |
| **Models (default)** | Cerebras `gpt-oss-120b` via LiteLLM; SDK mode optional (Bedrock Claude Sonnet 4) | `gemini-3-flash-preview` for actions/planning; `gemini-3.1-pro-preview` for subtask reflection |
| **Tracing** | `TraceCollector` records every turn (thinking, tool calls, results, costs) → `runs/<ts>/` | `trace.py` zero-LLM JSONL; trace fields come from Gemini schema (`reasoning`, `confidence`, `evidence_sources`) |
| **Self-improvement** | None. Each run is independent. | MCP discovery: traffic → noise filter → 12-step learner → tool graphs persisted to `morphnet/sites/{site}/tools.json`. CU eventually replaced by API calls. |
| **Tested benchmarks** | WebGames (20%), Mind2Web (~0%), BrowseComp (~0%), WebArena (~0%) — pre-orphan-controls fixes | 7 sites × 20 tasks eval framework (`experiments/run_eval.sh`) |

---

## 3. Deep-dive: what each crawler file does

### 3.1 `crawler/master.py` (3043 LOC) — the orchestrator-as-driver

This is the agent's main loop, but unlike morphnet's planner-orchestrator, it does **no task decomposition**. It does five things:

1. **Owns a FIFO page queue.** State across pages is `journey: list[PageSummary]`, an `_action_log` of every action ever taken, and entity registries (`_known_aids`, `_aid_type`, `_target_labels`) that persist across re-extractions. There is no subtask, no plan, no replanning.

2. **Drives the per-page loop.** For each URL:
   - `capture_and_extract()` produces `PageFilterOutput` (extraction.json) + V5 markdown.
   - The Jinja template renders the system prompt + user message.
   - `_run_raw_session()` calls the model in a tool-loop. Each model turn returns one `execute_actions` tool call with N actions in its `actions` array.

3. **Executes batches via `executor.py`.** Each action goes through `_prep()` (visibility check + scroll-into-view + hit-test for blockers) then a CDP-dispatched primitive. If any action in a batch fails, the batch stops cold (no retries, except a single click-blocker race retry).

4. **Manages the message thread.** After every successful batch, automatic re-extraction runs (`run_fast_re_extract`), fresh V5 is generated, and the thread is updated. In raw mode, V5 update is *deferred* via `_pending_v5` so it lands *after* the tool result message — the model sees the failure first, then the new state. This ordering matters: it's how the model knows "your last action failed; here's where the page is now."

5. **Detects navigation and stops.** If a click triggers a URL change, the batch terminates and master moves to the next page-task.

**The big architectural choice:** master treats the model as the planner. It assumes that given a faithful page representation + the original goal, the model can decide what to do without being told *which subtask to focus on right now*. This is the opposite of morphnet's bet.

### 3.2 `crawler/page_filter.py` (3066 LOC) — extraction pipeline

Single biggest file in the crawler. The pipeline:

1. **Massive JS payload.** One `page.evaluate()` runs all of:
   - **AID stamping.** `data-cdx-aid` attribute on every DOM element. New elements get new aids; existing aids are reused if unique and untouched. This is what gives the model stable element references *across re-renders*.
   - **Container collection.** Walks `main, article, section, form, nav, aside, div, dialog, [role='dialog']`.
   - **Block collection.** Text from `BLOCK_SELECTOR` nodes, deduplicated per-container.
   - **Control collection.** Inputs, buttons, links, draggables, canvases, sliders, plus cursor-style detection for resize handles.
   - **DnD library fingerprinting.** Detects react-beautiful-dnd, dnd-kit, html5-native via static attributes + runtime probes.
   - **HTML5 drop-zone probing.** For native drag, fires synthetic dragstart/dragover and patches `Event.preventDefault` to detect acceptors.

2. **Parallel AX tree fetch.** `Accessibility.getFullAXTree` via CDP runs in parallel with the JS payload. The two are merged: AX is authority for `disabled`, `ax_name`, `checked`, `expanded`, `focusable`, `has_popup`. The DOM is authority for identity (the aid) and geometry.

3. **Geometric blocking computation.** N² enumeration over containers, then for each candidate pair, `elementsFromPoint` 5-point hit-test (default `blocking_sample_points=5`, `blocking_hit_threshold=0.60`). If 60% of sample points hit a different container, that container is marked as a blocker.

4. **Per-control occlusion.** Same hit-test for every form control + button (default `occlusion_inset_px=6`, `occlusion_sample_points=5`). Output:
   ```python
   TargetOcclusion(checked=True, is_occluded=True, blocked_points=4, total_points=5, blocker_container_ids=["aid-12"])
   ```

5. **Container hierarchy DFS.** `_build_container_closure()` walks parent chains, retaining containers that have content or are in the blocker allowlist.

6. **Form bundling.** Controls grouped by form selector, ordered by DOM position. The output `FormControlGroup` has `container_id` for DOM-order tracking.

The JS payload is the load-bearing engineering choice — it's one network round-trip instead of dozens of `page.$()` calls. That makes extraction fast enough to run after every batch.

### 3.3 `crawler/master_markdown.py` (771 LOC) — V5 rendering

Turns `PageFilterOutput` into the markdown the LLM reads. The output has two top-level sections:

- **`## Overlays`** — fixed-position containers with `z-index ≥ 900`. These are the things that block the rest of the page (modals, cookie banners, sticky headers). Each overlay shows what it blocks: `aid-12 z=900 "Cookie Notice" blocks [aid-3, aid-4]`.

- **`## Content`** — everything else, rendered with indentation showing DOM nesting. Form controls are grouped under their form node, buttons get compact-rendered if there are >4 in a section.

Concrete shape:

```
## Overlays
`aid-99` z=1000 "Cookie consent" blocks [aid-1, aid-3]
  buttons(2): "Accept", "Reject"

## Content
`aid-1` <main>
  "Welcome to BookMyShow"
  form `aid-30`
    input[text] `aid-31` "City"
    button `aid-32` "Find shows"
  `aid-2` <section> "Trending"
    buttons(8): "Movie A", "Movie B", "Movie C"*, ... [* = blocked]
```

Key tricks:
- **Compact button rendering when count > 4.** Saves tokens on long lists.
- **Annotations in brackets, exception-only.** `[disabled]`, `[blocked by aid-X]`, `[scrollable]`, `[animated]`. If a state is the default, it's not rendered.
- **No semantic labels.** "Submit button" is not in the output. The model gets `button \`aid-32\` "Find shows"` and infers from text + context.

### 3.4 `crawler/templates/page_agent.j2` — the prompt

The Jinja template is the policy. Headlines:

- **"NEVER output text. Only tool calls. After successful actions, stop."**
- **"One call. One batch. Markdown-first by default."**
- The single tool is `execute_actions` taking `{"actions": [...]}` — a heterogeneous array of actions: `click`, `type_text`, `copy_paste`, `scroll`, `scroll_page`, `key_press`, `drag` (with sub-modes `target` / `slider` / `offset`), `draw` (canvas strokes!), `hover`, `wait_for_page_settle`, `sleep`, `probe_drop_zones`.
- `target_id` is a bare integer (the aid number). `copy_paste` takes `source_ids` (array — try multiple sources) plus a `target_id`.
- Recovery section is explicit: "1. Re-plan from current markdown. V5 is re-sent after every batch. 2. Full DOM fallback. Use dom.html and extraction.json if ambiguity remains."

The example batch in the template is 11 actions: dismiss overlay → click → scroll → copy_paste → click → drag (to target) → drag (slider) → drag (offset) → key_press → hover → draw. This is the team's mental model: solve the page in one shot.

### 3.5 `crawler/config.py` (162 LOC) — `ExtractorTuning`

All tuning knobs live in one Pydantic config. CLI-overridable. The notable ones:

| Knob | Default | What it does |
|---|---|---|
| `top_sections` | 6 | High-utility containers marked as `section_like` |
| `max_buttons_total` | 40 | Cap total buttons in extraction |
| `max_targets_total` | 120 | Cap total interactive targets |
| `section_node_limit` | 220 | JS query limit for containers |
| `control_node_limit` | 500 | JS query limit for controls |
| `section_block_min_chars` | 6 | Drop blocks shorter than this |
| `occlusion_inset_px` | 6 | Inset from BCR for hit-test corners |
| `occlusion_sample_points` | 5 | Number of hit-test points per element |
| `blocking_hit_threshold` | 0.60 | Fraction of points blocked → declare blocker |
| `overlay_candidate_min_z_index` | 900 | Z-index threshold for `## Overlays` section |
| `section_min_score` | 0.10 | Utility threshold for container inclusion |

There's no morphnet equivalent of this single-config-file pattern. Morphnet's tuning is distributed across `representation.py` constants and per-module enrichers. Worth considering whether to consolidate.

---

## 4. Concept-by-concept comparison

### 4.1 Task decomposition

**Crawler:** None. The page agent gets the full user goal every turn, and the orchestrator just advances when navigation happens.

**MorphNet:** AgentOccam branch/prune tree. Each planning step the LLM emits one of `continue` / `branch` / `prune` / `complete_task`. Pruned/completed branches digest into `BranchSummary` so the active context never grows unboundedly. Loop detection (40% word overlap on pruned attempts) injects a hard "try a fundamentally different approach" warning.

**The trade-off:** crawler's flat model is simpler and lets the LLM do more in one shot, but has nothing to escalate to when stuck on a page. Morphnet's tree is more bookkeeping-heavy but tracks *why* approaches failed across the entire task.

### 4.2 Element identity

**Crawler:** `data-cdx-aid` stamped on the actual DOM. Persistent across re-renders (as long as the element node survives). The model's `target_id: 42` reliably refers to the same element even if Chrome's accessibility tree shifts.

**MorphNet:** Integer IDs assigned per-extraction by `get_interactive_elements()`. Re-extraction reshuffles them. To compensate, morphnet re-extracts before *every* action, so the IDs the LLM sees are always fresh.

**Trade-off:** stamping is invasive (you mutate the page) but gives stability. MorphNet's per-extraction IDs are non-invasive but *require* re-extraction every turn — which makes batching dangerous (because IDs the LLM emitted in action 7 of a batch may be stale by then).

This is exactly why crawler can batch 10+ actions per turn and morphnet can't.

### 4.3 Extraction strategy

**Crawler:** One enormous JS payload that runs the entire DOM walk + AID stamping + DnD fingerprinting + drop-zone probing in a single `page.evaluate()`. Then AX merge in Python, then geometric occlusion via separate hit-test calls.

**MorphNet:** Chrome's AXTree is the primary source. CDP `Accessibility.getFullAXTree` is the one round-trip; everything else is post-processing. DOM is fetched separately when needed.

**Why crawler is going AX-heavier (V6).** The bet for V6 (spec only, lives on the `aria-snapshot-extraction` worktree, not on disk in the cloned repo) is that V5 misses 10–30% of interactive controls because heuristic queues for "what's clickable" diverge from what Chrome's a11y engine actually computes. Concrete numbers from their research: V5 captures 71/89 interactive elements on Google Forms (5.6% match rate after dedup), 57/72 on Metacritic (79%). V6 splits responsibility: AX owns roles + names + ARIA relationships; DOM owns identity (aid) + geometry (occlusion).

This puts crawler on a trajectory toward something *closer* to morphnet's architecture (AX-primary), while preserving its DOM-stamping advantage.

### 4.4 Page rendering format

**Crawler:** Markdown. Verbose but human-readable. `## Overlays` and `## Content` as top-level sections. Annotations bracketed. AIDs inline in code-fences.

**MorphNet:** TOON. Compact: `[5] btn"ADD"="Medium" req,foc |near:Vietnamese Cold Brew`. Achieves ~40% token savings vs. natural language, plus the depth-keyed context stack disambiguates generic buttons by attaching the nearest heading text.

**Trade-off:** TOON is more token-efficient at the cost of being harder to read on first glance. Markdown is more readable but pays in tokens. Both encode hierarchy via indentation.

### 4.5 Action emission

**Crawler:** One LLM call → batch of N actions in a single `execute_actions` tool call. N is unbounded; the example in the prompt has 11 actions. Sequential execution within the batch; first failure aborts the batch.

**MorphNet:** Default = one action per turn. Adaptive batching for forms via Plan-Then-Execute (one LLM call to plan the batch type, one to fill it). Even batched, the batch counts as 1 step against a 10-action budget per subtask.

**The fundamental difference:** crawler trusts the model to predict the post-conditions of action 1 well enough to plan action 2 without seeing the new page state. Morphnet doesn't — it re-extracts and re-prompts between every action.

Crawler can do this because (a) AIDs are stable, so action 7 in a batch references a real element, and (b) the markdown explicitly tells the model what mutations to expect (`## During Batch` section after a batch shows added/removed/changed nodes — if the model's prediction was wrong, it sees that on the next turn).

### 4.6 Re-extraction

Both systems converged on the same insight: **the model should not control re-extraction.** Morphnet bakes it into the action loop (every action triggers re-extraction). Crawler bakes it into the post-batch hook (every successful batch triggers re-extraction).

The crawler team learned this the hard way. There used to be a `re_extract` tool the model could call. It turned out the model would emit `re_extract` then in the *same* batch reference AIDs that didn't exist yet (or had since been replaced). After re-extraction, old `aid-42` could become `aid-58`, and the model would confidently emit `click target_id=42` against a stale handle.

The fix: removed the tool. Re-extraction is invisible to the model — the only thing it sees is fresh markdown on its next turn.

This is an argument *against* exposing extraction-control as a model affordance in any agent design.

### 4.7 Verification & recovery

**Crawler:** No reflector LLM call per action. Verification = "did the executor return success?" + "did the mutation observer record the expected DOM changes?" Recovery = "next turn after failure, with fresh V5 + failure tool-result in the thread."

**MorphNet:** Three-stage reflector. Stage 1 = deterministic signals (URL change, value change, HTTP status, ARIA alerts) — zero LLM calls. Stage 2 = AXTree diff (added/removed/changed nodes by signature) — zero LLM calls. Stage 3 = LLM with deterministic signals + diff as evidence — only ~2-3 LLM calls per subtask. Subtask reflection at the end uses Gemini Pro for deep evaluation and emits a `recommendation` enum that drives the planning tree (`prune` / `branch` / `retry` / `complete`).

**Trade-off:** crawler's lack of an explicit reflector saves LLM calls but means the model is responsible for noticing its own failures from the markdown alone. Morphnet's tiered reflector is more expensive but catches silent failures (e.g., a click that did nothing — no error, no URL change, no DOM mutation).

### 4.8 Self-improvement

**Crawler:** None. Every run starts cold. There is no `sites/` directory, no learned profile, no API tool extraction. The state across runs is just: source code + benchmarks.

**MorphNet:** This is the whole product. Every CU run captures HTTP traffic via the observer (CDP `Network.*` events + initiator stacks). The noise filter strips analytics/ads. The 12-step learner pipeline:
1. Filter noise (two-pass)
2. Build candidates (CU action windowing → group by endpoint → prefix-chain collapse)
3. Detect chains (response value → later request param)
4. Classify parameters via LLM (`user_intent` / `chained` / `website_generated`)
5. Discover entry points (5 strategies including reachable globals + extracted functions)
6. Verify HTTP + pipeline
7. Name the tool
8. Persist to `tools.json`

Subsequent runs check `find_candidates(site, subtask, url)` first and route to a learned graph if available. CU is the discovery path; MCP is the cheap execution path.

This is the actual difference between the two systems: crawler is a pure browser agent. Morphnet is a browser agent that's *trying to make itself unnecessary* on every site it touches.

### 4.9 Models

**Crawler:**
- Default: Cerebras `gpt-oss-120b` via LiteLLM. Fast, cheap, OpenAI-compatible function calling.
- SDK mode: Bedrock Claude Sonnet 4 (`apac.anthropic.claude-sonnet-4-20250514-v1:0`).
- LiteLLM means swapping to Gemini is a flag flip: `--raw-model gemini/gemini-2.5-flash`. The OpenAI-style tool schema works because LiteLLM adapts function-calling for Gemini.

**MorphNet:**
- Action loop / planner / param classifier: `gemini-3-flash-preview` (thinking budget 2048).
- Subtask reflection: `gemini-3.1-pro-preview` (thinking budget 4096).
- All calls go through `session_manager.call_gemini()` with structured output (`response_schema`).

For a head-to-head comparison, both can be on Gemini Flash. Crawler routes through LiteLLM; morphnet through the native SDK.

---

## 5. Design philosophy: where they agree

It's striking how much convergence there is despite the systems being built independently:

| Principle | Crawler | MorphNet |
|---|---|---|
| Schemas over string matching | `EXECUTE_ACTIONS_SCHEMA` defines the tool exactly | Every Gemini call has a structured output schema with `reasoning`, `confidence`, `evidence_sources` |
| Honest reporter, not classifier | "Don't pre-pick the submit button" | "Schemas over string matching... never regex/substring on model outputs" |
| Hierarchy over classification | Container DOM tree is the primary signal | Depth-keyed context stack + structural roles |
| On-demand extraction | `capture_and_extract()` runs only when needed | `session_manager` never bundles, each consumer pulls |
| Re-extraction not exposed to model | Removed `re_extract` tool — it caused AID hallucination | Every action triggers fresh extraction; LLM never controls it |
| Trace from schema, not parsing | Trace records model thinking + tool calls verbatim | `trace.py`: "Trace data comes FROM Gemini schema fields, never from post-hoc parsing" |
| No fallbacks, no silent degradation | "Fix infrastructure. Don't wrap in try/except to keep limping" | Same rule, enforced through reflector verdicts (not silent retries) |
| Cleaning ≠ classification | Lorem strip and dedup yes; utility scoring no | TextDedup yes; semantic dropping no |

The places they *don't* agree are exactly the places where they made different bets:
- **Decomposition.** Crawler says no; morphnet says yes.
- **Batch size.** Crawler 10+; morphnet 1-2.
- **DOM stamping.** Crawler yes; morphnet no.
- **Self-improvement.** Crawler none; morphnet via MCP discovery.

---

## 6. V5, V6, and that V8 thing

You asked. Three different things, one naming collision:

- **V5** = the current crawler page-rendering pipeline. `master_markdown.py` literally has `_MARKDOWN_VERSION = "v5"`. V5 = "the markdown the model reads, version 5." The whole pipeline producing V5 is `page_filter.py` → `master_markdown.py`.
- **V6** = a *bet*, not yet code. The DOM+AX synthesis idea: stop re-deriving roles/names/ARIA in JS, let Chrome's accessibility engine own that, keep DOM only for identity (aid) and geometry. Spec is Plan 030 on the `aria-snapshot-extraction` worktree (not in the cloned repo). May ship, may fold into V5 as targeted patches.
- **V8** in Chrome = the JavaScript engine. Unrelated. V8 runs JS *inside* Chrome — every `page.evaluate()` call you make is V8 evaluating that JS. The version-number collision is a coincidence.

V5→V6 in numbers: V5 misses controls because its heuristic "what's clickable" rules diverge from Chrome's a11y engine. Google Forms 5.6% match rate, Metacritic 79%, Excalidraw / DuckDuckGo over-count (false positives). Patching V5 doesn't converge — every site exposes a new gap.

If V6 ships, the crawler will look more like morphnet at the extraction layer (AX-primary), but with one big advantage morphnet doesn't have: stamped DOM identity that survives re-renders.

---

## 7. The build loop — discipline, not automation

This is in the crawler's onboarding doc as a non-negotiable rule:

> BUILD → EVALUATE → TEST → CHECKPOINT → **LIVE TEST** → OBSERVE
> Live test budget ≤ 2.5 min, ≤ $1 per run.

Important finding: **this is not implemented anywhere in code.** I checked the Makefile, `scripts/`, `tests/`. The Makefile has `make check` / `make format` / `make lint` — but those are code quality (ruff, project rule enforcement), not the agent build loop.

The build loop is a *team rule*, repeated in `docs/reflections.md` from 228 prior sessions of the user yelling at Claude when it claimed `PHASE_COMPLETE` without running the agent against a real site. The discipline is: no claim of done without an end-to-end run + budget cap + trace observation.

If you wanted to formalize it (in either system), it'd be a wrapper script: `run_with_budget.sh URL GOAL` that times out at 2.5 min, kills on cost > $1, and asserts on trace outputs.

Worth noting: morphnet has `experiments/run_eval.sh`, which is the closest thing — multi-task parallel runner with timing and resume support. It's the start of automation but doesn't enforce a per-run budget.

---

## 8. Proposed 10-website experiment

You asked for a head-to-head on 10 websites. Here's the design.

### 8.1 Sites and tasks

Mapping the 10 sites to existing morphnet site profiles:

| # | Site | URL | Existing morphnet profile? | Task category |
|---|---|---|---|---|
| 1 | Swiggy | https://www.swiggy.com | ✅ `swiggy_com/` | E-commerce / food |
| 2 | LEGO | https://www.lego.com | ✅ `lego_com/` | E-commerce |
| 3 | BookMyShow | https://in.bookmyshow.com | ✅ `bookmyshow_com/`, `in_bookmyshow_com/` | Booking |
| 4 | ConfirmTkt | https://www.confirmtkt.com | ✅ `confirmtkt_com/` | Booking |
| 5 | RedBus | https://www.redbus.in | ❌ cold start | Booking |
| 6 | Cleartrip | https://www.cleartrip.com | ✅ `cleartrip_com/` | Booking |
| 7 | Gemini | https://gemini.google.com | ❌ cold start, also requires login | Chat / login flow |
| 8 | Wikipedia | https://en.wikipedia.org | ❌ cold start | Reference / read-only |
| 9 | GitHub | https://github.com | ❌ cold start | Developer / search |
| 10 | Amazon.in | https://www.amazon.in | ❌ cold start | E-commerce |

Worth flagging: 4 of 10 have existing morphnet profiles (giving morphnet a "warm start" advantage on those — learned MCP tools or insights from prior runs). Crawler always starts cold. To make this fair we have two options: (a) accept the asymmetry and report on it, or (b) wipe morphnet's per-site state for these sites before the run. I'd recommend (a) — the warm-start ability *is* one of morphnet's design wins and the comparison should reflect it.

### 8.2 Task definitions (one per site)

These are read-only / observation tasks where possible, to avoid making real bookings. Each is sized to be solvable in ≤10 actions for morphnet (its budget) and a couple of batches for crawler.

```json
[
  {"site": "swiggy", "url": "https://www.swiggy.com",
   "task": "Find a popular pizza restaurant near Bangalore Indiranagar and report its rating and average price for two."},
  {"site": "lego", "url": "https://www.lego.com",
   "task": "Find the price of the LEGO Star Wars Millennium Falcon (75192) and report whether it is in stock."},
  {"site": "bookmyshow", "url": "https://in.bookmyshow.com",
   "task": "Find one movie currently running in Bangalore and report its name, language, and duration."},
  {"site": "confirmtkt", "url": "https://www.confirmtkt.com",
   "task": "Search for trains from Pune to Mumbai for tomorrow's date and report the name and departure time of the first train listed."},
  {"site": "redbus", "url": "https://www.redbus.in",
   "task": "Search for buses from Bangalore to Chennai for tomorrow and report the lowest fare shown."},
  {"site": "cleartrip", "url": "https://www.cleartrip.com",
   "task": "Search for flights from Delhi to Mumbai for the day after tomorrow and report the cheapest flight's airline and price."},
  {"site": "gemini", "url": "https://gemini.google.com",
   "task": "Open Gemini and report whether the page requires sign-in. If logged in, report the model name shown in the model picker."},
  {"site": "wikipedia", "url": "https://en.wikipedia.org",
   "task": "Find the Wikipedia article for 'Computer Use Agent' or the closest equivalent topic, and report the first paragraph of the lead section."},
  {"site": "github", "url": "https://github.com",
   "task": "Search for the repository 'anthropics/claude-code' and report the number of stars and the primary language."},
  {"site": "amazon", "url": "https://www.amazon.in",
   "task": "Search for 'Sony WH-1000XM5 headphones', open the first listing, and report its price and average customer rating."}
]
```

### 8.3 LLM configuration — both on Gemini

To make the comparison apples-to-apples on the model side, both systems use Gemini.

**MorphNet** uses Gemini natively today (`gemini-3-flash-preview` for actions, `gemini-3.1-pro-preview` for subtask reflection). No change needed.

**Crawler** routes through LiteLLM, which supports Gemini via `gemini/gemini-*` model IDs. The flag is:

```bash
uv run -m crawler.main \
  --url "<URL>" \
  --goal "<TASK>" \
  --agent-mode raw \
  --raw-model gemini/gemini-2.5-flash \
  --headless \
  --max-pages 10
```

LiteLLM picks up `GEMINI_API_KEY` from env. The OpenAI-style tool schema in `master.py` (`EXECUTE_ACTIONS_SCHEMA`) translates to Gemini function-calling automatically.

**Caveat to verify before launching:** crawler's prompt template was tuned for Cerebras GPT-OSS-120B. Gemini may emit slightly different tool-call shapes (LiteLLM normalizes them, but worth a smoke test on one site first). Plan: run **Wikipedia** as the smoke test (low-stakes, no overlays, no auth, simple search).

### 8.4 Run orchestration

Two harnesses, one results directory:

**MorphNet:** reuse `experiments/run_eval.sh` with a custom task file:
```bash
./experiments/run_eval.sh --task-file experiments/comparison_10_tasks.json --max-subtasks 15
```
Output: `results/eval_<ts>/{site}/result.json` + traces.

**Crawler:** write a parallel runner (~50 LOC bash) that reads the same `comparison_10_tasks.json`, runs `crawler.main` per task, and emits a comparable `result.json` per task. Output: `results/crawler_eval_<ts>/{site}/`.

Then a single analysis script reads both directories and produces a comparison table:

| Site | MorphNet success | Crawler success | MorphNet duration | Crawler duration | MorphNet $ | Crawler $ |

### 8.5 Cost & time estimate

Per-task budget:
- MorphNet: ~$0.30–$0.80, ~3–6 min (15 subtasks × ~5 LLM calls × Gemini Flash + Pro for reflection).
- Crawler: ~$0.10–$0.30, ~1–3 min (Gemini Flash, batched actions, no reflector LLM).

Total for 10 sites:
- MorphNet: ~$5–8, ~30–50 min serial (can parallelize 4-way → ~10-15 min wall clock).
- Crawler: ~$1.5–3, ~15–30 min serial.

**Combined: ~$7–11, ~30 min wall-clock with parallelism.**

### 8.6 What we'll learn

The comparison surfaces several questions cleanly:

1. **Does crawler's batch-of-N actions actually beat morphnet's one-at-a-time?** If yes, on which task types? My prior: batching wins on form-heavy pages (Cleartrip, RedBus, ConfirmTkt) where the page state is highly predictable. Morphnet's per-action re-extraction wins on dynamic pages where the model needs to react to unexpected state (Swiggy with its location modal, Amazon with its sponsored-result interleaving).

2. **Does morphnet's planning tree help on multi-step tasks?** Tasks 4, 5, 6 (booking) require multiple correct decisions in sequence. If crawler's flat model fails on a step, does it recover? If morphnet prunes a wrong subtask, does it find the right approach?

3. **How big is the warm-start advantage?** Morphnet's existing profiles for swiggy/lego/bookmyshow/confirmtkt/cleartrip should give it an edge. Quantifying it tells us whether the MCP-discovery investment pays off.

4. **Where do both systems fail in the same way?** This is the most interesting signal — failure modes shared by both architectures point at problems that *aren't* about extraction or planning, but about something deeper (e.g. sites with anti-bot detection, sites that require login).

### 8.7 What I want to confirm before launching

1. **Are the tasks the right ones?** I picked read-only tasks to avoid actually booking trains/flights. Want me to swap any?
2. **Do you want a smoke test first?** I'd suggest running Wikipedia on both systems as a sanity check before launching the full 10. That catches Gemini-LiteLLM-tool-call issues on crawler's side cheaply.
3. **Headed or headless?** Both default to different modes. Some sites (BookMyShow, Swiggy) detect headless and block. I'd run headed for everything, accept the 2× slowdown.
4. **Should I wipe morphnet's existing site profiles for these 5 sites before running?** Default is no (keep the warm-start advantage in the data, report on it). Switch to yes if you want a strict cold-start comparison.

Once you give the green light, I'll write the comparison runner, do the smoke test, then launch the full 10.

---

## 9. Summary — the one-line takeaway

**Crawler** bets that a faithful, geometrically-aware page representation + stable DOM identity lets a single LLM call solve a page in one shot.

**MorphNet** bets that a clean accessibility-tree representation + branch/prune planning + per-site API discovery gradually replaces the LLM with deterministic execution.

They're not actually competitors. They're betting on different *time horizons*: crawler is optimizing per-page agency, morphnet is optimizing per-site amortization. A natural synthesis would take crawler's V6 extraction (DOM+AX, stamped identity) and feed it into morphnet's MCP discovery loop — but that's a much bigger conversation.
