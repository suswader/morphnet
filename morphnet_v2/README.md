# morphnet_v2

A browser agent that combines:
- **`browser-challenge/crawler/`'s page handling** (extraction, V5 markdown, batched action loop) — superior to v1 morphnet's representation
- **`morphnet/`'s branch/prune planner** — recovers from stuck states by trying different approaches (Phase 3)
- **A new tool manager** (Phases 4–5) — captures HTTP+JS bundles in real time during CU and replays them deterministically. Replaces v1's broken learner stack.

See `draft.md` for the full build plan, architecture rationale, and the parity-driven build sequence. Read it before touching code.

This README is **living documentation** — it grows as each chunk lands. Every public function, every notes event, every architectural decision is documented here. No black boxes. If a line of code doesn't have an explanation in this README, it doesn't ship.

## Build status

- ✅ **Phase 1 complete (1.1–1.7)** — CLI, notes, Chrome lifecycle, page lifecycle, action dispatch (20 methods ported from crawler/executor), HTTP + script capture with per-site context, Gemini async + curl_cffi utilities. Phase 5 replay thesis validated end-to-end on cleartrip (tier-3 Akamai site): 17/17 captured endpoints replay successfully, parameter substitution proven (BLR→PNQ original → DEL→BOM modified, real flights returned).
- ✅ **Phase 2 complete (2.1–2.5)** — `page_filter` (2.1), `v5_markdown` + mutation observer + episodes (2.2), `raw_session` + `page_agent.j2` (2.3), `page_agent` integration class (2.4), Phase-2 stub planner (2.5, since retired by Phase 3).
- ✅ **Phase 3 complete (3.1–3.4)** — `PlanningTree` / `ToolRegistry` data structures, Gemini-native function-calling planner LLM (one function declaration per action; tools become typed `invoke_<tool_id>` functions), the trigger-driven `Orchestrator` routing loop, and `timeline.py` — an offline reader that aligns CU actions ↔ HTTP r/r ↔ JS scripts (with precise V8 coverage deltas) per step. `passthrough_planner.py` is gone — `SessionManager.run_task` now hands every task to the Orchestrator. `mutation_types.py` was extracted from `session_manager.py` to break the `sm → planner → page_agent → schemas → sm` import cycle once all imports moved to top-of-file.
- ⬜ **Phases 4–6** outlined in `draft.md`, chunked later. Tool lifecycle states clarified: `verified / failing / script_drift` (no `unverified / probationary` middle states); no auto-discard.

## Files in this directory

### `session_manager.py` (🟡 partial)

The single I/O boundary between morphnet_v2 and the outside world. Owns Chrome (raw CDP + Playwright), Gemini, `curl_cffi`, and the Chrome subprocess. File I/O is exempt — `notes.py` writes directly.

What's currently implemented:
- `_launch_chrome(port, headless, user_data_dir=None)` — launches Chrome with the LEARNINGS phase-6c stealth flags, a fresh `tempfile.mkdtemp` profile per launch (unless caller passes one), and explicit `--remote-debugging-port` + `--user-data-dir`. Cross-session state cannot bleed: profile, process, and port are all distinct per launch. 
- `CDPSession` — raw WebSocket primitive over Chrome's CDP. `cdp.send(method, params)` for any CDP method without depending on Playwright. Each `send` and each received event mirrors to `notes.log()`.
- `SessionManager` — `start()`, `close()`, async context manager; holds both `cdp` (raw `CDPSession`) and `page` (Playwright `Page`). Seven-step start sequence: notes attach → Chrome subprocess → raw CDP attach → `Network.enable` + `Page.enable` → `Page.addScriptToEvaluateOnNewDocument(NAV_CAPTURE_INIT_SCRIPT)` → subscribe `Page.loadEventFired` then `Page.navigate` (subscribe-before-fire to avoid the race) → eager Playwright attach via `connect_over_cdp` reusing the existing context/page.
- `run_task(...)` — shared single-task runner used both by this module's CLI and by `experiments/run_eval.py`. Single source of truth for the run-one flow.

#### Page lifecycle (Chunk 1.4)

Five primitives ported from `crawler/browser_tools.py` and `crawler/master.py`. None require the MutationObserver to be installed (chunk 2.2 lifts that and adds the lighter `wait_for_settle`). All timeouts are kwargs.

- `await sm.wait_for_page_ready(*, poll_interval_ms=200, stable_window_ms=800, max_wait_ms=5_000) -> bool` — polls two integers (HTML-length + visible-text-length) every `poll_interval_ms`. When both stay constant for `stable_window_ms` consecutive ms, returns `True`. Returns `False` if `max_wait_ms` elapsed first. Catches `Execution context was destroyed` mid-poll and retries (resets the stable window). Used after navigation or initial load.
- `sm.url_before_action: str` — public attribute. Caller sets it directly: `sm.url_before_action = sm.page.url`. Acts as the baseline for `check_navigation`. Chunk 1.5 action dispatch sets it before each action; chunk 2.4 page_agent sets it at the start of a page-processing block and again after navigation is detected.
- `sm.check_navigation() -> Optional[str]` — synchronous URL diff with fragment stripping. Returns the new URL if navigation occurred since `url_before_action` was set, else `None`. Hash-only changes (`/foo#a` → `/foo#b`) are not flagged.
- `await sm.wait_for_navigation(timeout_ms=2000) -> bool` — event-driven via Playwright `page.wait_for_url(predicate, wait_until="commit")`. Used by chunk 1.5 when the action's intent is navigation. Returns `True` if URL changed within timeout. The synchronous `check_navigation` fallback still catches navs that fire after this returned `False`, so a slow nav doesn't get lost.
- `await sm.wait_for_dom_content_loaded(timeout_ms=3000) -> bool` — wraps `page.wait_for_load_state("domcontentloaded")`. Used right after `wait_for_navigation` returned `True`, before extraction runs against the new doc.

#### Action dispatch (Chunk 1.5)

20 public methods ported faithfully from `crawler/executor.py`. Each takes element AIDs (assigned by chunk 2.1 page_filter via `data-cdx-aid`) and returns an `ActionResult` with structured failure info (`success`, `reason_code`, `fail_subtype`, `blocker_aid`, `blocker_probe`). Every call mirrors to `notes.log(data_type="action")` via `_log_action`.

**Pre-flight (`_prep`):** every action calls this first — verifies the element exists, is visible (`checkVisibility({checkOpacity, checkVisibilityCSS})`), is not `:disabled` / `[inert]`, scrolls into view, captures bounding rect, optionally hit-tests center + 4 corners against `elementFromPoint` to detect overlay occlusion. On block: returns rich blocker info (`blocker_aid`, `overlay_aid`, z-indexes, nearby buttons).

**Public methods:**
- Click / type / read / scroll: `click_target`, `type_target`, `scroll_target`, `scroll_page`, `read_text_target`, `copy_paste`
- Keyboard: `key_press`, `press_escape`
- Drag suite: `drag_target`, `drag_target_cdp_dispatch`, `drag_offset`, `drag_slider`, `drag_batch_synthetic`, `probe_drop_zones`
- Pen: `draw_strokes`
- Pointer: `hover_target`
- Lifecycle: `wait_for_page_settle` (stub until chunk 2.2 lifts mutation_observer), `sleep`, `check_dismiss`
- Escape hatch: `click_selector` (CSS-selector path when no AID available)

#### HTTP + script capture (Chunk 1.6)

Always-on capture via raw CDP. Subscribes `Network.requestWillBeSent`, `responseReceived`, `responseReceivedExtraInfo` (raw Set-Cookie per LEARNINGS phase 9), `loadingFinished`, `loadingFailed`, `requestServedFromCache`, plus `Debugger.scriptParsed` (with `setAsyncCallStackDepth=32`) and `Target.setAutoAttach` for iframes / service workers.

Body fetching: each `loadingFinished` fires a background `_finalize_request` task that calls `Network.getResponseBody` and writes raw bytes (no truncation, no base64 decode-skip — the never-truncate rule).

Script source: `_link_script_source` runs in background per `scriptParsed`. For HTTP-loaded scripts (~95%), reuses bytes from the matching Network response (zero extra CDP roundtrip). For inline / eval / worker scripts, falls back to `Debugger.getScriptSource`. Dedups by SHA256 — same content under different scriptIds shares one file.

**Per-site context** (`morphnet_v2/sites/{site_name}/`):
- `scripts/{sha256}.js` — bytes, deduped by content
- `scripts/index.json` — `{sha256: {url, length, first_seen_ms, runs[]}}`
Persists across runs. Phase 4+ adds `profile.json`, `graphs/{graph_id}.json`, `tools.json`, `bundle/{bundle_hash}/`.

**Public API:**
- `sm.get_traffic(since_ts_ms=0) -> list[CapturedRequest]` — completed requests, in-memory
- `sm.clear_traffic()` — drop the in-memory buffer; disk records via notes are unaffected
- `await sm.cookies_snapshot() -> list[dict]` — `Network.getAllCookies`, includes JS-set cookies that never appear in HTTP. Logged as `cookies_snapshot`.
- `await sm.get_script_source(script_id) -> Optional[str]` — cached or freshly fetched

#### Outside-world utilities (Chunk 1.7)

- `await sm.call_gemini(model, contents, response_schema=None, ...) -> Any` — Gemini call with optional structured-output schema. Returns parsed JSON when schema given, else raw text. Pairs prompt + response in notes via shared `call_id` (uuid hex). Retries 3 attempts with exponential backoff on transient failures; one extra retry with doubled `max_output_tokens` on JSON-decode failure (truncation recovery). **Concurrency: many callers can `await call_gemini` simultaneously — the genai async client handles them in parallel via httpx's connection pool. No internal queue.**
- `await sm.make_http_session(impersonate="chrome") -> curl_cffi.requests.Session` — curl_cffi session with Chrome TLS/JA4 impersonation + cookies from the live Playwright context. Default `"chrome"` auto-tracks the latest version curl_cffi knows about (matches LEARNINGS phase 8c; passed Akamai for cleartrip in chunk 1.6 proof). Cookies are snapshotted at call time — call again if you need fresh state mid-session.

The genai client is initialized at module-load time from `GEMINI_API_KEY` / `GOOGLE_API_KEY` (read via `python-dotenv`). Importing fails fast if neither is set.

**Future direction (not now):** wrap `call_gemini` behind a thin LLM-agnostic protocol so Anthropic / OpenAI clients can drop in. Defer until we have a reason to switch.

#### Schema discipline

**Use `response_schema` for every LLM call where you'll parse the result.** The retry-on-JSON-decode logic only protects schema-using calls; freeform text calls fail loudly on malformed output. The planner / classifier / naming / param-extraction LLM calls (Phases 3-5) all use schemas — module-level constants paired with the call site.

### `notes.py` (✅ done, 401 LOC)

Lazy structured logger. Records every byte we get from the outside world.

**Core rule: never truncate.** Bodies, cookies, init scripts — always raw and verbatim.

Two layers:
1. `record.jsonl` — the timeline. One line per `log()` call. Lightweight, scannable.
2. Per-type files — the actual artifact (HTTP body, screenshot, prompt).

Parallel-experiment-safe via `contextvars`: each asyncio task tree gets its own active Notes.

Usage:
```python
from morphnet_v2 import notes
notes.log(data_type="prompt", data=prompt_str, model="gemini-3-flash-preview")
notes.log(data_type="screenshot", data=jpeg_bytes)

# experiment runner
notes.attach("experiments/results_v2", site_name="swiggy")
# ...run...
notes.detach()
```

Type-dispatched store handlers in `_store_*` write to:
- CDP messages → `cdp/messages.jsonl`
- HTTP → `http/index.jsonl` + `http/bodies/{rid}.{req,resp}`
- Cookies → `http/cookies/all_cookies_{ts}.json`, `http/cookies/set_events.jsonl`
- Page artefacts → `page/{ts}_{html,axtree,screenshot}`
- Script sources → `scripts/{script_id}.js`
- Events → `events/{navigation,console,exceptions}.jsonl`
- LLM calls → `llm/{call_id}.json`
- Actions → `actions/{action_id}.json`
- Metadata → `metadata.json`

### `page_filter.py` (✅ done, 3319 LOC)

The extraction engine. Lifted byte-for-byte from `browser-challenge/crawler/page_filter.py` + `crawler/schemas.py` + `crawler/config.py:ExtractorTuning`, collapsed into one file. Verified via AST diff: zero unexpected drift; the only changes vs crawler are the two documented boundary deltas. JS payloads (~32k chars + two ~5k/7k JS calls) match crawler byte-for-byte.

**Boundary deltas vs crawler:**
- `PageFilter(sm, use_justext=False, tuning=None)` — takes `sm: SessionManager` instead of having `page: Page` passed per call. Stores `sm` once, reuses.
- `_collect_axtree()` calls `self._sm.cdp.send("Accessibility.getFullAXTree")` instead of opening a fresh `page.context.new_cdp_session(page)` per call. Same page-target WebSocket → byte-equivalent CDP traffic, no per-call session open/detach.
- `build_aid_to_ax_map(cdp, ax_nodes)` — signature changed from `client` to `cdp` to make the contract explicit (any object with `.send(method, params)`). v2 passes `sm.cdp`; crawler passes a Playwright CDP session. Both have the same `.send` API.
- `_collect_blocking_relations` + `_collect_occlusion` take no `page` arg — use `self._sm.page.evaluate(...)` internally.

**Public API:**
- `await PageFilter(sm).run(snapshot, aid_allowlist=None) → PageFilterOutput`
  - `snapshot: PageSnapshot(url, title, html)` — caller composes it (`sm.page.url`, `await sm.page.title()`, `await sm.page.content()`).
  - `aid_allowlist` — set of AIDs that must be retained in the output (used after a blocked action, so the blocking container stays visible to the agent).
  - Returns `PageFilterOutput`: `containers[]`, `forms[]`, `buttons[]`, `actions[]`, plus aggregates (`container_count`, `blocked_action_count`, etc.) and DnD metadata (`dnd_library`, `synthetic_drag_accepted`).
- `pf.last_timing: dict[str, float]` — per-stage millisecond breakdown of the last `.run()` (js_collect, ax_map, ax_merge, python_build, blocking_and_occlusion, finalize, total).
- `pf.last_aid_to_ax_map: dict[str, dict]` — AID → AXNode mapping from the last run.

**Pipeline (one `.run()` call):**
1. **JS payload + AXTree fetch in parallel** — `_collect_payload` (~720 LOC of JS) walks the DOM, stamps every interactive node with `data-cdx-aid="aid-N"`, and returns `{containers, blocks, controls, dropZoneAids, dndLibrary, syntheticDragAccepted, pageEpoch}`. In parallel, `Accessibility.getFullAXTree` via `sm.cdp` returns the screen-reader view. AX failure is non-fatal — the run continues with JS data only.
2. **AX merge** — `build_aid_to_ax_map` calls `DOM.getDocument(depth=-1)` (single CDP roundtrip), walks the doc tree in Python to find every `data-cdx-aid` and pair it with its AXNode by `backendDOMNodeId`. Then `merge_axtree` overwrites JS-collected fields with authoritative AX signals: `ax_name`, `ax_role`, `disabled`, `checked` (tristate), `expanded`, `selected`, `current_value`, `has_popup`, `ax_description`, `focusable`, `ax_modal`. The "JS heuristic loses to AXTree on the same field" rule keeps the output canonical.
3. **Block filtering** — `_filter_blocks_global` strips lorem-like text, splits camelCase, dedupes by template key, suppresses repeated templates (≥3 occurrences with low unique-token ratio), applies utility/noise scores.
4. **Control selection** — `_build_controls` + `_select_controls` rank by `utility - noise`, keep explicit-form controls unconditionally, then top form-associated, then top standalone capped at `max_targets_total` (120).
5. **Container closure** — `_build_container_closure` keeps only containers that have a control/block, plus their parent chain to the root. `_build_containers` builds `ContainerEntity` per kept ID with text-block summaries, scoring, geometry, drop-zone flag.
6. **Form/button extraction** — `_build_forms` produces three passes: explicit `<form>` elements, then container-level pseudo-forms with ≥2 input-like controls, then sibling-aggregation for radio/checkbox groups across wrapper containers, finally a draggable-piece pass. `_build_buttons` takes button-like controls not already in a form, capped at `max_buttons_total` (40).
7. **Blocking relations** — `_collect_blocking_relations` does Python N² geometric pair enumeration (rectangle intersection + ancestry exclusion + viewport + high-z/fixed filter) → one JS `page.evaluate` that samples 5 points per target via `elementsFromPoint` → 60% hit threshold + cycle prevention.
8. **Occlusion** — `_collect_occlusion` runs another JS `page.evaluate`, 5-point sampling per button/form. Skips per-element occlusion inside scrollable containers (would report scroll-state as permanent blocking). Builds `TargetOcclusion` per target.
9. **Container ref attach + label classification** — back-fills `control_refs`/`button_refs`/`form_refs` on each container, marks `overlay_like` (semantic + geometric) and `section_like`.
10. **Action candidates** — `_build_actions` produces `click_button`/`complete_form` candidates sorted by `blocked_now` then `priority_score`, capped at `max_actions_total` (40).
11. **Optional justext text extraction** — if `use_justext=True`, runs `justext.justext(html)` on the raw HTML for a clean prose layer.

**Schemas** (all `extra="forbid", strict=True`, lifted from `crawler/schemas.py`):
- `PageSnapshot` — caller-provided `{url, title, html}`.
- `ViewportGeometry` — `{x, y, w, h}` as viewport ratios.
- `TargetOcclusion` — per-element pixel-occlusion result with `blocker_container_ids`.
- `FormBlockerStatus` — form-level blocking summary.
- `ContainerEntity` (28 fields) — every container's structural + classification + AX state.
- `FormControl` (32 fields) — every interactable's geometry + AX + slider metadata.
- `FormControlGroup`, `FormEntity`, `ButtonEntity`, `ActionCandidate`, `PageFilterOutput`.
- `ExtractorTuning` (105 fields) — every threshold, every score weight, every cap. Override via constructor.

**Module-level helpers:**
- `merge_axtree(payload, aid_to_ax)` — applied to raw JS payload before Python builders. Mutates in place.
- `build_aid_to_ax_map(cdp, ax_nodes)` — one CDP roundtrip + recursive Python walk.
- `_ax_prop_raw`, `_ax_bool`, `_ax_tristate`, `_ax_string` — normalize AXNode property values to schema types.

**Faithfulness verification:** `python3 -c "..."` AST diff vs crawler trio reports `Total unexpected drifts: 0`. JS payloads (3 of them) byte-identical. Schemas field-identical across all 12 models. Every method except the 5 expected boundary-port methods is byte-identical to crawler.

### `computer_use/` (✅ chunk 2.2 done — V5 markdown + mutation observer + episodes)

CU-only modules. Lifted byte-for-byte from crawler. Page interaction is gated by `SessionManager` — these modules expose JS strings + pure-Python helpers; sm methods are the only legal entry point for callers.

#### `computer_use/__init__.py`

Empty marker — makes `computer_use` a package.

#### `computer_use/schemas.py` (40 LOC)

Single home for CU-side type imports. Two halves:
1. **Re-exports from `morphnet_v2/page_filter.py`** — `ButtonEntity`, `ContainerEntity`, `FormControl`, `FormEntity`, `FormControlGroup`, `PageFilterOutput`, `PageSnapshot`, `ViewportGeometry`, `TargetOcclusion`, `FormBlockerStatus`, `ActionCandidate`. Lets `v5_markdown.py` import via `from .schemas import ...` exactly like `crawler/master_markdown.py` (zero AST drift).
2. **Mutation/episode schemas** — `MutationNodeRef`, `RawMutationRecord`, `TextDelta`, `AttrDelta`, `SubjectEpisode`. Defined here because they're CU-only — the planner doesn't see mutations. Byte-identical to `crawler/schemas.py:261-368`.

#### `computer_use/v5_markdown.py` (771 LOC, byte-identical to `crawler/master_markdown.py`)

Renders `PageFilterOutput` → the Markdown the LLM reads each turn.

**Public API:**
- `render_master_markdown(extraction) -> str`
- `render_master_markdown_with_meta(extraction) -> (str, meta_dict)` — meta is `{version, source_url, page_epoch, rendered_aids}`.

**V5 layout:**
- Header: URL, title, counts (`47 containers, 12 buttons, 3 forms, 1 blocking`).
- `## Overlays` — fixed-position containers (popups, modals). Each line: AID, z-index, heading, flags, blocking targets. Inline buttons + form controls below.
- `## Content` — the rest of the tree. Pruned of "scaffold" containers (empty wrappers with no text/controls) unless they're DnD drop zones.
- Each container line: AID, semantic tag (`<form>`, `<nav>`, ...), heading, flags (`overlay`, `modal`, `scrollable`, `animated`), blocking relations.
- Inline buttons in DOM order. >4 buttons → compact one-liner.
- Inline form controls with `_control_annotations` (exception-only: `disabled`, `checked`, `expanded`, `hasPopup`, `cursor:not-allowed`, `focusable`).
- Sliders show `[value/max, orientation]`. Canvases show `[WxH]`. Resize-handles show `[cursor: ...]`.

**DnD scaffold exemption** (`build_scaffold_exempt`): empty containers near draggable controls survive scaffold pruning so drop zones stay visible. Score = `exp(-0.2 * tree_distance) + (0.2 if 6 ≤ text_len ≤ 20 else 0)`. >0.5 → exempt. Library-fingerprinted droppables are always exempt.

#### `computer_use/mutation_observer.py` (139 LOC after refactor)

JS strings + pure-Python helper. **Page interaction lives on `SessionManager`** — see the sm methods below. These constants are not for direct use.

**Module exports:**
- `_OBSERVER_INJECT_JS`, `_FLUSH_JS`, `_PEEK_JS`, `_DISCONNECT_JS`, `_WAIT_SETTLE_JS` — JS constants (byte-identical to crawler).
- `summarize_mutations(records) -> str` — debug print helper (compact NEW/REMOVED/CHANGED/TEXT_CHANGED summary).

**JS observer behavior** (in `_OBSERVER_INJECT_JS`):
- Idempotent install — disconnects + reinstalls if already present, with fresh baseline.
- Watches `childList`, `attributes`, `characterData` on `document.body` subtree.
- Tracked attributes only: `disabled`, `hidden`, `aria-hidden`, `aria-disabled`, `class`, `aria-checked`, `aria-selected`. Class changes filtered to those affecting `hidden`/`disabled` substring.
- Stamps positioned containers (fixed/absolute with z-index) and interactive elements with `data-cdx-aid` (continuing PageFilter's counter via `maxAid` scan).
- Module-level counters (`__cdxObsCounter`, `__cdxSeqCounter`) persist across batches; per-batch state (`_rootRefs`, `_subjectRefs`) clears on `__cdx_flush`.
- 200-record buffer cap (oldest evicted) — pull-model, Python flushes after each action.

#### `computer_use/mutation_episode_builder.py` (251 LOC, byte-identical to crawler)

Synthesizes per-element life-stories from raw observer records.

**Public API:**
- `build_episodes(records) -> list[SubjectEpisode]` — group by `subject.obs_id`. Tracks `appeared_after_step`, `disappeared_after_step`, `text_first/last`, `text_deltas`, `attr_deltas`, hint fields.
- `apply_persistence_results(episodes, disconnected_ids, final_step_index)` — mark vanished subjects.
- `reconcile_episodes(episodes, known_aids)` — set `present_in_final_extraction` against post-batch V5.
- `format_batch_events(episodes, surface_all_deltas=False) -> str` → `## During Batch` Markdown block.

**Suppression rule** (default `surface_all_deltas=False`): single-delta episodes (A→B) on subjects still present in V5 are suppressed — agent has A from action result + B from next V5 extraction. Surfaces only:
1. Multi-delta episodes (A→B→C, where B would be lost).
2. Lifecycle pairs (appeared + disappeared in the batch — agent never sees them otherwise).

#### `SessionManager` mutation methods (chunk 2.2 wiring)

All mutation page-touching lives here. Internals route to the JS constants in `mutation_observer.py`.

- `await sm.install_mutation_observer() -> str` — inject the observer. Returns `'v2 installed (maxAid: N)'`. Idempotent.
- `await sm.flush_mutations(batch_id=None) -> list[RawMutationRecord]` — drain JS buffer, construct Pydantic records. Generates `batch_id` UUID if omitted.
- `await sm.peek_mutation_count() -> int` — buffer length without consuming.
- `await sm.wait_for_settle(quiet_ms=80, max_ms=500) -> int` — light settle. Returns mutation count seen during the wait. Heavy settle (no observer needed) is `wait_for_page_ready` from chunk 1.4.
- `await sm.disconnect_mutation_observer() -> None` — cleanup.

The chunk 1.5 `wait_for_page_settle` action is now real — it calls `sm.wait_for_settle()` and reports the new mutation count in the action result message.

**Faithfulness verification:** AST diff vs crawler reports zero unexpected drifts across `master_markdown.py` (32 funcs identical), `mutation_observer.py` (5 JS constants identical, `summarize_mutations` identical), `mutation_episode_builder.py` (6 funcs identical), and the 5 mutation schemas (every field identical). The page-touching helpers crawler had at module level (`install_observer`, `flush_mutations`, `peek_mutation_count`, `wait_for_settle`, `disconnect_observer`) are intentionally **not** lifted — their job moves to sm methods to honor the I/O boundary. One pre-existing pyright error in `v5_markdown.py:184` (`list[str | None]` vs `list[str]`) is inherited from crawler verbatim.

### `computer_use/raw_session.py` (✅ chunk 2.3 done — the LLM tool loop)

The multi-turn LLM loop for ONE CU **step** (one page-worth of CU activity — see `project_morphnet_v2_step_terminology`). Lifted from `crawler/master.py:_run_raw_session` with three v2 adaptations: Gemini-native function calling (instead of LiteLLM-normalized tool calls), pure dependency injection (no `MasterOrchestrator` coupling), and the deterministic success rule per architecture rule 11.

**Public surface:**
- `render_page_agent_prompt(user_goal, dnd_library) -> str` — loads `morphnet_v2/prompts/page_agent.j2` (Jinja2) and renders. Returns the system-instruction text. `dnd_library` gates the `probe_drop_zones` action via the template's `{% if dnd_library == 'html5-native' %}` block.
- `build_tools_schema(dnd_library) -> list[Tool]` — constructs Gemini function declarations for `execute_actions` (one OBJECT with all action fields, `action` is the required enum discriminator) + `report` (single `message: string`). The `probe_drop_zones` action is added to the enum only when `dnd_library == 'html5-native'`.
- `class RawSessionRunner(*, sm, model, max_turns, batch_executor, dnd_library=None, max_output_tokens=8192, thinking_budget=2048, temperature=0.7)` — holds dependencies. `batch_executor: Callable[[list[dict]], Awaitable[BatchResult]]` is the chunk 2.4 callback that runs the actions + does the post-batch re-extract + returns the new V5.
- `await runner.run(user_goal, initial_v5) -> StepResult` — drives the loop once.

**Per-step loop body (per turn):**
1. `sm.call_gemini(model=..., contents=messages, tools=tools_schema, system_instruction=prompt, ...)` — function-calling mode.
2. `_parse_response(resp)` — **typed extraction, no string matching.** Iterates `candidate.content.parts` and routes by attribute: `part.function_call is not None` → tool call; `part.thought` → reasoning trace (mirrored to notes as `data_type="thinking"`); `part.text is not None` → text content (for the nudge case). Extracts token counts from `resp.usage_metadata`.
3. Append the model's raw parts to `messages` (preserves Gemini's thread invariants — function_call followed by function_response).
4. **No function_call branch:** if model emitted plain text → one-shot user nudge ("use the tool, don't emit text"), continue. Otherwise → exit `NO_TOOL_CALL`.
5. **Per function_call:**
   - `report` → store `message`, set exit_reason=`REPORT`, break.
   - `execute_actions` → `await batch_executor(actions)` → `BatchResult`. Append `function_response` Content. Update `action_log`, accumulators. Run spiral detection. If `batch.new_v5` is set → store as `pending_v5`. If `batch.navigated` → exit_reason=`NAVIGATED`, break.
   - Unknown tool → error function_response, continue.
6. After processing this turn's tool calls: apply `pending_v5` via `_rotate_v5`; if spiral triggered, append nudge user message + reset counts; break on exit_reason.
7. After max_turns elapsed → exit_reason=`MAX_TURNS`.
8. Synthesize `StepResult` (see rule below).

**V5-deletion fix (`_rotate_v5`).** The initial V5 at message index 0 is **never deleted**. When a fresh V5 arrives mid-step, we delete the previously-appended V5 (which lives at some index > 0) and append the new one. Cost: one extra V5 stays in the thread. Gain: Gemini's strict validator accepts the thread — the very first model `function_call` must be preceded by a user turn (index 0 IS that turn). Without this fix, the first call returns INVALID_ARGUMENT.

**Deterministic success rule (`_synthesize_success`).** Architecture rule 11. The planner's `tree_update.outcome` consumes `StepResult.success`:
```
success = (exit_reason == NAVIGATED)
       OR (exit_reason == REPORT AND last_batch_clean)
```
- `NAVIGATED` — URL actually changed. The navigating action by definition succeeded (mechanical signal). Trust.
- `REPORT` — model called the report tool. Trust only if the most recent batch executed cleanly (`last_batch_clean`). Catches the LLM-hallucination case: model gives up after failed clicks and calls `report("done")` anyway → `last_batch_clean=False` → `success=False`.
- Other exit_reasons (`MAX_TURNS` / `SPIRAL` / `NO_TOOL_CALL`) → `False`.

**Spiral detection (`_update_spiral_counts`).** Lifted verbatim from `crawler/master.py:2577-2613`. Regex-parses the `BatchResult.text` for failure lines (`(type|click|action) failed aid=aid-N`); counts `(action_kind, aid)` pairs; after 3 consecutive failures on the same pair, the loop appends a "stop, re-read the page, try a different approach" user message and resets the counter. A successful `clicked/typed/copied` on the same AID resets that AID's count mid-step. The regex is the ONLY string-matching path in this module — see `draft.md` carve-out #15. The planning tree's `detect_repeated_approaches` (Phase 3) is the cross-page backup for loop detection.

**Action-log extraction (`_extract_action_lines`).** Pulls per-action result lines out of `BatchResult.text` for `StepResult.action_log`. Filters out the `## During Batch` heading + its bullet lines (mutation events stay in the function_response sent back to the model, but they're not "actions" in the planner's sense). Result is `list[str]` like `["[1] clicked aid-5", "[2] typed 'Alan Turing' into aid-5", ...]`.

**Notes mirroring.** Per turn: `thinking_text` → `notes.log(data_type="thinking", turn=...)` for diagnostic visibility (function-calling mode has no `reasoning` field in the response). Every `sm.call_gemini` already pairs prompt+response via call_id. Every `batch_executor` action goes through `sm._log_action` → `notes.log(data_type="action")`. No extra notes-writing in this file.

**Prompts policy.** All prompts live in `morphnet_v2/prompts/` as `.j2` files (per `feedback_prompts_in_separate_dir`). `page_agent.j2` is byte-identical to `crawler/templates/page_agent.j2`. Future prompts go in the same directory.

**No freeform LLM generations.** `sm.call_gemini` now requires exactly one of `response_schema` (structured output) or `tools` (function-calling) — raises `ValueError` if both or neither (per `feedback_no_freeform_llm`). The CU loop uses the `tools` path.

### `computer_use/page_agent.py` (✅ chunk 2.4 done — CU integration class, 1472 LOC)

The CU policy layer. Faithfully lifted from `crawler/master.py:MasterOrchestrator` per-page logic. Owns all the state + action dispatch + post-batch pipeline that crawler kept on its orchestrator.

**Lifted methods (22 from crawler, full parity):**
- `resolve_target` (int|str → "aid-N")
- `_register_entity`, `_build_label_maps` (entity registry + label maps)
- `enrich_aid_refs`, `_resolve_to_visible`, `check_target_blocked` (reference enrichment)
- `process_mutations` (live blocking-graph maintenance — keeps in-memory extraction in sync between re-extracts)
- `filter_ax_probe_candidate_aids`, `_ax_probe_new_nodes`, `apply_ax_modal_signals` (AX modal probing)
- `_ax_enable_push`, `_ax_disable_push`, `drain_ax_updates`, `process_ax_updates`, `_build_ax_backend_map` (AX push events — vestigial per crawler's note that Playwright doesn't deliver them; lifted for parity)
- `_format_blocker_probe`, `_enrich_unknown_blocker` (blocker probe formatting)
- `record_action`, `resolve_step_refs` (action log + `$stepN` resolution)
- `run_re_extract`, `run_fast_re_extract` (heavy + light re-extraction paths)
- `_run_one_action` (the full 380-LOC action-dispatch switch with intent handling, drag-mode validation, navigation detection)

**v2-only methods (the two integration points):**
- `run_step(user_goal) -> StepResult` — single public entrypoint. Reset state → wait_for_page_ready → initial extract + install observer + render V5 → instantiate `RawSessionRunner` with `batch_executor` closure → drive loop → return result.
- `_execute_batch(actions) -> BatchResult` — the `batch_executor` body. Lifted from crawler's `_handle_execute_actions`. Drag-batch pre-scan, per-action mutation flush + nav check, blocker-aware click retry (one-shot), post-batch persistence + re-extract + episode formatting.

**Three new sm wrappers added (chunk 2.2 section):**
- `await sm.mark_mutation_step(n)` — wraps `window.__cdx_markStep(N)` so mutations during action N get tagged with step_index.
- `await sm.check_persistence() -> list[str]` — wraps `window.__cdx_checkPersistence()` to find observer roots that vanished during the batch.
- `await sm.clear_observer_refs()` — wraps the clear-refs JS after persistence check.

Plus AX-push helpers (`sm.enable_ax_push`, `sm.disable_ax_push`) — best-effort, vestigial per crawler.

**Parity verification:** AST diff vs crawler shows 23 shared methods. After stripping docstrings + type annotations, only 4 trivial differences remain: `@staticmethod` on `resolve_target` (it never used self), import path on `_ax_tristate`, explicit `bool()` wrap on a `re.search` truthiness check, and dropping inline `dict[str, str]` annotations on `_build_label_maps` empties (the attrs are typed in `__init__`). **Zero behavioral drift.**

**Open optimization candidates** (per `feedback_no_simplification_lifts` — note in `draft.md`, revisit after 2.6 parity):
- `process_mutations` live blocking-graph maintenance: works correctly but might be removable if v2's per-batch re-extract proves sufficient (crawler kept it as an optimization to skip extra re-extracts).
- AX push events: vestigial in crawler (Playwright doesn't deliver them); v2's raw sm.cdp MIGHT actually deliver them. Possible future improvement.
- `_enrich_unknown_blocker`: marginally redundant since the post-batch re-extract surfaces the blocker on the next turn via `aid_allowlist`. Worth keeping for mid-batch model feedback.
- `resolve_step_refs($stepN)`: model rarely uses these in practice; `copy_paste` covers the canonical case.

### `passthrough_planner.py` (✅ retired in Chunk 3.3)

Removed. The Phase-2 stub planner (linear `PageAgent.run_step` loop with crawler-style fingerprint-based loop detection) was replaced by the Orchestrator in `planner.py`. `SessionManager.run_task` now builds the Orchestrator lazily; no public surface change. `PageAgent.last_extraction` (added as a 2.4 addendum to support the retired fingerprint check) stays — the Orchestrator doesn't read it, but it's free and may help future debugging.

### `timeline.py` (✅ chunk 3.4 — offline temporal representation, ~250 LOC)

Reads one task's notes dir and emits `step_frames.json` aligning three streams per CU step:

1. **CU actions** — `actions/{aid}.json` + `record.jsonl` action rows in the step window.
2. **HTTP request/response pairs** — paired by `request_id` from `http/index.jsonl`. Each `LinkedRequest` carries the full `initiator_stack` plus `initiator_scripts` (deduped scriptIds that appear anywhere in that stack — the causal evidence that JS issued this network call).
3. **JS scripts** — one `ScriptUse` per scriptId touched in the step. `evidence ∈ {parsed, initiator_stack, coverage}`. The `coverage` channel uses `Profiler.takePreciseCoverage` snapshots taken at step boundaries; `timeline._coverage_delta` subtracts adjacent snapshots per-function and yields `executed_functions` + `coverage_delta_count` (per-step call-count delta).

**Step windows** are bracketed by `step_boundary` events the Orchestrator emits at `tree.branch(...)` (phase=`start`) and at `tree.complete_current` / `tree.prune` (phase=`end`). The payload is stored via `notes._store_misc` (no custom handler) and carries `{coverage, url}`.

**Public surface:**
- `dataclass StepFrame(step_node_id, start_ts_ms, end_ts_ms, start_url, end_url, actions, requests, scripts, timeline)`
- `dataclass LinkedRequest`, `ScriptUse`, `TimelineEvent`
- `build(notes_dir: Path) → list[StepFrame]` — offline reader. No mutation, no LLM, no browser.
- `write_step_frames(notes_dir) → Path` — writes `step_frames.json` next to `record.jsonl`.
- CLI: `uv run python -m morphnet_v2.timeline <notes_dir>`.

**Why offline.** Phase 4 tool_builder will eventually build StepFrames in real time, but emitting offline now de-risks the 100-task eval run: zero new hot-path code, and the schema becomes the contract real-time synthesis must satisfy.

**SessionManager additions** (chunk 3.4): `Profiler.enable` + `Profiler.startPreciseCoverage(callCount=True, detailed=True)` at session start (2 lines added to `start()`); new `async take_coverage_snapshot()` method that returns the cumulative coverage list. Orchestrator calls it at every step boundary via the private `_log_step_boundary(phase, step_node_id)` helper.

### `mutation_types.py` (✅ chunk 3.3 — cycle-break extraction)

Pure-stdlib + Pydantic module that owns the CU-side mutation primitives (`MutationNodeRef`, `RawMutationRecord`, `records_from_raw`, `summarize_mutations`). Lifted unchanged out of `session_manager.py` when chunk 3.3 moved all imports to top-of-file and exposed the cycle `session_manager → planner → page_agent → schemas → session_manager`. Anything in the codebase can import from here without re-creating that cycle. Re-exported through `computer_use/schemas.py` so existing import paths (e.g., `from .schemas import MutationNodeRef`) keep working byte-for-byte.

### `planner.py` (✅ chunks 3.1 + 3.2 + 3.3 done — PlanningTree + ToolRegistry + Planner LLM + Orchestrator)

Chunk 3.1 added the pure-stdlib data structures (`PlanNode`, `PlanningTree`, `ToolEntry`, `ToolRegistry`). Chunk 3.2 added the Gemini-native function-calling planner (`SlotDef`, `PlannerDecision`, `build_planner_function_declarations`, `render_planner_prompt`, `async call_planner`). Chunk 3.3 added the `Orchestrator` class — the trigger-driven routing loop that owns PageAgent/PageFilter (+ future ToolExecutor) internally and is built lazily by `SessionManager.run_task`.

**`PlanNode`** — one step in the planning tree.
- `node_id, parent_id, kind: "cu"|"tool"`
- `tool_id, tool_user_intent` — set iff `kind="tool"`
- `summary, outcome` — written in the NEXT iteration by the planner's `tree_update`
- `children: list[str]`
- No `intent` field. CU steps don't need one (CU reads the task + the tree directly). Tool steps are identified by `tool_id` + `tool_user_intent` — the tool's name IS the descriptor.
- `status` is derived (not stored): a node is `"active"` iff it equals `tree.current_id`, else `"completed"`/`"pruned"` from `outcome`.

**`PlanningTree`** — per-task branch/prune memory.
- `create_root(task: str) → "plan_0"` — stores the user task at tree level (rendered at the top of `get_context_for_planning`). Idempotent only on first call.
- `branch(*, kind, tool_id?, tool_user_intent?) → new_node_id` — appends child under current, makes it current. `kind="tool"` requires `tool_id`.
- `complete_current(summary)` / `prune(summary)` — write outcome + 1-line summary, move current to parent.
- `get_context_for_planning() → str` — renders task header + tree. Per-node displays kind, tool_id (if any), outcome, summary. Current node marked `← CURRENT FOCUS`.
- `detect_repeated_approaches() → str | None` — scans pruned summaries for >40% word-overlap clusters of 2+. Returns a warning string for the planner prompt, or None.
- `to_mermaid() / save_visualization()` — Mermaid graph (green=success, red=failure, blue=active) persisted via `notes.log(data_type="planning_tree")` to `planning/{ts}.mermaid`.

**`SlotDef`** — one tool parameter (chunk 3.2).
- `name`, `type ∈ {string, number, integer, boolean, array, object}`, `required: bool`.
- `description` — text shown in the Gemini function declaration.
- `examples: list[str]` — values that worked in past invocations. Embedded into the slot's description as "Past values that worked: 'X', 'Y'". Phase 4's `tool_builder` populates this from observed runs; Phase 3 leaves it empty.

**`ToolEntry`** — one registered tool.
- `tool_id` doubles as the human-readable name.
- `capability_statement` — one-sentence description the planner reads.
- `slots: list[SlotDef]` — what params the planner must produce when invoking this tool. Translated into per-tool Gemini function-call schemas at planner-call time.
- `lifecycle ∈ {verified, trusted, failing, discarded}` (4 internal states; `discarded` is never shown to the planner).
- `success_rate: float` (0.0–1.0), `total_runs: int` (divisor + min-runs gate).

**`ToolRegistry`** — per-task tool lifecycle.
- `register(tool)`, `record_success(tool_id)`, `record_failure(tool_id, reason)` — counters + deterministic threshold transitions.
- Thresholds (private class constants): `MIN_RUNS_BEFORE_DOWNGRADE=3`, `PROMOTE_THRESHOLD=0.8`, `FAILING_THRESHOLD=0.5`, `DISCARD_THRESHOLD=0.2`.
- `available_for_planner()` filters out `discarded`, orders trusted → verified → failing.
- `format_for_planner()` renders one line per tool: id, capability, lifecycle, success%.
- Phase 3: orchestrator constructs an empty registry per task. Phase 4 populates it. Phase 5 reads from it. Persistence to `morphnet_v2/sites/{site}/tools.json` lands in Phase 4.

**Why this shape (vs old morphnet's `morphnet_orchestrator.py:238-497`).**
- Dropped `BranchSummary` dataclass — replaced with a single `summary: str` written by the planner LLM in `tree_update`. Action-level detail (per-click results) stays in `StepResult` on the orchestrator side; only the planner's distillation reaches the tree.
- Dropped `PlanNode.intent` — redundant with `kind` + `tool_id` + planner's distilled summaries.
- Tool lifecycle pulled out of the tree into `ToolRegistry` — tools are persistent across tasks; tree is per-task. They share only `tool_id` strings.
- Lifecycle transitions are deterministic threshold rules. No LLM judgment.

---

**Chunk 3.2 — Planner LLM call + function-calling design.**

The planner uses **Gemini's native function-calling** (same mode as CU's `execute_actions`), not `response_schema`. Each turn, the planner sees a set of function declarations and emits exactly one function_call. The function name IS the `planning_action`; the function args ARE the typed structured response. No free-form dict, no hallucinated tool params.

**Function declarations exposed each turn:**

- `continue_cu` — hand back to the CU (browser) agent.
- `complete_task` — return `final_answer` synthesized from prior step summaries.
- `give_up` — no viable next move.
- `invoke_<tool_id>` — generated dynamically from `registry.available_for_planner()`. Each tool gets its own function with typed slot params from its `SlotDef`s.

Every function declaration carries the same 5 common fields (built by `_common_planner_props`):
- `tree_update_outcome` (nullable string enum success|failure)
- `tree_update_summary` (nullable string)
- `reasoning` (required string)
- `confidence` (required number 0.0–1.0)
- `evidence_sources` (required string array)

Tool slots are added to `invoke_<tool_id>` declarations from `ToolEntry.slots`. Required slots become required function params; optional slots are not in the `required` list. Slot descriptions include past examples that worked.

**Public API (chunk 3.2):**

- `PlannerDecision` dataclass — typed extract from the function_call: `planning_action`, `tool_id`, `tool_user_intent`, `final_answer`, `tree_update_outcome`, `tree_update_summary`, `reasoning`, `confidence`, `evidence_sources`.
- `build_planner_function_declarations(registry) → list[FunctionDeclaration]` — assembles the three static actions plus one `invoke_<tool>` per registered tool.
- `render_planner_prompt(*, task, tree, registry, trigger, browser_state) → str` — renders `prompts/planner.j2`. `browser_state` shape depends on trigger:
  - `task_start`: `{"url": str, "v5": str}`
  - `cu_returned`: `{"url", "v5", "cu_success", "cu_exit_reason", "cu_total_actions", "cu_failed_actions", "cu_report_message", "cu_action_history"}`
  - `tool_returned`: `{"url", "tool_id", "tool_user_intent", "tool_http_status", "tool_success", "tool_error", "tool_response_digest"}` — no V5 (browser is stale; orchestrator re-renders before any CU fallback)
- `async call_planner(sm, *, task, tree, registry, trigger, browser_state, model="gemini-3-flash-preview", thinking_budget=2048, max_output_tokens=8192, temperature=0.4) → PlannerDecision` — renders, calls Gemini in tools-mode, parses the function_call. Single LLM call per turn.

**`prompts/planner.j2`** — Jinja template. Order: role framing (distill/route/compile/terminate) → task → planning history (linear list of steps with summaries) → trigger-specific block (mechanical signals + action history + V5 for cu_returned; tool response digest for tool_returned) → summary-writing instruction with task-type examples (read/locate, multi-page aggregation, action-with-errors). No markdown tables — bullets only (tables are token noise for an LLM).

**Why function-calling instead of `response_schema`.**
- Schema-enforced tool params with no hallucination risk — Gemini either generates valid args or the call fails at the API level.
- No discriminated-union problem — each tool has its own function with its own slots; no need to enumerate all 10 tools' params in one schema and ask the model to set most fields to "NA".
- Mirrors CU's design (`execute_actions` is also a function declaration). Consistent across the codebase.

---

**Chunk 3.3 — Orchestrator routing loop.**

`Orchestrator(*, sm, max_steps=10, max_turns_per_step=60)` owns the per-task trigger-driven state machine. `SessionManager.run_task(task)` constructs one lazily and forwards the call. The Orchestrator owns PageAgent + PageFilter internally (Phase 5 will add ToolExecutor); SessionManager does not construct or pass any higher-layer module — preserving `feedback_dependency_direction`.

**Loop shape (per iteration):**
1. `call_planner(...)` with the current `trigger` ∈ `{task_start, cu_returned, tool_returned}` and `browser_state` dict.
2. Apply `tree_update_outcome` + `tree_update_summary` to the in-flight node (skip only on `task_start`; default to `success` + placeholder summary if the LLM omitted them, so the tree stays structurally valid).
3. Check termination: `complete_task` / `give_up` set `tree.task_exit` and return; `iteration == max_steps` clamps to `max_steps` exit.
4. Dispatch the new step. `continue_cu` → `tree.branch(kind="cu")` then `page_agent.run_step(task)`; next trigger = `cu_returned`. `invoke_tool` → `tree.branch(kind="tool", tool_id, tool_user_intent)` then a Phase-5 ToolExecutor replay (Phase 3 synthesizes a 503 failure since no executor exists yet).
5. Rebuild `browser_state` for the next planner turn:
   - `cu_returned`: fresh V5 (from PageAgent's last extraction or a re-extract if navigated) + mechanical signals from `StepResult` + full action history (untruncated, per `feedback_never_truncate`).
   - `tool_returned`: tool_id + tool_user_intent + HTTP status + error + response digest. No V5 (browser may be stale; orchestrator re-renders only when the next decision routes back to CU).

**Token + result bookkeeping.** Every planner-call's `input_tokens` / `output_tokens` and every CU step's totals get accumulated onto `PlanningTree.total_input_tokens` / `total_output_tokens`. At termination the Orchestrator sets `tree.task_exit`, `tree.final_answer`, `tree.set_final_url(...)`, and calls `tree.save_visualization()` exactly once (Mermaid graph to disk). `tree.success` and `tree.step_count` are derived properties — no separate `TaskResult` dataclass.

**No reflector** (architecture rule 11). Step outcome comes from the planner LLM's `tree_update.outcome`, derived from deterministic mechanical signals shown in the prompt (`StepResult.success` and `HTTP status`). No second-guessing in code.

**task_metadata is write-only.** SessionManager stores `task_metadata` (which carries `expected_answer` in eval runs) only to feed `notes.log(data_type="metadata", ...)`. It is never propagated to the Orchestrator's call path; no LLM prompt can see it. See `feedback_task_metadata_write_only`.

### Files pending (documented as chunks land)

- `experiments/parity_v2.py` + `experiments/diff_parity.py` (Chunk 2.6) — 50-task side-by-side parity experiment vs crawler (now compares Orchestrator vs crawler directly)
- `tool_builder.py`, `tool_executor.py` (Phases 4–5) — outlined in `draft.md`

## Reading order

1. `draft.md` — full architecture, core ideas, build plan
2. This README — what's currently in the codebase
3. The source files themselves
