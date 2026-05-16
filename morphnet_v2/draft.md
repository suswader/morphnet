# morphnet_v2 — build plan

A merge of two systems we benchmarked on the same 50-task corpus:

- **`browser-challenge/crawler/`** — page-handling layer is genuinely better than ours. Concrete reasons listed below. Score: 20/50 correct on Gemini.
- **`morphnet/`** — branch/prune planner is the part that works. Learner / manifest / executor stack is broken (19 successful CU runs → 2 unexecutable graphs). Score: 11/50, with most failures caused by a Chrome lifecycle bug we've now fixed in v2.

morphnet_v2 takes the best of each, with a strict layering rule, a thesis about how API tools can be captured-and-replayed cheaply, and a discipline that **every line shipped is understood and documented in `README.md`**. No black boxes.

The build is **parity-driven**: instead of jumping straight to a fancy planner, we first lift crawler's CU stack into morphnet_v2, build a stub pass-through planner, and verify that running tasks through morphnet_v2 produces near-identical results to running them through crawler directly. Only when that parity holds do we add the real planner with branch/prune.

## What crawler actually does better (concrete reasons we lift it)

1. **`data-cdx-aid` DOM stamping** (`crawler/page_filter.py` JS payload). Element IDs written into the live DOM as attributes; survive across re-renders. Morphnet's per-extraction integer IDs reshuffle on every re-render. **This single property is what makes batched action emission possible** — the model can plan 11 actions referencing AIDs and trust they're still valid by action 7.
2. **Single mega `page.evaluate()` extraction.** AID stamping + container/block/control collection + DnD fingerprinting + drop-zone probing in one round-trip. Morphnet pipeline does multiple separate evaluates.
3. **5-point hit-test occlusion.** Every form control + button gets probed at 5 points (center + 4 inset corners). Catches CSS-based occlusion (cookie banners, z-index 999 overlays) that AXTree-only misses.
4. **Geometric blocking detection.** N² container enumeration with hit-tests. V5 markdown explicitly tells the LLM "aid-12 is blocked by aid-5." Morphnet doesn't — agent has to figure it out from visual inspection alone.
5. **V5 markdown's two-section structure** (`## Overlays` for fixed-position blockers, `## Content` for everything else). Surfaces blocking explicitly. Compact button rendering when count > 4. Exception-only annotations.
6. **Mutation observer + episodes.** DOM mutations between extractions surface as a `## During Batch` section in the next V5. The LLM sees what changed after each batch without re-extracting + diffing manually.
7. **Batched action emission.** Prompt explicitly says "one call, one batch, markdown-first." 11+ actions per LLM turn vs morphnet's 1.
8. **DOM stability detection** (`crawler/browser_tools.py:_wait_for_dom_stability`) — polls innerHTML/innerText length, treats stable for N ms as ready. Plus `_prep` (visibility check + scroll-into-view + hit-test) before every action. Crawler does this WITHOUT depending heavily on Playwright primitives, which makes it faster and more reliable than morphnet's Playwright-dependent approach.
9. **Spiral detection.** `(action_type, aid)` failure pairs tracked; if same pair fails 3+ times, model gets nudged toward a different approach.
10. **Stealth launch flags + `start_new_session=True`** for Chrome. We already lifted this into `session_manager.py:_launch_chrome`; eliminated the 31/50 morphnet Chrome lifecycle failures.

## Read this first — five core ideas

### 1. Session manager is the only layer that talks to Chrome

Everything above session_manager — planner, computer_use, tool_builder, tool_executor — speaks Python. None of them touch Playwright directly, none of them shell out to `curl`. session_manager owns: Chrome subprocess lifecycle, raw CDP WebSocket, Playwright bindings, page-load detection, retry-on-failed-interaction, network capture, JS source capture, screenshots, action execution, HTTP replay via `curl_cffi`.

If a layer above wants to know "did the page load?" it calls `await session.wait_for_page_ready()`. It does NOT poll. It does NOT retry. It does NOT inspect Playwright objects.

### 2. Two routes from a step, with trigger-driven decisions

```
                ┌─────────────────────────────┐
                │ planner: routing decision   │
                │ at trigger points only      │
                └──────────────┬──────────────┘
                               │
            ┌──────────────────┴────────────────┐
            ▼                                   ▼
   ┌──────────────────┐              ┌──────────────────────┐
   │ tool_executor    │              │ computer_use         │
   │ replay HTTP+JS   │              │ extract → V5 → LLM   │
   │ via curl_cffi    │              │ → action → repeat    │
   └──────────────────┘              └──────────────────────┘
            │                                   │
            │                  ┌────────────────┘
            ▼                  ▼
   ┌─────────────────────────────────────┐
   │ tool_builder (real-time, async)     │
   │ - observes traffic during CU         │
   │ - builds graphs as nodes accumulate  │
   │ - verifies, names, registers         │
   └─────────────────────────────────────┘
```

A **step** = one page-worth of CU activity OR one tool invocation. The planner runs at exactly three trigger points: task start, CU returned, tool returned. At each trigger it picks `continue_cu`, `invoke_tool`, `complete_task`, or `give_up`. The planner does NOT pre-decompose steps; they emerge from trigger transitions.

The four routing outcomes:

- **CU pass** → `tree.complete_current()` with `SubtaskResult.success` as the deterministic signal. Next trigger fires.
- **CU fail** → `tree.prune()` with reason from CU's action log. Planner picks next route on next trigger.
- **Tool pass** → step marked complete via `ReplayResult.http_status == 200`. If the planner then picks `continue_cu`, the orchestrator `goto`s the URL the tool's JS yielded during replay (the URL the page would have navigated to if the action ran via UI). Otherwise no navigation.
- **Tool fail** → step marked failed (status != 200). Browser is in its active state (whatever was rendered before the tool ran — there is no "known URL" to re-render to). Planner can pick `continue_cu` to resume from that active state, pick another tool, or `give_up`.

No A/B testing loop in v1 — investigation of tool failures is manual, conservative (we don't auto-discard).

### 3. Three parameter types per HTTP request — the tool capture thesis

Every API call has three kinds of params:

- **`user_intent`** — search query, source/destination, date. The planner fills these at replay.
- **`chained`** — extracted from a previous response on this same flow. Station codes from autocomplete, session IDs.
- **`website_generated`** — CSRF tokens, request IDs, signed timestamps. Generated by the website's JS at runtime.

The classical learner (morphnet v1) tried to find the *exact* minified JS function that generates each website-generated value. 5 strategies. All failed for both real graphs we built — `invocations=None` on every node.

The v2 approach skips that problem:

> **At capture time, record every JS file the website executed during the captured flow. At replay time, evaluate all of those scripts in original order against a fresh document, then send the HTTP request with our `user_intent` and `chained` slots substituted, and let whatever JS that ran the first time fill in the `website_generated` slots again.**

We don't need to know which minified function does what. We run all of them; the intended functions hit their intended slots; the rest are no-ops or harmless.

> **Caveat: this thesis is to be re-examined when Phase 1 (session_manager), Phase 2 (CU lift + parity), and Phase 3 (real planner) are all in place.** At that point we have a real temporal representation in front of us (CU actions + captured traffic + executed JS bundle) and can examine real cases before locking the replay strategy. Phases 4–6 (tool building + replay + eval) implement the conclusions.

### 4. Computer use IS tool capture infrastructure — real-time, not on-success

The v1 mistake: graphs only built when a CU subtask completed successfully. Wrong abstraction. Most CU activity emits nodes worth crystallizing well before the subtask wraps up.

**The chaining algorithm in detail:**

> Observer records request/response pairs in real time. As soon as a pair lands:
>
> 1. Scan request's params (URL query string + body fields). For each `{param_name: value}`:
> 2. Look in **prior responses** for a matching value (same task / same session).
> 3. If found → this is a `chained` param with edge `(prior_response_field → this_request_param)`.
> 4. If the value matches a prior CU action's `typed_value` → this is a `user_intent` param.
> 5. Otherwise → `website_generated`.
>
> **Normalization is critical.** Trivial values like `"1"`, `"true"`, short strings, common dates match coincidentally across many requests and produce wrong edges. The matcher must:
> - Skip values shorter than 2 characters.
> - Skip values in a `_TRIVIAL_VALUES` denylist (`"true"`, `"false"`, `"0"`–`"9"`, common HTTP verbs, etc.).
> - Match exact strings only (no fuzzy match — fuzzy creates false chains).
> - Allow multi-edge: if the same value appears in 3 prior response fields, record all 3 candidates and let downstream classification disambiguate.
>
> Subtask success/failure does NOT gate this process. Candidates from a failed/pruned subtask get dropped at task end; from a successful subtask, get verified and registered.

The temporal representation has four facets per CU run:

1. **Action timeline** — which AID was clicked/typed at which timestamp, with what value.
2. **HTTP traffic** — every XHR/fetch with full headers + bodies.
3. **JS bundle** — every script the page loaded.
4. **DOM snapshots before/after** — useful mostly for CSRF tokens; heavy on disk. **Risk note**: we use minified scripts to do extraction from DOM (e.g. button-click handlers), not raw DOM diffing. If this breaks somewhere — e.g. a CSRF embedded as a hidden input that only the HTML reveals — we'll need to revisit. Run an experiment when Phase 4 ships.

**Backpressure isn't a worry yet.** Simple version: each (request, response) pair fires a synchronous classifier call on the same task. If CU's batch loop overruns, we deal with it then.

### 5. The planner needs a clean tool surface — exact shape is an open question

Tools live in `morphnet_v2/sites/{site}/tools.json`. The planner sees only:
- name
- one-sentence capability statement
- lifecycle status
- list of input slots (the `user_intent` params)

> **Caveat: the exact shape of how tools are presented to the planner is a research item.** Lifecycle states in v1 are `verified` (default after discovery-time verification), `failing` (3 consecutive use-time failures), and `script_drift` (captured SHA256s don't match live scripts). We DON'T have `unverified` / `probationary` middle states — discovery-time verification is the only gate. How we surface this to the planner without overwhelming it (token budget, ranking, freshness) is still unknown. Initial implementation dumps everything; we iterate after measuring.

## Architecture rules (load-bearing)

1. **Boundary rule.** session_manager owns ALL non-Python-process I/O. File I/O is exempt — `notes.py` writes directly. No layer above session_manager touches Playwright or CDP.
2. **Page handling lives in session_manager.** Page-load stability detection, retry-on-failed-interaction, popup dismissal, navigation detection, action dispatch — session_manager's responsibility, lifted from crawler. CU asks `session.click(aid)` and trusts retries already happened.
3. **Notes is dumb.** Records what's handed to it. No interpretation, no truncation, no parsing. Lazy subdir creation.
4. **Never truncate raw outside-world data.** HTTP bodies, cookies, init scripts, JS sources — verbatim, byte-for-byte.
5. **Lazy persistence.** Files/subdirs created the moment something is written into them.
6. **Generalist code.** No site-specific branches in core.
7. **One reusable tool, not five.** v1 had 5 JS entry-point discovery strategies. v2 has 1: replay all observed JS.
8. **Selective lift, not blind copy-paste.** When porting from crawler, every chunk reads the source first, identifies which lines belong in `session_manager.py` (touch Chrome/CDP/Playwright) vs which stay in `computer_use/` (pure logic on top of session_manager's primitives). We don't `cp -r crawler/ morphnet_v2/computer_use/`.
9. **Don't break crawler during the lift.** Crawler stays intact in `browser-challenge/`. We can run it side-by-side as a baseline to validate parity.
10. **Every line is documented in `README.md`.** No black boxes.
11. **Reflection is fine-tuning, not core.** Past reflectors caused FP/FN problems. **Success/failure of every step uses deterministic mechanical signals** — `SubtaskResult.success` (synthesized from per-action `ActionResult.success` returned by crawler's executor, already lifted in Phase 1 chunk 1.5) for CU steps, `ReplayResult.http_status == 200` for tool steps. No LLM-based verdict layer. Spiral detection inside CU stays regex-based per carve-out 15; refactor later. Reflection (and the AXTree-diff stage etc.) gets added back only when we have data showing it's needed.

## Module layout

```
morphnet_v2/
  README.md              ← living module guide (every chunk updates this)
  draft.md               ← this file
  __init__.py
  session_manager.py     ← I/O boundary; Chrome lifecycle; page lifecycle; action dispatch; HTTP+JS capture; Gemini; curl_cffi
  notes.py               ← lazy structured logger (✅ shipped)
  page_filter.py         ← lifted extraction engine (~3000 LOC, kept separate due to size)
  planner.py             ← branch/prune tree; planner schema + LLM call; orchestrator routing loop (Phase 3)
  computer_use/          ← lifted from crawler; everything that's not Chrome-touching
    __init__.py
    v5_markdown.py       ← lifted from master_markdown.py
    mutation_observer.py
    mutation_episodes.py
    raw_session.py       ← refactor of crawler/master.py:_run_raw_session w/ DI + V5 fix
    page_agent.py        ← integration: extract → render → run loop → execute
    templates/
      page_agent.j2      ← lifted verbatim
  planner.py             ← PlanningTree + ToolRegistry + planner LLM + Orchestrator (Phase 3, ✅ shipped)
  mutation_types.py      ← MutationNodeRef / RawMutationRecord (cycle-break extraction, ✅ shipped)
  tool_builder.py        ← real-time graph construction (Phase 4, outlined)
  tool_executor.py       ← HTTP+JS replay + A/B testing + routing (Phase 5, outlined)
  sites/{site}/          ← per-site state
    profile.json
    captures/{ts}.json
    tools.json
    bundle/{hash}/
```

---

# Build chunks

Order is dependency- and parity-driven. Each chunk has the same shape:

- **Why we need this**
- **What "done" looks like**
- **The work** — concrete steps, justified
- **Why this approach** — non-obvious decisions and what we rejected
- **README delta**

Status: ✅ done · 🟡 partial · ⬜ not started.

---

## Phase 1 — `session_manager.py` (the I/O boundary, complete)

Phase ends when session_manager can navigate any site, dispatch any action with retries, capture all HTTP traffic, and capture every JS source. All later phases consume these primitives. **Everything in this phase is either already done in v2 or is a port from crawler — we're not inventing new infra.**

### Chunk 1.1 — CLI skeleton ✅ done

**Why we need this.** Entry point.

**Done state.** `uv run python -m morphnet_v2.session_manager --url https://example.com` parses args, attaches notes, runs `run_task`.

---

### Chunk 1.2 — `notes.py` ✅ done

**Why we need this.** Every chunk below produces artifacts. Lazy persistence + parallel safety + type dispatch were prerequisites; without these, every later chunk would reinvent logging.

**Done state.** 401 LOC shipped.

---

### Chunk 1.3 — Chrome launch + lifecycle ✅ done in v2

**Why we need this.** Chrome must start, accept CDP, and not contaminate the next session's state.

**Done state.** v2 has: `_launch_chrome` with the LEARNINGS phase-6c stealth flags, fresh `tempfile.mkdtemp` profile per launch (so cookies / storage / bot-management tokens never bleed across sessions), explicit `--user-data-dir` + `--remote-debugging-port` per launch, raw `CDPSession` primitive over the per-page WebSocket. Sanity-checked: cross-session contamination is impossible because profile, process, and CDP port are all distinct per launch.


> **Note**: prefer crawler's chrome management as inspiration for any future tweaks. If we discover a stability gap during the parity experiment in Phase 2, port the missing crawler logic rather than morphnet's.

---

### Chunk 1.4 — Page lifecycle (port from crawler)

**Why we need this.** Without this, every action call from CU has to hand-roll its own retry, page-load wait, popup dismissal. **Crawler's lifecycle is significantly faster and more reliable than morphnet's because morphnet leans heavily on Playwright primitives** (slow round-trips, racy on fast-navigating pages); crawler does most lifecycle work via cheap `page.evaluate()` polls and CDP events.

**Done state.** `await session.wait_for_page_ready()`, `await session.dismiss_popups()`, `session.check_navigation()` all work against any site. Action methods (Chunk 1.5) inherit the same retry posture.

**The work:**
- [ ] Read `crawler/browser_tools.py:_wait_for_dom_stability` (line 191). Understand the polling logic. Port the heuristic to `session.wait_for_page_ready()`.
- [ ] **Verify whether the `Execution context was destroyed` error is a real crawler race or an artefact of model speed.** Check by running the failing log against crawler with gpt-oss vs gemini. If the race fires regardless of model, it's structural — wrap `page.evaluate()` in retry-on-context-destroyed. If it only fires with gemini's faster batching, it's a behaviour-frequency issue, not a structural bug — port crawler's logic as-is and document the concern.
- [ ] Port the popup-dismissal pattern. Generic, not site-specific.
- [ ] Port navigation detection (`Page.frameNavigated` + URL diff).
- [ ] **Update `README.md`** with a "Page lifecycle" section.

**Why this approach.** Crawler's lifecycle is tuned against 50+ real sites and avoids Playwright's per-call overhead. We rejected reinventing because the wins are too small relative to weeks of tuning. The selective part is verifying the supposed bug before fixing it — don't introduce defensive code without evidence.

**README delta.** "Page lifecycle" — `wait_for_page_ready`, `dismiss_popups`, `check_navigation`, retry policy, what `notes` event each emits.

---

### Chunk 1.5 — Action dispatch (port from `crawler/executor.py`, keep crawler intact)

**Why we need this.** CU and tool replay both need to dispatch actions through the same primitives. Per the boundary rule, dispatch lives in session_manager. **Important constraint: don't modify crawler itself during this port.** We need crawler intact as the parity-experiment baseline in Phase 2.

**Done state.** Public methods on session_manager: `click(aid)`, `type(aid, text)`, `scroll(...)`, `drag(...)`, `draw(...)`, `key_press(...)`, `hover(...)`, `wait_for_page_settle(...)`, `sleep(...)`, `copy_paste(source_ids, target)`. Each returns an `ActionResult` and writes one `notes.log(data_type="action", ...)` per call. Crawler's `executor.py` is untouched.

**The work:**
- [ ] Read `crawler/executor.py` end-to-end. Understand what `_prep` does (visibility + scroll + hit-test), what `ActionResult` looks like, which logic is shared.
- [ ] Port to `session_manager.py` as private helpers + public method wrappers. **Copy, don't move** — we leave the source file untouched.
- [ ] Don't keep an `Executor` class in v2 — methods live on `SessionManager` directly.
- [ ] Mirror result via `notes.log(data_type="action", ...)`.
- [ ] **Update `README.md`** with an "Action dispatch" section.

**Why this approach.** Boundary rule: session_manager owns Chrome interaction. Action dispatch is Chrome interaction. Putting it elsewhere creates two seams between Python and Chrome — exactly what the boundary rule prevents. Morphnet currently uses Playwright for actions; Playwright is robust, but crawler has refined the dispatch with `_prep` and blocker-probe retries that are demonstrably better. We port the refinement; we keep crawler intact for baseline.

**README delta.** "Action dispatch" — list every public method, its retry policy, what's logged.

---

### Chunk 1.6 — HTTP traffic + JS source capture (CDP subscriptions) + bot-detection experiment

**Why we need this.** Phase 4's tool_builder reads from this. Without complete HTTP capture and complete JS source capture, real-time graph construction has nothing to work with. **Capture must not get us flagged as a bot — we need to verify this empirically.**

**Done state.** After navigating to confirmtkt and doing one search, `notes/{site}/http/index.jsonl` has the auto-suggest + train-search calls with full bodies untruncated; `notes/{site}/scripts/{sha}.js` has the source of every JS file the page loaded. **Bot-detection experiment passes**: monitoring cookies before/after capture shows no new bot-flag cookies (`_abck`, `incap_ses_*`, etc.) on baseline-friendly sites.

**The work:**
- [ ] Subscribe `Network.requestWillBeSent`, `responseReceived`, `loadingFinished`, `responseReceivedExtraInfo`, `requestServedFromCache`. Mirror to `notes.log(data_type="http_request"|"http_response"|"cookie_set", ...)` with full bodies untruncated.
- [ ] Public methods: `session.get_traffic(since_ts)`, `session.clear_traffic()`.
- [ ] Subscribe `Debugger.scriptParsed`. Lazy-fetch source via `Debugger.getScriptSource`. Deduplicate by SHA256.
- [ ] Mirror script source to `notes.log(data_type="script_source", script_id=..., sha=...)` and write bytes to `notes/{site}/scripts/{sha}.js`.
- [ ] **Filter ONLY at the consumer.** Capture everything; tool_builder decides what's noise.
- [ ] **Bot-detection experiment**: take a known-friendly site (Wikipedia) and a known-protected site (BookMyShow). Run two CU sessions: (a) with capture disabled, (b) with capture enabled. Compare cookies after a fixed action sequence. Look for new bot-related cookies (`_abck`, `incap_ses_*`, `__cf_bm`, `datadome`). Report findings to README. If capture flags us, defer enabling `Debugger.*` until after `Page.loadEventFired` (sidesteps the Akamai/PerimeterX `Debugger.enable` early-check signal).
- [ ] **Update `README.md`** with an "HTTP + JS capture" section + experiment findings.

**Why this approach.** Capture-everything-filter-at-consumer is reversible; capture-time filtering isn't. CDP `Debugger.getScriptSource` is the only way to grab JS source bytes; Playwright doesn't expose this. The experiment ensures we know whether enabling the CDP `Debugger` domain trips bot detection in our specific environment before we commit to it.

**README delta.** "HTTP + JS capture" — CDP events subscribed, where artifacts land, bot-detection experiment results, fallback strategy if flagged.

---

### Chunk 1.7 — `call_gemini_async` + `make_http_session`

**Why we need this.** Planner, classifier, naming all call Gemini. tool_executor calls `curl_cffi`. Centralized utility = consistent retry, logging, impersonation.

**Done state.** `result = await session.call_gemini(model="gemini-3-flash-preview", contents=[prompt], response_schema=SCHEMA)` returns parsed JSON or raises after 3 backoff attempts. `sess = session.make_http_session()` returns a Chrome-impersonating `curl_cffi.Session` with cookies refreshed from the live browser.

**The work:**
- [ ] Lift `call_gemini_async` from `morphnet/session_manager.py`. Adapt to write through `notes.log(data_type="prompt", ...)`.
- [ ] `make_http_session()` returns `curl_cffi.requests.Session(impersonate=f"chrome{ver}")` matching launched Chrome's version. Cookies come from `session._context.cookies()`.
- [ ] **Update `README.md`** with a "Utilities" section.

**Why this approach.** Start with Gemini for speed of iteration. **Future direction (not this chunk): make `call_gemini` LLM-agnostic** — wrap a thin protocol that any structured-output-capable LLM can implement (Gemini, Anthropic, OpenAI). Defer until we have a reason to switch.

**README delta.** "Utilities" — `call_gemini`, `make_http_session`. Note about LLM-agnostic future direction.

**Phase 1 milestone — ✅ REACHED.** session_manager handles every piece of Chrome / outside-world I/O across 34 public methods. README documents each. The Phase 5 replay thesis was validated end-to-end on cleartrip (tier-3 Akamai-protected): 17/17 captured endpoints replayed via curl_cffi with matching status codes; parameter substitution proven (BLR→PNQ original capture, DEL→BOM modified replay returned 246 actual flights with the server echoing the new search intent). Per-site context (`morphnet_v2/sites/{site}/scripts/`) populates across runs and dedups by SHA256. Bot-detection experiment was deferred per LEARNINGS phases 4/5/10 already validating the CDP capture surface on Chrome 148 — we'll add it back if a future site flags us.

---

## Phase 2 — Lift CU + pass-through planner + parity experiment

This phase ends with a confidence check: **morphnet_v2 (session_manager + computer_use + pass-through planner) produces near-identical results to running crawler directly on the same task list.** If parity holds, the lift is correct and we can build the real planner on top. If not, we diagnose what we broke.

The pass-through planner is intentionally trivial — it has no branch/prune, no schema, no LLM call of its own. It just takes a task and hands it straight to `page_agent.run_subtask`. This is the smallest possible planner that lets us run morphnet_v2 end-to-end, and it isolates the variable: any behavioural difference vs crawler MUST come from the lift, not from the planner.

### Chunk 2.1 — `page_filter.py` (extraction engine, lift)

**Why we need this.** This is THE core capability — the bridge from "Chrome is open" to "we have a structured representation." The planner's V5 view, CU's input, tool_builder's element references all derive from `PageFilterOutput`.

**Done state.** `await PageFilter().run(page, snapshot) → PageFilterOutput` returns a non-empty extraction with containers, buttons, AIDs against any site. Output shape exactly matches crawler's.

**The work (✅ done):**
- [x] Lift `crawler/page_filter.py` + `crawler/schemas.py` + `crawler/config.py:ExtractorTuning` into a single `morphnet_v2/page_filter.py`. JS payloads byte-identical. Every schema field lifted verbatim (no pruning — `feedback_parity_test_before_extend`: prove faithfulness before pruning).
- [x] Boundary deltas: `PageFilter(sm, ...)` instead of `page: Page` per call; `_collect_axtree` + `build_aid_to_ax_map` route AXTree fetch through `sm.cdp.send(...)` (same page-target WebSocket as crawler's `page.context.new_cdp_session(page)` — protocol-byte-equivalent).
- [x] AST diff vs crawler trio: zero unexpected drift (only the 7 documented boundary-port methods differ).
- [x] `lorem-text` + `justext` added to pyproject.toml dependencies.
- [x] `README.md` updated with "Page filter" section: every public class, every method, every pipeline stage.

**Why this approach.** PageFilter is genuinely self-contained — keep it that way.

---

### Chunk 2.2 — `computer_use/v5_markdown.py` + mutation observer + episodes (lift)

**Why we need this.** V5 is what the LLM reads. Mutation episodes are the "## During Batch" feedback. Together they're the page representation surface.

**Done state.** `render_v5(extraction) → str` produces the markdown the LLM reads. `await session.flush_mutations() → list[RawMutationRecord]` plus `format_batch_events(episodes)` produces the "## During Batch" snippet.

**The work:**
- [ ] Copy `crawler/master_markdown.py` → `morphnet_v2/computer_use/v5_markdown.py`. Adjust imports.
- [ ] Copy `crawler/mutation_observer.py` and `mutation_episode_builder.py` → `computer_use/mutation_observer.py`, `mutation_episodes.py`.
- [ ] Wire `flush_mutations` into session_manager's public surface. (The observer touches the page so its installation belongs in session_manager; the episode-building logic that consumes records stays in `computer_use/mutation_episodes.py`.)
- [ ] **Heavy/light settle split (carry-over from chunk 1.4 sweep).** Crawler has TWO settle mechanisms: `_wait_for_dom_stability` in `browser_tools.py` (~5s, polls HTML/text length, no observer needed — already lifted as `wait_for_page_ready` in chunk 1.4) AND `wait_for_settle` in `mutation_observer.py:447` (~500ms, requires the MutationObserver to be installed first, returns mutation count). The light version is what `executor.wait_for_page_settle` calls. When 2.2 lifts the observer, also expose `wait_for_settle` so chunk 1.5's `wait_for_page_settle` agent action can call it. Without 2.2, that action either no-ops or falls back to `asyncio.sleep`.
- [ ] **Update `README.md`** with a "V5 markdown + mutation episodes" section.

**Why this approach.** Crawler's V5 is heavily tuned (compact buttons, exception-only annotations, two-section structure). Mutation observer is load-bearing for batched action emission.

**README delta.** "V5 markdown + mutation episodes" — markdown sections, when each appears, what mutations look like.

---

### Chunk 2.3 — `computer_use/raw_session.py` + `prompts/page_agent.j2` (LLM tool loop)

**Why we need this.** The CU loop. 100+ tuned details (text-nudge handling, JSON parse error recovery, navigation detection, batch-failure handling, spiral detection, V5-deletion bug fix). Lifting saves weeks.

**Done state.** A CU step runs end-to-end via this loop: the model emits a batched action call, the actions execute via session_manager (behind `batch_executor`), and the loop terminates on navigation / report / max_turns / spiral / no_tool_call. Returns `StepResult` with deterministic `success` consumed by the planner.

**The work (✅ done):**
- [x] Copy `crawler/templates/page_agent.j2` → `morphnet_v2/prompts/page_agent.j2` verbatim (byte-identical). Per `feedback_prompts_in_separate_dir` — prompts go in `prompts/`, not in a `templates/` subdir inside a code package.
- [x] `render_page_agent_prompt(user_goal, dnd_library) -> str` — Jinja2 loader + render helper. Default `.j2` for all morphnet_v2 prompts going forward.
- [x] `build_tools_schema(dnd_library) -> list[Tool]` — Gemini function declarations for `execute_actions` + `report`. `probe_drop_zones` action gated on `dnd_library == 'html5-native'`.
- [x] `class RawSessionRunner(*, sm, model, max_turns, batch_executor, dnd_library=None, ...)` — pure DI. `batch_executor` is the chunk-2.4 callback.
- [x] `async run(user_goal, initial_v5) -> StepResult` — body lifted with the V5-deletion fix preserved (initial V5 at index 0 is never deleted).
- [x] Spiral detection — regex-based, lifted verbatim from `crawler/master.py:2577-2613` (carve-out #15). Only string-matching path in the file.
- [x] Deterministic success rule (`_synthesize_success`) per architecture rule 11: `NAVIGATED OR (REPORT AND last_batch_clean)`. Catches LLM-hallucinated report after failed clicks.
- [x] Thinking traces → `notes.log(data_type="thinking", ...)` for diagnostic observability (function-calling mode has no `reasoning` field).
- [x] Extended `sm.call_gemini` to accept `tools` kwarg. ENFORCED: exactly one of `response_schema` or `tools` is mandatory — no freeform generations (per `feedback_no_freeform_llm`).
- [x] Added `BatchResult`, `SessionExit`, `StepResult` schemas to `computer_use/schemas.py`.
- [x] Smoke-tested: prompt renders (dnd-conditional verified), tools schema builds (12 vs 13 enum entries), spiral detection works (3-fail trigger, success-resets), success synthesis matches truth table. Pyright clean.
- [x] README updated with "Raw session loop" section.

**Why this approach.** Pure DI decouples the loop from any Chrome / page-extraction concerns. The loop is testable with a mock `batch_executor`. Gemini-native function calling (not LiteLLM) keeps Phase 1's `call_gemini` boundary intact and avoids importing LiteLLM.

---

### Chunk 2.4 — `computer_use/page_agent.py` (integration class)

**Why we need this.** The CU policy layer. Owns all per-page state crawler kept on `MasterOrchestrator`: action dispatch, mutation handling, blocking-graph maintenance, AX modal probing, re-extraction, batch execution. The planner-facing entrypoint is `run_step(user_goal) -> StepResult`.

**Done state.** A full CU step runs end-to-end via `PageAgent.run_step`: initial V5 → loop turns → batched actions through full dispatch → mutation flush + post-batch re-extract → spiral detection + V5 rotation → exit on navigation/report/max_turns. Returns `StepResult.success` deterministically.

**The work (✅ done — full crawler lift, no skips):**
- [x] `class PageAgent(*, sm, page_filter, model, max_turns=60, ...)` — DI on sm + page_filter + LLM config.
- [x] `run_step(user_goal)` — public entrypoint. Reset state → `wait_for_page_ready` → snapshot + extract + install_observer + build_ax_backend_map + ax_enable_push → render initial V5 → instantiate `RawSessionRunner` with `batch_executor` closure → drive → ax_disable_push on exit → return `StepResult`.
- [x] `_execute_batch` — lifted from crawler's `_handle_execute_actions`. Drag-batch pre-scan, per-action `mark_mutation_step` + nav baseline + dispatch + flush_mutations + nav check, blocker-aware click retry (one-shot, click-only), batch-stop on failure, post-batch persistence + re-extract + episodes.
- [x] `_run_one_action` — the full 380-LOC dispatch switch lifted verbatim. All 13 action kinds (click, type_text, scroll, scroll_page, copy_paste, key_press, drag×3, draw, hover, wait_for_page_settle, sleep, re_extract, read_text, probe_drop_zones). Intent handling (`dismiss` → `check_dismiss` + synthetic node_removed mutation; `navigate` → optimistic `wait_for_navigation(2000)`). Drag-mode validation. Error messages mirror crawler exactly so spiral regex stays compatible.
- [x] State management: `_known_aids`, `_rendered_aids`, `_aid_type`, `_container_id_set`, `_container_labels`, `_target_labels`, `_cmap`, `_blocker_allowlist`, `_dnd_library`, `_synthetic_drag_ok`, `_navigated`, `_action_log`, `_step_results`, `_batch_step_results`, `_batch_step_raw`, `_failed_action_count`, `_drag_batch_ranges`, `_ax_backend_to_aid`, `_ax_update_buffer`.
- [x] `_build_label_maps` + `_register_entity` — entity registry rebuilt after every re-extract.
- [x] `enrich_aid_refs` + `_resolve_to_visible` — replaces `aid-N` with `aid-N (label)` in action messages; walks parent chain to nearest non-scaffold visible container.
- [x] `process_mutations` — live blocking-graph maintenance. Updates `_current_extraction` in-place as overlays appear/disappear during a batch. Used by `_enrich_unknown_blocker` and synthetic dismiss-mutation injection.
- [x] AX modal probing: `filter_ax_probe_candidate_aids`, `_ax_probe_new_nodes` (routes CDP through `sm.cdp` instead of crawler's per-call `page.context.new_cdp_session`), `apply_ax_modal_signals`.
- [x] AX push events: `_ax_enable_push`, `_ax_disable_push`, `drain_ax_updates`, `process_ax_updates`, `_build_ax_backend_map`. Vestigial per crawler's note ("Playwright's new_cdp_session doesn't deliver Accessibility.nodesUpdated"); v2's raw sm.cdp MIGHT deliver them — lifted as-is, treated as best-effort.
- [x] Blocker probe: `_format_blocker_probe`, `_enrich_unknown_blocker` (uses `dataclasses.replace` since v2's `ActionResult` is `@dataclass`, not Pydantic).
- [x] `record_action`, `resolve_step_refs` — action log + `$stepN` text-ref resolution.
- [x] `run_fast_re_extract` (no DOM-stability wait) + `run_re_extract` (with wait via `sm.wait_for_page_ready`).
- [x] Three new sm wrappers added to chunk 2.2 mutation section: `mark_mutation_step`, `check_persistence`, `clear_observer_refs`. Plus AX-push helpers: `enable_ax_push`, `disable_ax_push`.
- [x] Imports `PlaywrightError` via `rebrowser_playwright.async_api` (matches sm's playwright dependency).

**Faithfulness verification.** AST diff vs `crawler/master.py:MasterOrchestrator` shows 23 shared methods after lift. Stripping docstrings + type annotations: only 4 trivial cosmetic differences remain (`@staticmethod` decorator on `resolve_target`, import path adjustment on `_ax_tristate`, explicit `bool()` wrap for type narrowing, dropping inline empty-dict annotations). **Zero behavioral drift.** Pyright clean across all of morphnet_v2.

**Optimization candidates (revisit after 2.6 parity)** — these were briefly considered as skips per `feedback_no_simplification_lifts` but are LIFTED IN FULL for now:
- `process_mutations` live blocking-graph (~500 LOC) — works correctly but might be removable if v2's per-batch re-extract proves sufficient.
- AX push events — vestigial in crawler; v2 might actually deliver them.
- `_enrich_unknown_blocker` — marginally redundant since `aid_allowlist` surfaces blockers in next V5.
- `resolve_step_refs($stepN)` — model rarely uses these.

**Why this approach.** The user explicitly directed full faithful replication of crawler, no architectural simplifications until 2.6 parity testing confirms what's actually necessary. Past chunks (2.1, 2.2, 2.3) were already faithful by AST diff; this chunk continues the pattern with full lift of all per-page MasterOrchestrator state + methods.

**README delta.** New "page_agent.py" section listing every lifted method with crawler line refs, the three new sm wrappers, the parity verification results, and the optimization candidates queued for after 2.6.

---

### Chunk 2.5 — `passthrough_planner.py` (stub)

**Why we need this.** This is the planner-shaped slot in morphnet_v2's architecture, but with no decision logic — it just hands the task to CU. Purpose: make morphnet_v2's pipeline runnable end-to-end so we can compare it against crawler. Replaced in Phase 3 by the real `planner.py`.

**Done state.** `await PassthroughPlanner(session, page_agent).run_task(task) → TaskResult` runs the task as a single CU invocation and returns whatever CU returned. No tree, no schema, no LLM call.

**The work (✅ done):**
- [x] `morphnet_v2/passthrough_planner.py`. Class `PassthroughPlanner(*, sm, page_agent, max_steps=10)`.
- [x] `async run_task(task: str) → TaskResult` — **loops** `page_agent.run_step(task)` across pages until termination (NOT a single call as draft originally said — multi-page tasks need the loop, and crawler does the same). Termination: `COMPLETED` on report, `STUCK` on non-nav exit, `LOOP_DETECTED` on repeated fingerprint, `MAX_STEPS` on budget exhaustion.
- [x] Loop fingerprint detection via `_build_fingerprint(url, extraction)` — lifted verbatim from `crawler/master.py:537-545`.
- [x] Journey tracking — `PageSummary(url, summary, actions)` per step, aggregated into `TaskResult.journey`. Mirrors crawler.
- [x] Step summary helper — `_build_step_summary(action_log)` produces `"clicked aid-5 → typed 'X' → ..."` (lifted from crawler's `_build_journey_summary`).
- [x] Added `PageAgent.last_extraction` property so pass-through can compute fingerprints without re-extracting.
- [x] Schemas: `TaskExit` enum (`COMPLETED` / `MAX_STEPS` / `STUCK` / `LOOP_DETECTED`), `PageSummary`, `TaskResult` (Pydantic, `extra="forbid"`).
- [x] README updated with "Pass-through planner (Phase 2)" section.

**Why this approach.** Smallest planner that makes morphnet_v2 runnable end-to-end. Mirrors crawler's outer page loop so chunk 2.6 parity test compares apples-to-apples — same per-page CU semantics, same loop-detection, same journey building. The only thing missing vs crawler is the auto-click optimization (single unblocked button → click without LLM); skipped because it's a latency win, not a correctness one. Lift later if 2.6 shows it matters.

**README delta.** "Pass-through planner" — class signature, per-iteration loop body, schemas, fingerprint helper, journey building, the `last_extraction` PageAgent addition.

---

### Chunk 2.6 — Parity experiment (the actual confidence check)

**Why we need this.** Verifies the lift is correct. Without this, we'd build the real planner on top of an unverified CU stack and any later bug would be ambiguous (planner issue? lift issue?). With this, the question collapses: if morphnet_v2-with-pass-through ≈ crawler, the lift is correct.

**Done state.** A 5-task subset of `experiments/comparison_50_tasks.json` (pick known-deterministic ones: `wikipedia_einstein_birth`, `github_cpython_latest_release`, `gemini_landing_state`, plus 2 lego/confirmtkt tasks) is run twice — once via crawler directly, once via morphnet_v2 with pass-through planner. Both runs use the same Gemini model and the same task definition. Comparison report shows: extraction shapes near-identical, action sequences near-identical, final answers match in spirit. Differences are diagnosed and either fixed or documented.

**The work:**
- [ ] Build `experiments/parity_v2.py`: takes a task list, runs each task via crawler (`crawler.main`) and via morphnet_v2 (`SessionManager + PassthroughPlanner + PageAgent`), captures both runs to disk.
- [ ] Build `experiments/diff_parity.py`: for each task, compares the two runs across (a) page extraction shape (count of containers / buttons / forms — should match within ±5%), (b) action types emitted by the model (should match in spirit, exact-aid mismatches OK if the AIDs themselves match the same elements), (c) final answer text (LLM grader, like our existing `grade_results.py`).
- [ ] Run both flows on the 5-task subset. Investigate any non-trivial differences. Fix or document.
- [ ] **Update `README.md`** with a "Parity experiment" section + measured results.

**Why this approach.** Diff-driven validation. The closer morphnet_v2's behaviour is to crawler's, the more confident we are that the lift didn't break anything. We rejected direct unit testing of every lifted function because the surface is too big — diff-testing on real outputs is a much higher leverage check.

**README delta.** "Parity experiment" — methodology, tasks chosen, measured deltas, diagnosis of any divergence.

**Phase 2 milestone.** Parity holds (or known divergences are documented and accepted). morphnet_v2 ≈ crawler on the chosen task subset. We can move to Phase 3.

---

## Phase 3 — `planner.py` (the recovery brain, replaces pass-through)

Phase ends when the real planner with branch/prune replaces the pass-through. **Reflector is intentionally absent** — past reflectors caused FP/FN problems. Step success/failure uses deterministic mechanical signals only: `SubtaskResult.success` (CU; synthesized from structured `ActionResult.success` per action) and `ReplayResult.http_status == 200` (tool). See Chunk 3.2 for the schema and Chunk 3.3 for the trigger-driven loop.

### Chunk 3.1 — `PlanningTree` + dataclasses (lift verbatim)

**Why we need this.** Branch/prune is the load-bearing recovery mechanism. Without it, when a step fails the agent retries the same approach forever.

**Done state.** `tree = PlanningTree(); tree.create_root(...)` → branch → prune → complete_current → `get_context_for_planning()` produces readable text. `detect_repeated_approaches()` catches loops. `to_mermaid()` saves the visualization to `notes/`.

**The work:**
- [ ] Copy `BranchSummary`, `PlanNode`, `PlanningTree` from `morphnet/morphnet_orchestrator.py:238–497` to `planner.py`. Zero morphnet imports — pure stdlib.
- [ ] Wire `tree.save_visualization()` to write Mermaid via `notes`.
- [ ] **Update `README.md`** — new "Planning tree" section.

**Why this approach.** Zero dependencies, ~400 LOC well-tested. No reason to re-implement.

**README delta.** "Planning tree" — `branch`, `prune`, `complete_current`, the four planner scenarios, where Mermaid lands.

---

### Chunk 3.2 — Planner schema + LLM call

**Why we need this.** Without these, no routing decisions get emitted.

**Done state.** `await planner.call(task, current_browser_state, tree, available_tools) → dict` returns parsed JSON matching the schema. The schema captures a routing decision (NOT a step decomposition).

**Terminology.** `step` (page-or-tool unit) replaces `subtask` throughout Phase 3+. Every node in the tree is a step.

**What the planner is NOT:**
- Not a step decomposer. It doesn't plan two moves ahead. There is no "list of pending steps" anywhere.
- Not a CU-instructor. It doesn't say "click button X."
- Not an action verifier. There is no reflector module. Step success/failure comes from deterministic mechanical signals (described below).

**The work:**
- [ ] Adapt `ORCHESTRATOR_SCHEMA` from `morphnet/morphnet_orchestrator.py:47–129`. Final v2 fields:
  - `planning_action` ∈ `{continue_cu, invoke_tool, complete_task, give_up}`
  - `tool_id` — if `invoke_tool`
  - `tool_user_intent` — slot values for the tool call (planner generates these from task + tree state)
  - `tree_update` — `{outcome: success|failure, summary: <1-line>}` for the just-ended step (none on task-start trigger)
  - `final_answer` — if `complete_task`
  - `reasoning`, `confidence`, `evidence_sources` — schema-first observability
- [ ] **Drop** these fields from the lifted schema: `next_subtask` (steps emerge from triggers, not decomposition), `branch_intent` / `prune_reason` (fold into `tree_update`), `urgency` (use `max_steps` budget instead), `task_success` (use `complete_task` action's `final_answer` presence), and `goto_url` (the tool's JS computes the URL; orchestrator handles it deterministically, not the planner).
- [ ] **CU context = the tree itself**, rendered via the lifted `PlanningTree.get_context_for_planning()`. NO bespoke instruction string from the planner. CU sees: original task + tree (which carries every completed step's summary and every failed step's failure reason). CU figures out the next move on its own.
- [ ] **Deterministic success signals** the planner consumes for `tree_update.outcome`:
  - CU step: `SubtaskResult.success` (already synthesized by crawler's lifted `_run_raw_session`, which reads structured `ActionResult.success` per action from `crawler/executor.py`).
  - Tool step: `ReplayResult.http_status == 200`.
  - No LLM evaluation. No AXTree diff. No second-guessing.
- [ ] `async planner.call(...)` — assemble prompt with budgeted sections: original task, current browser state (URL + V5 if case-a; URL + pending tool response if case-b), tree text (via `get_context_for_planning`), available tools list (name + capability statement + lifecycle status). Call `session.call_gemini(model="gemini-3.1-pro-preview", response_schema=PLANNER_SCHEMA)`.
- [ ] **Update `README.md`** with a "Planner schema + LLM call" section.

**Why this approach.** Trigger-driven routing is dramatically simpler than the v1 "emit next_subtask" framing. Crawler runs without any subtask decomposition and hits 20/50; the planner's marginal value over crawler is the cross-page tree memory, not step planning. Schema reflects that: routing decision + tree update, nothing else.

**README delta.** "Planner schema + LLM call" — fields, prompt assembly, model, notes events. Explicit note that CU context = the tree (no instruction string).

---

### Chunk 3.3 — Orchestrator (trigger-driven routing loop, no reflector) ✅

**Status (May 12):** Shipped. `Orchestrator` lives in `planner.py`. `SessionManager.run_task(task)` builds it lazily; `passthrough_planner.py` has been deleted. There is no separate `TaskResult` — `PlanningTree` itself carries `task_exit`, `final_answer`, `total_input_tokens`, `total_output_tokens`, plus derived `success` / `step_count` / `final_url` / `journey()`. The Orchestrator's `__init__` takes only `sm`, `max_steps`, `max_turns_per_step` — it builds PageAgent + PageFilter (and, in Phase 5, ToolExecutor) internally. To keep all imports at top-of-file we also extracted `MutationNodeRef` / `RawMutationRecord` to `mutation_types.py` to break the `sm → planner → page_agent → schemas → sm` cycle.

**Why we need this.** The actual loop that ties planner + CU + tool_executor. Lives in `planner.py` because routing IS planning. **No reflector** — deterministic mechanical signals only (see chunk 3.2).

**Done state.** `tree = await sm.run_task(task) → PlanningTree` runs end-to-end on a real task. `tree.success`, `tree.final_answer`, `tree.total_input_tokens`, `tree.total_output_tokens`, `tree.journey()` populated. Planning tree shows correct step nodes with deterministic outcomes.

**Trigger-driven loop:**

```
1. Task start
   → planner.decide(task, browser_state, empty_tree, cached_tools_for_site)
   → dispatch (continue_cu / invoke_tool / complete_task / give_up)

2. CU returned (case a — page rendered, V5 available)
   → planner.decide(task, current_url+V5, tree, tools)
   → dispatch

3. Tool returned, success (case b — browser unchanged from before tool ran)
   → planner.decide(task, current_url+stale_V5+replay_response, tree, tools)
   → if planner picks continue_cu: orchestrator goto(replay.yielded_url) FIRST, then invoke CU
   → if planner picks invoke_tool or complete_task: no navigation

4. Tool returned, failure (case b — browser unchanged, page is "active state")
   → planner.decide(...)
   → if planner picks continue_cu: CU resumes from the active state (NO re-render to a "known URL"; there isn't one)
   → if planner picks another tool or give_up: dispatch accordingly

Termination — return TaskResult when ANY of:
   - planner emits complete_task (with final_answer)
   - planner emits give_up
   - max_steps = 10 exhausted (matches crawler's max_pages default; one planner trigger per page transition, so step budget = page budget)
   - unrecoverable infrastructure error (browser died, etc.)
```

**The work:**
- [x] In `planner.py`: `Orchestrator(*, sm, max_steps=10, max_turns_per_step=60)`. PageAgent + PageFilter built lazily inside `_ensure_page_agent()`; not passed by caller.
- [x] `async run_task(task) → PlanningTree` — trigger-driven loop. Each iteration: call `call_planner(...)`, apply `tree_update` BEFORE branching (with safe defaults if the LLM forgot a field), check termination, dispatch on `planning_action`.
- [x] Step success/failure: planner's `tree_update_outcome` is the source of truth (LLM reads mechanical signals: `StepResult.success` for CU, HTTP status for tool). Tree's `complete_current(summary)` on success, `prune(summary)` on failure. No reflector.
- [x] Each CU invocation has its own internal budget: `max_turns_per_step=60` (lifted from crawler), plumbed sm → orchestrator → page_agent.
- [x] **Update `README.md`** with an "Orchestrator routing loop" section.

**Phase 5 cleanup notes:**
- `Orchestrator` has a Phase-3 defensive branch in `invoke_tool` that synthesizes an HTTP 503 when `_tool_executor is None`. Remove this branch once Phase 5 lands a real `tool_executor`. The branch should be unreachable in practice (registry is empty, so no `invoke_<tool>` declarations are exposed), but kept defensive in case the planner LLM ever hallucinates a function name.
- The placeholder for `_tool_executor.replay()` assumes `.http_status`, `.body`, `.error` attributes — the actual contract gets defined in Phase 5.

**Worked example — cheapest 2AC across 5 days DEL→BLR on confirmtkt:**

Task: `find cheapest 2AC across 5 days DEL→BLR`. `start_url=confirmtkt.com`. `max_steps=10`. Fresh registry.

- **Trigger 1 (task start).** Tree = root only. No tools. → Planner: `continue_cu`.
- **CU runs.** Fills DEL→BLR + day-1 date, hits search → results page. Observer captures the search API; `tool_builder` runs discovery-time verification → passes → `search_trains` registered as `verified`. CU's loop ends. `SubtaskResult.success=True`, day-1 fare = ₹2000.
- **Trigger 2 (CU returned, case a).** Browser on day-1 results. Tree = [root → step-1 CU ✓ "day 1 ₹2000"]. Tools = [`search_trains`]. Step budget = 9. → Planner: `invoke_tool` for day 2.
- **Tool replay runs.** HTTP 200. Response has day-2 trains + `yielded_url=confirmtkt.com/trains/DEL-BLR/day2`.
- **Trigger 3 (tool returned, success, case b).** Browser still on day-1 results (stale, no navigation occurred because planner didn't pick `continue_cu`). Tree += [step-2 tool ✓ "day 2 ₹1800"]. Step budget = 8. → Planner: `invoke_tool` for day 3.
- (Triggers 4, 5, 6 — days 3, 4, 5 tool calls, all HTTP 200. Tree grows to 5 step nodes. Step budget = 5.)
- **Trigger 7 (tool returned).** All 5 days collected. → Planner: `complete_task`, `final_answer="cheapest 2AC is day-N at ₹X on train Y"`.

7 triggers consumed, 5 within `max_steps`. CU was invoked once; the four `yielded_url`s were unused (we never transitioned back to CU). Tools collapsed 4 page navigations into 4 parallel API calls.

**Worked failure scenario — CU fail, retry, loop-detect:**

Same task. CU's first attempt fails because cookie modal blocks the search form.

- **Trigger 1 (task start).** → Planner: `continue_cu`.
- **CU runs.** Cookie modal blocks search form. Per-action `ActionResult.success=False` with `fail_subtype=blocked`. CU's spiral detector fires after 3 same-(action, aid) failures. Loop ends. `SubtaskResult.success=False`.
- **Trigger 2 (CU returned, case a, failed).** Browser on home with modal. Tree += [step-1 CU ✗ "search form blocked by cookie modal"]. Step budget = 9. → Planner: `continue_cu`. (Tree now carries the failure; CU at next invocation will see it and try something different.)
- **CU runs again.** Sees tree, knows search-without-dismiss didn't work. Dismisses modal first, then searches. `SubtaskResult.success=True`.
- **Trigger 3 (CU returned, success).** Tree = [step-1 ✗, step-2 ✓ "dismissed modal, day 1 fare ₹2000"]. Continues as the success case above.
- **Loop-detect variant.** If the second CU attempt also fails the same way, `PlanningTree.detect_repeated_approaches()` fires (two prunes with overlapping intent words). → Planner: either `continue_cu` once more with the tree explicit about "STOP retrying search," or `give_up`. CU reads the tree and either picks something fundamentally different (refresh, navigate elsewhere) or the planner gives up.

This is the across-page memory crawler lacks. Crawler would just keep clicking.

**Why this approach.** Trigger-driven routing with deterministic signals is dramatically simpler than reflector-based step-outcome synthesis. We're explicitly mirroring crawler's no-reflector trust model (which hit 20/50) and adding only the cross-page memory layer (which is the morphnet planner's marginal contribution). Reflection becomes a Phase 6 fine-tuning concern if we measure that deterministic signals are insufficient.

**README delta.** "Orchestrator routing loop" — the four triggers, deterministic signals consumed at each, the `goto(yielded_url)` behavior on `continue_cu` after tool success, the 4 termination conditions, what `TaskResult` contains.

**Phase 3 milestone.** `Orchestrator.run_task("...")` runs to completion against a real Wikipedia / GitHub / Gemini / confirmtkt task. Planning tree updates correctly with deterministic step outcomes. Same 50-task corpus from Phase 2 parity experiment re-run with the real planner; expect ≥ crawler's 20/50 and likely better on multi-step tasks where branch/prune helps (e.g., bot-detection-friendly sites that crawler loop-detects out of).

### Chunk 3.4 — Temporal representation (offline, ready for Phase 4 grounding)

**Status.** Shipped before the 100-task run so each task's notes dir gains a per-step alignment of (CU actions ↔ HTTP request/response ↔ JS scripts parsed/initiated/precisely-executed). Tool synthesis in Phase 4 consumes `StepFrame[]` directly.

**Done state.** `uv run python -m morphnet_v2.timeline <notes_dir>` writes `<notes_dir>/step_frames.json`. For each CU step the file carries: the action records that fired in window, every HTTP r/r pair with initiator-stack-resolved scriptIds, and a `ScriptUse` per script with evidence ∈ {parsed, initiator_stack, coverage} and per-step function-level coverage deltas.

**Where the three streams come from (no new capture surface — all already in v2 notes):**
- CU actions: `actions/{aid}.json` + `record.jsonl` `type=action` rows (chunk 1.5).
- HTTP r/r: `http/index.jsonl` + `http/bodies/{rid}.req|resp` (chunk 1.6). Pairing by `request_id`. Each request carries `initiator_stack` frames with `scriptId`.
- JS scripts: `scripts/{sid}.js` source bodies (chunk 1.6). Execution evidence is the union of (a) scriptIds that appear in any initiator_stack of in-window requests and (b) precise coverage deltas from `Profiler.takePreciseCoverage` snapshots captured at step boundaries.

**Revert log — the entire chunk:**

1. `morphnet_v2/session_manager.py` (after line 1208) — 2 new CDP send lines:
   ```python
   await self._cdp.send("Profiler.enable")
   await self._cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
   ```
2. `morphnet_v2/session_manager.py` (just above `cookies_snapshot`) — 1 new method `async def take_coverage_snapshot(self) -> list[dict]` (~8 lines, calls `Profiler.takePreciseCoverage`).
3. `morphnet_v2/planner.py` `Orchestrator.run_task` — 3 call-site additions:
   - After `tree.complete_current` / `tree.prune`: `await self._log_step_boundary("end", closing_node_id)` (+ 1 line capturing `closing_node_id = tree._current_id` before the close).
   - After `tree.branch(kind="cu")`: `await self._log_step_boundary("start", tree._current_id)`.
   - After `tree.branch(kind="tool", ...)`: `await self._log_step_boundary("start", tree._current_id)`.
4. `morphnet_v2/planner.py` — 1 new private async method `_log_step_boundary(phase, step_node_id)` (~9 lines) just above the `# ---- browser_state builders` separator. Calls `sm.take_coverage_snapshot()` + `notes.log(data_type="step_boundary", ...)`. Uses the default `_store_misc` handler in notes.py (zero changes to notes.py).
5. `morphnet_v2/timeline.py` — NEW file. Revert: `rm morphnet_v2/timeline.py`.
6. `morphnet_v2/README.md` — new "Chunk 3.4 — Temporal representation" subsection.
7. `morphnet_v2/draft.md` — this section.

**Why this approach.** Offline reader is the safest path before a 100-task run: zero risk to the orchestrator's hot path, and the schema becomes the contract Phase 4's real-time `tool_builder` must match. Precise V8 coverage is captured once per step boundary — `Profiler.takePreciseCoverage` is cheap (single CDP roundtrip per call) and gives us function-level grounding for tool synthesis (which JS function actually issued each network request).

**Trade-off.** `Profiler.startPreciseCoverage(detailed=True, callCount=True)` adds ~5–15% CPU overhead per page. Acceptable for evals; toggle off in production by reverting the two lines in §1 above.

**Known issues observed in the 100-task run (NOT fixed — capture-first policy):**

1. **`Profiler.takePreciseCoverage` payloads are heavy.** Bookmyshow: uniform 14.5 MB/snapshot. cleartrip: median 5.9 MB, max 14.5 MB. Within-task growth up to 4.6× (cumulative counters never reset; we never call `Profiler.stopPreciseCoverage`). 280 scripts × 44k functions × 67k ranges per bookmyshow snapshot; 39% of functions actually executed (the rest are paid-for but uncalled). User's call (2026-05-14): we have the RAM, keep capturing — re-examine after tool_builder lands.
2. **CDP `_read_loop` is fatal on Chrome death.** When the CDP WebSocket drops (Chrome OOM / renderer crash / etc.), `session_manager.py:1036`'s background asyncio task throws `websockets.ConnectionClosedError` unwrapped, killing the orchestrator process. The 100-task run died at task ~68 (cleartrip OTP-dismiss loop) for this reason — the per-task `try/except` in `run_eval.py` couldn't catch a sibling Task's exception. Fix when we stabilise: wrap the `async for raw in self._ws:` in `try/except (ConnectionClosedError, ConnectionClosedOK)` that sets `self._cdp_alive = False` and returns cleanly. ~5 lines.
3. **Bookmyshow PageFilter crash — `escapeCss` returns input unchanged on bookmyshow's live page.** Custom regex `/([ #;?%&,.+*~\\':"!^$\[\]()=>|\/@])/g` works on fresh `page.set_content` but matches nothing inside bookmyshow's live SPA — even though `String.prototype.replace` and `RegExp.prototype[Symbol.replace]` both probe as native. Sidestep when we stabilise: replace `escapeCss` body with `CSS.escape(String(value))` (verified working in the same control probe). One-line fix in `page_filter.py:895-896`. The current `try/catch` fail-soft in `_collect_occlusion` (item 12) prevents this from crashing the task; the missed-target occlusion entries silently degrade extraction quality on bookmyshow only.

**Deferred fixes (noted, not yet landed):**

4. **New-tab navigation hijacks the session.** Amazon's sponsored product cards and similar `target="_blank"` / `window.open()` patterns spawn a fresh Chrome tab on click. Our `sm.page` stays bound to the originating tab, so the agent sees no URL change / no mutations / no V5 refresh and burns retries thinking the click failed. Observed empirically on amazon_in task 2 (bookshelf) on 2026-05-15. Fix design (CU-scoped, ~28 LOC, designed but not landed): pre-click DOM walk from clicked aid → set `target='_self'` on first `<a>`/`<form>` ancestor; init script overrides `window.open` gated by a `window.__cuClickActive` flag that `_click_by_aid` sets before dispatch and clears in `finally`. The flag-gate keeps ad-popup scripts unaffected — only deliberate CU clicks get the same-tab forcing. Apply when click-task discovery becomes priority.

5. **`type_text` actions do not record the typed value.** `_log_action("type_text", result, aid=aid)` is called without `text=text_value` in `extra`, so the typed string never reaches `actions/{aid}.json`. Without this, `user_intent_text` slot detection (Rule 3 in tool_builder) cannot fire — the empirical FN/FP test returned 0/0/0 for user-intent because the extractor had nothing to extract. Three-line fix: pass `text=value` through to `_log_action` at the type-action call site. Lights up user_intent_text classification on the existing 45-task corpus.

6. **Click actions do not record DOM context of the clicked element.** Needed for `user_intent_dom` slot detection (Rule 4 in tool_builder) — e.g., extracting `data-asin` from a clicked Amazon product card. The action record currently stores only `aid` + dispatch coordinates. Fix design (~30 LOC): in `_click_by_aid`, before dispatch, capture `outerHTML` (truncated to ~500 chars), `closest('[data-*]').attributes`, and `href` of the clicked element + its top 3 ancestors. Store under `actions/{aid}.json.dom_context`. Without this, Rule 4 stays inactive and the entire click-to-id chain class can only be inferred via HTML-body chain detection (Rule 2 extended).

**Open follow-up (Phase 4 will land):**
- Real-time `tool_builder.py` consumes `StepFrame[]` and emits candidate `ToolEntry`s with `SlotDef`s derived from request bodies + response shapes.
- Optional: hash `script_source` once at the per-site context and reference by sha256 in `ScriptUse` instead of relative path. Cheap upgrade once tool_builder is up.

---

# Phases 4–6 — outlined, detailed later

These phases are critical to morphnet_v2's full vision but are deferred until Phase 3 ships. Outlined here so the build-order makes sense; chunked in detail when we get there.

### Chunk 4.0 — `enumerate_mode` flag on PageFilter (✅ shipped 2026-05-15)

**Revert log:**
1. `morphnet_v2/page_filter.py` `PageFilter.run()` signature — added `enumerate_mode: bool = False` kwarg; forwarded to `_collect_payload`.
2. `morphnet_v2/page_filter.py` `_collect_payload(enumerate_mode=False)` — added kwarg.
3. `morphnet_v2/page_filter.py` JS args block — `containerNodeLimit` and `controlNodeLimit` jump from `_tuning` defaults (220 / 500) to **3000** when enumerate_mode; `maxDataAttrsPerContainer` jumps 8 → **30**; new `enumerateMode` boolean arg passed through.
4. `morphnet_v2/page_filter.py` JS container candidate loop — `if (!visible(node)) continue;` becomes `if (!args.enumerateMode && !visible(node)) continue;`. CU's normal viewport-mode behaviour is unchanged.

**Verification result** (against captured Amazon search HTML, 1.68 MB body, set_content offline):
- Default mode: 6 containers with data-asin
- Enumerate mode: **69 containers with data-asin, 32 distinct ASINs** including `B0B1PXM75C` (the one the discovery agent clicked)

This is the foundation Chunk 4.3 (HTML response indexing) builds on.

### Chunk 4.1 — `tool_builder.py` skeleton + offline HTML extractor (✅ shipped 2026-05-15)

**Revert log:**
1. `morphnet_v2/tool_builder.py` — NEW file, ~190 LOC. Revert = `rm morphnet_v2/tool_builder.py`. Defines:
   - Type aliases: `SlotKind = Literal["chained","user_intent_text","captured"]`; `DispatchKind = Literal["rest","graphql","json_rpc","form_post","page_navigate"]`
   - Dataclasses: `ScriptRef(script_id, url, sha256)`; `SlotSource(kind, chain_source, response_jmespath, html_attribute, container_signature, list_selector_needed, captured_examples, observed_values, required)`; `SampleRequest`; `SampleResponse`; `ToolCandidate(cluster_key, dispatch_kind, sample_request, sample_response, slots, constants, call_count, tasks_seen, initiator_scripts, rule_trace, tool_id, capability_statement, slot_descriptions, verdict, verdict_reason)`
   - Public stubs: `build_candidates(notes_dir)` raises NotImplementedError; `build_site_registry(eval_run_dir, site)` raises NotImplementedError
   - Working helper: `async extract_response_html(html, base_url, sm) → PageFilterOutput` — sets sm.page content to captured HTML, runs `page_filter.run(enumerate_mode=True)`, returns the structured output

**No other files modified.** No production code path touches tool_builder yet — pure additive.

**Verification:** module imports clean; dataclasses instantiate correctly.

**Next:** Chunk 4.2 (junk gate + scriptId classifier + dispatch-identity clustering).

### Chunks 4.2–4.8 — full tool_builder pipeline (✅ shipped 2026-05-15)

All landed in a single extension of `morphnet_v2/tool_builder.py` (now ~700 LOC total). Plus two new prompt files. Revert: trim `tool_builder.py` back to the Chunk-4.1 skeleton (the file's top dataclasses block is intact, everything below `# Chunk 4.2` is new) OR `rm morphnet_v2/tool_builder.py morphnet_v2/prompts/tool_finaliser.j2 morphnet_v2/prompts/list_selector.j2`.

**Revert log per chunk:**
- **4.2** — `is_noise_host`, `is_noise_path`, `is_static`, `classify_script_url`, `is_junk_response`, `cluster_identity`, `parse_graphql_operation`. Pure functions, no I/O. Junk gate: status ≥ 400 OR body too small for declared MIME (html < 500B, json < 30B, other < 30B). No body-key string matching (dropped per user feedback). Cluster identity differentiates REST / GraphQL (operationName from body) / JSON-RPC (method from body) / form_post / page_navigate.
- **4.3** — value normalisation (`normalise_value` URL-decode/lowercase/date-parse), entropy filter (`looks_low_entropy`), nonce regex (`looks_like_nonce`), `tokens` + `token_overlap` with token-subset scoring (Jaccard / max with subset boost to 1.0/0.9). `IndexEntry` dataclass; `index_json_body` (recursive JSON walk); `index_html_body` (uses Chunk 4.0's enumerate_mode page_filter offline). `detect_chain` with comma-split for composite values. `link_causality` for initiator_stack ∩ app-code scriptIds.
- **4.4** — `detect_user_intent_text` (placeholder, fires 0× until instrumentation 0a lands), `make_captured_source` with deterministic required/optional inference (False iff any observed value is empty/null/undefined).
- **4.5** — `graph_isolation_filter`. Computes chain_in (slot count of kind=chained), chain_out (downstream consumers of this cluster's response in other clusters' slots), user_intent_count. Drops nodes where all three are 0.
- **4.6** — `finalise_cluster` (one Gemini Flash call with `response_schema` from `FINALISER_RESPONSE_SCHEMA`) + `morphnet_v2/prompts/tool_finaliser.j2` (Jinja). Multiplexes name + description + verdict + slot descriptions in one call. Wraps `sm.call_gemini`.
- **4.7** — `build_candidates(notes_dir, sm)` orchestrates all phases. CLI: `python -m morphnet_v2.tool_builder <notes_dir> [--site <s>]`. Writes `tool_candidates.json` next to `record.jsonl`.
- **4.8** — `build_site_registry(eval_run_dir, site, out_path?)`. Cross-task merge by `cluster_key`. `_merge_slot_sources` preserves kind precedence (chained > user_intent_text > captured), dedupes examples, ANDs required flags. Writes `morphnet_v2/sites/{site}/tools.json`. `_rehydrate_candidate` / `_rehydrate_slot_source` reload from JSON. `load_tools_for_site` (in tool_executor.py) does the read side.

### Chunks 5.0–5.4 — tool_executor + 3× test (✅ shipped 2026-05-15)

**Revert log:**
- **5.0** — Two surgical edits in `morphnet_v2/planner.py`:
  - `Orchestrator.__init__` — added `self._registry = ToolRegistry()` + `self._seed_registry_from_site()` method that loads `morphnet_v2/sites/{site}/tools.json` if present, registering each tool's `ToolEntry` so planner's `build_planner_function_declarations()` surfaces `invoke_<tool_id>` declarations automatically.
  - `Orchestrator.run_task` — `registry = ToolRegistry()` line replaced with `registry = self._registry` (use the seeded one).
  - `Orchestrator.run_task` — the Phase-3 `_tool_executor is None` defensive 503 branch replaced with: lazy-construct `ToolExecutor` on first invoke (when `self._registry._tools` is non-empty); pass `_user_task` and `_tool_id` into planner-values for the list_selector context.
- **5.1** — `morphnet_v2/tool_executor.py` (NEW, ~320 LOC). `ReplayResult` dataclass. `ToolExecutor(sm, tools_by_id)` class. `replay(tool_id, planner_slot_values) → ReplayResult`. Slot resolution per kind: chained → `_resolve_chained` (either live_page via page_filter enumerate_mode + list_selector, or upstream cluster_key via cached response). HTTP dispatch via `sm.make_http_session()` (existing curl_cffi client). `_build_request` overlays constants + resolved slots into URL query / body. `_eval_simple_path` for JMESPath-ish navigation. `load_tools_for_site(site)` reads tools.json into `{tool_id → ToolCandidate}`.
- **5.2** — `morphnet_v2/prompts/list_selector.j2` (NEW Jinja prompt). `_list_select` in `ToolExecutor` calls `sm.call_gemini` with `LIST_SELECTOR_RESPONSE_SCHEMA` returning 1-based chosen_index.
- **5.3** — Lifecycle update integrated into Chunk 5.0's orchestrator wiring — `registry.record_success` / `record_failure` called per replay outcome.
- **5.4** — `experiments/run_tool_test.py` (NEW, ~180 LOC). Three task variants (Mumbai→Delhi 5 Jun, Chennai→Bangalore 30 May, Pune→Howrah 15 Jun). State file `results/tool_test_state.json` keeps the shared session_dir across invocations. After Run 1: synthesises via `tool_builder.build_candidates`, aggregates via `build_site_registry`, copies result to `morphnet_v2/sites/confirmtkt/tools.json` so Run 2 + 3 pick it up. CLI: `python experiments/run_tool_test.py {1,2,3,report}`.

**No other files modified.** All Phase 4+5 changes are isolated to: new files (`tool_builder.py`, `tool_executor.py`, two prompts, `run_tool_test.py`) plus three small edits in `planner.py` (registry seeding, registry use in run_task, ToolExecutor lazy-construct + invoke).

## Phase 4 — `tool_builder.py` (real-time graph construction)

Builds tool candidates as CU runs, not at end-of-subtask. Implements the chaining algorithm from Core Idea 4 (request/response pair scanning, value normalization, edge construction). Includes naming via Gemini, registry, and lifecycle (described below).

**Tool lifecycle (v1, deterministic):**
- **verified** — default after a successful discovery-time verification replay. Tool is in `tools.json`, available to the planner.
- **failing** — 3 consecutive execution failures since the last success. Tool stays in the registry; planner sees the status and prefers another tool / CU.
- **script_drift** — captured-script SHA256s don't match the live page's current scripts (lower-priority detection per user; we implement but don't optimize).

Transitions: discovery → verification replay → pass = `verified`, registered in `tools.json`; fail = NOT registered (no entry written). Per-use outcome: failure increments the failing counter; one success resets to `verified` regardless of prior state. Pre-execution check compares current page's loaded-script SHA256 set vs the tool's captured SHA256 set; mismatch → `script_drift`.

**We don't auto-discard tools.** Discarding is a problem when we've built many; investigation is more valuable than deletion. **We don't separate read tools from write tools.** Same flow; we want to observe write-tool replay impact empirically before designing separation.

Critical sub-tasks (to be chunked in detail later):
- Per-step Observer with real-time event emission
- Param classifier (3 types) with normalization to avoid false chains
- Chain edge builder
- Naming + capability statements via Gemini
- Registry + lifecycle (states above)
- Discovery-time verification replay (the gate that decides whether to register)

## Phase 5 — `tool_executor.py` (replay + A/B testing)

HTTP + JS-bundle replay (the conclusion of Core Idea 3, reviewed at end of Phase 3 with real captured data). A/B testing loop for failed tools (Core Idea 2 scenario 4). Tool routing dispatch — re-adds `routing` enum to the planner schema.

Critical sub-tasks:
- HTTP + JS bundle replay via curl_cffi + fresh CDP target
- A/B testing (re-run captured CU actions, diff against stored capture)
- Tool routing in the orchestrator

## Phase 6 — Eval + fine-tuning

`experiments/run_eval_v2.py` + `grade_results_v2.py` running against the 50-task corpus. Compare to crawler's 20/50 baseline and morphnet's 11/50.

Fine-tuning items considered here:
- Reflector (if simple-signal verdicts prove insufficient)
- Embedding ranking for `find_candidates`
- LLM-agnostic Gemini wrapper
- Multi-tab handling
- Tool-graph DAG branching

---

# Caveats and review points

These are explicit "we'll come back to this":

1. **Core Idea 3 (3-param-types / run-all-JS)** — review at end of Phase 3 with real captured data.
2. **Core Idea 5 (planner-tool-surface representation)** — research item; initial impl dumps everything.
3. **Reflector deferred to Phase 6 fine-tuning.** Simple action-result signals until then.
4. **DOM snapshot value** — heavy on disk; useful mostly for CSRF token capture. Risk if we miss a flow that needs them.
5. **Backpressure on the classifier queue** — not a worry until CU's batched loop overruns.
6. **`Execution context was destroyed` error in crawler** — verified to be a real race in `_collect_stability_probe`, but its frequency may be inflated by gemini's batched-action speed. Port-and-fix is justified, but the fix should be minimal (retry on context-destroyed) until we have data showing more is needed.
7. **LLM-agnostic `call_gemini`** — Phase 6 fine-tuning.
8. **`max_steps` and `max_turns_per_page` budgets are mirrored from crawler.** `max_steps=10` (= crawler's `max_pages` CLI default), `max_turns_per_page=60` (= crawler's `master.py` default). One planner trigger fires per page transition, so step budget = page budget. Tune per task / per site later if budget-induced false stops appear.

# Deferred / future work (logged Phase 1 → revisit later)

8. **Bot-detection experiment formalization.** Chunk 1.6 spec called for a Wikipedia-vs-BookMyShow capture-on/off cookie diff. We deferred per the user's call: LEARNINGS phases 4/5/10 already validated `Network.enable` + `Debugger.enable` + `Target.setAutoAttach` on Chrome 148 across 4 protected sites. The cleartrip end-to-end replay (tier-3 Akamai) further confirms our capture isn't flagged in practice. **Re-add the formal experiment if a future eval site fails replay** with status drift between original and replay calls — that's the signal we got fingerprinted.

9. **Cleartrip parameter substitution data point.** Validated empirically: `from`, `source_header`, `to`, `destination_header`, `depart_date`, `class`, `adults`, `childs`, `infants` are all pure `user_intent` slots — server accepts arbitrary substitution without re-running JS. **This may NOT generalize.** Swiggy was bound to fresh tokens per LEARNINGS phase 8b vs 8c. Phase 4's tool_builder must run the same kind of substitution-validation experiment per-site before locking a graph.

10. **Run-all vs subset script-replay policy.** Phase 5 replay must decide: (a) re-run every captured script in original order against a fresh context (safe but slow), (b) re-run only initiator-stack scripts (fast but misses state-defining inline scripts like Swiggy's `<script>window._csrfToken = ...</script>`), (c) hybrid based on URL / inline detection. Cleartrip didn't need ANY JS replay — captured cookies + curl_cffi were enough. Other sites will. Decide with real data when Phase 4 has graphs to replay.

11. **SHA256-keyed script filenames.** Currently scripts on disk are filed as `scripts/{script_id}.js` because notes' existing `_store_script_source` keys by `script_id`. Functionally OK (we still dedup by SHA256 — only one file per unique source), but the per-run forensic dir uses script_id as filename while the per-site context uses sha256. Cosmetic inconsistency. Fix in notes.py if it ever bites us.

12. **Drift detection.** When a graph's referenced SHA256 doesn't match the current capture's SHA256, the design is to A/B test the graph (run cached, run fresh CU on same intent, compare outputs) rather than auto-rebuild. Implementation lands in Phase 5 tool_executor. The data substrate (per-site script index with `runs[]` history) is already populated.

13. **LLM stress test.** Chunk 1.7 includes parallel-execution claims for `call_gemini`. Verified by inspection (genai async client uses httpx pool, no shared lock). Add an empirical stress test if we observe surprising serialization in Phase 3+ planner runs.

14. **Async CDP handlers (chunk 2.2 sanity-check carve-out).** `CDPSession._read_loop` originally launched any coroutine returned by a handler via fire-and-forget `asyncio.create_task(res)` — untracked, uncancellable, exceptions silently swallowed. We removed that branch (handlers must be sync). All current handlers (`_on_request`, `_on_response`, `_on_response_extra`, `_on_loading_finished`, `_on_loading_failed`, `_on_request_from_cache`, `_on_script_parsed`) are sync — they mutate state then return, with explicit `asyncio.create_task(...)` calls anchored into `self._capture_tasks` for follow-up work like `_finalize_request` and `_link_script_source`. **When a future chunk needs an async CDP handler**, the right move is: have the handler do `task = asyncio.create_task(...); self._capture_tasks.add(task); task.add_done_callback(self._capture_tasks.discard)` inside its sync body — same pattern as `_on_loading_finished`. Do NOT reintroduce the iscoroutine branch in `_read_loop` — that's what we just removed because it was untraceable.

15. **Spiral detection IS our current reflection mechanism (chunk 2.3 carve-out).** v2 currently has no separate "reflector" module. The only signal v2 has for "this subtask is failing" is the regex-based spiral detector lifted from crawler's `_run_raw_session`: it parses `aid=aid-NNN` failure lines out of `BatchResult.text` and counts consecutive same-action-same-aid failures. After 3 hits on the same (action_type, aid) pair, it injects a "re-read the page, try a different approach" nudge user message and resets. **Why:** crawler chose to encode action results as text and the spiral detector reads them back. **The risk:** it's regex on a string the action layer formats — `feedback_no_string_matching` says don't do this. If we ever rename "type failed" to "fill failed", spiral detection silently breaks. **The fix when we revisit:** add a structured `BatchResult.failures: list[{action_kind, aid}]` side-channel and have spiral detection consume that instead of regex-on-text. **Why we're lifting verbatim for now:** parity with crawler in chunk 2.6 requires the same detection behavior, and we don't yet have a strong mental model for what richer reflection should look like (LLM-based subtask reflector? deterministic per-action verdict? AXTree-diff?). Revisit after Phase 3 planner is operating — the planner's branch-prune signal may obviate per-subtask reflection entirely (the planner is the reflector).

    **Cross-reference:** Phase 3 chunk 3.3's deterministic step-outcome signals rely on `SubtaskResult.success` (synthesized from structured per-action `ActionResult.success`) for CU steps and `ReplayResult.http_status == 200` for tool steps — NOT on the regex spiral signal. Spiral detection is one of several internal CU-loop signals that feed into `SubtaskResult.success`; if we ever refactor it to consume structured `BatchResult.failures`, the planner's contract doesn't change.

16. **Mutation noise filtering (deferred from May-12 swiggy investigation).** Mutations whose subject is `<img>`, `<svg>`, `<picture>`, `<source>`, `<style>`, or `<script>` should be filtered at the observer JS level (before they enter the records buffer). These convey no actionable info to the agent and currently flood `## During Batch` on image-heavy pages — we observed 78 `obs-N appeared tag=img` entries in a single batch on swiggy after typing an autocomplete query. Fix when revisiting the mutation observer pipeline.

17. **Replace `_ACTION_WORDS` verb list with structural scoring (deferred from May-12 hazelnut investigation).** Today the button utility score uses a hardcoded verb list (`_ACTION_WORDS` in `page_filter.py:608`) to bonus action-taking buttons. We expanded the list with commerce verbs (add, buy, order, cart, checkout, save, remove, delete, confirm, pay, send) because "Add" buttons on swiggy tied with "More Details" / "See more information" at the same utility score, lost the DOM-order tiebreaker, and got dropped past the 40-button cap. **The verb list is fragile** — any new commerce/booking/social verb the list misses ("Reserve", "Vote", "Like") will keep silently losing. **The real fix when we revisit:** replace the verb-list bonus with a structural signal like a short-label CTA bonus (length 1-8 chars gets +0.10) so the rule is content-agnostic. Also reconsider `max_buttons_total: 40` — on dense action surfaces (product catalogs, search results with many cards), 40 is too aggressive.

18. **Exponential backoff after batch failure (deferred from May-12 — flagged risky to edit).** When a batch fails (e.g. `not_found` on a stale AID), the post-batch flow currently re-extracts with the standard 800ms DOM-stability window. If the failure stems from "page hasn't settled yet from the previous action," the next batch fires on the same un-settled page and fails the same way. Proposed: track consecutive batch failures, add an exponential extra sleep (1.6s → 3.2s → cap 5s) before re-extract. Skipped on user instruction because the change touches active loop code and breaks should be avoided unless certain — revisit with care.

# Cross-cutting

- **README is not optional.** Every chunk includes a README delta. No code lands without README updates.
- Logging: stdlib `logging.getLogger(__name__)`. `info` lifecycle, `warning` best-effort, `exception` inside event-loop handlers.
- Errors: raise on hard failures; log+continue on best-effort.
- Async: every I/O method yields the loop. `asyncio.to_thread` for sync C-API if any sneak in.
- Verification: each chunk ships with an inline verification step. No chunk is "done" without it.
- Reviews: stop and review between every chunk.
- **Don't break crawler during the lift.** Crawler in `browser-challenge/` stays intact as the parity baseline.
