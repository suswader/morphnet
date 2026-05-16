# MorphNet — Things to Fix

A consolidated list of bugs and gaps surfaced during the 50-task crawler-vs-morphnet comparison experiment (2026-05-09 → 2026-05-10). Sorted by blast radius.

---

## 1. Chrome process management on macOS — the runtime-blocker

**Severity:** CRITICAL. Currently kills 31/50 morphnet runs with `BrowserType.connect_over_cdp: Browser context management is not supported`. Also poisons bot-detection state on remaining sites.

### Root cause

`session_manager.py:_launch_chrome` (line 2632) and the cleanup code at line 3151–3156 don't reliably tear down Chrome between tasks on macOS. Chrome's "single-instance-per-user-data-dir" policy then routes new task launches into a stale, half-killed Chrome from the previous task.

### Symptoms (all variants of the same bug)

| Observed | Why it happens |
|---|---|
| 31/50 morphnet tasks fail at `connect_over_cdp` | Stale Chrome's browser context is owned; new connection's `Browser.setDownloadBehavior` is rejected. |
| Multiple unrelated sites in one Chrome window | Chrome's IPC forwards new launch URLs to the surviving Chrome → opens new tabs in it. |
| Dormant new-tab Chrome windows piling up on screen | Same IPC opens a window instead of a tab; the controlling subprocess errored out at `connect_over_cdp` and never used it. |
| "Something went wrong when opening your profile" modal | `shutil.rmtree(profile_dir)` partially succeeds (deletes files Chrome doesn't have open) → corrupted profile → Chrome's modal blocks browser context. |
| Bing search for "Www Swiggy Com" | Stale Chrome had focus on a new-tab search box from a previous task; CU's `type` action landed there, not on the intended target. |

### Failure sequence (the macOS-specific chain)

1. Task 1 finishes → `chrome_proc.terminate()` → only the **launcher** subprocess is killed.
2. Chrome's actual browser process keeps running (separate PIDs on macOS), still holding the lock on `chrome-morphnet-9301/`, still listening on port 9301.
3. Task 2 starts → `_launch_chrome` calls `shutil.rmtree(profile_dir, ignore_errors=True)` → **wipe partially succeeds** because the surviving Chrome has open file handles on `Local State`, `SingletonLock`, etc.
4. New Chrome subprocess starts with `--user-data-dir=...chrome-morphnet-9301`.
5. Either:
   a. New Chrome detects the surviving Chrome → forwards args via IPC → exits → existing Chrome opens a new tab. Subprocess connects via CDP, but `Browser.setDownloadBehavior` is rejected because the existing Chrome's context is owned.
   b. New Chrome detects partial corruption from the half-wiped profile → shows the "profile error" modal → modal owns context → same `setDownloadBehavior` rejection.
6. Subprocess fails. `chrome_proc.terminate()` runs on a launcher PID that's already dead. Chrome stays alive. **Repeats for every subsequent task.**

### Fix

**Two changes in `session_manager.py`:**

1. **`_launch_chrome` (line 2698)** — launch Chrome in its own process group:
   ```python
   return subprocess.Popen(
       cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
       start_new_session=True,   # NEW — process group ownership
   )
   ```

2. **`finally` block at line 3151** — process-group kill, then poll for `SingletonLock` clear, then re-attempt `rmtree`:
   ```python
   finally:
       import os, signal
       try:
           os.killpg(os.getpgid(chrome_proc.pid), signal.SIGTERM)
           chrome_proc.wait(timeout=5)
       except (ProcessLookupError, subprocess.TimeoutExpired):
           try:
               os.killpg(os.getpgid(chrome_proc.pid), signal.SIGKILL)
               chrome_proc.wait(timeout=5)
           except (ProcessLookupError, subprocess.TimeoutExpired):
               pass
       # Wait until SingletonLock clears before letting next task launch
       profile_dir = Path(__file__).parent.parent / ".tmp" / "chrome-profiles" / f"chrome-morphnet-{args.port}"
       lock_file = profile_dir / "SingletonLock"
       for _ in range(30):
           if not lock_file.exists():
               break
           time.sleep(0.1)
   ```

**Optional defense-in-depth** in `experiments/run_comparison.py` worker — give each task a unique port (`port_base + worker_idx + task_count_so_far`) so even a leaked Chrome from a previous task can't collide with the next one. Profile dirs are port-keyed, so unique ports = unique dirs = guaranteed isolation.

### Already applied (incomplete, doesn't fix the lifecycle issue)

- Added `SearchEngineChoiceTrigger` to `--disable-features` to suppress the EU search-engine-choice modal that was a related but separate trigger of the same `connect_over_cdp` error pattern.
- Added a single retry on `Browser context management is not supported` in `start()`.

These help with the first-launch case but don't address the across-task Chrome leak.

---

## 2. Learner pipeline — graphs created but unexecutable

**Severity:** HIGH. After 19 actual CU runs only 2 graphs were created, and both have `invocations=None` (no JS entry points found). Today's executor cannot run them, even though the captured HTTP endpoints are demonstrably live and useful (verified via `replay_graph_endpoints.py` — both replayed status 200 with the original captures).

### What's working

- **Observer** captures HTTP request/response pairs cleanly.
- **Noise filter** (Brave adblock + EasyPrivacy + EasyList + supplementary domains) is reasonable.
- **Parameter classification** (LLM, step 6) labels params as `user_intent` / `chained` / `website_generated` correctly.
- **Naming** (LLM, step 11) generates clean capability statements and descriptions.

### What's broken

#### 2.1 Entry-point discovery (step 7) is unreliable

The 5 strategies (reachable global / framework dispatch / extracted function / DOM replay / fetch replay) all failed for both observed graphs. The persisted `required_globals` field shows the discovery code is producing nonsense:

| Graph | `required_globals` | Diagnosis |
|---|---|---|
| `lego_com` | `['0','1','2','3','4','5','6','7','8','9']` | Array indices, not real globals — the discovery code matched something incorrectly. |
| `confirmtkt_com` | `['Object','Function','Array','Number','parseFloat','parseInt','Boolean','String','Symbol','Date']` | JS built-ins. Means discovery couldn't find any application-specific globals; fell back to listing built-ins. Useless as preconditions. |

#### 2.2 Persistence without verification

Even though both graphs have `invocations=None`, they were still **saved to disk**. The pipeline should mark these as unexecutable or discard them entirely. Currently they sit in `tools.json` giving a false impression of progress.

#### 2.3 Lifecycle state machine drops to `None`

Both graphs have `lifecycle: None` instead of `unverified` / `probationary` / `verified`. Steps 8 (HTTP verification) and 9 (pipeline verification) were skipped because `invocations` was empty, so the lifecycle field never got set. The state machine isn't tracking these orphaned graphs.

#### 2.4 Edges field is corrupt

Both graphs' `edges` arrays show `None.None → None.None` for every edge, despite `core_parameters` being correctly tagged with chained sources. The chain detection (step 5) is identifying flow correctly, but the edge serialization step is dropping the source/target node IDs and field names. Bug in the graph object's `to_dict` or in the persistence path.

### Fix — add a 6th entry-point strategy: pure HTTP replay

This is the high-leverage fix. Use the captured request as a recipe (URL template + headers + body shape), substitute user_intent params, send via `curl_cffi` outside the browser. We did this manually in `experiments/replay_graph_endpoints.py` — it worked first-shot for both graphs.

Shifts the executor model from "must invoke through the page's JS" to "JS preferred, HTTP replay otherwise". Much more robust because:
- Doesn't depend on JS bundle introspection.
- Survives bundle hash changes.
- No race against Chrome's CDP state.
- Works for any captured XHR/fetch endpoint.

Estimated effort: 1–2 days. Requires:
- Implementing `_try_http_replay` strategy in `learner.py` (parallels existing `_try_extracted_function` etc.)
- Adapter in `executor.py` to dispatch HTTP replay via `curl_cffi.Session(impersonate=...)` when the chosen invocation type is `http_replay`.
- Updating preconditions check to skip `required_globals` for HTTP-replay-only graphs.
- Re-cookie-injection from the live browser context for stateful endpoints.

### Fix — discard or quarantine empty-invocation graphs

Until 2.5 ships, change `manifest.py` so graphs with `invocations=None` are NOT persisted to `graphs/`. Either:
- Drop them in step 12 (registry persistence) with a clear log line.
- Persist them to a `failed_graphs/` subdir for forensic inspection but keep them out of the main registry.

### Fix — repair the edge serializer

Walk through `learner.py` step 5 → graph object construction. Edges are being built from `chain_candidates` (we see them used correctly in classification) but the `GraphEdge` fields are landing as `None`. Likely a field-name mismatch between what step 5 produces and what `GraphEdge.__init__` expects.

---

## 3. Smaller issues found along the way

### 3.1 Crawler V5 thread management broke Gemini turn ordering

- **File:** `browser-challenge/crawler/master.py:1621` (`_update_v5`).
- **Bug:** Deleted the V5 message at index 1 (the only `user` message), leaving the conversation as `[system, assistant(tool_call), tool, user(V5_new)]`. Gemini's validator rejects every `function_call` not preceded by a user/function_response — including the first one.
- **Fix:** Already applied. Preserve the message at index 1; only delete previously-appended V5s.

### 3.2 LiteLLM's `gemini/` provider mis-orders tool-result turns

- **Workaround:** Already applied in `experiments/run_comparison.py`. Route crawler through Gemini's OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`) by setting `OPENAI_API_BASE` + `OPENAI_API_KEY` and using `--raw-model openai/gemini-2.5-flash`.
- **Long-term:** Either patch LiteLLM's gemini adapter or stay on the OpenAI-compat endpoint indefinitely.

### 3.3 Crawler's `_collect_stability_probe` races navigation

- **File:** `browser-challenge/crawler/browser_tools.py:179-189`.
- **Bug:** Calls `page.evaluate()` while a navigation is in flight; the JS execution context gets recreated and the `evaluate` errors out with `Execution context was destroyed`.
- **Affected:** ~10 of 50 crawler runs in our experiment.
- **Fix:** Wrap the `evaluate` call in a retry-on-context-destroyed loop, or `await page.wait_for_load_state("domcontentloaded")` before probing.

### 3.4 Crawler's loop detector trips on multi-gate sites

- **File:** `browser-challenge/crawler/master.py` (loop detection logic).
- **Bug:** Treats `lego.com/en-in` (post-consent) as the same page as `lego.com/en-in` (pre-consent) because the URL fingerprint matches. Aborts with `loop_detected` after 3 turns of consent flow.
- **Fix:** Loop detector should incorporate DOM signature delta or cookie-state delta in the fingerprint, not just URL.

---

## 4. Test debt — things to add to the smoke test before the next big run

- A **3-task sequential smoke test** on one morphnet worker (different sites, different ports) verifying:
  - No leftover Chrome processes after subprocess exit.
  - No `SingletonLock` files in profile dirs after subprocess exit.
  - No stale Chrome windows visible.
  - Each task gets a fresh CDP session with no cross-task state.
- A **graph-execution smoke test** that runs the captured graphs through the executor with synthetic user_intent and asserts the response shape matches.

---

## 5. Priority order

1. **Fix #1 (Chrome process management)** — blocks all morphnet experiments. ~2 hours of work.
2. **Fix #2.2 + #2.4 (don't persist orphan graphs, repair edge serializer)** — quick wins, prevents misleading state. ~1 hour.
3. **Fix #2.5 (HTTP replay strategy)** — high-leverage, unlocks already-captured semantic value. ~1–2 days.
4. **Fix #2.1 (entry-point discovery)** — mostly subsumed by #2.5. Worth deeper investigation only if HTTP replay isn't sufficient.
5. **Fix #3.3 (crawler probe race)** — improves crawler success rate by ~20%. Independent of morphnet work. ~1 hour.
