# LEARNINGS — what we actually established

Empirical results from Phases 0-8 against `bot-detector.rebrowser.net`,
`creepjs`, and 17 production sites (tier 0-5). Where this contradicts
`How Internet Works.md` (the design-document we started with), the
empirical result wins and is called out explicitly.

## TL;DR

The "minimal viable" Chrome-via-raw-CDP setup for production scraping is
much smaller than the textbook implies and often reduces to:

```
chrome --remote-debugging-port=PORT \
       --remote-allow-origins=* \
       --user-data-dir=/path/to/fresh/profile \
       --no-first-run --no-default-browser-check \
       --window-size=1920,1080 \
       --use-gl=angle --use-angle=default \
       [--user-agent="Mozilla/5.0 ... Chrome/147.0.0.0 ..."]   # only if --headless=new
```

**Almost no JS init scripts.** The textbook's recommended `navigator.webdriver`
override is actively harmful (makes detector score *worse*). The WebRTC and
MediaDevices patches solve problems Chrome 147 doesn't have by default.

**Network/Debugger/Runtime CDP commands are mostly free** as long as
`Runtime.enable` is never called at session start.

**The hot-path production pattern** is `live headed Chrome (state) + curl_cffi
(transport)` — one shared Chrome session feeds cookies + CSRF + JS state to a
curl_cffi worker that does the actual API calls. Validated end-to-end against
Swiggy: 5-stage flow in 3.7s, byte-perfect responses, **8-15× faster than a
computer-use agent** on the same intent.

The detection boundary that actually matters in practice is **session-state
binding** (CSRF tied to cookies tied to IP-reputation), not the exotic
TLS/JS-fingerprinting the textbook focuses on.

---

## Architecture validated (Phase 8)

Hybrid pattern: **live headed Chrome** owns session state, **curl_cffi** owns
HTTP transport.

```
┌──────────────────────────┐         ┌──────────────────────────┐
│  Live headed Chrome      │         │  curl_cffi.Session       │
│  (Phase 6c min-viable    │         │  (impersonate="chrome")  │
│   flags, --remote-debug) │         │                          │
│                          │         │                          │
│  • cookie jar (incl.     │   ─►    │  • cookies loaded from   │
│    HttpOnly aws-waf-token│         │    Network.getAllCookies │
│  • window._csrfToken     │   ─►    │  • _csrf substituted     │
│  • runs site JS to keep  │         │    into POST bodies      │
│    state fresh           │         │  • 5 stages in ~3.7s     │
└──────────────────────────┘         └──────────────────────────┘
   live state extraction               replay via JA4-matched HTTP
   (CDP: Network.getAllCookies          to the same endpoints
   + Runtime.evaluate)
```

**What this validated for Swiggy:**

| stage | endpoint | captured | live-replay | byte match |
|---|---|---|---|---|
| 1 | POST /dapi/misc/place-autocomplete | 200 | 200 | size differs (session IDs) |
| 2 | POST /dapi/misc/address-recommend | 200 | 200 | size differs (session IDs) |
| 3 | GET /dapi/homepagev2/getCards | 200 | 200 | **byte-identical** (308,309 = 308,309) |
| 4 | GET /dapi/restaurants/search/v3 | 200 | 200 | size +7.5% (inventory drift) |
| 5 | GET /dapi/menu/pl | 200 | 200 | **byte-identical** (709,140 = 709,140) |

Total: **3.69s** end-to-end including state extraction. ~30-60s on a CU
agent for the same intent → **8-15× speedup**.

Detail in `experiments/phase_08_swiggy_direct_api/8c_live_state/FINDINGS.md`.

---

## Things `How Internet Works.md` got wrong, that we empirically corrected

The textbook is mostly right about *categories* of detection (TLS, H2,
headers, JS env, cookies, IP). It's wrong about specific *defenses* — many
recommended patches are decorative, redundant, or actively harmful on
modern Chrome.

### 1. Don't patch `navigator.webdriver` to undefined

**Textbook says:** "navigator.webdriver=undefined via --disable-blink-features=AutomationControlled"

**Empirical (Phase 6a-6f):**
- Native value on raw-CDP Chrome is `false` (the W3C-spec value for non-WebDriver Chrome).
- The flag `--disable-blink-features=AutomationControlled` is a **no-op** on our launch profile (we never had `--enable-automation`, so webdriver was never going to be `true`).
- The JS patch `Object.defineProperty(navigator, 'webdriver', {get: () => undefined})` actually **flips bot-detector.rebrowser.net's verdict from green to red** because `undefined` is unreachable through normal browser runtime. The detector flags it as deliberate property deletion.

**Action:** Don't apply the patch on modern Chrome. Native `false` already passes.

→ `memory/project_webdriver_undefined_makes_score_worse.md`

### 2. `Page.addScriptToEvaluateOnNewDocument` requires `Page.enable`

**Textbook says:** "patch navigator.webdriver via `Page.addScriptToEvaluateOnNewDocument` before navigation"

**Empirical (Phase 6d-6e):** the script registers fine (CDP returns `{identifier: '1'}`) but Chrome 147 silently does NOT execute it on subsequent documents unless `Page.enable` has been called on the same session first. Verified by canary line `window.__patchRan = Date.now()` — `false` everywhere until we added `Page.enable`.

**Action:** always pair `Page.addScriptToEvaluateOnNewDocument` with `Page.enable`. Verify with a canary line; don't trust silent CDP success.

→ `memory/project_addscript_needs_page_enable.md`

### 3. Most "stealth flags" are no-ops; the real wins are launch flags

**Phase 6a-6c progression**: across `bot-detector.rebrowser.net`'s 11 probes:
- 6a (no stealth): 10/11 ✅, 1 🟠 viewport
- 6b (`--disable-blink-features=AutomationControlled`): unchanged
- 6c (+ `--window-size=1920,1080 --use-gl=angle --use-angle=default`): **11/11 ✅**

**Adding the JS patches in 6d/6e/6f *regressed* the score** (1 🔴 from the `webdriver=undefined` patch). The WebRTC SDP scrubber was redundant (Chrome 147's default IP-handling policy already hides local IPs). The MediaDevices `enumerateDevices` spoof never substituted — Chrome on macOS without permission already returns 3 default-y entries.

**Action:** the bulk of stealth value is in launch flags that produce real-world signal values (real window size, real GPU rendering). JS patches that override Chrome's natural state mostly create new tells.

→ `experiments/phase_06_stealth_patches/SYNTHESIS.md`

### 4. `Page.navigate` is wire-indistinguishable from a typed URL

**Textbook implies:** programmatic navigation lacks user-gesture signals (`Sec-Fetch-User: ?1`).

**Empirical (Phase 3):** byte-identical Sec-Fetch headers between `Page.navigate` and the manual baseline:

```
sec-fetch-site: none
sec-fetch-mode: navigate
sec-fetch-user: ?1   ← present in both, including Page.navigate
sec-fetch-dest: document
upgrade-insecure-requests: 1
```

`Page.navigate`'s default `transitionType: "typed"` is classified as user-initiated by Chrome's network service (DevTools uses `Page.navigate` for address-bar typing — the header has to be set or DevTools-driven browsing would look bot-like to every site).

**Action:** don't budget defense effort against `Sec-Fetch-User` for top-level CDP-driven navigation. It's not a meaningful signal there.

→ `memory/project_page_navigate_emits_sec_fetch_user.md`

### 5. `--headless=new` still leaks "HeadlessChrome" in legacy User-Agent

**Textbook implies:** modern `--headless=new` is "closer to headed than legacy headless".

**Empirical (Phase 7 headless sweep):** Chrome 147 with `--headless=new` sends `HeadlessChrome/147.0.0.0` in the legacy `User-Agent` header, vs `Chrome/147.0.0.0` headed. **One token different. All other request fields are byte-identical** (TLS, H2, Sec-CH-UA Client Hints say "Google Chrome" not "HeadlessChrome", Sec-Fetch-*, Accept-*, etc.).

7 of 17 sites escalated headed → headless. **6 of 7 (myntra/zomato/bookmyshow/makemytrip/amazon_in/lego) recovered to their headed verdict with just `--user-agent="...Chrome/147..."`** (one-flag fix). Reddit was the only site doing JS-runtime headless detection beyond the UA regex.

**Action:** if you must use `--headless=new`, always pass `--user-agent` with "HeadlessChrome" stripped. ~85% of headless escalation is a 1990s-style UA-string regex.

→ `memory/project_headless_chrome_still_in_ua_chrome_147.md`

### 6. AWS WAF on Swiggy uses a JS-set token, not Set-Cookie

**Textbook focuses on:** Cloudflare `cf_clearance`, Akamai `_abck`, DataDome — all set via HTTP `Set-Cookie`.

**Empirical (Phase 0):** Swiggy is on AWS WAF (not Cloudflare or Akamai). Its `aws-waf-token` cookie is set by `challenge.js` via `document.cookie` — never appears in HTTP `Set-Cookie` events. An HTTP-layer parser sees it being **sent** on subsequent requests but never sees it being **set**.

**Action:** any cookie analysis must inspect both `cookies_set` (HTTP-set) and `cookies_sent` (Cookie request headers) streams. Edge-cookie classifiers must include `aws-waf-*` prefix. Tier 2 in our site list shouldn't assume Cloudflare.

→ `memory/project_swiggy_aws_waf.md`

### 7. AWS WAF detects via *absence of telemetry*, not request fingerprint (revised)

**Initial finding (Phase 7 vs baseline):** Swiggy hard-blocked our automation despite byte-perfect requests because our `challenge.js` never made the telemetry callbacks to `*.edge.sdk.awswaf.com`. We hypothesized this was gesture-gated.

**Revised (Phase 8c):** the telemetry-absence hypothesis was wrong about WHAT triggers the absence. A fresh standalone session (no gestures, no mouse moves) DID get the SDK to phone home. The Phase 7 hard-block was IP-rate-limit stacking from running 6 sites sequentially before swiggy's turn.

**Action:** for multi-site automation from a single residential IP, pace requests and cool down between sites. Sequential automated runs accumulate WAF suspicion.

→ `memory/project_aws_waf_detects_by_telemetry_absence.md` (note the update)

### 8. Swiggy validates CSRF–session binding (Phase 8 keystone)

**Discovered Phase 8:** `_csrf` is bound to the session that owns `__SW`/`_guest_tid`/`_sid` cookies. Mixing them produces 403:

| | cookies | CSRF | POST endpoints | GET endpoints |
|---|---|---|---|---|
| 8b verbatim | stale | stale | ✅ 200 | ❌ 202 (WAF challenge — captured aws-waf-token doesn't satisfy high-trust GETs) |
| 8c first | live | stale | ❌ 403 (CSRF mismatch) | ✅ 200 |
| 8c final | **live** | **live** | **✅ 200** | **✅ 200** |

**Where the CSRF lives:** `<script>window._csrfToken = "..."</script>` inline in the swiggy.com homepage HTML. Not in a meta tag, not in cookies, not in `window.csrfToken`. Discovered forensically by searching the captured homepage HTML for the captured CSRF value.

**AWS WAF differential trust per endpoint:** low-value POST (place-autocomplete, address-recommend) accept stale tokens; high-value GET (getCards, search, menu) require fresh `aws-waf-token`. Single token; different trust thresholds per endpoint.

**Action:** Always extract `(cookies, _csrfToken)` together at the same moment from the same live Chrome. Don't mix and match.

→ `memory/project_swiggy_csrf_cookie_binding.md`

---

## Per-tier detection reality (Phase 7 sweep, headed mode, corrected verdicts)

Tier list from `experiments/_shared/site_list.py`:

| tier | sites | what we saw | mechanism |
|---|---|---|---|
| **0 control** | wikipedia, github | `clean` both | no detection at all |
| **1 light** | reddit*, flipkart, myntra | mostly `soft-flag` | first-party + light Cloudflare; cookies issued, page renders. *Reddit fires "Prove your humanity" challenges sometimes (variance) |
| **2 CDN bot mgmt** | swiggy, zomato, confirmtkt | `clean` to `hard-block` | AWS WAF on swiggy (we mislabeled tier 2 as Cloudflare in CLAUDE.md — it's mixed) |
| **3 custom JS** | bookmyshow, cleartrip, makemytrip | `soft-flag` mostly | Cloudflare + Akamai stack; pages render but post-load 4xx on some API endpoints — that's degradation not block |
| **4 enterprise** | amazon.in, airbnb, nike | `clean` to `hard-block` | DataDome (airbnb), Akamai BMP (nike). Stochastic — same setup loads sometimes, hard-blocks other times |
| **5 hostile** | lego, leboncoin, vinted | mostly `soft-flag` | CF Waiting Room + DataDome stacks; we passed through both on lego and vinted |

**Key insight (revised after Phase 9):** "soft-flag" doesn't mean we're being
detected as a bot at the wire fingerprint layer. `__cf_bm`, `cf_clearance`,
`_abck`, `datadome`, `aws-waf-token` all fire on human browser sessions too.
The Phase 7 `SIGNALS.md` diff confirmed every bot-management cookie *name*
that fired for our automation also fired for the human baseline.

**But cookie names ≠ cookie values.** Phase 9 re-parsed the same netlogs
without the 16-char Set-Cookie value truncation that
`experiments/_shared/observers.py` applied at line 189, and decoded `_abck`'s
internal structure. Result: on 2 of 4 Akamai sites (myntra, nike) the cookie
*value's* field 1 read `-1` (sensor data not validated as human) for our
automation, while the human baseline reached `0` (validated). On 2 sites
(cleartrip, makemytrip) automation also reached `0`. So we are
**name-indistinguishable but verdict-distinguishable** at the cookie layer
on Akamai-protected sites, and per-site outcomes vary even with identical
launch flags.

→ `experiments/phase_09_cookie_value_decode/FINDINGS.md`

→ `experiments/phase_07_multi_site_sweep/SWEEP_RESULTS.md` (headed verdicts)
→ `experiments/phase_07_multi_site_sweep/HEADED_VS_HEADLESS.md` (mode diff)
→ `experiments/phase_07_multi_site_sweep/SIGNALS.md` (baseline-vs-automation diff)

### Headless penalty (after corrected verdict logic)

**7 of 17 sites escalate headed → headless** — and 6 of 7 are fixed by
`--user-agent` override removing "HeadlessChrome". The remaining one
(reddit) does have JS-runtime headless detection.

| site | headed | headless | headless+UA-fix |
|---|---|---|---|
| myntra, zomato, bookmyshow, makemytrip, amazon_in, lego | (mixed) | hard-block | **recovers** |
| reddit | soft-flag | hard-block | still hard-block (real JS detection) |

---

## Per-phase summary

| phase | what it tested | finding |
|---|---|---|
| 0 | manual baseline on Swiggy with `--log-net-log` | Swiggy uses AWS WAF (not CF). `aws-waf-token` is JS-set. 19 challenge headers from `*.awswaf.com` |
| 1 | bare CDP attach, no commands sent | invisible — verdict identical to manual |
| 2 | single `Runtime.evaluate` (no `Runtime.enable`) | invisible — `navigator.webdriver=false`, page renders, no cookies change |
| 3 | programmatic `Page.navigate` | wire-indistinguishable; `Sec-Fetch-User: ?1` set the same as typed URL |
| 4 | `Network.enable` + event capture | invisible — no escalation in cookies or status |
| 5 | `Debugger.enable` + `setAsyncCallStackDepth` on 4 sites (swiggy/confirmtkt/bookmyshow/lego) | clean across all 4. Apparent swiggy soft-flag was a false positive (timing artifact in `aws-waf-token` rotation) |
| 6 (a-c) | launch flags only — no JS | flag-based stealth converges at 6c (window-size + ANGLE) → 11/11 green on bot-detector |
| 6 (d-f) | layered JS patches | webdriver=undefined → red (regression). WebRTC scrubber + MediaDevices spoof were no-ops on Chrome 147 with default flags. `Page.enable` requirement discovered |
| 7 headed | 17-site sweep | 5 clean / 9 soft-flag / 3 hard-block (corrected). Cookie diff vs human baseline: zero. |
| 7 headless | same sweep with `--headless=new` | 7 sites escalate; 6 fixed by `--user-agent` override |
| 8a | manual capture of full Swiggy flow with full request/response bodies | 211 swiggy requests captured; 5 stages identified |
| 8b | verbatim curl_cffi replay with stale captured state | 2/5 stages succeed; high-trust GETs hit WAF challenge |
| 8c | live-Chrome state extraction + curl_cffi replay | **5/5 stages succeed in 3.7s, byte-perfect on deterministic endpoints** |
| 9 | re-parse Phase 7 netlogs, decode `_abck` and `aws-waf-token` value structure | `_abck` field 1 = `-1` (bot) on myntra/nike auto vs `0` (human) on baseline. Cookie *names* matched but *values* didn't — overturns "wire-indistinguishable at cookie layer" |
| 10 | bot-detector.rebrowser.net against raw-CDP no-enable / raw-CDP+`Runtime.enable` / Playwright `connect_over_cdp` eager / Playwright lazy | All 5 variants pass `runtimeEnableLeak` AND `pwInitScripts` on Chrome 148 + Playwright 1.59. The classic V8-inspector eager-format leak does not fire; Playwright 1.59 doesn't inject main-world tells via `connect_over_cdp`. Textbook is dated. |

---

## Operational rules that emerged

1. **Profiles are one-shot.** Once a Chrome profile has been issued
   bot-management cookies, it's contaminated as a test subject. Delete
   `experiments/phase_NN_*/profile/` before each run.

2. **Raw CDP only.** No Playwright (injects `__pwInitScripts`, calls
   `Runtime.enable` on attach), no Selenium (sets `navigator.webdriver=true`).
   Drive Chrome with `subprocess.Popen` + raw WebSocket. Required since
   Chrome 111: `--remote-allow-origins=*` (or our Origin gets rejected).

3. **Lazy CDP domain enable.** Never call `Runtime.enable` at session start —
   it triggers `consoleAPICalled`-family side effects bot detectors check.
   `Network.enable` and `Debugger.enable` are safer.

4. **Headed mode by default.** `--headless=new` works for ~12 of 17 sites with
   the UA override; for the rest, headed is the only choice.

5. **`curl_cffi` not `requests`** for any direct API call against a CDN-protected
   target. Plain `requests` has the wrong JA4, wrong HTTP/2 fingerprint,
   wrong header order. `impersonate="chrome"` (no version, auto-tracks) is the
   right invocation.

6. **State extraction is one bundle**: cookies + CSRF must come from the same
   live Chrome at the same moment. Mixing fails.

7. **IP rate-limit stacking is real.** 6+ sequential automated runs from one
   residential IP escalate WAF suspicion enough to break sites that pass
   standalone (Phase 7 swiggy did this twice). Pace and cool down.

---

## What's still unknown / next experiments

1. **State TTL.** How long does the (cookies, csrf) bundle stay valid? Test at
   +5/+15/+60 min. Sets the live-Chrome session reuse policy in production.

2. **Multi-intent reuse.** How many curl_cffi replays can we do off one live
   state extraction before something rotates or invalidates?

3. **CSRF rotation.** Does `_csrfToken` rotate per session-action or only on
   re-navigation? Per-action would force more frequent state harvesting.

4. **JS-module replay** for `queryUniqueId`, `metaData`, `trackingId`. We
   kept these as captured for short-window replay. For longer windows or
   substituted user intent, we'd need to actually re-invoke the page's JS
   modules to regenerate them — production MorphNet's harder problem.

5. **Tier 4-5 sites under Phase 8 pattern.** We validated only Swiggy. Does
   the live-state + curl_cffi pattern hold against Akamai BMP (nike, lego),
   DataDome (airbnb, leboncoin), or do they bind state to TLS session?

6. **Reddit's residual JS-runtime detection** (the 1 of 7 headless escalations
   that the UA fix didn't recover). What specifically does Reddit probe?

7. **xvfb / headed-but-hidden** — middle ground between headed (real GPU,
   visible window) and `--headless=new` (no visible window, JS leaks). Run
   Phase 7 sweep with xvfb to see if it's the best of both.

---

## File map

For each phase, the load-bearing artifact:

```
LEARNINGS.md                                                          ← you are here
How Internet Works.md                                                 (textbook, partially obsoleted)
CLAUDE.md                                                             (rules for future sessions)

experiments/_baseline/SUMMARY.md                                      Phase 0 — manual Swiggy
experiments/phase_01_cdp_attach/{SUMMARY,DIFF}.md                     Phase 1
experiments/phase_02_runtime_evaluate/INTUITION.md                    Phase 2 — empirical writeup
experiments/phase_03_page_navigate/{INTUITION,first_request_headers}.md Phase 3 — Sec-Fetch finding
experiments/phase_04_network_enable/INTUITION.md                      Phase 4
experiments/phase_05_debugger_enable/AGGREGATE.md                     Phase 5 — multi-site
experiments/phase_05_debugger_enable/swiggy_no_debugger/COMPARISON.md  ↑ control variant
experiments/phase_06_stealth_patches/SYNTHESIS.md                     Phase 6 — stealth-patches table
experiments/phase_07_multi_site_sweep/SWEEP_RESULTS.md                Phase 7 headed
experiments/phase_07_multi_site_sweep/HEADED_VS_HEADLESS.md           Phase 7 mode comparison
experiments/phase_07_multi_site_sweep/SIGNALS.md                      Phase 7 vs human baseline
experiments/phase_08_swiggy_direct_api/8c_live_state/FINDINGS.md      Phase 8 — production architecture validated
experiments/phase_09_cookie_value_decode/FINDINGS.md                  Phase 9 — _abck verdict flag decode
experiments/phase_10_runtime_enable_impact/SYNTHESIS.md               Phase 10 — Runtime.enable + Playwright connectOverCDP impact

memory/MEMORY.md                                                      project memory index
memory/project_*.md                                                   per-finding rationale + how-to-apply
```
