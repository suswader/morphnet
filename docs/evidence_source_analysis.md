# Evidence Source Analysis: What Drives MorphNet Decisions?

**Dataset:** 345 runs across 14 site domains, 10,133 action steps, 39,304 reasoning samples  
**Sites with meaningful sample sizes:** confirmtkt.com (162 runs), in.bookmyshow.com (53), lego.com (47), cleartrip.com (25), youtube.com (17)  
**Date:** 2026-04-29

---

## Executive Summary

**AXTree is the dominant evidence source across every module, every site, and every task type — accounting for 43.3% of all evidence references.** DOM extraction contributes only 0.2% of evidence references and is essentially unused by the models. Screenshots matter in specific, narrow situations (8.3% of CU reasoning) but are not a primary decision driver. The biggest lever for improving task success isn't adding more evidence types — it's ensuring the AXTree is complete and well-labeled.

### Key Findings

1. **DOM is nearly useless.** 0.2% of all evidence references. When it appears in reasoning, it's always in the *negative* sense — "DOM doesn't show this" — never as positive evidence for a decision. Removing DOM extraction from the planner prompt would have negligible impact.

2. **AXTree is the single source of truth.** 49.8% of planner evidence, 52.7% of CU evidence, 43.9% of reflector evidence. The model references AXTree elements using TOON notation (`[N] btn"label"`) in ~90% of its decisions.

3. **Screenshots are a fallback, not a primary input.** Used in 8.3% of CU reasoning, almost exclusively when AXTree fails — missing iframe content, unlabeled elements, ambiguous modals. Cleartrip shows 15% screenshot usage (complex filter modals) vs YouTube at 4% (well-structured AXTree).

4. **Traffic/API data is critical for the reflector and MCP path.** 9.9% of reflector evidence and the dominant signal for `securedapi.confirmtkt.com` (39.7%) where MCP executor runs.

5. **Page content (visible text) is the #2 signal for CU** at 21.7% — the model reads text from the page to understand context, confirm state, and extract data.

6. **Failed tasks show more traffic references and fewer page_content references** than successful ones, suggesting failures correlate with complex HTTP interactions and limited readable content.

---

## 1. Evidence Source Usage by Module

### Planner (Orchestrator)

| Source | Share | Role |
|--------|-------|------|
| AXTree | 49.8% | Primary: reads page structure to decompose tasks |
| Planning tree | 10.2% | Reviews subtask outcomes for next-step decisions |
| Action history | 10.0% | Tracks what's been attempted to avoid repetition |
| Website profile | 9.4% | Learned site insights (Cloudflare blocks, UI patterns) |
| URL | 7.1% | Confirms navigation state |
| Page content | 6.9% | Reads visible text for context |
| Screenshot | **0.4%** | Almost never used |
| DOM | **0.3%** | Essentially unused |

**Reasoning analysis (5,366 planner samples):**
- 95.1% reference previous actions/steps
- 88.1% explicitly mention AXTree
- 65.6% reference the planning tree
- 29.3% reference website insights
- 10.9% mention URL
- **0.4% mention DOM**
- **2.4% mention screenshots**

**Takeaway:** The planner relies on AXTree for page understanding and the planning tree for decision continuity. DOM extraction adds no value to planning. The planner almost never looks at screenshots.

### CU Agent (Computer Use)

| Source | Share | Role |
|--------|-------|------|
| AXTree | 52.7% | Primary: element identification and action targeting |
| Page content | 21.7% | Reads visible text to understand page state |
| Action history | 12.1% | Tracks what was just done |
| URL | 5.1% | Confirms page location |
| Element state | 4.4% | Checks enabled/disabled, expanded/collapsed |
| Screenshot | **3.3%** | Fallback when AXTree is insufficient |
| DOM | **0.1%** | Unused |

**Reasoning analysis (22,710 CU samples):**
- 72.5% reference previous actions
- 67.1% reference specific element IDs (`[N]` notation)
- 41.2% explicitly mention AXTree
- **8.3% mention screenshots** — almost exclusively when:
  - reCAPTCHA/iframe content is visible but has no AXTree representation
  - Filter modals have poorly-labeled elements needing visual confirmation
  - Dialog state is ambiguous (e.g., "filter dialog is actually open in the screenshot")
- **0.2% mention DOM**

**Takeaway:** CU agent lives in the AXTree. Screenshots are a bailout for when the AXTree fails. DOM is irrelevant.

### Reflector

| Source | Share | Role |
|--------|-------|------|
| AXTree | 43.9% | Primary: checks current page state after action |
| URL | 22.5% | URL changes are a core deterministic signal |
| Traffic | 9.9% | HTTP status codes for success/failure signals |
| Page content | 9.6% | Verifies content changes |
| Element state | 8.0% | Element count diffs (added/removed) |
| Action history | 2.7% | What was attempted |
| ARIA | 1.4% | Alert/error signals |
| Screenshot | **1.0%** | Almost never used |
| DOM | **1.0%** | Almost never used |

**Reasoning analysis (11,228 reflector samples):**
- 15.8% explicitly mention AXTree
- 13.6% mention URL
- 12.5% mention errors/failures
- **0.9% mention screenshots**
- **1.0% mention DOM**

**Takeaway:** Reflector operates on structured signals — AXTree diffs, URL changes, HTTP codes. It never looks at screenshots and never uses DOM. Its three-stage pipeline (deterministic → AXTree diff → LLM) means most decisions happen without any LLM call at all.

---

## 2. Evidence Source Usage by Site

### Tier 1: Primary Sites (>25 runs)

| Site | Runs | AXTree | Page Content | Traffic | Screenshot | DOM | Success Rate |
|------|------|--------|-------------|---------|------------|-----|-------------|
| confirmtkt.com | 162 | 44.1% | 14.0% | 10.7% | 1.8% | 0% | 51.9% (84/162) |
| in.bookmyshow.com | 53 | 48.6% | 11.8% | 16.9% | 1.9% | 0% | 1.9% (1/53) |
| lego.com | 47 | 42.7% | 23.1% | 12.8% | 1.4% | 0% | 14.9% (7/47) |
| cleartrip.com | 25 | 39.6% | 4.8% | 28.8% | 4.1% | 0% | 4.0% (1/25) |

### Tier 2: Secondary Sites (5-25 runs)

| Site | Runs | AXTree | Page Content | Traffic | Screenshot | DOM |
|------|------|--------|-------------|---------|------------|-----|
| youtube.com | 17 | 44.0% | 6.4% | 34.3% | 0.9% | 0% |
| swiggy (via WAF) | 10 | 35.6% | 19.1% | 14.2% | 1.3% | 0% |
| securedapi.confirmtkt.com | 5 | 20.4% | 4.6% | **39.7%** | 0% | 0% |

### Site-Specific Patterns

**cleartrip.com — Highest screenshot usage (15% of CU reasoning)**
- Complex date pickers and filter modals with poorly labeled AXTree elements
- 61.2% of action steps had screenshots enabled (highest of any site)
- CU agent explicitly said things like "element [307] appears to be the checkbox for '5-star' within that modal, which is not explicitly labeled in the AXTree but is visible in the overlay"
- Average AXTree size: 13,732 chars (large pages with many unlabeled elements)

**lego.com — Highest page_content share (23.1%)**
- Product listing pages with extensive visible text (product names, prices, descriptions)
- CU agent reads product cards to identify specific LEGO sets
- Average AXTree size: 8,368 chars but heavily text-dense
- ARIA signals appeared here (0.3%) — likely from modal/accordion states

**youtube.com — Highest traffic share (34.3%)**
- YouTube's SPA architecture generates heavy API traffic for every interaction
- Low screenshot need (4% CU) because YouTube has excellent AXTree structure
- Average AXTree size: 18,589 chars (large, but well-structured)

**confirmtkt.com — Highest MCP executor usage (49 runs with executor)**
- The only site with mature MCP tools, so traffic/API evidence is consumed differently
- Planner shows 27% API/executor references (vs. 1-2% for other sites)
- `securedapi.confirmtkt.com` runs show 98% API/executor refs in planner — when MCP tools are available, the planner relies on API responses, not AXTree

**in.bookmyshow.com — Lowest screenshot engagement (14.2% of steps)**
- Most steps run without screenshots despite complex UI
- Only 1 successful run out of 53 (frequent Cloudflare blocks)
- When failures happen, the model tries to read the AXTree for error pages

**reddit.com — Highest AXTree size (avg 46,644 chars)**
- Reddit generates massive AXTree due to nested comment threads
- DOM was referenced 0.8% — higher than most, because Reddit's dynamic content sometimes needs DOM-level inspection
- reCAPTCHA was the primary blocker — "visible in screenshot but missing from AXTree"

---

## 3. Evidence by Task Type

| Task Type | Runs | AXTree | Traffic | Page Content | Screenshot | DOM |
|-----------|------|--------|---------|-------------|------------|-----|
| Search | 228 | 43.1% | 18.9% | 12.3% | 2.5% | 0% |
| Navigation | 59 | 42.8% | 14.5% | 20.4% | 1.5% | 0% |
| Transactional | 20 | 47.9% | 13.5% | 15.6% | 2.5% | 0% |
| Verification | 18 | 46.7% | 10.6% | 5.0% | 1.8% | 0% |
| Comparison | 11 | 46.4% | 7.8% | 4.7% | 1.2% | 0% |

**Key patterns:**
- **Navigation tasks** have the highest page_content share (20.4%) — the model reads more text to understand where it is on content-heavy pages
- **Search tasks** have the highest traffic share (18.9%) — search triggers API calls that generate traffic evidence
- **Transactional tasks** have the highest AXTree share (47.9%) — form filling requires precise element identification
- **DOM is 0% across every task type** — there is no task category where DOM extraction provides meaningful value

---

## 4. Screenshot Deep Dive

### When Screenshots Are Used

Screenshots are referenced in CU agent reasoning in three specific patterns:

**Pattern 1: AXTree-invisible elements (iframes, dynamic content)**
> "The reCAPTCHA iframe and its internal elements are not present in the accessibility tree. Despite visual evidence in screenshots, the lack of DOM/AXTree representation prevents further progress."

This is the most common pattern. The model sees something in the screenshot that has zero representation in the AXTree — typically iframes (reCAPTCHA, payment widgets) or dynamically injected overlays.

**Pattern 2: Ambiguous/unlabeled elements needing visual confirmation**
> "Based on the visual layout, element [307] appears to be the checkbox for '5-star' within that modal, which is not explicitly labeled in the AXTree but is visible in the overlay."

Cleartrip dominates this pattern. Its filter modals use generic `div` elements with no accessible labels, forcing the model to cross-reference the screenshot to identify which element ID maps to which UI control.

**Pattern 3: State confirmation when AXTree is ambiguous**
> "The previous attempt to click 'Search filters' [20] resulted in an error, but the filter dialog is actually open in the screenshot."

The model uses the screenshot to resolve contradictions — when an action reports failure but the UI actually changed, or vice versa.

### Screenshot Usage by Site (Step-Level)

| Site | Steps with Screenshot | Steps without | % with Screenshot |
|------|----------------------|---------------|-------------------|
| cleartrip.com | 729 | 462 | **61.2%** |
| featuregates.org | 168 | 281 | 37.4% |
| unpkg.com | 124 | 241 | 34.0% |
| analytics.swiggy.com | 38 | 81 | 31.9% |
| confirmtkt.com | 615 | 1,612 | 27.6% |
| lego.com | 682 | 2,361 | 22.4% |
| in.bookmyshow.com | 279 | 1,684 | **14.2%** |
| youtube.com | 40 | 261 | **13.3%** |
| reddit.com | 2 | 11 | 15.4% |

**Correlation with AXTree quality:** Sites with well-structured AXTree (YouTube, BookMyShow) need fewer screenshots. Sites with poorly-labeled elements (Cleartrip) need more screenshots. This suggests that improving AXTree enrichment (better label association, unnamed element recovery) could reduce screenshot dependency further.

---

## 5. Success vs Failure Patterns

| Source | Successful (108 runs) | Failed (42 runs) | Differential |
|--------|----------------------|-------------------|-------------|
| AXTree | 44.5% | 43.3% | +1.2% (stable) |
| **Traffic** | **15.4%** | **22.0%** | **-6.6%** |
| **Page content** | **14.0%** | **8.7%** | **+5.3%** |
| Action history | 10.0% | 8.5% | +1.5% |
| URL | 5.9% | 6.3% | -0.4% |
| Screenshot | 1.9% | 2.4% | -0.4% |
| Planning tree | 1.8% | 3.0% | -1.2% |

**Significant differentials:**

1. **Traffic is 6.6% higher in failures.** Failed runs generate more HTTP traffic references because: (a) the model gets stuck in retry loops generating more requests, (b) error codes (403, 404, 413) appear in failure reasoning, (c) Cloudflare/WAF blocks are detected via HTTP responses.

2. **Page content is 5.3% higher in successes.** Successful runs read more meaningful text from pages — product names, train schedules, search results. When the model can read the page content, it succeeds more. When content is blocked or hidden behind overlays, it fails.

3. **Planning tree is 1.2% higher in failures.** Failed runs spend more time re-planning and re-evaluating the planning tree, which makes sense — failures trigger more orchestrator calls and more subtask reflections.

4. **AXTree is stable across success and failure.** It's always the primary source regardless of outcome, confirming it's the foundation, not a differentiator.

---

## 6. Edge Cases Where Specific Sources Become Critical

### When Screenshots Become Essential

- **Cleartrip hotel filters:** 353-446 screenshot references in a single run. The star-rating filter modal uses generic elements with no accessible labels. Without screenshots, the agent cannot identify which checkbox is "5 star."
- **reCAPTCHA on Reddit:** The captcha is visible in the screenshot but has zero AXTree representation. The model explicitly states: "Despite visual evidence in screenshots, the lack of DOM/AXTree representation prevents further progress."
- **BookMyShow popups:** Modal dialogs for city selection and movie details sometimes render before AXTree updates, creating a window where only the screenshot shows the current state.

### When Traffic/API Data Becomes Essential

- **securedapi.confirmtkt.com (MCP executor runs):** Traffic is the #1 evidence source at 39.7%. When MCP tools are available, the system bypasses CU entirely and reads API responses. The planner shows 98% API/executor references.
- **Cleartrip:** Traffic share is 28.8% (highest among CU-driven sites). Cleartrip's flight/hotel search triggers complex API calls; the reflector uses HTTP status codes to verify whether searches actually executed.
- **YouTube:** Traffic share is 34.3%. YouTube's SPA generates extensive API calls for every navigation, and the observer captures rich traffic data.

### When ARIA Signals Appear

ARIA is rare overall (0.2%) but spikes in specific contexts:
- **analytics.swiggy.com:** 2.9% ARIA share — Swiggy uses ARIA roles extensively for its dynamic food ordering UI
- **YouTube (Despacito run):** 24 ARIA references — YouTube's media player exposes ARIA states for playback controls
- **confirmtkt.com (Ranchi-Kolkata run):** 14 ARIA references — form validation errors use aria-invalid signals

### When DOM Actually Matters (0.2% — Almost Never)

The only consistent pattern where DOM is explicitly mentioned:
- **Reddit reCAPTCHA:** "The lack of DOM/AXTree representation for the captcha prevents further progress" — DOM is cited as a negative signal (something expected but missing)
- **BookMyShow Delhi-NCR events:** 8 DOM references in a 30-step plan — likely when the page has dynamic content loaded via JavaScript that appears in DOM before AXTree hydration
- **featuregates.org (Cleartrip via WAF):** 9 DOM references — when Cloudflare challenge pages render, the DOM shows the challenge structure before the AXTree exposes interactive elements

In every case, DOM is referenced to explain a problem, never as the primary basis for a decision.

---

## 7. The MCP Transition: How Evidence Sources Shift

The most dramatic evidence shift occurs between CU-driven runs and MCP executor-driven runs:

| Metric | CU-driven (confirmtkt.com, 113 runs) | MCP-driven (securedapi, 5 runs) |
|--------|--------------------------------------|----------------------------------|
| AXTree | 44.1% | 20.4% |
| Traffic/API | 10.7% | 39.7% |
| MCP tools | 0.8% | 5.6% |
| Planner API refs | 27% | **98%** |
| Page content | 14.0% | 4.6% |

When MCP tools are available and the executor runs API calls directly, the entire evidence hierarchy inverts. AXTree drops from dominant to secondary, and traffic/API responses become the primary decision driver. This confirms the MCP pipeline is working as designed — replacing browser-level observation with API-level data.

---

## 8. Recommendations

### Immediate Actions

1. **Remove DOM extraction from the planner prompt.** It consumes tokens but contributes 0.3% of planner evidence (24 mentions out of 5,366 reasoning samples) and 0% actionable decisions. This will reduce prompt size with zero impact on performance.

2. **Consider removing DOM from CU agent input.** 0.1% of CU evidence, 0.2% of CU reasoning. The AXTree already provides everything the CU agent needs. Saving these tokens frees space for larger AXTree representations.

3. **Keep screenshots but make them conditional.** Screenshots matter on sites with poorly-labeled AXTree elements (Cleartrip 61%, Swiggy 32%) but are wasted on well-structured sites (YouTube 13%, BookMyShow 14%). Consider triggering screenshots only when:
   - Previous action failed and AXTree shows no change
   - Current page has many unlabeled interactive elements
   - The site profile indicates screenshot dependency

4. **Invest in AXTree enrichment quality.** Since AXTree is the dominant evidence source (43-53% across all modules), improving its quality has the highest ROI:
   - Better label association for unlabeled elements (reduces screenshot dependency)
   - Better iframe content exposure (addresses the reCAPTCHA gap)
   - Better lazy-loaded content detection (addresses page_content gaps on product listings)

### Strategic Observations

5. **Page content is a success predictor.** 14% in successes vs 8.7% in failures. Tasks succeed more when the model can read meaningful text. Consider enriching the AXTree with more visible text content where possible.

6. **Traffic volume is a failure predictor.** 15.4% in successes vs 22% in failures. High traffic references correlate with stuck states, retry loops, and error responses. Consider adding circuit breakers for traffic-heavy reflector evaluations.

7. **The MCP transition validates the architecture.** When MCP tools mature for a site, evidence shifts from AXTree (visual) to API (structured data), exactly as designed. Accelerating MCP tool discovery for high-volume sites (BookMyShow, Cleartrip) should be a priority.

---

## Appendix: Methodology

**Evidence classification:** 13 evidence categories defined by regex patterns, applied to:
- `evidence_sources` arrays in trace events
- `reasoning` text in trace events
- `response.evidence_sources` in step files

**Reasoning pattern analysis:** 14 pattern detectors applied to 39,304 reasoning text samples:
- Element ID references (`[N]` notation)
- AXTree/screenshot/DOM/URL explicit mentions
- Previous action references
- Error/failure mentions
- Form/field mentions
- Planning tree references
- Website insight references

**Task classification:** Rules-based classification into search/navigation/transactional/comparison/verification/other based on task description keywords.

**Success/failure:** Determined by `result.success` field in trace events. 108 successes, 42 failures, remainder had no explicit result.
