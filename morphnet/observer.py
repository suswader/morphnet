"""
observer.py — Always-on recording layer for MorphNet.

Captures CU actions, HTTP traffic with stack traces, script sources,
and DOM state continuously across all subtasks within a task. Makes no
decisions. At task end, produces a single observation containing the
full traffic — the learner needs the complete workflow to build graphs
that can replay an entire task.

Lifecycle:
  start_task()    — sets up CDP, begins accumulating
  start_subtask() — marks subtask boundary (no reset)
  end_subtask()   — collects nav events for this subtask (no teardown)
  end_task()      — tears down CDP, returns full observation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Optional
from urllib.parse import urlparse

from morphnet.manifest import (
    CUAction,
    HTTPRequest,
    ScriptSource,
    DOMSnapshot,
    NavigationEvent,
    SubtaskObservation,
    save_observation,
    save_bundle_metadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise URL filter — analytics, telemetry, tracking domains
# ---------------------------------------------------------------------------

_NOISE_DOMAINS = frozenset({
    "googletagmanager.com", "google-analytics.com", "analytics.google.com",
    "segment.io", "segment.com", "cdn.segment.com", "api.segment.io",
    "mixpanel.com", "api.mixpanel.com",
    "hotjar.com", "static.hotjar.com", "script.hotjar.com",
    "facebook.com", "facebook.net", "connect.facebook.net",
    "clevertap.com", "clevertap-prod.com",
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "google.com/pagead", "adservice.google.com",
    "sentry.io", "browser.sentry-cdn.com",
    "newrelic.com", "bam.nr-data.net", "js-agent.newrelic.com",
    "datadog-agent", "browser-intake-datadoghq.com",
    "fullstory.com", "edge.fullstory.com", "rs.fullstory.com",
    "clarity.ms", "clarity.microsoft.com",
    "amplitude.com", "api.amplitude.com",
    "heap.io", "heapanalytics.com",
    "intercom.io", "widget.intercom.io",
    "crisp.chat",
    "cdn.cookielaw.org", "geolocation.onetrust.com",
    "fonts.googleapis.com", "fonts.gstatic.com",
    "browser-intake-us5-datadoghq.com",
})


def _is_noise_url(url: str) -> bool:
    """Check if a URL belongs to a known analytics/telemetry domain."""
    try:
        host = urlparse(url).hostname or ""
        parts = host.split(".")
        # Check domain at various depths: foo.bar.com → bar.com, foo.bar.com
        for i in range(len(parts)):
            domain = ".".join(parts[i:])
            if domain in _NOISE_DOMAINS:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Request type classification
# ---------------------------------------------------------------------------

def _classify_request_type(
    url: str,
    method: str,
    content_type: str,
    body: Optional[str],
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Classify HTTP request type and extract protocol-specific identifiers.

    Returns: (request_type, graphql_operation_name, graphql_query_hash, jsonrpc_method)
    """
    graphql_op = None
    graphql_hash = None
    jsonrpc_method = None

    # Try GraphQL detection
    if body:
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                if "query" in parsed or "mutation" in parsed:
                    graphql_op = parsed.get("operationName")
                    query_str = parsed.get("query") or parsed.get("mutation") or ""
                    if query_str:
                        graphql_hash = hashlib.sha256(query_str.encode()).hexdigest()[:12]
                    return ("graphql", graphql_op, graphql_hash, None)
                if "jsonrpc" in parsed:
                    jsonrpc_method = parsed.get("method")
                    return ("json_rpc", None, None, jsonrpc_method)
        except (json.JSONDecodeError, TypeError):
            pass

    # Form-encoded
    if content_type and "form-urlencoded" in content_type:
        return ("form", None, None, None)

    # Multipart
    if content_type and "multipart" in content_type:
        return ("form", None, None, None)

    # Default: REST
    return ("rest", None, None, None)


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class Observer:
    """Always-on capture layer. Records everything, decides nothing.

    Accumulates traffic across all subtasks within a task. The learner
    needs the full workflow (auto-suggest → search → navigate) to build
    graphs that replay an entire task via one executor call.

    Lifecycle:
      start_task    — CDP setup, begin accumulating
      start_subtask — mark boundary (no reset)
      end_subtask   — collect nav events (no teardown)
      end_task      — CDP teardown, return full observation
    """

    def __init__(self, session_manager: Any):
        self._session = session_manager
        self._cdp_session = None

        # Task-level state (accumulates across subtasks)
        self._site: str = ""
        self._task_description: str = ""
        self._task_start_url: str = ""
        self._task_start_ts: int = 0

        self._cu_actions: list[CUAction] = []
        self._http_requests: list[HTTPRequest] = []
        self._scripts: dict[str, ScriptSource] = {}
        self._dom_snapshots: list[DOMSnapshot] = []
        self._navigation_events: list[NavigationEvent] = []
        self._framework_fingerprint: dict = {}

        # Current subtask marker
        self._subtask_id: str = ""

        # Request tracking for response matching
        self._pending_requests: dict[str, dict] = {}
        self._last_snapshot_ts: float = 0
        self._last_url: str = ""
        self._active: bool = False

        # Background snapshot task
        self._snapshot_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_task(self, site: str, task_description: str) -> None:
        """Begin recording for a task. Sets up CDP and starts accumulating."""
        self._site = site
        self._task_description = task_description
        self._task_start_ts = int(time.time() * 1000)
        self._task_start_url = self._session.page.url if self._session.page else ""
        self._last_url = self._task_start_url

        # Reset all accumulation state
        self._cu_actions = []
        self._http_requests = []
        self._scripts = {}
        self._dom_snapshots = []
        self._navigation_events = []
        self._pending_requests = {}
        self._framework_fingerprint = {}
        self._active = True

        # Set up CDP session for Network events
        await self._setup_cdp()

        # Framework fingerprinting
        self._framework_fingerprint = await self._detect_frameworks()

        # Initial DOM snapshot
        await self._take_dom_snapshot()

        # Start periodic snapshot task
        self._snapshot_task = asyncio.create_task(self._periodic_snapshots())

        logger.info("Observer started task on %s", site)

    async def start_subtask(self, subtask_id: str, site: str, description: str) -> None:
        """Mark a subtask boundary. No reset — traffic keeps accumulating."""
        self._subtask_id = subtask_id

        # If start_task wasn't called (backward compat), do a full setup
        if not self._active:
            await self.start_task(site, description)
            self._subtask_id = subtask_id
            return

        logger.info("Observer marking subtask boundary: %s", subtask_id)

    async def end_subtask(self, subtask_id: str, reflector_verdict: str) -> None:
        """Mark subtask end. Collects navigation events but keeps recording."""
        # Collect navigation events accumulated during this subtask
        await self._collect_navigation_events()

        # DOM snapshot at subtask boundary
        await self._take_dom_snapshot()

        logger.info(
            "Observer subtask boundary %s: %d total HTTP requests, %d total CU actions so far",
            subtask_id, len(self._http_requests), len(self._cu_actions),
        )

    async def end_task(self) -> SubtaskObservation:
        """Finalize recording for the entire task. Returns full observation."""
        self._active = False

        # Cancel periodic snapshot task
        if self._snapshot_task and not self._snapshot_task.done():
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass

        # Final DOM snapshot
        await self._take_dom_snapshot()

        # Final navigation event collection
        await self._collect_navigation_events()

        # Detach CDP session
        await self._teardown_cdp()

        # Compute bundle hash
        bundle_hash = self._compute_bundle_hash()

        end_url = self._session.page.url if self._session.page else self._task_start_url

        observation = SubtaskObservation(
            subtask_id=f"task_{self._task_start_ts}",
            site=self._site,
            start_url=self._task_start_url,
            end_url=end_url,
            subtask_description=self._task_description,
            start_timestamp_ms=self._task_start_ts,
            end_timestamp_ms=int(time.time() * 1000),
            cu_actions=list(self._cu_actions),
            http_requests=list(self._http_requests),
            scripts=dict(self._scripts),
            dom_snapshots=list(self._dom_snapshots),
            navigation_events=list(self._navigation_events),
            framework_fingerprint=self._framework_fingerprint,
            bundle_hash=bundle_hash,
            reflector_verdict="task_complete",
        )

        # Persist
        save_observation(observation)
        if self._framework_fingerprint and bundle_hash:
            save_bundle_metadata(self._site, bundle_hash, self._framework_fingerprint)

        logger.info(
            "Observer ended task: %d CU actions, %d HTTP requests, %d scripts, %d snapshots, %d nav events",
            len(self._cu_actions), len(self._http_requests),
            len(self._scripts), len(self._dom_snapshots), len(self._navigation_events),
        )
        return observation

    async def _collect_navigation_events(self) -> None:
        """Drain pushState/replaceState events from the page into our accumulator."""
        try:
            page = self._session.page
            if not page or page.is_closed():
                return
            raw_nav = await page.evaluate(
                "() => { const k = Symbol.for('_mn_nav'); const evts = window[k] || []; window[k] = []; return evts; }"
            )
            for evt in raw_nav:
                self._navigation_events.append(NavigationEvent(
                    timestamp_ms=evt.get("ts", 0),
                    nav_type=evt.get("type", "pushState"),
                    url=evt.get("url", ""),
                ))
        except Exception as exc:
            logger.debug("Failed to collect navigation events: %s", exc)

    # ------------------------------------------------------------------
    # CU action capture (called from computer_use.py)
    # ------------------------------------------------------------------

    async def record_cu_action(
        self,
        action_type: str,
        target: dict,
        value: Optional[str],
        reasoning: str,
    ) -> None:
        """Record a CU browser action with its reasoning.

        Args:
            action_type: click, type, select, scroll, etc.
            target: dict with keys: element_id, selector, attributes, text, ax_node_id
            value: typed text, selected value, etc.
            reasoning: CU's rationale for this action.
        """
        if not self._active:
            return

        action = CUAction(
            timestamp_ms=int(time.time() * 1000),
            subtask_id=self._subtask_id,
            action_type=action_type,
            target_selector=target.get("selector", ""),
            target_attributes=target.get("attributes", {}),
            target_text=target.get("text", ""),
            target_ax_node_id=target.get("ax_node_id"),
            typed_value=value,
            cu_reasoning=reasoning,
        )
        self._cu_actions.append(action)
        logger.debug("Recorded CU action: %s (target: %s)", action_type, target.get("selector", "")[:60])

    # ------------------------------------------------------------------
    # CDP setup and event handlers
    # ------------------------------------------------------------------

    async def _setup_cdp(self) -> None:
        """Enable CDP domains for HTTP and script capture."""
        try:
            self._cdp_session = await self._session._context.new_cdp_session(self._session.page)

            # Enable domains. Skip Runtime.enable: triggers consoleAPICalled
            # side effects that bot detectors flag (LEARNINGS rule #3).
            await self._cdp_session.send("Network.enable")
            await self._cdp_session.send("Debugger.enable")
            await self._cdp_session.send("Debugger.setAsyncCallStackDepth", {"maxDepth": 32})

            # Auto-attach to service workers and iframes
            try:
                await self._cdp_session.send("Target.setAutoAttach", {
                    "autoAttach": True,
                    "waitForDebuggerOnStart": False,
                    "flatten": True,
                })
            except Exception:
                pass  # Not critical if Target.setAutoAttach fails

            # Register event handlers
            self._cdp_session.on("Network.requestWillBeSent", self._on_request_will_be_sent)
            self._cdp_session.on("Network.responseReceived", self._on_response_received)
            self._cdp_session.on("Network.loadingFinished", self._on_loading_finished)
            self._cdp_session.on("Debugger.scriptParsed", self._on_script_parsed)

            logger.debug("CDP domains enabled for observer")
        except Exception as exc:
            logger.warning("CDP setup for observer failed: %s", exc)
            self._cdp_session = None

    async def _teardown_cdp(self) -> None:
        """Detach the CDP session."""
        if self._cdp_session:
            try:
                await self._cdp_session.detach()
            except Exception:
                pass
            self._cdp_session = None

    def _on_request_will_be_sent(self, params: dict) -> None:
        """Handle Network.requestWillBeSent — capture request details and initiator stack."""
        try:
            request = params.get("request", {})
            url = request.get("url", "")

            # Skip noise
            if _is_noise_url(url):
                return

            # Only capture XHR/Fetch (skip images, stylesheets, etc.)
            resource_type = params.get("type", "")
            if resource_type not in ("XHR", "Fetch", "Document"):
                return

            request_id = params.get("requestId", "")
            method = request.get("method", "GET")
            headers = request.get("headers", {})
            body = request.get("postData")
            content_type = headers.get("content-type", headers.get("Content-Type", ""))

            # Extract initiator stack
            initiator = params.get("initiator", {})
            initiator_type = initiator.get("type", "other")
            stack_frames = []
            stack = initiator.get("stack")
            if stack:
                stack_frames = self._extract_stack_frames(stack)

            # Classify request type
            req_type, gql_op, gql_hash, rpc_method = _classify_request_type(
                url, method, content_type, body,
            )

            self._pending_requests[request_id] = {
                "timestamp_ms": int(time.time() * 1000),
                "subtask_id": self._subtask_id,
                "request_id": request_id,
                "url": url,
                "method": method,
                "headers": dict(headers),
                "body": body,
                "request_type": req_type,
                "graphql_operation_name": gql_op,
                "graphql_query_hash": gql_hash,
                "jsonrpc_method": rpc_method,
                "initiator_stack": stack_frames,
                "initiator_type": initiator_type,
            }
        except Exception as exc:
            logger.debug("Observer request capture error: %s", exc)

    def _extract_stack_frames(self, stack: dict) -> list[dict]:
        """Recursively extract call frames from a CDP stack trace."""
        frames = []
        for frame in stack.get("callFrames", []):
            frames.append({
                "scriptId": frame.get("scriptId", ""),
                "functionName": frame.get("functionName", ""),
                "lineNumber": frame.get("lineNumber", 0),
                "columnNumber": frame.get("columnNumber", 0),
                "url": frame.get("url", ""),
            })
        # Follow async parent stacks
        parent = stack.get("parent")
        if parent and len(frames) < 32:
            frames.extend(self._extract_stack_frames(parent))
        return frames[:32]  # Cap at async stack depth

    def _on_response_received(self, params: dict) -> None:
        """Handle Network.responseReceived — capture response headers and status."""
        try:
            request_id = params.get("requestId", "")
            pending = self._pending_requests.get(request_id)
            if not pending:
                return

            response = params.get("response", {})
            pending["response_status"] = response.get("status", 0)
            pending["response_headers"] = dict(response.get("headers", {}))
            pending["response_time_ms"] = int(response.get("timing", {}).get("receiveHeadersEnd", 0))
        except Exception as exc:
            logger.debug("Observer response capture error: %s", exc)

    def _on_loading_finished(self, params: dict) -> None:
        """Handle Network.loadingFinished — fetch response body and finalize."""
        request_id = params.get("requestId", "")
        pending = self._pending_requests.pop(request_id, None)
        if not pending:
            return

        # Schedule async response body fetch
        if self._cdp_session:
            asyncio.ensure_future(self._fetch_response_body(request_id, pending))

    async def _fetch_response_body(self, request_id: str, pending: dict) -> None:
        """Fetch response body via CDP and finalize the HTTPRequest record."""
        response_body = None
        try:
            if self._cdp_session:
                result = await self._cdp_session.send("Network.getResponseBody", {
                    "requestId": request_id,
                })
                response_body = result.get("body")
                if result.get("base64Encoded") and response_body:
                    # Skip binary responses
                    response_body = None
        except Exception:
            pass  # Response may have been evicted

        http_req = HTTPRequest(
            timestamp_ms=pending["timestamp_ms"],
            subtask_id=pending["subtask_id"],
            request_id=pending["request_id"],
            url=pending["url"],
            method=pending["method"],
            headers=pending["headers"],
            body=pending.get("body"),
            request_type=pending["request_type"],
            graphql_operation_name=pending.get("graphql_operation_name"),
            graphql_query_hash=pending.get("graphql_query_hash"),
            jsonrpc_method=pending.get("jsonrpc_method"),
            response_status=pending.get("response_status", 0),
            response_headers=pending.get("response_headers", {}),
            response_body=response_body,
            response_time_ms=pending.get("response_time_ms", 0),
            initiator_stack=pending.get("initiator_stack", []),
            initiator_type=pending.get("initiator_type", "other"),
        )
        self._http_requests.append(http_req)

        # Capture script sources referenced in stack frames
        for frame in http_req.initiator_stack:
            script_id = frame.get("scriptId", "")
            if script_id and script_id not in self._scripts:
                await self._capture_script_source(script_id, frame.get("url", ""))

    def _on_script_parsed(self, params: dict) -> None:
        """Handle Debugger.scriptParsed — record script metadata for later source fetching."""
        # We don't fetch sources eagerly for all scripts — only for those
        # referenced in initiator stacks (done in _fetch_response_body).
        # This handler is a no-op but could be extended for pre-caching.
        pass

    async def _capture_script_source(self, script_id: str, url: str) -> None:
        """Fetch and store a script source via Debugger.getScriptSource."""
        if script_id in self._scripts:
            return
        try:
            if not self._cdp_session:
                return
            result = await self._cdp_session.send("Debugger.getScriptSource", {
                "scriptId": script_id,
            })
            source = result.get("scriptSource", "")
            if not source or len(source) < 10:
                return

            content_hash = hashlib.sha256(source.encode()).hexdigest()
            is_module = url.endswith(".mjs") or "type=module" in url

            self._scripts[script_id] = ScriptSource(
                script_id=script_id,
                url=url,
                content_hash=content_hash,
                source=source,
                is_module=is_module,
            )
            logger.debug("Captured script %s (%s, %d bytes)", script_id, url[:60], len(source))
        except Exception as exc:
            logger.debug("Failed to capture script %s: %s", script_id, exc)

    # ------------------------------------------------------------------
    # DOM snapshots
    # ------------------------------------------------------------------

    async def _take_dom_snapshot(self) -> None:
        """Take an AXTree + storage snapshot at the current moment."""
        if not self._active:
            return
        try:
            page = self._session.page
            if not page or page.is_closed():
                return

            # AXTree
            ax_tree = await self._session.get_raw_accessibility_tree()

            # DOM content hash (body innerHTML hash for quick comparison)
            try:
                inner_html = await page.evaluate("() => document.body ? document.body.innerHTML.slice(0, 50000) : ''")
                dom_hash = hashlib.sha256(inner_html.encode()).hexdigest()
            except Exception:
                dom_hash = ""

            # Storage keys
            storage_keys = {"localStorage": [], "sessionStorage": [], "cookies": []}
            try:
                cookies = await self._session.get_cookies()
                storage_keys["cookies"] = [c.get("name", "") for c in cookies]
            except Exception:
                pass
            try:
                storage = await self._session.get_storage()
                storage_keys["localStorage"] = list(storage.get("local_storage", {}).keys())
                storage_keys["sessionStorage"] = list(storage.get("session_storage", {}).keys())
            except Exception:
                pass

            snapshot = DOMSnapshot(
                timestamp_ms=int(time.time() * 1000),
                subtask_id=self._subtask_id,
                url=page.url,
                ax_tree=ax_tree or {},
                dom_content_hash=dom_hash,
                storage_keys=storage_keys,
            )
            self._dom_snapshots.append(snapshot)
            self._last_snapshot_ts = time.time()
            self._last_url = page.url

        except Exception as exc:
            logger.debug("DOM snapshot failed: %s", exc)

    async def _periodic_snapshots(self) -> None:
        """Take DOM snapshots every 5 seconds if no other snapshot was taken."""
        while self._active:
            await asyncio.sleep(5)
            if not self._active:
                break
            # Check if URL changed (triggers snapshot)
            try:
                current_url = self._session.page.url if self._session.page else ""
                if current_url != self._last_url:
                    await self._take_dom_snapshot()
                    continue
            except Exception:
                pass
            # Idle snapshot if none taken in 5s
            if time.time() - self._last_snapshot_ts >= 4.5:
                await self._take_dom_snapshot()

    async def take_snapshot_on_url_change(self) -> None:
        """Explicitly take a snapshot when URL has changed. Called by session_manager or CU."""
        if self._active:
            await self._take_dom_snapshot()

    # ------------------------------------------------------------------
    # Framework fingerprinting
    # ------------------------------------------------------------------

    async def _detect_frameworks(self) -> dict:
        """Detect frontend frameworks and reachable globals in the current page.

        Probes window.* for React, Redux, Apollo, Vue, Angular, Next.js.
        Enumerates non-standard window properties up to depth 3.
        """
        page = self._session.page
        if not page or page.is_closed():
            return {}

        try:
            fingerprint = await page.evaluate("""() => {
                const result = {
                    frameworks: [],
                    globals: {},
                };

                // React
                if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) result.frameworks.push('react');
                if (window.__REACT_QUERY_STATE__) result.frameworks.push('react-query');

                // Redux
                if (window.__REDUX_DEVTOOLS_EXTENSION__ || window.__REDUX_STORE__) result.frameworks.push('redux');
                if (window.store && typeof window.store.dispatch === 'function') result.frameworks.push('redux-store');

                // Apollo
                if (window.__APOLLO_CLIENT__) result.frameworks.push('apollo');

                // Vue
                if (window.__VUE_DEVTOOLS_GLOBAL_HOOK__) result.frameworks.push('vue');
                if (window.__NUXT__) result.frameworks.push('nuxt');
                if (window.__INITIAL_STATE__) result.frameworks.push('vue-ssr');

                // Angular
                if (window.ng) result.frameworks.push('angular');
                if (window.Zone) result.frameworks.push('angular-zone');

                // Next.js
                if (window.__NEXT_DATA__) result.frameworks.push('nextjs');
                const nextScripts = document.querySelectorAll('script[src*="_next/"]');
                if (nextScripts.length > 0 && !result.frameworks.includes('nextjs')) {
                    result.frameworks.push('nextjs');
                }

                // Enumerate non-standard window properties (depth 1-3)
                const BROWSER_GLOBALS = new Set([
                    'window', 'self', 'document', 'location', 'navigator', 'history',
                    'screen', 'performance', 'console', 'crypto', 'fetch', 'alert',
                    'confirm', 'prompt', 'setTimeout', 'setInterval', 'clearTimeout',
                    'clearInterval', 'requestAnimationFrame', 'cancelAnimationFrame',
                    'addEventListener', 'removeEventListener', 'dispatchEvent',
                    'getComputedStyle', 'matchMedia', 'open', 'close', 'focus', 'blur',
                    'print', 'scroll', 'scrollTo', 'scrollBy', 'resizeTo', 'resizeBy',
                    'moveTo', 'moveBy', 'postMessage', 'atob', 'btoa',
                    'localStorage', 'sessionStorage', 'indexedDB', 'caches',
                    'origin', 'length', 'name', 'status', 'closed', 'frames',
                    'parent', 'top', 'opener', 'frameElement', 'customElements',
                    'visualViewport', 'styleMedia', 'isSecureContext', 'crossOriginIsolated',
                    'chrome', 'webkitRequestAnimationFrame', 'webkitCancelAnimationFrame',
                ]);

                const nonStandard = {};
                const visited = new WeakSet();
                function walkGlobals(obj, path, depth) {
                    if (depth > 3 || !obj) return;
                    try { if (typeof obj === 'object') { if (visited.has(obj)) return; visited.add(obj); } } catch(e) { return; }
                    const keys = depth === 0
                        ? Object.getOwnPropertyNames(obj)
                        : Object.keys(obj);
                    for (const key of keys) {
                        if (depth === 0 && (BROWSER_GLOBALS.has(key) || (key.startsWith('__') && key.endsWith('__')) || key.startsWith('on'))) continue;
                        try {
                            const val = obj[key];
                            const t = typeof val;
                            const fullPath = depth === 0 ? key : path + '.' + key;
                            if (t === 'function') {
                                nonStandard[fullPath] = 'function';
                            } else if (t === 'object' && val !== null) {
                                nonStandard[fullPath] = 'object';
                                walkGlobals(val, fullPath, depth + 1);
                            }
                        } catch(e) {}
                    }
                }
                walkGlobals(window, '', 0);
                result.globals = nonStandard;

                return result;
            }""")
            return fingerprint
        except Exception as exc:
            logger.debug("Framework fingerprinting failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Bundle hash
    # ------------------------------------------------------------------

    def _compute_bundle_hash(self) -> str:
        """Combine content hashes of all captured scripts into a single bundle identity."""
        if not self._scripts:
            return ""
        hashes = sorted(s.content_hash for s in self._scripts.values())
        combined = "|".join(hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
