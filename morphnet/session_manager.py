"""
session_manager.py — Browser session, state extraction, action execution, traffic capture.

Owns the Chrome CDP connection. Every other MorphNet module operates through this.
No LLM calls. No task interpretation. Pure infrastructure.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from curl_cffi import requests as cffi_requests
from playwright.async_api import (
    async_playwright,
    Page,
    BrowserContext,
    Browser,
    Playwright,
    Response,
)

import httpx
from google import genai
from google.genai import types as genai_types

from morphnet.trace import TaskTrace

logger = logging.getLogger(__name__)

# Load .env file from project root (GEMINI_API_KEY, etc.)
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Suppress Node.js DEP0169 (url.parse deprecation) from Playwright internals
os.environ.setdefault("NODE_OPTIONS", "--no-deprecation")

SITES_DIR = Path(__file__).parent / "sites"

_STRIP_TAGS_RE = re.compile(
    r'<(script|style|noscript|svg|link\s)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
_STRIP_INLINE_STYLE_RE = re.compile(r'\s+style="[^"]*"', re.IGNORECASE)
_STRIP_CLASS_RE = re.compile(r'\s+class="[^"]*"', re.IGNORECASE)
_COLLAPSE_WHITESPACE_RE = re.compile(r'\n\s*\n+')


# ---------------------------------------------------------------------------
# Shared Gemini Inference Utility
# ---------------------------------------------------------------------------

_prompt_counter: int = 0


def _save_prompt_log(
    path: Path,
    model: str,
    system_instruction: str | None,
    contents: list[Any],
    generation_config: dict | None,
) -> None:
    """Save prompt contents to a text file for debugging. Skips binary image data."""
    try:
        lines: list[str] = [
            f"Model: {model}",
            f"Config: {json.dumps(generation_config or {}, default=str)}",
            "",
        ]
        if system_instruction:
            lines.append("=== SYSTEM INSTRUCTION ===")
            lines.append(system_instruction)
            lines.append("")
        lines.append("=== CONTENTS ===")
        for i, item in enumerate(contents):
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict) and "mime_type" in item:
                lines.append(f"[Image: {item['mime_type']}]")
            else:
                lines.append(f"[Non-text content: {type(item).__name__}]")
            if i < len(contents) - 1:
                lines.append("---")
        path.write_text("\n".join(lines), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to save prompt log: %s", exc)


def call_gemini(
    *,
    model: str,
    contents: list[Any],
    generation_config: dict | None = None,
    response_schema: Any | None = None,
    system_instruction: str | None = None,
    prompt_log_dir: Path | None = None,
) -> Any:
    """Shared Gemini inference utility. Each module provides its own model,
    schema, prompt, and config — this function just handles the call.

    Returns the parsed response object (structured output if response_schema
    is provided, raw text otherwise).
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "in .env or environment."
        )
    client = genai.Client(api_key=api_key)

    # Normalize contents: convert raw image dicts to proper Part objects.
    # Callers pass {"mime_type": "image/jpeg", "data": "<base64>"} for convenience;
    # the google.genai SDK requires genai_types.Part with inline_data.
    normalized: list[Any] = []
    for item in contents:
        if isinstance(item, dict) and "mime_type" in item and "data" in item:
            normalized.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type=item["mime_type"],
                        data=base64.b64decode(item["data"]),
                    )
                )
            )
        else:
            normalized.append(item)

    gc = dict(generation_config or {})
    # Gemini 3 models default to "high" thinking which consumes output tokens.
    # Set a thinking budget so structured output isn't truncated.
    gc.setdefault("max_output_tokens", 8192)
    gc.setdefault("thinking_config", genai_types.ThinkingConfig(thinking_budget=2048))

    config = genai_types.GenerateContentConfig(**gc)
    if system_instruction is not None:
        config.system_instruction = system_instruction
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    # Save prompt to disk for debugging
    if prompt_log_dir is not None:
        global _prompt_counter
        _prompt_counter += 1
        prompt_log_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_log_dir / f"{_prompt_counter:03d}_{model}.txt"
        _save_prompt_log(prompt_file, model, system_instruction, contents, generation_config)

    # Retry on transient network errors (server disconnect, timeout, etc.)
    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=normalized,
                config=config,
            )
            break
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_exc = exc
            logger.warning("Gemini API transient error (attempt %d/3): %s", _attempt + 1, exc)
            time.sleep(2 ** _attempt)  # 1s, 2s, 4s backoff
        except genai.errors.ClientError as exc:
            # Gemini 400 "Unable to process input image" — transient, retry with same payload
            if "unable to process input image" in str(exc).lower():
                last_exc = exc
                logger.warning("Gemini image processing error (attempt %d/3), retrying: %s", _attempt + 1, exc)
                time.sleep(2 ** _attempt)
                continue
            raise
    else:
        raise last_exc  # type: ignore[misc]

    # Log token usage for performance analysis
    usage = getattr(response, "usage_metadata", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
        thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        logger.debug(
            "Gemini %s: prompt=%d output=%d thinking=%d total=%d",
            model, prompt_tokens, candidates_tokens, thoughts_tokens, total_tokens,
        )

    # If structured output was requested, parse the JSON response
    if response_schema is not None:
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning("Truncated JSON from %s, retrying with larger budget. Raw: %s",
                           model, (response.text or "")[:200])
            # Retry with more tokens and less thinking
            gc["max_output_tokens"] = 16384
            gc["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=2048)
            config2 = genai_types.GenerateContentConfig(**gc)
            if system_instruction is not None:
                config2.system_instruction = system_instruction
            config2.response_mime_type = "application/json"
            config2.response_schema = response_schema
            response2 = client.models.generate_content(
                model=model, contents=normalized, config=config2,
            )
            return json.loads(response2.text)
    return response.text


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Screenshot:
    image_base64: str           # Full-page JPEG as base64
    url: str                    # URL when screenshot was taken
    timestamp: float
    viewport_height: int
    viewport_width: int
    full_page_height: int       # Total scrollable page height


@dataclass
class InteractiveElement:
    element_id: int                 # Stable numeric ID (matches AXTree / DOM / SoM)
    tag: str                        # HTML tag: button, a, input, select, textarea, div, …
    role: str                       # ARIA role or inferred role
    name: str                       # Accessible name / visible text / placeholder
    element_type: str | None        # input type: text, password, checkbox, submit, …
    value: str | None               # Current value for inputs/selects
    bounding_box: dict              # {x, y, width, height} in page coordinates
    is_visible: bool                # CSS-visible AND inside current viewport
    attributes: dict                # Key attrs: href, data-testid, name, id, placeholder, type
    states: list[str]               # focused, checked, expanded, disabled, required, selected
    selector: str                   # Playwright-compatible selector for action execution
    fingerprint: str                # Structural identity for stable ID assignment


@dataclass
class CapturedRequest:
    url: str
    method: str
    request_headers: dict
    response_headers: dict
    request_body: str | None
    request_body_parsed: dict | None
    response_body: str | None
    response_body_parsed: Any | None
    status_code: int
    resource_type: str              # "xhr" or "fetch"
    timestamp: float
    request_content_type: str
    response_content_type: str
    # Derived — populated by classify_request()
    protocol: str | None = None         # rest | graphql | jsonrpc | form | multipart | unknown
    endpoint_identity: str | None = None
    is_state_changing: bool | None = None

    # ------------------------------------------------------------------
    def classify_request(self) -> None:
        """Populate derived fields: protocol, endpoint_identity, is_state_changing.

        Attempts body parsing first, then classifies in priority order:
        GraphQL -> JSON-RPC -> URL-encoded form -> Multipart -> REST.
        """
        self._parse_bodies()
        url_path = urlparse(self.url).path

        if self._classify_graphql(url_path):
            return
        if self._classify_jsonrpc(url_path):
            return
        if self._classify_form(url_path):
            return
        if self._classify_multipart(url_path):
            return
        self._classify_rest(url_path)

    # --- body parsing helpers -------------------------------------------

    def _parse_bodies(self) -> None:
        # Request body
        if self.request_body and self.request_body_parsed is None:
            ct = self.request_content_type.lower()
            if "json" in ct or "graphql" in ct:
                try:
                    self.request_body_parsed = json.loads(self.request_body)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif "x-www-form-urlencoded" in ct:
                try:
                    self.request_body_parsed = {
                        k: v[0] if len(v) == 1 else v
                        for k, v in parse_qs(self.request_body).items()
                    }
                except Exception:
                    pass
        # Response body
        if self.response_body and self.response_body_parsed is None:
            if "json" in self.response_content_type.lower():
                try:
                    self.response_body_parsed = json.loads(self.response_body)
                except (json.JSONDecodeError, TypeError):
                    pass

    # --- protocol classifiers -------------------------------------------

    def _classify_graphql(self, url_path: str) -> bool:
        is_graphql_url = any(
            url_path.rstrip("/").endswith(suffix)
            for suffix in ("/graphql", "/gql", "/graphql/")
        )
        body = self.request_body_parsed
        has_query_field = (
            isinstance(body, dict)
            and isinstance(body.get("query"), str)
            and any(
                body["query"].lstrip().startswith(kw)
                for kw in ("query", "mutation", "subscription", "{")
            )
        )
        if not (is_graphql_url or has_query_field):
            return False

        self.protocol = "graphql"
        if isinstance(body, dict):
            op_name = body.get("operationName")
            if op_name:
                self.endpoint_identity = op_name
            else:
                # First keyword after query/mutation
                q = (body.get("query") or "").lstrip()
                for prefix in ("query ", "mutation ", "subscription "):
                    if q.startswith(prefix):
                        rest = q[len(prefix):].strip()
                        name = rest.split("(")[0].split("{")[0].split(" ")[0].strip()
                        self.endpoint_identity = name or prefix.strip()
                        break
                else:
                    self.endpoint_identity = "anonymous_query"
            # Mutations are state-changing regardless of HTTP method
            query_text = (body.get("query") or "").lstrip()
            self.is_state_changing = query_text.startswith("mutation")
        else:
            self.endpoint_identity = f"graphql:{url_path}"
            self.is_state_changing = True
        return True

    def _classify_jsonrpc(self, url_path: str) -> bool:
        body = self.request_body_parsed
        if not isinstance(body, dict):
            return False
        if "jsonrpc" not in body or "method" not in body:
            return False
        self.protocol = "jsonrpc"
        self.endpoint_identity = body["method"]
        method_name = str(body["method"]).lower()
        state_prefixes = ("set", "create", "delete", "update", "remove", "add", "put", "patch", "insert")
        self.is_state_changing = any(method_name.startswith(p) for p in state_prefixes) or True
        return True

    def _classify_form(self, url_path: str) -> bool:
        if "x-www-form-urlencoded" not in self.request_content_type.lower():
            return False
        self.protocol = "form"
        self.endpoint_identity = f"{self.method} {url_path}"
        self.is_state_changing = self.method.upper() == "POST"
        return True

    def _classify_multipart(self, url_path: str) -> bool:
        if "multipart/form-data" not in self.request_content_type.lower():
            return False
        self.protocol = "multipart"
        self.endpoint_identity = f"{self.method} {url_path}"
        self.is_state_changing = True
        return True

    def _classify_rest(self, url_path: str) -> None:
        self.protocol = "rest"
        self.endpoint_identity = f"{self.method} {url_path}"
        self.is_state_changing = self.method.upper() in ("POST", "PUT", "DELETE", "PATCH")


@dataclass
class ActionResult:
    success: bool
    error: str | None = None
    navigation_occurred: bool = False
    new_url: str | None = None
    status_code: int | None = None
    page_ready: bool = True
    elements_stale: bool = False    # True after navigation



# ---------------------------------------------------------------------------
# JavaScript: enumerate interactive elements (depth-first, document order)
# ---------------------------------------------------------------------------

_ENUMERATE_ELEMENTS_JS = """() => {
    const results = [];
    const interactiveTags = new Set([
        'button', 'a', 'input', 'select', 'textarea', 'details', 'summary'
    ]);
    const interactiveRoles = new Set([
        'button', 'link', 'menuitem', 'tab', 'checkbox', 'radio', 'switch',
        'slider', 'combobox', 'searchbox', 'option', 'menuitemcheckbox',
        'menuitemradio', 'treeitem', 'gridcell', 'spinbutton',
        'scrollbar', 'progressbar', 'textbox'
    ]);
    // Only check cursor:pointer on tags plausibly made interactive via JS
    const cursorCheckTags = new Set([
        'div', 'span', 'li', 'td', 'img', 'svg', 'p', 'section', 'article', 'tr', 'th', 'label'
    ]);

    function isInteractive(el) {
        const tag = el.tagName.toLowerCase();
        if (tag === 'option') return false;
        if (interactiveTags.has(tag)) return true;
        const role = el.getAttribute('role');
        if (role && interactiveRoles.has(role)) return true;
        const tabIndex = el.getAttribute('tabindex');
        if (tabIndex !== null && parseInt(tabIndex, 10) >= 0) return true;
        // Inline event handlers
        if (el.hasAttribute('onclick') || el.hasAttribute('onmousedown') || el.hasAttribute('ontouchstart')) return true;
        // data-testid — testing infrastructure marks intentionally interactive elements
        if (el.hasAttribute('data-testid')) {
            const testId = el.getAttribute('data-testid');
            const containerRe = /^(page|content|wrapper|container|layout|root|app|main|section)/i;
            if (!containerRe.test(testId)) return true;
        }
        // Cursor check for non-native tags
        if (cursorCheckTags.has(tag)) {
            try {
                if (window.getComputedStyle(el).cursor === 'pointer') return true;
            } catch (e) {}
        }
        // ARIA landmark role="search" with few children (likely the click target itself)
        if (role === 'search') {
            try {
                if (window.getComputedStyle(el).cursor === 'pointer') return true;
            } catch (e) {}
            if (el.children.length <= 3) return true;
        }
        // ARIA state attributes suggesting interactivity
        if (cursorCheckTags.has(tag)) {
            if (el.hasAttribute('aria-haspopup') || el.hasAttribute('aria-expanded') ||
                el.hasAttribute('aria-pressed') || el.hasAttribute('aria-selected')) {
                return true;
            }
        }
        // React/Preact synthetic event handlers — detect onClick on framework-managed elements.
        // Only check near-leaf elements (children <= 5) to avoid crashing heavy DOMs.
        if (cursorCheckTags.has(tag) && el.children.length <= 5) {
            try {
                const names = Object.getOwnPropertyNames(el);
                for (let i = 0; i < names.length; i++) {
                    const k = names[i];
                    if (k.startsWith('__reactProps$') || k.startsWith('__reactEventHandlers$')) {
                        const props = el[k];
                        if (props && (props.onClick || props.onMouseDown || props.onTouchStart)) {
                            return true;
                        }
                        break;  // Only one React fiber key per element
                    }
                }
            } catch (e) {}
        }
        return false;
    }

    function getRole(el) {
        const explicit = el.getAttribute('role');
        if (explicit) return explicit;
        const tag = el.tagName.toLowerCase();
        const type = (el.getAttribute('type') || '').toLowerCase();
        if (tag === 'input') {
            const m = {
                text: 'textbox', email: 'textbox', url: 'textbox', tel: 'textbox',
                password: 'textbox', search: 'searchbox', number: 'spinbutton',
                checkbox: 'checkbox', radio: 'radio', range: 'slider',
                submit: 'button', reset: 'button', button: 'button',
                date: 'textbox', time: 'textbox', datetime: 'textbox',
                'datetime-local': 'textbox', month: 'textbox', week: 'textbox',
                color: 'textbox', file: 'button', hidden: 'none',
            };
            return m[type] || 'textbox';
        }
        const tagMap = {
            button: 'button', a: 'link', select: 'combobox',
            textarea: 'textbox', details: 'group', summary: 'button',
        };
        return tagMap[tag] || tag;
    }

    function getAccessibleName(el) {
        // Priority: aria-label > aria-labelledby > label[for] > placeholder > title > innerText
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return ariaLabel.trim();

        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const parts = labelledBy.split(/\\s+/).map(id => {
                const ref = document.getElementById(id);
                return ref ? ref.textContent.trim() : '';
            }).filter(Boolean);
            if (parts.length) return parts.join(' ');
        }

        if (el.id) {
            const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
            if (lbl) return lbl.textContent.trim();
        }
        // Walk parent to find wrapping <label>
        let parent = el.parentElement;
        while (parent) {
            if (parent.tagName === 'LABEL') return parent.textContent.trim();
            parent = parent.parentElement;
        }

        if (el.placeholder) return el.placeholder.trim();
        if (el.title) return el.title.trim();

        const text = (el.innerText || el.textContent || '').trim();
        return text.length > 80 ? text.substring(0, 80) + '...' : text;
    }

    function buildSelector(el) {
        // Priority: data-testid > #id > [name] > text > css path
        const testId = el.getAttribute('data-testid');
        if (testId) return '[data-testid="' + testId + '"]';

        if (el.id) {
            try {
                if (document.querySelectorAll('#' + CSS.escape(el.id)).length === 1) {
                    return '#' + CSS.escape(el.id);
                }
            } catch (e) {}
        }

        const name = el.getAttribute('name');
        if (name) {
            const sel = el.tagName.toLowerCase() + '[name="' + name + '"]';
            try {
                if (document.querySelectorAll(sel).length === 1) return sel;
            } catch (e) {}
        }

        // text= selector for buttons and links with short unique text
        const tag = el.tagName.toLowerCase();
        if ((tag === 'button' || tag === 'a') && el.innerText) {
            const txt = el.innerText.trim();
            if (txt.length > 0 && txt.length <= 50) {
                const sameTag = [...document.querySelectorAll(tag)];
                const matches = sameTag.filter(e => (e.innerText || '').trim() === txt);
                if (matches.length === 1) return tag + ':has-text("' + txt.replace(/"/g, '\\\\"') + '")';
            }
        }

        return getCSSPath(el);
    }

    function getCSSPath(el) {
        const parts = [];
        let cur = el;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            let part = cur.tagName.toLowerCase();
            if (cur.id) {
                try {
                    if (document.querySelectorAll('#' + CSS.escape(cur.id)).length === 1) {
                        parts.unshift('#' + CSS.escape(cur.id));
                        break;
                    }
                } catch (e) {}
            }
            const parent = cur.parentElement || (cur.getRootNode && cur.getRootNode()).host;
            if (parent && parent.children) {
                const siblings = [...parent.children].filter(s => s.tagName === cur.tagName);
                if (siblings.length > 1) {
                    const idx = siblings.indexOf(cur) + 1;
                    part += ':nth-of-type(' + idx + ')';
                }
            }
            parts.unshift(part);
            cur = cur.parentElement;
        }
        return parts.join(' > ');
    }

    function getStates(el) {
        const states = [];
        if (document.activeElement === el) states.push('focused');
        if (el.checked) states.push('checked');
        if (el.getAttribute('aria-expanded') === 'true') states.push('expanded');
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') states.push('disabled');
        if (el.required || el.getAttribute('aria-required') === 'true') states.push('required');
        if (el.selected) states.push('selected');
        return states;
    }

    function getAttributes(el) {
        const attrs = {};
        const want = ['href', 'data-testid', 'name', 'id', 'placeholder', 'type', 'action', 'method', 'target'];
        for (const a of want) {
            const v = el.getAttribute(a);
            if (v !== null && v !== '') attrs[a] = v;
        }
        // Collect data-* attributes (often useful for MCP discovery)
        for (const a of el.attributes) {
            if (a.name.startsWith('data-') && a.name !== 'data-testid' && a.value) {
                attrs[a.name] = a.value;
            }
        }
        return attrs;
    }

    function getValue(el) {
        const tag = el.tagName.toLowerCase();
        if (tag === 'select') {
            if (el.selectedIndex >= 0 && el.options[el.selectedIndex]) {
                return el.options[el.selectedIndex].text;
            }
            return null;
        }
        if (typeof el.value === 'string') return el.value || null;
        return el.getAttribute('value') || null;
    }

    function walk(node) {
        if (!node || node.nodeType !== Node.ELEMENT_NODE) return;

        if (isInteractive(node)) {
            const rect = node.getBoundingClientRect();
            if (rect.width > 0 || rect.height > 0) {
                const tag = node.tagName.toLowerCase();
                const role = getRole(node);
                const rawName = getAccessibleName(node);
                const selector = buildSelector(node);
                const nameNorm = (rawName || '').toLowerCase().replace(/[0-9]+/g, '').trim();

                results.push({
                    tag: tag,
                    role: role,
                    name: rawName,
                    element_type: tag === 'input' ? (node.getAttribute('type') || 'text') : null,
                    value: getValue(node),
                    bounding_box: {
                        x: Math.round(rect.x + window.scrollX),
                        y: Math.round(rect.y + window.scrollY),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height),
                    },
                    is_visible: (function() {
                        if (rect.width <= 0 || rect.height <= 0) return false;
                        const cs = window.getComputedStyle(node);
                        if (cs.display === 'none') return false;
                        if (rect.bottom <= 0 || rect.top >= window.innerHeight) return false;
                        if (cs.visibility === 'hidden') {
                            // Angular/PrimeNG hide native inputs behind styled overlays.
                            // If it has a valid bbox and is a real form control, still interactable.
                            const formTags = new Set(['input', 'select', 'textarea']);
                            return formTags.has(tag) && rect.width >= 10 && rect.height >= 10;
                        }
                        return true;
                    })(),
                    attributes: getAttributes(node),
                    states: getStates(node),
                    selector: selector,
                    fingerprint: tag + '|' + role + '|' + selector + '|' + nameNorm,
                });
            }
        }

        // Recurse into shadow DOM
        if (node.shadowRoot) {
            for (const child of node.shadowRoot.children) walk(child);
        }
        // Recurse into children
        for (const child of node.children) walk(child);
    }

    walk(document.body);
    return results;
}"""



# ---------------------------------------------------------------------------
# JavaScript: extract meta tokens (CSRF, form keys, auth tokens)
# ---------------------------------------------------------------------------

_META_TOKENS_JS = """() => {
    const tokens = {};

    // 1. Meta tags
    const metaNames = ['csrf-token', '_csrf', 'csrf-param', 'csrf_token',
                        'authenticity_token', 'X-CSRF-TOKEN'];
    for (const name of metaNames) {
        const el = document.querySelector('meta[name="' + name + '"]');
        if (el && el.content) {
            tokens['meta_' + name] = {
                value: el.content,
                source: 'meta_tag',
                selector: 'meta[name="' + name + '"]'
            };
        }
    }

    // 2. Hidden form fields
    const hiddenNames = ['_token', 'csrfmiddlewaretoken', 'authenticity_token',
                         'form_key', 'csrf_token', '_csrf', '__RequestVerificationToken',
                         'nonce', 'wp_nonce'];
    for (const name of hiddenNames) {
        const el = document.querySelector('input[type="hidden"][name="' + name + '"]');
        if (el && el.value) {
            tokens['hidden_' + name] = {
                value: el.value,
                source: 'hidden_field',
                selector: 'input[type="hidden"][name="' + name + '"]'
            };
        }
    }
    // Also grab ALL hidden inputs (some have site-specific names)
    document.querySelectorAll('input[type="hidden"]').forEach(el => {
        if (el.name && el.value && !tokens['hidden_' + el.name]) {
            tokens['hidden_' + el.name] = {
                value: el.value,
                source: 'hidden_field',
                selector: 'input[type="hidden"][name="' + el.name + '"]'
            };
        }
    });

    // 3. JS variables commonly holding tokens
    const jsVars = [
        'window.csrfToken', 'window._token', 'window.__csrf',
        'window.__INITIAL_STATE__', 'window.Laravel',
    ];
    for (const path of jsVars) {
        try {
            const val = eval(path);
            if (val && typeof val === 'string') {
                tokens['js_' + path] = {value: val, source: 'js_variable', key: path};
            } else if (val && typeof val === 'object') {
                // Check common nested keys
                for (const k of ['csrfToken', 'csrf_token', 'token', '_token']) {
                    if (val[k]) {
                        tokens['js_' + path + '.' + k] = {
                            value: String(val[k]), source: 'js_variable', key: path + '.' + k
                        };
                    }
                }
            }
        } catch (e) {}
    }

    // 4. localStorage / sessionStorage auth tokens
    const storageKeys = ['token', 'authToken', 'auth_token', 'access_token',
                         'jwt', 'id_token', 'session', 'csrf'];
    for (const store of ['localStorage', 'sessionStorage']) {
        try {
            const s = window[store];
            for (let i = 0; i < s.length; i++) {
                const key = s.key(i);
                const lk = key.toLowerCase();
                if (storageKeys.some(sk => lk.includes(sk))) {
                    tokens[store + '_' + key] = {
                        value: s.getItem(key),
                        source: store,
                        key: key
                    };
                }
            }
        } catch (e) {}
    }

    return tokens;
}"""


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """Persistent Chrome browser session for MorphNet.

    Provides: page state extraction, action execution, network traffic capture,
    cookie/session management.  Zero LLM calls — pure infrastructure.
    """

    def __init__(
        self,
        start_url: str,
        task_prompt: str,
        evaluation_mode: bool = False,
        evaluation_benchmark: str = "webarena",
        headless: bool = True,
        chrome_cdp_url: str = "http://localhost:9222",
        viewport_width: int = 1440,
        viewport_height: int = 900,
        site_name: str | None = None,
        trace: TaskTrace | None = None,
    ):
        # Config — stored, not interpreted
        self.start_url = start_url
        self.task_prompt = task_prompt
        self.evaluation_mode = evaluation_mode
        self.evaluation_benchmark = evaluation_benchmark
        self.headless = headless
        self.chrome_cdp_url = chrome_cdp_url
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        if site_name is None and start_url:
            hostname = urlparse(start_url).hostname or ""
            if hostname.startswith("www."):
                hostname = hostname[4:]
            self.site_name = hostname.replace(".", "_") if hostname else None
        else:
            self.site_name = site_name

        # Browser handles — populated by start()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

        # curl_cffi session for MCP HTTP replay
        self.http_session: cffi_requests.Session = cffi_requests.Session(impersonate="chrome124")

        # Site configuration
        self._site_profile: dict | None = None
        self._credentials: dict | None = None
        self._noise_domains: set[str] = set()

        # Element discovery state (stable IDs across same-page scans)
        self._previous_elements: list[InteractiveElement] = []
        self._previous_url: str = ""

        # Screenshot history
        self._screenshot_history: list[Screenshot] = []

        # Network traffic capture buffer
        self._captured_traffic: list[CapturedRequest] = []

        # Decision trace (optional — modules work without it)
        self._trace = trace

    # ===================================================================
    # Trace Helper
    # ===================================================================

    def _log(
        self,
        event_type: str,
        summary: str,
        **kwargs,
    ) -> str | None:
        """Log to trace if available. No-ops when trace is None."""
        if self._trace is None:
            return None
        return self._trace.log("session_manager", event_type, summary, **kwargs)

    # ===================================================================
    # Lifecycle
    # ===================================================================

    async def start(self) -> None:
        """Initialise browser connection, traffic capture, navigate to start_url."""
        # 1. Load site config
        self._noise_domains = self._load_noise_domains()
        self._load_site_config()

        # 2. Connect to Chrome via CDP
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.connect_over_cdp(self.chrome_cdp_url)
        self._context = self._browser.contexts[0]
        self.page = (
            self._context.pages[0]
            if self._context.pages
            else await self._context.new_page()
        )

        # 3. Viewport
        await self.page.set_viewport_size({
            "width": self.viewport_width,
            "height": self.viewport_height,
        })

        # 4. Anti-detection — patch CDP/automation fingerprints
        await self.page.add_init_script("""(() => {
            // 1. navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {get: () => false});

            // 2. Remove CDP-injected cdc_ variables on window/document
            //    Chrome DevTools protocol injects cdc_adoQpoasnfa76pfcZLmcfl_* properties.
            const cleanCDC = (obj) => {
                for (const key of Object.keys(obj)) {
                    if (key.startsWith('cdc_') || key.startsWith('$cdc_')) {
                        delete obj[key];
                    }
                }
            };
            cleanCDC(window);
            cleanCDC(document);

            // 3. chrome.runtime — must exist but .connect() etc should look normal
            if (!window.chrome) window.chrome = {};
            if (!window.chrome.runtime) {
                window.chrome.runtime = {
                    connect: function() {},
                    sendMessage: function() {},
                    id: undefined,
                };
            }

            // 4. Permissions API — hide "denied" for notifications (bot signal)
            const origQuery = window.Permissions?.prototype?.query;
            if (origQuery) {
                window.Permissions.prototype.query = function(params) {
                    if (params.name === 'notifications') {
                        return Promise.resolve({state: 'prompt', onchange: null});
                    }
                    return origQuery.call(this, params);
                };
            }

            // 5. navigator.plugins — must be non-empty (headless/CDP can have empty)
            if (navigator.plugins.length === 0) {
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer',
                         description: 'Portable Document Format', length: 1},
                        {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                         description: '', length: 1},
                        {name: 'Native Client', filename: 'internal-nacl-plugin',
                         description: '', length: 2},
                    ],
                });
            }

            // 6. navigator.languages — ensure populated
            if (!navigator.languages || navigator.languages.length === 0) {
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            }

            // 7. Prevent iframe contentWindow detection of automation
            const origHTMLIFrameElement = HTMLIFrameElement.prototype.__lookupGetter__('contentWindow');
            if (origHTMLIFrameElement) {
                Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
                    get: function() {
                        const w = origHTMLIFrameElement.call(this);
                        if (w) {
                            try {
                                Object.defineProperty(w.navigator, 'webdriver', {get: () => false});
                            } catch(e) {}
                        }
                        return w;
                    }
                });
            }
        })()""")

        # 5. Traffic capture
        await self._setup_traffic_capture(self.page)
        # Also capture on any future pages opened in the context
        self._context.on("page", lambda p: asyncio.ensure_future(self._setup_traffic_capture(p)))

        # 6. Navigate to start URL
        try:
            await self.page.goto(self.start_url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as exc:
            logger.warning("Initial navigation may have timed out: %s", exc)

        # 7. Wait for page readiness
        await self.wait_for_page_ready()

        # 8. Dismiss blocking popups (cookie banners, age gates, surveys)
        await self.dismiss_popups()

        # 9. Sync cookies to curl_cffi
        await self.sync_cookies_to_http_session()

        logger.info("SessionManager started — %s", self.start_url)
        self._start_time = time.time()
        self._log("session_started", f"Session started: {self.start_url}", detail={
            "start_url": self.start_url,
            "site_name": self.site_name,
            "evaluation_mode": self.evaluation_mode,
            "headless": self.headless,
            "viewport": f"{self.viewport_width}x{self.viewport_height}",
            "noise_domains_count": len(self._noise_domains),
            "has_site_profile": self._site_profile is not None,
            "has_credentials": self._credentials is not None,
        }, outcome="success")

    async def close(self) -> None:
        """Clean shutdown. Does NOT kill the Chrome process (it persists)."""
        try:
            self.http_session.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        self._browser = None
        self._context = None
        self.page = None
        logger.info("SessionManager closed")
        self._log("session_closed", "Session closed", detail={
            "duration_s": round(time.time() - getattr(self, "_start_time", time.time()), 2),
            "screenshots_taken": len(self._screenshot_history),
            "traffic_summary": self.get_traffic_summary(),
        }, outcome="success")

    # ===================================================================
    # Page Reattachment (TargetClosedError recovery)
    # ===================================================================

    async def reattach_page(self) -> bool:
        """Reattach to an active page after TargetClosedError.

        When an action opens a new tab or causes the current page to close,
        self.page becomes stale.  This method picks the best surviving page
        from the browser context, re-sets self.page, re-injects the
        anti-detection script, and re-registers traffic capture.

        Returns True if reattachment succeeded, False if no usable page exists.
        """
        if self._context is None:
            return False

        pages = self._context.pages
        if not pages:
            return False

        # Prefer the page whose URL matches the site we're automating.
        # Fall back to the last (most recently opened) page.
        from urllib.parse import urlparse
        target_host = urlparse(self.start_url).netloc
        best: Page | None = None
        for p in reversed(pages):
            try:
                if p.is_closed():
                    continue
            except Exception:
                continue
            if best is None:
                best = p
            try:
                if target_host and target_host in p.url:
                    best = p
                    break
            except Exception:
                pass

        if best is None:
            return False

        old_url = "unknown"
        try:
            old_url = self.page.url if self.page else "none"
        except Exception:
            pass

        self.page = best
        # Re-inject anti-detection (init_script runs on next navigation)
        try:
            await self.page.evaluate(
                "Object.defineProperty(navigator, 'webdriver', {get: () => false})"
            )
        except Exception:
            pass
        # Re-register traffic capture
        try:
            await self._setup_traffic_capture(self.page)
        except Exception:
            pass

        logger.info("Reattached page: %s → %s (%d pages in context)",
                     old_url[:60], self.page.url[:60], len(pages))
        self._log("page_reattached", f"Reattached: {self.page.url}", detail={
            "old_url": old_url,
            "new_url": self.page.url,
            "context_pages": len(pages),
        })

        # Close stale extra tabs (keep only the reattached page)
        for p in pages:
            if p is not self.page:
                try:
                    if not p.is_closed():
                        await p.close()
                except Exception:
                    pass

        return True

    # ===================================================================
    # Site Configuration
    # ===================================================================

    def _load_noise_domains(self) -> set[str]:
        """Load noise domains from ./sites/noise_domains.txt, fall back to defaults."""
        noise_file = SITES_DIR / "noise_domains.txt"
        if noise_file.exists():
            domains: set[str] = set()
            for line in noise_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    domains.add(line)
            return domains
        return {
            "google-analytics.com", "googletagmanager.com",
            "segment.io", "segment.com",
            "mixpanel.com", "sentry.io", "hotjar.com",
            "doubleclick.net", "googlesyndication.com",
            "facebook.net", "fbcdn.net",
            "newrelic.com", "nr-data.net",
            "cloudflareinsights.com",
        }

    def _load_site_config(self) -> None:
        """Load profile.json and credentials.json for the configured site_name."""
        if not self.site_name:
            return
        site_dir = SITES_DIR / self.site_name
        if not site_dir.is_dir():
            site_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created new site directory: %s", site_dir)
        # Profile
        profile_path = site_dir / "profile.json"
        if profile_path.exists():
            try:
                self._site_profile = json.loads(profile_path.read_text())
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse profile.json for %s: %s", self.site_name, exc)
        # Credentials
        creds_path = site_dir / "credentials.json"
        if creds_path.exists():
            try:
                self._credentials = json.loads(creds_path.read_text())
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse credentials.json for %s: %s", self.site_name, exc)
        self._log("site_config_loaded", f"Site config loaded: {self.site_name}", detail={
            "site_name": self.site_name,
            "has_profile": self._site_profile is not None,
            "has_credentials": self._credentials is not None,
            "profile_keys": list(self._site_profile.keys()) if self._site_profile else [],
        }, outcome="success")

    def get_credentials(self) -> dict | None:
        return self._credentials

    def get_site_profile(self) -> dict | None:
        return self._site_profile

    # ===================================================================
    # Traffic Capture
    # ===================================================================

    def _is_noise_url(self, url: str) -> bool:
        """Check URL against the noise domain blocklist."""
        hostname = urlparse(url).hostname or ""
        return any(
            hostname == d or hostname.endswith(f".{d}")
            for d in self._noise_domains
        )

    async def _setup_traffic_capture(self, page: Page) -> None:
        """Attach response listener for real-time API traffic capture."""

        async def _on_response(response: Response) -> None:
            request = response.request
            if request.resource_type not in ("xhr", "fetch"):
                return
            if self._is_noise_url(request.url):
                return
            try:
                request_headers = await request.all_headers()
                response_headers = await response.all_headers()
                request_body = request.post_data

                response_body: str | None = None
                resp_ct = response_headers.get("content-type", "")
                if any(t in resp_ct for t in ("json", "graphql", "xml", "text/plain", "form-urlencoded")):
                    try:
                        response_body = await response.text()
                    except Exception:
                        pass  # Page navigated away

                captured = CapturedRequest(
                    url=request.url,
                    method=request.method,
                    request_headers=request_headers,
                    response_headers=response_headers,
                    request_body=request_body,
                    request_body_parsed=None,
                    response_body=response_body,
                    response_body_parsed=None,
                    status_code=response.status,
                    resource_type=request.resource_type,
                    timestamp=time.time(),
                    request_content_type=request_headers.get("content-type", ""),
                    response_content_type=resp_ct,
                )
                captured.classify_request()
                self._captured_traffic.append(captured)
                self._log("traffic_captured", f"{captured.method} {urlparse(request.url).path} → {response.status}", detail={
                    "url": request.url,
                    "method": captured.method,
                    "status_code": response.status,
                    "protocol": captured.protocol,
                    "endpoint_identity": captured.endpoint_identity,
                    "is_state_changing": captured.is_state_changing,
                })
            except Exception:
                pass  # Never break the browsing session

        page.on("response", _on_response)

    # --- Traffic access -------------------------------------------------

    def get_captured_traffic(self, since_timestamp: float = 0) -> list[CapturedRequest]:
        if since_timestamp <= 0:
            return list(self._captured_traffic)
        return [r for r in self._captured_traffic if r.timestamp >= since_timestamp]

    def get_traffic_for_endpoint(self, endpoint_identity: str) -> list[CapturedRequest]:
        return [r for r in self._captured_traffic if r.endpoint_identity == endpoint_identity]

    def clear_traffic(self) -> None:
        self._captured_traffic.clear()

    def get_traffic_summary(self) -> dict:
        from collections import Counter
        protocols = Counter(r.protocol for r in self._captured_traffic)
        statuses = Counter(r.status_code for r in self._captured_traffic)
        endpoints = {r.endpoint_identity for r in self._captured_traffic if r.endpoint_identity}
        return {
            "total_requests": len(self._captured_traffic),
            "by_protocol": dict(protocols),
            "by_status_code": dict(statuses),
            "unique_endpoints": sorted(endpoints),
        }

    # ===================================================================
    # Page Readiness
    # ===================================================================

    async def wait_for_page_ready(self, timeout_ms: int = 10_000) -> bool:
        """Wait for network idle + DOM mutation settling."""
        assert self.page is not None, "SessionManager not started"
        t0 = time.time()
        network_ok = True
        dom_ok = True

        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            network_ok = False

        # MutationObserver-based DOM settling
        try:
            await self.page.evaluate("""() => {
                return new Promise((resolve) => {
                    let timer;
                    const target = document.body || document.documentElement;
                    if (!target) { resolve(true); return; }
                    const observer = new MutationObserver(() => {
                        clearTimeout(timer);
                        timer = setTimeout(() => { observer.disconnect(); resolve(true); }, 300);
                    });
                    observer.observe(target, {childList: true, subtree: true, attributes: true});
                    timer = setTimeout(() => { observer.disconnect(); resolve(true); }, 500);
                });
            }""")
        except Exception:
            dom_ok = False

        # Check for error/stuck pages and auto-reload
        try:
            page_url = self.page.url
            page_title = await self.page.title()
            is_error_page = (
                "chrome-error://" in page_url
                or page_title in (
                    "This site can't be reached",
                    "This page isn't working",
                    "No internet",
                )
                or "ERR_" in page_title
            )
            if not is_error_page:
                # Also check page content for reload prompts
                body_text = await self.page.evaluate(
                    "(document.body && document.body.innerText || '').slice(0, 500)"
                )
                is_error_page = any(
                    marker in body_text
                    for marker in ("ERR_CONNECTION", "ERR_TIMED_OUT", "ERR_NAME",
                                   "This site can't be reached", "Press reload",
                                   "Try again", "took too long to respond")
                    if marker in body_text
                )
            if is_error_page:
                logger.warning("Error page detected (%s), reloading...", page_title[:40])
                self._log("error_page_reload", f"Reloading error page: {page_title[:40]}", detail={
                    "url": page_url,
                    "title": page_title,
                })
                await self.page.reload(wait_until="domcontentloaded", timeout=30_000)
                await self.page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception as exc:
            logger.debug("Error page check failed (non-fatal): %s", exc)

        self._log("page_ready_check", f"Page ready: network={'ok' if network_ok else 'timeout'}, DOM={'settled' if dom_ok else 'timeout'}", detail={
            "url": self.page.url,
            "network_idle": network_ok,
            "dom_settled": dom_ok,
            "timeout_ms": timeout_ms,
        }, duration_ms=round((time.time() - t0) * 1000, 2))
        return True

    # ===================================================================
    # Popup / Modal Dismissal
    # ===================================================================

    # Button texts that dismiss popups, ordered by preference.
    # Checked case-insensitively via exact text match on the Playwright locator.
    _DISMISS_TEXTS = (
        "Accept All", "Accept all cookies", "Accept Cookies", "Accept",
        "Allow All", "Allow all cookies", "Allow",
        "Got it", "I understand", "OK", "Okay",
        "Continue", "Proceed",
        "No thanks", "No, thanks", "Not now", "Maybe later",
        "No", "Decline", "Reject", "Reject All",
        "Close", "Dismiss", "Skip",
        "I agree", "Agree", "Agree & Continue",
        "Confirm",
    )

    async def dismiss_popups(self, max_rounds: int = 3) -> int:
        """Dismiss blocking popups/modals (cookie banners, age gates, surveys).

        Scans the AXTree for dialog/alertdialog/modal nodes. For each,
        finds visible buttons matching common dismiss text patterns and
        clicks them via Playwright. Repeats up to max_rounds to handle
        stacked popups (e.g. LEGO: age gate → survey → cookie banner).

        Returns the number of popups dismissed.
        """
        assert self.page is not None
        dismissed = 0

        for _round in range(max_rounds):
            # Get current AXTree to check for dialogs
            axtree = await self.get_raw_accessibility_tree()
            if not axtree:
                break

            # Find dialog nodes in the AXTree
            dialog_nodes = self._find_dialog_nodes(axtree)
            if not dialog_nodes:
                break

            # Try to click a dismiss button
            clicked = False
            for text in self._DISMISS_TEXTS:
                try:
                    btn = self.page.get_by_role("button", name=text, exact=False)
                    if await btn.count() > 0 and await btn.first.is_visible():
                        await btn.first.click(timeout=3_000)
                        clicked = True
                        dismissed += 1
                        logger.info("Popup dismissed (round %d): clicked button '%s'", _round + 1, text)
                        self._log("popup_dismissed", f"Auto-dismissed popup: '{text}'", detail={
                            "round": _round + 1,
                            "button_text": text,
                            "dialog_count": len(dialog_nodes),
                        })
                        await self.wait_for_page_ready(timeout_ms=3_000)
                        break
                except Exception:
                    continue

            if not clicked:
                # Try links as well (some dismiss buttons are <a> tags)
                for text in self._DISMISS_TEXTS:
                    try:
                        link = self.page.get_by_role("link", name=text, exact=False)
                        if await link.count() > 0 and await link.first.is_visible():
                            await link.first.click(timeout=3_000)
                            clicked = True
                            dismissed += 1
                            logger.info("Popup dismissed (round %d): clicked link '%s'", _round + 1, text)
                            self._log("popup_dismissed", f"Auto-dismissed popup: '{text}'", detail={
                                "round": _round + 1,
                                "link_text": text,
                                "dialog_count": len(dialog_nodes),
                            })
                            await self.wait_for_page_ready(timeout_ms=3_000)
                            break
                    except Exception:
                        continue

            if not clicked:
                # No dismiss button found — stop trying
                logger.debug("Popup detected but no dismiss button matched (round %d)", _round + 1)
                break

        if dismissed:
            logger.info("Dismissed %d popup(s) total", dismissed)
        return dismissed

    @staticmethod
    def _find_dialog_nodes(node: dict, depth: int = 0) -> list[dict]:
        """Recursively find dialog/alertdialog/modal nodes in AXTree."""
        results: list[dict] = []
        role = (node.get("role") or "").lower()
        if role in ("dialog", "alertdialog"):
            results.append(node)
        # Also detect modal-like patterns: nodes with "modal" in name
        name = (node.get("name") or "").lower()
        if "modal" in name or "popup" in name or "overlay" in name:
            results.append(node)
        if depth < 10:  # Don't recurse too deep
            for child in node.get("children", []):
                results.extend(SessionManager._find_dialog_nodes(child, depth + 1))
        return results

    # ===================================================================
    # Screenshots
    # ===================================================================

    async def take_screenshot(self) -> Screenshot:
        """Capture full-page JPEG screenshot. Appends to history.

        SoM annotation is owned by computer_use.py, not session_manager.
        """
        assert self.page is not None, "SessionManager not started"

        raw = await self.page.screenshot(full_page=True, type="jpeg", quality=85)
        dimensions = await self.page.evaluate("""() => ({
            viewportHeight: window.innerHeight,
            viewportWidth: window.innerWidth,
            fullPageHeight: document.documentElement.scrollHeight,
        })""")

        screenshot = Screenshot(
            image_base64=base64.b64encode(raw).decode(),
            url=self.page.url,
            timestamp=time.time(),
            viewport_height=dimensions["viewportHeight"],
            viewport_width=dimensions["viewportWidth"],
            full_page_height=dimensions["fullPageHeight"],
        )
        self._screenshot_history.append(screenshot)
        self._log("screenshot_taken", f"Screenshot: {self.page.url}", detail={
            "url": self.page.url,
            "viewport": f"{dimensions['viewportWidth']}x{dimensions['viewportHeight']}",
            "full_page_height": dimensions["fullPageHeight"],
            "history_size": len(self._screenshot_history),
        })
        return screenshot

    def get_screenshot_history(self, last_n: int | None = None) -> list[Screenshot]:
        if last_n is None:
            return list(self._screenshot_history)
        return self._screenshot_history[-last_n:]

    def clear_screenshot_history(self) -> None:
        self._screenshot_history.clear()

    # ===================================================================
    # Element Discovery (stable IDs across same-page scans)
    # ===================================================================

    async def get_interactive_elements(self, element_limit: int = 200) -> list[InteractiveElement]:
        """Discover all interactable elements with stable IDs across consecutive scans."""
        assert self.page is not None, "SessionManager not started"

        current_url = self.page.url
        raw_elements = await self._enumerate_elements_js()

        if element_limit > 0 and len(raw_elements) > element_limit:
            raw_elements = self._hierarchical_filter(raw_elements, element_limit)

        same_page = self._is_same_page(current_url, self._previous_url)

        if same_page and self._previous_elements:
            # Reuse IDs from previous scan via fingerprint matching
            old_fp_to_id = {el.fingerprint: el.element_id for el in self._previous_elements}
            used_ids: set[int] = set()
            matched: list[InteractiveElement] = []
            unmatched: list[InteractiveElement] = []

            for el in raw_elements:
                old_id = old_fp_to_id.get(el.fingerprint)
                if old_id is not None and old_id not in used_ids:
                    el.element_id = old_id
                    used_ids.add(old_id)
                    matched.append(el)
                else:
                    unmatched.append(el)

            # Assign fresh IDs to new/unmatched elements
            next_id = max(used_ids, default=0) + 1
            for el in unmatched:
                while next_id in used_ids:
                    next_id += 1
                el.element_id = next_id
                used_ids.add(next_id)
                matched.append(el)

            # Sort by document order (top→bottom, left→right) for consistent presentation
            elements = sorted(
                matched,
                key=lambda e: (e.bounding_box.get("y", 0), e.bounding_box.get("x", 0)),
            )
        else:
            # Fresh page or first scan — sequential IDs in document order
            for i, el in enumerate(raw_elements, start=1):
                el.element_id = i
            elements = raw_elements

        # Trace: element discovery stats
        if same_page and self._previous_elements:
            n_matched = len(elements) - len(unmatched)
            n_new = len(unmatched)
        else:
            n_matched = 0
            n_new = len(elements)
        self._log("elements_discovered", f"Found {len(elements)} interactive elements", detail={
            "url": current_url,
            "total_elements": len(elements),
            "same_page": same_page,
            "matched_ids_reused": n_matched,
            "new_ids_assigned": n_new,
        })

        self._previous_elements = elements
        self._previous_url = current_url
        return elements

    async def _enumerate_elements_js(self) -> list[InteractiveElement]:
        """Run the JS enumerator and convert raw dicts to InteractiveElement objects."""
        assert self.page is not None
        raw: list[dict] = await self.page.evaluate(_ENUMERATE_ELEMENTS_JS)
        return [
            InteractiveElement(
                element_id=0,  # Assigned later by get_interactive_elements
                tag=r["tag"],
                role=r["role"],
                name=r.get("name", ""),
                element_type=r.get("element_type"),
                value=r.get("value"),
                bounding_box=r.get("bounding_box", {}),
                is_visible=r.get("is_visible", False),
                attributes=r.get("attributes", {}),
                states=r.get("states", []),
                selector=r.get("selector", ""),
                fingerprint=r.get("fingerprint", ""),
            )
            for r in raw
        ]

    @staticmethod
    def _is_same_page(current_url: str, previous_url: str) -> bool:
        """Compare scheme + netloc + path. Ignore query params and fragments."""
        if not previous_url:
            return False
        curr = urlparse(current_url)
        prev = urlparse(previous_url)
        return (
            curr.scheme == prev.scheme
            and curr.netloc == prev.netloc
            and curr.path == prev.path
        )

    @staticmethod
    def _hierarchical_filter(elements: list[InteractiveElement], target: int = 150) -> list[InteractiveElement]:
        """When element count is too high, keep structural + navigational elements.

        Returns ~50-150 elements regardless of input size by keeping:
        - Headings, nav, forms, search, landmarks
        - First interactive element per section
        - Unique-name actions (cart, checkout, login)
        - Section summaries for collapsed groups
        """
        STRUCTURAL_ROLES = frozenset({
            "heading", "navigation", "banner", "main", "contentinfo",
            "complementary", "search", "form", "tab", "tablist",
            "menu", "menubar", "toolbar", "searchbox",
        })
        STRUCTURAL_TAGS = frozenset({
            "h1", "h2", "h3", "h4", "h5", "h6", "nav", "header",
            "footer", "main", "aside", "form", "summary", "details",
        })
        UNIQUE_KEYWORDS = frozenset({
            "cart", "checkout", "login", "sign in", "register", "search",
            "menu", "account", "profile", "settings", "home", "back",
        })

        structural: list[InteractiveElement] = []
        rest: list[InteractiveElement] = []

        for el in elements:
            role_lower = el.role.lower()
            tag_lower = el.tag.lower()
            name_lower = (el.name or "").lower()

            is_structural = (
                role_lower in STRUCTURAL_ROLES
                or tag_lower in STRUCTURAL_TAGS
                or any(kw in name_lower for kw in UNIQUE_KEYWORDS)
            )
            if is_structural:
                structural.append(el)
            else:
                rest.append(el)

        # Fill remaining budget from non-structural elements.
        # Prioritize visible elements over hidden ones — on element-heavy pages
        # (LEGO: 700+ elements), hidden dropdown menus fill the budget before
        # visible content (filters, products, pagination) gets a chance.
        # Within each group, preserve DOM order for stable presentation.
        remaining_budget = max(0, target - len(structural))
        visible_rest = [el for el in rest if el.is_visible]
        hidden_rest = [el for el in rest if not el.is_visible]
        kept = structural + (visible_rest + hidden_rest)[:remaining_budget]

        # Deduplicate by name
        kept = SessionManager._deduplicate_by_name(kept)

        # Add a summary element noting how many were filtered
        if len(elements) > len(kept):
            kept.append(InteractiveElement(
                element_id=0,
                tag="section_summary",
                role="note",
                name=f"[{len(elements) - len(kept)} more interactive elements on page — scroll or search to explore]",
                element_type=None,
                value=None,
                bounding_box={"x": 0, "y": 0, "width": 0, "height": 0},
                is_visible=False,
                attributes={},
                states=[],
                selector="",
                fingerprint="section_summary",
            ))

        return kept

    @staticmethod
    def _deduplicate_by_name(elements: list[InteractiveElement]) -> list[InteractiveElement]:
        """Keep first element for each name, skip duplicates (e.g. 15 'ADD' buttons)."""
        seen_names: dict[str, int] = {}
        result: list[InteractiveElement] = []

        for el in elements:
            name = (el.name or "").strip().lower()
            if not name or len(name) > 40:
                # Long names are likely unique content, keep them
                result.append(el)
                continue

            count = seen_names.get(name, 0)
            if count < 2:
                # Keep first two of each name
                result.append(el)
                seen_names[name] = count + 1
            else:
                seen_names[name] = count + 1

        return result

    # ===================================================================
    # DOM Tree Extraction
    # ===================================================================

    async def get_dom_tree(self, max_length: int = 200_000) -> str:
        """Get cleaned DOM via page.content() + Python-side stripping.

        ~150ms for even multi-MB pages (vs 26+ seconds for the old JS walker).
        """
        assert self.page is not None
        try:
            raw_html = await self.page.content()
        except Exception as exc:
            logger.warning("page.content() failed: %s", exc)
            return ""

        cleaned = _STRIP_TAGS_RE.sub('', raw_html)
        cleaned = _STRIP_INLINE_STYLE_RE.sub('', cleaned)
        cleaned = _STRIP_CLASS_RE.sub('', cleaned)
        cleaned = _COLLAPSE_WHITESPACE_RE.sub('\n', cleaned)

        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "\n<!-- DOM truncated -->"
            self._log("dom_truncated", f"DOM truncated from {len(raw_html)} to {max_length} chars")

        return cleaned.strip()

    # ===================================================================
    # Accessibility Tree Extraction
    # ===================================================================

    async def get_raw_accessibility_tree(self) -> dict | None:
        """Return the accessibility tree as a nested dict via CDP.

        Uses Accessibility.getFullAXTree (CDP) instead of the deprecated
        page.accessibility.snapshot(). The CDP response is a flat node array;
        we convert it to the nested {role, name, children, ...} format that
        every consumer module expects.
        """
        assert self.page is not None
        try:
            cdp = await self._context.new_cdp_session(self.page)
            result = await cdp.send("Accessibility.getFullAXTree")
            await cdp.detach()
            return self._cdp_axtree_to_nested(result)
        except Exception as exc:
            logger.warning("AXTree extraction failed: %s", exc)
            return None

    @staticmethod
    def _cdp_axtree_to_nested(cdp_result: dict) -> dict | None:
        """Convert CDP flat node array to nested tree.

        CDP returns {"nodes": [{nodeId, role, name, value, properties, childIds, ignored}, ...]}.
        Consumers expect nested {role, name, value, level, checked, ..., children: [...]}.

        Ignored nodes are transparent — their children are promoted to the
        parent level. This handles React/SPA wrapper divs that CDP marks as
        ignored but whose children contain all the actual page content.
        """
        nodes = cdp_result.get("nodes", [])
        if not nodes:
            return None

        by_id: dict[str, dict] = {}
        for node in nodes:
            nid = node.get("nodeId")
            if nid:
                by_id[nid] = node

        def _val(obj: dict | None) -> Any:
            if obj is None:
                return None
            return obj.get("value")

        def _build_children(node: dict) -> list[dict]:
            """Recursively build children, promoting ignored nodes' children."""
            children: list[dict] = []
            for child_id in node.get("childIds", []):
                child_node = by_id.get(child_id)
                if child_node:
                    children.extend(_build_nodes(child_node))
            return children

        def _build_nodes(node: dict) -> list[dict]:
            """Build nested representation. Returns a LIST — ignored nodes
            promote their children to the parent level."""
            if node.get("ignored", False):
                return _build_children(node)

            role = _val(node.get("role")) or "none"
            name = _val(node.get("name")) or ""
            value = _val(node.get("value"))

            result: dict[str, Any] = {"role": role, "name": name}
            if value is not None:
                result["value"] = value

            for prop in node.get("properties", []):
                pname = prop.get("name", "")
                pval = _val(prop.get("value"))
                if pname and pval is not None:
                    result[pname] = pval

            children = _build_children(node)
            if children:
                result["children"] = children

            return [result]

        root = nodes[0]
        built = _build_nodes(root)
        return built[0] if built else None

    # ===================================================================
    # Meta Token Extraction
    # ===================================================================

    async def extract_meta_tokens(self) -> dict:
        """Extract CSRF tokens, form keys, auth tokens with source annotations."""
        assert self.page is not None
        tokens: dict = {}

        # JS-based extraction (meta tags, hidden fields, JS vars, storage)
        try:
            js_tokens = await self.page.evaluate(_META_TOKENS_JS)
            tokens.update(js_tokens)
        except Exception as exc:
            logger.warning("JS meta token extraction failed: %s", exc)

        # Cookies (double-submit CSRF pattern)
        try:
            cookies = await self._context.cookies() if self._context else []
            csrf_cookie_names = {"xsrf-token", "csrf_token", "_csrf", "csrftoken", "x-csrf-token"}
            for cookie in cookies:
                if cookie["name"].lower() in csrf_cookie_names:
                    tokens[f"cookie_{cookie['name']}"] = {
                        "value": cookie["value"],
                        "source": "cookie",
                        "key": cookie["name"],
                    }
        except Exception:
            pass

        # Trace: log token names and sources (NOT values — security)
        self._log("meta_tokens_extracted", f"Extracted {len(tokens)} tokens", detail={
            "token_sources": {
                name: info.get("source", "unknown") if isinstance(info, dict) else "raw"
                for name, info in tokens.items()
            },
        })
        return tokens

    async def get_cookies(self) -> list[dict]:
        """Get all cookies for the current context."""
        return await self._context.cookies() if self._context else []

    async def get_storage(self) -> dict:
        """Get localStorage and sessionStorage."""
        result = {"local_storage": {}, "session_storage": {}}
        try:
            result["local_storage"] = await self.page.evaluate(
                "() => { const o = {}; for (let i = 0; i < localStorage.length; i++) "
                "{ const k = localStorage.key(i); o[k] = localStorage.getItem(k); } return o; }"
            )
        except Exception:
            pass
        try:
            result["session_storage"] = await self.page.evaluate(
                "() => { const o = {}; for (let i = 0; i < sessionStorage.length; i++) "
                "{ const k = sessionStorage.key(i); o[k] = sessionStorage.getItem(k); } return o; }"
            )
        except Exception:
            pass
        return result

    # ===================================================================
    # Action Execution
    # ===================================================================

    def _resolve_element(self, element_id: int) -> InteractiveElement | None:
        """Find an InteractiveElement by its SoM ID."""
        for el in self._previous_elements:
            if el.element_id == element_id:
                return el
        return None

    async def execute_action(self, action: dict) -> ActionResult:
        """Execute a structured action from the CU agent.

        action dict keys:
            action_type: click | type | select | scroll | press_key | navigate | hover | go_back | wait
            element_id:  SoM ID (for click, type, select, hover)
            text:        text to type / URL to navigate / key name to press
            value:       option value for select
            direction:   "up" | "down" for scroll
            scroll_amount: int (wheel clicks, default 3)
            clear_first: bool (clear field before typing, default True)
        """
        assert self.page is not None, "SessionManager not started"
        action_type = action.get("action_type", "")
        url_before = self.page.url
        t0 = time.time()

        try:
            match action_type:
                case "click":
                    result = await self._action_click(action)
                case "type":
                    result = await self._action_type(action)
                case "select":
                    result = await self._action_select(action)
                case "scroll":
                    result = await self._action_scroll(action)
                case "press_key":
                    result = await self._action_press_key(action)
                case "navigate":
                    result = await self._action_navigate(action)
                case "hover":
                    result = await self._action_hover(action)
                case "go_back":
                    result = await self._action_go_back()
                case "wait":
                    result = await self._action_wait(action)
                case "note":
                    # No-op action — CU agent records reasoning without browser interaction.
                    # The text is captured in the trace log below; nothing to execute.
                    result = ActionResult(success=True)
                case _:
                    result = ActionResult(success=False, error=f"Unknown action_type: {action_type}")
        except Exception as exc:
            result = ActionResult(success=False, error=str(exc))

        # Detect navigation
        url_after = self.page.url
        if not self._is_same_page(url_before, url_after):
            result.navigation_occurred = True
            result.new_url = url_after
            result.elements_stale = True
            self._previous_elements = []  # Reset stable IDs for new page

        # Page readiness after potentially-navigating actions
        if action_type in ("click", "navigate", "go_back", "press_key", "select"):
            try:
                await self.wait_for_page_ready(timeout_ms=5_000)
            except Exception:
                result.page_ready = False

        # Trace: log every action with full context
        el = self._resolve_element(action.get("element_id", -1))
        self._log(
            "action_executed",
            f"{action_type}: {el.name if el else action.get('text', '')[:50]}",
            detail={
                "action": action,
                "element": {"id": el.element_id, "name": el.name, "selector": el.selector, "role": el.role} if el else None,
                "url_before": url_before,
                "url_after": self.page.url,
                "navigation_occurred": result.navigation_occurred,
                "page_ready": result.page_ready,
            },
            outcome="success" if result.success else "failure",
            error=result.error,
            duration_ms=round((time.time() - t0) * 1000, 2),
        )
        return result

    # --- Individual action implementations ------------------------------

    async def _action_click(self, action: dict) -> ActionResult:
        element_id = action.get("element_id")
        if element_id is None:
            return ActionResult(success=False, error="click requires element_id")
        el = self._resolve_element(element_id)
        if not el:
            return ActionResult(success=False, error=f"Element {element_id} not found")

        try:
            await self.page.locator(el.selector).first.click(timeout=5_000)
            return ActionResult(success=True)
        except Exception:
            pass
        # Fallback: force click (handles overlays)
        try:
            await self.page.locator(el.selector).first.click(force=True, timeout=5_000)
            self._log("fallback_used", f"Force click on [{element_id}] {el.name}", detail={
                "element_id": element_id, "approach": "force_click",
                "reason": "Normal click failed, likely covered by overlay",
            })
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=f"Click failed: {exc}")

    async def _action_type(self, action: dict) -> ActionResult:
        """Type text into an element. Fallback chain: click+type → fill → contenteditable.

        Prefers character-by-character typing (keyboard.type) over fill() because:
        - keyboard.type fires keydown/keypress/input/keyup per character
        - This triggers autocomplete APIs, React synthetic events, debounced handlers
        - fill() only fires input+change once, which many autocomplete/controlled
          components ignore (Google Places, Swiggy location, search suggestions)

        CRITICAL: Does NOT press Enter after typing. Enter is a separate press_key action.
        """
        element_id = action.get("element_id")
        text = action.get("text", "")
        clear_first = action.get("clear_first", True)

        if element_id is None:
            return ActionResult(success=False, error="type requires element_id")
        el = self._resolve_element(element_id)
        if not el:
            return ActionResult(success=False, error=f"Element {element_id} not found")

        locator = self.page.locator(el.selector).first

        if clear_first:
            # Approach 1: click → select all → type character by character
            # Fires per-character keydown/keyup events — works with autocomplete,
            # React controlled inputs, and any framework's event system.
            try:
                await locator.click(timeout=3_000)
                # Triple-click selects all text in the field
                await locator.click(click_count=3, timeout=1_000)
                await self.page.keyboard.type(text, delay=50)
                return ActionResult(success=True)
            except Exception:
                pass

            # Approach 2: fill() — atomic clear + set. Works for simple form inputs
            # where Approach 1 fails (e.g. elements that can't be clicked).
            try:
                await locator.fill(text, timeout=3_000)
                self._log("fallback_used", f"fill() on [{element_id}]", detail={
                    "element_id": element_id, "approach": "fill",
                    "reason": "click+type failed, using atomic fill()",
                })
                return ActionResult(success=True)
            except Exception:
                pass

            # Approach 3: contenteditable — set via JS
            try:
                await locator.evaluate(
                    "(node, text) => { node.textContent = text; "
                    "node.dispatchEvent(new Event('input', {bubbles: true})); }",
                    text,
                )
                self._log("fallback_used", f"JS contenteditable on [{element_id}]", detail={
                    "element_id": element_id, "approach": "contenteditable_js",
                    "reason": "click+type and fill() both failed, using JS textContent",
                })
                return ActionResult(success=True)
            except Exception as exc:
                return ActionResult(success=False, error=f"All type approaches failed: {exc}")
        else:
            # Append mode: click to focus, then type at cursor position
            try:
                await locator.click(timeout=3_000)
                await self.page.keyboard.type(text, delay=50)
                return ActionResult(success=True)
            except Exception as exc:
                return ActionResult(success=False, error=f"Append type failed: {exc}")

    async def _action_select(self, action: dict) -> ActionResult:
        element_id = action.get("element_id")
        value = action.get("value", "")

        if element_id is None:
            return ActionResult(success=False, error="select requires element_id")
        el = self._resolve_element(element_id)
        if not el:
            return ActionResult(success=False, error=f"Element {element_id} not found")

        locator = self.page.locator(el.selector).first

        # Try by value first, then by label
        try:
            await locator.select_option(value=value, timeout=3_000)
            return ActionResult(success=True)
        except Exception:
            pass
        try:
            await locator.select_option(label=value, timeout=3_000)
            self._log("fallback_used", f"Select by label on [{element_id}]", detail={
                "element_id": element_id, "approach": "select_by_label",
                "reason": "select_option(value=...) failed, matched by label text instead",
            })
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=f"Select failed: {exc}")

    async def _action_scroll(self, action: dict) -> ActionResult:
        direction = action.get("direction", "down")
        amount = action.get("scroll_amount", 3)
        delta_y = 120 * amount * (1 if direction == "down" else -1)
        try:
            await self.page.mouse.wheel(0, delta_y)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=f"Scroll failed: {exc}")

    async def _action_press_key(self, action: dict) -> ActionResult:
        key = action.get("text", "")
        if not key:
            return ActionResult(success=False, error="press_key requires text (key name)")
        try:
            await self.page.keyboard.press(key)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=f"Key press failed: {exc}")

    async def _action_navigate(self, action: dict) -> ActionResult:
        url = action.get("text", "")
        if not url:
            return ActionResult(success=False, error="navigate requires text (URL)")
        try:
            response = await self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            status = response.status if response else None
            return ActionResult(success=True, status_code=status)
        except Exception as exc:
            return ActionResult(success=False, error=f"Navigation failed: {exc}")

    async def _action_hover(self, action: dict) -> ActionResult:
        element_id = action.get("element_id")
        if element_id is None:
            return ActionResult(success=False, error="hover requires element_id")
        el = self._resolve_element(element_id)
        if not el:
            return ActionResult(success=False, error=f"Element {element_id} not found")
        try:
            await self.page.locator(el.selector).first.hover(timeout=5_000)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=f"Hover failed: {exc}")

    async def _action_go_back(self) -> ActionResult:
        try:
            await self.page.go_back(wait_until="domcontentloaded", timeout=10_000)
            return ActionResult(success=True)
        except Exception as exc:
            return ActionResult(success=False, error=f"Go back failed: {exc}")

    async def _action_wait(self, action: dict) -> ActionResult:
        seconds = action.get("scroll_amount", 1)  # Reuse scroll_amount field or default 1s
        if isinstance(action.get("text"), (int, float)):
            seconds = action["text"]
        await asyncio.sleep(max(0.1, min(seconds, 10)))  # Bounded: 100ms to 10s
        return ActionResult(success=True)

    # ===================================================================
    # Cookie / Session Management
    # ===================================================================

    async def sync_cookies_to_http_session(self) -> None:
        """Copy browser cookies to the curl_cffi session for MCP API replay."""
        if not self._context:
            return
        cookies = await self._context.cookies()
        for cookie in cookies:
            self.http_session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain", ""),
                path=cookie.get("path", "/"),
            )

    def get_http_session(self) -> cffi_requests.Session:
        return self.http_session

    # ===================================================================
    # CDP Utilities
    # ===================================================================

    async def get_element_event_listeners(self, selector: str) -> list[dict]:
        """Get JS event listeners on an element via CDP DOMDebugger. Expensive — on-demand only."""
        assert self.page is not None
        try:
            cdp = await self._context.new_cdp_session(self.page)
            # Resolve element to a remote object
            result = await cdp.send("Runtime.evaluate", {
                "expression": f"document.querySelector('{selector}')",
                "returnByValue": False,
            })
            object_id = result.get("result", {}).get("objectId")
            if not object_id:
                return []
            # Get event listeners
            listeners_result = await cdp.send("DOMDebugger.getEventListeners", {
                "objectId": object_id,
            })
            listeners = listeners_result.get("listeners", [])
            return [
                {
                    "type": listener["type"],
                    "once": listener.get("once", False),
                    "passive": listener.get("passive", False),
                    "handler_preview": listener.get("handler", {}).get("description", "")[:200],
                }
                for listener in listeners
            ]
        except Exception as exc:
            logger.warning("CDP event listener discovery failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import shutil
    import subprocess
    import platform

    parser = argparse.ArgumentParser(description="MorphNet — Run a web task")
    parser.add_argument("--url", required=True, help="Start URL")
    parser.add_argument("--task", required=True, help="Natural language task description")
    parser.add_argument("--headless", default="true", choices=["true", "false"],
                        help="Run Chrome in headless mode (default: true)")
    parser.add_argument("--port", type=int, default=9222,
                        help="Chrome remote debugging port (default: 9222)")
    parser.add_argument("--max-subtasks", type=int, default=15,
                        help="Maximum subtasks before stopping (default: 15)")
    parser.add_argument("--site", default=None,
                        help="Site name from ./sites/ for profile and credentials")
    args = parser.parse_args()

    def _find_chrome() -> str:
        """Find Chrome binary on this system."""
        system = platform.system()
        if system == "Darwin":
            path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            if Path(path).exists():
                return path
        elif system == "Linux":
            for name in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
                found = shutil.which(name)
                if found:
                    return found
        elif system == "Windows":
            for path in [
                Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
                Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
                Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
            ]:
                if path.exists():
                    return str(path)
        raise FileNotFoundError(
            "Chrome not found. Install Google Chrome or pass a running CDP endpoint via --port."
        )

    def _launch_chrome(port: int, headless: bool) -> subprocess.Popen:
        """Launch Chrome with remote debugging enabled."""
        chrome_bin = _find_chrome()
        profile_dir = Path.home() / f"chrome-morphnet-{port}"
        cmd = [
            chrome_bin,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
        ]
        if headless:
            cmd.append("--headless=new")
        logger.info("Launching Chrome: %s", " ".join(cmd[:3]) + " ...")
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_for_cdp(port: int, timeout: int = 15) -> None:
        """Poll Chrome's CDP endpoint until it responds."""
        import urllib.request
        url = f"http://localhost:{port}/json/version"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("Chrome CDP ready on port %d", port)
                        return
            except Exception:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"Chrome CDP not responding on port {port} after {timeout}s")

    async def _run() -> None:
        from morphnet.morphnet_orchestrator import MorphNetOrchestrator
        from morphnet.trace import TaskTrace

        trace = TaskTrace(task_prompt=args.task)
        session = SessionManager(
            start_url=args.url,
            task_prompt=args.task,
            headless=args.headless == "true",
            chrome_cdp_url=f"http://localhost:{args.port}",
            site_name=args.site,
            trace=trace,
        )
        await session.start()

        orchestrator = MorphNetOrchestrator(session=session, trace=trace)
        result = await orchestrator.run_task(args.task, max_subtasks=args.max_subtasks)

        print()
        print("=" * 50)
        print(f"Success: {result.success}")
        print(f"Answer:  {result.final_answer}")
        print(f"Subtasks: {result.subtasks_completed} | Actions: {result.total_actions}")
        print(f"Trace:   {trace.output_dir}/trace.jsonl")
        print("=" * 50)

        await session.close()
        trace.close()

    # --- Launch Chrome and run ------------------------------------------------
    is_headless = args.headless == "true"
    print(f"MorphNet — {'headless' if is_headless else 'visible'} mode")
    print(f"Task: {args.task}")
    print(f"URL:  {args.url}")
    print()

    chrome_proc = _launch_chrome(args.port, is_headless)
    try:
        _wait_for_cdp(args.port)
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True)
    finally:
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome_proc.kill()
