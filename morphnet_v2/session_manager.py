"""
session_manager.py — v2

Single I/O boundary between morphnet and the outside world.
Owns Chrome (raw CDP + Playwright), Gemini, curl_cffi, and the Chrome
subprocess. File I/O is exempt — notes 

Building incrementally — see ./draft.md for the full plan.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from uuid import uuid4

from pydantic import BaseModel, ConfigDict

import curl_cffi.requests as cffi_requests
import httpx
import websockets
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from rebrowser_playwright.async_api import (
    Browser,
    Error as PlaywrightError,
    Page,
    Playwright,
    async_playwright,
)

from morphnet_v2 import notes
from morphnet_v2.mutation_types import (
    MutationNodeRef,
    RawMutationRecord,
    records_from_raw,
    summarize_mutations,
)
from morphnet_v2.planner import Orchestrator

load_dotenv()  # populate GEMINI_API_KEY / GOOGLE_API_KEY from .env at import time

# genai client — initialized at module load. API key MUST be in env;
# importing fails fast if not set so we don't ship a broken binary.
_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY / GOOGLE_API_KEY not set. Add it to .env or env "
        "before importing morphnet_v2.session_manager."
    )
_gemini = genai.Client(api_key=_api_key)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Chrome
# ─────────────────────────────────────────────────────────────────

CHROME_PATH: str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CHROME_FLAGS_BASE: list[str] = [
    "--remote-allow-origins=*",          # required since Chrome 111
    "--no-first-run",
    "--no-default-browser-check",
    "--use-gl=angle",
    "--use-angle=default",
    "--disable-features=IsolateOrigins,site-per-process",
]

HEADLESS_UA_OVERRIDE: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

CDP_READY_TIMEOUT_S: float = 15.0
NAVIGATION_TIMEOUT_S: float = 30.0

# Page-lifecycle defaults (chunk 1.4). All overridable per call.
DOM_STABILITY_POLL_MS: int = 200          # how often to probe DOM size
DOM_STABILITY_WINDOW_MS: int = 800         # consecutive unchanged time before "stable"
DOM_STABILITY_MAX_WAIT_MS: int = 5_000     # hard cap before giving up
NAV_WAIT_TIMEOUT_MS: int = 2_000           # event-driven wait_for_url cap
DOMCL_TIMEOUT_MS: int = 3_000              # waitForLoadState("domcontentloaded") cap

# DOM probe (lifted from crawler/browser_tools.py:_collect_stability_probe).
# Two cheap integers tell us whether the DOM is still settling.
_DOM_PROBE_JS: str = """
() => ({
    htmlChars: document.documentElement?.outerHTML?.length ?? 0,
    textChars: document.body?.innerText?.length ?? 0,
})
"""


# Init script — wraps history.pushState / replaceState and listens for
# popstate / hashchange. Records into a buffer on window.__mn_nav_handle.
# Loaded via Page.addScriptToEvaluateOnNewDocument so it runs before page JS;
# requires Page.enable on the same session first.
NAV_CAPTURE_INIT_SCRIPT: str = """
(() => {
    if (window.__mn_nav_handle) return;
    const buffer = [];
    window.__mn_nav_handle = { buffer };
    const wrap = (orig, kind) => function() {
        const result = orig.apply(this, arguments);
        try {
            const u = arguments[2] != null
                ? new URL(arguments[2], location.href).href
                : location.href;
            buffer.push({ ts: Date.now(), kind, url: u });
        } catch (_) {}
        return result;
    };
    history.pushState = wrap(history.pushState.bind(history), 'pushState');
    history.replaceState = wrap(history.replaceState.bind(history), 'replaceState');
    window.addEventListener('popstate', () => {
        buffer.push({ ts: Date.now(), kind: 'popstate', url: location.href });
    });
    window.addEventListener('hashchange', () => {
        buffer.push({ ts: Date.now(), kind: 'hashchange', url: location.href });
    });
})();
"""


# ─────────────────────────────────────────────────────────────────
# Action JS primitives (chunk 1.5)
# Lifted verbatim from crawler/executor.py. All actions resolve elements
# by `[data-cdx-aid="..."]` — the AID stamping is what page_filter (chunk
# 2.1) puts into the live DOM so the agent can address elements stably
# across re-renders. Note: until 2.1 lands, AIDs aren't on the page yet,
# so these methods will return `not_found` against a fresh Chrome — that
# is expected and not a bug.
# ─────────────────────────────────────────────────────────────────

_PREP_JS: str = """(args) => {
    const [aid, opts] = args;
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return { error: 'not_found' };
    if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}))
        return { error: 'hidden' };
    if (opts.checkDisabled && (el.matches(':disabled') || !!el.closest('[inert]')))
        return { error: 'disabled' };

    el.scrollIntoView({ block: 'center', behavior: 'instant' });
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return { error: 'zero_size' };
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;

    if (opts.hitTest) {
        const top = document.elementFromPoint(cx, cy);
        if (top && !el.contains(top) && !top.contains(el)) {
            if (opts.hitTest === 'corners') {
                const w = r.width, h = r.height;
                const ix = Math.max(10, w * 0.2), iy = Math.max(10, h * 0.2);
                const corners = [[ix,iy],[w-ix,iy],[ix,h-iy],[w-ix,h-iy]];
                for (const [px,py] of corners) {
                    const at = document.elementFromPoint(r.left+px, r.top+py);
                    if (at && el.contains(at))
                        return { x: r.left+px, y: r.top+py, w, h, corner: true };
                }
            }
            const topAid = top.getAttribute('data-cdx-aid') || null;
            const topZ = parseInt(getComputedStyle(top).zIndex) || 0;
            const topText = (top.textContent || '').trim().slice(0, 60);
            let overlay = top.parentElement;
            while (overlay && overlay !== document.body) {
                const os = getComputedStyle(overlay);
                if (os.position === 'fixed' || os.position === 'absolute') break;
                overlay = overlay.parentElement;
            }
            if (!overlay || overlay === document.body) overlay = top;
            const overlayAid = overlay.getAttribute('data-cdx-aid') || null;
            const overlayZ = parseInt(getComputedStyle(overlay).zIndex) || 0;
            const btns = [...overlay.querySelectorAll('button')].slice(0, 5).map(b => ({
                aid: b.getAttribute('data-cdx-aid'),
                text: (b.textContent || '').trim().slice(0, 30)
            }));
            return {
                error: 'blocked', blocker_aid: topAid, overlay_aid: overlayAid,
                z: topZ, overlay_z: overlayZ, text: topText, buttons: btns
            };
        }
    }
    return { x: cx, y: cy, w: r.width, h: r.height };
}"""

_FILL_JS: str = """(args) => {
    const [aid, text] = args;
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return { error: 'not_found' };
    const tag = el.tagName.toLowerCase();
    const isInput = (tag === 'input' || tag === 'textarea');
    const isEditable = el.isContentEditable;
    if (!isInput && !isEditable)
        return { error: 'not_editable', tag, isContentEditable: false };
    el.focus();
    if (el.select) el.select();
    else document.getSelection().selectAllChildren(el);
    const ok = document.execCommand('insertText', false, text);
    return ok ? { ok: true } : { error: 'exec_failed', tag };
}"""

_READ_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return { error: 'not_found' };
    return { text: (el.innerText || '').slice(0, 2000) };
}"""

_DISMISS_CHECK_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return 'removed';
    if (!document.body.contains(el)) return 'detached';
    if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}))
        return 'hidden';
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return 'hidden';
    if (r.right < 0 || r.left > window.innerWidth ||
        r.bottom < 0 || r.top > window.innerHeight)
        return 'offscreen';
    return 'still_visible';
}"""

_GET_ANIMATIONS_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return { count: 0 };
    const anims = el.getAnimations({subtree: true});
    const running = anims.filter(a => a.playState === 'running');
    return { count: running.length };
}"""

_AWAIT_ANIMATIONS_JS: str = """async (aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return 'no_element';
    const anims = el.getAnimations({subtree: true});
    if (anims.length === 0) return 'no_animations';
    await Promise.race([
        Promise.all(anims.map(a => a.finished.catch(() => null))),
        new Promise(r => setTimeout(r, 500))
    ]);
    return 'finished';
}"""

_PROBE_BLOCKER_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    if (cx < 0 || cx >= window.innerWidth || cy < 0 || cy >= window.innerHeight)
        return null;
    const top = document.elementFromPoint(cx, cy);
    if (!top || el.contains(top) || top.contains(el)) return null;
    const topAid = top.getAttribute('data-cdx-aid') || null;
    const topZ = parseInt(getComputedStyle(top).zIndex) || 0;
    const topText = (top.textContent || '').trim().slice(0, 60);
    return { topElement: { aid: topAid, z: topZ, text: topText } };
}"""

_DESCRIBE_BLOCKER_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return null;
    const bcr = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const tag = el.tagName.toLowerCase();
    const text = (el.textContent || "").trim().slice(0, 60);
    const w = Math.round(bcr.width);
    const h = Math.round(bcr.height);
    const x = Math.round(bcr.x);
    const y = Math.round(bcr.y);
    const z = parseInt(cs.zIndex, 10) || 0;
    const pos = cs.position;
    const anim = cs.animationName && cs.animationName !== "none" ? "animated"
        : (parseFloat(cs.transitionDuration) > 0 ? "animated" : "");
    return { tag, text, w, h, x, y, z, pos, anim };
}"""


# Drag suite JS — lifted from inline blocks in drag_*/probe_drop_zones methods.

_DRAG_SYNTHETIC_PAIR_JS: str = """(args) => {
    const [srcAid, tgtAid] = args;
    const src = document.querySelector(`[data-cdx-aid="${srcAid}"]`);
    const tgt = document.querySelector(`[data-cdx-aid="${tgtAid}"]`);
    if (!src) return { error: 'source_not_found' };
    if (!tgt) return { error: 'target_not_found' };

    const data = src.textContent.trim();
    const dt = new DataTransfer();
    dt.setData('text/plain', data);

    src.dispatchEvent(new DragEvent('dragstart', { dataTransfer: dt, bubbles: true, cancelable: true }));
    tgt.dispatchEvent(new DragEvent('dragover',  { dataTransfer: dt, bubbles: true, cancelable: true }));
    tgt.dispatchEvent(new DragEvent('drop',      { dataTransfer: dt, bubbles: true, cancelable: true }));
    src.dispatchEvent(new DragEvent('dragend',   { dataTransfer: dt, bubbles: true }));

    return { data, after: tgt.textContent.trim().slice(0, 30) };
}"""

_DRAG_SYNTHETIC_BATCH_JS: str = """(pairs) => {
    const results = [];
    for (const [srcAid, tgtAid] of pairs) {
        const src = document.querySelector(`[data-cdx-aid="${srcAid}"]`);
        const tgt = document.querySelector(`[data-cdx-aid="${tgtAid}"]`);
        if (!src || !tgt) {
            results.push({src: srcAid, tgt: tgtAid, ok: false, err: 'not_found'});
            continue;
        }
        const data = src.textContent.trim();
        const dt = new DataTransfer();
        dt.setData('text/plain', data);

        src.dispatchEvent(new DragEvent('dragstart', { dataTransfer: dt, bubbles: true, cancelable: true }));
        tgt.dispatchEvent(new DragEvent('dragover',  { dataTransfer: dt, bubbles: true, cancelable: true }));
        tgt.dispatchEvent(new DragEvent('drop',      { dataTransfer: dt, bubbles: true, cancelable: true }));
        src.dispatchEvent(new DragEvent('dragend',   { dataTransfer: dt, bubbles: true }));

        results.push({src: srcAid, tgt: tgtAid, data});
    }
    return results;
}"""

_READ_TEXT_CONTENT_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    return el ? el.textContent.trim() : '';
}"""

_READ_TEXT_TRUNC_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    return el ? el.textContent.trim().slice(0, 30) : null;
}"""

_DISPATCH_DRAGEND_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (el) el.dispatchEvent(new DragEvent('dragend', {bubbles: true}));
}"""

_DETECT_SLIDER_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return { error: 'not_found' };
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role');
    const bcr = el.getBoundingClientRect();
    if (tag === 'input' && el.type === 'range') {
        return {
            type: 'native-range',
            min: el.min !== '' ? +el.min : 0,
            max: el.max !== '' ? +el.max : 100,
            step: el.step !== '' ? +el.step : 1,
            value: +el.value,
            orientation: bcr.height > bcr.width ? 'vertical' : 'horizontal',
            track: { left: bcr.left, top: bcr.top, width: bcr.width, height: bcr.height }
        };
    }
    if (role === 'slider') {
        const parent = el.parentElement?.getBoundingClientRect() || bcr;
        return {
            type: 'aria-slider',
            min: +(el.getAttribute('aria-valuemin') || '0'),
            max: +(el.getAttribute('aria-valuemax') || '100'),
            value: +(el.getAttribute('aria-valuenow') || '0'),
            orientation: el.getAttribute('aria-orientation')
                || (parent.height > parent.width ? 'vertical' : 'horizontal'),
            track: { left: parent.left, top: parent.top, width: parent.width, height: parent.height },
            thumb: { left: bcr.left, top: bcr.top, width: bcr.width, height: bcr.height }
        };
    }
    const parent = el.parentElement?.getBoundingClientRect() || bcr;
    return {
        type: 'generic',
        orientation: parent.width >= parent.height ? 'horizontal' : 'vertical',
        track: { left: parent.left, top: parent.top, width: parent.width, height: parent.height },
        thumb: { left: bcr.left, top: bcr.top, width: bcr.width, height: bcr.height }
    };
}"""

_SLIDER_READBACK_JS: str = """(aid) => {
    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
    if (!el) return null;
    if (el.tagName === 'INPUT') return el.value;
    return el.getAttribute('aria-valuenow');
}"""

_PROBE_DROP_SETUP_JS: str = """() => {
    window.__cdx_drop_targets = [];
    window.__cdx_orig_pd = Event.prototype.preventDefault;
    window.__cdx_current_target = null;
    Event.prototype.preventDefault = function() {
        if (this.type === 'dragover' && window.__cdx_current_target) {
            window.__cdx_drop_targets.push(window.__cdx_current_target);
        }
        return window.__cdx_orig_pd.call(this);
    };
}"""

_PROBE_DROP_TEARDOWN_JS: str = """() => {
    Event.prototype.preventDefault = window.__cdx_orig_pd;
    const aids = [...new Set(window.__cdx_drop_targets)];
    delete window.__cdx_drop_targets;
    delete window.__cdx_orig_pd;
    delete window.__cdx_current_target;
    return aids;
}"""

# ─────────────────────────────────────────────────────────────────
# Mutation-observer JS — lifted from chunk 2.2's mutation_observer.py.
# Lives at sm-level because sm is what runs these JS calls (the
# observer is a Chrome-side concern). The Pydantic helpers and
# schemas are below alongside ActionResult.
# ─────────────────────────────────────────────────────────────────

_OBSERVER_INJECT_JS = """
() => {
  // Always reinstall (baseline must match current page state)
  if (window.__cdxMutationObserver) {
    window.__cdxMutationObserver.disconnect();
    window.__cdxMutationObserver = null;
  }

  const INTERACTIVE_SELECTOR =
    'button, input, select, textarea, a[href], form, dialog, ' +
    '[role="button"], [draggable="true"], canvas, video, audio';
  const SIGNIFICANT_ATTRS = new Set([
    'disabled', 'hidden', 'aria-hidden', 'aria-disabled', 'class',
    'aria-checked', 'aria-selected',
  ]);

  // --- Module-level counters (persist across batches) ---
  // _obsCounter: monotonically increasing, never resets. Globally unique obs_ids.
  if (typeof window.__cdxObsCounter === 'undefined') window.__cdxObsCounter = 0;
  // _seqCounter: monotonically increasing record sequence number.
  if (typeof window.__cdxSeqCounter === 'undefined') window.__cdxSeqCounter = 0;

  // --- Per-batch state (cleared by Python after checkPersistence) ---
  // NOTE: _rootRefs and _subjectRefs accumulate across per-action flushes within
  // a batch. They are only cleared by Python after checkPersistence returns.
  // This is intentional — persistence checking needs element refs from the
  // entire batch. Normal batches (5-12 actions) produce bounded entries.
  const _rootRefs = new Map();     // Element -> obs_id (new subtree roots this batch)
  const _subjectRefs = new Map();  // Element -> obs_id (any episode owner incl fallback)
  let _currentStepIndex = null;

  // Read max AID counter from existing stamped elements
  let maxAid = 0;
  for (const el of document.querySelectorAll('[data-cdx-aid]')) {
    const m = el.getAttribute('data-cdx-aid').match(/aid-(\\d+)/);
    if (m) maxAid = Math.max(maxAid, parseInt(m[1]));
  }

  const buffer = [];

  function stampAid(el) {
    if (el.hasAttribute('data-cdx-aid')) return el.getAttribute('data-cdx-aid');
    maxAid++;
    const aid = 'aid-' + maxAid;
    el.setAttribute('data-cdx-aid', aid);
    return aid;
  }

  function ensureObsId(el) {
    if (_rootRefs.has(el)) return _rootRefs.get(el);
    if (_subjectRefs.has(el)) return _subjectRefs.get(el);
    const existingAid = el.getAttribute && el.getAttribute('data-cdx-aid');
    if (existingAid) return existingAid;
    window.__cdxObsCounter++;
    const oid = 'obs-' + window.__cdxObsCounter;
    _subjectRefs.set(el, oid);
    return oid;
  }

  function makeSubject(el, obsId) {
    const tag = (el.tagName || 'unknown').toLowerCase();
    const role = el.getAttribute ? el.getAttribute('role') : null;
    const aid = el.getAttribute ? el.getAttribute('data-cdx-aid') : null;
    const text = (el.textContent || '').trim().slice(0, 120);

    let z = null;
    let topLayer = null;
    let pointerBlocking = null;
    // Note: for removed nodes, getComputedStyle returns empty/default styles.
    // The try/catch handles throws; null results are harmless (z_index etc stay null).
    try {
      const style = getComputedStyle(el);
      if (style.position === 'fixed' || style.position === 'absolute') {
        const zi = parseInt(style.zIndex);
        if (!isNaN(zi)) {
          z = zi;
          topLayer = zi > 0;
          if (style.position === 'fixed') {
            const rect = el.getBoundingClientRect();
            pointerBlocking = rect.width > window.innerWidth * 0.5
                           && rect.height > window.innerHeight * 0.5;
          }
        }
      }
    } catch (e) {}

    const interactive = el.matches ? el.matches(INTERACTIVE_SELECTOR) : false;
    let controlKind = null;
    if (interactive) {
      controlKind = tag;
      if (tag === 'a') controlKind = 'link';
      if (role === 'button') controlKind = 'button';
    }

    // Parent identity
    let parentAid = null;
    let parentObsId = null;
    const p = el.parentElement;
    if (p) {
      parentAid = p.getAttribute ? p.getAttribute('data-cdx-aid') : null;
      if (_rootRefs.has(p)) parentObsId = _rootRefs.get(p);
      else if (_subjectRefs.has(p)) parentObsId = _subjectRefs.get(p);
    }

    return {
      obs_id: obsId,
      aid: aid,
      parent_obs_id: parentObsId,
      parent_aid: parentAid,
      tag: tag,
      role: role || null,
      control_kind: controlKind,
      text_preview: text || null,
      top_layer_hint: topLayer,
      pointer_blocking_hint: pointerBlocking,
      interactive_hint: interactive || null,
      z_index: z,
    };
  }

  // Three-step text_changed ancestor walk
  // Priority: data-cdx-aid first (more specific — individual button/control),
  // then _rootRefs (coarser — container that introduced the subtree).
  function resolveTextOwner(startEl) {
    let el = startEl;
    // Step 1: check data-cdx-aid (most specific ancestor wins)
    while (el) {
      if (el.hasAttribute && el.hasAttribute('data-cdx-aid')) {
        const aid = el.getAttribute('data-cdx-aid');
        return { el: el, obsId: aid };
      }
      el = el.parentElement;
    }
    // Step 2: check _rootRefs (new subtree roots from this batch)
    el = startEl;
    while (el) {
      if (_rootRefs.has(el)) return { el: el, obsId: _rootRefs.get(el) };
      el = el.parentElement;
    }
    // Step 3: fallback to startEl
    return { el: startEl, obsId: ensureObsId(startEl) };
  }

  function pushRecord(rec) {
    window.__cdxSeqCounter++;
    rec.seq = window.__cdxSeqCounter;
    rec.step_index = _currentStepIndex;
    buffer.push(rec);
    if (buffer.length > 200) buffer.splice(0, buffer.length - 200);
  }

  const observer = new MutationObserver((mutations) => {
    const ts = Date.now();
    for (const m of mutations) {
      // --- childList: added/removed subtree roots + new text nodes ---
      if (m.type === 'childList') {
        for (const node of m.addedNodes) {
          // New text node → text_changed via ancestor walk
          if (node.nodeType === 3) {
            const text = (node.textContent || '').trim();
            if (!text) continue;
            const parent = node.parentElement;
            if (!parent) continue;
            const owner = resolveTextOwner(parent);
            pushRecord({
              ts_ms: ts, op: 'text_changed',
              subject: makeSubject(owner.el, owner.obsId),
              text_before: '', text_after: text.slice(0, 120),
            });
            continue;
          }
          // Element node → subtree root
          if (node.nodeType !== 1) continue;

          // Stamp positioned containers + interactive elements (they need AIDs)
          if (node.matches && !node.hasAttribute('data-cdx-aid')) {
            try {
              const style = getComputedStyle(node);
              if ((style.position === 'fixed' || style.position === 'absolute')
                  && style.zIndex !== 'auto') {
                stampAid(node);
              }
            } catch (e) {}
          }
          if (node.matches && node.matches(INTERACTIVE_SELECTOR)) stampAid(node);
          if (node.querySelectorAll) {
            for (const el of node.querySelectorAll(INTERACTIVE_SELECTOR)) stampAid(el);
          }

          // Record as subtree root (broad capture — no content filter)
          const obsId = ensureObsId(node);
          _rootRefs.set(node, obsId);

          let subtreeSize = 1;
          if (node.querySelectorAll) {
            subtreeSize = node.querySelectorAll('*').length + 1;
          }

          pushRecord({
            ts_ms: ts, op: 'node_added',
            subject: makeSubject(node, obsId),
            subtree_size_hint: subtreeSize > 1 ? subtreeSize : null,
          });
        }

        for (const node of m.removedNodes) {
          if (node.nodeType !== 1) continue;
          // Record removal for the root
          const aid = node.getAttribute && node.getAttribute('data-cdx-aid');
          const obsId = _rootRefs.has(node) ? _rootRefs.get(node)
                      : _subjectRefs.has(node) ? _subjectRefs.get(node)
                      : aid || null;
          if (obsId) {
            pushRecord({
              ts_ms: ts, op: 'node_removed',
              subject: makeSubject(node, obsId),
              removed_ts_ms: ts,
            });
          }
          // Record removal for stamped children
          if (node.querySelectorAll) {
            for (const el of node.querySelectorAll('[data-cdx-aid]')) {
              const childAid = el.getAttribute('data-cdx-aid');
              pushRecord({
                ts_ms: ts, op: 'node_removed',
                subject: makeSubject(el, childAid),
                removed_ts_ms: ts,
              });
            }
          }
        }
        continue;
      }

      // --- Attribute changes on stamped elements ---
      if (m.type === 'attributes') {
        const attr = m.attributeName;
        if (!attr || !SIGNIFICANT_ATTRS.has(attr)) continue;
        const el = m.target;
        if (!el.hasAttribute || !el.hasAttribute('data-cdx-aid')) continue;
        if (attr === 'class') {
          const oldCls = m.oldValue || '';
          const newCls = el.className || '';
          const nc = typeof newCls === 'string' ? newCls : '';
          const changed = (
            oldCls.includes('hidden') !== nc.includes('hidden') ||
            oldCls.includes('disabled') !== nc.includes('disabled')
          );
          if (!changed) continue;
        }
        const aid = el.getAttribute('data-cdx-aid');
        pushRecord({
          ts_ms: ts, op: 'attr_changed',
          subject: makeSubject(el, aid),
          attr_field: attr,
          attr_before: m.oldValue,
          attr_after: el.getAttribute(attr),
        });
      }

      // --- Text content changes (characterData) ---
      if (m.type === 'characterData') {
        const textNode = m.target;
        const parent = textNode.parentElement;
        if (!parent) continue;
        const oldText = (m.oldValue || '').trim().slice(0, 120);
        const newText = (textNode.textContent || '').trim().slice(0, 120);
        if (oldText === newText) continue;
        const owner = resolveTextOwner(parent);
        pushRecord({
          ts_ms: ts, op: 'text_changed',
          subject: makeSubject(owner.el, owner.obsId),
          text_before: oldText, text_after: newText,
        });
      }
    }
    window.__cdxMutationSeqNo = (window.__cdxMutationSeqNo || 0) + mutations.length;
  });

  observer.observe(document.body, {
    childList: true,
    attributes: true,
    characterData: true,
    subtree: true,
    attributeOldValue: true,
    characterDataOldValue: true,
    attributeFilter: Array.from(SIGNIFICANT_ATTRS),
  });

  // --- Public API ---
  window.__cdxMutationBuffer = buffer;
  window.__cdxMutationObserver = observer;
  window.__cdxRootRefs = _rootRefs;
  window.__cdxSubjectRefs = _subjectRefs;

  window.__cdx_markStep = function(n) {
    _currentStepIndex = n;
  };

  window.__cdx_checkPersistence = function() {
    const disconnected = [];
    for (const [el, obsId] of _rootRefs) {
      if (!el.isConnected) disconnected.push(obsId);
    }
    return disconnected;
  };

  window.__cdx_flush = function() {
    const snapshot = buffer.splice(0, buffer.length);
    return { mutations: snapshot, count: snapshot.length };
  };

  return 'v2 installed (maxAid: ' + maxAid + ')';
}
"""

_FLUSH_JS = """
() => {
  if (typeof window.__cdx_flush === 'function') return window.__cdx_flush();
  return { mutations: [], count: 0 };
}
"""

_PEEK_JS = """
() => {
  const buf = window.__cdxMutationBuffer;
  if (!buf) return { count: 0 };
  return { count: buf.length };
}
"""

_DISCONNECT_JS = """
() => {
  if (window.__cdxMutationObserver) {
    window.__cdxMutationObserver.disconnect();
    window.__cdxMutationObserver = null;
  }
  window.__cdxMutationBuffer = null;
  window.__cdxMutationBaseline = null;
  return { disconnected: true };
}
"""


_WAIT_SETTLE_JS = """
(args) => new Promise((resolve) => {
    const quietMs = args[0];
    const maxMs = args[1];
    const buf = window.__cdxMutationBuffer;
    if (!buf) { resolve(0); return; }
    const start = Date.now();
    const startSeq = window.__cdxMutationSeqNo || 0;
    let lastSeq = startSeq;
    let lastChangeAt = start;
    let done = false;

    const finish = () => {
        if (done) return;
        done = true;
        resolve((window.__cdxMutationSeqNo || 0) - startSeq);
    };

    // setTimeout watchdog: guarantees max_ms cap even if rAF is paused
    setTimeout(finish, maxMs);

    const check = () => {
        if (done) return;
        const now = Date.now();
        const curSeq = window.__cdxMutationSeqNo || 0;
        if (curSeq !== lastSeq) { lastSeq = curSeq; lastChangeAt = now; }
        if (now - lastChangeAt >= quietMs) { finish(); return; }
        if (now - start >= maxMs) { finish(); return; }
        setTimeout(check, 20);
    };
    setTimeout(check, 20);
})
"""


# Used by _extract_code in copy_paste — short alphanumeric tokens.
_CODE_PATTERN: re.Pattern = re.compile(r"\b[A-Za-z0-9]{4,10}\b")


# ── Mutation observer pure-Python helpers ─────────────────────
# `records_from_raw` constructs Pydantic records from raw JS dicts.
# These now live in morphnet_v2/mutation_types.py — re-exported below
# so any existing `from morphnet_v2.session_manager import ...` callers
# keep working. New code should import from mutation_types directly.
# `summarize_mutations` is a debug formatter used by page_agent's batch loop.

# records_from_raw + summarize_mutations now live in morphnet_v2.mutation_types.
# Imported at top of file; re-exported here for backwards compatibility with
# any callers still using `from morphnet_v2.session_manager import ...`.



def _compact_pw_error(exc: Exception) -> tuple[str, str, Optional[str]]:
    """Extract a compact agent-facing message from a Playwright exception.
    Returns (compact_message, full_raw_log, blocker_aid)."""
    raw = str(exc)
    blocker_match = re.search(r'data-cdx-aid="(aid-\d+)"[^>]*>.*?intercepts pointer events', raw)
    if blocker_match:
        return f"blocked by {blocker_match.group(1)}", raw, blocker_match.group(1)
    if "element is not enabled" in raw:
        return "element is disabled", raw, None
    timeout_match = re.search(r"Timeout (\d+)ms exceeded", raw)
    if timeout_match:
        return f"timeout {timeout_match.group(1)}ms", raw, None
    return raw.split("\n")[0][:200], raw, None


# ─────────────────────────────────────────────────────────────────
# ActionResult — the wire shape every action returns
# ─────────────────────────────────────────────────────────────────
@dataclass
class ActionResult:
    success: bool
    message: str
    reason_code: Optional[str] = None
    fail_subtype: Optional[str] = None        # not_found | hidden | disabled | zero_size | blocked
    raw_log: Optional[str] = None              # full Playwright log; not for agent
    blocker_probe: Optional[dict] = None       # rich blocker info from _prep
    blocker_aid: Optional[str] = None          # AID of element that intercepted pointer events
    navigation_occurred: bool = False          # context died / URL changed during action — caller should abort batch and re-extract


# MutationNodeRef and RawMutationRecord now live in morphnet_v2.mutation_types.
# Imported at top of file; re-exported here for backwards compatibility.


# ─────────────────────────────────────────────────────────────────
# HTTP + script capture (chunk 1.6)
# CapturedRequest is the in-memory wire shape consumers query via
# session.get_traffic(). ScriptInfo tracks per-scriptId metadata
# and the (lazy) source bytes.
# ─────────────────────────────────────────────────────────────────

JS_MIMES: frozenset[str] = frozenset({
    "application/javascript",
    "application/x-javascript",
    "application/ecmascript",
    "text/javascript",
})


def _is_js_response(mime: Optional[str], url: str) -> bool:
    """True if this response carries JavaScript source — by content-type or URL ext."""
    if mime:
        primary = mime.split(";", 1)[0].strip().lower()
        if primary in JS_MIMES:
            return True
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(".js") or path.endswith(".mjs") or path.endswith(".cjs")


@dataclass
class CapturedRequest:
    """One HTTP exchange observed via CDP Network domain."""
    request_id: str
    ts_ms: int
    url: str
    method: str
    request_headers: dict[str, str]
    request_body: Optional[bytes] = None
    status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: Optional[bytes] = None
    response_mime: Optional[str] = None
    initiator_type: Optional[str] = None       # script | parser | preload | other
    initiator_stack: list[dict] = field(default_factory=list)
    body_error: Optional[str] = None            # if getResponseBody failed
    from_cache: bool = False
    error: Optional[str] = None                 # for loadingFailed cases


@dataclass
class ScriptInfo:
    """Metadata + (lazy) source for one V8-parsed script."""
    script_id: str
    url: str
    v8_hash: str                                # Chrome-internal content hash
    length: int
    sha256: Optional[str] = None                # set after source linked
    source: Optional[str] = None                # decoded UTF-8 text


def _launch_chrome(
    *,
    chrome_path: str = CHROME_PATH,
    port: int,
    user_data_dir: Path | None,
    headless: bool,
    viewport: tuple[int, int] = (1920, 1080),
) -> subprocess.Popen:
    """Launch Chrome. If user_data_dir is None, a fresh tmpdir is used."""
    if user_data_dir is None:
        user_data_dir = Path(tempfile.mkdtemp(prefix="mn_v2_chrome_"))

    flags = [
        *CHROME_FLAGS_BASE,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        f"--window-size={viewport[0]},{viewport[1]}",
    ]
    if headless:
        flags.append("--headless=new")
        flags.append(f"--user-agent={HEADLESS_UA_OVERRIDE}")

    return subprocess.Popen(
        [chrome_path, *flags],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ─────────────────────────────────────────────────────────────────
# Raw CDP — direct WebSocket, no Playwright
# ─────────────────────────────────────────────────────────────────

async def _wait_for_page_target(port: int) -> dict:
    """Poll /json/list until Chrome reports a page target.

    Doubles as the readiness probe — a successful /json/list implies Chrome
    is alive. Returns the page tab dict (id, webSocketDebuggerUrl, ...).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + CDP_READY_TIMEOUT_S
    async with httpx.AsyncClient() as http:
        while loop.time() < deadline:
            try:
                r = await http.get(f"http://localhost:{port}/json/list", timeout=1.0)
                if r.status_code == 200:
                    page = next((t for t in r.json() if t.get("type") == "page"), None)
                    if page:
                        return page
            except Exception:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError(f"No page target on Chrome CDP port {port} within {CDP_READY_TIMEOUT_S}s")


class CDPSession:
    """Raw CDP WebSocket client. Request/response + event dispatch.

    Connects directly to Chrome's debug WebSocket. No Playwright. Each
    send() and each received event is mirrored to notes.log() automatically.
    """

    def __init__(self, port: int):
        self.port = port
        self.target_id: Optional[str] = None
        self._ws: Optional[Any] = None
        self._next_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._handlers: dict[str, list[Callable]] = {}
        self._read_task: Optional[asyncio.Task] = None

    async def attach(self) -> None:
        """Wait for a page target, then open the WS to it and start the read loop."""
        page = await _wait_for_page_target(self.port)
        self.target_id = page["id"]
        self._ws = await websockets.connect(
            page["webSocketDebuggerUrl"],
            max_size=128 * 1024 * 1024,
        )
        self._read_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            await self._ws.close()

    async def send(self, method: str, params: dict | None = None) -> dict:
        """Send a CDP method on the page WS and await the response."""
        if self._ws is None:
            raise RuntimeError("CDP not attached")
        self._next_id += 1
        msg_id = self._next_id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        msg = {"id": msg_id, "method": method, "params": params or {}}
        await self._ws.send(json.dumps(msg))
        notes.log(data_type="cdp_send", data=msg, method=method)
        return await fut

    def subscribe(self, event: str, handler: Callable[[dict], Any]) -> Callable[[], None]:
        """Subscribe to a CDP event. Returns an unsubscribe callable."""
        self._handlers.setdefault(event, []).append(handler)

        def unsub() -> None:
            try:
                self._handlers[event].remove(handler)
            except (ValueError, KeyError):
                pass

        return unsub

    async def _read_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                payload = json.loads(raw)
                if "id" in payload:
                    fut = self._pending.pop(payload["id"], None)
                    if fut and not fut.done():
                        if "error" in payload:
                            fut.set_exception(
                                RuntimeError(payload["error"].get("message", "CDP error"))
                            )
                        else:
                            fut.set_result(payload.get("result", {}))
                elif "method" in payload:
                    notes.log(data_type="cdp_event", data=payload, event=payload["method"])
                    for h in list(self._handlers.get(payload["method"], ())):
                        try:
                            # Handlers MUST be sync — anything that needs
                            # await should schedule its own anchored task via
                            # SessionManager._capture_tasks. Returning a
                            # coroutine here would create an untracked task
                            # that close() can't cancel.
                            h(payload.get("params", {}))
                        except Exception:
                            logger.exception("CDP handler raised on %s", payload["method"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("CDP read loop terminated unexpectedly")


# ─────────────────────────────────────────────────────────────────
# SessionManager — one Chrome lifecycle, one experiment
# ─────────────────────────────────────────────────────────────────

class SessionManager:
    """One Chrome session: launch → CDP attach → navigate → tear down.

    All non-Python-process I/O for one experiment / task / run flows through
    this object. Optional `results_dir` causes notes to attach automatically,
    so every CDP message gets logged to {results_dir}/{ts}-{site}/.
    """

    def __init__(
        self,
        *,
        start_url: str,
        site_name: Optional[str] = None,
        headless: bool = False,
        port: int = 9222,
        viewport: tuple[int, int] = (1920, 1080),
        user_data_dir: Path | None = None,
        results_dir: Path | str | None = None,
        task_metadata: dict | None = None,
        max_steps: int = 10,
        max_turns_per_step: int = 60,
    ):
        """All callers go through SessionManager. Calling `await sm.run_task(task)`
        lazily builds an `Orchestrator` (which owns PageAgent + ToolExecutor +
        PageFilter internally) and hands the task to it.

        `task_metadata` is a WRITE-ONLY trace tag. It gets persisted to
        metadata.json on disk for forensics (eval-time grading joins on label/
        site/expected_answer). It MUST NEVER be read into any LLM-facing code
        path — see `feedback_task_metadata_write_only` memory entry. The only
        legitimate reader is the `notes.log(data_type="metadata", ...)` call
        inside `start()`.

        Example:
            async with SessionManager(start_url="...") as sm:
                tree = await sm.run_task("...")
                print(tree.success, tree.final_answer)
        """
        self.start_url = start_url
        self.site_name = site_name or notes.site_name_from_url(start_url)
        self.headless = headless
        self.port = port
        self.viewport = viewport
        self.user_data_dir = user_data_dir
        self.results_dir = Path(results_dir) if results_dir else None
        # WRITE-ONLY trace tag — DO NOT propagate to any LLM prompt or
        # high-layer module. Only the metadata.json write site at line ~1318
        # may read this. See feedback_task_metadata_write_only memory.
        self.task_metadata = task_metadata or {}
        self._max_steps = max_steps
        self._max_turns_per_step = max_turns_per_step
        self._orchestrator: Optional[Any] = None  # lazily built on first run_task

        self._chrome_proc: Optional[subprocess.Popen] = None
        self._cdp: Optional[CDPSession] = None
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._notes_attached: bool = False
        # Baseline URL for check_navigation(). Callers set this directly:
        #     sm.url_before_action = sm.page.url
        # Chunk 1.5 action dispatch sets it before each action; chunk 2.4
        # page_agent sets it at the start of a page-processing block.
        self.url_before_action: str = ""

        # capture state (chunk 1.6)
        self._pending: dict[str, dict] = {}          # request_id → partial record
        self._traffic: dict[str, CapturedRequest] = {}
        self._scripts: dict[str, ScriptInfo] = {}     # script_id → metadata + source
        self._script_hashes: set[str] = set()         # SHA256s already written to notes
        self._js_bytes_by_url: dict[str, bytes] = {}  # cache so scriptParsed can reuse Network bytes
        self._capture_tasks: set[asyncio.Task] = set()

    # ── public access ─────────────────────────────────────────────
    @property
    def cdp(self) -> CDPSession:
        if self._cdp is None:
            raise RuntimeError("CDP not attached — call start() first")
        return self._cdp

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Playwright Page not attached — call start() first")
        return self._page

    # ── lifecycle ─────────────────────────────────────────────────
    async def start(self) -> None:
        # Partial-failure cleanup: `__aexit__` is NOT called when `__aenter__`
        # raises, so without this we'd leak the Chrome subprocess, notes
        # handle, playwright resources, and tempdir if any step below throws.
        # `close()` is idempotent and safe under any partial state.
        try:
            # 1. Notes — attach FIRST so every CDP message that follows is logged.
            if self.results_dir is not None:
                notes.attach(self.results_dir, self.site_name)
                self._notes_attached = True
                notes.log(
                    data_type="metadata",
                    data={
                        "start_url": self.start_url,
                        "site_name": self.site_name,
                        "headless": self.headless,
                        "viewport": list(self.viewport),
                        "port": self.port,
                        **self.task_metadata,
                    },
                )

            # 2. Chrome subprocess.
            self._chrome_proc = _launch_chrome(
                port=self.port,
                user_data_dir=self.user_data_dir,
                headless=self.headless,
                viewport=self.viewport,
            )

            # 3. Raw CDP attach (waits for /json/list to surface a page target).
            self._cdp = CDPSession(self.port)
            await self._cdp.attach()

            # 4. Enable domains we need. Runtime.enable held back deliberately —
            #    consoleAPICalled-family events some bot detectors look for.
            await self._cdp.send("Network.enable")
            await self._cdp.send("Page.enable")

            # 4b. Wire up HTTP + script capture (chunk 1.6) BEFORE navigating so
            #     we capture from the very first request. LEARNINGS phase 4/5/10
            #     confirm Network.enable + Debugger.enable don't trip detection
            #     on Chrome 148+ across the protected sites we care about.
            self._cdp.subscribe("Network.requestWillBeSent",         self._on_request)
            self._cdp.subscribe("Network.responseReceived",          self._on_response)
            self._cdp.subscribe("Network.responseReceivedExtraInfo", self._on_response_extra)
            self._cdp.subscribe("Network.loadingFinished",           self._on_loading_finished)
            self._cdp.subscribe("Network.loadingFailed",             self._on_loading_failed)
            self._cdp.subscribe("Network.requestServedFromCache",    self._on_request_from_cache)

            await self._cdp.send("Debugger.enable")
            await self._cdp.send("Debugger.setAsyncCallStackDepth", {"maxDepth": 32})
            self._cdp.subscribe("Debugger.scriptParsed", self._on_script_parsed)

            await self._cdp.send("Profiler.enable")
            await self._cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})

            # auto-attach to iframes / service workers / web workers; events
            # come through this same WebSocket with a sessionId field we ignore.
            try:
                await self._cdp.send("Target.setAutoAttach", {
                    "autoAttach": True,
                    "waitForDebuggerOnStart": False,
                    "flatten": True,
                })
            except Exception:
                logger.debug("Target.setAutoAttach failed (non-fatal)")

            # 5. Inject nav-capture init script BEFORE navigating so it runs on
            #    the first document. Page.enable above is required for it to
            #    fire on subsequent navigations.
            await self._cdp.send(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": NAV_CAPTURE_INIT_SCRIPT},
            )

            # 6. Subscribe to Page.loadEventFired BEFORE Page.navigate to avoid
            #    the race where load fires before our handler is registered.
            loop = asyncio.get_event_loop()
            load_fut: asyncio.Future = loop.create_future()
            unsub = self._cdp.subscribe(
                "Page.loadEventFired",
                lambda _params: load_fut.set_result(None) if not load_fut.done() else None,
            )
            try:
                await self._cdp.send("Page.navigate", {"url": self.start_url})
                await asyncio.wait_for(load_fut, timeout=NAVIGATION_TIMEOUT_S)
            finally:
                unsub()

            # 7. Attach Playwright eagerly over the same Chrome (LEARNINGS phase 10:
            #    Runtime.enable on connect_over_cdp is invisible on Chrome 148+).
            #    Reuses the existing context/page rather than opening new ones.
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.connect_over_cdp(
                f"http://localhost:{self.port}"
            )
            ctx = (
                self._browser.contexts[0]
                if self._browser.contexts
                else await self._browser.new_context()
            )
            self._page = ctx.pages[0] if ctx.pages else await ctx.new_page()

            logger.info(
                "SessionManager started: %s (port=%d, headless=%s)",
                self.start_url, self.port, self.headless,
            )
        except BaseException:
            with suppress(Exception):
                await self.close()
            raise

    async def close(self) -> None:
        """Idempotent teardown — Playwright, CDP, Chrome subprocess, notes."""
        # Cancel any in-flight capture background tasks first (chunk 1.6)
        if self._capture_tasks:
            for task in list(self._capture_tasks):
                task.cancel()
            with suppress(Exception):
                await asyncio.gather(*self._capture_tasks, return_exceptions=True)
            self._capture_tasks.clear()

        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
            self._page = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

        if self._cdp is not None:
            try:
                await self._cdp.close()
            except Exception:
                logger.exception("CDP close failed")
            self._cdp = None

        if self._chrome_proc is not None:
            if self._chrome_proc.poll() is None:
                self._chrome_proc.terminate()
                try:
                    self._chrome_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._chrome_proc.kill()
            self._chrome_proc = None

        if self._notes_attached:
            notes.detach()
            self._notes_attached = False

        logger.info("SessionManager closed")

    # ── page lifecycle (chunk 1.4) ────────────────────────────────
    async def wait_for_page_ready(
        self,
        *,
        poll_interval_ms: int = DOM_STABILITY_POLL_MS,
        stable_window_ms: int = DOM_STABILITY_WINDOW_MS,
        max_wait_ms: int = DOM_STABILITY_MAX_WAIT_MS,
    ) -> bool:
        """Poll the DOM (HTML + visible-text length) every poll_interval_ms.
        When both lengths stay unchanged for stable_window_ms consecutive ms,
        consider the page settled and return True. Return False on timeout.

        Lifted from crawler/browser_tools.py:_wait_for_dom_stability.
        """
        stable_elapsed = 0
        waited = 0
        last_html = -1
        last_text = -1
        while waited < max_wait_ms:
            try:
                probe = await self.page.evaluate(_DOM_PROBE_JS)
                html_chars = int(probe["htmlChars"])
                text_chars = int(probe["textChars"])
            except Exception:
                # E.g. "Execution context was destroyed" mid-poll. Reset and retry.
                stable_elapsed = 0
                last_html = -1
                last_text = -1
                await asyncio.sleep(poll_interval_ms / 1000)
                waited += poll_interval_ms
                continue

            if html_chars == last_html and text_chars == last_text:
                stable_elapsed += poll_interval_ms
            else:
                stable_elapsed = 0
            last_html = html_chars
            last_text = text_chars

            if stable_elapsed >= stable_window_ms:
                return True

            await self.page.wait_for_timeout(poll_interval_ms)
            waited += poll_interval_ms
        return False

    def check_navigation(self) -> Optional[str]:
        """Synchronous URL diff with fragment stripping. Returns the new URL
        if navigation occurred since `url_before_action` was set, else None.
        Hash-only changes (`/foo#a` → `/foo#b`) are not treated as navigation.
        """
        current = self.page.url
        if current.split("#", 1)[0] != self.url_before_action.split("#", 1)[0]:
            return current
        return None

    async def wait_for_navigation(self, timeout_ms: int = NAV_WAIT_TIMEOUT_MS) -> bool:
        """Event-driven wait for the URL to change (fragment-aware). Used by
        chunk 1.5 when the action's intent is navigation. Returns True if URL
        changed within timeout, False otherwise. The synchronous check_navigation
        fallback still catches navs that fire after this returns False.
        """
        old_bare = self.page.url.split("#", 1)[0]
        try:
            await self.page.wait_for_url(
                lambda url: url.split("#", 1)[0] != old_bare,
                timeout=timeout_ms,
                wait_until="commit",
            )
            return True
        except Exception:
            return False

    async def wait_for_dom_content_loaded(self, timeout_ms: int = DOMCL_TIMEOUT_MS) -> bool:
        """Wait for the new doc's DOMContentLoaded event after a navigation.
        Returns True if fired within timeout, False otherwise. Caller typically
        invokes this right after wait_for_navigation returned True, before
        running extraction against the new page.
        """
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            return True
        except Exception:
            return False

    # ── HTTP + script capture (chunk 1.6) ──────────────────────────
    # Synchronous handlers. They DO NOT await; they update dicts and
    # spawn background tasks for any work that requires CDP roundtrips
    # (getResponseBody, getScriptSource). Read loop never blocks.

    def _on_request(self, params: dict) -> None:
        try:
            rid = params["requestId"]
            request = params.get("request", {})
            initiator = params.get("initiator", {})
            self._pending[rid] = {
                "ts_ms": int(time.time() * 1000),
                "url": request.get("url", ""),
                "method": request.get("method", "GET"),
                "request_headers": dict(request.get("headers", {})),
                "request_body": request.get("postData"),
                "initiator_type": initiator.get("type", "other"),
                "initiator_stack": self._extract_stack_frames(initiator.get("stack")),
            }
        except Exception:
            logger.exception("on_request handler failed")

    def _on_response(self, params: dict) -> None:
        try:
            rid = params.get("requestId")
            if rid not in self._pending:
                return
            response = params.get("response", {})
            self._pending[rid]["status"] = response.get("status", 0)
            self._pending[rid]["response_headers"] = dict(response.get("headers", {}))
            self._pending[rid]["response_mime"] = response.get("mimeType")
        except Exception:
            logger.exception("on_response handler failed")

    def _on_response_extra(self, params: dict) -> None:
        # Network.responseReceivedExtraInfo carries raw Set-Cookie headers
        # untruncated. LEARNINGS phase 9: Chrome truncates Set-Cookie in the
        # `responseReceived` event, so the value diff that exposed _abck=−1
        # only became visible by reading THIS event. Log everything verbatim.
        try:
            notes.log(
                data_type="cookie_set",
                data=params,
                request_id=params.get("requestId"),
            )
        except Exception:
            logger.exception("on_response_extra handler failed")

    def _on_loading_finished(self, params: dict) -> None:
        rid = params.get("requestId")
        if rid not in self._pending:
            return
        task = asyncio.create_task(self._finalize_request(rid))
        self._capture_tasks.add(task)
        task.add_done_callback(self._capture_tasks.discard)

    def _on_loading_failed(self, params: dict) -> None:
        try:
            rid = params.get("requestId")
            if rid is None:
                return
            pending = self._pending.pop(rid, None)
            if pending is None:
                return
            captured = CapturedRequest(
                request_id=rid,
                ts_ms=pending["ts_ms"],
                url=pending["url"],
                method=pending["method"],
                request_headers=pending["request_headers"],
                request_body=self._encode_body(pending.get("request_body")),
                status=pending.get("status", 0),
                response_headers=pending.get("response_headers", {}),
                response_body=None,
                response_mime=pending.get("response_mime"),
                initiator_type=pending.get("initiator_type"),
                initiator_stack=pending.get("initiator_stack", []),
                error=params.get("errorText"),
            )
            self._traffic[rid] = captured
            notes.log(
                data_type="http_request",
                data=captured.request_body,
                request_id=rid,
                url=captured.url,
                method=captured.method,
                request_headers=captured.request_headers,
                initiator_type=captured.initiator_type,
                initiator_stack=captured.initiator_stack,
            )
            notes.log(
                data_type="http_response",
                data=None,
                request_id=rid,
                status=captured.status,
                response_headers=captured.response_headers,
                response_mime=captured.response_mime,
                error=captured.error,
            )
        except Exception:
            logger.exception("on_loading_failed handler failed")

    def _on_request_from_cache(self, params: dict) -> None:
        try:
            rid = params.get("requestId")
            if rid in self._pending:
                self._pending[rid]["from_cache"] = True
        except Exception:
            logger.exception("on_request_from_cache handler failed")

    def _on_script_parsed(self, params: dict) -> None:
        try:
            sid = params.get("scriptId")
            if not sid:
                return
            self._scripts[sid] = ScriptInfo(
                script_id=sid,
                url=params.get("url", ""),
                v8_hash=params.get("hash", ""),
                length=params.get("length", 0),
            )
            task = asyncio.create_task(self._link_script_source(sid))
            self._capture_tasks.add(task)
            task.add_done_callback(self._capture_tasks.discard)
        except Exception:
            logger.exception("on_script_parsed handler failed")

    # ── capture-side background tasks ──────────────────────────────

    async def _finalize_request(self, rid: str) -> None:
        """Fetch response body, build CapturedRequest, log to notes."""
        pending = self._pending.pop(rid, None)
        if pending is None:
            return

        body_bytes: Optional[bytes] = None
        body_error: Optional[str] = None
        try:
            result = await self.cdp.send("Network.getResponseBody", {"requestId": rid})
            body_str = result.get("body", "") or ""
            if result.get("base64Encoded"):
                try:
                    body_bytes = base64.b64decode(body_str)
                except Exception as e:
                    body_error = f"base64 decode: {e!r}"
            else:
                body_bytes = body_str.encode("utf-8")
        except Exception as e:
            body_error = repr(e)

        # Side-stash JS bodies so scriptParsed can reuse them (no extra CDP call)
        if body_bytes is not None and _is_js_response(pending.get("response_mime"), pending["url"]):
            self._js_bytes_by_url[pending["url"]] = body_bytes

        captured = CapturedRequest(
            request_id=rid,
            ts_ms=pending["ts_ms"],
            url=pending["url"],
            method=pending["method"],
            request_headers=pending["request_headers"],
            request_body=self._encode_body(pending.get("request_body")),
            status=pending.get("status", 0),
            response_headers=pending.get("response_headers", {}),
            response_body=body_bytes,
            response_mime=pending.get("response_mime"),
            initiator_type=pending.get("initiator_type"),
            initiator_stack=pending.get("initiator_stack", []),
            body_error=body_error,
            from_cache=pending.get("from_cache", False),
        )
        self._traffic[rid] = captured

        notes.log(
            data_type="http_request",
            data=captured.request_body,
            request_id=rid,
            url=captured.url,
            method=captured.method,
            request_headers=captured.request_headers,
            initiator_type=captured.initiator_type,
            initiator_stack=captured.initiator_stack,
        )
        notes.log(
            data_type="http_response",
            data=captured.response_body,
            request_id=rid,
            status=captured.status,
            response_headers=captured.response_headers,
            response_mime=captured.response_mime,
            body_error=body_error,
            from_cache=captured.from_cache,
        )

    async def _link_script_source(self, sid: str) -> None:
        """Attach source bytes to a ScriptInfo. Reuse Network bytes if we have
        them for this URL; otherwise call Debugger.getScriptSource. Dedup on
        SHA256 so we don't write the same content twice."""
        info = self._scripts.get(sid)
        if info is None or info.sha256 is not None:
            return

        body: Optional[bytes] = None
        if info.url and info.url in self._js_bytes_by_url:
            body = self._js_bytes_by_url[info.url]
        else:
            try:
                result = await self.cdp.send("Debugger.getScriptSource", {"scriptId": sid})
                source = result.get("scriptSource", "") or ""
                body = source.encode("utf-8")
            except Exception:
                return  # eviction, target gone, etc. — leave info.sha256 None

        if body is None or len(body) == 0:
            return
        sha = hashlib.sha256(body).hexdigest()
        info.sha256 = sha
        info.source = body.decode("utf-8", errors="replace")

        if sha not in self._script_hashes:
            self._script_hashes.add(sha)
            # Per-run forensic copy via notes (results_v2/{ts}-{site}/scripts/)
            notes.log(
                data_type="script_source",
                data=info.source,
                script_id=sid,
                sha256=sha,
                url=info.url,
                length=info.length,
            )
            # Persistent per-site context (morphnet_v2/sites/{site}/scripts/)
            # — survives across runs so Phase 5 graph replays have the bytes
            # they need without re-capturing.
            try:
                self._save_to_site_context(sha, info)
            except Exception:
                logger.exception("site-context write failed for sha=%s", sha[:8])

    # ── per-site context (chunk 1.6) ───────────────────────────────
    # morphnet_v2/sites/{site_name}/ persists across runs. Phase 4+ will add
    # profile.json, graphs/, tools.json, bundle/. Chunk 1.6 only writes the
    # scripts/ subtree — the rest is reserved for later phases.

    def _save_to_site_context(self, sha: str, info: ScriptInfo) -> None:
        """Mirror a captured script to the per-site context.

        Layout:
            morphnet_v2/sites/{site}/scripts/{sha256}.js   ← bytes (deduped)
            morphnet_v2/sites/{site}/scripts/index.json    ← metadata index

        Drift detection works off this — a graph remembers the SHA256s it
        depends on; on later runs we compare current SHA256s against the
        graph's expected ones. Phase 5 does the A/B test on mismatch.
        """
        if info.source is None:
            return
        scripts_dir = Path(__file__).parent / "sites" / self.site_name / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        bytes_path = scripts_dir / f"{sha}.js"
        if not bytes_path.exists():
            bytes_path.write_text(info.source, encoding="utf-8")

        index_path = scripts_dir / "index.json"
        try:
            index = (
                json.loads(index_path.read_text(encoding="utf-8"))
                if index_path.exists() else {}
            )
        except Exception:
            index = {}
        now_ms = int(time.time() * 1000)
        entry = index.setdefault(sha, {
            "url": info.url,
            "length": info.length,
            "first_seen_ms": now_ms,
            "runs": [],
        })
        # Keep latest known URL (a script may be served from a versioned URL
        # that changes per deploy; the SHA256 stays constant).
        if info.url and entry.get("url") != info.url:
            entry["url"] = info.url
        entry["runs"].append(now_ms)
        # Cap the tail — we don't need every run's timestamp forever.
        if len(entry["runs"]) > 50:
            entry["runs"] = entry["runs"][-50:]

        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    # ── capture-side helpers ───────────────────────────────────────

    def _extract_stack_frames(self, stack: Optional[dict]) -> list[dict]:
        """Flatten an initiator.stack into a list of frames, depth-capped at 32."""
        if not stack:
            return []
        frames: list[dict] = []
        for frame in stack.get("callFrames", []):
            frames.append({
                "scriptId": frame.get("scriptId", ""),
                "functionName": frame.get("functionName", ""),
                "lineNumber": frame.get("lineNumber", 0),
                "columnNumber": frame.get("columnNumber", 0),
                "url": frame.get("url", ""),
            })
            if len(frames) >= 32:
                return frames
        parent = stack.get("parent")
        if parent and len(frames) < 32:
            frames.extend(self._extract_stack_frames(parent))
        return frames[:32]

    @staticmethod
    def _encode_body(body: Any) -> Optional[bytes]:
        if body is None:
            return None
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        if isinstance(body, str):
            return body.encode("utf-8", errors="replace")
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

    # ── capture-side public API ────────────────────────────────────

    def get_traffic(self, since_ts_ms: int = 0) -> list[CapturedRequest]:
        """Completed requests since since_ts_ms. Returns a list of dataclass
        references (no copy). Pending requests are NOT included."""
        if since_ts_ms <= 0:
            return list(self._traffic.values())
        return [r for r in self._traffic.values() if r.ts_ms >= since_ts_ms]

    def clear_traffic(self) -> None:
        """Drop in-memory _traffic + _pending. Disk records via notes are
        unaffected — this is for callers that want to limit memory growth."""
        self._traffic.clear()
        self._pending.clear()

    async def take_coverage_snapshot(self) -> list[dict]:
        """Cumulative V8 coverage since Profiler.startPreciseCoverage. Callers
        diff adjacent snapshots offline to get per-step execution."""
        try:
            result = await self.cdp.send("Profiler.takePreciseCoverage")
        except Exception as e:
            logger.warning("take_coverage_snapshot failed: %s", e)
            return []
        return result.get("result", [])

    async def cookies_snapshot(self) -> list[dict]:
        """Network.getAllCookies snapshot of the cookie jar — includes JS-set
        cookies that never appear in any HTTP header. Logged as
        cookies_snapshot to notes."""
        try:
            result = await self.cdp.send("Network.getAllCookies")
        except Exception as e:
            logger.warning("cookies_snapshot failed: %s", e)
            return []
        cookies = result.get("cookies", [])
        notes.log(data_type="cookies_snapshot", data=cookies)
        return cookies

    async def get_script_source(self, script_id: str) -> Optional[str]:
        """Source bytes for a scriptId. Returns cached source if linked, else
        fetches now via Debugger.getScriptSource. None on failure."""
        info = self._scripts.get(script_id)
        if info is not None and info.source is not None:
            return info.source
        if info is None:
            try:
                result = await self.cdp.send("Debugger.getScriptSource", {"scriptId": script_id})
                return result.get("scriptSource")
            except Exception:
                return None
        await self._link_script_source(script_id)
        return self._scripts[script_id].source if script_id in self._scripts else None

    # ── mutation observer (chunk 2.2) ──────────────────────────────
    # Page interaction lives here. JS payloads + the pure-Python
    # summarize_mutations helper live in
    # morphnet_v2/computer_use/mutation_observer.py (byte-identical to
    # crawler's JS). Pydantic schemas live in
    # morphnet_v2/computer_use/schemas.py.

    async def install_mutation_observer(self) -> str:
        """Inject the MutationObserver. Idempotent — reinstalls with fresh baseline."""
        return await self.page.evaluate(_OBSERVER_INJECT_JS)

    async def flush_mutations(
        self, batch_id: Optional[str] = None,
    ) -> list[RawMutationRecord]:
        """Drain JS mutation buffer into typed RawMutationRecord list.

        Wraps the JS flush + Pydantic conversion in one call. Generates a
        UUID `batch_id` if not provided. Skips malformed records silently
        (lifted from crawler's flush_mutations behavior).
        """
        result = await self.page.evaluate(_FLUSH_JS)
        return records_from_raw(result.get("mutations", []) or [], batch_id=batch_id)

    async def peek_mutation_count(self) -> int:
        """Buffer length without consuming."""
        result = await self.page.evaluate(_PEEK_JS)
        return result.get("count", 0)

    async def wait_for_settle(self, quiet_ms: int = 80, max_ms: int = 500) -> int:
        """Light settle: wait until mutation buffer is quiet for `quiet_ms`. Returns mutation count.
        Heavy settle is `wait_for_page_ready` (chunk 1.4) — HTML/text-length polling, no observer.
        """
        return await self.page.evaluate(_WAIT_SETTLE_JS, [quiet_ms, max_ms])

    async def mark_mutation_step(self, step: int) -> None:
        """Tag subsequent mutations with this step index (window.__cdx_markStep(N)).
        Used by chunk 2.4's batch_executor before each action."""
        with suppress(PlaywrightError):
            await self.page.evaluate(
                f"window.__cdx_markStep && window.__cdx_markStep({step})"
            )

    async def check_persistence(self) -> list[str]:
        """Return obs_ids of subjects that disconnected from the DOM during the batch.
        Used by chunk 2.4 post-batch to apply persistence results to episodes."""
        raw = await self.page.evaluate(
            "window.__cdx_checkPersistence && window.__cdx_checkPersistence() || []"
        )
        return raw if isinstance(raw, list) else []

    async def clear_observer_refs(self) -> None:
        """Drop the observer's per-batch element refs after persistence check."""
        with suppress(PlaywrightError):
            await self.page.evaluate(
                "window.__cdxRootRefs && window.__cdxRootRefs.clear();"
                " window.__cdxSubjectRefs && window.__cdxSubjectRefs.clear()"
            )

    # ── AX push events (chunk 2.4) — vestigial per crawler ────────
    # Crawler discovered that Playwright's new_cdp_session doesn't deliver
    # Accessibility.nodesUpdated events. v2 routes through our raw sm.cdp,
    # which COULD deliver them, but for parity we lift crawler's setup
    # verbatim and treat the subscription as best-effort.

    async def enable_ax_push(self) -> Callable[[], None] | None:
        """Enable Accessibility.* events and subscribe to nodesUpdated.
        Returns an unsubscribe callable, or None if enable failed.
        The caller drains via the buffer the handler writes to."""
        try:
            await self._cdp.send("Accessibility.enable") if self._cdp else None
        except Exception:
            return None
        return None  # subscribe is owned by PageAgent — pattern mirrors crawler

    async def disable_ax_push(self) -> None:
        """Disable Accessibility events. Best-effort cleanup."""
        with suppress(Exception):
            if self._cdp is not None:
                await self._cdp.send("Accessibility.disable")

    async def disconnect_mutation_observer(self) -> None:
        """Stop observing and clean up window globals."""
        await self.page.evaluate(_DISCONNECT_JS)

    # ── action dispatch (chunk 1.5) ────────────────────────────────
    # Lifted faithfully from crawler/executor.py. Each public method is an
    # action the LLM can emit. `_log_action` mirrors every result to notes.
    # `_prep` is the shared pre-flight (find/visibility/scroll/blocker probe).

    def _log_action(self, kind: str, result: ActionResult, **extra: Any) -> ActionResult:
        """Mirror an ActionResult to notes and return it. Returning lets
        callers do `return self._log_action(...)` in one line."""
        notes.log(
            data_type="action",
            data={
                "kind": kind,
                "success": result.success,
                "message": result.message,
                "reason_code": result.reason_code,
                "fail_subtype": result.fail_subtype,
                "blocker_aid": result.blocker_aid,
                **extra,
            },
        )
        return result

    async def _safe_evaluate(
        self,
        expression: str,
        arg: Any = None,
        *,
        settle_timeout_ms: int = 5_000,
    ) -> tuple[Any, str]:
        """Wrap page.evaluate with navigation-race recovery.

        Returns (result, status):
          status='ok'         — eval succeeded, result is the page output.
          status='navigated'  — JS context was destroyed (page mid-navigation).
                                Waited up to settle_timeout_ms for the new
                                context. AIDs from the old page are stale —
                                caller MUST NOT retry the same expression;
                                instead, return ActionResult(success=True,
                                navigation_occurred=True) so the batch exits
                                cleanly and re-extraction stamps fresh AIDs.
        Any other Playwright error is re-raised so callers can handle it.
        """
        try:
            result = await self.page.evaluate(expression, arg)
            return result, "ok"
        except PlaywrightError as e:
            if "context was destroyed" not in str(e).lower():
                raise
            with suppress(PlaywrightError, asyncio.TimeoutError):
                await self.page.wait_for_load_state(
                    "domcontentloaded", timeout=settle_timeout_ms,
                )
            return None, "navigated"

    async def _prep(
        self,
        aid: str,
        *,
        hit_test: bool | str = False,
        check_disabled: bool = False,
    ) -> dict | ActionResult:
        """Shared element prep: find, checkVisibility, scroll, BCR, optional hit-test.
        Returns dict {x, y, w, h} on success, or ActionResult on error."""
        opts: dict = {}
        if hit_test:
            opts["hitTest"] = hit_test
        if check_disabled:
            opts["checkDisabled"] = True
        result, status = await self._safe_evaluate(_PREP_JS, [aid, opts])
        if status == "navigated":
            return ActionResult(
                success=True,
                navigation_occurred=True,
                message=f"page navigated during prep aid={aid}",
                reason_code="context_destroyed",
            )
        if "error" in result:
            error = result["error"]
            if error == "blocked":
                return ActionResult(
                    success=False,
                    message=f"action failed aid={aid}: blocked by {result.get('blocker_aid', '?')}",
                    reason_code="preflight_blocked",
                    fail_subtype="blocked",
                    blocker_probe=result,
                    blocker_aid=result.get("blocker_aid"),
                )
            return ActionResult(
                success=False,
                message=f"action failed aid={aid}: {error}",
                reason_code="preflight_failed",
                fail_subtype=error,
            )
        return result

    async def _describe_blocker(self, aid: str) -> Optional[str]:
        """One-line description of a blocker element, or None if gone."""
        info = await self.page.evaluate(_DESCRIBE_BLOCKER_JS, aid)
        if info is None:
            return None
        parts = [f"{info['w']}x{info['h']}px", info["tag"]]
        if info["text"]:
            parts.append(f'"{info["text"]}"')
        parts.append(f"at ({info['x']},{info['y']})")
        parts.append(f"z={info['z']}")
        if info["pos"] in ("fixed", "absolute", "relative"):
            parts.append(info["pos"])
        if info["anim"]:
            parts.append(info["anim"])
        return " ".join(parts)

    async def _probe_blocker(self, aid: str) -> Optional[dict]:
        """Post-click blocker probe for race-condition cases."""
        try:
            return await self.page.evaluate(_PROBE_BLOCKER_JS, aid)
        except Exception:
            return None

    async def check_dismiss(self, aid: str) -> str:
        """Check if a dismiss target is gone. Returns status string."""
        status = await self.page.evaluate(_DISMISS_CHECK_JS, aid)
        if status == "still_visible":
            anim_info = await self.page.evaluate(_GET_ANIMATIONS_JS, aid)
            if anim_info["count"] > 0:
                await self.page.evaluate(_AWAIT_ANIMATIONS_JS, aid)
                status = await self.page.evaluate(_DISMISS_CHECK_JS, aid)
        return status

    async def _cdp_drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        *,
        steps: int = 20,
        hold_ms: int = 80,
    ) -> None:
        """Low-level CDP drag: move to start, mousedown, hold, move to end, mouseup."""
        await self.page.mouse.move(start_x, start_y)
        await self.page.mouse.down()
        try:
            await asyncio.sleep(hold_ms / 1000)
            await self.page.mouse.move(end_x, end_y, steps=steps)
        finally:
            await self.page.mouse.up()

    @staticmethod
    def _extract_code(raw_text: str) -> Optional[str]:
        """Extract a short code-like value from container text, ignoring labels."""
        lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
        for line in reversed(lines):
            if len(line) > 20:
                continue
            match = _CODE_PATTERN.fullmatch(line)
            if match:
                return match.group()
        matches = _CODE_PATTERN.findall(raw_text)
        if matches:
            return matches[-1]
        short_lines = [line for line in lines if len(line) <= 20]
        if short_lines:
            return short_lines[-1]
        return None

    # ── public action methods ──────────────────────────────────────

    async def click_target(self, aid: str, timeout_ms: int = 1_500) -> ActionResult:
        """Click via CDP: _prep(hit_test, check_disabled) → page.mouse.click."""
        prep = await self._prep(aid, hit_test=True, check_disabled=True)
        if isinstance(prep, ActionResult):
            return self._log_action("click", prep, aid=aid)
        print(f"  [FAST_CLICK {aid}] dispatch at ({prep['x']:.0f}, {prep['y']:.0f})")
        await self.page.mouse.click(prep["x"], prep["y"])
        return self._log_action(
            "click",
            ActionResult(success=True, message=f"clicked aid={aid}"),
            aid=aid,
        )

    async def type_target(self, aid: str, text: str, timeout_ms: int = 1_500) -> ActionResult:
        """Click to focus, then type via execCommand('insertText')."""
        prep = await self._prep(aid)
        if isinstance(prep, ActionResult):
            return self._log_action("type_text", prep, aid=aid)
        await self.page.mouse.click(prep["x"], prep["y"])
        result, status = await self._safe_evaluate(_FILL_JS, [aid, text])
        if status == "navigated":
            return self._log_action(
                "type_text",
                ActionResult(
                    success=True,
                    navigation_occurred=True,
                    message=f"page navigated during type aid={aid}",
                    reason_code="context_destroyed",
                ),
                aid=aid,
            )
        if result.get("error"):
            error = result["error"]
            if error == "not_editable":
                tag = result.get("tag", "?")
                return self._log_action(
                    "type_text",
                    ActionResult(
                        success=False,
                        message=f"type failed aid={aid}: element is <{tag}>, not an input or editable",
                        reason_code="type_failed",
                    ),
                    aid=aid,
                )
            return self._log_action(
                "type_text",
                ActionResult(
                    success=False,
                    message=f"type failed aid={aid}: {error}",
                    reason_code="type_failed",
                ),
                aid=aid,
            )
        return self._log_action(
            "type_text",
            ActionResult(success=True, message=f"typed aid={aid}"),
            aid=aid,
        )

    async def scroll_target(
        self, aid: str, pixels: int = 0, timeout_ms: int = 1_500
    ) -> ActionResult:
        """Scroll element into view; optionally scroll WITHIN it by `pixels`."""
        prep = await self._prep(aid)
        if isinstance(prep, ActionResult):
            return self._log_action("scroll", prep, aid=aid, pixels=pixels)
        if pixels != 0:
            await self.page.evaluate(
                """(args) => {
                    const [aid, px] = args;
                    const el = document.querySelector(`[data-cdx-aid="${aid}"]`);
                    if (el) el.scrollTop += px;
                }""",
                [aid, pixels],
            )
            return self._log_action(
                "scroll",
                ActionResult(success=True, message=f"scrolled aid={aid} by {pixels}px"),
                aid=aid, pixels=pixels,
            )
        return self._log_action(
            "scroll",
            ActionResult(success=True, message=f"scrolled to aid={aid}"),
            aid=aid,
        )

    async def scroll_page(self, pixels: int = 500) -> ActionResult:
        """Scroll the page window by `pixels` (positive = down)."""
        try:
            await self.page.evaluate(f"window.scrollBy(0, {pixels})")
            return self._log_action(
                "scroll_page",
                ActionResult(success=True, message=f"scrolled page by {pixels}px"),
                pixels=pixels,
            )
        except Exception as exc:
            return self._log_action(
                "scroll_page",
                ActionResult(
                    success=False,
                    message=f"scroll_page failed: {exc}",
                    reason_code="scroll_page_failed",
                ),
                pixels=pixels,
            )

    async def read_text_target(self, aid: str, timeout_ms: int = 1_500) -> ActionResult:
        """Read live innerText of an element by aid."""
        result = await self.page.evaluate(_READ_JS, aid)
        if result.get("error"):
            return self._log_action(
                "read_text",
                ActionResult(
                    success=False,
                    message=f"read_text failed aid={aid}: {result['error']}",
                    reason_code="read_text_failed",
                ),
                aid=aid,
            )
        return self._log_action(
            "read_text",
            ActionResult(success=True, message=result["text"][:2000]),
            aid=aid,
        )

    async def copy_paste(
        self,
        source_aids: list[str],
        target_aid: str,
        timeout_ms: int = 1_500,
    ) -> ActionResult:
        """Try each source, extract the code value from first hit, type into target."""
        tgt_prep = await self._prep(target_aid)
        if isinstance(tgt_prep, ActionResult):
            return self._log_action(
                "copy_paste",
                ActionResult(
                    success=False,
                    message=f"target {tgt_prep.message}",
                    reason_code="copy_target_failed",
                ),
                source_aids=source_aids, target_aid=target_aid,
            )
        tried: list[str] = []
        for source_aid in source_aids:
            read_result = await self.page.evaluate(_READ_JS, source_aid)
            if read_result.get("error"):
                tried.append(f"{source_aid}:{read_result['error']}")
                continue
            raw_text = read_result["text"]
            code = self._extract_code(raw_text)
            if code is None:
                tried.append(f"{source_aid}:no_code")
                continue
            fill_result = await self.page.evaluate(_FILL_JS, [target_aid, code])
            if fill_result.get("error"):
                return self._log_action(
                    "copy_paste",
                    ActionResult(
                        success=False,
                        message=f"copy_paste fill failed: {fill_result['error']}",
                        reason_code="copy_paste_failed",
                    ),
                    source_aids=source_aids, target_aid=target_aid,
                )
            return self._log_action(
                "copy_paste",
                ActionResult(
                    success=True,
                    message=f"copied '{code}' from {source_aid} to target (tried: {tried})",
                ),
                source_aids=source_aids, target_aid=target_aid,
            )
        return self._log_action(
            "copy_paste",
            ActionResult(
                success=False,
                message=f"no code found in any source (tried: {tried})",
                reason_code="copy_no_code",
            ),
            source_aids=source_aids, target_aid=target_aid,
        )

    async def wait_for_page_settle(self, max_ms: int = 100) -> ActionResult:
        """Light settle action wrapper — caps at max_ms and reports new mutation count.
        Requires `install_mutation_observer()` to have been called this session.
        """
        capped = min(max_ms, 1_000)
        try:
            count = await self.wait_for_settle(quiet_ms=80, max_ms=capped)
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return self._log_action(
                "wait_for_page_settle",
                ActionResult(
                    success=False,
                    message=f"wait_for_page_settle failed: {compact}",
                    reason_code="settle_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                ),
                max_ms=max_ms,
            )
        return self._log_action(
            "wait_for_page_settle",
            ActionResult(success=True, message=f"page settled ({count} mutations within {capped}ms)"),
            max_ms=max_ms,
        )

    async def sleep(self, ms: int = 1000) -> ActionResult:
        """Sleep for a fixed duration (capped at 10s)."""
        capped = min(ms, 10_000)
        await asyncio.sleep(capped / 1000.0)
        return self._log_action(
            "sleep",
            ActionResult(success=True, message=f"slept {capped}ms"),
            ms=ms,
        )

    async def key_press(self, keys: list[str]) -> ActionResult:
        """Press one or more keyboard keys / combos in sequence."""
        if not keys:
            return self._log_action(
                "key_press",
                ActionResult(
                    success=False,
                    message="key_press requires a non-empty keys list",
                    reason_code="key_press_empty",
                ),
            )
        pressed: list[str] = []
        for key in keys:
            try:
                await self.page.keyboard.press(key)
                pressed.append(key)
            except Exception as exc:
                compact, raw, blocker = _compact_pw_error(exc)
                return self._log_action(
                    "key_press",
                    ActionResult(
                        success=False,
                        message=f"key_press failed at '{key}' (pressed: {pressed}): {compact}",
                        reason_code="key_press_failed",
                        raw_log=raw,
                        blocker_aid=blocker,
                    ),
                    keys=keys,
                )
        return self._log_action(
            "key_press",
            ActionResult(success=True, message=f"pressed keys: {', '.join(pressed)}"),
            keys=keys,
        )

    async def click_selector(self, selector: str, timeout_ms: int = 5_000) -> ActionResult:
        """Click by CSS selector — Playwright locator (used for DnD special clicks)."""
        try:
            target = self.page.locator(selector).first
            await target.scroll_into_view_if_needed(timeout=timeout_ms)
            await target.click(timeout=timeout_ms)
            return self._log_action(
                "click_selector",
                ActionResult(success=True, message=f"clicked selector={selector}"),
                selector=selector,
            )
        except Exception as exc:
            return self._log_action(
                "click_selector",
                ActionResult(
                    success=False,
                    message=f"click failed selector={selector}: {exc}",
                ),
                selector=selector,
            )

    async def drag_target(
        self,
        source_aid: str,
        target_aid: str,
        steps: int = 20,
        hold_ms: int = 80,
    ) -> ActionResult:
        """Drag via CDP mouse: _prep both elements → mouse down+move+up."""
        src = await self._prep(source_aid, hit_test=True)
        if isinstance(src, ActionResult):
            return self._log_action(
                "drag", src, source_aid=source_aid, target_aid=target_aid, mode="target",
            )
        tgt = await self._prep(target_aid)
        if isinstance(tgt, ActionResult):
            return self._log_action(
                "drag", tgt, source_aid=source_aid, target_aid=target_aid, mode="target",
            )
        try:
            await self._cdp_drag(
                src["x"], src["y"], tgt["x"], tgt["y"],
                steps=steps, hold_ms=hold_ms,
            )
            return self._log_action(
                "drag",
                ActionResult(success=True, message=f"dragged {source_aid} to {target_aid}"),
                source_aid=source_aid, target_aid=target_aid, mode="target",
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return self._log_action(
                "drag",
                ActionResult(
                    success=False,
                    message=f"drag failed {source_aid} -> {target_aid}: {compact}",
                    reason_code="drag_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                ),
                source_aid=source_aid, target_aid=target_aid, mode="target",
            )

    async def drag_batch_synthetic(self, pairs: list[tuple[str, str]]) -> ActionResult:
        """Batch drag via synthetic DragEvent — all drags in one JS call.

        Executes all (source_aid, target_aid) pairs in a single page.evaluate.
        Doesn't verify drops landed — mutation observer + post-batch V5 are the
        verification layer.
        """
        for src_aid, tgt_aid in pairs:
            src = await self._prep(src_aid, hit_test=True)
            if isinstance(src, ActionResult):
                return self._log_action("drag_batch", src, pairs=pairs)
            tgt = await self._prep(tgt_aid)
            if isinstance(tgt, ActionResult):
                return self._log_action("drag_batch", tgt, pairs=pairs)
        try:
            result = await self.page.evaluate(
                _DRAG_SYNTHETIC_BATCH_JS,
                [[s, t] for s, t in pairs],
            )
            for r in result:
                print(f"  [DRAG_BATCH {r['src']} -> {r['tgt']}] data='{r.get('data', '')}'")
            return self._log_action(
                "drag_batch",
                ActionResult(success=True, message=f"batch dragged {len(pairs)} pieces"),
                pairs=pairs,
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return self._log_action(
                "drag_batch",
                ActionResult(
                    success=False,
                    message=f"batch drag failed: {compact}",
                    reason_code="drag_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                ),
                pairs=pairs,
            )

    async def drag_target_cdp_dispatch(
        self,
        source_aid: str,
        target_aid: str,
        *,
        use_synthetic: bool = True,
    ) -> ActionResult:
        """Drag for html5-native DnD — bypasses mousedown pipeline.

        use_synthetic=True: synthetic DragEvent via page.evaluate (no _prep,
            no scrollIntoView side-effects, fastest).
        use_synthetic=False: CDP Input.dispatchDragEvent (isTrusted=true), needs
            _prep for coordinates + dragend cleanup.
        """
        try:
            if use_synthetic:
                return await self._drag_synthetic_path(source_aid, target_aid)
            return await self._drag_cdp_native_path(source_aid, target_aid)
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return self._log_action(
                "drag_cdp",
                ActionResult(
                    success=False,
                    message=f"drag failed {source_aid} -> {target_aid}: {compact}",
                    reason_code="drag_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                ),
                source_aid=source_aid, target_aid=target_aid, use_synthetic=use_synthetic,
            )

    async def _drag_synthetic_path(self, source_aid: str, target_aid: str) -> ActionResult:
        result = await self.page.evaluate(
            _DRAG_SYNTHETIC_PAIR_JS, [source_aid, target_aid],
        )
        if result.get("error"):
            return self._log_action(
                "drag_cdp",
                ActionResult(
                    success=False,
                    message=f"drag failed {source_aid} -> {target_aid}: {result['error']}",
                    reason_code="drag_failed",
                ),
                source_aid=source_aid, target_aid=target_aid, use_synthetic=True,
            )
        print(
            f"  [DRAG_SYNTH {source_aid} -> {target_aid}]"
            f" data='{result['data']}' after='{result['after']}'"
        )
        return self._log_action(
            "drag_cdp",
            ActionResult(success=True, message=f"dragged {source_aid} to {target_aid}"),
            source_aid=source_aid, target_aid=target_aid, use_synthetic=True,
        )

    async def _drag_cdp_native_path(self, source_aid: str, target_aid: str) -> ActionResult:
        src = await self._prep(source_aid, hit_test=True)
        if isinstance(src, ActionResult):
            return self._log_action(
                "drag_cdp", src,
                source_aid=source_aid, target_aid=target_aid, use_synthetic=False,
            )
        tgt = await self._prep(target_aid)
        if isinstance(tgt, ActionResult):
            return self._log_action(
                "drag_cdp", tgt,
                source_aid=source_aid, target_aid=target_aid, use_synthetic=False,
            )

        piece_data = await self.page.evaluate(_READ_TEXT_CONTENT_JS, source_aid)
        before_text = await self.page.evaluate(_READ_TEXT_CONTENT_JS, target_aid)

        cdp = await self.page.context.new_cdp_session(self.page)
        try:
            await cdp.send("Input.setInterceptDrags", {"enabled": True})
            drag_data = {
                "items": [{"mimeType": "text/plain", "data": piece_data}],
                "dragOperationsMask": 19,
            }
            await cdp.send("Input.dispatchDragEvent", {
                "type": "dragEnter",
                "x": int(src["x"]), "y": int(src["y"]),
                "data": drag_data,
            })
            await cdp.send("Input.dispatchDragEvent", {
                "type": "dragOver",
                "x": int(tgt["x"]), "y": int(tgt["y"]),
                "data": drag_data,
            })
            await cdp.send("Input.dispatchDragEvent", {
                "type": "drop",
                "x": int(tgt["x"]), "y": int(tgt["y"]),
                "data": drag_data,
            })
            await cdp.send("Input.setInterceptDrags", {"enabled": False})
        finally:
            await cdp.detach()

        await self.page.evaluate(_DISPATCH_DRAGEND_JS, source_aid)

        post_text = await self.page.evaluate(_READ_TEXT_TRUNC_JS, target_aid)
        if post_text == before_text:
            # dragend wasn't enough — give drag manager 50ms to reset
            await asyncio.sleep(0.05)

        print(
            f"  [DRAG_CDP {source_aid} -> {target_aid}]"
            f" data='{piece_data}' after='{post_text}'"
        )
        return self._log_action(
            "drag_cdp",
            ActionResult(success=True, message=f"dragged {source_aid} to {target_aid}"),
            source_aid=source_aid, target_aid=target_aid, use_synthetic=False,
        )

    async def drag_slider(
        self,
        aid: str,
        percent: float,
        steps: int = 20,
        hold_ms: int = 50,
    ) -> ActionResult:
        """Drag a slider to target percentage (0-100). Native range: click; ARIA/generic: drag."""
        if not (0 <= percent <= 100):
            return self._log_action(
                "drag_slider",
                ActionResult(
                    success=False,
                    message=f"drag_slider: percent must be 0-100, got {percent}",
                    reason_code="slider_invalid_percent",
                ),
                aid=aid, percent=percent,
            )

        prep = await self._prep(aid, hit_test="corners")
        if isinstance(prep, ActionResult):
            return self._log_action("drag_slider", prep, aid=aid, percent=percent)

        info = await self.page.evaluate(_DETECT_SLIDER_JS, aid)

        if isinstance(info, dict) and info.get("error"):
            return self._log_action(
                "drag_slider",
                ActionResult(
                    success=False,
                    message=f"drag_slider failed aid={aid}: {info['error']}",
                    reason_code="slider_detect_failed",
                ),
                aid=aid, percent=percent,
            )

        slider_type = info["type"]
        track = info["track"]
        orientation = info.get("orientation", "horizontal")
        ratio = percent / 100.0
        if orientation == "horizontal":
            target_x = track["left"] + ratio * track["width"]
            target_y = track["top"] + track["height"] / 2
        else:
            # vertical: 0% at bottom, 100% at top
            target_x = track["left"] + track["width"] / 2
            target_y = track["top"] + track["height"] - ratio * track["height"]

        try:
            if slider_type == "native-range":
                # Native range: click at target position; browser moves thumb
                await self.page.mouse.click(target_x, target_y)
            else:
                thumb = info.get("thumb", track)
                thumb_cx = thumb["left"] + thumb["width"] / 2
                thumb_cy = thumb["top"] + thumb["height"] / 2
                await self._cdp_drag(
                    thumb_cx, thumb_cy, target_x, target_y,
                    steps=steps, hold_ms=hold_ms,
                )

            readback = await self.page.evaluate(_SLIDER_READBACK_JS, aid)
            value_msg = f" (value now: {readback})" if readback is not None else ""
            return self._log_action(
                "drag_slider",
                ActionResult(success=True, message=f"slider {aid} set to {percent}%{value_msg}"),
                aid=aid, percent=percent,
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return self._log_action(
                "drag_slider",
                ActionResult(
                    success=False,
                    message=f"drag_slider failed {aid}: {compact}",
                    reason_code="slider_drag_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                ),
                aid=aid, percent=percent,
            )

    async def drag_offset(
        self,
        aid: str,
        offset_x: int = 0,
        offset_y: int = 0,
        steps: int = 20,
        hold_ms: int = 50,
    ) -> ActionResult:
        """Drag an element by pixel offset from its current center."""
        prep = await self._prep(aid, hit_test=True)
        if isinstance(prep, ActionResult):
            return self._log_action(
                "drag_offset", prep, aid=aid, offset_x=offset_x, offset_y=offset_y,
            )
        target_x = prep["x"] + offset_x
        target_y = prep["y"] + offset_y
        try:
            await self._cdp_drag(
                prep["x"], prep["y"], target_x, target_y,
                steps=steps, hold_ms=hold_ms,
            )
            return self._log_action(
                "drag_offset",
                ActionResult(success=True, message=f"dragged {aid} by ({offset_x}, {offset_y})px"),
                aid=aid, offset_x=offset_x, offset_y=offset_y,
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return self._log_action(
                "drag_offset",
                ActionResult(
                    success=False,
                    message=f"drag_offset failed {aid}: {compact}",
                    reason_code="drag_offset_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                ),
                aid=aid, offset_x=offset_x, offset_y=offset_y,
            )

    async def draw_strokes(
        self, aid: str, strokes: list[list[list[int]]],
    ) -> ActionResult:
        """Draw strokes on element via CDP Input.dispatchMouseEvent.

        Each stroke is a list of [offset_x, offset_y] points relative to the
        element's top-left. mousePressed at first point, mouseMoved through
        intermediates, mouseReleased at last point. All events isTrusted=true.
        """
        prep = await self._prep(aid, hit_test=True)
        if isinstance(prep, ActionResult):
            return self._log_action("draw", prep, aid=aid)

        # Element top-left from _prep center + dimensions
        left = prep["x"] - prep["w"] / 2
        top = prep["y"] - prep["h"] / 2

        # Validate and filter strokes (each needs >= 2 points with >= 2 coords)
        valid_strokes = [
            [p for p in stroke if isinstance(p, list) and len(p) >= 2] for stroke in strokes
        ]
        valid_strokes = [s for s in valid_strokes if len(s) >= 2]
        if not valid_strokes:
            return self._log_action(
                "draw",
                ActionResult(
                    success=False,
                    message=f"draw failed aid={aid}: no valid strokes (each needs >= 2 points)",
                    reason_code="draw_no_valid_strokes",
                ),
                aid=aid,
            )

        # CDP fields match Playwright's crInput.js exactly:
        #   mousePressed:  button="left", buttons=1, clickCount=1
        #   mouseMoved:    button="left", buttons=1
        #   mouseReleased: button="left", buttons=0, clickCount=1
        client = await self.page.context.new_cdp_session(self.page)
        try:
            for stroke in valid_strokes:
                x0 = left + stroke[0][0]
                y0 = top + stroke[0][1]
                await client.send("Input.dispatchMouseEvent", {
                    "type": "mousePressed",
                    "x": x0, "y": y0,
                    "button": "left", "buttons": 1, "clickCount": 1,
                })
                for point in stroke[1:]:
                    px = left + point[0]
                    py = top + point[1]
                    await client.send("Input.dispatchMouseEvent", {
                        "type": "mouseMoved",
                        "x": px, "y": py,
                        "button": "left", "buttons": 1,
                    })
                xn = left + stroke[-1][0]
                yn = top + stroke[-1][1]
                await client.send("Input.dispatchMouseEvent", {
                    "type": "mouseReleased",
                    "x": xn, "y": yn,
                    "button": "left", "buttons": 0, "clickCount": 1,
                })
        finally:
            with suppress(Exception):
                await client.detach()

        total_points = sum(len(s) for s in valid_strokes)
        print(
            f"  [DRAW {aid}] {len(valid_strokes)} strokes,"
            f" {total_points} points via CDP dispatch"
        )
        return self._log_action(
            "draw",
            ActionResult(
                success=True,
                message=f"drew {len(valid_strokes)} strokes on aid={aid} ({total_points} points)",
            ),
            aid=aid,
        )

    async def probe_drop_zones(
        self,
        draggable_aid: str,
        container_aids: list[str],
        steps: int = 5,
        hold_ms: int = 80,
    ) -> ActionResult:
        """Physical drop-zone probe: drag a piece over containers, collect which accept drops.

        Uses _prep + page.mouse to start a real drag, sweep over container
        centers, and detect which elements fire dragover + preventDefault.
        Returns the list of drop-zone AIDs in the message.
        """
        src = await self._prep(draggable_aid, hit_test=True)
        if isinstance(src, ActionResult):
            return self._log_action("probe_drop_zones", src, draggable_aid=draggable_aid)

        # Collect container centers via _prep (no hit-test for targets)
        targets: list[tuple[str, float, float]] = []
        for cid in container_aids:
            tgt = await self._prep(cid)
            if isinstance(tgt, ActionResult):
                continue  # skip unreachable containers
            targets.append((cid, tgt["x"], tgt["y"]))

        if not targets:
            return self._log_action(
                "probe_drop_zones",
                ActionResult(
                    success=False,
                    message="probe_drop_zones: no reachable containers to probe",
                    reason_code="probe_no_targets",
                ),
                draggable_aid=draggable_aid,
            )

        # Patch preventDefault to collect dragover acceptors
        await self.page.evaluate(_PROBE_DROP_SETUP_JS)

        sweep_error: Optional[Exception] = None
        try:
            await self.page.mouse.move(src["x"], src["y"])
            await self.page.mouse.down()
            await asyncio.sleep(hold_ms / 1000)

            for cid, tx, ty in targets:
                await self.page.evaluate(
                    "(aid) => { window.__cdx_current_target = aid; }", cid,
                )
                await self.page.mouse.move(tx, ty, steps=steps)

            await self.page.mouse.up()
        except Exception as exc:
            sweep_error = exc
            with suppress(Exception):
                await self.page.mouse.up()
        finally:
            try:
                drop_zone_aids = await self.page.evaluate(_PROBE_DROP_TEARDOWN_JS)
            except Exception:
                drop_zone_aids = []

        interrupted = (
            f" (sweep interrupted: {type(sweep_error).__name__})" if sweep_error else ""
        )

        if not drop_zone_aids:
            return self._log_action(
                "probe_drop_zones",
                ActionResult(
                    success=True,
                    message=f"probe_drop_zones: 0 drop zones found"
                    f" (no container accepted dragover){interrupted}",
                ),
                draggable_aid=draggable_aid,
            )

        return self._log_action(
            "probe_drop_zones",
            ActionResult(
                success=True,
                message=f"probe_drop_zones: {len(drop_zone_aids)} drop zones found"
                f"{interrupted}: " + ", ".join(drop_zone_aids),
            ),
            draggable_aid=draggable_aid,
        )

    async def hover_target(
        self, aid: str, timeout_ms: int = 1_500, duration_ms: int = 0,
    ) -> ActionResult:
        """Hover via CDP: _prep(hit_test='corners') → page.mouse.move."""
        prep = await self._prep(aid, hit_test="corners")
        if isinstance(prep, ActionResult):
            return self._log_action("hover", prep, aid=aid)
        await self.page.mouse.move(prep["x"], prep["y"])
        if duration_ms > 0:
            await asyncio.sleep(duration_ms / 1000)
        suffix = f" (held {duration_ms}ms)" if duration_ms > 0 else ""
        return self._log_action(
            "hover",
            ActionResult(success=True, message=f"hovered aid={aid}{suffix}"),
            aid=aid,
        )

    async def press_escape(self) -> ActionResult:
        """Press Escape (popup-dismiss helper). Equivalent to key_press(['Escape'])."""
        try:
            await self.page.keyboard.press("Escape")
            return self._log_action(
                "press_escape",
                ActionResult(success=True, message="pressed key=Escape"),
            )
        except Exception as exc:
            return self._log_action(
                "press_escape",
                ActionResult(
                    success=False,
                    message=f"key press failed key=Escape: {exc}",
                ),
            )

    # ── outside-world utilities (chunk 1.7) ────────────────────────
    # Gemini and curl_cffi flow through here so notes can mirror the full I/O
    # and the boundary rule is enforced. Concurrent call_gemini invocations
    # run in true parallel via the genai async client — no internal queue.

    async def call_gemini(
        self,
        *,
        model: str,
        contents: list[Any],
        response_schema: Any | None = None,
        tools: list[Any] | None = None,
        system_instruction: str | None = None,
        thinking_budget: int = 2048,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> Any:
        """Call Gemini. Exactly one of `response_schema` or `tools` MUST be set —
        no freeform text generations.

        - `response_schema=...` → returns parsed JSON dict matching the schema.
          One extra retry with doubled `max_output_tokens` on JSON-decode failure.
        - `tools=[...]` → returns the raw `GenerateContentResponse` so the caller
          can iterate `candidates[0].content.parts` for function_calls / thoughts
          and read `usage_metadata` for token accounting. Used by CU's tool loop.

        Pairs prompt + response in notes via a shared call_id. Retries 3 attempts
        with exponential backoff on transient failure.

        Concurrency: this method does NOT serialize calls. The underlying httpx
        pool inside the genai async client handles concurrent calls.
        """
        if (response_schema is None) == (tools is None):
            raise ValueError(
                "call_gemini requires exactly one of response_schema or tools "
                "(freeform text generations not allowed)"
            )
        call_id = uuid.uuid4().hex[:12]
        notes.log(
            data_type="prompt",
            data=contents,
            call_id=call_id,
            model=model,
            system_instruction=system_instruction,
            mode="schema" if response_schema is not None else "tools",
            thinking_budget=thinking_budget,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        cfg: dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "thinking_config": genai_types.ThinkingConfig(thinking_budget=thinking_budget),
        }
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if response_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = response_schema
        else:
            cfg["tools"] = tools

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = await _gemini.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(**cfg),
                )
                if response_schema is not None:
                    try:
                        result = json.loads(resp.text or "")
                    except json.JSONDecodeError as je:
                        # Truncation recovery — bump budget once and retry.
                        if attempt == 0:
                            cfg["max_output_tokens"] = max(max_output_tokens * 2, 16384)
                            last_err = je
                            continue
                        raise
                else:
                    result = resp                   # tools mode — return raw response
                notes.log(
                    data_type="response",
                    data=result,
                    call_id=call_id,
                    model=model,
                    attempt=attempt,
                    success=True,
                )
                return result
            except Exception as e:
                last_err = e
                notes.log(
                    data_type="response",
                    data={"error_type": type(e).__name__, "detail": repr(e)[:500]},
                    call_id=call_id,
                    model=model,
                    attempt=attempt,
                    success=False,
                )
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"call_gemini failed after 3 attempts: {last_err}") from last_err

    async def make_http_session(
        self, *, impersonate: str = "chrome",
    ) -> cffi_requests.Session:
        """curl_cffi session with cookies + Chrome TLS/JA4 impersonation.

        Used by Phase 5 replay path (tool_executor). Default `impersonate="chrome"`
        auto-tracks the latest version curl_cffi knows about — same setting that
        passed Akamai for cleartrip in our chunk-1.6 proof. Pin with
        `impersonate="chrome131"` etc. if needed.

        Cookies snapshotted from the live Playwright context at call time —
        call again if you need refreshed state mid-session.
        """
        sess = cffi_requests.Session(impersonate=impersonate)  # type: ignore[arg-type]
        if self._page is not None:
            try:
                pw_cookies = await self._page.context.cookies()
                for c in pw_cookies:
                    name = c.get("name")
                    value = c.get("value")
                    if not name or value is None:
                        continue
                    try:
                        sess.cookies.set(
                            name=name,
                            value=value,
                            domain=c.get("domain") or "",
                            path=c.get("path", "/"),
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("cookie sync to http session failed: %s", e)
        return sess

    # ── top-level task dispatch ───────────────────────────────────
    # All external callers (CLI, experiments, parity harness) call
    # `sm.run_task(task)`. sm lazily builds the `Orchestrator` (Phase 3)
    # on first call. The Orchestrator owns PageAgent + ToolExecutor (Phase 5)
    # internally — sm does not construct or pass those.
    #
    # IMPORTANT: do not read `self.task_metadata` here or anywhere in the
    # Orchestrator's call path. It carries `expected_answer` in eval runs;
    # leaking it into any LLM prompt would invalidate the eval.

    async def run_task(self, task: str) -> Any:
        """Drive one task end-to-end. Returns the `PlanningTree` populated by
        the Orchestrator (carries task_exit, success, final_answer, totals,
        journey). Multiple `run_task` calls within one sm session reuse the
        same Orchestrator instance.
        """
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(
                sm=self,
                max_steps=self._max_steps,
                max_turns_per_step=self._max_turns_per_step,
            )
        return await self._orchestrator.run_task(task)

    # ── async context manager ─────────────────────────────────────
    async def __aenter__(self) -> SessionManager:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# ─────────────────────────────────────────────────────────────────
# CLI entry point — `python -m morphnet_v2.session_manager`.
#
# Note: `SessionManager.run_task` lazily builds the Orchestrator, which in
# turn builds PageAgent + PageFilter (and Phase-5 ToolExecutor). The CLI
# only constructs SessionManager — higher layers stay encapsulated.
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    def _str_bool(s: str) -> bool:
        return s.strip().lower() in ("true", "1", "yes", "y")

    p = argparse.ArgumentParser(
        prog="python -m morphnet_v2.session_manager",
        description=(
            "Run one task end-to-end via the full morphnet_v2 pipeline "
            "(SessionManager → Orchestrator → PageAgent / ToolExecutor). "
            "Smoke-test mode if --task is omitted."
        ),
    )
    p.add_argument("--url", required=True, help="Start URL.")
    p.add_argument("--task", default=None,
                   help="Task description (drives CU pipeline if given).")
    p.add_argument("--headless", type=_str_bool, default=False,
                   help="true|false. Default: false (headed).")
    p.add_argument("--port", type=int, default=9222, help="CDP port (default 9222).")
    p.add_argument("--results-dir", default=None,
                   help="If given, notes writes to {results_dir}/{ts}-{site}/.")
    p.add_argument("--max-steps", type=int, default=10,
                   help="Planner step budget (= max branches under the planning tree root).")
    p.add_argument("--max-turns-per-step", type=int, default=60,
                   help="Max LLM turns within one CU step.")
    args = p.parse_args()

    async def _run() -> None:
        print(f"\n=== {args.url} ===")
        if args.task:
            print(f"task: {args.task}")
        async with SessionManager(
            start_url=args.url,
            headless=args.headless,
            port=args.port,
            results_dir=Path(args.results_dir) if args.results_dir else None,
            task_metadata={"task": args.task} if args.task else None,
            max_steps=args.max_steps,
            max_turns_per_step=args.max_turns_per_step,
        ) as sm:
            ready = await sm.wait_for_page_ready()
            print(f"page settled: {ready}")
            if args.task is None:
                return
            tree = await sm.run_task(args.task)
            print(
                f"\n=== TASK DONE ===\n"
                f"  exit:        {tree.task_exit}\n"
                f"  success:     {tree.success}\n"
                f"  steps:       {tree.step_count}\n"
                f"  tokens:      in={tree.total_input_tokens}, out={tree.total_output_tokens}\n"
                f"  final_url:   {tree.final_url}\n"
                f"  final_answer: {tree.final_answer!r}\n"
            )

    asyncio.run(_run())

