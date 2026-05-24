import asyncio
import re
from contextlib import suppress
from typing import Any

from playwright.async_api import Page

from browser_agent.extraction import (
    REFERENCE_GATEWAY_JS,
    ExtractResult,
    build_legacy_reference_gateway_payload,
    build_reference_gateway_payload,
)

from .schemas import ActionResult

# ---------------------------------------------------------------------------
# Shared JS primitives — used by all CDP-dispatched actions
# ---------------------------------------------------------------------------

_PREP_JS = (
    """(args) => {
    const [aid, opts, refCtx] = args;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
    if (!el) return { error: 'not_found' };
    if (!el.checkVisibility({checkVisibilityCSS: true, checkOpacity: true}))
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
            const topAid = __cdxAidForElement(top, refCtx);
            const topZ = parseInt(getComputedStyle(top).zIndex) || 0;
            const topText = (top.textContent || '').trim().slice(0, 60);
            let overlay = top.parentElement;
            while (overlay && overlay !== document.body) {
                const os = getComputedStyle(overlay);
                if (os.position === 'fixed' || os.position === 'absolute') break;
                overlay = overlay.parentElement;
            }
            if (!overlay || overlay === document.body) overlay = top;
            const overlayAid = __cdxAidForElement(overlay, refCtx);
            const overlayZ = parseInt(getComputedStyle(overlay).zIndex) || 0;
            const btns = [...overlay.querySelectorAll('button')].slice(0, 5).map(b => ({
                aid: __cdxAidForElement(b, refCtx),
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
)

_FILL_JS = (
    """(args) => {
    const [aid, text, refCtx] = args;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
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
)

_READ_JS = (
    """(args) => {
    const aid = Array.isArray(args) ? args[0] : args;
    const refCtx = Array.isArray(args) ? args[1] : null;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
    if (!el) return { error: 'not_found' };
    return { text: (el.innerText || '').slice(0, 2000) };
}"""
)

# ---------------------------------------------------------------------------
# Post-action verification JS (unchanged from Phase 1-3)
# ---------------------------------------------------------------------------

_DISMISS_CHECK_JS = (
    """(args) => {
    const aid = Array.isArray(args) ? args[0] : args;
    const refCtx = Array.isArray(args) ? args[1] : null;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
    if (!el) return 'removed';
    if (!document.body.contains(el)) return 'detached';
    if (!el.checkVisibility({checkVisibilityCSS: true, checkOpacity: true}))
        return 'hidden';
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return 'hidden';
    if (r.right < 0 || r.left > window.innerWidth ||
        r.bottom < 0 || r.top > window.innerHeight)
        return 'offscreen';
    return 'still_visible';
}"""
)

_GET_ANIMATIONS_JS = (
    """(args) => {
    const aid = Array.isArray(args) ? args[0] : args;
    const refCtx = Array.isArray(args) ? args[1] : null;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
    if (!el) return { count: 0 };
    const anims = el.getAnimations({subtree: true});
    const running = anims.filter(a => a.playState === 'running');
    return { count: running.length };
}"""
)

_AWAIT_ANIMATIONS_JS = (
    """async (args) => {
    const aid = Array.isArray(args) ? args[0] : args;
    const refCtx = Array.isArray(args) ? args[1] : null;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
    if (!el) return 'no_element';
    const anims = el.getAnimations({subtree: true});
    if (anims.length === 0) return 'no_animations';
    await Promise.race([
        Promise.all(anims.map(a => a.finished.catch(() => null))),
        new Promise(r => setTimeout(r, 500))
    ]);
    return 'finished';
}"""
)

_PROBE_BLOCKER_JS = (
    """(args) => {
    const aid = Array.isArray(args) ? args[0] : args;
    const refCtx = Array.isArray(args) ? args[1] : null;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    if (cx < 0 || cx >= window.innerWidth || cy < 0 || cy >= window.innerHeight)
        return null;
    const top = document.elementFromPoint(cx, cy);
    if (!top || el.contains(top) || top.contains(el)) return null;
    const topAid = __cdxAidForElement(top, refCtx);
    const topZ = parseInt(getComputedStyle(top).zIndex) || 0;
    const topText = (top.textContent || '').trim().slice(0, 60);
    return { topElement: { aid: topAid, z: topZ, text: topText } };
}"""
)

# ---------------------------------------------------------------------------
# Blocker description JS (used by master.py)
# ---------------------------------------------------------------------------

_DESCRIBE_BLOCKER_JS = (
    """(args) => {
    const aid = Array.isArray(args) ? args[0] : args;
    const refCtx = Array.isArray(args) ? args[1] : null;
"""
    + REFERENCE_GATEWAY_JS
    + """
    const el = __cdxResolveAid(aid, refCtx);
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
)


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------


def _compact_pw_error(exc: Exception) -> tuple[str, str, str | None]:
    """Extract a compact message from a Playwright exception.

    Returns (compact_message, full_raw_log, blocker_aid).
    The compact message is what the agent sees; raw_log goes to the trace file.
    blocker_aid is the AID of the intercepting element (or None for non-blocker errors).
    """
    raw = str(exc)
    # Extract the "intercepts pointer events" blocker if present
    blocker_match = re.search(r'data-cdx-aid="(aid-\d+)"[^>]*>.*?intercepts pointer events', raw)
    if blocker_match:
        blocker_aid = blocker_match.group(1)
        return f"blocked by {blocker_aid}", raw, blocker_aid

    if "element is not enabled" in raw:
        return "element is disabled", raw, None

    timeout_match = re.search(r"Timeout (\d+)ms exceeded", raw)
    if timeout_match:
        return f"timeout {timeout_match.group(1)}ms", raw, None

    first_line = raw.split("\n")[0][:200]
    return first_line, raw, None


def _reported_blocker_aid(prep_result: dict[str, Any]) -> str | None:
    """Return the agent-facing blocker owner from a runtime hit-test result."""
    overlay_aid = prep_result.get("overlay_aid")
    if isinstance(overlay_aid, str) and overlay_aid:
        return overlay_aid
    blocker_aid = prep_result.get("blocker_aid")
    if isinstance(blocker_aid, str) and blocker_aid:
        return blocker_aid
    return None


class Executor:
    """Tool executor for browser interactions."""

    def __init__(self) -> None:
        self._cdp_session: Any = None  # persistent CDP session for input dispatch
        self._cdp_page: Page | None = None  # page the session belongs to
        self._extract_result: ExtractResult | None = None

    def set_extract_result(self, extract_result: ExtractResult | None) -> None:
        """Set the active representation context used by aid resolution."""
        self._extract_result = extract_result

    def _reference_gateway_payload(self) -> dict[str, Any]:
        if self._extract_result is None:
            payload = build_legacy_reference_gateway_payload()
        else:
            payload = build_reference_gateway_payload(self._extract_result)
        return payload.model_dump(mode="json")

    async def _get_cdp(self, page: Page) -> Any:
        """Return a persistent CDP session for the given page."""
        if self._cdp_session is None or self._cdp_page is not page:
            if self._cdp_session is not None:
                with suppress(Exception):
                    await self._cdp_session.detach()
            self._cdp_session = await page.context.new_cdp_session(page)
            self._cdp_page = page
        return self._cdp_session

    # ------------------------------------------------------------------
    # Shared prep — find, checkVisibility, scroll, BCR, optional hit-test
    # ------------------------------------------------------------------

    async def _prep(
        self,
        page: Page,
        aid: str,
        *,
        hit_test: bool | str = False,
        check_disabled: bool = False,
    ) -> dict | ActionResult:
        """Shared element prep for all CDP-dispatched actions.

        Returns dict with {x, y, w, h} on success, or ActionResult on error.
        """
        opts: dict = {}
        if hit_test:
            opts["hitTest"] = hit_test
        if check_disabled:
            opts["checkDisabled"] = True
        result = await page.evaluate(_PREP_JS, [aid, opts, self._reference_gateway_payload()])
        if "error" in result:
            error = result["error"]
            if error == "blocked":
                reported_blocker = _reported_blocker_aid(result)
                blocker_label = reported_blocker or "?"
                return ActionResult(
                    success=False,
                    message=f"action failed aid={aid}: blocked by {blocker_label}",
                    reason_code="preflight_blocked",
                    fail_subtype="blocked",
                    blocker_probe=result,
                    blocker_aid=reported_blocker,
                )
            return ActionResult(
                success=False,
                message=f"action failed aid={aid}: {error}",
                reason_code="preflight_failed",
                fail_subtype=error,
            )
        return result

    # ------------------------------------------------------------------
    # Blocker helpers (used by master.py)
    # ------------------------------------------------------------------

    async def _describe_blocker(self, page: Page, aid: str) -> str | None:
        """Return a one-line description of a blocker element, or None if gone."""
        info = await page.evaluate(_DESCRIBE_BLOCKER_JS, [aid, self._reference_gateway_payload()])
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

    async def _probe_blocker(self, page: Page, aid: str) -> dict | None:
        """Post-click blocker probe for race condition cases."""
        try:
            return await page.evaluate(_PROBE_BLOCKER_JS, [aid, self._reference_gateway_payload()])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Dismiss verification (unchanged from Phase 1-3)
    # ------------------------------------------------------------------

    async def check_dismiss(self, page: Page, aid: str) -> str:
        """Check if a dismiss target is gone. Returns status string."""
        status = await page.evaluate(_DISMISS_CHECK_JS, [aid, self._reference_gateway_payload()])
        if status == "still_visible":
            anim_info = await page.evaluate(
                _GET_ANIMATIONS_JS,
                [aid, self._reference_gateway_payload()],
            )
            if anim_info["count"] > 0:
                await page.evaluate(_AWAIT_ANIMATIONS_JS, [aid, self._reference_gateway_payload()])
                status = await page.evaluate(
                    _DISMISS_CHECK_JS,
                    [aid, self._reference_gateway_payload()],
                )
        return status

    # ------------------------------------------------------------------
    # Action methods — all use shared _prep() + CDP dispatch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Shared CDP drag primitive — mousedown → hold → mousemove → mouseup
    # ------------------------------------------------------------------

    async def _cdp_drag(
        self,
        page: Page,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        steps: int = 20,
        hold_ms: int = 80,
    ) -> None:
        """Low-level CDP drag: move to start, mousedown, hold, move to end, mouseup.

        Mouse is always released via try/finally — callers don't need their own
        safety cleanup (double mouse.up on released mouse is a no-op).
        """
        await page.mouse.move(start_x, start_y)
        await page.mouse.down()
        try:
            await asyncio.sleep(hold_ms / 1000)
            await page.mouse.move(end_x, end_y, steps=steps)
        finally:
            await page.mouse.up()

    # ------------------------------------------------------------------
    # Action methods — all use shared _prep() + CDP dispatch
    # ------------------------------------------------------------------

    async def click_target(
        self,
        page: Page,
        aid: str,
        timeout_ms: int = 1_500,
    ) -> ActionResult:
        """Click via CDP dispatch: _prep(hitTest, checkDisabled) → page.mouse.click."""
        prep = await self._prep(page, aid, hit_test=True, check_disabled=True)
        if isinstance(prep, ActionResult):
            return prep
        print(f"  [FAST_CLICK {aid}] dispatch at ({prep['x']:.0f}, {prep['y']:.0f})")
        await page.mouse.click(prep["x"], prep["y"])
        return ActionResult(success=True, message=f"clicked aid={aid}")

    async def type_target(
        self,
        page: Page,
        aid: str,
        text: str,
        timeout_ms: int = 1_500,
    ) -> ActionResult:
        """Click to focus, then type text via execCommand('insertText').

        Clicks the element first (like a human would) to activate focus,
        then inserts text. Returns descriptive error if the element isn't editable.
        """
        prep = await self._prep(page, aid)
        if isinstance(prep, ActionResult):
            return prep
        # Click to focus — a human always clicks before typing
        await page.mouse.click(prep["x"], prep["y"])
        result = await page.evaluate(_FILL_JS, [aid, text, self._reference_gateway_payload()])
        if result.get("error"):
            error = result["error"]
            if error == "not_editable":
                tag = result.get("tag", "?")
                return ActionResult(
                    success=False,
                    message=f"type failed aid={aid}: element is <{tag}>, not an input or editable",
                    reason_code="type_failed",
                )
            return ActionResult(
                success=False,
                message=f"type failed aid={aid}: {error}",
                reason_code="type_failed",
            )
        return ActionResult(success=True, message=f"typed aid={aid}")

    async def select_option(
        self,
        page: Page,
        aid: str,
        value: str,
    ) -> ActionResult:
        """Select an option in a <select> element by visible label text."""
        result = await page.evaluate(
            """(args) => {
                const [aid, value, refCtx] = args;
"""
            + REFERENCE_GATEWAY_JS
            + """
                const el = __cdxResolveAid(aid, refCtx);
                if (!el) return { error: 'not_found' };
                if (el.tagName.toLowerCase() !== 'select')
                    return { error: 'not_select', tag: el.tagName.toLowerCase() };
                for (const opt of el.options) {
                    if (opt.text.trim() === value.trim()) {
                        el.value = opt.value;
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        return { ok: true, selected: opt.value, label: opt.text.trim() };
                    }
                }
                const available = Array.from(el.options).map(o => o.text.trim()).slice(0, 10);
                return { error: 'option_not_found', available };
            }""",
            [aid, value, self._reference_gateway_payload()],
        )
        if result.get("ok"):
            return ActionResult(
                success=True,
                message=f"selected '{result.get('label', value)}' in aid={aid}",
            )
        error = result.get("error", "unknown")
        if error == "option_not_found":
            avail = result.get("available", [])
            return ActionResult(
                success=False,
                message=f"select failed aid={aid}: option '{value}' not found. Available: {avail}",
                reason_code="select_option_not_found",
            )
        return ActionResult(
            success=False,
            message=f"select failed aid={aid}: {error}",
            reason_code="select_failed",
        )

    async def scroll_target(
        self,
        page: Page,
        aid: str,
        pixels: int = 0,
        timeout_ms: int = 1_500,
    ) -> ActionResult:
        """Scroll at an element via CDP mouseWheel dispatch.

        Trusted event — browser handles native scroll and bubbling.
        If the element is scrollable, it scrolls. If not, the event
        bubbles to the nearest scrollable ancestor.
        """
        prep = await self._prep(page, aid)
        if isinstance(prep, ActionResult):
            return prep
        if pixels != 0:
            cdp = await self._get_cdp(page)
            await cdp.send(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseWheel",
                    "x": int(prep["x"]),
                    "y": int(prep["y"]),
                    "deltaX": 0,
                    "deltaY": pixels,
                },
            )
            return ActionResult(success=True, message=f"scrolled aid={aid} by {pixels}px")
        return ActionResult(success=True, message=f"scrolled to aid={aid}")

    async def scroll_page(
        self,
        page: Page,
        pixels: int = 500,
    ) -> ActionResult:
        """Scroll the page window by the given pixels (positive = down)."""
        try:
            await page.evaluate(f"window.scrollBy(0, {pixels})")
            return ActionResult(success=True, message=f"scrolled page by {pixels}px")
        except Exception as exc:
            return ActionResult(
                success=False,
                message=f"scroll_page failed: {exc}",
                reason_code="scroll_page_failed",
            )

    async def read_text_target(
        self,
        page: Page,
        aid: str,
        timeout_ms: int = 1_500,
    ) -> ActionResult:
        """Read the live innerText of an element by aid."""
        result = await page.evaluate(_READ_JS, [aid, self._reference_gateway_payload()])
        if result.get("error"):
            return ActionResult(
                success=False,
                message=f"read_text failed aid={aid}: {result['error']}",
                reason_code="read_text_failed",
            )
        return ActionResult(success=True, message=result["text"][:2000])

    async def copy_paste_click(
        self,
        page: Page,
        source_aid: str,
        pattern: str,
        target_aid: str,
        click_aid: str,
    ) -> ActionResult:
        """Read source innerText, find regex matches, try each: type into target, click button.

        Reports honestly what happened. Navigation is not this tool's concern —
        the nav listener handles that at the session level.
        """
        # Read source
        read_result = await page.evaluate(_READ_JS, [source_aid, self._reference_gateway_payload()])
        if read_result.get("error"):
            return ActionResult(
                success=False,
                message=f"source {source_aid}: {read_result['error']}",
                reason_code="copy_source_failed",
            )
        raw_text = read_result["text"]

        # Find matches
        try:
            matches = re.findall(pattern, raw_text)
        except re.error as e:
            return ActionResult(
                success=False,
                message=f"invalid regex pattern: {e}",
                reason_code="copy_bad_pattern",
            )

        if not matches:
            preview = raw_text[:200].replace("\n", " ").strip()
            return ActionResult(
                success=False,
                message=(f'no matches for /{pattern}/ in {source_aid} (source text: "{preview}")'),
                reason_code="copy_no_match",
            )

        # Deduplicate preserving order, cap at 10
        seen: set[str] = set()
        unique: list[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        unique = unique[:10]

        # Try each match: clear → type → click.
        # Nav listener cancels the task if any click triggers navigation.
        # Click blocked → stop and return click_target's result as-is.
        tried: list[str] = []
        for match in unique:
            await page.evaluate(
                _FILL_JS,
                [target_aid, "", self._reference_gateway_payload()],
            )
            fill_result = await page.evaluate(
                _FILL_JS,
                [target_aid, match, self._reference_gateway_payload()],
            )
            if fill_result.get("error"):
                tried.append(f"'{match}': fill failed")
                continue

            click_result = await self.click_target(page, click_aid)
            if not click_result.success:
                return click_result
            tried.append(f"'{match}': typed and clicked")

        # All matches tried
        return ActionResult(
            success=True,
            message=f"tried {len(tried)} matches: {', '.join(tried)}",
        )

    async def wait_for_page_settle(self, page: Page, max_ms: int = 100) -> ActionResult:
        """Wait until DOM mutations stop, then return."""
        from .mutation_observer import wait_for_settle

        quiet_ms = min(60, max_ms // 2)
        count = await wait_for_settle(page, quiet_ms=quiet_ms, max_ms=max_ms)
        return ActionResult(
            success=True,
            message=f"page settled ({count} mutations in {max_ms}ms window)",
        )

    async def sleep(self, page: Page, ms: int = 1000) -> ActionResult:
        """Sleep for a fixed duration. Use for timed content reveals."""
        # page param unused — kept for API consistency with other executor methods
        capped = min(ms, 10_000)  # hard cap at 10 seconds
        await asyncio.sleep(capped / 1000.0)
        return ActionResult(
            success=True,
            message=f"slept {capped}ms",
        )

    async def key_press(self, page: Page, keys: list[str]) -> ActionResult:
        """Press one or more keyboard keys/combos in sequence."""
        if not keys:
            return ActionResult(
                success=False,
                message="key_press requires a non-empty keys list",
                reason_code="key_press_empty",
            )
        pressed: list[str] = []
        for key in keys:
            try:
                await page.keyboard.press(key)
                pressed.append(key)
            except Exception as exc:
                compact, raw, blocker = _compact_pw_error(exc)
                return ActionResult(
                    success=False,
                    message=f"key_press failed at '{key}' (pressed: {pressed}): {compact}",
                    reason_code="key_press_failed",
                    raw_log=raw,
                    blocker_aid=blocker,
                )
        return ActionResult(success=True, message=f"pressed keys: {', '.join(pressed)}")

    async def click_selector(
        self,
        page: Page,
        selector: str,
        timeout_ms: int = 5_000,
    ) -> ActionResult:
        """Click by CSS selector — keeps Playwright locator (used for DnD special clicks)."""
        try:
            target = page.locator(selector).first
            await target.scroll_into_view_if_needed(timeout=timeout_ms)
            await target.click(timeout=timeout_ms)
            return ActionResult(success=True, message=f"clicked selector={selector}")
        except Exception as exc:
            return ActionResult(
                success=False,
                message=f"click failed selector={selector}: {exc}",
            )

    async def drag_target(
        self,
        page: Page,
        source_aid: str,
        target_aid: str,
        steps: int = 20,
        hold_ms: int = 80,
    ) -> ActionResult:
        """Drag via CDP mouse: _prep both elements → mouse down+move+up."""
        # Prep source with hit-test — fail early if blocked
        src = await self._prep(page, source_aid, hit_test=True)
        if isinstance(src, ActionResult):
            return src

        # Prep target — no hit-test (target may be empty slot, not visually prominent)
        tgt = await self._prep(page, target_aid)
        if isinstance(tgt, ActionResult):
            return tgt

        try:
            await self._cdp_drag(
                page,
                src["x"],
                src["y"],
                tgt["x"],
                tgt["y"],
                steps=steps,
                hold_ms=hold_ms,
            )
            return ActionResult(
                success=True,
                message=f"dragged {source_aid} to {target_aid}",
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return ActionResult(
                success=False,
                message=f"drag failed {source_aid} -> {target_aid}: {compact}",
                reason_code="drag_failed",
                raw_log=raw,
                blocker_aid=blocker,
            )

    async def drag_batch_synthetic(
        self,
        page: Page,
        pairs: list[tuple[str, str]],
    ) -> ActionResult:
        """Batch drag via synthetic DragEvent — all drags in one JS call.

        Executes all (source_aid, target_aid) pairs in a single page.evaluate.
        No CDP sessions, no drag manager state issues. Does not verify whether
        drops landed — the mutation observer and post-batch V5 are the
        verification layer.
        """
        # Prep all pairs first — fail fast on blocked/missing elements
        for src_aid, tgt_aid in pairs:
            src = await self._prep(page, src_aid, hit_test=True)
            if isinstance(src, ActionResult):
                return src
            tgt = await self._prep(page, tgt_aid)
            if isinstance(tgt, ActionResult):
                return tgt

        try:
            result = await page.evaluate(
                """(args) => {
                    const [pairs, refCtx] = args;
"""
                + REFERENCE_GATEWAY_JS
                + """
                    const results = [];
                    for (const [srcAid, tgtAid] of pairs) {
                        const src = __cdxResolveAid(srcAid, refCtx);
                        const tgt = __cdxResolveAid(tgtAid, refCtx);
                        if (!src || !tgt) {
                            results.push({src: srcAid, tgt: tgtAid, ok: false, err: 'not_found'});
                            continue;
                        }
                        const data = src.textContent.trim();
                        const dt = new DataTransfer();
                        dt.setData('text/plain', data);

                        src.dispatchEvent(new DragEvent('dragstart', {
                            dataTransfer: dt, bubbles: true, cancelable: true
                        }));
                        tgt.dispatchEvent(new DragEvent('dragover', {
                            dataTransfer: dt, bubbles: true, cancelable: true
                        }));
                        tgt.dispatchEvent(new DragEvent('drop', {
                            dataTransfer: dt, bubbles: true, cancelable: true
                        }));
                        src.dispatchEvent(new DragEvent('dragend', {
                            dataTransfer: dt, bubbles: true
                        }));

                        results.push({src: srcAid, tgt: tgtAid, data});
                    }
                    return results;
                }""",
                [[[s, t] for s, t in pairs], self._reference_gateway_payload()],
            )

            for r in result:
                print(f"  [DRAG_BATCH {r['src']} -> {r['tgt']}] data='{r.get('data', '')}'")

            return ActionResult(
                success=True,
                message=f"batch dragged {len(pairs)} pieces",
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return ActionResult(
                success=False,
                message=f"batch drag failed: {compact}",
                reason_code="drag_failed",
                raw_log=raw,
                blocker_aid=blocker,
            )

    async def drag_target_cdp_dispatch(
        self,
        page: Page,
        source_aid: str,
        target_aid: str,
        *,
        use_synthetic: bool = True,
    ) -> ActionResult:
        """Drag for html5-native DnD — bypasses mousedown pipeline.

        When use_synthetic=True (Tier 1 probe confirmed isTrusted=false accepted):
          Dispatches synthetic DragEvent directly via page.evaluate. No _prep
          needed — uses querySelector, not coordinates. No scrollIntoView side
          effects that could interfere with the page's DOM update cycle.
        When use_synthetic=False (page checks isTrusted):
          Uses CDP Input.dispatchDragEvent (isTrusted=true). Needs _prep for
          coordinates, dragend cleanup, and conditional sleep between drags.
        """
        try:
            if use_synthetic:
                # Synthetic path: no _prep, no scrollIntoView, no coordinates.
                # Everything happens in one synchronous JS call.
                result = await page.evaluate(
                    """(args) => {
                        const [srcAid, tgtAid, refCtx] = args;
"""
                    + REFERENCE_GATEWAY_JS
                    + """
                        const src = __cdxResolveAid(srcAid, refCtx);
                        const tgt = __cdxResolveAid(tgtAid, refCtx);
                        if (!src) return { error: 'source_not_found' };
                        if (!tgt) return { error: 'target_not_found' };

                        const data = src.textContent.trim();
                        const dt = new DataTransfer();
                        dt.setData('text/plain', data);

                        src.dispatchEvent(new DragEvent('dragstart', {
                            dataTransfer: dt, bubbles: true, cancelable: true
                        }));
                        tgt.dispatchEvent(new DragEvent('dragover', {
                            dataTransfer: dt, bubbles: true, cancelable: true
                        }));
                        tgt.dispatchEvent(new DragEvent('drop', {
                            dataTransfer: dt, bubbles: true, cancelable: true
                        }));
                        src.dispatchEvent(new DragEvent('dragend', {
                            dataTransfer: dt, bubbles: true
                        }));

                        return { data, after: tgt.textContent.trim().slice(0, 30) };
                    }""",
                    [source_aid, target_aid, self._reference_gateway_payload()],
                )

                if result.get("error"):
                    return ActionResult(
                        success=False,
                        message=f"drag failed {source_aid} -> {target_aid}: {result['error']}",
                        reason_code="drag_failed",
                    )

                print(
                    f"  [DRAG_SYNTH {source_aid} -> {target_aid}]"
                    f" data='{result['data']}' after='{result['after']}'"
                )
                return ActionResult(
                    success=True,
                    message=f"dragged {source_aid} to {target_aid}",
                )
            else:
                # CDP dispatch path — isTrusted=true, needs coordinates
                src = await self._prep(page, source_aid, hit_test=True)
                if isinstance(src, ActionResult):
                    return src
                tgt = await self._prep(page, target_aid)
                if isinstance(tgt, ActionResult):
                    return tgt

                piece_data = await page.evaluate(
                    """(aid) => {
                        const [targetAid, refCtx] = aid;
"""
                    + REFERENCE_GATEWAY_JS
                    + """
                        const el = __cdxResolveAid(targetAid, refCtx);
                        return el ? el.textContent.trim() : '';
                    }""",
                    [source_aid, self._reference_gateway_payload()],
                )
                before_text = await page.evaluate(
                    """(aid) => {
                        const [targetAid, refCtx] = aid;
"""
                    + REFERENCE_GATEWAY_JS
                    + """
                        const el = __cdxResolveAid(targetAid, refCtx);
                        return el ? el.textContent.trim() : '';
                    }""",
                    [target_aid, self._reference_gateway_payload()],
                )

                cdp = await page.context.new_cdp_session(page)
                try:
                    await cdp.send("Input.setInterceptDrags", {"enabled": True})
                    drag_data = {
                        "items": [{"mimeType": "text/plain", "data": piece_data}],
                        "dragOperationsMask": 19,
                    }
                    await cdp.send(
                        "Input.dispatchDragEvent",
                        {
                            "type": "dragEnter",
                            "x": int(src["x"]),
                            "y": int(src["y"]),
                            "data": drag_data,
                        },
                    )
                    await cdp.send(
                        "Input.dispatchDragEvent",
                        {
                            "type": "dragOver",
                            "x": int(tgt["x"]),
                            "y": int(tgt["y"]),
                            "data": drag_data,
                        },
                    )
                    await cdp.send(
                        "Input.dispatchDragEvent",
                        {
                            "type": "drop",
                            "x": int(tgt["x"]),
                            "y": int(tgt["y"]),
                            "data": drag_data,
                        },
                    )
                    await cdp.send("Input.setInterceptDrags", {"enabled": False})
                finally:
                    await cdp.detach()

                # Synthetic dragend to reset drag manager
                await page.evaluate(
                    """(aid) => {
                        const [targetAid, refCtx] = aid;
"""
                    + REFERENCE_GATEWAY_JS
                    + """
                        const el = __cdxResolveAid(targetAid, refCtx);
                        if (el) el.dispatchEvent(new DragEvent('dragend', {bubbles: true}));
                    }""",
                    [source_aid, self._reference_gateway_payload()],
                )

                post_text = await page.evaluate(
                    """(aid) => {
                        const [targetAid, refCtx] = aid;
"""
                    + REFERENCE_GATEWAY_JS
                    + """
                        const el = __cdxResolveAid(targetAid, refCtx);
                        return el ? el.textContent.trim().slice(0, 30) : null;
                    }""",
                    [target_aid, self._reference_gateway_payload()],
                )
                if post_text == before_text:
                    # dragend wasn't enough — give drag manager 50ms to reset
                    await asyncio.sleep(0.05)

                print(
                    f"  [DRAG_CDP {source_aid} -> {target_aid}]"
                    f" data='{piece_data}' after='{post_text}'"
                )
                return ActionResult(
                    success=True,
                    message=f"dragged {source_aid} to {target_aid}",
                )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return ActionResult(
                success=False,
                message=f"drag failed {source_aid} -> {target_aid}: {compact}",
                reason_code="drag_failed",
                raw_log=raw,
                blocker_aid=blocker,
            )

    async def drag_slider(
        self,
        page: Page,
        aid: str,
        percent: float,
        steps: int = 20,
        hold_ms: int = 50,
    ) -> ActionResult:
        """Drag a slider to a target percentage (0-100).

        For native <input type="range">: click directly at the target track position
        (browser moves thumb to click point). For ARIA/generic sliders: drag thumb
        from current position to computed target.
        """
        if not (0 <= percent <= 100):
            return ActionResult(
                success=False,
                message=f"drag_slider: percent must be 0-100, got {percent}",
                reason_code="slider_invalid_percent",
            )

        # Shared prep: visibility, scroll-into-view, blocker detection.
        # hit_test="corners" — slider thumbs are often small, center-only misses.
        prep = await self._prep(page, aid, hit_test="corners")
        if isinstance(prep, ActionResult):
            return prep

        # Detect slider type and geometry via JS (post-prep, element is now in view)
        info = await page.evaluate(
            """(args) => {
                const [aid, refCtx] = args;
"""
            + REFERENCE_GATEWAY_JS
            + """
                const el = __cdxResolveAid(aid, refCtx);
                if (!el) return { error: 'not_found' };

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role');
                const bcr = el.getBoundingClientRect();

                // Native range input
                // Slider classification — keep in sync with page_filter.py (~line 898)
                if (tag === 'input' && el.type === 'range') {
                    return {
                        type: 'native-range',
                        min: el.min !== '' ? +el.min : 0,
                        max: el.max !== '' ? +el.max : 100,
                        step: el.step !== '' ? +el.step : 1,
                        value: +el.value,
                        orientation: bcr.height > bcr.width ? 'vertical' : 'horizontal',
                        track: { left: bcr.left, top: bcr.top,
                                 width: bcr.width, height: bcr.height }
                    };
                }

                // ARIA slider
                if (role === 'slider') {
                    const parent = el.parentElement?.getBoundingClientRect() || bcr;
                    return {
                        type: 'aria-slider',
                        min: +(el.getAttribute('aria-valuemin') || '0'),
                        max: +(el.getAttribute('aria-valuemax') || '100'),
                        value: +(el.getAttribute('aria-valuenow') || '0'),
                        orientation: el.getAttribute('aria-orientation')
                            || (parent.height > parent.width ? 'vertical' : 'horizontal'),
                        track: { left: parent.left, top: parent.top,
                                 width: parent.width, height: parent.height },
                        thumb: { left: bcr.left, top: bcr.top,
                                 width: bcr.width, height: bcr.height }
                    };
                }

                // Generic: treat element as thumb, parent as track
                const parent = el.parentElement?.getBoundingClientRect() || bcr;
                return {
                    type: 'generic',
                    orientation: parent.width >= parent.height ? 'horizontal' : 'vertical',
                    track: { left: parent.left, top: parent.top,
                             width: parent.width, height: parent.height },
                    thumb: { left: bcr.left, top: bcr.top,
                             width: bcr.width, height: bcr.height }
                };
            }""",
            [aid, self._reference_gateway_payload()],
        )

        if isinstance(info, dict) and info.get("error"):
            return ActionResult(
                success=False,
                message=f"drag_slider failed aid={aid}: {info['error']}",
                reason_code="slider_detect_failed",
            )

        slider_type = info["type"]
        track = info["track"]
        orientation = info.get("orientation", "horizontal")
        ratio = percent / 100.0

        if orientation == "horizontal":
            target_x = track["left"] + ratio * track["width"]
            target_y = track["top"] + track["height"] / 2
        else:
            # Vertical: 0% at bottom, 100% at top
            target_x = track["left"] + track["width"] / 2
            target_y = track["top"] + track["height"] - ratio * track["height"]

        try:
            if slider_type == "native-range":
                # Native range: click at target position (browser moves thumb automatically)
                await page.mouse.click(target_x, target_y)
            else:
                # ARIA/generic: drag from thumb center to target
                thumb = info.get("thumb", track)
                thumb_cx = thumb["left"] + thumb["width"] / 2
                thumb_cy = thumb["top"] + thumb["height"] / 2
                await self._cdp_drag(
                    page,
                    thumb_cx,
                    thumb_cy,
                    target_x,
                    target_y,
                    steps=steps,
                    hold_ms=hold_ms,
                )

            # Read back the value for confirmation
            readback = await page.evaluate(
                """(args) => {
                    const [aid, refCtx] = args;
"""
                + REFERENCE_GATEWAY_JS
                + """
                    const el = __cdxResolveAid(aid, refCtx);
                    if (!el) return null;
                    if (el.tagName === 'INPUT') return el.value;
                    return el.getAttribute('aria-valuenow');
                }""",
                [aid, self._reference_gateway_payload()],
            )
            value_msg = f" (value now: {readback})" if readback is not None else ""
            return ActionResult(
                success=True,
                message=f"slider {aid} set to {percent}%{value_msg}",
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return ActionResult(
                success=False,
                message=f"drag_slider failed {aid}: {compact}",
                reason_code="slider_drag_failed",
                raw_log=raw,
                blocker_aid=blocker,
            )

    async def drag_offset(
        self,
        page: Page,
        aid: str,
        offset_x: int = 0,
        offset_y: int = 0,
        steps: int = 20,
        hold_ms: int = 50,
    ) -> ActionResult:
        """Drag an element by a pixel offset from its current center."""
        prep = await self._prep(page, aid, hit_test=True)
        if isinstance(prep, ActionResult):
            return prep

        target_x = prep["x"] + offset_x
        target_y = prep["y"] + offset_y

        try:
            await self._cdp_drag(
                page,
                prep["x"],
                prep["y"],
                target_x,
                target_y,
                steps=steps,
                hold_ms=hold_ms,
            )
            return ActionResult(
                success=True,
                message=f"dragged {aid} by ({offset_x}, {offset_y})px",
            )
        except Exception as exc:
            compact, raw, blocker = _compact_pw_error(exc)
            return ActionResult(
                success=False,
                message=f"drag_offset failed {aid}: {compact}",
                reason_code="drag_offset_failed",
                raw_log=raw,
                blocker_aid=blocker,
            )

    async def draw_strokes(
        self,
        page: Page,
        aid: str,
        strokes: list[list[list[int]]],
    ) -> ActionResult:
        """Draw strokes on an element via CDP Input.dispatchMouseEvent.

        Each stroke is a list of [offset_x, offset_y] points relative to the
        element's top-left corner. For each stroke: mousePressed at first point,
        mouseMoved through intermediates, mouseReleased at last point.

        All events are isTrusted=true (CDP dispatch, not synthetic JS).
        Uses shared _prep() for scroll, visibility, hit-test, and blocker detection.
        """
        # 1. Shared prep: scroll, visibility, blocker detection
        prep = await self._prep(page, aid, hit_test=True)
        if isinstance(prep, ActionResult):
            return prep

        # Derive element top-left from _prep center + dimensions
        left = prep["x"] - prep["w"] / 2
        top = prep["y"] - prep["h"] / 2

        # 2. Validate and filter strokes (each needs >= 2 points with >= 2 coords)
        valid_strokes = [
            [p for p in stroke if isinstance(p, list) and len(p) >= 2] for stroke in strokes
        ]
        valid_strokes = [s for s in valid_strokes if len(s) >= 2]
        if not valid_strokes:
            return ActionResult(
                success=False,
                message=f"draw failed aid={aid}: no valid strokes (each needs >= 2 points)",
                reason_code="draw_no_valid_strokes",
            )

        # 3. Open CDP session and dispatch all strokes
        # CDP fields match Playwright's crInput.js exactly:
        #   mousePressed:  button="left", buttons=1, clickCount=1
        #   mouseMoved:    button="left", buttons=1  (button tracks last-pressed)
        #   mouseReleased: button="left", buttons=0, clickCount=1
        client = await page.context.new_cdp_session(page)
        try:
            for stroke in valid_strokes:
                x0 = left + stroke[0][0]
                y0 = top + stroke[0][1]
                await client.send(
                    "Input.dispatchMouseEvent",
                    {
                        "type": "mousePressed",
                        "x": x0,
                        "y": y0,
                        "button": "left",
                        "buttons": 1,
                        "clickCount": 1,
                    },
                )
                for point in stroke[1:]:
                    px = left + point[0]
                    py = top + point[1]
                    await client.send(
                        "Input.dispatchMouseEvent",
                        {"type": "mouseMoved", "x": px, "y": py, "button": "left", "buttons": 1},
                    )
                xn = left + stroke[-1][0]
                yn = top + stroke[-1][1]
                await client.send(
                    "Input.dispatchMouseEvent",
                    {
                        "type": "mouseReleased",
                        "x": xn,
                        "y": yn,
                        "button": "left",
                        "buttons": 0,
                        "clickCount": 1,
                    },
                )
        finally:
            with suppress(Exception):
                await client.detach()

        total_points = sum(len(s) for s in valid_strokes)
        print(
            f"  [DRAW {aid}] {len(valid_strokes)} strokes, {total_points} points via CDP dispatch"
        )
        return ActionResult(
            success=True,
            message=f"drew {len(valid_strokes)} strokes on aid={aid} ({total_points} points)",
        )

    async def probe_drop_zones(
        self,
        page: Page,
        draggable_aid: str,
        container_aids: list[str],
        steps: int = 5,
        hold_ms: int = 80,
    ) -> ActionResult:
        """Physical drop zone probe: drag a piece over containers, collect which accept drops.

        Uses _prep() + CDP page.mouse to start a real drag, sweep over container centers,
        and detect which elements fire dragover + preventDefault. Returns a list of drop zone
        AIDs in the message.
        """
        # Prep source with hit-test — fail early if blocked
        src = await self._prep(page, draggable_aid, hit_test=True)
        if isinstance(src, ActionResult):
            return src

        # Collect container centers via _prep (no hit-test for targets)
        targets: list[tuple[str, float, float]] = []
        for cid in container_aids:
            tgt = await self._prep(page, cid)
            if isinstance(tgt, ActionResult):
                continue  # skip unreachable containers
            targets.append((cid, tgt["x"], tgt["y"]))

        if not targets:
            return ActionResult(
                success=False,
                message="probe_drop_zones: no reachable containers to probe",
                reason_code="probe_no_targets",
            )

        # Setup interceptor: patch preventDefault to collect dragover acceptors
        await page.evaluate("""() => {
            window.__cdx_drop_targets = [];
            window.__cdx_orig_pd = Event.prototype.preventDefault;
            window.__cdx_current_target = null;
            Event.prototype.preventDefault = function() {
                if (this.type === 'dragover' && window.__cdx_current_target) {
                    window.__cdx_drop_targets.push(window.__cdx_current_target);
                }
                return window.__cdx_orig_pd.call(this);
            };
        }""")

        sweep_error: Exception | None = None
        try:
            # Physical drag: mousedown on source, sweep over targets, mouseup
            await page.mouse.move(src["x"], src["y"])
            await page.mouse.down()
            await asyncio.sleep(hold_ms / 1000)

            for cid, tx, ty in targets:
                # Set current target AID before moving
                await page.evaluate("(aid) => { window.__cdx_current_target = aid; }", cid)
                await page.mouse.move(tx, ty, steps=steps)

            await page.mouse.up()
        except Exception as exc:
            sweep_error = exc
            # Safety: always release mouse
            with suppress(Exception):
                await page.mouse.up()
        finally:
            # Collect results and clean up — page may have crashed/navigated
            try:
                drop_zone_aids = await page.evaluate("""() => {
                    Event.prototype.preventDefault = window.__cdx_orig_pd;
                    const aids = [...new Set(window.__cdx_drop_targets)];
                    delete window.__cdx_drop_targets;
                    delete window.__cdx_orig_pd;
                    delete window.__cdx_current_target;
                    return aids;
                }""")
            except Exception:
                drop_zone_aids = []

        interrupted = f" (sweep interrupted: {type(sweep_error).__name__})" if sweep_error else ""

        if not drop_zone_aids:
            return ActionResult(
                success=True,
                message=f"probe_drop_zones: 0 drop zones found"
                f" (no container accepted dragover){interrupted}",
            )

        return ActionResult(
            success=True,
            message=f"probe_drop_zones: {len(drop_zone_aids)} drop zones found{interrupted}: "
            + ", ".join(drop_zone_aids),
        )

    async def hover_target(
        self, page: Page, aid: str, timeout_ms: int = 1_500, duration_ms: int = 0
    ) -> ActionResult:
        """Hover via CDP: _prep(hitTest='corners') → page.mouse.move."""
        prep = await self._prep(page, aid, hit_test="corners")
        if isinstance(prep, ActionResult):
            return prep
        await page.mouse.move(prep["x"], prep["y"])
        if duration_ms > 0:
            await asyncio.sleep(duration_ms / 1000)
        suffix = f" (held {duration_ms}ms)" if duration_ms > 0 else ""
        return ActionResult(success=True, message=f"hovered aid={aid}{suffix}")

    async def press_escape(self, page: Page) -> ActionResult:
        try:
            await page.keyboard.press("Escape")
            return ActionResult(success=True, message="pressed key=Escape")
        except Exception as exc:
            return ActionResult(
                success=False,
                message=f"key press failed key=Escape: {exc}",
            )
