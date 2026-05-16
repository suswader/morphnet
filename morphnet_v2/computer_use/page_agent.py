"""
morphnet_v2/computer_use/page_agent.py

The CU integration class. Faithful lift of the per-page logic from
`browser-challenge/crawler/master.py:MasterOrchestrator`. Owns:

  - Per-step state: known_aids / rendered_aids / aid_type / container_labels /
    target_labels / container_id_set / cmap / blocker_allowlist / dnd_library /
    synthetic_drag_ok / current_extraction / navigated / action_log / step_results.
  - Label-map building (`_build_label_maps`, `_register_entity`).
  - Reference enrichment (`enrich_aid_refs`, `_resolve_to_visible`, `check_target_blocked`).
  - Live blocking-graph maintenance (`process_mutations`) — keeps the in-memory
    extraction in sync with mutation events between re-extracts. Used by
    `_enrich_unknown_blocker` and downstream feedback enrichment.
  - AX modal probing (`filter_ax_probe_candidate_aids`, `_ax_probe_new_nodes`,
    `apply_ax_modal_signals`, `_build_ax_backend_map`).
  - AX push events (`_ax_enable_push`, `_ax_disable_push`, `drain_ax_updates`,
    `process_ax_updates`) — vestigial per crawler's note: Playwright's
    new_cdp_session doesn't deliver Accessibility.nodesUpdated. v2 routes
    through sm.cdp which COULD deliver them; lifted for parity, treated as
    best-effort. MutationObserver covers the same signals.
  - Blocker probe formatting (`_format_blocker_probe`, `_enrich_unknown_blocker`).
  - Re-extract (`run_fast_re_extract`).
  - Step-ref resolution (`resolve_step_refs` for `$stepN` text refs).
  - Action dispatch — the full `_run_one_action` switch with intent handling.
  - Batch execution — `_execute_batch` with drag-batch pre-scan, per-action
    mutation flush, blocker-retry, navigation guard, post-batch persistence +
    re-extract + episode formatting.
  - Top-level `run_step(user_goal)` orchestration that wires everything up
    and drives `RawSessionRunner`.

All page-touching I/O routes through `sm.*` methods (boundary rule). No
direct Playwright / CDP calls from this module.

Skips vs crawler — none material. Crawler's `_TeeStream`, log-file plumbing,
trace-collector wiring, and per-page-index disk-file naming are session/run
concerns; they don't affect CU correctness and are not lifted into PageAgent.
"""

from __future__ import annotations

import dataclasses
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

from rebrowser_playwright.async_api import Error as PlaywrightError

from morphnet_v2.computer_use.mutation_episode_builder import (
    apply_persistence_results,
    build_episodes,
    format_batch_events,
    reconcile_episodes,
)
from morphnet_v2.mutation_types import summarize_mutations
from morphnet_v2.computer_use.raw_session import RawSessionRunner
from morphnet_v2.computer_use.schemas import (
    BatchResult,
    MutationNodeRef,
    PageFilterOutput,
    PageSnapshot,
    RawMutationRecord,
    StepResult,
)
from morphnet_v2.computer_use.v5_markdown import render_master_markdown_with_meta

if TYPE_CHECKING:
    from morphnet_v2.page_filter import PageFilter
    from morphnet_v2.session_manager import ActionResult, SessionManager


# ---------------------------------------------------------------------------
# Module constants (lifted from crawler/master.py)
# ---------------------------------------------------------------------------

_BUTTON_TAGS = frozenset({"button", "input", "select", "textarea", "a"})


# ---------------------------------------------------------------------------
# NamedTuples — internal result types lifted from crawler
# ---------------------------------------------------------------------------


class _ProbeResult(NamedTuple):
    """Result of formatting a blocker probe from the executor."""
    resolved: bool   # True if blocker identified and agent can act on it
    message: str


class _StepDispatchResult(NamedTuple):
    """Structured result from _run_one_action."""
    text: str         # model-facing string (goes into results[])
    success: bool     # False => batch should stop (unless plausibly dismissed)
    navigated: bool   # True => batch stops, mutation_lines cleared
    raw: str = ""     # raw executor message without step prefix (for $stepN resolution)


# ---------------------------------------------------------------------------
# PageAgent — the CU integration class
# ---------------------------------------------------------------------------


class PageAgent:
    """One CU step driver. Lifted from crawler's MasterOrchestrator per-page logic.

    Lifecycle: instantiate once per session (built lazily by the Phase 3
    Orchestrator). Call `run_step(user_goal)` once per step. Multiple
    `run_step` calls in sequence are valid within one session — per-step state
    resets at the top of each call.
    """

    def __init__(
        self,
        *,
        sm: SessionManager,
        page_filter: PageFilter,
        model: str,
        max_turns: int = 60,
        max_output_tokens: int = 8192,
        thinking_budget: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self._sm = sm
        self._page_filter = page_filter
        self._model = model
        self._max_turns = max_turns
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget
        self._temperature = temperature

        # State mirrored from crawler/master.py:856-911. Reset/refreshed across
        # _build_label_maps() + run_step() resets — see those methods.
        self._current_extraction: Optional[PageFilterOutput] = None
        self._known_aids: set[str] = set()
        self._rendered_aids: set[str] = set()
        self._aid_type: dict[str, str] = {}
        self._container_id_set: set[str] = set()
        self._container_labels: dict[str, str] = {}
        self._target_labels: dict[str, str] = {}
        self._cmap: dict[str, Any] = {}
        self._dnd_library: Optional[str] = None
        self._synthetic_drag_ok: Optional[bool] = None
        self._blocker_allowlist: set[str] = set()
        self._navigated: bool = False
        self._action_log: list[str] = []
        self._step_results: dict[int, str] = {}
        self._batch_step_results: dict[int, str] = {}
        self._batch_step_raw: dict[int, str] = {}
        self._last_action_result: Any = None
        self._failed_action_count: int = 0
        self._drag_batch_ranges: list[tuple[int, int]] = []
        # AX push event state. Buffer is filled by the subscription callback in
        # `_ax_enable_push`. Vestigial in crawler (no events delivered); lifted
        # for parity, drained at batch boundaries in _execute_batch.
        self._ax_update_buffer: list[dict] = []
        self._ax_unsub: Optional[Any] = None  # CDP unsubscribe callable
        self._ax_backend_to_aid: dict[int, str] = {}

    @property
    def last_extraction(self) -> Optional[PageFilterOutput]:
        """The most recent PageFilterOutput (post-batch re-extract or initial).
        Exposed so the planner can compute page fingerprints for loop detection
        (mirrors crawler's `_build_fingerprint` use of `_current_extraction`)."""
        return self._current_extraction

    # ── Top-level: one CU step ──────────────────────────────────────

    async def run_step(self, user_goal: str) -> StepResult:
        """Drive one CU step end-to-end: extract → render → loop → return."""
        # Reset per-step state (crawler does this implicitly via page_index advance
        # in run(); v2 makes it explicit per step).
        self._blocker_allowlist = set()
        self._action_log = []
        self._failed_action_count = 0
        self._navigated = False
        self._step_results = {}
        self._batch_step_results = {}
        self._batch_step_raw = {}
        self._last_action_result = None
        self._ax_update_buffer = []

        # Heavy settle then initial extraction (mirrors crawler's
        # capture_and_extract_on_page in master.py:_run_inner).
        await self._sm.wait_for_page_ready()
        snapshot = PageSnapshot(
            url=self._sm.page.url,
            title=await self._sm.page.title(),
            html=await self._sm.page.content(),
        )
        self._current_extraction = await self._page_filter.run(
            snapshot, aid_allowlist=self._blocker_allowlist or None
        )
        self._dnd_library = self._current_extraction.dnd_library
        self._synthetic_drag_ok = self._current_extraction.synthetic_drag_accepted
        self._build_label_maps()
        await self._sm.install_mutation_observer()
        self._build_ax_backend_map()
        await self._ax_enable_push()

        # Render the initial V5.
        initial_v5, meta = render_master_markdown_with_meta(self._current_extraction)
        self._rendered_aids = cast(set[str], meta.get("rendered_aids") or set())

        # batch_executor closure — chunk 2.3 injects this into RawSessionRunner.
        async def batch_executor(actions: list[dict[str, Any]]) -> BatchResult:
            return await self._execute_batch(actions)

        runner = RawSessionRunner(
            sm=self._sm,
            model=self._model,
            max_turns=self._max_turns,
            batch_executor=batch_executor,
            dnd_library=self._dnd_library,
            max_output_tokens=self._max_output_tokens,
            thinking_budget=self._thinking_budget,
            temperature=self._temperature,
        )
        try:
            result = await runner.run(user_goal, initial_v5)
        finally:
            # Tear down AX push session for this step. Re-enabled on next run_step.
            await self._ax_disable_push()
        return result

    # ── resolve_target ──────────────────────────────────────────────

    @staticmethod
    def resolve_target(raw_id: int | str | None) -> Optional[str]:
        """Convert an element ID to 'aid-N' string. Returns None if invalid.

        Accepts integer (from model tool calls) or 'aid-N' string. Lifted
        verbatim from crawler/master.py:956-967.
        """
        if raw_id is None:
            return None
        if isinstance(raw_id, int):
            return f"aid-{raw_id}"
        if isinstance(raw_id, str) and raw_id.startswith("aid-") and raw_id[4:].isdigit():
            return raw_id
        return None

    # ── Label maps + entity registry ────────────────────────────────

    def _register_entity(
        self,
        aid: str,
        *,
        entity_type: Optional[str] = None,
        text: Optional[str] = None,
        z_index: Optional[int] = None,
        backend_dom_node_id: Optional[int] = None,
        _source: str = "",
    ) -> None:
        """Register or augment an entity. Lifted from crawler/master.py:969-1029.

        Upsert semantics: non-None params overwrite; None params leave existing
        values. On first registration with entity_type=None, defaults to
        "container". Called from:
          - `_build_label_maps()` — extraction-time, always passes entity_type
          - `process_mutations()` — mutation-time, always passes entity_type
          - `_ax_probe_new_nodes()` — AX probe, entity_type=None (augment only)
        """
        is_new = aid not in self._known_aids
        self._known_aids.add(aid)

        if entity_type is not None:
            self._aid_type[aid] = entity_type
        elif aid not in self._aid_type:
            self._aid_type[aid] = "container"

        resolved_type = self._aid_type[aid]

        if resolved_type == "container":
            self._container_id_set.add(aid)

        if resolved_type == "container" and z_index is not None:
            label = (text or "")[:60]
            self._container_labels[aid] = f'z={z_index} "{label}"'

        if text:
            self._target_labels[aid] = text[:60]

        if backend_dom_node_id is not None:
            self._ax_backend_to_aid[backend_dom_node_id] = aid

        if _source:
            verb = "registered" if is_new else "augmented"
            rtype = entity_type or self._aid_type.get(aid, "?")
            parts = [f"  [ENTITY] {verb} {aid} type={rtype} source={_source}"]
            if text:
                parts.append(f'text="{text[:40]}"')
            if z_index is not None:
                parts.append(f"z={z_index}")
            if backend_dom_node_id is not None:
                parts.append(f"backend={backend_dom_node_id}")
            print(" ".join(parts))

    def _build_label_maps(self) -> None:
        """Build container and target label lookups from current extraction.
        Lifted from crawler/master.py:1031-1056."""
        self._container_labels = {}
        self._target_labels = {}
        self._known_aids = set()
        self._aid_type = {}
        self._container_id_set = set()
        self._cmap = {}
        if self._current_extraction is None:
            return
        self._cmap = {c.container_id: c for c in self._current_extraction.containers}
        for ctr in self._current_extraction.containers:
            label = ctr.heading or ctr.summary[:60]
            self._register_entity(
                ctr.container_id,
                entity_type="container",
                text=label,
                z_index=ctr.z_index,
            )
        for btn in self._current_extraction.buttons:
            self._register_entity(btn.button_id, entity_type="button", text=btn.text)
        for form in self._current_extraction.forms:
            for group in form.groups:
                for ctl in group.controls:
                    label = ctl.label or ctl.text or ""
                    self._register_entity(ctl.control_id, entity_type="control", text=label)

    # ── Reference enrichment ────────────────────────────────────────

    def enrich_aid_refs(self, message: str) -> str:
        """Enrich aid-XXX references with human-readable labels.
        Lifted from crawler/master.py:1058-1078."""
        if not message:
            return message

        def _replace(m: re.Match) -> str:
            aid = m.group(0)
            if aid in self._container_id_set:
                resolved = self._resolve_to_visible(aid)
                label = self._container_labels.get(resolved, "")
                if label:
                    return f"{resolved} ({label})"
                return resolved
            label = self._target_labels.get(aid, "")
            if label:
                return f'{aid} ("{label}")'
            return aid

        return re.sub(r"aid-\d+", _replace, message)

    def _resolve_to_visible(self, aid: str) -> str:
        """Walk up parent chain to nearest non-scaffold container.
        Lifted from crawler/master.py:1080-1113."""
        if self._current_extraction is None:
            return aid
        cmap = self._cmap
        current = aid
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            c = cmap.get(current)
            if c is None:
                return current
            if (
                c.text_blocks
                or c.control_refs
                or c.button_refs
                or c.form_refs
                or c.is_blocking
                or c.is_drop_zone
            ):
                return current
            if c.parent_container_id:
                current = c.parent_container_id
            else:
                break
        return aid

    def check_target_blocked(self, target_id: str) -> Optional[str]:
        """Return a reason string if the target is occluded/blocked, else None.
        Lifted from crawler/master.py:1115-1151."""
        if self._current_extraction is None:
            return None
        for btn in self._current_extraction.buttons:
            if btn.button_id == target_id:
                if btn.occlusion.checked and btn.occlusion.is_occluded:
                    blocker = (
                        btn.occlusion.blocker_container_ids[0]
                        if btn.occlusion.blocker_container_ids
                        else "unknown"
                    )
                    blocker = self._resolve_to_visible(blocker)
                    label = self._container_labels.get(blocker, "")
                    if label:
                        return f"{target_id} covered by {blocker} ({label})"
                    return f"{target_id} covered by {blocker}"
                return None
        for form in self._current_extraction.forms:
            for group in form.groups:
                for ctl in group.controls:
                    if ctl.control_id == target_id:
                        if ctl.occlusion.checked and ctl.occlusion.is_occluded:
                            blocker = (
                                ctl.occlusion.blocker_container_ids[0]
                                if ctl.occlusion.blocker_container_ids
                                else "unknown"
                            )
                            blocker = self._resolve_to_visible(blocker)
                            label = self._container_labels.get(blocker, "")
                            if label:
                                return f"{target_id} covered by {blocker} ({label})"
                            return f"{target_id} covered by {blocker}"
                        return None
        return None

    # ── Live blocking-graph maintenance ─────────────────────────────

    def process_mutations(
        self,
        mutations: list[RawMutationRecord],
    ) -> tuple[list[str], set[str], list[str]]:
        """Process mutation types: update live extraction blocking state.
        Returns (new_overlay_aids, gone_aids, unblocked_aids).
        Lifted from crawler/master.py:1153-1254."""
        if self._current_extraction is None:
            return [], set(), []

        gone_aids: set[str] = set()
        for m in mutations:
            if m.op == "node_removed" and m.subject.aid:
                gone_aids.add(m.subject.aid)

        for m in mutations:
            if m.op != "attr_changed":
                continue
            aid = m.subject.aid
            attr = m.attr_field
            new_val = m.attr_after
            is_now_hidden = (
                (attr == "class" and bool(re.search(r"\bhidden\b", new_val or "")))
                or (attr == "hidden" and new_val is not None)
                or (attr == "aria-hidden" and new_val == "true")
            )
            if is_now_hidden and aid:
                gone_aids.add(aid)

        unblocked: list[str] = []
        if gone_aids:
            for ctr in self._current_extraction.containers:
                if ctr.container_id in gone_aids:
                    ctr.is_blocking = False
                    ctr.blocks_container_ids = []

            for btn in self._current_extraction.buttons:
                if btn.occlusion.is_occluded:
                    remaining = [
                        b for b in btn.occlusion.blocker_container_ids if b not in gone_aids
                    ]
                    if len(remaining) < len(btn.occlusion.blocker_container_ids):
                        btn.occlusion.blocker_container_ids = remaining
                        if not remaining:
                            btn.occlusion.is_occluded = False
                            btn.occlusion.blocked_points = 0
                        unblocked.append(btn.button_id)

            for form in self._current_extraction.forms:
                for group in form.groups:
                    for ctl in group.controls:
                        if ctl.occlusion.is_occluded:
                            remaining = [
                                b for b in ctl.occlusion.blocker_container_ids if b not in gone_aids
                            ]
                            if len(remaining) < len(ctl.occlusion.blocker_container_ids):
                                ctl.occlusion.blocker_container_ids = remaining
                                if not remaining:
                                    ctl.occlusion.is_occluded = False
                                    ctl.occlusion.blocked_points = 0
                                unblocked.append(ctl.control_id)

            if unblocked:
                print(f"  [UNBLOCKED] {', '.join(unblocked)} (blocker gone)")

        new_overlay_aids: list[str] = []
        for m in mutations:
            if m.op != "node_added" or not m.subject.aid:
                continue
            aid = m.subject.aid
            z = m.subject.z_index or 0
            text = (m.subject.text_preview or "")[:60]
            tag = m.subject.tag or ""
            role = m.subject.role or ""
            entity_type = "button" if tag in _BUTTON_TAGS or role == "button" else "container"
            self._register_entity(
                aid,
                entity_type=entity_type,
                text=text,
                z_index=z,
                _source="mutation",
            )
            if entity_type == "container":
                new_overlay_aids.append(aid)

        return new_overlay_aids, gone_aids, unblocked

    # ── AX modal probing ────────────────────────────────────────────

    def filter_ax_probe_candidate_aids(self, overlay_aids: list[str]) -> list[str]:
        """Return only AIDs that have been registered (stamped).
        Lifted from crawler/master.py:1260-1266."""
        return [aid for aid in overlay_aids if aid in self._known_aids]

    async def _ax_probe_new_nodes(
        self, aids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Probe AX-candidate nodes for modal/dialog signals.
        Lifted from crawler/master.py:1268-1322 — routes CDP calls through sm.cdp."""
        if not aids:
            return {}
        aids = aids[:5]  # cap to bound worst-case CDP round-trips

        out: dict[str, dict[str, Any]] = {}
        try:
            doc = await self._sm.cdp.send("DOM.getDocument", {"depth": 0})
            root_id = doc["root"]["nodeId"]
            for aid in aids:
                try:
                    result = await self._sm.cdp.send(
                        "DOM.querySelector",
                        {"nodeId": root_id, "selector": f'[data-cdx-aid="{aid}"]'},
                    )
                    if not result.get("nodeId"):
                        continue

                    desc = await self._sm.cdp.send(
                        "DOM.describeNode", {"nodeId": result["nodeId"]},
                    )
                    backend_id = desc["node"]["backendNodeId"]

                    ax_result = await self._sm.cdp.send(
                        "Accessibility.getPartialAXTree",
                        {"backendNodeId": backend_id, "fetchRelatives": False},
                    )
                    nodes = ax_result.get("nodes", [])
                    if not nodes:
                        continue

                    root_node = nodes[0]
                    props = {
                        p["name"]: p.get("value", {}).get("value")
                        for p in root_node.get("properties", [])
                    }
                    out[aid] = {
                        "role": root_node.get("role", {}).get("value"),
                        "name": root_node.get("name", {}).get("value"),
                        "modal": bool(props.get("modal", False)),
                        "ignored": bool(root_node.get("ignored", False)),
                    }
                    self._register_entity(
                        aid, backend_dom_node_id=backend_id, _source="ax_probe",
                    )
                except (PlaywrightError, KeyError, ConnectionError):
                    continue
        except (PlaywrightError, KeyError, ConnectionError):
            return {}
        return out

    def apply_ax_modal_signals(self, ax_map: dict[str, dict[str, Any]]) -> None:
        """Apply AX modal signals to current extraction's containers.
        Lifted from crawler/master.py:1324-1341."""
        if self._current_extraction is None:
            return
        for aid, ax_info in ax_map.items():
            if not ax_info.get("modal"):
                continue
            for ctr in self._current_extraction.containers:
                if ctr.container_id == aid:
                    ctr.ax_modal = True
                    if not ctr.is_blocking:
                        ctr.is_blocking = True
                    break

    # ── AX push events (best-effort — see module docstring) ────────

    async def _ax_enable_push(self) -> None:
        """Enable Accessibility events + subscribe to nodesUpdated.
        Lifted from crawler/master.py:1347-1375. Vestigial — Playwright doesn't
        deliver these in crawler; v2's raw sm.cdp MIGHT deliver them. The
        MutationObserver covers the same signals either way."""
        await self._ax_disable_push()
        try:
            await self._sm.cdp.send("Accessibility.enable")
            self._ax_update_buffer = []

            def _on_nodes_updated(params: dict) -> None:
                nodes = params.get("nodes", [])
                self._ax_update_buffer.extend(nodes)

            self._ax_unsub = self._sm.cdp.subscribe(
                "Accessibility.nodesUpdated", _on_nodes_updated,
            )
        except (PlaywrightError, ConnectionError):
            self._ax_unsub = None

    async def _ax_disable_push(self) -> None:
        """Disable Accessibility events. Best-effort.
        Lifted from crawler/master.py:1377-1385."""
        if self._ax_unsub is not None:
            with suppress(Exception):
                self._ax_unsub()
            self._ax_unsub = None
        with suppress(Exception):
            await self._sm.cdp.send("Accessibility.disable")
        self._ax_update_buffer = []

    def drain_ax_updates(self) -> list[dict]:
        """Drain the nodesUpdated buffer.
        Lifted from crawler/master.py:1387-1393."""
        if not self._ax_update_buffer:
            return []
        pending = list(self._ax_update_buffer)
        self._ax_update_buffer.clear()
        return pending

    def process_ax_updates(self, updates: list[dict]) -> None:
        """Process buffered nodesUpdated events.
        Lifted from crawler/master.py:1395-1452."""
        if self._current_extraction is None or not updates:
            return

        for node in updates:
            backend_id = node.get("backendDOMNodeId")
            if not backend_id:
                continue
            aid = self._ax_backend_to_aid.get(backend_id)
            if not aid:
                continue
            if aid not in self._known_aids:
                continue

            props = {p["name"]: p.get("value", {}).get("value") for p in node.get("properties", [])}
            aid_type = self._aid_type.get(aid)

            if aid_type == "container":
                for ctr in self._current_extraction.containers:
                    if ctr.container_id == aid:
                        modal = bool(props.get("modal", False))
                        if modal and not ctr.ax_modal:
                            ctr.ax_modal = True
                            ctr.is_blocking = True
                        elif not modal and ctr.ax_modal:
                            ctr.ax_modal = False
                        break

            elif aid_type == "control":
                for form in self._current_extraction.forms:
                    found = False
                    for group in form.groups:
                        for ctrl in group.controls:
                            if ctrl.control_id == aid:
                                disabled = bool(props.get("disabled", False))
                                if ctrl.disabled != disabled:
                                    ctrl.disabled = disabled
                                checked_raw = props.get("checked")
                                if checked_raw is not None:
                                    from morphnet_v2.page_filter import _ax_tristate
                                    new_checked = _ax_tristate(
                                        node.get("properties", []), "checked",
                                    )
                                    if ctrl.checked != new_checked:
                                        ctrl.checked = new_checked
                                found = True
                                break
                        if found:
                            break

    def _build_ax_backend_map(self) -> None:
        """Build backendDOMNodeId -> aid mapping from last AX tree fetch.
        Lifted from crawler/master.py:1454-1471."""
        self._ax_backend_to_aid = {}
        if self._page_filter is None:
            return
        last_ax_map = self._page_filter.last_aid_to_ax_map
        if not last_ax_map:
            return
        for aid, ax_node in last_ax_map.items():
            backend_id = ax_node.get("backendDOMNodeId")
            if backend_id:
                self._ax_backend_to_aid[backend_id] = aid

    # ── Blocker probe formatting + unknown-blocker enrichment ──────

    def _format_blocker_probe(self, probe: dict[str, Any]) -> _ProbeResult:
        """Format a blocker probe into an actionable message.
        Lifted from crawler/master.py:1533-1573."""
        overlay_aid = probe.get("overlay_aid") or probe.get("overlayContainer", {}).get("aid")
        blocker_aid = probe.get("blocker_aid") or probe.get("topElement", {}).get("aid")
        overlay_z = probe.get("overlay_z") or probe.get("overlayContainer", {}).get("z", 0)
        top_z = probe.get("z", 0) or probe.get("topElement", {}).get("z", 0)
        top_text = probe.get("text", "")[:40] or probe.get("topElement", {}).get("text", "")[:40]
        buttons = probe.get("buttons", [])

        resolved_aid = overlay_aid or blocker_aid
        if resolved_aid:
            label = self._container_labels.get(resolved_aid, "")
            btn_strs = []
            for b in buttons:
                b_aid = b.get("aid")
                b_text = b.get("text", "")
                if b_aid:
                    btn_strs.append(f'{b_aid} "{b_text}"')
                elif b_text:
                    btn_strs.append(f'"{b_text}"')
            msg = f"blocked by {resolved_aid}"
            if label:
                msg += f" ({label})"
            elif top_text:
                msg += f" z={overlay_z or top_z} {top_text!r}"
            if btn_strs:
                msg += f" — buttons: {', '.join(btn_strs)}"
            self._blocker_allowlist.add(resolved_aid)
            print(f"  [BLOCKER PROBE] resolved=True {resolved_aid} {msg}")
            return _ProbeResult(resolved=True, message=msg)

        msg = f"blocked by unstamped element z={overlay_z or top_z} {top_text!r}"
        print(f"  [BLOCKER PROBE] resolved=False {msg}")
        return _ProbeResult(resolved=False, message=msg)

    async def _enrich_unknown_blocker(self, result: ActionResult) -> ActionResult:
        """If result has a blocker_aid not in _rendered_aids, describe it inline.
        Lifted from crawler/master.py:1646-1653. Crawler's ActionResult is Pydantic
        (uses model_copy); v2's is a @dataclass — use dataclasses.replace."""
        if not result.blocker_aid or result.blocker_aid in self._rendered_aids:
            return result
        desc = await self._sm._describe_blocker(result.blocker_aid)
        if not desc:
            return result
        return dataclasses.replace(result, message=f"{result.message} ({desc})")

    # ── Action log + step refs ──────────────────────────────────────

    def record_action(self, description: str, result: ActionResult) -> int:
        """Append to action log + track failures.
        Lifted from crawler/master.py:1655-1662."""
        step = len(self._action_log) + 1
        self._action_log.append(f"[step {step}] {description} -> {result.message}")
        self._step_results[step] = result.message
        self._last_action_result = result
        if not result.success:
            self._failed_action_count += 1
        return step

    def resolve_step_refs(self, text: str) -> str:
        """Replace $stepN refs in text with the result of step N.
        Lifted from crawler/master.py:1664-1676."""
        def replacer(match: re.Match) -> str:
            n = int(match.group(1))
            raw = self._batch_step_raw.get(n)
            if raw is not None:
                return raw
            formatted = self._batch_step_results.get(n)
            if formatted is not None:
                return formatted
            return match.group(0)
        return re.sub(r"\$step(\d+)", replacer, text)

    # ── Re-extract (post-batch) ─────────────────────────────────────

    async def run_re_extract(self) -> str:
        """Run fresh extraction on the current page with heavy DOM-stability wait,
        return V5 markdown. Lifted from crawler/master.py:1678-1711 (minus the
        disk-file side-effects — notes already mirrors snapshots/V5/extractions).

        Used for initial-extraction-after-navigation scenarios. The post-batch
        path uses `run_fast_re_extract` instead since the observer already saw
        the DOM stabilize.
        """
        if self._page_filter is None:
            return "ERROR: page_filter not initialized"
        await self._sm.wait_for_page_ready()
        title = await self._sm.page.title()
        html = await self._sm.page.content()
        snapshot = PageSnapshot(url=self._sm.page.url, title=title, html=html)
        try:
            extraction = await self._page_filter.run(
                snapshot, aid_allowlist=self._blocker_allowlist or None,
            )
        except Exception as exc:
            return f"ERROR: re-extract failed: {exc}"
        self._current_extraction = extraction
        self._dnd_library = extraction.dnd_library
        self._synthetic_drag_ok = extraction.synthetic_drag_accepted
        self._build_label_maps()
        self._build_ax_backend_map()
        markdown, meta = render_master_markdown_with_meta(extraction)
        self._rendered_aids = cast(set[str], meta.get("rendered_aids") or set())
        await self._sm.install_mutation_observer()
        return markdown

    async def run_fast_re_extract(self) -> str:
        """Re-extract without separate DOM-stability wait.
        Lifted from crawler/master.py:1575-1619 — drops the per-page disk-file
        side-effects (notes already mirrors snapshots/V5/extractions).
        Re-installs the mutation observer after with a fresh baseline."""
        if self._page_filter is None:
            return "ERROR: page_filter not initialized"
        title = await self._sm.page.title()
        html = await self._sm.page.content()
        snapshot = PageSnapshot(url=self._sm.page.url, title=title, html=html)
        try:
            extraction = await self._page_filter.run(
                snapshot, aid_allowlist=self._blocker_allowlist or None,
            )
        except Exception as exc:
            return f"ERROR: re-extract failed: {exc}"
        self._current_extraction = extraction
        self._dnd_library = extraction.dnd_library
        self._synthetic_drag_ok = extraction.synthetic_drag_accepted
        self._build_label_maps()
        markdown, meta = render_master_markdown_with_meta(extraction)
        self._rendered_aids = cast(set[str], meta.get("rendered_aids") or set())

        # Re-install the observer — PageFilter just stamped fresh AIDs, so the
        # observer's internal maxAid is stale (crawler/master.py:1612-1616).
        await self._sm.install_mutation_observer()
        self._build_ax_backend_map()
        return markdown

    # ── Action dispatch (the big switch — _run_one_action) ─────────

    async def _run_one_action(
        self, action: dict[str, Any], step: int, batch_pos: int,
    ) -> _StepDispatchResult:
        """Execute a single action, return structured result.
        Lifted faithfully from crawler/master.py:1719-2098."""
        kind = action.get("action", "")
        raw_target = action.get("target_id")
        target_id = self.resolve_target(raw_target) or ""
        raw_source = action.get("source_id")
        source_id = self.resolve_target(raw_source) or ""
        intent = action.get("intent", "")

        def _t(msg: str) -> str:
            return self.enrich_aid_refs(msg)

        if kind == "click":
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: invalid target_id {raw_target!r}",
                    success=False, navigated=False,
                )
            aid = target_id
            result = await self._sm.click_target(aid)

            if result.blocker_probe:
                probe_result = self._format_blocker_probe(result.blocker_probe)
                self.record_action(f"click {target_id}", result)
                return _StepDispatchResult(
                    text=f"[{batch_pos}] click failed {target_id}: {probe_result.message}",
                    success=False, navigated=False,
                )

            self.record_action(f"click {target_id}", result)

            if intent == "dismiss" and result.success:
                dismiss_status = await self._sm.check_dismiss(aid)
                if dismiss_status in ("removed", "hidden", "offscreen", "detached"):
                    print(f"  [DISMISS OK] {target_id} -> {dismiss_status}")
                    # Synthetic node_removed mutation — update blocking graph.
                    self.process_mutations(
                        [RawMutationRecord(
                            seq=0,
                            batch_id="synthetic",
                            ts_ms=0,
                            op="node_removed",
                            subject=MutationNodeRef(
                                obs_id=aid, aid=aid, tag="synthetic-dismiss",
                            ),
                        )]
                    )
                else:
                    print(f"  [DISMISS MISS] {target_id} -> {dismiss_status}")

            if intent == "navigate" and result.success:
                try:
                    await self._sm.wait_for_navigation(timeout_ms=2000)
                    self._navigated = True
                except Exception:
                    pass

            if result.navigation_occurred:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [CONTEXT DESTROYED — page navigated]",
                    success=True, navigated=True,
                )

            if not self._navigated:
                nav = self._sm.check_navigation()
                if nav:
                    self._navigated = True
                    return _StepDispatchResult(
                        text=f"[{batch_pos}] {_t(result.message)} [NAVIGATION DETECTED -> {nav}]",
                        success=result.success, navigated=True,
                    )

            if self._navigated:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [NAVIGATED]",
                    success=result.success, navigated=True,
                )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "type_text":
            raw_text = action.get("text", "")
            resolved = self.resolve_step_refs(raw_text)
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: invalid target_id {raw_target!r}",
                    success=False, navigated=False,
                )
            aid = target_id
            result = await self._sm.type_target(aid, resolved)
            result = await self._enrich_unknown_blocker(result)
            self.record_action(f"type {target_id} '{resolved}'", result)
            if result.navigation_occurred:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [CONTEXT DESTROYED — page navigated]",
                    success=True, navigated=True,
                )
            nav = self._sm.check_navigation()
            if nav:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [NAVIGATION DETECTED -> {nav}]",
                    success=result.success, navigated=True,
                )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "read_text":
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: invalid target_id {raw_target!r}",
                    success=False, navigated=False,
                )
            aid = target_id
            result = await self._sm.read_text_target(aid)
            self.record_action(f"read_text {target_id}", result)
            return _StepDispatchResult(
                text=f"[{batch_pos}] {result.message}",
                success=result.success, navigated=False, raw=result.message,
            )

        if kind == "copy_paste":
            raw_source_ids = action.get("source_ids", [])
            source_aids: list[str] = []
            for sid in raw_source_ids:
                resolved = self.resolve_target(sid)
                if resolved is not None:
                    source_aids.append(resolved)
            if not source_aids:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: no valid sources in {raw_source_ids}",
                    success=False, navigated=False,
                )
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: invalid target_id {raw_target!r}",
                    success=False, navigated=False,
                )
            result = await self._sm.copy_paste(source_aids, target_id)
            self.record_action(
                f"copy_paste [{','.join(source_aids)}]->{target_id}", result,
            )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "scroll":
            pixels = int(action.get("pixels", 300))
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: invalid target_id {raw_target!r}",
                    success=False, navigated=False,
                )
            aid = target_id
            result = await self._sm.scroll_target(aid, pixels=pixels)
            self.record_action(f"scroll {target_id} {pixels}px", result)
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "scroll_page":
            pixels = int(action.get("pixels", 500))
            result = await self._sm.scroll_page(pixels=pixels)
            self.record_action(f"scroll_page {pixels}px", result)
            return _StepDispatchResult(
                text=f"[{batch_pos}] {result.message}",
                success=result.success, navigated=False,
            )

        if kind == "key_press":
            keys = action.get("keys", [])
            result = await self._sm.key_press(keys)
            self.record_action(f"key_press {keys}", result)
            nav = self._sm.check_navigation()
            if nav:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [NAVIGATION DETECTED -> {nav}]",
                    success=result.success, navigated=True,
                )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "drag":
            if not source_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: drag requires 'source_id'",
                    success=False, navigated=False,
                )
            source_resolved = source_id

            drag_mode = action.get("drag_mode")
            if not drag_mode:
                return _StepDispatchResult(
                    text=(
                        f"[{batch_pos}] ERROR: drag requires 'drag_mode' "
                        "(one of: target, slider, offset)"
                    ),
                    success=False, navigated=False,
                )

            if drag_mode == "slider":
                if "percent" not in action:
                    return _StepDispatchResult(
                        text=f"[{batch_pos}] ERROR: drag_mode 'slider' requires 'percent'",
                        success=False, navigated=False,
                    )
                if target_id:
                    return _StepDispatchResult(
                        text=f"[{batch_pos}] ERROR: drag_mode 'slider' forbids 'target_id'",
                        success=False, navigated=False,
                    )
                result = await self._sm.drag_slider(source_resolved, action["percent"])

            elif drag_mode == "offset":
                dx = action.get("offset_x", 0)
                dy = action.get("offset_y", 0)
                if "offset_x" not in action and "offset_y" not in action:
                    return _StepDispatchResult(
                        text=(
                            f"[{batch_pos}] ERROR: drag_mode 'offset' requires "
                            "offset_x and/or offset_y"
                        ),
                        success=False, navigated=False,
                    )
                if target_id or "percent" in action:
                    return _StepDispatchResult(
                        text=(
                            f"[{batch_pos}] ERROR: drag_mode 'offset' forbids "
                            "'target_id'/'percent'"
                        ),
                        success=False, navigated=False,
                    )
                result = await self._sm.drag_offset(source_resolved, dx, dy)

            elif drag_mode == "target":
                if not target_id:
                    return _StepDispatchResult(
                        text=f"[{batch_pos}] ERROR: drag_mode 'target' requires 'target_id'",
                        success=False, navigated=False,
                    )
                if "percent" in action or "offset_x" in action or "offset_y" in action:
                    return _StepDispatchResult(
                        text=(
                            f"[{batch_pos}] ERROR: drag_mode 'target' forbids "
                            "'percent'/'offset_x'/'offset_y'"
                        ),
                        success=False, navigated=False,
                    )
                if self._dnd_library == "html5-native":
                    result = await self._sm.drag_target_cdp_dispatch(
                        source_resolved, target_id,
                        use_synthetic=bool(self._synthetic_drag_ok),
                    )
                else:
                    result = await self._sm.drag_target(source_resolved, target_id)

            else:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: unknown drag_mode '{drag_mode}'",
                    success=False, navigated=False,
                )

            result = await self._enrich_unknown_blocker(result)
            self.record_action(f"drag {source_id} mode={drag_mode}", result)
            nav = self._sm.check_navigation()
            if nav:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [NAVIGATION DETECTED -> {nav}]",
                    success=result.success, navigated=True,
                )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "draw":
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: draw requires 'target_id'",
                    success=False, navigated=False,
                )
            strokes = action.get("strokes")
            if not strokes or not isinstance(strokes, list):
                return _StepDispatchResult(
                    text=(
                        f"[{batch_pos}] ERROR: draw requires 'strokes' "
                        "(array of stroke paths)"
                    ),
                    success=False, navigated=False,
                )
            result = await self._sm.draw_strokes(target_id, strokes)
            result = await self._enrich_unknown_blocker(result)
            self.record_action(f"draw {target_id} ({len(strokes)} strokes)", result)
            nav = self._sm.check_navigation()
            if nav:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [NAVIGATION DETECTED -> {nav}]",
                    success=result.success, navigated=True,
                )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "hover":
            if not target_id:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: invalid target_id {raw_target!r}",
                    success=False, navigated=False,
                )
            duration_ms = int(action.get("duration", 0))
            result = await self._sm.hover_target(target_id, duration_ms=duration_ms)
            result = await self._enrich_unknown_blocker(result)
            self.record_action(f"hover {target_id}", result)
            nav = self._sm.check_navigation()
            if nav:
                self._navigated = True
                return _StepDispatchResult(
                    text=f"[{batch_pos}] {_t(result.message)} [NAVIGATION DETECTED -> {nav}]",
                    success=result.success, navigated=True,
                )
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "wait_for_page_settle":
            max_ms = int(action.get("max_ms", 100))
            result = await self._sm.wait_for_page_settle(max_ms=max_ms)
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "sleep":
            ms = int(action.get("delay_ms", 1000))
            result = await self._sm.sleep(ms=ms)
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        if kind == "re_extract":
            # Post-batch pipeline always re-extracts, but allow the model to
            # request an explicit one. The actual re-extraction happens via
            # the post-batch path; here we just acknowledge.
            return _StepDispatchResult(
                text=f"[{batch_pos}] re_extract requested (handled post-batch)",
                success=True, navigated=False,
            )

        if kind == "probe_drop_zones":
            if self._dnd_library != "html5-native":
                return _StepDispatchResult(
                    text=(
                        f"[{batch_pos}] ERROR: probe_drop_zones only available "
                        "when dnd_library is 'html5-native'"
                    ),
                    success=False, navigated=False,
                )
            raw_container_ids = action.get("container_ids", [])
            if not source_id or not raw_container_ids:
                return _StepDispatchResult(
                    text=(
                        f"[{batch_pos}] ERROR: probe_drop_zones requires "
                        "'source_id' and 'container_ids'"
                    ),
                    success=False, navigated=False,
                )
            resolved_cids: list[str] = []
            for cid in raw_container_ids:
                resolved = self.resolve_target(cid)
                if resolved is not None:
                    resolved_cids.append(resolved)
            if not resolved_cids:
                return _StepDispatchResult(
                    text=f"[{batch_pos}] ERROR: no valid containers in {raw_container_ids}",
                    success=False, navigated=False,
                )
            result = await self._sm.probe_drop_zones(
                draggable_aid=source_id, container_aids=resolved_cids,
            )
            self.record_action(f"probe_drop_zones {source_id}", result)
            return _StepDispatchResult(
                text=f"[{batch_pos}] {_t(result.message)}",
                success=result.success, navigated=False,
            )

        return _StepDispatchResult(
            text=f"[{batch_pos}] ERROR: unknown action '{kind}'",
            success=False, navigated=False,
        )

    # ── Batch execution (the post-tool-call body) ──────────────────

    async def _execute_batch(self, actions: list[dict[str, Any]]) -> BatchResult:
        """Execute a batch of browser actions. Lifted from crawler's
        `_handle_execute_actions` (master.py:2100-2364)."""
        results: list[str] = []
        batch_navigated = False
        last_executed_kind = ""
        self._batch_step_results = {}
        self._batch_step_raw = {}
        all_batch_mutations: list[RawMutationRecord] = []
        n_executed = 0
        n_failed = 0
        last_batch_clean = True  # vacuously True if no actions execute

        # Phase 0: pre-scan drag-batch ranges for html5-native + synthetic-OK pages.
        self._drag_batch_ranges = []
        if self._dnd_library == "html5-native" and self._synthetic_drag_ok:
            i = 0
            while i < len(actions):
                if (
                    actions[i].get("action") == "drag"
                    and actions[i].get("drag_mode") == "target"
                ):
                    j = i + 1
                    while j < len(actions) and (
                        actions[j].get("action") == "drag"
                        and actions[j].get("drag_mode") == "target"
                    ):
                        j += 1
                    if j - i > 1:
                        self._drag_batch_ranges.append((i, j))
                    i = j
                else:
                    i += 1
            if self._drag_batch_ranges:
                print(
                    f"  [DRAG_BATCH] grouping "
                    f"{sum(e - s for s, e in self._drag_batch_ranges)} drags "
                    f"into {len(self._drag_batch_ranges)} batch(es)"
                )

        drag_batch_done: set[int] = set()
        step_result: Optional[_StepDispatchResult] = None

        for i, action in enumerate(actions):
            if i in drag_batch_done:
                continue

            # Drag-batch path.
            batch_range = next(
                ((s, e) for s, e in self._drag_batch_ranges if s == i), None,
            )
            if batch_range is not None:
                s, e = batch_range
                pairs: list[tuple[str, str]] = []
                for k in range(s, e):
                    a = actions[k]
                    src_aid = self.resolve_target(a.get("source_id"))
                    tgt_aid = self.resolve_target(a.get("target_id"))
                    if src_aid and tgt_aid:
                        pairs.append((src_aid, tgt_aid))

                await self._sm.mark_mutation_step(i + 1)
                result = await self._sm.drag_batch_synthetic(pairs)
                step = len(self._action_log) + 1
                self.record_action(f"drag_batch {len(pairs)} pairs", result)
                result_str = f"[{i + 1}-{e}] {result.message}"
                results.append(result_str)
                n_executed += len(pairs)
                if not result.success:
                    n_failed += len(pairs)
                    last_batch_clean = False
                for k in range(s, e):
                    drag_batch_done.add(k)

                try:
                    mutations = await self._sm.flush_mutations()
                    if mutations:
                        all_batch_mutations.extend(mutations)
                        summary = summarize_mutations(mutations)
                        print(f"  [MUTATIONS after step {step}] ({len(mutations)} new)")
                        for line in summary.split("\n"):
                            print(f"    {line}")
                        new_overlay_aids, _gone, _unblocked = self.process_mutations(mutations)
                        if new_overlay_aids:
                            ax_candidates = self.filter_ax_probe_candidate_aids(new_overlay_aids)
                            if ax_candidates:
                                ax_map = await self._ax_probe_new_nodes(ax_candidates)
                                if ax_map:
                                    self.apply_ax_modal_signals(ax_map)
                    ax_updates = self.drain_ax_updates()
                    if ax_updates:
                        self.process_ax_updates(ax_updates)
                except (PlaywrightError, KeyError, ConnectionError):
                    pass

                self._step_results[step] = result_str
                self._batch_step_results[i + 1] = result_str
                last_executed_kind = "drag"

                if not result.success:
                    if result.blocker_aid:
                        self._blocker_allowlist.add(result.blocker_aid)
                    remaining = len(actions) - e
                    if remaining > 0:
                        results.append(
                            f"[batch stopped — drag batch failed, "
                            f"{remaining} remaining skipped]"
                        )
                    break
                continue

            # Single action path.
            step = len(self._action_log) + 1
            batch_pos = i + 1
            self._last_action_result = None
            await self._sm.mark_mutation_step(batch_pos)
            self._sm.url_before_action = self._sm.page.url
            try:
                step_result = await self._run_one_action(action, step, batch_pos)
            except PlaywrightError as exc:
                err = str(exc)
                kind_str = action.get("action", "?")
                tid = action.get("target_id", "?")
                if "context was destroyed" in err.lower():
                    self._navigated = True
                    print(f"  [CONTEXT DESTROYED in {kind_str} {tid}] {err[:120]}")
                    step_result = _StepDispatchResult(
                        text=f"[{batch_pos}] {kind_str} {tid} — page navigated mid-action",
                        success=True, navigated=True,
                    )
                else:
                    print(f"  [PLAYWRIGHT ERROR in {kind_str} {tid}] {err[:200]}")
                    step_result = _StepDispatchResult(
                        text=f"[{batch_pos}] {kind_str} {tid} failed: {err[:120]}",
                        success=False, navigated=False,
                    )
            result_str = step_result.text
            results.append(result_str)
            n_executed += 1
            if not step_result.success:
                n_failed += 1
                last_batch_clean = False
            kind = action.get("action", "")

            # Blocker-aware retry — click actions only, one shot.
            if kind == "click":
                failed_result = self._last_action_result
                if failed_result and not failed_result.success:
                    if failed_result.blocker_aid:
                        blocker_before_retry = failed_result.blocker_aid
                        self._action_log.pop()
                        self._failed_action_count -= 1
                        await self._sm.mark_mutation_step(batch_pos)
                        self._sm.url_before_action = self._sm.page.url
                        try:
                            step_result = await self._run_one_action(action, step, batch_pos)
                        except PlaywrightError as exc:
                            err = str(exc)
                            tid = action.get("target_id", "?")
                            if "context was destroyed" in err.lower():
                                self._navigated = True
                                step_result = _StepDispatchResult(
                                    text=f"[{batch_pos}] click {tid} — page navigated during retry",
                                    success=True, navigated=True,
                                )
                            else:
                                step_result = _StepDispatchResult(
                                    text=f"[{batch_pos}] click {tid} retry failed: {err[:120]}",
                                    success=False, navigated=False,
                                )
                        result_str = step_result.text
                        results[-1] = result_str
                        if step_result.success:
                            n_failed -= 1
                            last_batch_clean = (n_failed == 0)
                        print(
                            f"  [RETRY FIRE {action.get('target_id', '')}] "
                            f"blocker_aid={blocker_before_retry}, retried"
                        )
                    else:
                        print(
                            f"  [RETRY SKIP {action.get('target_id', '')}] "
                            f"blocker_aid=None, no retry"
                        )

            last_executed_kind = kind
            if kind == "re_extract":
                # Re-extract: discard buffered mutations (will be re-rendered).
                with suppress(PlaywrightError, KeyError, ConnectionError):
                    await self._sm.flush_mutations()
            else:
                try:
                    mutations = await self._sm.flush_mutations()
                    if mutations:
                        all_batch_mutations.extend(mutations)
                        summary = summarize_mutations(mutations)
                        print(f"  [MUTATIONS after step {step}] ({len(mutations)} new)")
                        for line in summary.split("\n"):
                            print(f"    {line}")
                        new_overlay_aids, _gone, _unblocked = self.process_mutations(mutations)
                        if new_overlay_aids:
                            ax_candidates = self.filter_ax_probe_candidate_aids(new_overlay_aids)
                            if ax_candidates:
                                ax_map = await self._ax_probe_new_nodes(ax_candidates)
                                if ax_map:
                                    self.apply_ax_modal_signals(ax_map)
                    ax_updates = self.drain_ax_updates()
                    if ax_updates:
                        self.process_ax_updates(ax_updates)
                except (PlaywrightError, KeyError, ConnectionError):
                    pass

            self._step_results[step] = result_str
            self._batch_step_results[batch_pos] = result_str
            if step_result.raw:
                self._batch_step_raw[batch_pos] = step_result.raw
            remaining = len(actions) - i - 1

            if step_result.navigated:
                batch_navigated = True
                all_batch_mutations.clear()
                if remaining > 0:
                    results.append(
                        f"[batch stopped — navigated, {remaining} remaining skipped]"
                    )
                break

            if not step_result.success:
                if self._last_action_result and self._last_action_result.blocker_aid:
                    self._blocker_allowlist.add(self._last_action_result.blocker_aid)
                if remaining > 0:
                    results.append(
                        f"[batch stopped — action failed, {remaining} remaining skipped]"
                    )
                break

        # Phase 2: post-batch — persistence → re-extract → episodes → format.
        batch_ended_with_re_extract = last_executed_kind == "re_extract"
        disconnected_ids: list[str] = []
        new_v5: Optional[str] = None
        if not batch_navigated and not batch_ended_with_re_extract:
            if all_batch_mutations:
                disconnected_ids = await self._sm.check_persistence()
                await self._sm.clear_observer_refs()

            try:
                post_batch_md = await self.run_fast_re_extract()
                if not post_batch_md.startswith("ERROR"):
                    new_v5 = post_batch_md
            except Exception as exc:
                print(f"  [WARN] post-batch re-extract failed: {exc}")

            if all_batch_mutations:
                episodes = build_episodes(all_batch_mutations)
                if episodes:
                    final_step = len(self._action_log)
                    if disconnected_ids:
                        apply_persistence_results(episodes, disconnected_ids, final_step)
                    reconcile_episodes(episodes, self._known_aids)
                    during_batch = format_batch_events(episodes)
                    if during_batch:
                        results.append(f"\n{during_batch}")

        executed = len([r for r in results if not r.startswith("[batch stopped")])
        total = len(actions)
        skipped = total - executed
        if skipped > 0:
            stop_reason = (
                "navigated"
                if step_result is not None and step_result.navigated
                else "failed"
            )
            print(
                f"  [BATCH] {total} planned, {executed} executed, "
                f"{skipped} skipped ({stop_reason})"
            )
        else:
            print(f"  [BATCH] {total} planned, {executed} executed")

        return BatchResult(
            text="\n".join(results),
            navigated=batch_navigated,
            new_v5=new_v5,
            n_actions_executed=n_executed,
            n_actions_failed=n_failed,
            last_batch_clean=last_batch_clean,
        )