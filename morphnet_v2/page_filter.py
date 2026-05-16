"""
morphnet_v2/page_filter.py

The bridge from "Chrome is open" → structured page representation. Faithfully
lifted from browser-challenge/crawler/page_filter.py + crawler/schemas.py +
crawler/config.py (ExtractorTuning).

Two boundary deltas vs crawler (everything else byte-for-byte):
  1. Takes `sm: SessionManager` instead of `page: Page`.
  2. Routes `Accessibility.getFullAXTree` + `DOM.getDocument` through `sm.cdp`
     (the same page-target WebSocket crawler opens via
     `page.context.new_cdp_session(page)`), so no separate CDP session per call.

JS payloads (`_collect_payload`, `_collect_blocking_relations`,
`_collect_occlusion`) still go through `sm.page.evaluate(...)` — `sm.page` is
owned by sm, so this is boundary-clean.

Output: PageFilterOutput. Downstream — V5 markdown (chunk 2.2), CU input
(chunk 2.3), tool_builder element refs (Phase 4) — all consume this.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse
import json as _json_dbg

from lorem_text import lorem as lorem_gen
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, ConfigDict, Field

from . import notes

if TYPE_CHECKING:
    from morphnet_v2.session_manager import SessionManager


# ═════════════════════════════════════════════════════════════════════════════
# Schemas — lifted from crawler/schemas.py (PageFilter-consumed entities only)
# Mutation/episode schemas are out of scope (chunk 2.2).
# ═════════════════════════════════════════════════════════════════════════════

AXTristate = Literal["checked", "unchecked", "mixed"]
AXExpandedState = Literal["expanded", "collapsed", "mixed"]


class ActionIntent(StrEnum):
    """Optional intent hint on click actions. Enables smarter post-processing."""

    dismiss = "dismiss"
    navigate = "navigate"


ActionKind = Literal["click_button", "complete_form"]


class PageSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    url: str
    title: str
    html: str


class ViewportGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    x: float
    y: float
    w: float
    h: float


class TargetOcclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    checked: bool
    is_occluded: bool
    blocked_points: int
    total_points: int
    primary_blocker_selector: str | None = None
    primary_blocker_z_index: int | None = None
    blocker_container_ids: list[str] = Field(default_factory=list)
    estimated_overlap_ratio: float | None = None
    occlusion_unknown_until_visible: bool


class FormBlockerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    is_blocked: bool
    blocker_selector: str | None = None
    blocker_container_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class ContainerEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    container_id: str
    selector: str
    tag: str
    role: str | None = None
    parent_container_id: str | None = None
    heading: str | None = None
    summary: str
    text_blocks: list[str] = Field(default_factory=list)
    control_refs: list[str] = Field(default_factory=list)
    form_refs: list[str] = Field(default_factory=list)
    button_refs: list[str] = Field(default_factory=list)
    geometry: ViewportGeometry
    dom_order: int
    z_index: int
    pointer_blocking: bool
    fixed_position: bool
    scrollable: bool
    has_animation: bool
    data_attributes: dict[str, str] = Field(default_factory=dict)
    overlay_like: bool
    section_like: bool
    is_blocking: bool
    blocks_container_ids: list[str] = Field(default_factory=list)
    utility_score: float
    noise_score: float
    reason_codes: list[str] = Field(default_factory=list)
    ax_modal: bool = False
    is_drop_zone: bool = False


class FormControl(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    control_id: str
    owner_container_id: str
    text: str | None = None
    selector: str
    tag: str
    role: str | None = None
    type: str | None = None
    name: str | None = None
    label: str | None = None
    options: list[str] = Field(default_factory=list)
    geometry: ViewportGeometry
    visible_now: bool
    in_viewport_now: bool
    occlusion: TargetOcclusion
    ax_name: str | None = None
    ax_role: str | None = None
    disabled: bool = False
    ax_ignored: bool = False
    checked: AXTristate | None = None
    expanded: AXExpandedState | None = None
    selected: bool = False
    current_value: str | None = None
    has_popup: str | None = None
    ax_description: str | None = None
    focusable: bool = False
    cursor: str | None = None
    draggable: bool = False
    duplicate_id: bool = False
    slider_min: float | None = None
    slider_max: float | None = None
    slider_value: float | None = None
    slider_orientation: Literal["horizontal", "vertical"] | None = None


class FormControlGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    container_id: str
    controls: list[FormControl] = Field(default_factory=list)


class FormEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    form_id: str
    owner_container_id: str
    selector: str
    prompt_heading: str | None = None
    prompt_text: str | None = None
    groups: list[FormControlGroup] = Field(default_factory=list)
    form_blocker: FormBlockerStatus


class ButtonEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    button_id: str
    owner_container_id: str
    text: str | None = None
    selector: str
    href: str | None = None
    geometry: ViewportGeometry
    occlusion: TargetOcclusion
    utility_score: float
    noise_score: float
    reason_codes: list[str] = Field(default_factory=list)
    visible_now: bool
    in_viewport_now: bool
    z_index: int
    ax_name: str | None = None
    disabled: bool = False
    has_popup: str | None = None
    cursor: str | None = None


class ActionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_id: str
    kind: ActionKind
    form_id: str | None = None
    button_id: str | None = None
    priority_score: float
    blocked_now: bool
    blocker_container_id: str | None = None


class PageFilterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    page_url: str
    page_title: str | None = None
    containers: list[ContainerEntity]
    forms: list[FormEntity]
    buttons: list[ButtonEntity]
    actions: list[ActionCandidate]
    # Inputs/textareas/selects that didn't make it into a `form` because they
    # don't cluster with siblings (modern React sites: lone <input> in a <div>,
    # no <form> ancestor). Rendered by v5_markdown as bare controls under
    # their owner container, in the same shape as form-internal controls.
    orphan_controls: list[FormControl] = Field(default_factory=list)
    container_count: int
    important_container_count: int
    form_count: int
    button_count: int
    action_count: int
    blocked_action_count: int
    dropped_button_count: int = 0
    blocking_container_ids: list[str]
    blocking_container_count: int
    has_blocking_containers: bool
    uncertain_items: list[str] = Field(default_factory=list)
    dnd_library: str | None = None
    synthetic_drag_accepted: bool | None = None
    justext_text: str | None = None
    justext_paragraphs: list[str] = Field(default_factory=list)
    justext_paragraph_count: int = 0
    page_epoch: int


# ═════════════════════════════════════════════════════════════════════════════
# ExtractorTuning — lifted from crawler/config.py
# ═════════════════════════════════════════════════════════════════════════════


class ExtractorTuning(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    top_sections: int = 6
    max_buttons_total: int = 40
    max_actions_total: int = 40
    top_snippets: int = 20
    top_overlays: int = 10
    section_node_limit: int = 220
    action_node_limit: int = 260
    control_node_limit: int = 500
    overlay_node_limit: int = 60
    overlay_dismiss_button_limit: int = 24
    section_block_min_chars: int = 6
    action_text_capture_chars: int = 280
    overlay_text_capture_chars: int = 500
    snippet_max_chars: int = 400
    uncertain_band: float = 0.20
    duplicate_summary_min_len: int = 24
    duplicate_summary_overlap_ratio: float = 0.72
    duplicate_summary_containment_ratio: float = 0.55
    duplicate_summary_token_overlap_ratio: float = 0.80
    duplicate_same_heading_token_overlap_ratio: float = 0.68

    max_targets_total: int = 120
    max_form_associated_controls: int = 80
    max_forms_total: int = 20
    max_options_per_form: int = 80
    max_controls_per_form: int = 80
    form_prompt_text_max_chars: int = 180
    form_min_controls_for_container: int = 2
    form_min_non_button_controls_for_container: int = 1
    form_min_non_button_ratio_for_pseudo: float = 0.30
    max_section_target_refs: int = 20
    max_occlusion_targets: int = 60
    occlusion_sample_points: int = 5
    occlusion_inset_px: int = 6

    section_min_score: float = 0.10
    section_semantic_container_bonus: float = 0.24
    section_heading_bonus: float = 0.20
    section_text_len_min: int = 80
    section_text_len_max: int = 2400
    section_text_band_bonus: float = 0.22
    section_text_too_short_threshold: int = 40
    section_text_too_short_penalty: float = 0.10
    section_action_density_step: float = 0.01
    section_action_density_cap: float = 0.15
    section_visible_area_threshold: float = 0.04
    section_visible_area_bonus: float = 0.12
    section_high_link_density_threshold: float = 0.45
    section_high_link_density_penalty: float = 0.16
    section_noise_hit_penalty_step: float = 0.08
    section_noise_hit_penalty_cap: float = 0.30
    section_repeated_template_min_count: int = 3
    section_repeated_template_penalty: float = 0.12
    section_repeated_template_block_suppress_min_count: int = 3
    section_repeated_template_low_info_token_max: int = 20
    section_repeated_template_unique_token_ratio_max: float = 0.55
    section_fixed_penalty: float = 0.10
    block_low_info_token_max: int = 5
    block_low_info_unique_token_ratio_max: float = 0.45
    block_keep_utility_min: float = 0.18
    block_drop_noise_minus_utility_min: float = 0.25

    action_min_utility: float = 0.15
    action_form_control_bonus: float = 0.22
    action_link_bonus: float = 0.14
    action_role_button_bonus: float = 0.06
    action_in_main_bonus: float = 0.14
    action_reasonable_size_min: float = 0.0003
    action_reasonable_size_max: float = 0.25
    action_reasonable_size_bonus: float = 0.10
    action_action_word_bonus: float = 0.24
    action_noise_hit_penalty_step: float = 0.10
    action_noise_hit_penalty_cap: float = 0.45
    action_fixed_small_target_area_threshold: float = 0.003
    action_fixed_small_target_penalty: float = 0.08
    action_non_navigational_href_penalty: float = 0.15
    action_external_domain_penalty: float = 0.08
    action_very_low_text_penalty: float = 0.05

    overlay_candidate_min_area_ratio: float = 0.12
    overlay_candidate_min_z_index: int = 900
    overlay_candidate_min_width_px: int = 180
    overlay_candidate_min_height_px: int = 90
    overlay_hit_point_limit: int = 6
    overlay_blocking_hit_points_for_cta: int = 4
    overlay_blocking_area_ratio: float = 0.20
    overlay_semantic_blocking_min_signals: int = 2
    overlay_semantic_blocking_min_area_ratio: float = 0.08
    overlay_semantic_signal_weight: float = 0.07
    overlay_semantic_signal_cap: float = 0.24
    overlay_hit_test_blocking_bonus: float = 0.40
    overlay_cta_blocked_weight: float = 0.06
    overlay_cta_blocked_cap: float = 0.18
    overlay_center_blocked_bonus: float = 0.10
    overlay_geometric_bonus: float = 0.16
    overlay_blocking_surface_bonus: float = 0.12
    overlay_weak_signal_noise_penalty: float = 0.08
    overlay_weak_signal_area_threshold: float = 0.10
    overlay_pointer_passthrough_penalty: float = 0.12
    overlay_flow_words_bonus: float = 0.20
    overlay_noise_hit_penalty_step: float = 0.10
    overlay_noise_hit_penalty_cap: float = 0.45
    overlay_base_noise: float = 0.25
    overlay_dedupe_iou_threshold: float = 0.70
    overlay_dedupe_text_overlap_threshold: float = 0.65
    overlay_dedupe_z_index_delta: int = 3
    blocking_sample_points: int = 5
    blocking_hit_threshold: float = 0.60
    blocking_min_candidate_area_ratio: float = 0.005


# ═════════════════════════════════════════════════════════════════════════════
# Private atoms (intermediate Python structures during build)
# ═════════════════════════════════════════════════════════════════════════════

_LOREM_VOCAB = frozenset(re.findall(r"[a-zA-Z]+", lorem_gen.words(2000).lower()))
_LOREM_MIN_TOKENS = 6
_LOREM_OVERLAP_THRESHOLD = 0.40


@dataclass(frozen=True)
class _BlockAtom:
    block_id: str
    container_id: str
    text: str
    dom_order: int


@dataclass(frozen=True)
class _ControlAtom:
    control_id: str
    selector: str
    container_id: str
    form_selector: str | None
    tag: str
    role: str | None
    type: str | None
    name: str | None
    label: str | None
    text: str | None
    href: str | None
    options: list[str]
    geometry: ViewportGeometry
    z_index: int
    in_viewport_now: bool
    visible_now: bool
    dom_order: int
    cursor: str | None
    draggable: bool
    duplicate_id: bool
    utility_score: float
    noise_score: float
    reason_codes: list[str]
    ax_name: str | None = None
    ax_role: str | None = None
    disabled: bool = False
    ax_ignored: bool = False
    checked: AXTristate | None = None
    expanded: AXExpandedState | None = None
    selected: bool = False
    current_value: str | None = None
    has_popup: str | None = None
    ax_description: str | None = None
    focusable: bool = False
    slider_min: float | None = None
    slider_max: float | None = None
    slider_value: float | None = None
    slider_orientation: Literal["horizontal", "vertical"] | None = None


logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# AX helpers — normalize raw AXNode properties to schema types
# ═════════════════════════════════════════════════════════════════════════════


def _ax_prop_raw(props: list[dict[str, Any]], name: str) -> Any:
    """Extract raw value from AXNode properties list. Returns None if not found.
    AX properties are: {name, value: {type, value}} where type varies
    (boolean, tristate, token, idref, string, etc.)."""
    for p in props:
        if p.get("name") == name:
            return p.get("value", {}).get("value")
    return None


def _ax_bool(props: list[dict[str, Any]], name: str) -> bool:
    """Normalize an AX property to bool. Handles:
    - boolean type: True/False -> True/False
    - tristate type: "true"/"false"/"mixed" -> True/False/True
    - missing -> False"""
    raw = _ax_prop_raw(props, name)
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw != "false"  # "true" and "mixed" both -> True
    return bool(raw)


def _ax_tristate(
    props: list[dict[str, Any]], name: str
) -> Literal["checked", "unchecked", "mixed"] | None:
    """Normalize an AX tristate property. Handles:
    - boolean: True -> "checked", False -> "unchecked"
    - tristate: "true" -> "checked", "false" -> "unchecked", "mixed" -> "mixed"
    - missing -> None (property doesn't apply to this element)"""
    raw = _ax_prop_raw(props, name)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "checked" if raw else "unchecked"
    if isinstance(raw, str):
        if raw == "true":
            return "checked"
        if raw == "mixed":
            return "mixed"
        return "unchecked"
    return "checked" if raw else "unchecked"


def _ax_string(props: list[dict[str, Any]], name: str) -> str | None:
    """Normalize an AX property to str|None. Returns None if missing or empty."""
    raw = _ax_prop_raw(props, name)
    if not raw:
        return None
    return str(raw)


_EXPANDED_MAP: dict[str, str] = {
    "checked": "expanded",
    "unchecked": "collapsed",
    "mixed": "mixed",
}

# AX ignored reasons that mean "explicitly hidden" vs "no semantic role".
# ariaHiddenElement/Subtree: author set aria-hidden="true"
# notRendered: display:none, visibility:hidden, etc.
# uninteresting: no AX role (canvas, video, plain divs) — NOT hidden.
_AX_HIDDEN_REASONS = frozenset({"ariaHiddenElement", "ariaHiddenSubtree", "notRendered"})


def merge_axtree(payload: dict[str, Any], aid_to_ax: dict[str, dict[str, Any]]) -> None:
    """Overwrite JS-collected fields with authoritative AXTree signals.

    `payload` is the raw dict from `_collect_payload()`.
    Its shape is: {"containers": [...], "blocks": [...], "controls": [...], "pageEpoch": int}
    """
    for control in payload["controls"]:
        ax = aid_to_ax.get(control["id"])
        if not ax:
            continue
        props = ax.get("properties", [])

        ax_name = ax.get("name", {}).get("value")
        if ax_name and not control.get("duplicateId"):
            control["ax_name"] = ax_name
        ax_role = ax.get("role", {}).get("value")
        if ax_role:
            control["ax_role"] = ax_role
        control["disabled"] = _ax_bool(props, "disabled")
        if ax.get("ignored", False):
            reasons = {r.get("name") for r in ax.get("ignoredReasons", [])}
            control["ax_ignored"] = bool(reasons & _AX_HIDDEN_REASONS)
        else:
            control["ax_ignored"] = False

        control["checked"] = _ax_tristate(props, "checked")
        _exp_raw = _ax_tristate(props, "expanded")
        control["expanded"] = _EXPANDED_MAP.get(_exp_raw) if _exp_raw else None
        control["selected"] = _ax_bool(props, "selected")
        # AX value can be null for React-controlled inputs even when DOM has a value
        # (e.g. swiggy's location input). Prefer AX (it's the canonical accessibility
        # value), fall back to the live DOM value captured by the JS extractor.
        ax_val = _ax_string(props, "value")
        control["current_value"] = ax_val if ax_val else control.get("domValue")

        control["has_popup"] = _ax_string(props, "hasPopup")
        control["ax_description"] = ax.get("description", {}).get("value")
        control["focusable"] = _ax_bool(props, "focusable")

    for container in payload["containers"]:
        ax = aid_to_ax.get(container["id"])
        if not ax:
            continue
        props = ax.get("properties", [])
        container["ax_modal"] = _ax_bool(props, "modal")


async def build_aid_to_ax_map(
    cdp: Any, ax_nodes: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Map data-cdx-aid values to their AXTree node.

    Uses DOM.getDocument(depth=-1) to fetch the full DOM tree in a single CDP call,
    then walks it in Python to find all [data-cdx-aid] elements and their backendNodeIds.

    `cdp` is any object with `.send(method, params)` — pass `sm.cdp` in v2;
    crawler passed a `page.context.new_cdp_session(page)` client. Both target
    the same page WebSocket so behavior is identical.
    """
    ax_by_backend: dict[int, dict[str, Any]] = {
        node["backendDOMNodeId"]: node for node in ax_nodes if "backendDOMNodeId" in node
    }

    doc = await cdp.send("DOM.getDocument", {"depth": -1})

    aid_to_ax: dict[str, dict[str, Any]] = {}

    def _walk(node: dict[str, Any]) -> None:
        attrs = node.get("attributes", [])  # flat: ["attr1", "val1", "attr2", "val2", ...]
        try:
            idx = attrs.index("data-cdx-aid")
            aid = attrs[idx + 1]
        except (ValueError, IndexError):
            aid = None

        if aid:
            backend_id = node.get("backendNodeId")
            if backend_id is not None:
                ax_node = ax_by_backend.get(backend_id)
                if ax_node:
                    aid_to_ax[aid] = ax_node

        for child in node.get("children", []):
            _walk(child)

    _walk(doc["root"])
    return aid_to_ax


# ═════════════════════════════════════════════════════════════════════════════
# PageFilter
# ═════════════════════════════════════════════════════════════════════════════


class PageFilter:
    # Words that earn a control the `action_word_match` utility bonus.
    # Match is word-boundary (re-anchored in _contains_phrase) so "add" matches
    # "Add to cart" but NOT "address" or "padded". Commerce verbs were added
    # after the swiggy hazelnut investigation where 114 "Add" buttons tied with
    # 100+ "More Details" / "See more information" buttons at the same utility
    # score and got dropped past the button cap. See draft.md for the planned
    # structural replacement (length/role-based scoring instead of a word list).
    _ACTION_WORDS: tuple[str, ...] = (
        "next",
        "continue",
        "start",
        "open",
        "view",
        "submit",
        "verify",
        "search",
        "apply",
        "register",
        "login",
        "sign in",
        "proceed",
        "accept",
        "decline",
        "close",
        "dismiss",
        "reveal",
        # Commerce / cart actions
        "add",
        "buy",
        "order",
        "cart",
        "checkout",
        "save",
        "remove",
        "delete",
        "confirm",
        "pay",
        "send",
    )

    _NOISE_WORDS: tuple[str, ...] = (
        "sponsored",
        "promo",
        "install",
        "download",
        "newsletter",
        "subscribe",
        "upsell",
        "cookie",
        "consent",
        "advertisement",
        "advertising",
    )

    def __init__(
        self,
        sm: SessionManager,
        use_justext: bool = False,
        tuning: ExtractorTuning | None = None,
    ) -> None:
        self._sm = sm
        self._use_justext = use_justext
        self._tuning = tuning if tuning is not None else ExtractorTuning()
        self.last_timing: dict[str, float] = {}
        self.last_aid_to_ax_map: dict[str, dict[str, Any]] = {}

    async def _collect_axtree(self) -> dict[str, Any]:
        """Fetch the full AXTree via the SessionManager-owned CDP."""
        return await self._sm.cdp.send("Accessibility.getFullAXTree")

    async def run(
        self,
        snapshot: PageSnapshot,
        aid_allowlist: set[str] | None = None,
        enumerate_mode: bool = False,
    ) -> PageFilterOutput:
        page = self._sm.page
        t0 = time.perf_counter()

        # Fetch JS payload and AXTree in parallel — AX is additive, failures are non-fatal.
        payload_task = self._collect_payload(enumerate_mode=enumerate_mode)
        ax_task = self._collect_axtree()
        results = await asyncio.gather(payload_task, ax_task, return_exceptions=True)
        payload_result, ax_result = results[0], results[1]
        if isinstance(payload_result, BaseException):
            raise payload_result
        payload: dict[str, Any] = payload_result

        # Source-localized log: this is where the JS extraction payload is built.
        # Records the raw universe of containers/controls/blocks BEFORE any filter
        # rules apply, so we can later compare against V5 to see what got dropped.
        # All three artifacts (HTML, AXTree, JS payload) share extraction_id so
        # they correlate to a single PageFilter run.
        extraction_id = f"{int(time.time() * 1000)}_{id(snapshot) & 0xFFFF:04x}"
        notes.log(
            data_type="pf_raw_payload",
            data=payload,
            extraction_id=extraction_id,
            url=snapshot.url,
        )
        notes.log(
            data_type="page_html",
            data=snapshot.html,
            extraction_id=extraction_id,
            url=snapshot.url,
            title=snapshot.title,
        )
        if not isinstance(ax_result, BaseException):
            notes.log(
                data_type="page_axtree",
                data=ax_result,
                extraction_id=extraction_id,
                url=snapshot.url,
            )

        t_ax_start = time.perf_counter()
        ax_map_ms = 0.0
        ax_merge_ms = 0.0
        if not isinstance(ax_result, BaseException):
            try:
                ax_tree: dict[str, Any] = ax_result
                aid_to_ax = await build_aid_to_ax_map(self._sm.cdp, ax_tree["nodes"])
                ax_map_ms = (time.perf_counter() - t_ax_start) * 1000
                self.last_aid_to_ax_map = aid_to_ax
                t_merge = time.perf_counter()
                merge_axtree(payload, aid_to_ax)
                ax_merge_ms = (time.perf_counter() - t_merge) * 1000
            except (PlaywrightError, KeyError, ConnectionError) as exc:
                logger.warning("AX enrichment skipped (merge): %s", exc)
                self.last_aid_to_ax_map = {}
        else:
            logger.warning("AX enrichment skipped (fetch): %s", ax_result)
            self.last_aid_to_ax_map = {}

        t_js = time.perf_counter()

        blocks, uncertain_block_ids = self._filter_blocks_global(payload["blocks"])
        controls, uncertain_control_ids = self._build_controls(
            controls_raw=payload["controls"],
            page_url=snapshot.url,
        )
        controls = self._select_controls(controls)

        drop_zone_aids: set[str] = set(payload.get("dropZoneAids", []))
        kept_container_ids, parent_by_id = self._build_container_closure(
            containers_raw=payload["containers"],
            blocks=blocks,
            controls=controls,
            drop_zone_aids=drop_zone_aids,
            aid_allowlist=aid_allowlist,
        )
        dnd_library: str | None = payload.get("dndLibrary")
        synthetic_drag_accepted: bool | None = payload.get("syntheticDragAccepted")
        containers = self._build_containers(
            containers_raw=payload["containers"],
            kept_container_ids=kept_container_ids,
            parent_by_id=parent_by_id,
            blocks=blocks,
            controls=controls,
            drop_zone_aids=drop_zone_aids,
        )
        container_by_id = {container.container_id: container for container in containers}

        forms, orphan_controls = self._build_forms(
            controls=controls, container_by_id=container_by_id,
        )
        form_control_ids = self._collect_form_control_ids(forms)
        buttons, dropped_button_count = self._build_buttons(
            controls=controls,
            container_by_id=container_by_id,
            exclude_ids=form_control_ids,
        )
        t_build = time.perf_counter()

        blocking_by_container = await self._collect_blocking_relations(containers)
        containers = self._apply_blocking_relations(containers, blocking_by_container)
        container_by_id = {container.container_id: container for container in containers}

        container_handle_specs = [
            {
                "container_id": container.container_id,
                "parent_container_id": container.parent_container_id or "",
            }
            for container in containers
        ]
        root_container_id = next(
            (
                container.container_id
                for container in containers
                if container.parent_container_id is None
            ),
            None,
        )
        target_specs = self._build_occlusion_target_specs(buttons, forms, container_by_id)
        occlusion_map = await self._collect_occlusion(
            target_specs,
            container_handle_specs,
            root_container_id,
        )
        t_occlusion = time.perf_counter()

        buttons = self._apply_button_occlusion(buttons, occlusion_map)
        forms = self._apply_form_occlusion(forms, occlusion_map, blocking_by_container)

        containers = self._attach_container_refs(
            containers=containers,
            controls=controls,
            buttons=buttons,
            forms=forms,
            parent_by_id=parent_by_id,
        )
        containers = self._classify_container_labels(containers)
        actions = self._build_actions(forms=forms, buttons=buttons)

        justext_text, justext_paragraphs = self._extract_justext(snapshot.html)
        blocking_container_ids = [
            container.container_id for container in containers if container.is_blocking
        ]

        important_containers = sorted(
            [container for container in containers if container.section_like],
            key=lambda item: (
                item.utility_score - item.noise_score,
                item.geometry.w * item.geometry.h,
                item.z_index,
            ),
            reverse=True,
        )[: self._tuning.top_sections]

        uncertain_items = uncertain_block_ids + uncertain_control_ids
        t_finalize = time.perf_counter()

        self.last_timing = {
            "js_collect_ms": round((t_js - t0) * 1000, 1),
            "ax_map_ms": round(ax_map_ms, 1),
            "ax_merge_ms": round(ax_merge_ms, 1),
            "python_build_ms": round((t_build - t_js) * 1000, 1),
            "blocking_and_occlusion_ms": round((t_occlusion - t_build) * 1000, 1),
            "finalize_ms": round((t_finalize - t_occlusion) * 1000, 1),
            "total_ms": round((t_finalize - t0) * 1000, 1),
        }

        # `page` reference retained for symmetry with crawler — actual page evals
        # happen via self._sm.page inside _collect_payload / _collect_*.
        del page

        return PageFilterOutput(
            page_url=snapshot.url,
            page_title=snapshot.title,
            containers=containers,
            forms=forms,
            buttons=buttons,
            actions=actions,
            orphan_controls=orphan_controls,
            container_count=len(containers),
            important_container_count=len(important_containers),
            form_count=len(forms),
            button_count=len(buttons),
            action_count=len(actions),
            blocked_action_count=sum(1 for action in actions if action.blocked_now),
            dropped_button_count=dropped_button_count,
            blocking_container_ids=blocking_container_ids,
            blocking_container_count=len(blocking_container_ids),
            has_blocking_containers=bool(blocking_container_ids),
            uncertain_items=uncertain_items,
            dnd_library=dnd_library,
            synthetic_drag_accepted=synthetic_drag_accepted,
            justext_text=justext_text,
            justext_paragraphs=justext_paragraphs,
            justext_paragraph_count=len(justext_paragraphs),
            page_epoch=int(payload["pageEpoch"]),
        )

    async def _collect_payload(self, enumerate_mode: bool = False) -> dict[str, Any]:
        return await self._sm.page.evaluate(
            """
            async (args) => {
              // Wait for two rAFs so the browser completes a full layout-paint cycle
              // before we start measuring. Without this, getBoundingClientRect() can
              // return {0,0,0,0} for elements that exist in the DOM but haven't been
              // laid out yet, causing visible() to drop them and producing empty/partial
              // extractions on freshly-rendered pages.
              await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

              const vw = window.innerWidth || 1;
              const vh = window.innerHeight || 1;
              const viewportArea = Math.max(1, vw * vh);

              const escapeCss = (value) =>
                String(value).replace(/([ #;?%&,.+*~\\':"!^$\[\]()=>|\/@])/g, "\\$1");

              const selectorFor = (el) => {
                if (!el) return "body";
                if (el.id) return "#" + escapeCss(el.id);
                const parts = [];
                let current = el;
                let depth = 0;
                while (
                  current &&
                  current.nodeType === 1 &&
                  current.tagName.toLowerCase() !== "html" &&
                  depth < 8
                ) {
                  const tag = current.tagName.toLowerCase();
                  const parent = current.parentElement;
                  if (!parent) {
                    parts.unshift(tag);
                    break;
                  }
                  const siblings = Array.from(parent.children).filter(
                    (x) => x.tagName === current.tagName
                  );
                  const idx = Math.max(1, siblings.indexOf(current) + 1);
                  parts.unshift(`${tag}:nth-of-type(${idx})`);
                  current = parent;
                  depth += 1;
                }
                return parts.length ? parts.join(" > ") : "body";
              };

              const normText = (value) => String(value || "").replace(/\s+/g, " ").trim();

              const visible = (el) => {
                if (!el) return false;
                if (el === document.body) return true;
                const rect = el.getBoundingClientRect();
                if (rect.width <= 1 || rect.height <= 1) return false;
                const style = window.getComputedStyle(el);
                if (
                  style.display === "none" ||
                  style.visibility === "hidden" ||
                  style.opacity === "0"
                ) {
                  return false;
                }
                return true;
              };

              const inViewport = (el) => {
                const rect = el.getBoundingClientRect();
                return !(
                  rect.bottom < 0 ||
                  rect.right < 0 ||
                  rect.top > vh ||
                  rect.left > vw
                );
              };

              const toRatio = (value, maxValue) => {
                if (!Number.isFinite(value) || !Number.isFinite(maxValue) || maxValue <= 0) {
                  return 0;
                }
                return value / maxValue;
              };

              const resolveLabel = (el) => {
                const aria = normText(el.getAttribute("aria-label") || "");
                if (aria) return aria;
                const title = normText(el.getAttribute("title") || "");
                if (title) return title;
                const placeholder = normText(el.getAttribute("placeholder") || "");
                if (placeholder) return placeholder;
                const idAttr = normText(el.id || "");
                if (idAttr) {
                  const qs = `[id="${escapeCss(idAttr)}"]`;
                  const isDuplicate = document.querySelectorAll(qs).length > 1;
                  const scope = isDuplicate ? el.parentElement : document;
                  if (scope) {
                    const byFor = scope.querySelector(`label[for="${escapeCss(idAttr)}"]`);
                    if (byFor) {
                      const txt = normText(byFor.innerText || byFor.textContent || "");
                      if (txt) return txt;
                    }
                  }
                }
                const parentLabel = el.closest("label");
                if (parentLabel) {
                  const txt = normText(parentLabel.innerText || parentLabel.textContent || "");
                  if (txt) return txt;
                }
                const txt = normText(el.innerText || el.textContent || "");
                if (txt) return txt;
                return null;
              };

              const BLOCK_TAGS = [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "li",
                "dt",
                "dd",
                "label",
                "legend",
                "button",
                "td",
                "th",
                "blockquote",
                "pre",
                "code",
                "summary",
                "div",
                "section",
                "article",
                "form",
                "main",
                "aside",
                "nav",
              ];
              const BLOCK_SELECTOR = BLOCK_TAGS.join(", ");
              const BLOCK_SET = new Set(BLOCK_TAGS);

              // Tags whose textContent is never meaningful to a CU agent —
              // <style>: CSS rules, <script>: JS code, <noscript>: JS-disabled fallback.
              // Without this skip, styled-components dumps the entire stylesheet into
              // V5 text_blocks via the textContent fallback (innerText respects
              // display:none but textContent does not).
              const SKIP_TEXT_TAGS = new Set(["style", "script", "noscript"]);

              const ownTextContent = (el) => {
                let text = "";
                for (const node of el.childNodes) {
                  if (node.nodeType === Node.TEXT_NODE) {
                    text += node.textContent || "";
                  } else if (node.nodeType === Node.ELEMENT_NODE) {
                    const tag = node.tagName.toLowerCase();
                    if (SKIP_TEXT_TAGS.has(tag)) continue;
                    if (!BLOCK_SET.has(tag) && visible(node)) {
                      text += node.innerText || node.textContent || "";
                    }
                  }
                }
                return text;
              };

              const hasOwnText = (el) =>
                normText(ownTextContent(el)).length > 0;

              const isTextBearingLeaf = (el) => hasOwnText(el);

              const containerNodes = [];
              const containerSet = new Set();

              const bodyNode = document.body;
              containerNodes.push(bodyNode);
              containerSet.add(bodyNode);

              const candidates = Array.from(
                document.querySelectorAll(
                  [
                    "main",
                    "article",
                    "section",
                    "form",
                    "nav",
                    "aside",
                    "div",
                    "dialog",
                    "[role='dialog']",
                    "[role='alertdialog']",
                    "[aria-modal='true']",
                  ].join(", ")
                )
              );
              for (const node of candidates) {
                if (!args.enumerateMode && !visible(node)) continue;
                if (containerSet.has(node)) continue;
                containerSet.add(node);
                containerNodes.push(node);
                if (containerNodes.length >= args.containerNodeLimit) break;
              }

              const extractDataAttributes = (node) => {
                const out = {};
                let count = 0;
                for (const attr of Array.from(node.attributes || [])) {
                  const name = String(attr.name || "").toLowerCase();
                  if (!(name.startsWith("data-") || name.startsWith("aria-"))) {
                    continue;
                  }
                  if (
                    /^data-(react|v-|ng-|ember-|vue|svelte|astro|next|nuxt)/.test(name)
                  ) {
                    continue;
                  }
                  const value = normText(attr.value || "");
                  if (!value) continue;
                  if (value.length > args.dataAttrMaxValueChars) continue;
                  out[name] = value;
                  count += 1;
                  if (count >= args.maxDataAttrsPerContainer) break;
                }
                return out;
              };

              // Stable handle stamping: assign or reuse data-cdx-aid on DOM nodes
              // Scan existing aids: seed counter from max, count occurrences to detect clones.
              // seenAids is NOT pre-seeded — only tracks aids assigned this pass,
              // so re-extraction correctly reuses existing stamps.
              let aidCounter = 1;
              const aidDomCounts = {};
              for (const el of document.querySelectorAll('[data-cdx-aid]')) {
                const val = el.getAttribute('data-cdx-aid');
                const m = /^aid-(\d+)$/.exec(val);
                if (m) {
                  aidDomCounts[val] = (aidDomCounts[val] || 0) + 1;
                  const n = parseInt(m[1], 10) + 1;
                  if (n > aidCounter) aidCounter = n;
                }
              }
              const seenAids = new Set();
              const stampAid = (node) => {
                let aid = node.getAttribute('data-cdx-aid');
                if (!aid || !/^aid-\d+$/.test(aid)
                    || (aidDomCounts[aid] || 0) > 1
                    || seenAids.has(aid)) {
                  aid = `aid-${aidCounter}`;
                  aidCounter += 1;
                  node.setAttribute('data-cdx-aid', aid);
                }
                seenAids.add(aid);
                return aid;
              };
              const containerIdByNode = new Map();
              const containers = [];
              for (let idx = 0; idx < containerNodes.length; idx += 1) {
                const node = containerNodes[idx];
                const rect = node.getBoundingClientRect();
                const style = window.getComputedStyle(node);
                const role = normText(node.getAttribute("role") || "") || null;
                const headingNode = node.querySelector("h1,h2,h3,h4");
                const heading = headingNode ? normText(headingNode.innerText || "") : "";
                const id = stampAid(node);
                const overflowX = style.overflowX || "";
                const overflowY = style.overflowY || "";
                const scrollableX =
                  (overflowX === "scroll" || overflowX === "auto") &&
                  node.scrollWidth > node.clientWidth + 5;
                const scrollableY =
                  (overflowY === "scroll" || overflowY === "auto") &&
                  node.scrollHeight > node.clientHeight + 5;
                const hasAnimation =
                  style.animationName !== "none" ||
                  ((style.transitionProperty || "none") !== "none" &&
                    parseFloat(style.transitionDuration || "0") > 0);
                containerIdByNode.set(node, id);
                containers.push({
                  id,
                  selector: selectorFor(node),
                  tag: node.tagName.toLowerCase(),
                  role,
                  heading,
                  domOrder: idx + 1,
                  xRatio: toRatio(rect.left, vw),
                  yRatio: toRatio(rect.top, vh),
                  wRatio: toRatio(rect.width, vw),
                  hRatio: toRatio(rect.height, vh),
                  zIndex: Number.parseInt(style.zIndex || "0", 10) || 0,
                  pointerBlocking: style.pointerEvents !== "none",
                  fixedPosition: style.position === "fixed" || style.position === "sticky",
                  scrollable: scrollableX || scrollableY,
                  hasAnimation,
                  dataAttributes: extractDataAttributes(node),
                  areaRatio: (rect.width * rect.height) / viewportArea,
                });
              }

              for (let idx = 0; idx < containerNodes.length; idx += 1) {
                const node = containerNodes[idx];
                const container = containers[idx];
                let parentId = null;
                let current = node.parentElement;
                while (current) {
                  if (containerIdByNode.has(current)) {
                    parentId = containerIdByNode.get(current);
                    break;
                  }
                  current = current.parentElement;
                }
                container.parentContainerId = parentId;
              }

              const nearestContainerId = (el) => {
                let current = el;
                while (current) {
                  const maybeId = containerIdByNode.get(current);
                  if (maybeId) return maybeId;
                  current = current.parentElement;
                }
                return containers[0].id;
              };

              const blocks = [];
              let blockCounter = 0;
              const blockSeenByContainer = new Map();
              const blockNodes = Array.from(document.querySelectorAll(BLOCK_SELECTOR));
              for (const node of blockNodes) {
                if (!visible(node)) continue;
                if (!isTextBearingLeaf(node)) continue;
                const text = normText(ownTextContent(node));
                if (text.length < args.sectionBlockMinChars) continue;
                const containerId = nearestContainerId(node);
                const key = text.toLowerCase();
                if (!blockSeenByContainer.has(containerId)) {
                  blockSeenByContainer.set(containerId, new Set());
                }
                const seen = blockSeenByContainer.get(containerId);
                if (seen.has(key)) continue;
                seen.add(key);
                blockCounter += 1;
                blocks.push({
                  id: `blk-${blockCounter}`,
                  containerId,
                  text,
                  domOrder: blockCounter,
                });
              }

              const controlNodes = Array.from(
                document.querySelectorAll(
                  [
                    "input",
                    "select",
                    "textarea",
                    "button",
                    "a",
                    "[role='button']",
                    "[onclick]",
                    "[draggable='true']",
                    "[data-rbd-drag-handle-draggable-id]",
                    "[data-rfd-drag-handle-draggable-id]",
                    "[aria-roledescription='draggable']",
                    "[contenteditable='true']",
                    "[role='slider']",
                    "[role='tab']",
                    "[role='menuitem']",
                    "[role='radio']",
                    "[role='checkbox']",
                    "[role='switch']",
                    "canvas",
                    "video",
                    "audio",
                  ].join(", ")
                )
              )
                .filter((el) => {
                  const type = normText(el.getAttribute("type") || "").toLowerCase();
                  return !(el.tagName.toLowerCase() === "input" && type === "hidden");
                })
                .slice(0, args.controlNodeLimit);

              const controls = [];
              let controlDomOrder = 0;
              for (const node of controlNodes) {
                const rect = node.getBoundingClientRect();
                const style = window.getComputedStyle(node);
                const tag = node.tagName.toLowerCase();
                const role = normText(node.getAttribute("role") || "") || null;
                const type = normText(node.getAttribute("type") || "") || null;
                const name = normText(node.getAttribute("name") || "") || null;
                const href = normText(node.getAttribute("href") || "") || null;
                const label = resolveLabel(node);
                const ownText = normText(
                  node.innerText ||
                  node.value ||
                  node.getAttribute("aria-label") ||
                  node.getAttribute("title") ||
                  ""
                );
                const formRoot = node.closest("form");
                const options = [];
                if (tag === "select") {
                  const optionNodes = Array.from(node.querySelectorAll("option")).slice(0, 200);
                  for (const optionNode of optionNodes) {
                    const optionText = normText(
                      optionNode.innerText || optionNode.textContent || ""
                    );
                    if (optionText) options.push(optionText);
                  }
                }

                controlDomOrder += 1;
                const ctlAid = stampAid(node);
                const nodeId = normText(node.id || "");
                const duplicateId = nodeId
                  ? document.querySelectorAll(`[id="${escapeCss(nodeId)}"]`).length > 1
                  : false;
                // Capture the live DOM input value as a separate field. The AXTree
                // can come back with `value: null` for React-controlled inputs (e.g.
                // swiggy clears its location input after autocomplete consumes the
                // pick — DOM still has the chosen address, AX does not). The Python
                // side prefers AX value when present and falls back to this. Skip
                // password/hidden/file types for privacy/no-signal reasons.
                const valueTagOK = (tag === "input" || tag === "textarea" || tag === "select");
                const valueTypeOK = !(tag === "input" && ["password","hidden","file"].includes((type||"").toLowerCase()));
                const domValue = (valueTagOK && valueTypeOK) ? (node.value || null) : null;
                controls.push({
                  id: ctlAid,
                  selector: selectorFor(node),
                  containerId: nearestContainerId(node),
                  formSelector: formRoot ? selectorFor(formRoot) : null,
                  tag,
                  role,
                  type,
                  name,
                  label,
                  href,
                  text: ownText || null,
                  options,
                  domValue,
                  xRatio: toRatio(rect.left, vw),
                  yRatio: toRatio(rect.top, vh),
                  wRatio: toRatio(rect.width, vw),
                  hRatio: toRatio(rect.height, vh),
                  zIndex: Number.parseInt(style.zIndex || "0", 10) || 0,
                  inViewportNow: inViewport(node),
                  visibleNow: visible(node),
                  areaRatio: (rect.width * rect.height) / viewportArea,
                  isFixed: style.position === "fixed" || style.position === "sticky",
                  cursor: style.cursor || null,
                  draggable: node.draggable === true
                    || node.getAttribute('data-rbd-drag-handle-draggable-id') !== null
                    || node.getAttribute('data-rfd-drag-handle-draggable-id') !== null
                    || node.getAttribute('aria-roledescription') === 'draggable',
                  duplicateId,
                  domOrder: controlDomOrder,
                  // Slider metadata — keep in sync with executor.py:drag_slider
                  sliderMin: (tag === 'input' && type === 'range')
                    ? (node.min !== '' ? +node.min : 0)
                    : (role === 'slider' ? (+(node.getAttribute('aria-valuemin') || '0')) : null),
                  sliderMax: (tag === 'input' && type === 'range')
                    ? (node.max !== '' ? +node.max : 100)
                    : (role === 'slider' ? (+(node.getAttribute('aria-valuemax') || '100')) : null),
                  sliderValue: (tag === 'input' && type === 'range')
                    ? (+node.value)
                    : (role === 'slider' ? (+(node.getAttribute('aria-valuenow') || '0')) : null),
                  sliderOrientation: (tag === 'input' && type === 'range')
                    ? (rect.height > rect.width ? 'vertical' : 'horizontal')
                    : (role === 'slider'
                      ? (node.getAttribute('aria-orientation')
                        || (rect.height > rect.width ? 'vertical' : 'horizontal'))
                      : null),
                });
              }

              // Library fingerprint: identify DnD library from static DOM markers.
              let dndLibrary =
                document.querySelector('[data-rbd-droppable-id]') ? 'react-beautiful-dnd' :
                document.querySelector('[data-rfd-droppable-id]') ? 'hello-pangea' :
                document.querySelector('[aria-roledescription="draggable"]') ? 'dnd-kit' :
                null;

              // Walk up DOM to find nearest already-registered container.
              const resolveParent = (el) => {
                let cur = el.parentElement;
                while (cur) {
                  const pid = containerIdByNode.get(cur);
                  if (pid) return pid;
                  cur = cur.parentElement;
                }
                return containers.length > 0 ? containers[0].id : null;
              };

              // Force-add a droppable element if not already in containerIdByNode,
              // then return its aid (works for both existing and newly-added elements).
              const forceAddDropZone = (el) => {
                if (!containerIdByNode.has(el)) {
                  const rect = el.getBoundingClientRect();
                  const style = window.getComputedStyle(el);
                  const id = stampAid(el);
                  containers.push({
                    id,
                    selector: selectorFor(el),
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute('role') || null,
                    heading: (() => {
                      const h = el.querySelector('h1,h2,h3,h4,h5,h6');
                      const t = h ? h.textContent : el.textContent;
                      return t.trim().slice(0, 200);
                    })(),
                    domOrder: containers.length + 1,
                    xRatio: toRatio(rect.left, vw),
                    yRatio: toRatio(rect.top, vh),
                    wRatio: toRatio(rect.width, vw),
                    hRatio: toRatio(rect.height, vh),
                    zIndex: Number.parseInt(style.zIndex || '0', 10) || 0,
                    pointerBlocking: style.pointerEvents !== 'none',
                    fixedPosition: style.position === 'fixed' || style.position === 'sticky',
                    scrollable: false,
                    hasAnimation: false,
                    dataAttributes: extractDataAttributes(el),
                    areaRatio: (rect.width * rect.height) / viewportArea,
                    parentContainerId: resolveParent(el),
                  });
                  containerIdByNode.set(el, id);
                }
                return el.getAttribute('data-cdx-aid');
              };

              // Collect drop zone aids — always built fresh here, not via old logic.
              const dropZoneAids = [];

              // Change A: rbd / hello-pangea — static droppable attribute on each slot.
              if (dndLibrary === 'react-beautiful-dnd') {
                for (const el of document.querySelectorAll('[data-rbd-droppable-id]')) {
                  const aid = forceAddDropZone(el);
                  if (aid) dropZoneAids.push(aid);
                }
              } else if (dndLibrary === 'hello-pangea') {
                for (const el of document.querySelectorAll('[data-rfd-droppable-id]')) {
                  const aid = forceAddDropZone(el);
                  if (aid) dropZoneAids.push(aid);
                }
              }

              // Change B: dnd-kit — BCR probe to discover droppables (no static attrs).
              // Ordering matters: pointerdown BEFORE patching getBCR so dnd-kit's PointerSensor
              // registers the drag-start without our patch interfering. After pointerdown, we
              // patch, then fire pointermove to trigger collision detection (RAF-based in dnd-kit),
              // then wait 3 frames so the collision loop runs before we collect callers.
              if (dndLibrary === 'dnd-kit') {
                const draggable = document.querySelector('[aria-roledescription="draggable"]');
                if (draggable) {
                  const bcrCallers = [];
                  const origBCR = Element.prototype.getBoundingClientRect;
                  try {
                    // Step 1: get rect and fire pointerdown BEFORE patching.
                    // isPrimary + pressure required for dnd-kit's PointerSensor to activate.
                    const rect = draggable.getBoundingClientRect();
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top + rect.height / 2;
                    // TODO(read-only extraction): this pointerdown fires dnd-kit's PointerSensor
                    // which calls onDragStart callbacks — app state may mutate. The clean fix
                    // is to run the probe in an isolated throwaway page context so the live
                    // planning page is never touched. Deferred: requires isolated-page
                    // infrastructure. For now, onDragCancel (via pointerup) resets drag state
                    // and the probe runs pre-planning so any residue is overwritten by re-render.
                    draggable.dispatchEvent(new PointerEvent('pointerdown', {
                      bubbles: true, cancelable: true,
                      clientX: cx, clientY: cy, pointerId: 1,
                      isPrimary: true, pressure: 0.5,
                    }));
                    // Step 2: patch AFTER pointerdown so only post-drag-start BCR calls captured.
                    Element.prototype.getBoundingClientRect = function() {
                      bcrCallers.push(this);
                      return origBCR.call(this);
                    };
                    // Dispatch on both draggable and document — dnd-kit listens on both.
                    draggable.dispatchEvent(new PointerEvent('pointermove', {
                      bubbles: true, cancelable: true,
                      clientX: cx + 20, clientY: cy + 20, pointerId: 1,
                      isPrimary: true, pressure: 0.5,
                    }));
                    document.dispatchEvent(new PointerEvent('pointermove', {
                      bubbles: true, cancelable: true,
                      clientX: cx + 20, clientY: cy + 20, pointerId: 1,
                      isPrimary: true, pressure: 0.5,
                    }));
                    // Step 3: wait 5 RAF frames for dnd-kit's RAF-based collision detection.
                    for (let i = 0; i < 5; i++) await new Promise(r => requestAnimationFrame(r));
                    document.dispatchEvent(new PointerEvent('pointerup', {
                      bubbles: true, cancelable: true,
                      clientX: cx + 20, clientY: cy + 20, pointerId: 1,
                    }));
                  } finally {
                    // Always restore — the try/finally guarantees cleanup even on throw.
                    // Residual drag state (dnd-kit's internal active descriptor) resolves
                    // naturally: pointerup fires onDragCancel, and the next React render
                    // resets any stale drag-overlay state. No DOM mutation lingers.
                    Element.prototype.getBoundingClientRect = origBCR;
                  }
                  // Collect unique callers that are actual droppables, not layout ancestors.
                  // dnd-kit's collision detection calls BCR on registered droppables AND on
                  // ancestor nodes during sensor position tracking — exclude the latter.
                  const draggableAncestors = new Set();
                  let _p = draggable.parentElement;
                  while (_p) { draggableAncestors.add(_p); _p = _p.parentElement; }

                  const seen = new Set();
                  for (const el of bcrCallers) {
                    if (seen.has(el)) continue;
                    seen.add(el);
                    if (el.getAttribute('aria-roledescription') === 'draggable') continue;
                    if (draggableAncestors.has(el)) continue; // layout/wrapper, not a drop zone
                    if (el === document.body || el === document.documentElement) continue;
                    const aid = forceAddDropZone(el);
                    if (aid) dropZoneAids.push(aid);
                  }
                }
              }

              // Change C: Native HTML5 DnD — dragover+preventDefault probe (experimental).
              // Follows dnd-kit BCR probe pattern: pick a draggable, fake a drag sequence,
              // observe which elements respond with preventDefault on dragover.
              // Risk: synthetic DragEvent may not trigger the same code paths as real drags.
              // If this fails, Tier 2 (agent-invoked physical probe) is the fallback.
              let syntheticDragAccepted = null;  // null = no draggables, true/false = probed
              if (dndLibrary === null) {
                const draggable = document.querySelector('[draggable="true"]');
                if (draggable) {
                  // Set dndLibrary to 'html5-native' so downstream knows draggables exist.
                  // This is set regardless of whether the probe finds drop zones —
                  // the agent needs to know the page uses native HTML5 DnD.
                  dndLibrary = 'html5-native';

                  const dropTargets = [];
                  const origPreventDefault = Event.prototype.preventDefault;

                  // Step 1: fire dragstart on the draggable to initiate drag session
                  const rect = draggable.getBoundingClientRect();
                  const cx = rect.left + rect.width / 2;
                  const cy = rect.top + rect.height / 2;

                  const dt = new DataTransfer();
                  draggable.dispatchEvent(new DragEvent('dragstart', {
                    bubbles: true, cancelable: true, dataTransfer: dt,
                    clientX: cx, clientY: cy,
                  }));

                  // Step 2: patch preventDefault to detect who calls it on dragover
                  let currentTarget = null;
                  Event.prototype.preventDefault = function() {
                    if (this.type === 'dragover' && currentTarget) {
                      dropTargets.push(currentTarget);
                    }
                    return origPreventDefault.call(this);
                  };

                  let syntheticProbeCount = 0;
                  try {
                    // Step 3: fire dragover on candidate containers (capped at 200)
                    const candidates = document.querySelectorAll(
                      'div, section, ul, ol, main, article, aside, [role]'
                    );
                    let probed = 0;
                    for (const el of candidates) {
                      if (probed >= 200) break;
                      if (el === draggable) continue;
                      if (el.getAttribute('draggable') === 'true') continue;
                      const r = el.getBoundingClientRect();
                      if (r.width === 0 || r.height === 0) continue;
                      currentTarget = el;
                      el.dispatchEvent(new DragEvent('dragover', {
                        bubbles: true, cancelable: true, dataTransfer: dt,
                        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
                      }));
                      probed++;
                    }
                    syntheticProbeCount = probed;
                  } finally {
                    Event.prototype.preventDefault = origPreventDefault;
                    currentTarget = null;
                  }

                  // Step 4: fire dragend to clean up
                  draggable.dispatchEvent(new DragEvent('dragend', {
                    bubbles: true, cancelable: true, dataTransfer: dt,
                  }));

                  // Step 5: dedupe, prune descendants, and force-add as drop zones.
                  const unique = [...new Set(dropTargets)].filter(
                    el => el !== document.body && el !== document.documentElement
                  );
                  const pruned = unique.filter(el =>
                    !unique.some(other => other !== el && other.contains(el))
                  );
                  for (const el of pruned) {
                    const aid = forceAddDropZone(el);
                    if (aid) dropZoneAids.push(aid);
                  }

                  // Flag: did any element call preventDefault on our synthetic dragover?
                  // If yes → page accepts isTrusted=false → synthetic drag is safe.
                  // If no (probed > 0 but 0 hits) → page may check isTrusted → need CDP.
                  syntheticDragAccepted = syntheticProbeCount > 0 && dropTargets.length > 0;
                }
              }

              return {
                containers,
                blocks,
                controls,
                dropZoneAids,
                dndLibrary,
                syntheticDragAccepted,
                pageEpoch: Math.floor(performance.now()),
              };
            }
            """,
            {
                "containerNodeLimit": 3000 if enumerate_mode else self._tuning.section_node_limit,
                "controlNodeLimit": 3000 if enumerate_mode else self._tuning.control_node_limit,
                "sectionBlockMinChars": self._tuning.section_block_min_chars,
                "maxDataAttrsPerContainer": 30 if enumerate_mode else 8,
                "dataAttrMaxValueChars": 200,
                "enumerateMode": bool(enumerate_mode),
            },
        )

    def _filter_blocks_global(
        self, blocks_raw: list[dict[str, Any]]
    ) -> tuple[list[_BlockAtom], list[str]]:
        template_counts: Counter[str] = Counter()
        cleaned_blocks: list[tuple[dict[str, Any], str]] = []

        for raw in blocks_raw:
            text = self._split_stuck_camel_tokens(self._strip_lorem_text(str(raw["text"])))
            if not text:
                continue
            template_key = self._normalize_for_template(text)
            if template_key:
                template_counts[template_key] += 1
            cleaned_blocks.append((raw, text))

        kept: list[_BlockAtom] = []
        uncertain_ids: list[str] = []
        seen_global: set[str] = set()

        for raw, text in cleaned_blocks:
            normalized_key = self._normalize_for_dedupe(text)
            if not normalized_key:
                continue
            if normalized_key in seen_global:
                continue
            template_key = self._normalize_for_template(text)
            repeat_count = template_counts.get(template_key, 0)
            utility, noise = self._score_text_block(text, repeat_count)

            should_drop = (
                utility < self._tuning.block_keep_utility_min
                and (noise - utility) >= self._tuning.block_drop_noise_minus_utility_min
                and abs(utility - noise) >= self._tuning.uncertain_band
            )
            if should_drop:
                continue
            if self._should_suppress_repeated_block(text, repeat_count) and utility <= noise:
                continue

            if abs(utility - noise) < self._tuning.uncertain_band:
                uncertain_ids.append(str(raw["id"]))

            kept.append(
                _BlockAtom(
                    block_id=str(raw["id"]),
                    container_id=str(raw["containerId"]),
                    text=text,
                    dom_order=int(raw["domOrder"]),
                )
            )
            seen_global.add(normalized_key)

        kept.sort(key=lambda item: item.dom_order)
        return kept, uncertain_ids

    def _build_controls(
        self,
        controls_raw: list[dict[str, Any]],
        page_url: str,
    ) -> tuple[list[_ControlAtom], list[str]]:
        page_host = urlparse(page_url).netloc.lower()
        controls: list[_ControlAtom] = []
        uncertain_ids: list[str] = []

        for raw in controls_raw:
            utility, noise, reasons = self._score_control(raw, page_host)
            geometry = self._geometry_from_raw(raw)
            control = _ControlAtom(
                control_id=str(raw["id"]),
                selector=str(raw["selector"]),
                container_id=str(raw["containerId"]),
                form_selector=str(raw["formSelector"]) if raw.get("formSelector") else None,
                tag=str(raw["tag"]),
                role=str(raw["role"]) if raw.get("role") else None,
                type=str(raw["type"]) if raw.get("type") else None,
                name=str(raw["name"]) if raw.get("name") else None,
                label=str(raw["label"]) if raw.get("label") else None,
                text=str(raw["text"]) if raw.get("text") else None,
                href=str(raw["href"]) if raw.get("href") else None,
                options=[str(option) for option in raw.get("options", [])],
                geometry=geometry,
                z_index=int(raw["zIndex"]),
                in_viewport_now=bool(raw["inViewportNow"]),
                visible_now=bool(raw["visibleNow"]),
                dom_order=int(raw["domOrder"]),
                cursor=str(raw["cursor"]) if raw.get("cursor") else None,
                draggable=bool(raw.get("draggable", False)),
                duplicate_id=bool(raw.get("duplicateId", False)),
                utility_score=utility,
                noise_score=noise,
                reason_codes=reasons,
                ax_name=str(raw["ax_name"]) if raw.get("ax_name") else None,
                ax_role=str(raw["ax_role"]) if raw.get("ax_role") else None,
                disabled=bool(raw.get("disabled", False)),
                ax_ignored=bool(raw.get("ax_ignored", False)),
                checked=raw.get("checked"),
                expanded=raw.get("expanded"),
                selected=bool(raw.get("selected", False)),
                # Prefer AX-merged value; fall back to live DOM value (JS-captured).
                # `merge_axtree` only writes current_value for controls that have an
                # AX entry, so for unmatched controls we still want the DOM value here.
                current_value=(
                    str(raw["current_value"]) if raw.get("current_value")
                    else (str(raw["domValue"]) if raw.get("domValue") else None)
                ),
                has_popup=str(raw["has_popup"]) if raw.get("has_popup") else None,
                ax_description=(str(raw["ax_description"]) if raw.get("ax_description") else None),
                focusable=bool(raw.get("focusable", False)),
                slider_min=float(raw["sliderMin"]) if raw.get("sliderMin") is not None else None,
                slider_max=float(raw["sliderMax"]) if raw.get("sliderMax") is not None else None,
                slider_value=(
                    float(raw["sliderValue"]) if raw.get("sliderValue") is not None else None
                ),
                slider_orientation=raw.get("sliderOrientation"),
            )
            controls.append(control)
            if abs(utility - noise) < self._tuning.uncertain_band:
                uncertain_ids.append(control.control_id)

        return controls, uncertain_ids

    def _select_controls(self, controls: list[_ControlAtom]) -> list[_ControlAtom]:
        by_id: dict[str, _ControlAtom] = {}

        explicit_form_controls = sorted(
            [
                control
                for control in controls
                if control.form_selector is not None and not control.ax_ignored
            ],
            key=lambda item: item.dom_order,
        )
        for control in explicit_form_controls:
            by_id[control.control_id] = control

        form_associated_controls = sorted(
            [
                control
                for control in controls
                if control.form_selector is None
                and self._is_input_like(control)
                and not control.ax_ignored
            ],
            key=lambda item: (
                item.utility_score - item.noise_score,
                -item.dom_order,
            ),
            reverse=True,
        )
        for control in form_associated_controls[: self._tuning.max_form_associated_controls]:
            by_id[control.control_id] = control

        standalone_controls = sorted(
            controls,
            key=lambda item: (item.utility_score - item.noise_score, item.dom_order),
            reverse=True,
        )
        standalone_count = 0
        for control in standalone_controls:
            if control.control_id in by_id:
                continue
            if control.ax_ignored:
                continue
            if (
                control.utility_score < self._tuning.action_min_utility
                and not self._is_input_like(control)
                and not control.draggable
                and control.tag not in ("canvas", "video", "audio")
            ):
                continue
            by_id[control.control_id] = control
            standalone_count += 1
            if standalone_count >= self._tuning.max_targets_total:
                break

        selected = list(by_id.values())
        selected.sort(key=lambda item: item.dom_order)
        return selected

    def _build_container_closure(
        self,
        containers_raw: list[dict[str, Any]],
        blocks: list[_BlockAtom],
        controls: list[_ControlAtom],
        drop_zone_aids: set[str] | None = None,
        aid_allowlist: set[str] | None = None,
    ) -> tuple[set[str], dict[str, str | None]]:
        parent_by_id: dict[str, str | None] = {}
        root_id = ""
        for raw in containers_raw:
            container_id = str(raw["id"])
            parent_id = str(raw["parentContainerId"]) if raw.get("parentContainerId") else None
            parent_by_id[container_id] = parent_id
            if parent_id is None and not root_id:
                root_id = container_id

        all_container_ids: set[str] = {str(raw["id"]) for raw in containers_raw}

        kept: set[str] = set()
        for block in blocks:
            kept.add(block.container_id)
        for control in controls:
            kept.add(control.container_id)
        if drop_zone_aids:
            for aid in drop_zone_aids:
                if aid in all_container_ids:
                    kept.add(aid)
        if aid_allowlist:
            for aid in aid_allowlist:
                if aid in all_container_ids:
                    kept.add(aid)

        frontier = list(kept)
        while frontier:
            current = frontier.pop()
            if current not in parent_by_id:
                continue
            parent_id = parent_by_id[current]
            if parent_id is None:
                continue
            if parent_id in kept:
                continue
            kept.add(parent_id)
            frontier.append(parent_id)

        if root_id:
            kept.add(root_id)

        return kept, parent_by_id

    def _build_containers(
        self,
        containers_raw: list[dict[str, Any]],
        kept_container_ids: set[str],
        parent_by_id: dict[str, str | None],
        blocks: list[_BlockAtom],
        controls: list[_ControlAtom],
        drop_zone_aids: set[str] | None = None,
    ) -> list[ContainerEntity]:
        blocks_by_container: dict[str, list[_BlockAtom]] = defaultdict(list)
        for block in blocks:
            if block.container_id in kept_container_ids:
                blocks_by_container[block.container_id].append(block)

        controls_by_container: dict[str, list[_ControlAtom]] = defaultdict(list)
        for control in controls:
            if control.container_id in kept_container_ids:
                controls_by_container[control.container_id].append(control)

        containers: list[ContainerEntity] = []
        for raw in containers_raw:
            container_id = str(raw["id"])
            if container_id not in kept_container_ids:
                continue

            container_blocks = sorted(
                blocks_by_container.get(container_id, []),
                key=lambda item: item.dom_order,
            )
            text_blocks = [block.text for block in container_blocks]
            summary = "\n".join(text_blocks).strip()
            if not summary:
                heading = str(raw["heading"]).strip() if raw.get("heading") else ""
                summary = heading

            container_controls = sorted(
                controls_by_container.get(container_id, []),
                key=lambda item: item.dom_order,
            )
            control_refs = [control.control_id for control in container_controls]

            utility, noise, reason_codes = self._score_container(raw, text_blocks, control_refs)

            containers.append(
                ContainerEntity(
                    container_id=container_id,
                    selector=str(raw["selector"]),
                    tag=str(raw["tag"]),
                    role=str(raw["role"]) if raw.get("role") else None,
                    parent_container_id=parent_by_id.get(container_id),
                    heading=str(raw["heading"]) if raw.get("heading") else None,
                    summary=summary,
                    text_blocks=text_blocks,
                    control_refs=control_refs,
                    form_refs=[],
                    button_refs=[],
                    geometry=self._geometry_from_raw(raw),
                    dom_order=int(raw["domOrder"]),
                    z_index=int(raw["zIndex"]),
                    pointer_blocking=bool(raw["pointerBlocking"]),
                    fixed_position=bool(raw["fixedPosition"]),
                    scrollable=bool(raw.get("scrollable", False)),
                    has_animation=bool(raw.get("hasAnimation", False)),
                    data_attributes={
                        str(name): str(value)
                        for name, value in raw.get("dataAttributes", {}).items()
                    },
                    overlay_like=False,
                    section_like=False,
                    is_blocking=False,
                    blocks_container_ids=[],
                    utility_score=utility,
                    noise_score=noise,
                    reason_codes=reason_codes,
                    ax_modal=bool(raw.get("ax_modal", False)),
                    is_drop_zone=(drop_zone_aids is not None and str(raw["id"]) in drop_zone_aids),
                )
            )

        containers.sort(
            key=lambda item: (
                self._selector_depth(item.selector),
                item.geometry.y,
                item.geometry.x,
            )
        )
        return containers

    def _score_container(
        self,
        raw_container: dict[str, Any],
        text_blocks: list[str],
        control_refs: list[str],
    ) -> tuple[float, float, list[str]]:
        utility = 0.0
        noise = 0.0
        reasons: list[str] = []

        heading = str(raw_container["heading"]).strip() if raw_container.get("heading") else ""
        text = " ".join(text_blocks).strip()
        text_len = len(text)
        area_ratio = float(raw_container["areaRatio"])
        tag = str(raw_container["tag"]).lower()

        if tag in ("main", "article", "section", "form", "dialog"):
            utility += 0.18
            reasons.append("semantic_container")

        if heading:
            utility += 0.18
            reasons.append("has_heading")

        if text_len >= 40:
            utility += 0.16
            reasons.append("text_present")
        elif text_len < 12:
            noise += 0.20
            reasons.append("text_too_short")

        if 0.01 <= area_ratio <= 0.80:
            utility += 0.10
            reasons.append("reasonable_area")

        if control_refs:
            utility += min(0.24, 0.04 * len(control_refs))
            reasons.append("has_controls")

        noise_hits = self._count_noise_hits(" ".join((heading, text)))
        if noise_hits > 0:
            noise += min(0.35, 0.12 * noise_hits)
            reasons.append("noise_keyword")

        return self._clamp(utility, 0.0, 1.0), self._clamp(noise, 0.0, 1.0), reasons

    async def _collect_blocking_relations(
        self,
        containers: list[ContainerEntity],
    ) -> dict[str, list[str]]:
        """Geometric blocking: N² bbox enumeration → per-target EFP verification.

        1. Python enumerates all candidate blocker→target pairs via rectangle
           intersection + ancestry exclusion + viewport + high-z/fixed filter.
           No cap — pure in-memory geometry on data we already have.
        2. Group candidate blockers by target.
        3. One JS page.evaluate: for each unique target, sample 5 points on its
           bbox (center + 4 inset corners), call elementsFromPoint, check every
           candidate blocker against those shared stacks. Return per-pair hits.
        4. Python applies 60% hit threshold + cycle prevention.
        """
        by_id = {container.container_id: container for container in containers}
        ancestor_sets = self._build_ancestor_sets(containers)

        targets_map: dict[str, list[str]] = defaultdict(list)

        for blocker in containers:
            if not blocker.pointer_blocking:
                continue
            blocker_area = blocker.geometry.w * blocker.geometry.h
            if blocker_area < self._tuning.blocking_min_candidate_area_ratio:
                continue
            if not self._intersects_viewport(blocker.geometry):
                continue

            for target in containers:
                if blocker.container_id == target.container_id:
                    continue
                if blocker.container_id in ancestor_sets.get(target.container_id, set()):
                    continue
                if target.container_id in ancestor_sets.get(blocker.container_id, set()):
                    continue
                target_area = target.geometry.w * target.geometry.h
                if target_area < self._tuning.blocking_min_candidate_area_ratio:
                    continue
                if not self._intersects_viewport(target.geometry):
                    continue
                if not self._geometry_intersects(blocker.geometry, target.geometry):
                    continue
                if not (
                    blocker.fixed_position
                    or target.fixed_position
                    or blocker.z_index >= self._tuning.overlay_candidate_min_z_index
                    or target.z_index >= self._tuning.overlay_candidate_min_z_index
                ):
                    continue
                targets_map[target.container_id].append(blocker.container_id)

        if not targets_map:
            return {}

        target_specs = []
        for target_id, blocker_ids in targets_map.items():
            target_specs.append(
                {
                    "target_id": target_id,
                    "target_aid": target_id,
                    "blocker_ids": blocker_ids,
                    "blocker_aids": blocker_ids,
                }
            )

        raw_result = await self._sm.page.evaluate(
            """
            ({ targets, insetPx, samplePoints }) => {
              const vw = window.innerWidth || 1;
              const vh = window.innerHeight || 1;

              const clipRectToViewport = (rect) => {
                const left = Math.max(0, rect.left);
                const top = Math.max(0, rect.top);
                const right = Math.min(vw, rect.right);
                const bottom = Math.min(vh, rect.bottom);
                const width = right - left;
                const height = bottom - top;
                if (width <= 1 || height <= 1) return null;
                return { left, top, right, bottom, width, height };
              };

              const pointsForRect = (rect, maxPoints) => {
                const inset = Math.min(
                  insetPx, Math.max(1, Math.min(rect.width, rect.height) / 4)
                );
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                const candidates = [
                  [centerX, centerY],
                  [rect.left + inset, rect.top + inset],
                  [rect.right - inset, rect.top + inset],
                  [rect.left + inset, rect.bottom - inset],
                  [rect.right - inset, rect.bottom - inset],
                ];
                return candidates.slice(0, Math.max(1, Math.min(maxPoints, candidates.length)));
              };

              /* Pre-build AID→node map for all unique AIDs */
              const aidSet = new Set();
              for (const t of targets) {
                aidSet.add(t.target_aid);
                for (const bAid of t.blocker_aids) aidSet.add(bAid);
              }
              const nodeByAid = new Map();
              for (const aid of aidSet) {
                const node = document.querySelector(`[data-cdx-aid="${aid}"]`);
                if (node) nodeByAid.set(aid, node);
              }

              let missingTargetNodes = 0;
              const out = [];

              for (const spec of targets) {
                const targetNode = nodeByAid.get(spec.target_aid);
                if (!targetNode) { missingTargetNodes++; continue; }

                const rawRect = targetNode.getBoundingClientRect();
                if (rawRect.width <= 1 || rawRect.height <= 1) continue;
                const rect = clipRectToViewport(rawRect);
                if (!rect) continue;

                /* Sample points ONCE for this target */
                const points = pointsForRect(rect, samplePoints);
                const stacks = [];
                for (const [xRaw, yRaw] of points) {
                  const x = Math.min(vw - 1, Math.max(0, Math.floor(xRaw)));
                  const y = Math.min(vh - 1, Math.max(0, Math.floor(yRaw)));
                  stacks.push(document.elementsFromPoint(x, y));
                }

                /* Check each candidate blocker against the shared stacks */
                for (let bi = 0; bi < spec.blocker_aids.length; bi++) {
                  const blockerAid = spec.blocker_aids[bi];
                  const blockerId = spec.blocker_ids[bi];
                  const blockerNode = nodeByAid.get(blockerAid);
                  if (!blockerNode) continue;

                  let hits = 0;
                  let valid = 0;
                  for (const stack of stacks) {
                    if (!stack || stack.length === 0) continue;
                    let foundBlocker = false;
                    let foundTarget = false;
                    for (const el of stack) {
                      const isBlocker = el === blockerNode || blockerNode.contains(el);
                      const isTarget = el === targetNode || targetNode.contains(el);
                      if (isBlocker && !foundTarget) {
                        foundBlocker = true;
                        break;
                      }
                      if (isTarget && !foundBlocker) {
                        foundTarget = true;
                        break;
                      }
                    }
                    if (foundBlocker) {
                      let targetPresent = false;
                      for (const el of stack) {
                        if (el === targetNode || targetNode.contains(el)) {
                          targetPresent = true;
                          break;
                        }
                      }
                      if (targetPresent) { hits++; valid++; }
                    } else if (foundTarget) {
                      valid++;
                    }
                  }
                  if (valid <= 0) continue;
                  out.push({
                    blocker_id: blockerId,
                    target_id: spec.target_id,
                    hits,
                    valid,
                  });
                }
              }

              return { edges: out, missingTargetNodes };
            }
            """,
            {
                "targets": target_specs,
                "insetPx": self._tuning.occlusion_inset_px,
                "samplePoints": self._tuning.blocking_sample_points,
            },
        )

        missing_target_count = int(raw_result.get("missingTargetNodes", 0))
        if missing_target_count:
            print(f"  [BLOCKING] missing target nodes: {missing_target_count}")

        edge_candidates: list[dict[str, Any]] = []
        for raw in raw_result.get("edges", []):
            blocker_id = str(raw["blocker_id"])
            target_id = str(raw["target_id"])
            if blocker_id == target_id:
                continue
            if blocker_id not in by_id or target_id not in by_id:
                continue
            if blocker_id in ancestor_sets.get(target_id, set()):
                continue
            if target_id in ancestor_sets.get(blocker_id, set()):
                continue
            hits = int(raw["hits"])
            valid = int(raw["valid"])
            ratio = hits / valid if valid > 0 else 0.0
            if ratio < self._tuning.blocking_hit_threshold or hits <= 0:
                continue
            edge_candidates.append(
                {
                    "blocker_id": blocker_id,
                    "target_id": target_id,
                    "ratio": ratio,
                    "hits": hits,
                }
            )

        edge_candidates.sort(
            key=lambda item: (
                float(item["ratio"]),
                int(item["hits"]),
                str(item["blocker_id"]),
                str(item["target_id"]),
            ),
            reverse=True,
        )

        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edge_candidates:
            blocker_id = str(edge["blocker_id"])
            target_id = str(edge["target_id"])
            if self._path_exists(adjacency, target_id, blocker_id):
                continue
            adjacency[blocker_id].add(target_id)

        validated: dict[str, list[str]] = {}
        for blocker_id, targets in adjacency.items():
            cleaned_targets: list[str] = []
            for target_id in sorted(targets, key=lambda item: by_id[item].dom_order):
                if blocker_id == target_id:
                    continue
                if blocker_id in ancestor_sets.get(target_id, set()):
                    continue
                if target_id in ancestor_sets.get(blocker_id, set()):
                    continue
                cleaned_targets.append(target_id)
            if cleaned_targets:
                validated[blocker_id] = cleaned_targets
        return validated

    def _build_ancestor_sets(self, containers: list[ContainerEntity]) -> dict[str, set[str]]:
        parent_by_id = {
            container.container_id: container.parent_container_id for container in containers
        }
        ancestors: dict[str, set[str]] = {}
        for container_id in parent_by_id:
            seen: set[str] = set()
            current = parent_by_id.get(container_id)
            while current is not None:
                if current in seen:
                    break
                seen.add(current)
                current = parent_by_id.get(current)
            ancestors[container_id] = seen
        return ancestors

    def _path_exists(
        self,
        adjacency: dict[str, set[str]],
        source_id: str,
        target_id: str,
    ) -> bool:
        if source_id == target_id:
            return True
        stack = [source_id]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            for next_id in adjacency.get(current, set()):
                if next_id not in seen:
                    stack.append(next_id)
        return False

    def _apply_blocking_relations(
        self,
        containers: list[ContainerEntity],
        blocking_map: dict[str, list[str]],
    ) -> list[ContainerEntity]:
        output: list[ContainerEntity] = []
        for container in containers:
            blocked_ids = blocking_map.get(container.container_id, [])
            output.append(
                container.model_copy(
                    update={
                        "is_blocking": bool(blocked_ids),
                        "blocks_container_ids": blocked_ids,
                    }
                )
            )
        return output

    def _build_forms(
        self,
        controls: list[_ControlAtom],
        container_by_id: dict[str, ContainerEntity],
    ) -> tuple[list[FormEntity], list[FormControl]]:
        """Returns (forms, orphan_controls). Orphans are input-like, visible,
        labeled controls that didn't get claimed by any form — typically a lone
        <input> in a React-managed <div> with no <form> ancestor (Path A fix).
        """
        controls_by_explicit_form: dict[str, list[_ControlAtom]] = defaultdict(list)
        controls_by_container: dict[str, list[_ControlAtom]] = defaultdict(list)

        for control in controls:
            if control.container_id not in container_by_id:
                continue
            if control.form_selector is not None:
                controls_by_explicit_form[control.form_selector].append(control)
            else:
                controls_by_container[control.container_id].append(control)

        forms: list[FormEntity] = []
        form_counter = 1

        explicit_control_ids: set[str] = set()
        for selector, form_controls in controls_by_explicit_form.items():
            ordered_controls = sorted(form_controls, key=lambda item: item.dom_order)
            explicit_control_ids.update(control.control_id for control in ordered_controls)
            form = self._build_form_entity(
                form_id=f"form-{form_counter}",
                selector=selector,
                owner_container_id=self._majority_owner_container(ordered_controls),
                controls=ordered_controls,
                container_by_id=container_by_id,
            )
            if form is not None:
                forms.append(form)
                form_counter += 1

        for container_id, container_controls in controls_by_container.items():
            candidate_controls = [
                control
                for control in sorted(container_controls, key=lambda item: item.dom_order)
                if control.control_id not in explicit_control_ids
            ]
            if not candidate_controls:
                continue

            has_media = any(c.tag in ("canvas", "video", "audio") for c in candidate_controls)
            if (
                len(candidate_controls) < self._tuning.form_min_controls_for_container
                and not has_media
            ):
                continue
            non_button_count = sum(
                1 for control in candidate_controls if self._is_input_like(control)
            )
            if non_button_count < self._tuning.form_min_non_button_controls_for_container:
                continue
            ratio = non_button_count / max(1, len(candidate_controls))
            if ratio < self._tuning.form_min_non_button_ratio_for_pseudo:
                continue

            form = self._build_form_entity(
                form_id=f"form-{form_counter}",
                selector=container_by_id[container_id].selector,
                owner_container_id=container_id,
                controls=candidate_controls,
                container_by_id=container_by_id,
            )
            if form is not None:
                forms.append(form)
                form_counter += 1

        # Sibling aggregation: radio/checkbox groups where each option is in its own wrapper.
        claimed_control_ids: set[str] = set(explicit_control_ids)
        for form in forms:
            for group in form.groups:
                for ctrl in group.controls:
                    claimed_control_ids.add(ctrl.control_id)

        children_map: dict[str, list[str]] = defaultdict(list)
        for container in container_by_id.values():
            pid = container.parent_container_id
            if pid and pid in container_by_id:
                children_map[pid].append(container.container_id)

        for parent_id, child_ids in children_map.items():
            sibling_controls: list[_ControlAtom] = []
            for child_id in child_ids:
                child_controls = [
                    c
                    for c in controls_by_container.get(child_id, [])
                    if c.control_id not in claimed_control_ids and self._is_input_like(c)
                ]
                if len(child_controls) != 1:
                    continue
                sibling_controls.append(child_controls[0])

            if len(sibling_controls) < self._tuning.form_min_controls_for_container:
                continue

            roles = {c.role for c in sibling_controls if c.role}
            if len(roles) > 1:
                continue
            if not roles:
                type_keys = {(c.tag, c.type) for c in sibling_controls}
                if len(type_keys) > 1:
                    continue

            ordered = sorted(sibling_controls, key=lambda item: item.dom_order)
            form = self._build_form_entity(
                form_id=f"form-{form_counter}",
                selector=container_by_id[parent_id].selector,
                owner_container_id=parent_id,
                controls=ordered,
                container_by_id=container_by_id,
            )
            if form is not None:
                forms.append(form)
                form_counter += 1
                for ctrl in ordered:
                    claimed_control_ids.add(ctrl.control_id)

        # Draggable piece pass — group unclaimed draggables by container.
        draggable_by_container: dict[str, list[_ControlAtom]] = defaultdict(list)
        for control in controls:
            if (
                control.draggable
                and control.control_id not in claimed_control_ids
                and control.container_id in container_by_id
            ):
                draggable_by_container[control.container_id].append(control)

        for container_id, drag_controls in draggable_by_container.items():
            ordered = sorted(drag_controls, key=lambda item: item.dom_order)
            form = self._build_form_entity(
                form_id=f"form-{form_counter}",
                selector=container_by_id[container_id].selector,
                owner_container_id=container_id,
                controls=ordered,
                container_by_id=container_by_id,
            )
            if form is not None:
                forms.append(form)
                form_counter += 1
                for ctrl in ordered:
                    claimed_control_ids.add(ctrl.control_id)

        capped_forms = forms[: self._tuning.max_forms_total]

        # Path A — orphan controls. Input-like, visible, labeled controls that
        # no form claimed. These render in V5 as bare children of their owner
        # container (e.g. swiggy's lone <input id="location"> with no <form>).
        # Limit to a generous cap to avoid runaway noise on pathological pages.
        capped_form_control_ids: set[str] = set()
        for form in capped_forms:
            for group in form.groups:
                for ctrl in group.controls:
                    capped_form_control_ids.add(ctrl.control_id)
        orphan_controls: list[FormControl] = []
        for control in controls:
            if control.control_id in capped_form_control_ids:
                continue
            if not self._is_input_like(control):
                continue
            if not control.visible_now or not control.in_viewport_now:
                continue
            if control.tag == "input" and (control.type or "").lower() == "hidden":
                continue
            label_text = (control.label or "").strip() or (control.name or "").strip()
            if not label_text and control.tag != "select":
                # Select boxes can be useful even without a label; everything
                # else needs SOME hint of purpose to be worth rendering.
                continue
            if control.container_id not in container_by_id:
                continue
            orphan_controls.append(
                FormControl(
                    control_id=control.control_id,
                    owner_container_id=control.container_id,
                    text=control.text,
                    selector=control.selector,
                    tag=control.tag,
                    role=control.role,
                    type=control.type,
                    name=control.name,
                    label=control.label,
                    options=control.options[: self._tuning.max_options_per_form],
                    geometry=control.geometry,
                    visible_now=control.visible_now,
                    in_viewport_now=control.in_viewport_now,
                    occlusion=self._default_occlusion(control.visible_now),
                    ax_name=control.ax_name,
                    ax_role=control.ax_role,
                    disabled=control.disabled,
                    ax_ignored=control.ax_ignored,
                    checked=control.checked,
                    expanded=control.expanded,
                    selected=control.selected,
                    current_value=control.current_value,
                    has_popup=control.has_popup,
                    ax_description=control.ax_description,
                    focusable=control.focusable,
                    cursor=control.cursor,
                    draggable=control.draggable,
                    duplicate_id=control.duplicate_id,
                    slider_min=control.slider_min,
                    slider_max=control.slider_max,
                    slider_value=control.slider_value,
                    slider_orientation=control.slider_orientation,
                )
            )

        return capped_forms, orphan_controls

    def _build_form_entity(
        self,
        form_id: str,
        selector: str,
        owner_container_id: str,
        controls: list[_ControlAtom],
        container_by_id: dict[str, ContainerEntity],
    ) -> FormEntity | None:
        grouped: dict[str, list[_ControlAtom]] = defaultdict(list)
        for control in controls:
            grouped[control.container_id].append(control)

        groups: list[FormControlGroup] = []
        for container_id, group_controls in sorted(grouped.items(), key=lambda item: item[0]):
            sorted_controls = sorted(group_controls, key=lambda item: item.dom_order)
            output_controls: list[FormControl] = [
                self._to_form_control(control)
                for control in sorted_controls[: self._tuning.max_controls_per_form]
            ]
            if not output_controls:
                continue
            groups.append(
                FormControlGroup(
                    container_id=container_id,
                    controls=output_controls,
                )
            )

        if not groups:
            return None

        owner = container_by_id.get(owner_container_id)
        prompt_heading = owner.heading if owner is not None else None
        prompt_text = None
        if owner is not None:
            if owner.text_blocks:
                prompt_text = owner.text_blocks[0]
            elif owner.summary:
                prompt_text = owner.summary

        return FormEntity(
            form_id=form_id,
            owner_container_id=owner_container_id,
            selector=selector,
            prompt_heading=prompt_heading,
            prompt_text=prompt_text,
            groups=groups,
            form_blocker=FormBlockerStatus(is_blocked=False),
        )

    def _majority_owner_container(self, controls: list[_ControlAtom]) -> str:
        counts: Counter[str] = Counter(control.container_id for control in controls)
        return counts.most_common(1)[0][0]

    def _collect_form_control_ids(self, forms: list[FormEntity]) -> set[str]:
        control_ids: set[str] = set()
        for form in forms:
            for group in form.groups:
                for control in group.controls:
                    control_ids.add(control.control_id)
        return control_ids

    def _build_buttons(
        self,
        controls: list[_ControlAtom],
        container_by_id: dict[str, ContainerEntity],
        exclude_ids: set[str],
    ) -> tuple[list[ButtonEntity], int]:
        button_controls = [
            control
            for control in controls
            if self._is_button_like(control)
            and control.container_id in container_by_id
            and control.control_id not in exclude_ids
        ]
        ranked = sorted(
            button_controls,
            key=lambda item: (item.utility_score - item.noise_score, -item.dom_order),
            reverse=True,
        )
        before_cap_count = len(ranked)

        buttons: list[ButtonEntity] = []
        for control in ranked:
            buttons.append(
                ButtonEntity(
                    button_id=control.control_id,
                    owner_container_id=control.container_id,
                    text=control.text,
                    selector=control.selector,
                    href=control.href,
                    geometry=control.geometry,
                    occlusion=self._default_occlusion(control.visible_now),
                    utility_score=control.utility_score,
                    noise_score=control.noise_score,
                    reason_codes=control.reason_codes,
                    visible_now=control.visible_now,
                    in_viewport_now=control.in_viewport_now,
                    z_index=control.z_index,
                    ax_name=control.ax_name,
                    disabled=control.disabled,
                    has_popup=control.has_popup,
                    cursor=control.cursor,
                )
            )
            if len(buttons) >= self._tuning.max_buttons_total:
                break

        buttons.sort(key=lambda item: item.geometry.y)
        dropped_count = max(0, before_cap_count - len(buttons))
        return buttons, dropped_count

    @staticmethod
    def _has_scrollable_ancestor(
        owner_container_id: str,
        container_by_id: dict[str, ContainerEntity],
    ) -> bool:
        """Walk the parent chain to check if any ancestor is scrollable."""
        current = owner_container_id
        while current:
            c = container_by_id.get(current)
            if c is None:
                break
            if c.scrollable:
                return True
            current = c.parent_container_id
        return False

    def _build_occlusion_target_specs(
        self,
        buttons: list[ButtonEntity],
        forms: list[FormEntity],
        container_by_id: dict[str, ContainerEntity],
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for form in forms:
            if form.form_id in seen_ids:
                continue
            seen_ids.add(form.form_id)
            # Skip form-level occlusion inside scrollable containers — per-element checking
            # there reports scroll-position-dependent state as permanent blocking.
            if self._has_scrollable_ancestor(form.owner_container_id, container_by_id):
                continue
            specs.append(
                {
                    "target_id": form.form_id,
                    "selector": form.selector,
                    "owner_container_id": form.owner_container_id,
                    "visible_now": False,
                    "in_viewport_now": False,
                    "target_kind": "form",
                }
            )

        for button in buttons:
            if button.button_id in seen_ids:
                continue
            seen_ids.add(button.button_id)
            if self._has_scrollable_ancestor(button.owner_container_id, container_by_id):
                continue
            specs.append(
                {
                    "target_id": button.button_id,
                    "selector": button.selector,
                    "owner_container_id": button.owner_container_id,
                    "visible_now": button.visible_now,
                    "in_viewport_now": button.in_viewport_now,
                    "target_kind": "control",
                }
            )

        for form in forms:
            for group in form.groups:
                for control in group.controls:
                    if control.control_id in seen_ids:
                        continue
                    seen_ids.add(control.control_id)
                    if self._has_scrollable_ancestor(control.owner_container_id, container_by_id):
                        continue
                    specs.append(
                        {
                            "target_id": control.control_id,
                            "selector": control.selector,
                            "owner_container_id": control.owner_container_id,
                            "visible_now": control.visible_now,
                            "in_viewport_now": control.in_viewport_now,
                            "target_kind": "control",
                        }
                    )

        return specs[: self._tuning.max_occlusion_targets]

    async def _collect_occlusion(
        self,
        target_specs: list[dict[str, Any]],
        container_handle_specs: list[dict[str, str]],
        root_container_id: str | None,
    ) -> dict[str, dict[str, Any]]:
        check_targets = [
            {
                "target_id": item["target_id"],
                "selector": item["selector"],
                "owner_container_id": item["owner_container_id"],
            }
            for item in target_specs
            if item.get("target_kind") == "form"
            or (bool(item["visible_now"]) and bool(item["in_viewport_now"]))
        ]
        if not check_targets:
            return {}

        return await self._sm.page.evaluate(
            """
            ({ targets, containers, rootContainerId, insetPx, samplePoints }) => {
              const vw = window.innerWidth || 1;
              const vh = window.innerHeight || 1;

              const clipRectToViewport = (rect) => {
                const left = Math.max(0, rect.left);
                const top = Math.max(0, rect.top);
                const right = Math.min(vw, rect.right);
                const bottom = Math.min(vh, rect.bottom);
                const width = right - left;
                const height = bottom - top;
                if (width <= 1 || height <= 1) return null;
                return { left, top, right, bottom, width, height };
              };

              const containerIds = new Set();
              const parentOf = new Map();
              for (const item of containers) {
                const containerId = String(item.container_id || "");
                if (!containerId) continue;
                containerIds.add(containerId);
                const pid = String(item.parent_container_id || "");
                if (pid) parentOf.set(containerId, pid);
              }

              const isAncestorOf = (candidateId, descendantId) => {
                let current = descendantId;
                const visited = new Set();
                while (current) {
                  if (visited.has(current)) break;
                  visited.add(current);
                  if (current === candidateId) return true;
                  current = parentOf.get(current);
                }
                return false;
              };

              const containerIdForNode = (node) => {
                let current = node;
                while (current && current.nodeType === 1) {
                  const aid = current.getAttribute("data-cdx-aid");
                  if (containerIds.has(aid)) {
                    if (rootContainerId && aid === rootContainerId) {
                      current = current.parentElement;
                      continue;
                    }
                    return aid;
                  }
                  current = current.parentElement;
                }
                return null;
              };

              const pointsForRect = (rect, maxPoints) => {
                const inset = Math.min(insetPx, Math.max(1, Math.min(rect.width, rect.height) / 4));
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                const candidates = [
                  [centerX, centerY],
                  [rect.left + inset, rect.top + inset],
                  [rect.right - inset, rect.top + inset],
                  [rect.left + inset, rect.bottom - inset],
                  [rect.right - inset, rect.bottom - inset],
                ];
                return candidates.slice(0, Math.max(1, Math.min(maxPoints, candidates.length)));
              };

              const result = {};
              for (const target of targets) {
                let node = null;
                try { node = document.querySelector(target.selector); }
                catch (e) { node = null; }
                if (!node) {
                  result[target.target_id] = {
                    checked: false,
                    is_occluded: false,
                    blocked_points: 0,
                    total_points: 0,
                    primary_blocker_selector: null,
                    blocker_container_ids: [],
                    primary_blocker_z_index: null,
                    estimated_overlap_ratio: null,
                  };
                  continue;
                }

                const rawRect = node.getBoundingClientRect();
                if (rawRect.width <= 1 || rawRect.height <= 1) {
                  result[target.target_id] = {
                    checked: false,
                    is_occluded: false,
                    blocked_points: 0,
                    total_points: 0,
                    primary_blocker_selector: null,
                    blocker_container_ids: [],
                    primary_blocker_z_index: null,
                    estimated_overlap_ratio: null,
                  };
                  continue;
                }
                const rect = clipRectToViewport(rawRect);
                if (!rect) {
                  result[target.target_id] = {
                    checked: false,
                    is_occluded: false,
                    blocked_points: 0,
                    total_points: 0,
                    primary_blocker_selector: null,
                    blocker_container_ids: [],
                    primary_blocker_z_index: null,
                    estimated_overlap_ratio: null,
                  };
                  continue;
                }

                const points = pointsForRect(rect, samplePoints);
                let blockedPoints = 0;
                const blockerHits = new Map();

                for (const [xRaw, yRaw] of points) {
                  const x = Math.min(vw - 1, Math.max(0, Math.floor(xRaw)));
                  const y = Math.min(vh - 1, Math.max(0, Math.floor(yRaw)));
                  const stack = document.elementsFromPoint(x, y);
                  if (!stack || stack.length === 0) continue;

                  /* Walk stack top-to-bottom. Find ALL containers above
                     the target node — not just the topmost. This detects
                     the full blocking chain (2->3 even when 1 is above both). */
                  let foundTarget = false;
                  let pointBlocked = false;
                  for (const el of stack) {
                    if (el === node || node.contains(el)) {
                      foundTarget = true;
                      break;
                    }
                    const cid = containerIdForNode(el);
                    if (!cid) continue;
                    const isOwnerOrAncestor = cid === target.owner_container_id
                      || isAncestorOf(cid, target.owner_container_id);
                    if (isOwnerOrAncestor) {
                      foundTarget = true;
                      break;
                    }
                    pointBlocked = true;
                    blockerHits.set(cid, (blockerHits.get(cid) || 0) + 1);
                  }
                  if (pointBlocked) blockedPoints += 1;
                }

                const blockerIds = [];
                for (const [containerId, hitCount] of blockerHits.entries()) {
                  if (hitCount > 0) blockerIds.push(containerId);
                }

                const totalPoints = points.length;
                const overlapRatio = totalPoints > 0 ? blockedPoints / totalPoints : 0;

                result[target.target_id] = {
                  checked: true,
                  is_occluded: blockedPoints > 0,
                  blocked_points: blockedPoints,
                  total_points: totalPoints,
                  primary_blocker_selector: null,
                  blocker_container_ids: blockerIds,
                  primary_blocker_z_index: null,
                  estimated_overlap_ratio: overlapRatio,
                };
              }

              return result;
            }
            """,
            {
                "targets": check_targets,
                "containers": container_handle_specs,
                "rootContainerId": root_container_id,
                "insetPx": self._tuning.occlusion_inset_px,
                "samplePoints": self._tuning.occlusion_sample_points,
            },
        )

    def _apply_button_occlusion(
        self,
        buttons: list[ButtonEntity],
        occlusion_map: dict[str, dict[str, Any]],
    ) -> list[ButtonEntity]:
        output: list[ButtonEntity] = []
        for button in buttons:
            raw = occlusion_map.get(button.button_id)
            if raw is None:
                output.append(button)
                continue
            output.append(
                button.model_copy(
                    update={
                        "occlusion": TargetOcclusion(
                            checked=bool(raw["checked"]),
                            is_occluded=bool(raw["is_occluded"]),
                            blocked_points=int(raw["blocked_points"]),
                            total_points=int(raw["total_points"]),
                            primary_blocker_selector=(
                                str(raw["primary_blocker_selector"])
                                if raw.get("primary_blocker_selector")
                                else None
                            ),
                            blocker_container_ids=[
                                str(b) for b in raw.get("blocker_container_ids", [])
                            ],
                            primary_blocker_z_index=(
                                int(raw["primary_blocker_z_index"])
                                if raw.get("primary_blocker_z_index") is not None
                                else None
                            ),
                            estimated_overlap_ratio=(
                                float(raw["estimated_overlap_ratio"])
                                if raw.get("estimated_overlap_ratio") is not None
                                else None
                            ),
                            occlusion_unknown_until_visible=False,
                        )
                    }
                )
            )
        return output

    def _apply_form_occlusion(
        self,
        forms: list[FormEntity],
        occlusion_map: dict[str, dict[str, Any]],
        blocking_by_container: dict[str, list[str]],
    ) -> list[FormEntity]:
        reverse_blocking: dict[str, list[str]] = defaultdict(list)
        for blocker_id, blocked_ids in blocking_by_container.items():
            for blocked_id in blocked_ids:
                reverse_blocking[blocked_id].append(blocker_id)

        output_forms: list[FormEntity] = []
        for form in forms:
            updated_groups: list[FormControlGroup] = []
            blocked_by: list[str] = []

            for group in form.groups:
                updated_controls: list[FormControl] = []
                for control in group.controls:
                    raw = occlusion_map.get(control.control_id)
                    if raw is None:
                        updated_controls.append(control)
                        continue
                    occlusion = TargetOcclusion(
                        checked=bool(raw["checked"]),
                        is_occluded=bool(raw["is_occluded"]),
                        blocked_points=int(raw["blocked_points"]),
                        total_points=int(raw["total_points"]),
                        primary_blocker_selector=(
                            str(raw["primary_blocker_selector"])
                            if raw.get("primary_blocker_selector")
                            else None
                        ),
                        blocker_container_ids=[
                            str(b) for b in raw.get("blocker_container_ids", [])
                        ],
                        primary_blocker_z_index=(
                            int(raw["primary_blocker_z_index"])
                            if raw.get("primary_blocker_z_index") is not None
                            else None
                        ),
                        estimated_overlap_ratio=(
                            float(raw["estimated_overlap_ratio"])
                            if raw.get("estimated_overlap_ratio") is not None
                            else None
                        ),
                        occlusion_unknown_until_visible=False,
                    )
                    if occlusion.is_occluded and occlusion.blocker_container_ids:
                        blocked_by.extend(occlusion.blocker_container_ids)
                    updated_controls.append(control.model_copy(update={"occlusion": occlusion}))

                updated_groups.append(group.model_copy(update={"controls": updated_controls}))

            blocker_id = None
            is_blocked = False

            # Source 1: form-level direct occlusion (compositor truth, highest priority)
            form_occlusion = occlusion_map.get(form.form_id)
            if (
                form_occlusion
                and bool(form_occlusion.get("checked"))
                and bool(form_occlusion.get("is_occluded"))
            ):
                candidates = form_occlusion.get("blocker_container_ids", [])
                if candidates:
                    blocker_id = str(candidates[0])
                    is_blocked = True

            # Source 2: per-control occlusion
            if not is_blocked and blocked_by:
                blocker_id = Counter(blocked_by).most_common(1)[0][0]
                is_blocked = True

            # Source 3: container-to-container blocking
            if not is_blocked:
                blockers = reverse_blocking.get(form.owner_container_id, [])
                if blockers:
                    blocker_id = blockers[0]
                    is_blocked = True

            output_forms.append(
                form.model_copy(
                    update={
                        "groups": updated_groups,
                        "form_blocker": FormBlockerStatus(
                            is_blocked=is_blocked,
                            blocker_container_id=blocker_id,
                            reason_codes=["container_blocked"] if is_blocked else [],
                        ),
                    }
                )
            )

        return output_forms

    def _attach_container_refs(
        self,
        containers: list[ContainerEntity],
        controls: list[_ControlAtom],
        buttons: list[ButtonEntity],
        forms: list[FormEntity],
        parent_by_id: dict[str, str | None] | None = None,
    ) -> list[ContainerEntity]:
        surviving_ids = {c.container_id for c in containers}

        def _resolve_container(cid: str) -> str:
            """Walk up the full parent chain to find the nearest surviving container."""
            if cid in surviving_ids:
                return cid
            if not parent_by_id:
                return cid
            visited: set[str] = set()
            current = cid
            while current and current not in visited:
                visited.add(current)
                parent = parent_by_id.get(current)
                if parent and parent in surviving_ids:
                    return parent
                current = parent
            return containers[0].container_id if containers else cid

        control_refs_by_container: dict[str, list[str]] = defaultdict(list)
        for control in controls:
            control_refs_by_container[_resolve_container(control.container_id)].append(
                control.control_id
            )

        button_refs_by_container: dict[str, list[str]] = defaultdict(list)
        for button in buttons:
            button_refs_by_container[button.owner_container_id].append(button.button_id)

        form_refs_by_container: dict[str, list[str]] = defaultdict(list)
        for form in forms:
            form_refs_by_container[form.owner_container_id].append(form.form_id)

        output: list[ContainerEntity] = []
        for container in containers:
            output.append(
                container.model_copy(
                    update={
                        "control_refs": self._dedupe_preserve_order(
                            control_refs_by_container.get(container.container_id, [])
                        ),
                        "button_refs": self._dedupe_preserve_order(
                            button_refs_by_container.get(container.container_id, [])
                        ),
                        "form_refs": self._dedupe_preserve_order(
                            form_refs_by_container.get(container.container_id, [])
                        ),
                    }
                )
            )
        return output

    def _classify_container_labels(
        self,
        containers: list[ContainerEntity],
    ) -> list[ContainerEntity]:
        output: list[ContainerEntity] = []
        for container in containers:
            heading = (container.heading or "").lower()
            summary = (container.summary or "").lower()
            tag = container.tag.lower()
            role = (container.role or "").lower()
            semantic_text = f"{heading} {summary}"
            area_ratio = container.geometry.w * container.geometry.h
            has_actions = bool(container.form_refs or container.button_refs)
            has_text = bool(container.text_blocks) or bool(summary.strip())

            semantic_overlay = (
                tag == "dialog"
                or role in ("dialog", "alertdialog")
                or self._contains_any(
                    semantic_text,
                    (
                        "popup",
                        "modal",
                        "overlay",
                        "cookie",
                        "consent",
                        "newsletter",
                        "alert",
                        "interstitial",
                        "subscribe",
                        "offer",
                        "deal",
                        "prize",
                        "notice",
                    ),
                )
            )
            geometric_overlay = (
                container.pointer_blocking
                and (
                    container.fixed_position
                    or container.z_index >= self._tuning.overlay_candidate_min_z_index
                )
                and area_ratio >= self._tuning.overlay_candidate_min_area_ratio
            )
            overlay_like = container.is_blocking or semantic_overlay or geometric_overlay

            semantic_section = tag in ("main", "article", "section", "form", "nav", "aside")
            action_context_signal = has_actions and self._contains_any(
                summary,
                (
                    "reveal",
                    "code",
                    "submit",
                    "select",
                    "enter",
                    "continue",
                    "proceed",
                    "step",
                ),
            )
            section_like = (
                (not overlay_like)
                and (not container.is_blocking)
                and has_text
                and (semantic_section or area_ratio >= 0.01 or action_context_signal)
            )

            output.append(
                container.model_copy(
                    update={
                        "overlay_like": overlay_like,
                        "section_like": section_like,
                    }
                )
            )
        return output

    def _build_actions(
        self,
        forms: list[FormEntity],
        buttons: list[ButtonEntity],
    ) -> list[ActionCandidate]:
        actions: list[ActionCandidate] = []

        for form in forms:
            actions.append(
                ActionCandidate(
                    action_id=f"act-form-{form.form_id}",
                    kind="complete_form",
                    form_id=form.form_id,
                    priority_score=0.70,
                    blocked_now=form.form_blocker.is_blocked,
                    blocker_container_id=form.form_blocker.blocker_container_id,
                )
            )

        for button in buttons:
            actions.append(
                ActionCandidate(
                    action_id=f"act-click-{button.button_id}",
                    kind="click_button",
                    button_id=button.button_id,
                    priority_score=self._clamp(button.utility_score - button.noise_score, 0.0, 1.0),
                    blocked_now=button.occlusion.is_occluded,
                    blocker_container_id=(
                        button.occlusion.blocker_container_ids[0]
                        if button.occlusion.blocker_container_ids
                        else None
                    ),
                )
            )

        actions.sort(
            key=lambda item: (
                item.blocked_now,
                -item.priority_score,
                item.action_id,
            )
        )
        return actions[: self._tuning.max_actions_total]

    def _default_occlusion(self, visible_now: bool) -> TargetOcclusion:
        return TargetOcclusion(
            checked=False,
            is_occluded=False,
            blocked_points=0,
            total_points=0,
            primary_blocker_selector=None,
            primary_blocker_z_index=None,
            estimated_overlap_ratio=None,
            occlusion_unknown_until_visible=not visible_now,
        )

    def _score_control(
        self,
        raw: dict[str, Any],
        page_host: str,
    ) -> tuple[float, float, list[str]]:
        utility = 0.0
        noise = 0.0
        reasons: list[str] = []

        tag = str(raw["tag"]).lower()
        role = str(raw["role"]).lower() if raw.get("role") else ""
        text = str(raw["text"]).lower() if raw.get("text") else ""
        href = str(raw["href"]) if raw.get("href") else None
        area_ratio = float(raw["areaRatio"])
        is_fixed = bool(raw["isFixed"])

        if tag in ("button", "input", "select", "textarea"):
            utility += self._tuning.action_form_control_bonus
            reasons.append("form_or_button_control")
        elif tag == "a":
            utility += self._tuning.action_link_bonus
            reasons.append("link_control")

        if role == "button":
            utility += self._tuning.action_role_button_bonus
            reasons.append("role_button")

        if (
            self._tuning.action_reasonable_size_min
            <= area_ratio
            <= self._tuning.action_reasonable_size_max
        ):
            utility += self._tuning.action_reasonable_size_bonus
            reasons.append("reasonable_size")

        if self._contains_any(text, self._ACTION_WORDS):
            utility += self._tuning.action_action_word_bonus
            reasons.append("action_word_match")

        noise_hits = self._count_noise_hits(text)
        if noise_hits > 0:
            noise += min(
                self._tuning.action_noise_hit_penalty_cap,
                noise_hits * self._tuning.action_noise_hit_penalty_step,
            )
            reasons.append("noise_keyword")

        if is_fixed and area_ratio < self._tuning.action_fixed_small_target_area_threshold:
            noise += self._tuning.action_fixed_small_target_penalty
            reasons.append("fixed_small_target")

        if href is not None:
            href_low = href.lower()
            if href_low.startswith("javascript:") or href_low.startswith("mailto:"):
                noise += self._tuning.action_non_navigational_href_penalty
                reasons.append("non_navigational_href")
            parsed = urlparse(href_low)
            if parsed.netloc and parsed.netloc != page_host:
                noise += self._tuning.action_external_domain_penalty
                reasons.append("external_domain")

        if len(text.strip()) <= 1 and tag in ("a", "button"):
            noise += self._tuning.action_very_low_text_penalty
            reasons.append("very_low_text_signal")

        return self._clamp(utility, 0.0, 1.0), self._clamp(noise, 0.0, 1.0), reasons

    def _score_text_block(self, text: str, repeat_count: int) -> tuple[float, float]:
        normalized = self._normalize_for_dedupe(text)
        if not normalized:
            return 0.0, 1.0

        tokens = normalized.split()
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / max(1, token_count)
        lower = normalized.lower()

        utility = 0.0
        noise = 0.0

        if self._contains_any(lower, self._ACTION_WORDS):
            utility += 0.30
        if 8 <= token_count <= 64:
            utility += 0.16
        if re.search(
            r"\b(?:step|code|option|select|submit|verify|continue|scroll|reveal)\b", lower
        ):
            utility += 0.20

        if self._contains_any(lower, self._NOISE_WORDS):
            noise += 0.40
        if self._is_lorem_like(lower):
            noise += 0.60
        if (
            token_count <= self._tuning.block_low_info_token_max
            and unique_ratio <= self._tuning.block_low_info_unique_token_ratio_max
        ):
            noise += 0.24
        if self._should_suppress_repeated_block(text, repeat_count):
            noise += 0.30

        return self._clamp(utility, 0.0, 1.0), self._clamp(noise, 0.0, 1.0)

    def _should_suppress_repeated_block(self, text: str, repeat_count: int) -> bool:
        if repeat_count < self._tuning.section_repeated_template_block_suppress_min_count:
            return False
        normalized = self._normalize_for_dedupe(text)
        if not normalized:
            return True
        tokens = normalized.split()
        token_count = len(tokens)
        if token_count <= self._tuning.section_repeated_template_low_info_token_max:
            return True
        unique_ratio = len(set(tokens)) / token_count
        return unique_ratio <= self._tuning.section_repeated_template_unique_token_ratio_max

    def _extract_justext(self, html: str) -> tuple[str | None, list[str]]:
        if not self._use_justext:
            return None, []
        if not html.strip():
            return None, []

        try:
            import justext
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "jusText is not installed. Install dependencies and retry with --use-justext."
            ) from exc

        paragraphs = justext.justext(html, justext.get_stoplist("English"))
        cleaned_paragraphs: list[str] = []
        seen: set[str] = set()

        for paragraph in paragraphs:
            if paragraph.is_boilerplate:
                continue
            text = str(paragraph.text).strip()
            if not text:
                continue
            text_key = self._normalize_for_dedupe(text)
            if len(text_key) < 8:
                continue
            if text_key in seen:
                continue
            seen.add(text_key)
            cleaned_paragraphs.append(text)

        if not cleaned_paragraphs:
            return None, []
        return "\n\n".join(cleaned_paragraphs), cleaned_paragraphs

    def _to_form_control(self, control: _ControlAtom) -> FormControl:
        """Convert internal _ControlAtom → public FormControl. Single source of
        truth for this mapping; called from _build_form_entity and the Path A
        orphan-controls collection in _build_forms."""
        return FormControl(
            control_id=control.control_id,
            owner_container_id=control.container_id,
            text=control.text,
            selector=control.selector,
            tag=control.tag,
            role=control.role,
            type=control.type,
            name=control.name,
            label=control.label,
            options=control.options[: self._tuning.max_options_per_form],
            geometry=control.geometry,
            visible_now=control.visible_now,
            in_viewport_now=control.in_viewport_now,
            occlusion=self._default_occlusion(control.visible_now),
            ax_name=control.ax_name,
            ax_role=control.ax_role,
            disabled=control.disabled,
            ax_ignored=control.ax_ignored,
            checked=control.checked,
            expanded=control.expanded,
            selected=control.selected,
            current_value=control.current_value,
            has_popup=control.has_popup,
            ax_description=control.ax_description,
            focusable=control.focusable,
            cursor=control.cursor,
            draggable=control.draggable,
            duplicate_id=control.duplicate_id,
            slider_min=control.slider_min,
            slider_max=control.slider_max,
            slider_value=control.slider_value,
            slider_orientation=control.slider_orientation,
        )

    def _is_input_like(self, control: _ControlAtom) -> bool:
        if control.tag in ("input", "select", "textarea", "canvas", "video", "audio"):
            return True
        return control.role in ("radio", "checkbox", "combobox", "option")

    def _is_button_like(self, control: _ControlAtom) -> bool:
        if control.tag in ("button", "a"):
            return True
        if control.role == "button":
            return True
        if control.tag == "input" and control.type is not None:
            return control.type.lower() in ("button", "submit", "image")
        return False

    def _geometry_from_raw(self, raw: dict[str, Any]) -> ViewportGeometry:
        return ViewportGeometry(
            x=float(raw["xRatio"]),
            y=float(raw["yRatio"]),
            w=float(raw["wRatio"]),
            h=float(raw["hRatio"]),
        )

    def _geometry_intersects(self, left: ViewportGeometry, right: ViewportGeometry) -> bool:
        left_x1 = left.x
        left_y1 = left.y
        left_x2 = left.x + left.w
        left_y2 = left.y + left.h

        right_x1 = right.x
        right_y1 = right.y
        right_x2 = right.x + right.w
        right_y2 = right.y + right.h

        overlap_x = left_x1 <= right_x2 and right_x1 <= left_x2
        overlap_y = left_y1 <= right_y2 and right_y1 <= left_y2
        return overlap_x and overlap_y

    def _intersects_viewport(self, geometry: ViewportGeometry) -> bool:
        x1 = geometry.x
        y1 = geometry.y
        x2 = geometry.x + geometry.w
        y2 = geometry.y + geometry.h
        return not (x2 <= 0.0 or y2 <= 0.0 or x1 >= 1.0 or y1 >= 1.0)

    def _selector_depth(self, selector: str) -> int:
        return selector.count(" > ")

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output

    def _normalize_for_dedupe(self, text: str) -> str:
        collapsed = " ".join(text.lower().split())
        return re.sub(r"\W+", " ", collapsed).strip()

    def _split_stuck_camel_tokens(self, text: str) -> str:
        if not text:
            return ""
        separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", separated)
        return " ".join(separated.split()).strip()

    def _normalize_for_template(self, text: str) -> str:
        normalized = self._normalize_for_dedupe(text)
        if not normalized:
            return ""
        return re.sub(r"\b\d+\b", "<n>", normalized).strip()

    def _count_noise_hits(self, text: str) -> int:
        normalized = self._normalize_for_dedupe(text)
        if not normalized:
            return 0
        hits = 0
        if self._contains_any(normalized, self._NOISE_WORDS):
            hits += 1
        if self._is_lorem_like(normalized):
            hits += 1
        return hits

    def _is_lorem_like(self, text: str) -> bool:
        if not text:
            return False
        tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        if len(tokens) < _LOREM_MIN_TOKENS:
            return False
        overlap = len(tokens & _LOREM_VOCAB)
        return overlap / len(tokens) >= _LOREM_OVERLAP_THRESHOLD

    def _strip_lorem_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            return ""
        if self._is_lorem_like(normalized):
            return ""
        return normalized

    def _clamp(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def _contains_any(self, text: str, words: tuple[str, ...]) -> bool:
        return any(self._contains_phrase(text, word) for word in words)

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        if not text:
            return False
        normalized = text.lower()
        phrase_normalized = phrase.lower().strip()
        if not phrase_normalized:
            return False
        escaped = re.escape(phrase_normalized).replace(r"\ ", r"\s+")
        return re.search(rf"(?<!\w){escaped}(?!\w)", normalized) is not None


# Suppress unused-import warning — used in TYPE_CHECKING block above
with suppress(Exception):
    _ = ActionIntent  # exported for downstream consumers (Phase 2+)
