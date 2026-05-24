"""
morphnet_v3/computer_use/v7_markdown.py

V5-format markdown renderer with V7-era improvements (survival reducer,
suppression sets, consumed_wrapper_ids, _truncate_display_label, narrower
keep-tag set). Lifted wholesale from `browser-challenge/crawler/master_markdown.py`
at V7 (`origin/main`).

Documented deltas vs V7 (morphnet only):
  5. `from .schemas import (...)` stays — re-export bridge in
     `computer_use/schemas.py` keeps the import line byte-identical to V7.
     `from .page_filter import (...)` becomes `from ..page_filter import (...)`
     because page_filter lives one level up in morphnet's layout.
  6. Filename is `v7_markdown.py` (not `master_markdown.py`) and
     `_MARKDOWN_VERSION = "v7"` (V7 itself keeps "v5"). The rendered output
     header reads `# Page Representation (V7)` so callers and humans can tell
     v3's V7-improved renderer apart from v2's V5 renderer at a glance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from ..page_filter import (
    _ax_bool,
    _ax_string,
    _ax_tristate,
    ax_description_value,
    ax_is_modal,
    ax_name_value,
    ax_props,
    ax_role_value,
)
from .schemas import (
    ContainerEntity,
    Control,
    FormEntity,
    PageFilterOutput,
)

_MARKDOWN_VERSION = "v7"

_NON_SEMANTIC_ROLES: frozenset[str] = frozenset({"generic", "none", "presentation"})


def _normalize_text(raw: str | None) -> str:
    return " ".join((raw or "").split()).strip()


def _value_or_none(raw: str | None) -> str:
    normalized = _normalize_text(raw)
    return normalized if normalized else "(no text)"


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------


def _build_children_map(containers: list[ContainerEntity]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for c in containers:
        if c.parent_container_id:
            children[c.parent_container_id].append(c.container_id)
    return dict(children)


def _build_forms_by_owner(forms: list[FormEntity]) -> dict[str, list[FormEntity]]:
    by_owner: dict[str, list[FormEntity]] = defaultdict(list)
    for f in forms:
        by_owner[f.owner_container_id].append(f)
    return dict(by_owner)


def _build_blocked_by_map(containers: list[ContainerEntity]) -> dict[str, list[str]]:
    blocked_by: dict[str, list[str]] = defaultdict(list)
    for c in containers:
        if c.is_blocking:
            for victim_id in c.blocks_container_ids:
                blocked_by[victim_id].append(c.container_id)
    return dict(blocked_by)


def _collect_form_control_button_ids(forms: list[FormEntity]) -> set[str]:
    ids: set[str] = set()
    for f in forms:
        for group in f.groups:
            for ctrl in group.controls:
                if ctrl.tag == "button":
                    ids.add(ctrl.control_id)
    return ids


def _build_suppressed_text(
    cmap: dict[str, ContainerEntity],
    children_map: dict[str, list[str]],
    standalone_by_owner: dict[str, list[Control]],
    forms_by_owner: dict[str, list[FormEntity]],
) -> dict[str, set[str]]:
    """Build per-container suppression sets: control text that should not echo in text_blocks.

    For each container, the suppression set is the union of:
    - Same-container standalone control text (buttons + orphans unified)
    - Same-container form control label/text
    - All descendant containers' control texts (cross-nesting suppression)
    """
    # Step 1: local control text per container
    local: dict[str, set[str]] = {}
    for cid in cmap:
        texts: set[str] = set()
        # Plan 026: unified loop. Previously buttons only suppressed
        # _value_or_none(text); orphans also suppressed label. Now both
        # suppress label+text — more accurate (a button's label matching
        # a text_block should suppress it).
        for ctrl in standalone_by_owner.get(cid, []):
            t = ctrl.label or ctrl.text or ""
            if t:
                texts.add(t)
            texts.add(_value_or_none(ctrl.text))
        for f in forms_by_owner.get(cid, []):
            for group in f.groups:
                for ctrl in group.controls:
                    t = ctrl.label or ctrl.text or ""
                    if t:
                        texts.add(t)
        local[cid] = texts

    # Step 2: bottom-up merge — each container's set includes all descendants'
    result: dict[str, set[str]] = {}

    def _merge(cid: str) -> set[str]:
        if cid in result:
            return result[cid]
        merged = set(local.get(cid, set()))
        for child_id in children_map.get(cid, []):
            merged |= _merge(child_id)
        result[cid] = merged
        return merged

    for cid in cmap:
        _merge(cid)

    # Plan 022 Phase 4: suppress ancestor headings.
    # If a leaf text block exactly matches an ancestor's heading, it would
    # render twice (as text and as the ancestor header). Add ancestor headings
    # to the suppression set so the duplicate text line is suppressed.
    for cid, container in cmap.items():
        ancestor_headings: set[str] = set()
        parent_id = container.parent_container_id
        while parent_id:
            parent = cmap.get(parent_id)
            if not parent:
                break
            if parent.heading:
                normalized = parent.heading.strip()
                if normalized:
                    ancestor_headings.add(normalized)
            parent_id = parent.parent_container_id
        if ancestor_headings:
            if cid in result:
                result[cid] |= ancestor_headings
            else:
                result[cid] = ancestor_headings

    return result


def _build_descendant_text_blocks(
    cmap: dict[str, ContainerEntity],
    children_map: dict[str, list[str]],
) -> dict[str, set[str]]:
    """For each container, collect all text_blocks from descendant containers (not self)."""
    result: dict[str, set[str]] = {}

    def _merge(cid: str) -> set[str]:
        if cid in result:
            return result[cid]
        texts: set[str] = set()
        for child_id in children_map.get(cid, []):
            child = cmap.get(child_id)
            if child:
                texts |= set(child.text_blocks)
            texts |= _merge(child_id)
        result[cid] = texts
        return texts

    for cid in cmap:
        _merge(cid)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _visual_tag(visual: dict[str, str]) -> str:
    """Build inline visual annotation from non-default computed styles."""
    if not visual:
        return ""
    parts = [f"{k}:{v}" for k, v in visual.items()]
    return " {" + ", ".join(parts) + "}"


_KEEP_SIGNAL_TAGS: frozenset[str] = frozenset({"nav", "form", "header", "footer", "main", "aside"})


def _compute_surviving(
    cmap: dict[str, ContainerEntity],
    children_map: dict[str, list[str]],
    standalone_by_owner: dict[str, list[Control]],
    forms_by_owner: dict[str, list[FormEntity]],
    suppressed_text: dict[str, set[str]],
    aria_owns: dict[str, list[str]],
    overlay_ids: set[str],
    consumed_wrapper_ids: set[str],
) -> set[str]:
    """Post-order reducer: decide which containers survive with named headers.

    Returns set of container IDs that get named headers in the markdown.

    For each container D, reduce children first, then:
    - K(D): hard structural keep-signals -> survive
    - Named (heading or AX name) -> survive
    - C(D): has unsuppressed text_blocks -> survive
    - 2+ payload peers (E + surviving children) -> survive
    - Otherwise -> collapse (not in surviving set)
    """
    surviving: set[str] = set()
    visited: set[str] = set()

    def _has_keep_signal(c: ContainerEntity, cid: str) -> bool:
        """K(D): narrow structural keep-signals."""
        role = ax_role_value(c.ax)
        if role and role not in _NON_SEMANTIC_ROLES:
            return True
        if c.tag in _KEEP_SIGNAL_TAGS:
            return True
        if c.is_blocking:
            return True
        if c.scrollable:
            return True
        if c.is_drop_zone:
            return True
        return bool(aria_owns.get(cid))

    def _reduce(cid: str) -> None:
        if cid in visited:
            return
        visited.add(cid)
        c = cmap.get(cid)
        if not c:
            return

        # Consumed wrappers are owned by the form renderer (Plan 024).
        # They don't participate in the surviving set at all.
        if cid in consumed_wrapper_ids:
            return

        # Post-order: reduce all children first
        for child_id in children_map.get(cid, []):
            _reduce(child_id)

        # Overlay roots always survive (rendered in overlay section)
        if cid in overlay_ids:
            surviving.add(cid)
            return

        # K(D): hard structural signal -> survive
        if _has_keep_signal(c, cid):
            surviving.add(cid)
            return

        # Named: heading or AX accessible name -> survive
        if ax_name_value(c.ax) or c.heading:
            surviving.add(cid)
            return

        # C(D): has unsuppressed text -> survive
        cid_suppressed = suppressed_text.get(cid, set())
        if any(tb not in cid_suppressed for tb in c.text_blocks):
            surviving.add(cid)
            return

        # E(D): count direct rendered leaves (individual items)
        direct_leaves = len(standalone_by_owner.get(cid, [])) + len(forms_by_owner.get(cid, []))

        # Surviving child containers
        surviving_children = sum(
            1 for child_id in children_map.get(cid, []) if child_id in surviving
        )

        # Groups 2+ payload peers -> survive
        if direct_leaves + surviving_children >= 2:
            surviving.add(cid)
            return

        # Singleton or empty -> collapse (not added to surviving)

    # Reduce from all roots
    for c in cmap.values():
        if not c.parent_container_id:
            _reduce(c.container_id)

    return surviving


def container_flags(c: ContainerEntity) -> str:
    flags: list[str] = []
    if ax_is_modal(c.ax):
        flags.append("modal")
    if c.is_drop_zone:
        has_content = bool(c.text_blocks or c.control_refs or c.button_refs or c.form_refs)
        flags.append("drop-zone" if has_content else "drop-zone: empty")
    if c.scrollable:
        flags.append("scrollable")
    if c.has_animation:
        flags.append("animated")
    if not flags:
        return ""
    return " [" + ", ".join(flags) + "]"


def _collect_tree_flags(
    cid: str,
    cmap: dict[str, ContainerEntity],
    children_map: dict[str, list[str]],
) -> set[str]:
    c = cmap.get(cid)
    if not c:
        return set()
    flags: set[str] = set()
    if c.scrollable:
        flags.add("scrollable")
    if c.has_animation:
        flags.add("animated")
    for child_id in children_map.get(cid, []):
        flags |= _collect_tree_flags(child_id, cmap, children_map)
    return flags


def _effective_container_blocker(
    container_id: str,
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
) -> str | None:
    """Walk up the container tree to find the nearest blocker."""
    visited: set[str] = set()
    current: str | None = container_id
    while current and current not in visited:
        visited.add(current)
        blockers = blocked_by_map.get(current)
        if blockers:
            return blockers[0]
        c = cmap.get(current)
        if c and c.parent_container_id:
            current = c.parent_container_id
        else:
            break
    return None


def _walk_to_surviving(
    aid: str | None,
    cmap: dict[str, ContainerEntity],
    surviving_ids: set[str],
) -> str | None:
    """Walk up parent chain to nearest container in the surviving set."""
    current = aid
    while current:
        c = cmap.get(current)
        if c is None:
            return current  # unknown container, return as-is
        if current in surviving_ids:
            return current  # visible in markdown, use this
        current = c.parent_container_id
    return aid  # fallback to original if chain exhausted


def _resolve_blockers(
    is_occluded: bool,
    direct_blockers: list[str],
    owner_container_id: str,
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    form_blocker_id: str | None = None,
    surviving_ids: set[str] | None = None,
) -> list[str]:
    """Return all effective blockers for a control/button, checking direct then inherited."""
    blockers: list[str] = []
    if is_occluded and direct_blockers:
        blockers = list(direct_blockers)
    elif is_occluded and form_blocker_id:
        # Fix A: only propagate form_blocker if this control is individually occluded
        blockers = [form_blocker_id]
    else:
        inherited = _effective_container_blocker(owner_container_id, blocked_by_map, cmap)
        if inherited:
            blockers = [inherited]
    # Fix E: if blocker is invisible in markdown, resolve to nearest surviving ancestor
    if surviving_ids is not None:
        blockers = [
            walked
            for b in blockers
            if (walked := _walk_to_surviving(b, cmap, surviving_ids)) is not None
        ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for b in blockers:
        if b not in seen:
            seen.add(b)
            result.append(b)
    return result


# ---------------------------------------------------------------------------
# AX annotation helpers
# ---------------------------------------------------------------------------

# Tags that are inherently focusable — [focusable] annotation is noise on these.
_INHERENTLY_FOCUSABLE: frozenset[str] = frozenset({"input", "button", "select", "textarea", "a"})


def _control_annotations(ctrl: Control) -> str:
    """Build exception-only annotation string for AX-enriched control fields.

    Only emits annotations when values are non-default. Returns empty string
    when all fields are default (zero extra tokens on first extraction).
    """
    props = ax_props(ctrl.ax)
    parts: list[str] = []
    if _ax_bool(props, "disabled"):
        parts.append("disabled")
    checked = _ax_tristate(props, "checked")
    if checked == "checked":
        parts.append("checked")
    elif checked == "mixed":
        parts.append("mixed")
    expanded = _ax_tristate(props, "expanded")
    if expanded == "checked":  # tristate "checked" = expanded state
        parts.append("expanded")
    if _ax_bool(props, "selected"):
        parts.append("selected")
    hp = _ax_string(props, "hasPopup")
    if hp:
        parts.append(f"hasPopup:{hp}")
    if ctrl.draggable:
        parts.append("draggable")
    if ctrl.href:
        parts.append(f"href:{ctrl.href}")
    if ctrl.cursor == "not-allowed":
        parts.append("cursor:not-allowed")
    if _ax_bool(props, "focusable") and ctrl.tag not in _INHERENTLY_FOCUSABLE:
        parts.append("focusable")
    if not parts:
        return ""
    return " [" + ", ".join(parts) + "]"


def _control_suffix(ctrl: Control) -> str:
    """Build suffix tokens for value and description — after annotations, before blocker."""
    suffix = ""
    cv = _ax_string(ax_props(ctrl.ax), "value")
    if cv:
        suffix += f' value="{cv}"'
    return suffix


def _control_hint(ctrl: Control, indent: str) -> str | None:
    """Build hint line for ax_description. Returns None if no hint."""
    desc = ax_description_value(ctrl.ax)
    if desc:
        return f'{indent}    (hint: "{desc}")'
    return None


def _truncate_display_label(
    display_text: str,
    ax_name: str | None,
    raw_text: str | None,
    max_len: int = 120,
) -> str:
    """Truncate long button display labels.

    If display_text fits, return as-is. If ax_name is distinct and short, prefer it.
    Otherwise truncate with ellipsis.
    """
    if len(display_text) <= max_len:
        return display_text
    if ax_name and ax_name != (raw_text or "") and len(ax_name) <= max_len:
        return ax_name
    return display_text[:max_len] + "..."


# ---------------------------------------------------------------------------
# Inline renderers
# ---------------------------------------------------------------------------

_RESIZE_CURSORS: frozenset[str] = frozenset(
    {
        "col-resize",
        "row-resize",
        "ew-resize",
        "ns-resize",
        "nw-resize",
        "ne-resize",
        "sw-resize",
        "se-resize",
    }
)


def _render_single_control(
    ctrl: Control,
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    form_blocker_id: str | None = None,
    truncate_label: bool = False,
    surviving_ids: set[str] | None = None,
) -> None:
    """Render one control line. Shared by form, orphan, and button paths."""
    ctype = ctrl.type or ""
    # Suppress AX name on duplicate-ID controls (clones share misleading names)
    ctrl_ax_name = None if ctrl.duplicate_id else ax_name_value(ctrl.ax)
    label = ctrl_ax_name or ctrl.label or ctrl.text or "(no label)"
    if truncate_label:
        label = _truncate_display_label(label, ctrl_ax_name, ctrl.text)
    type_prefix = ax_role_value(ctrl.ax) or ctrl.role or ctrl.tag

    # Blocker resolution — form path subtracts form-level blocker
    blockers = _resolve_blockers(
        ctrl.occlusion.is_occluded,
        ctrl.occlusion.blocker_container_ids,
        ctrl.owner_container_id,
        blocked_by_map,
        cmap,
        form_blocker_id=form_blocker_id,
        surviving_ids=surviving_ids,
    )
    if form_blocker_id:
        form_blocker_set: set[str] = {form_blocker_id}
        display_blockers = [b for b in blockers if b not in form_blocker_set]
    else:
        display_blockers = blockers
    ctrl_blocked = f" [blocked by {', '.join(display_blockers)}]" if display_blockers else ""

    type_str = f"[{ctype}]" if ctype else ""
    annotations = _control_annotations(ctrl)
    suffix = _control_suffix(ctrl)
    opts = ""
    if ctrl.options:
        shown = ctrl.options[:20]
        opts = " [options: " + ", ".join(f'"{o}"' for o in shown)
        if len(ctrl.options) > 20:
            opts += f", +{len(ctrl.options) - 20} more"
        opts += "]"

    # Canvas rendering: show dimensions so model knows it can draw
    if ctrl.tag == "canvas":
        w_px = round(ctrl.geometry.w * 1280)  # approx pixels
        h_px = round(ctrl.geometry.h * 720)
        lines.append(
            f"{indent}canvas `{ctrl.control_id}` [{w_px}x{h_px}]{annotations}{ctrl_blocked}"
        )
    # Slider-specific rendering: show value context
    elif ctrl.slider_max is not None:
        slider_val = ctrl.slider_value if ctrl.slider_value is not None else 0
        slider_max = ctrl.slider_max
        orient = ctrl.slider_orientation or "horizontal"
        lines.append(
            f'{indent}slider `{ctrl.control_id}` "{label}"'
            f" [{slider_val}/{slider_max}, {orient}]"
            f"{annotations}{ctrl_blocked}"
        )
    # Resize-handle rendering: cursor-based hint
    elif ctrl.cursor in _RESIZE_CURSORS:
        lines.append(
            f'{indent}resize-handle `{ctrl.control_id}` "{label}"'
            f" [cursor: {ctrl.cursor}]{annotations}{ctrl_blocked}"
        )
    else:
        lines.append(
            f'{indent}{type_prefix}{type_str} `{ctrl.control_id}` "{label}"'
            f"{annotations}{suffix}{opts}{ctrl_blocked}"
        )
    hint = _control_hint(ctrl, indent)
    if hint:
        lines.append(hint)


def _render_form_inline(
    form: FormEntity,
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    rendered_aids: set[str] | None = None,
    *,
    consumed_wrapper_ids: set[str] | None = None,
    render_consumed_wrapper_subtree: Callable[[str, set[str], int], None] | None = None,
    depth: int = 0,
    surviving_ids: set[str] | None = None,
) -> None:
    blocked = ""
    if form.form_blocker.is_blocked:
        blocker_id = form.form_blocker.blocker_container_id or "?"
        blocked = f" [blocked by {blocker_id}]"
    lines.append(f"{indent}form `{form.form_id}`{blocked}")
    if rendered_aids is not None:
        rendered_aids.add(form.form_id)

    form_blocker_id = (
        form.form_blocker.blocker_container_id if form.form_blocker.is_blocked else None
    )
    for group in form.groups:
        rendered_group_control_ids: set[str] = set()
        for ctrl in group.controls:
            _render_single_control(
                ctrl,
                indent + "  ",
                lines,
                blocked_by_map,
                cmap,
                form_blocker_id=form_blocker_id,
                surviving_ids=surviving_ids,
            )
            if rendered_aids is not None:
                rendered_aids.add(ctrl.control_id)
            rendered_group_control_ids.add(ctrl.control_id)

        # Plan 024: render consumed wrapper subtree after group controls.
        # depth + 1: wrapper text renders at the same indent as control lines
        # (one level deeper than the form header).
        if (
            consumed_wrapper_ids
            and group.container_id in consumed_wrapper_ids
            and render_consumed_wrapper_subtree is not None
        ):
            render_consumed_wrapper_subtree(
                group.container_id, rendered_group_control_ids, depth + 1
            )


def _render_standalone_controls(
    controls: list[Control],
    form_control_ids: set[str],
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    compact_threshold: int = 6,
    rendered_aids: set[str] | None = None,
    surviving_ids: set[str] | None = None,
) -> None:
    """Render standalone controls (both button-like and non-button).

    Compact mode fires only for button-like controls (6+).
    When compact is off, all controls interleave in DOM order.
    """
    controls = [
        c
        for c in controls
        if c.control_id not in form_control_ids
        and (rendered_aids is None or c.control_id not in rendered_aids)
    ]
    if not controls:
        return

    button_like = [c for c in controls if c.button_like]
    other = [c for c in controls if not c.button_like]

    if len(button_like) > compact_threshold:
        # Compact mode for button-like controls. Each entry STILL carries its
        # own aid so CU can address it (previously the compact form dropped
        # aids, which collapsed `<a>` links inside container groups into an
        # unaddressable bag — observable cause of multi-turn click loops where
        # CU could only click the parent container, never the actual link).
        parts: list[str] = []
        for b in button_like:
            b_ax_name = None if b.duplicate_id else ax_name_value(b.ax)
            bt = _value_or_none(b_ax_name or b.text)
            bt = _truncate_display_label(bt, b_ax_name, b.text)
            blockers = _resolve_blockers(
                b.occlusion.is_occluded,
                b.occlusion.blocker_container_ids,
                b.owner_container_id,
                blocked_by_map,
                cmap,
                surviving_ids=surviving_ids,
            )
            suffix = "*" if blockers else ""
            ann = _control_annotations(b)
            parts.append(f'{b.control_id} "{bt}"{ann}{suffix}')
        lines.append(f"{indent}buttons({len(button_like)}): {', '.join(parts)}")
        # Render non-button controls individually
        for ctrl in other:
            _render_single_control(
                ctrl, indent, lines, blocked_by_map, cmap, surviving_ids=surviving_ids
            )
            if rendered_aids is not None:
                rendered_aids.add(ctrl.control_id)
    else:
        # Sort by dom_order so buttons and orphans interleave faithfully
        # (production builds controls as buttons + orphans, not DOM order).
        controls = sorted(controls, key=lambda c: c.dom_order)
        for ctrl in controls:
            _render_single_control(
                ctrl,
                indent,
                lines,
                blocked_by_map,
                cmap,
                truncate_label=ctrl.button_like,
                surviving_ids=surviving_ids,
            )
            if rendered_aids is not None:
                rendered_aids.add(ctrl.control_id)


# ---------------------------------------------------------------------------
# Tree collectors for overlays (flatten children)
# ---------------------------------------------------------------------------


def _collect_tree_standalone(
    cid: str,
    standalone_by_owner: dict[str, list[Control]],
    children_map: dict[str, list[str]],
) -> list[Control]:
    result = list(standalone_by_owner.get(cid, []))
    for child_id in children_map.get(cid, []):
        result.extend(_collect_tree_standalone(child_id, standalone_by_owner, children_map))
    return result


def _collect_tree_forms(
    cid: str,
    forms_by_owner: dict[str, list[FormEntity]],
    children_map: dict[str, list[str]],
) -> list[FormEntity]:
    result = list(forms_by_owner.get(cid, []))
    for child_id in children_map.get(cid, []):
        result.extend(_collect_tree_forms(child_id, forms_by_owner, children_map))
    return result


def _collect_tree_text(
    cid: str,
    cmap: dict[str, ContainerEntity],
    children_map: dict[str, list[str]],
) -> list[str]:
    c = cmap.get(cid)
    if not c:
        return []
    result = list(c.text_blocks)
    for child_id in children_map.get(cid, []):
        result.extend(_collect_tree_text(child_id, cmap, children_map))
    return result


# Main render
# ---------------------------------------------------------------------------


def render_master_markdown(extraction: PageFilterOutput) -> str:
    markdown, _ = render_master_markdown_with_meta(extraction)
    return markdown


def render_master_markdown_with_meta(
    extraction: PageFilterOutput,
) -> tuple[str, dict[str, object]]:
    containers = extraction.containers
    forms = extraction.forms

    cmap: dict[str, ContainerEntity] = {c.container_id: c for c in containers}
    children_map = _build_children_map(containers)
    forms_by_owner = _build_forms_by_owner(forms)
    blocked_by_map = _build_blocked_by_map(containers)
    form_control_ids = _collect_form_control_button_ids(forms)
    rendered_aids: set[str] = set()

    standalone_by_owner: dict[str, list[Control]] = defaultdict(list)
    for ctrl in extraction.controls:
        standalone_by_owner[ctrl.owner_container_id].append(ctrl)

    # ---- Plan 024: consumed wrapper set ----
    consumed_wrapper_ids: set[str] = set(extraction.consumed_wrapper_ids)

    # ---- Text suppression sets (Phase 1 of Plan 021) ----
    suppressed_text = _build_suppressed_text(
        cmap, children_map, standalone_by_owner, forms_by_owner
    )
    # ---- ARIA relationship resolution (controls/owns → semantic placement) ----
    # Build backendDOMNodeId → AID mapping from all entities
    backend_to_aid: dict[int, str] = {}
    for c in containers:
        bid = c.ax.get("backendDOMNodeId")
        if bid is not None:
            backend_to_aid[bid] = c.container_id
    for f in forms:
        for g in f.groups:
            for ctrl in g.controls:
                bid = ctrl.ax.get("backendDOMNodeId")
                if bid is not None:
                    backend_to_aid[bid] = ctrl.control_id
    for ctrl in extraction.controls:
        bid = ctrl.ax.get("backendDOMNodeId")
        if bid is not None:
            backend_to_aid[bid] = ctrl.control_id

    # Resolve controls/owns → owned AIDs
    aria_owned_by: dict[str, str] = {}  # owned_aid → owner_aid
    aria_owns: dict[str, list[str]] = defaultdict(list)  # owner_aid → [owned_aids]

    def _resolve_aria_rels(aid: str, ax: dict[str, object]) -> None:
        properties = ax.get("properties")
        if not isinstance(properties, list):
            return
        for prop in properties:
            if not isinstance(prop, dict):
                continue
            name = prop.get("name")
            if name not in ("controls", "owns"):
                continue
            val = prop.get("value")
            related_raw = val.get("relatedNodes") if isinstance(val, dict) else None
            related = related_raw if isinstance(related_raw, list) else []
            for rn in related:
                if not isinstance(rn, dict):
                    continue
                bid = rn.get("backendDOMNodeId")
                if bid is not None:
                    target_aid = backend_to_aid.get(bid)
                    if target_aid and target_aid != aid and target_aid not in aria_owned_by:
                        aria_owned_by[target_aid] = aid
                        aria_owns[aid].append(target_aid)

    for c in containers:
        _resolve_aria_rels(c.container_id, c.ax)
    for f in forms:
        for g in f.groups:
            for ctrl in g.controls:
                _resolve_aria_rels(ctrl.control_id, ctrl.ax)
    for ctrl in extraction.controls:
        _resolve_aria_rels(ctrl.control_id, ctrl.ax)

    # Plan 022 Phase 3: only mark overlay roots, not descendants.
    # Descendants render hierarchically via _render_content.
    fixed_ids = {c.container_id for c in containers if c.fixed_position}
    overlay_ids: set[str] = set(fixed_ids)

    # Plan 025: two-pass container visibility.
    # Pass 2 — compute surviving set (containers that get named headers).
    surviving_ids = _compute_surviving(
        cmap,
        children_map,
        standalone_by_owner,
        forms_by_owner,
        suppressed_text,
        aria_owns,
        overlay_ids,
        consumed_wrapper_ids,
    )

    lines: list[str] = []

    # ---- Content infrastructure (Plan 022 Phase 3: moved before overlays) ----
    def _is_content_root(cid: str) -> bool:
        """Does this container have enough payload to anchor a content subtree?

        Containers that survive only because of heading/AX-name (but have no
        text_blocks, controls, forms, or structural keep signals) are walked
        through — they get their header when _render_content reaches them.
        """
        c = cmap.get(cid)
        if not c:
            return False
        # Structural keep signals → content root
        role = ax_role_value(c.ax)
        if role and role not in _NON_SEMANTIC_ROLES:
            return True
        if c.tag in _KEEP_SIGNAL_TAGS:
            return True
        if c.is_blocking:
            return True
        if c.is_drop_zone:
            return True
        # Has direct content → content root
        return bool(c.text_blocks or c.control_refs or c.button_refs or c.form_refs)

    def _find_content_roots() -> list[str]:
        roots = [c.container_id for c in containers if not c.parent_container_id]
        result: list[str] = []
        visited: set[str] = set()

        def walk(cid: str) -> None:
            if cid in visited or cid in overlay_ids:
                return
            visited.add(cid)
            if cid in consumed_wrapper_ids:
                return  # consumed by form renderer; do not descend
            c = cmap.get(cid)
            if not c:
                return
            if not _is_content_root(cid):
                for child_id in children_map.get(cid, []):
                    walk(child_id)
            else:
                result.append(cid)

        for rid in roots:
            walk(rid)
        return result

    effective_roots = _find_content_roots()
    content_walked: set[str] = set()

    def _render_content(cid: str, depth: int = 0) -> None:
        if cid in content_walked:
            return
        content_walked.add(cid)
        if cid in consumed_wrapper_ids:
            return  # consumed by form renderer; do not descend
        c = cmap.get(cid)
        if not c or cid in overlay_ids:
            return

        # Plan 025: container not in surviving set — collapse.
        # Skip header, promote payload and children at current depth.
        if cid not in surviving_ids:
            indent = "  " * depth
            cid_suppressed = suppressed_text.get(cid, set())
            for tb in c.text_blocks:
                if tb not in cid_suppressed:
                    lines.append(f'{indent}  "{tb}"')

            for f in forms_by_owner.get(cid, []):
                _render_form_inline(
                    f,
                    indent + "  ",
                    lines,
                    blocked_by_map,
                    cmap,
                    rendered_aids,
                    consumed_wrapper_ids=consumed_wrapper_ids,
                    render_consumed_wrapper_subtree=_render_consumed_wrapper_subtree,
                    depth=depth + 1,
                    surviving_ids=surviving_ids,
                )

            _render_standalone_controls(
                standalone_by_owner.get(cid, []),
                form_control_ids,
                indent + "  ",
                lines,
                blocked_by_map,
                cmap,
                rendered_aids=rendered_aids,
                surviving_ids=surviving_ids,
            )

            for child_id in children_map.get(cid, []):
                if child_id in aria_owned_by:
                    continue
                child = cmap.get(child_id)
                if not child or child_id in overlay_ids:
                    continue
                _render_content(child_id, depth)
            return

        indent = "  " * depth

        blocking_tag = ""
        if c.is_blocking and c.blocks_container_ids:
            blocking_tag = f" blocks [{', '.join(c.blocks_container_ids)}]"

        blocked_ids = blocked_by_map.get(cid, [])
        blocked_tag = ""
        if blocked_ids:
            blocked_tag = f" [blocked by {', '.join(blocked_ids)}]"

        z_tag = f" z={c.z_index}" if c.z_index > 0 else ""
        semantic_tag = ""
        if c.tag in (
            "form",
            "nav",
            "header",
            "footer",
            "main",
            "aside",
            "section",
            "article",
        ):
            semantic_tag = f" <{c.tag}>"
        # AX role prefix — Chrome's identity for this container
        ax_role = ax_role_value(c.ax)
        role_prefix = f"{ax_role} " if ax_role and ax_role not in _NON_SEMANTIC_ROLES else ""
        ax_name = ax_name_value(c.ax)
        heading = _normalize_text(c.heading)
        header = f"{indent}{role_prefix}`{cid}`{semantic_tag}{z_tag}"
        # Prefer AX name (Chrome's accessible name), fall back to DOM heading
        display_name = ax_name or heading
        if display_name:
            header += f' "{display_name}"'
        header += container_flags(c)
        header += blocking_tag + blocked_tag
        lines.append(header)
        rendered_aids.add(cid)

        cid_suppressed = suppressed_text.get(cid, set())
        for tb in c.text_blocks:
            if tb != heading and tb not in cid_suppressed:
                lines.append(f'{indent}  "{tb}"')

        for f in forms_by_owner.get(cid, []):
            _render_form_inline(
                f,
                indent + "  ",
                lines,
                blocked_by_map,
                cmap,
                rendered_aids,
                consumed_wrapper_ids=consumed_wrapper_ids,
                render_consumed_wrapper_subtree=_render_consumed_wrapper_subtree,
                depth=depth + 1,
                surviving_ids=surviving_ids,
            )

        _render_standalone_controls(
            standalone_by_owner.get(cid, []),
            form_control_ids,
            indent + "  ",
            lines,
            blocked_by_map,
            cmap,
            rendered_aids=rendered_aids,
            surviving_ids=surviving_ids,
        )

        # ARIA relationship: render owned containers under their semantic owner
        for owned_aid in aria_owns.get(cid, []):
            if owned_aid not in rendered_aids and owned_aid in cmap:
                lines.append(f"{indent}  (via aria-controls)")
                _render_content(owned_aid, depth + 1)

        for child_id in children_map.get(cid, []):
            # Skip containers owned by another element (rendered via aria-controls)
            if child_id in aria_owned_by:
                continue
            child = cmap.get(child_id)
            if not child or child_id in overlay_ids:
                continue
            _render_content(child_id, depth + 1)

    # ---- Plan 024: consumed wrapper subtree helper ----
    def _render_consumed_wrapper_subtree(
        wrapper_id: str, rendered_group_control_ids: set[str], depth: int
    ) -> None:
        """Render the full payload of a consumed wrapper inside the form path.

        Consumed wrappers are owned by the form renderer (Plan 024).
        They do not get a container header — just their text, standalone controls,
        sub-forms, and child containers.
        """
        content_walked.add(wrapper_id)
        rendered_aids.add(wrapper_id)
        wrapper = cmap.get(wrapper_id)
        if not wrapper:
            return

        indent = "  " * depth

        # Text blocks — same suppression/heading filtering as _render_content
        wrapper_suppressed = suppressed_text.get(wrapper_id, set())
        wrapper_heading = _normalize_text(wrapper.heading)
        for tb in wrapper.text_blocks:
            if tb != wrapper_heading and tb not in wrapper_suppressed:
                lines.append(f'{indent}"{tb}"')

        # Standalone controls (excluding the group's already-rendered direct controls)
        wrapper_controls = [
            c
            for c in standalone_by_owner.get(wrapper_id, [])
            if c.control_id not in rendered_group_control_ids
        ]
        _render_standalone_controls(
            wrapper_controls,
            form_control_ids,
            indent,
            lines,
            blocked_by_map,
            cmap,
            rendered_aids=rendered_aids,
            surviving_ids=surviving_ids,
        )

        # Sub-forms owned by wrapper
        for f in forms_by_owner.get(wrapper_id, []):
            _render_form_inline(
                f,
                indent,
                lines,
                blocked_by_map,
                cmap,
                rendered_aids,
                consumed_wrapper_ids=consumed_wrapper_ids,
                render_consumed_wrapper_subtree=_render_consumed_wrapper_subtree,
                depth=depth,
                surviving_ids=surviving_ids,
            )

        # Child containers — recurse via _render_content for non-consumed children
        for child_id in children_map.get(wrapper_id, []):
            if child_id not in consumed_wrapper_ids:
                _render_content(child_id, depth + 1)

    # ---- Header ----
    lines.append(f"# Page Representation ({_MARKDOWN_VERSION.upper()})")
    lines.append("")
    lines.append(f"url: `{extraction.page_url}`")
    lines.append(f'title: "{_value_or_none(extraction.page_title)}"')
    button_count = sum(1 for c in extraction.controls if c.button_like)
    lines.append(
        f"{len(containers)} containers, {button_count} buttons, "
        f"{len(forms)} forms, {extraction.blocking_container_count} blocking"
    )
    lines.append("")

    # ---- Overlays ----
    overlay_roots = [c for c in containers if c.container_id in fixed_ids]

    def _overlay_has_unblocked_buttons(c: ContainerEntity) -> bool:
        for ctrl in standalone_by_owner.get(c.container_id, []):
            if not ctrl.button_like:
                continue
            blockers = _resolve_blockers(
                ctrl.occlusion.is_occluded,
                ctrl.occlusion.blocker_container_ids,
                ctrl.owner_container_id,
                blocked_by_map,
                cmap,
                surviving_ids=surviving_ids,
            )
            if not blockers:
                return True
        for child_id in children_map.get(c.container_id, []):
            child = cmap.get(child_id)
            if child and _overlay_has_unblocked_buttons(child):
                return True
        return False

    overlay_roots.sort(key=lambda c: (not _overlay_has_unblocked_buttons(c), -c.z_index))

    if overlay_roots:
        lines.append("## Overlays")
        lines.append("")

        for c in overlay_roots:
            cid = c.container_id
            heading = _normalize_text(c.heading)

            blocking_tag = ""
            if c.is_blocking and c.blocks_container_ids:
                blocking_tag = f" blocks [{', '.join(c.blocks_container_ids)}]"

            # AX role prefix — Chrome's identity for this overlay
            ax_role = ax_role_value(c.ax)
            role_prefix = f"{ax_role} " if ax_role and ax_role not in _NON_SEMANTIC_ROLES else ""
            ax_name = ax_name_value(c.ax)
            display_name = ax_name or heading
            header = f"{role_prefix}`{cid}` z={c.z_index}"
            if display_name:
                header += f' "{display_name}"'
            tree_flags = _collect_tree_flags(cid, cmap, children_map)
            if tree_flags:
                header += " [" + ", ".join(sorted(tree_flags)) + "]"
            header += blocking_tag
            lines.append(header)
            rendered_aids.add(cid)
            content_walked.add(cid)

            # Plan 022 Phase 3: render overlay root's own content, then
            # children hierarchically via _render_content (not flat dump).
            cid_suppressed = suppressed_text.get(cid, set())
            for tb in c.text_blocks:
                if tb != heading and tb not in cid_suppressed:
                    lines.append(f'  "{tb}"')

            for f in forms_by_owner.get(cid, []):
                _render_form_inline(
                    f,
                    "  ",
                    lines,
                    blocked_by_map,
                    cmap,
                    rendered_aids,
                    consumed_wrapper_ids=consumed_wrapper_ids,
                    render_consumed_wrapper_subtree=_render_consumed_wrapper_subtree,
                    depth=1,
                    surviving_ids=surviving_ids,
                )

            _render_standalone_controls(
                standalone_by_owner.get(cid, []),
                form_control_ids,
                "  ",
                lines,
                blocked_by_map,
                cmap,
                rendered_aids=rendered_aids,
                surviving_ids=surviving_ids,
            )

            # Render children hierarchically
            for child_id in children_map.get(cid, []):
                _render_content(child_id, depth=1)

            lines.append("")

    # ---- Content Tree ----
    lines.append("## Content")
    lines.append("")

    for root_cid in effective_roots:
        _render_content(root_cid)

    lines.append("")

    # ---- Footer ----
    lines.append("---")
    lines.append("Resources: extraction_json, full_dom")

    meta: dict[str, object] = {
        "version": _MARKDOWN_VERSION,
        "source_url": extraction.page_url,
        "page_epoch": extraction.page_epoch,
        "rendered_aids": rendered_aids,
        "surviving_ids": surviving_ids,
    }
    return "\n".join(lines).rstrip() + "\n", meta
