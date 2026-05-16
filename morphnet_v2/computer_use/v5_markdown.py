from __future__ import annotations

import math
from collections import defaultdict

from .schemas import (
    ButtonEntity,
    ContainerEntity,
    FormControl,
    FormEntity,
    PageFilterOutput,
)

_MARKDOWN_VERSION = "v5"

# DnD proximity scoring: exponential decay rate.
# k=0.2 is needed for sibling-branch layouts (challenge: ~5 hops).
# exp(-0.2 * 5) + 0.2 text bonus = 0.57 > 0.5 threshold.
# k=0.7 would give exp(-3.5) ≈ 0.03 → below threshold → broken.
_DND_DECAY_K = 0.2


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


def _build_buttons_by_owner(buttons: list[ButtonEntity]) -> dict[str, list[ButtonEntity]]:
    by_owner: dict[str, list[ButtonEntity]] = defaultdict(list)
    for b in buttons:
        by_owner[b.owner_container_id].append(b)
    return dict(by_owner)


def _build_forms_by_owner(forms: list[FormEntity]) -> dict[str, list[FormEntity]]:
    by_owner: dict[str, list[FormEntity]] = defaultdict(list)
    for f in forms:
        by_owner[f.owner_container_id].append(f)
    return dict(by_owner)


def _build_orphan_controls_by_owner(
    orphans: list[FormControl],
) -> dict[str, list[FormControl]]:
    """Group Path A orphan controls (visible inputs not claimed by any form)
    by owner container so the content renderer can emit them inline."""
    by_owner: dict[str, list[FormControl]] = defaultdict(list)
    for c in orphans:
        by_owner[c.owner_container_id].append(c)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_scaffold(c: ContainerEntity) -> bool:
    return (
        not c.text_blocks
        and not c.control_refs
        and not c.button_refs
        and not c.form_refs
        and not c.is_blocking
    )


def container_flags(c: ContainerEntity) -> str:
    flags: list[str] = []
    if c.ax_modal:
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


def _resolve_scaffold_blocker(
    blocker_id: str | None,
    cmap: dict[str, ContainerEntity],
) -> str | None:
    """If blocker is a scaffold (invisible in markdown), walk up to nearest visible ancestor."""
    current = blocker_id
    while current:
        c = cmap.get(current)
        if c is None:
            return current  # unknown container, return as-is
        if not is_scaffold(c):
            return current  # visible in markdown, use this
        current = c.parent_container_id
    return blocker_id  # fallback to original if chain exhausted


def _resolve_blockers(
    is_occluded: bool,
    direct_blockers: list[str],
    owner_container_id: str,
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    form_blocker_id: str | None = None,
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
    # Fix E: if blocker is a scaffold, resolve to nearest visible ancestor.
    # _resolve_scaffold_blocker can return None (chain exhausted) — drop those.
    blockers = [r for b in blockers if (r := _resolve_scaffold_blocker(b, cmap)) is not None]
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


def _control_annotations(ctrl: FormControl) -> str:
    """Build exception-only annotation string for AX-enriched control fields.

    Only emits annotations when values are non-default. Returns empty string
    when all fields are default (zero extra tokens on first extraction).
    """
    parts: list[str] = []
    if ctrl.disabled:
        parts.append("disabled")
    if ctrl.checked == "checked":
        parts.append("checked")
    elif ctrl.checked == "mixed":
        parts.append("mixed")
    if ctrl.expanded == "expanded":
        parts.append("expanded")
    if ctrl.selected:
        parts.append("selected")
    if ctrl.has_popup:
        parts.append(f"hasPopup:{ctrl.has_popup}")
    if ctrl.cursor == "not-allowed":
        parts.append("cursor:not-allowed")
    if ctrl.focusable and ctrl.tag not in _INHERENTLY_FOCUSABLE:
        parts.append("focusable")
    if not parts:
        return ""
    return " [" + ", ".join(parts) + "]"


def _control_suffix(ctrl: FormControl) -> str:
    """Build suffix tokens for value and description — after annotations, before blocker."""
    suffix = ""
    if ctrl.current_value:
        suffix += f' value="{ctrl.current_value}"'
    return suffix


def _control_hint(ctrl: FormControl, indent: str) -> str | None:
    """Build hint line for ax_description. Returns None if no hint."""
    if ctrl.ax_description:
        return f'{indent}    (hint: "{ctrl.ax_description}")'
    return None


def _button_annotations(btn: ButtonEntity) -> str:
    """Build exception-only annotation string for AX-enriched button fields."""
    parts: list[str] = []
    if btn.disabled:
        parts.append("disabled")
    if btn.has_popup:
        parts.append(f"hasPopup:{btn.has_popup}")
    if btn.cursor == "not-allowed":
        parts.append("cursor:not-allowed")
    if btn.href:
        parts.append(f"href:{btn.href}")
    if not parts:
        return ""
    return " [" + ", ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Inline renderers
# ---------------------------------------------------------------------------


def _render_form_inline(
    form: FormEntity,
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    rendered_aids: set[str] | None = None,
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
    form_blocker_set: set[str] = {form_blocker_id} if form_blocker_id else set()
    for group in form.groups:
        for ctrl in group.controls:
            _render_one_control(
                ctrl, indent, lines, blocked_by_map, cmap,
                rendered_aids=rendered_aids,
                form_blocker_id=form_blocker_id,
                form_blocker_set=form_blocker_set,
            )


def _render_one_control(
    ctrl: FormControl,
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    *,
    rendered_aids: set[str] | None = None,
    form_blocker_id: str | None = None,
    form_blocker_set: set[str] | None = None,
) -> None:
    """Render a single FormControl as one V5 line (plus optional hint line).

    Shared by `_render_form_inline` (controls inside a form) and the orphan-
    control pass (visible inputs not claimed by any form, Path A). The output
    shape is identical so the LLM sees uniform input vocabulary.

    form_blocker_id/set are used by the form path to avoid double-tagging the
    form's own blocker on every child control; orphans pass None.
    """
    if form_blocker_set is None:
        form_blocker_set = set()
    ctype = ctrl.type or ""
    label = ctrl.ax_name or ctrl.label or ctrl.text or "(no label)"
    type_prefix = "draggable" if ctrl.draggable else (ctrl.ax_role or ctrl.role or ctrl.tag)
    blockers = _resolve_blockers(
        ctrl.occlusion.is_occluded,
        ctrl.occlusion.blocker_container_ids,
        ctrl.owner_container_id,
        blocked_by_map,
        cmap,
        form_blocker_id=form_blocker_id,
    )
    extra_blockers = [b for b in blockers if b not in form_blocker_set]
    ctrl_blocked = f" [blocked by {', '.join(extra_blockers)}]" if extra_blockers else ""
    type_str = f"[{ctype}]" if ctype else ""
    annotations = _control_annotations(ctrl)
    suffix = _control_suffix(ctrl)
    opts = ""
    if ctrl.options:
        shown = ctrl.options[:8]
        opts = " [options: " + ", ".join(f'"{o}"' for o in shown)
        if len(ctrl.options) > 8:
            opts += f", +{len(ctrl.options) - 8} more"
        opts += "]"

    if ctrl.tag == "canvas":
        w_px = round(ctrl.geometry.w * 1280)
        h_px = round(ctrl.geometry.h * 720)
        lines.append(
            f"{indent}  canvas `{ctrl.control_id}` [{w_px}x{h_px}]"
            f"{annotations}{ctrl_blocked}"
        )
    elif ctrl.slider_max is not None:
        slider_val = ctrl.slider_value if ctrl.slider_value is not None else 0
        slider_max = ctrl.slider_max
        orient = ctrl.slider_orientation or "horizontal"
        lines.append(
            f'{indent}  slider `{ctrl.control_id}` "{label}"'
            f" [{slider_val}/{slider_max}, {orient}]"
            f"{annotations}{ctrl_blocked}"
        )
    elif ctrl.cursor in (
        "col-resize", "row-resize", "ew-resize", "ns-resize",
        "nw-resize", "ne-resize", "sw-resize", "se-resize",
    ):
        lines.append(
            f'{indent}  resize-handle `{ctrl.control_id}` "{label}"'
            f" [cursor: {ctrl.cursor}]{annotations}{ctrl_blocked}"
        )
    else:
        lines.append(
            f'{indent}  {type_prefix}{type_str} `{ctrl.control_id}` "{label}"'
            f"{annotations}{suffix}{opts}{ctrl_blocked}"
        )
    if rendered_aids is not None:
        rendered_aids.add(ctrl.control_id)
    hint = _control_hint(ctrl, indent)
    if hint:
        lines.append(hint)


def _render_orphan_controls_inline(
    orphans: list[FormControl],
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    rendered_aids: set[str] | None = None,
) -> None:
    """Render Path A orphan controls — input-like elements that didn't get
    wrapped in a form (e.g. swiggy's lone <input id="location"> with no
    <form> ancestor). No `form form-N` wrapper line; each control emits one
    line at the container's indent. Uses the same per-control formatter as
    _render_form_inline so the model sees consistent input syntax."""
    for ctrl in orphans:
        _render_one_control(
            ctrl, indent, lines, blocked_by_map, cmap,
            rendered_aids=rendered_aids,
        )


def _render_buttons_inline(
    btns: list[ButtonEntity],
    form_control_ids: set[str],
    indent: str,
    lines: list[str],
    blocked_by_map: dict[str, list[str]],
    cmap: dict[str, ContainerEntity],
    compact_threshold: int = 4,
    rendered_aids: set[str] | None = None,
) -> None:
    btns = [b for b in btns if b.button_id not in form_control_ids]
    if not btns:
        return
    if len(btns) > compact_threshold:
        parts: list[str] = []
        for b in btns:
            bt = _value_or_none(b.ax_name or b.text)
            blockers = _resolve_blockers(
                b.occlusion.is_occluded,
                b.occlusion.blocker_container_ids,
                b.owner_container_id,
                blocked_by_map,
                cmap,
            )
            suffix = "*" if blockers else ""
            ann = _button_annotations(b)
            parts.append(f'"{bt}"{ann}{suffix}')
        lines.append(f"{indent}buttons({len(btns)}): {', '.join(parts)}")
    else:
        for b in btns:
            bt = _value_or_none(b.ax_name or b.text)
            blockers = _resolve_blockers(
                b.occlusion.is_occluded,
                b.occlusion.blocker_container_ids,
                b.owner_container_id,
                blocked_by_map,
                cmap,
            )
            blocked = f" [blocked by {', '.join(blockers)}]" if blockers else ""
            ann = _button_annotations(b)
            lines.append(f'{indent}button `{b.button_id}` "{bt}"{ann}{blocked}')
            if rendered_aids is not None:
                rendered_aids.add(b.button_id)


# ---------------------------------------------------------------------------
# Tree collectors for overlays (flatten children)
# ---------------------------------------------------------------------------


def _collect_tree_buttons(
    cid: str,
    buttons_by_owner: dict[str, list[ButtonEntity]],
    children_map: dict[str, list[str]],
) -> list[ButtonEntity]:
    result = list(buttons_by_owner.get(cid, []))
    for child_id in children_map.get(cid, []):
        result.extend(_collect_tree_buttons(child_id, buttons_by_owner, children_map))
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


def _collect_tree_orphan_controls(
    cid: str,
    orphan_by_owner: dict[str, list[FormControl]],
    children_map: dict[str, list[str]],
) -> list[FormControl]:
    result = list(orphan_by_owner.get(cid, []))
    for child_id in children_map.get(cid, []):
        result.extend(_collect_tree_orphan_controls(child_id, orphan_by_owner, children_map))
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


# ---------------------------------------------------------------------------
# DnD scaffold exemption
# ---------------------------------------------------------------------------


def build_scaffold_exempt(
    containers: list[ContainerEntity],
    forms: list[FormEntity],
    cmap: dict[str, ContainerEntity],
) -> set[str]:
    """Return container IDs that should be exempt from is_scaffold filtering.

    Containers near draggable controls get an exemption score based on
    LCA-based tree distance. Score > 0.5 → exempt. Library-fingerprinted
    drop zones are always exempt regardless of whether draggable controls
    were detected (e.g. when control limits trimmed them from forms).
    """
    # Always exempt fingerprinted drop zones — they're visible drag targets
    # even when the draggable pieces didn't make it into forms.
    exempt: set[str] = {c.container_id for c in containers if c.is_drop_zone}

    # Collect IDs of containers that own at least one draggable control
    draggable_owner_ids: set[str] = {
        ctrl.owner_container_id
        for form in forms
        for group in form.groups
        for ctrl in group.controls
        if ctrl.draggable
    }
    if not draggable_owner_ids:
        return exempt

    # Precompute ancestor chains for each draggable owner
    def _ancestor_chain(cid: str) -> list[str]:
        path: list[str] = []
        visited: set[str] = set()
        cur: str | None = cid
        while cur and cur not in visited:
            visited.add(cur)
            path.append(cur)
            c = cmap.get(cur)
            cur = c.parent_container_id if c else None
        return path

    # Precompute: ancestor_id → minimum hops from any draggable owner to that ancestor.
    # O(1) lookup per hop in _tree_distance instead of O(M) iteration over all owners.
    ancestor_to_min_owner_hops: dict[str, int] = {}
    for oid in draggable_owner_ids:
        for hop, ancestor in enumerate(_ancestor_chain(oid)):
            if ancestor not in ancestor_to_min_owner_hops:
                ancestor_to_min_owner_hops[ancestor] = hop
            else:
                ancestor_to_min_owner_hops[ancestor] = min(
                    ancestor_to_min_owner_hops[ancestor], hop
                )

    def _tree_distance(cid: str) -> int:
        """LCA-based tree distance: hops(cid→LCA) + min hops(any owner→LCA).
        O(D) per call — inner check is O(1) dict lookup.
        """
        visited: set[str] = set()
        cur: str | None = cid
        cid_hops = 0
        while cur and cur not in visited:
            if cur in ancestor_to_min_owner_hops:
                return cid_hops + ancestor_to_min_owner_hops[cur]
            visited.add(cur)
            c = cmap.get(cur)
            cur = c.parent_container_id if c else None
            cid_hops += 1
        return 999

    for c in containers:
        # Drop zones already in exempt from initial set comprehension above.
        if c.is_drop_zone:
            continue
        d = _tree_distance(c.container_id)
        text_len = len(c.summary.strip())
        score = math.exp(-_DND_DECAY_K * d) + (0.2 if 6 <= text_len <= 20 else 0.0)
        if score > 0.5:
            exempt.add(c.container_id)
    return exempt


# Main render
# ---------------------------------------------------------------------------


def render_master_markdown(extraction: PageFilterOutput) -> str:
    markdown, _ = render_master_markdown_with_meta(extraction)
    return markdown


def render_master_markdown_with_meta(
    extraction: PageFilterOutput,
) -> tuple[str, dict[str, object]]:
    containers = extraction.containers
    buttons = extraction.buttons
    forms = extraction.forms

    cmap: dict[str, ContainerEntity] = {c.container_id: c for c in containers}
    children_map = _build_children_map(containers)
    buttons_by_owner = _build_buttons_by_owner(buttons)
    forms_by_owner = _build_forms_by_owner(forms)
    orphan_controls_by_owner = _build_orphan_controls_by_owner(
        extraction.orphan_controls,
    )
    blocked_by_map = _build_blocked_by_map(containers)
    form_control_ids = _collect_form_control_button_ids(forms)
    rendered_aids: set[str] = set()

    # ---- DnD scaffold exemption ----
    # Containers near draggable controls are exempt from is_scaffold filtering
    # so drop zones (empty containers with placeholder text) survive to the markdown.
    scaffold_exempt: set[str] = build_scaffold_exempt(containers, forms, cmap)

    # Classify fixed containers and descendants as overlay
    fixed_ids = {c.container_id for c in containers if c.fixed_position}
    overlay_ids: set[str] = set()

    def _mark_overlay_tree(cid: str) -> None:
        overlay_ids.add(cid)
        for child_id in children_map.get(cid, []):
            _mark_overlay_tree(child_id)

    for fid in fixed_ids:
        _mark_overlay_tree(fid)

    lines: list[str] = []

    # ---- Header ----
    lines.append(f"# Page Representation ({_MARKDOWN_VERSION.upper()})")
    lines.append("")
    lines.append(f"url: `{extraction.page_url}`")
    lines.append(f'title: "{_value_or_none(extraction.page_title)}"')
    lines.append(
        f"{len(containers)} containers, {len(buttons)} buttons, "
        f"{len(forms)} forms, {extraction.blocking_container_count} blocking"
    )
    lines.append("")

    # ---- Overlays ----
    overlay_roots = [c for c in containers if c.container_id in fixed_ids]

    def _overlay_has_unblocked_buttons(c: ContainerEntity) -> bool:
        for b in buttons_by_owner.get(c.container_id, []):
            blockers = _resolve_blockers(
                b.occlusion.is_occluded,
                b.occlusion.blocker_container_ids,
                b.owner_container_id,
                blocked_by_map,
                cmap,
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

            header = f"`{cid}` z={c.z_index}"
            if heading:
                header += f' "{heading}"'
            tree_flags = _collect_tree_flags(cid, cmap, children_map)
            if tree_flags:
                header += " [" + ", ".join(sorted(tree_flags)) + "]"
            header += blocking_tag
            lines.append(header)
            rendered_aids.add(cid)

            all_text = _collect_tree_text(cid, cmap, children_map)
            all_buttons = _collect_tree_buttons(cid, buttons_by_owner, children_map)
            all_forms = _collect_tree_forms(cid, forms_by_owner, children_map)
            all_orphans = _collect_tree_orphan_controls(
                cid, orphan_controls_by_owner, children_map,
            )

            interactive_text: set[str] = {_value_or_none(b.text) for b in all_buttons}
            for f in all_forms:
                for group in f.groups:
                    for ctrl in group.controls:
                        interactive_text.add(ctrl.label or ctrl.text or "")
            for tb in all_text:
                if tb != heading and tb not in interactive_text:
                    lines.append(f'  "{tb}"')

            for f in all_forms:
                _render_form_inline(f, "  ", lines, blocked_by_map, cmap, rendered_aids)

            _render_buttons_inline(
                all_buttons,
                form_control_ids,
                "  ",
                lines,
                blocked_by_map,
                cmap,
                rendered_aids=rendered_aids,
            )

            # Path A — orphan inputs inside the overlay subtree (e.g. swiggy's
            # "Search for area, street name" inside the location side-drawer).
            _render_orphan_controls_inline(
                all_orphans, "  ", lines, blocked_by_map, cmap,
                rendered_aids=rendered_aids,
            )

            lines.append("")

    # ---- Content Tree ----
    lines.append("## Content")
    lines.append("")

    def _find_content_roots() -> list[str]:
        roots = [c.container_id for c in containers if not c.parent_container_id]
        result: list[str] = []
        visited: set[str] = set()

        def walk(cid: str) -> None:
            if cid in visited or cid in overlay_ids:
                return
            visited.add(cid)
            c = cmap.get(cid)
            if not c:
                return
            if is_scaffold(c) and cid not in scaffold_exempt:
                for child_id in children_map.get(cid, []):
                    walk(child_id)
            else:
                result.append(cid)

        for rid in roots:
            walk(rid)
        return result

    effective_roots = _find_content_roots()

    def _render_content(cid: str, depth: int = 0) -> None:
        c = cmap.get(cid)
        if not c or cid in overlay_ids:
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
        heading = _normalize_text(c.heading)
        header = f"{indent}`{cid}`{semantic_tag}{z_tag}"
        if heading:
            header += f' "{heading}"'
        header += container_flags(c)
        header += blocking_tag + blocked_tag
        lines.append(header)
        rendered_aids.add(cid)

        interactive_texts: set[str] = {
            _value_or_none(b.text) for b in buttons_by_owner.get(cid, [])
        }
        for f in forms_by_owner.get(cid, []):
            for group in f.groups:
                for ctrl in group.controls:
                    interactive_texts.add(ctrl.label or ctrl.text or "")
        for tb in c.text_blocks:
            if tb != heading and tb not in interactive_texts:
                lines.append(f'{indent}  "{tb}"')

        for f in forms_by_owner.get(cid, []):
            _render_form_inline(f, indent + "  ", lines, blocked_by_map, cmap, rendered_aids)

        _render_buttons_inline(
            buttons_by_owner.get(cid, []),
            form_control_ids,
            indent + "  ",
            lines,
            blocked_by_map,
            cmap,
            rendered_aids=rendered_aids,
        )

        # Path A — emit orphan controls (visible inputs that didn't get into any
        # form because they're alone in their container, e.g. swiggy's location).
        _render_orphan_controls_inline(
            orphan_controls_by_owner.get(cid, []),
            indent + "  ",
            lines,
            blocked_by_map,
            cmap,
            rendered_aids=rendered_aids,
        )

        for child_id in children_map.get(cid, []):
            child = cmap.get(child_id)
            if not child or child_id in overlay_ids:
                continue
            if (
                is_scaffold(child)
                and child_id not in scaffold_exempt
                and not children_map.get(child_id)
            ):
                continue
            _render_content(child_id, depth + 1)

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
    }
    return "\n".join(lines).rstrip() + "\n", meta
