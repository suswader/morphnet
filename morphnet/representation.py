"""
representation.py — Page representation pipeline for MorphNet.

CLEAN → COLLECT → STRUCTURE → FORMAT

Owns ALL AXTree-to-text transformations. Both CU agent and orchestrator
import from this module. Produces two distinct views from the same raw data:
  - CU view: section-based, interactive elements inline with context, footer excluded
  - Orchestrator view: text-only, full page including footer, no element IDs

Context tracking: a depth-keyed context_stack records the most recent
significant text at each AXTree depth during the walk. When a generic button
like "ADD" is encountered, the stack provides the nearest product name,
heading, or descriptive text — regardless of whether that text was a heading,
StaticText, or paragraph node.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from morphnet.session_manager import InteractiveElement


# ---------------------------------------------------------------------------
# Constants — role classification
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "textbox": "text field", "spinbutton": "number field",
    "combobox": "dropdown", "searchbox": "search field",
    "checkbox": "checkbox", "radio": "radio button",
    "switch": "toggle", "link": "link", "button": "button",
    "menuitem": "menu item", "tab": "tab", "slider": "slider",
}

# Roles that are pure rendering noise — make transparent (skip node, recurse children)
_NOISE_ROLES = frozenset({
    "inlinetextbox", "linebreak", "labeltext", "paragraph", "mark",
    "svg", "svgroot", "abbr", "superscript", "subscript", "ruby",
    "rubytext", "insertion", "deletion", "emphasis", "strong", "code",
    "time", "pre", "blockquote", "figcaption", "figure", "details",
    "summary",
})

# Structural roles that provide useful context even without an element ID
_STRUCTURAL_ROLES = frozenset({
    "heading", "navigation", "banner", "main", "contentinfo",
    "complementary", "form", "search", "region", "separator", "alert",
    "dialog", "alertdialog", "status", "log", "marquee", "timer",
    "toolbar", "menu", "menubar", "tablist", "tabpanel", "tree",
    "treegrid", "grid", "table", "list",
})

# Generic wrapper roles — always transparent
_GENERIC_ROLES = frozenset({
    "none", "generic", "genericcontainer", "group", "section",
    "article", "div", "span",
})

# Footer detection
_FOOTER_ROLES = frozenset({"contentinfo"})
_FOOTER_KEYWORDS = (
    "footer", "copyright", "\u00a9", "privacy policy", "terms of service",
    "all rights reserved", "cookie policy", "terms & conditions",
    "social links",
)

# CSS-class-like accessible names to exclude (framework artifacts)
_CSS_CLASS_PATTERN = re.compile(
    r"^[a-z]+-[a-z]+-[a-z]+$|^css-|^sc-|^styled-|^_[a-zA-Z0-9]+$",
    re.IGNORECASE,
)

# Generic element labels that benefit from nearby-text context
_GENERIC_LABELS = frozenset({
    "add", "remove", "delete", "edit", "view", "select", "more",
    "details", "go", "ok", "yes", "no", "cancel", "submit", "save",
    "update", "add to cart", "buy", "+", "-", "x",
})


# ===================================================================
# CLEAN phase — whitespace normalization, text quality filters
# ===================================================================

def _normalize_ws(s: str) -> str:
    """Normalize whitespace and strip Unicode icon characters for matching.

    AXTree names sometimes include trailing icon characters (e.g. \\ue923)
    from icon fonts that the JS enumerator doesn't capture.  Strip Private Use
    Area chars (U+E000–U+F8FF) plus common symbol blocks so the lookup matches.
    """
    # Remove Private Use Area characters (icon fonts like \\ue923)
    s = re.sub(r"[\ue000-\uf8ff]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def should_include_element(name: str | None, role: str) -> bool:
    """Filter out elements with CSS-generated or empty accessible names.

    Unnamed interactive elements (buttons, inputs) are kept — icon buttons
    are handled by Rule 3 during the walk. CSS-class names like "css-1a2b3c"
    or "sc-bdVTJa" are framework artifacts, not real accessible names.
    """
    if not name or not name.strip():
        # Keep unnamed interactive elements (icon buttons, empty fields)
        return role.lower() in (
            "button", "link", "checkbox", "radio", "switch",
            "textbox", "searchbox", "combobox", "spinbutton",
            "slider", "menuitem", "tab",
        )
    if _CSS_CLASS_PATTERN.match(name.strip()):
        return False
    return True


def compress_text(text: str, max_chars: int = 80) -> str:
    """Truncate long text at a sentence boundary.

    Preserves meaning better than hard truncation — cuts at the last
    sentence-ending punctuation within budget, or at last space with "...".
    """
    if len(text) <= max_chars:
        return text
    # Find last sentence-ending punctuation within budget
    for i in range(max_chars - 1, max_chars // 2, -1):
        if text[i] in ".!?":
            return text[: i + 1]
    # No sentence boundary — truncate at last space
    last_space = text.rfind(" ", 0, max_chars)
    if last_space > max_chars // 2:
        return text[:last_space] + "..."
    return text[:max_chars] + "..."


# ===================================================================
# COLLECT phase — element formatting, functional role inference
# ===================================================================

def _infer_functional_role(el: InteractiveElement) -> str:
    """Infer what an element functionally IS, not what its DOM tag is.

    A div with text "Search for restaurant" is a search trigger.
    A div with text "Add to Cart" is a button.
    """
    base_role = el.role.lower()
    name_lower = (el.name or "").lower()

    if base_role in _ROLE_MAP:
        return _ROLE_MAP[base_role]

    if base_role in ("div", "span", "section", "article", "li", "td", "img", "svg"):
        if any(kw in name_lower for kw in ("search", "find", "look up")):
            return "search trigger"
        if any(kw in name_lower for kw in ("add to cart", "buy", "purchase", "checkout")):
            return "button"
        if any(kw in name_lower for kw in ("sign in", "log in", "login", "sign up")):
            return "button"
        return "clickable area"

    return base_role


def format_element(el: InteractiveElement, nearby_context: str | None = None) -> str:
    """Natural language element description with optional nearby-text context.

    [5] button "ADD" — near: Vietnamese Cold Brew
    [6] dropdown "Size" — shows "Medium", collapsed
    [7] text field "Quantity" — contains "1", required
    """
    friendly = _infer_functional_role(el)
    parts: list[str] = [f"[{el.element_id}]", friendly]

    if el.name:
        parts.append(f'"{el.name}"')

    state_parts: list[str] = []
    if el.value is not None and el.value != "":
        match el.role:
            case "textbox" | "searchbox" | "spinbutton":
                state_parts.append(f'contains "{el.value}"')
            case "combobox":
                state_parts.append(f'shows "{el.value}"')
            case _:
                state_parts.append(f'value: "{el.value}"')
    elif el.role in ("textbox", "searchbox"):
        state_parts.append("empty")

    if "checked" in el.states:
        state_parts.append("checked")
    if "expanded" in el.states:
        state_parts.append("expanded")
    elif el.role == "combobox":
        state_parts.append("collapsed")
    if "disabled" in el.states:
        state_parts.append("disabled")
    if "required" in el.states:
        state_parts.append("required")
    if "focused" in el.states:
        state_parts.append("focused")

    if state_parts:
        parts.append("\u2014")
        parts.append(", ".join(state_parts))

    # Nearby-text context for generic/short-named elements
    if nearby_context and _should_add_nearby_context(el):
        if state_parts:
            parts.append(f"| near: {nearby_context}")
        else:
            parts.append(f"\u2014 near: {nearby_context}")

    return " ".join(parts)


def _should_add_nearby_context(el: InteractiveElement) -> bool:
    """Returns True when an element's name is generic and needs disambiguation."""
    if el.role.lower() not in ("button", "link", "div", "span", "clickable area",
                                "img", "svg", "li", "td"):
        return False
    name = (el.name or "").strip().lower()
    if not name:
        return True  # Unnamed elements always benefit from context
    if len(name) <= 12:
        return True
    return name in _GENERIC_LABELS


# ===================================================================
# STRUCTURE phase — context tracking, text dedup, footer detection
# ===================================================================

class _ContextStack:
    """Tracks the first significant text at each depth level per scope.

    During the tree walk, when we encounter text content at depth N,
    we store it ONLY if no context exists at that depth yet (first-text-wins).
    This ensures the product/item NAME (first text in a card) is used
    as context, not a subsequent description or price.

    When starting a new sibling subtree (e.g., next product card),
    call reset_scope(depth) to clear context at that depth and deeper,
    allowing the new sibling to establish its own context.

    This naturally associates "ADD" buttons with their product names
    even when the product name is a StaticText or paragraph node.
    """

    def __init__(self):
        self._stack: dict[int, str] = {}

    def update(self, depth: int, text: str) -> None:
        """Record first significant text at this depth. Clears deeper entries.

        First-text-wins: if context already exists at this depth, don't overwrite.
        This preserves the identifying name (first text) over descriptions/prices.
        """
        clean = text.strip() if text else ""
        if len(clean) < 2:
            return
        # First-text-wins at each depth within a scope
        if depth not in self._stack:
            self._stack[depth] = clean
        # Clear deeper entries — they belong to a previous sibling's subtree
        to_remove = [d for d in self._stack if d > depth]
        for d in to_remove:
            del self._stack[d]

    def reset_scope(self, depth: int) -> None:
        """Clear context at this depth and deeper.

        Called before processing each child in a sibling group so each
        child subtree establishes its own context independently.
        """
        to_remove = [d for d in self._stack if d >= depth]
        for d in to_remove:
            del self._stack[d]

    def get_nearest(self, depth: int, max_lookback: int = 4) -> str | None:
        """Find the nearest significant text at current or shallower depths.

        Prefers same-depth or one level up (sibling/parent context).
        """
        for d in range(depth, max(depth - max_lookback, -1), -1):
            if d in self._stack:
                ctx = self._stack[d]
                if len(ctx) > 2:
                    return ctx[:80]  # Cap context length for prompt brevity
        return None


class _TextDedup:
    """Rolling window + global frequency deduplication for StaticText nodes.

    Real AXTrees have heavy text repetition — the same text appears in
    nested nodes, siblings, and cousins. The parent-name check catches
    direct parent repetition, the rolling window catches nearby repeats,
    and the global frequency cap suppresses accessibility noise like
    "Skip to view cart button" (32x) or "Learn more" (24x).
    """

    def __init__(self, window: int = 3, max_repeats: int = 3):
        self._window = window
        self._recent: list[str] = []
        self._global_counts: dict[str, int] = {}
        self._max_repeats = max_repeats

    def is_duplicate(self, text: str) -> bool:
        """Returns True if normalized text was seen in the last N entries
        or has already appeared max_repeats times globally."""
        normalized = _normalize_ws(text)
        if not normalized:
            return True
        # Global frequency cap — suppress text that has appeared too many times
        count = self._global_counts.get(normalized, 0)
        if count >= self._max_repeats:
            return True
        # Rolling window — suppress nearby duplicates
        if normalized in self._recent:
            return True
        self._recent.append(normalized)
        if len(self._recent) > self._window:
            self._recent.pop(0)
        self._global_counts[normalized] = count + 1
        return False


def _is_footer_node(node: dict) -> bool:
    """Detect footer regions by role or content keywords.

    Checks keywords on ALL text-bearing nodes, not just structural ones.
    The keywords (©, privacy policy, social links, etc.) are specific enough
    to avoid false positives on main content.
    """
    role = node.get("role", "").lower()
    if role in _FOOTER_ROLES:
        return True
    name = (node.get("name", "") or "").lower()
    if not name:
        return False
    return any(kw in name for kw in _FOOTER_KEYWORDS)


# ===================================================================
# FORMAT phase — CU representation (sections, inline elements, context)
# ===================================================================

def build_cu_representation(
    raw_axtree: dict,
    interactive_elements: list[InteractiveElement],
    *,
    exclude_footer: bool = True,
    max_text_chars: int = 200,
) -> str:
    """Build section-based page representation for CU agent.

    Interactive elements appear INLINE with their context text.
    Generic buttons get nearby-text disambiguation via context_stack.
    Footer content excluded by default.

    Returns the full representation string including a Fields/Actionable
    quick-reference summary at the bottom.
    """
    # Pre-filter elements through quality check
    filtered_els = [
        el for el in interactive_elements
        if should_include_element(el.name, el.role)
    ]

    # Build lookup: (role_lower, name_lower_normalized) → list of InteractiveElement
    id_lookup: dict[tuple[str, str], list[InteractiveElement]] = defaultdict(list)
    name_only_lookup: dict[str, list[InteractiveElement]] = defaultdict(list)
    for el in filtered_els:
        norm_name = _normalize_ws(el.name or "")
        key = (el.role.lower(), norm_name)
        id_lookup[key].append(el)
        if norm_name and el.role.lower() in ("div", "span", "section", "article", "li", "td"):
            name_only_lookup[norm_name].append(el)

    lines: list[str] = []
    context = _ContextStack()
    dedup = _TextDedup(window=3)
    # Track which elements got nearby context during the walk
    context_map: dict[int, str] = {}
    # Track which element IDs were consumed (appeared in the tree)
    consumed_ids: set[int] = set()

    _cu_walk(
        raw_axtree, 0, "", lines,
        id_lookup, name_only_lookup,
        context, dedup, context_map, consumed_ids,
        exclude_footer=exclude_footer,
        max_text_chars=max_text_chars,
    )

    # Include ALL visible interactive elements in the summary.
    # Footer exclusion is handled by _is_footer_node() during the walk.
    # Elements that didn't match during the AXTree walk (due to role/name
    # mismatch between JS enumerator and AXTree) still need to be visible
    # to the agent — the summary is their safety net.
    summary_els = filtered_els
    summary = format_elements_summary(summary_els, context_map=context_map)
    if summary:
        lines.append("")
        lines.append(summary)

    return "\n".join(lines)


def _cu_walk(
    node: dict,
    depth: int,
    parent_name: str,
    lines: list[str],
    id_lookup: dict[tuple[str, str], list[InteractiveElement]],
    name_only_lookup: dict[str, list[InteractiveElement]],
    context: _ContextStack,
    dedup: _TextDedup,
    context_map: dict[int, str],
    consumed_ids: set[int],
    *,
    exclude_footer: bool = True,
    max_text_chars: int = 200,
    max_depth: int = 20,
) -> None:
    """Recursive walk producing section-based, context-rich output."""
    if depth > max_depth:
        return

    role = node.get("role", "none")
    role_lower = role.lower()
    name = node.get("name", "")
    children = node.get("children", [])

    # --- Footer exclusion ---
    if exclude_footer and _is_footer_node(node):
        return

    # --- Rule 1a: Always skip InlineTextBox (Blink rendering artifact) ---
    if role_lower == "inlinetextbox":
        return

    # --- Rule 1b: Skip redundant StaticText (parent already shows this text) ---
    if role_lower == "statictext":
        if name and _normalize_ws(name) == _normalize_ws(parent_name):
            return
        # Check if this StaticText matches an interactive element
        norm = _normalize_ws(name)
        key = (role_lower, norm)
        if key in id_lookup and id_lookup[key]:
            el = id_lookup[key].pop(0)
            consumed_ids.add(el.element_id)
            nearby = context.get_nearest(depth)
            if nearby and _should_add_nearby_context(el):
                context_map[el.element_id] = nearby
            lines.append(f"{'  ' * depth}{format_element(el, nearby)}")
            return
        if norm in name_only_lookup and name_only_lookup[norm]:
            el = name_only_lookup[norm].pop(0)
            consumed_ids.add(el.element_id)
            nearby = context.get_nearest(depth)
            if nearby and _should_add_nearby_context(el):
                context_map[el.element_id] = nearby
            lines.append(f"{'  ' * depth}{format_element(el, nearby)}")
            return
        # Non-interactive StaticText — dedup, compress, and update context
        if name.strip():
            if not dedup.is_duplicate(name):
                compressed = compress_text(name.strip(), max_text_chars)
                lines.append(f"{'  ' * depth}text \"{compressed}\"")
                context.update(depth, name.strip())
        return

    # --- Root WebArea — transparent, recurse with sibling collapse ---
    if role_lower == "webarea":
        _recurse_children_with_collapse(
            children, depth, name, lines, id_lookup, name_only_lookup,
            context, dedup, context_map, consumed_ids,
            exclude_footer=exclude_footer,
            max_text_chars=max_text_chars, max_depth=max_depth,
        )
        return

    # --- Noise roles — transparent (skip node, recurse children) ---
    if role_lower in _NOISE_ROLES:
        _recurse_children_with_collapse(
            children, depth, parent_name, lines, id_lookup, name_only_lookup,
            context, dedup, context_map, consumed_ids,
            exclude_footer=exclude_footer,
            max_text_chars=max_text_chars, max_depth=max_depth,
        )
        return

    # --- Tables → Markdown ---
    if role_lower == "table":
        _render_table_markdown(node, depth, lines)
        return

    # --- Lists → Markdown (simple lists only) ---
    if role_lower == "list":
        if _is_simple_list(node):
            _render_list_markdown(node, depth, lines)
            return
        # Complex list (product grids, filter groups, card layouts).
        # Walk each child individually — bypasses sibling collapse so
        # product cards and filter checkboxes aren't collapsed into
        # "[24 listitem items: (unnamed)]". Each listitem's subtree
        # (article → link + price + piece count) is preserved.
        for child in children:
            if child.get("children"):
                context.reset_scope(depth)
            _cu_walk(
                child, depth, parent_name, lines,
                id_lookup, name_only_lookup,
                context, dedup, context_map, consumed_ids,
                exclude_footer=exclude_footer,
                max_text_chars=max_text_chars,
                max_depth=max_depth,
            )
        return

    # --- Try to match to an InteractiveElement ---
    norm_name = _normalize_ws(name)
    key = (role_lower, norm_name)
    matched_el: InteractiveElement | None = None
    if key in id_lookup and id_lookup[key]:
        matched_el = id_lookup[key].pop(0)
    elif norm_name and norm_name in name_only_lookup and name_only_lookup[norm_name]:
        matched_el = name_only_lookup[norm_name].pop(0)

    # --- Rule 3: Collapse unnamed button/link with only image children ---
    if matched_el and not matched_el.name and role_lower in ("button", "link"):
        child_roles = [c.get("role", "").lower() for c in children]
        if all(r in ("img", "image", "svg", "svgroot", "presentation", "none", "")
               for r in child_roles):
            consumed_ids.add(matched_el.element_id)
            nearby = context.get_nearest(depth)
            if nearby:
                context_map[matched_el.element_id] = nearby
            lines.append(f"{'  ' * depth}{format_element(matched_el, nearby)} (icon)")
            return

    if matched_el:
        consumed_ids.add(matched_el.element_id)
        nearby = context.get_nearest(depth)
        if nearby and _should_add_nearby_context(matched_el):
            context_map[matched_el.element_id] = nearby
        lines.append(f"{'  ' * depth}{format_element(matched_el, nearby)}")
        # Don't recurse into interactive elements — their children are noise
        return

    # --- Generic/unnamed wrappers — transparent ---
    if role_lower in _GENERIC_ROLES and not name:
        _recurse_children_with_collapse(
            children, depth, name, lines, id_lookup, name_only_lookup,
            context, dedup, context_map, consumed_ids,
            exclude_footer=exclude_footer,
            max_text_chars=max_text_chars, max_depth=max_depth,
        )
        return

    # --- Non-structural — transparent with optional context text ---
    if role_lower not in _STRUCTURAL_ROLES:
        if name.strip():
            if not dedup.is_duplicate(name):
                compressed = compress_text(name.strip(), max_text_chars)
                lines.append(f"{'  ' * depth}text \"{compressed}\"")
                context.update(depth, name.strip())
        _recurse_children_with_collapse(
            children, depth, name, lines, id_lookup, name_only_lookup,
            context, dedup, context_map, consumed_ids,
            exclude_footer=exclude_footer,
            max_text_chars=max_text_chars, max_depth=max_depth,
        )
        return

    # --- Structural role — section boundary, update context ---
    indent = "  " * depth
    line = f"{indent}{role}"
    if name:
        line += f' "{name}"'
        context.update(depth, name)
    level = node.get("level")
    if level:
        line += f" \u2014 level {level}"
    lines.append(line)

    _recurse_children_with_collapse(
        children, depth + 1, name, lines, id_lookup, name_only_lookup,
        context, dedup, context_map, consumed_ids,
        exclude_footer=exclude_footer,
        max_text_chars=max_text_chars, max_depth=max_depth,
    )


def _recurse_children_with_collapse(
    children: list[dict],
    depth: int,
    parent_name: str,
    lines: list[str],
    id_lookup: dict,
    name_only_lookup: dict,
    context: _ContextStack,
    dedup: _TextDedup,
    context_map: dict[int, str],
    consumed_ids: set[int],
    *,
    exclude_footer: bool = True,
    max_text_chars: int = 200,
    max_depth: int = 20,
) -> None:
    """Recurse into children, collapsing repetitive sibling groups (Rule 4)."""
    if not children:
        return

    groups = _detect_repetitive_groups(children)

    for group_role, group_children in groups:
        if (group_role is not None
                and len(group_children) >= 5
                and group_role.lower() not in _GENERIC_ROLES
                and group_role.lower() != "statictext"):
            # Before collapsing, check if items would lose critical content.
            #
            # Three cases where collapse is destructive:
            # 1. Rich subtrees (product cards with nested interactive elements)
            # 2. Actionable control groups (filter buttons, sort options) —
            #    these are leaf-level interactive elements the agent needs to
            #    see individually to make informed choices. Collapsing
            #    "Rated 4+", "Rs 250+" behind "... and 5 more" hides
            #    strategically valuable UI controls.
            has_rich = _group_has_rich_content(group_children)
            is_actionable_group = (
                group_role.lower() in ("button", "link", "tab", "menuitem", "option")
                and all(
                    not child.get("children")
                    or len(child.get("children", [])) <= 1
                    for child in group_children
                )
            )
            if has_rich or is_actionable_group:
                # Rich content — walk each child individually, no collapse
                for child in group_children:
                    if child.get("children"):
                        context.reset_scope(depth)
                    _cu_walk(
                        child, depth, parent_name, lines,
                        id_lookup, name_only_lookup,
                        context, dedup, context_map, consumed_ids,
                        exclude_footer=exclude_footer,
                        max_text_chars=max_text_chars,
                        max_depth=max_depth,
                    )
                continue

            # Partial collapse — show first few items individually for context,
            # then summarize the rest. Agent sees top results with full detail
            # while keeping the representation bounded.
            _PARTIAL_SHOW = 3
            indent = "  " * depth

            # Walk the first few items through the full pipeline
            for child in group_children[:_PARTIAL_SHOW]:
                if child.get("children"):
                    context.reset_scope(depth)
                _cu_walk(
                    child, depth, parent_name, lines,
                    id_lookup, name_only_lookup,
                    context, dedup, context_map, consumed_ids,
                    exclude_footer=exclude_footer,
                    max_text_chars=max_text_chars,
                    max_depth=max_depth,
                )

            # Collapse the remaining items — consume id_lookup entries so they
            # still appear in the Fields/Actionable summary at the bottom
            remaining = group_children[_PARTIAL_SHOW:]
            if remaining:
                collapsed_ids: list[int] = []
                for child in remaining:
                    cname = _normalize_ws(child.get("name", ""))
                    crole = child.get("role", "").lower()
                    ckey = (crole, cname)
                    if ckey in id_lookup and id_lookup[ckey]:
                        el = id_lookup[ckey].pop(0)
                        collapsed_ids.append(el.element_id)
                        consumed_ids.add(el.element_id)
                    elif cname and cname in name_only_lookup and name_only_lookup[cname]:
                        el = name_only_lookup[cname].pop(0)
                        collapsed_ids.append(el.element_id)
                        consumed_ids.add(el.element_id)
                id_hint = ""
                if collapsed_ids:
                    id_hint = f" (IDs [{collapsed_ids[0]}]-[{collapsed_ids[-1]}])"
                lines.append(
                    f"{indent}... and {len(remaining)} more {group_role} items{id_hint}"
                )
        else:
            for child in group_children:
                # Reset context scope for container children (nodes with
                # children of their own, like product cards). Leaf nodes
                # (StaticText, buttons) within the same container share
                # context — that's how "ADD" gets linked to "Vietnamese Cold Brew".
                if child.get("children"):
                    context.reset_scope(depth)
                _cu_walk(
                    child, depth, parent_name, lines,
                    id_lookup, name_only_lookup,
                    context, dedup, context_map, consumed_ids,
                    exclude_footer=exclude_footer,
                    max_text_chars=max_text_chars,
                    max_depth=max_depth,
                )


def _detect_repetitive_groups(
    children: list[dict],
) -> list[tuple[str | None, list[dict]]]:
    """Group consecutive children by role.

    A group with role=None means these children don't form a repetitive block.
    A group with a role string means 2+ consecutive children share that role.
    Only groups with 5+ items get collapsed by the caller.
    """
    if not children:
        return []

    groups: list[tuple[str | None, list[dict]]] = []
    current_role = children[0].get("role", "")
    current_group: list[dict] = [children[0]]

    for child in children[1:]:
        child_role = child.get("role", "")
        if child_role == current_role:
            current_group.append(child)
        else:
            groups.append(
                (current_role if len(current_group) >= 2 else None, current_group)
            )
            current_role = child_role
            current_group = [child]

    groups.append(
        (current_role if len(current_group) >= 2 else None, current_group)
    )
    return groups


# ===================================================================
# Element summary — Fields + Actionable quick-reference
# ===================================================================

def format_elements_summary(
    elements: list[InteractiveElement],
    context_map: dict[int, str] | None = None,
) -> str:
    """Generate Fields + Actionable summary.

    When context_map is provided, generic buttons/links get
    "near: ..." annotations matching what appears inline in the tree.
    """
    fields: list[str] = []
    actionable: list[str] = []

    for el in elements:
        friendly = _infer_functional_role(el)

        if el.role in ("textbox", "searchbox", "spinbutton", "combobox"):
            val = f'"{el.value}"' if el.value else "empty"
            extra = ", required" if "required" in el.states else ""
            fields.append(
                f'  [{el.element_id}] "{el.name}" ({val}) \u2014 {friendly}{extra}'
            )
        elif el.role in ("checkbox", "radio", "switch"):
            state = "checked" if "checked" in el.states else "unchecked"
            fields.append(
                f'  [{el.element_id}] "{el.name}" ({state}) \u2014 {friendly}'
            )
        else:
            label = el.name or "(unnamed)"
            ctx = ""
            if context_map and el.element_id in context_map:
                ctx = f" | near: {context_map[el.element_id]}"
            actionable.append(
                f'  [{el.element_id}] "{label}" \u2014 {friendly}{ctx}'
            )

    lines: list[str] = []
    if fields:
        lines.append("--- Fields ---")
        lines.extend(fields)
    if actionable:
        if fields:
            lines.append("")
        lines.append("--- Actionable ---")
        lines.extend(actionable)
    return "\n".join(lines)


# ===================================================================
# Table / List rendering helpers
# ===================================================================

def _render_table_markdown(node: dict, depth: int, lines: list[str]) -> None:
    """Convert an AXTree table node to Markdown table format."""
    indent = "  " * depth
    rows: list[list[str]] = []
    headers: list[str] = []

    for child in node.get("children", []):
        role = child.get("role", "")
        if role in ("rowgroup", "row"):
            cells = _extract_row_cells(child)
            if cells:
                rows.append(cells)
        elif role == "columnheader":
            headers.append(child.get("name", ""))

    if not headers and rows:
        headers = rows.pop(0)

    if headers:
        lines.append(f"{indent}| {' | '.join(headers)} |")
        lines.append(f"{indent}| {' | '.join('---' for _ in headers)} |")
    for row in rows:
        while len(row) < len(headers):
            row.append("")
        lines.append(f"{indent}| {' | '.join(row[: len(headers) or len(row)])} |")


def _extract_row_cells(node: dict) -> list[str]:
    """Extract cell text from a table row or rowgroup."""
    cells: list[str] = []
    role = node.get("role", "")

    if role == "row":
        for child in node.get("children", []):
            cr = child.get("role", "")
            if cr in ("cell", "gridcell", "columnheader", "rowheader"):
                cells.append(child.get("name", ""))
    elif role == "rowgroup":
        for child in node.get("children", []):
            row_cells = _extract_row_cells(child)
            if row_cells:
                cells = row_cells
                break
    return cells


_INTERACTIVE_ROLES_FOR_COLLAPSE = frozenset({
    "button", "link", "checkbox", "radio", "switch",
    "textbox", "searchbox", "combobox", "spinbutton",
    "slider", "menuitem", "tab",
})


def _subtree_has_interactive(node: dict, max_depth: int = 4) -> bool:
    """Check if node's subtree contains any interactive elements within max_depth."""
    if max_depth <= 0:
        return False
    for child in node.get("children", []):
        role = (child.get("role") or "").lower()
        if role in _INTERACTIVE_ROLES_FOR_COLLAPSE:
            return True
        if _subtree_has_interactive(child, max_depth - 1):
            return True
    return False


def _group_has_rich_content(group_children: list[dict]) -> bool:
    """Check if a sibling group has rich content that would be lost by collapsing.

    Samples a few items and checks for interactive elements in their subtrees.
    Product cards (links, buttons), filter items (checkboxes), and similar
    content-rich groups will be detected and preserved.

    Simple groups (nav links with just text, repeated decorative items)
    are safe to collapse.
    """
    sample = group_children[:3]
    return any(_subtree_has_interactive(item) for item in sample)


def _is_simple_list(node: dict) -> bool:
    """Detect whether a list is simple (nav menus, breadcrumbs) or complex (product grids).

    Simple lists have flat listitems: listitem → link "Home", listitem → text "SHOP".
    Complex lists have deep listitems: listitem → article → link → image + text + button.

    Only simple lists should use the compact markdown renderer. Complex lists need
    full recursive walking so nested content (product names, prices, filters) isn't lost.
    """
    listitems = [c for c in node.get("children", []) if c.get("role") == "listitem"]
    if not listitems:
        return True  # Empty list, treat as simple

    for item in listitems[:5]:  # Sample first 5 items
        children = item.get("children", [])
        for child in children:
            child_role = (child.get("role") or "").lower()
            # Article, section, form inside a listitem → complex (product cards, filter groups)
            if child_role in ("article", "section", "form", "region"):
                return False
            # Child with its own deep children → complex
            grandchildren = child.get("children", [])
            if len(grandchildren) > 2:
                return False
        # Also check for interactive elements deeper in the subtree.
        # Catches cases like: listitem → generic → link → image + heading
        # where the 2-level check above misses the complexity.
        if _subtree_has_interactive(item):
            return False
    return True


def _render_list_markdown(node: dict, depth: int, lines: list[str]) -> None:
    """Convert an AXTree list node to Markdown list format."""
    indent = "  " * depth
    for child in node.get("children", []):
        if child.get("role") == "listitem":
            name = child.get("name", "")
            if name:
                lines.append(f"{indent}- {name}")
            else:
                for subchild in child.get("children", []):
                    sub_name = subchild.get("name", "")
                    if sub_name:
                        lines.append(f"{indent}- {sub_name}")
                        break


# ===================================================================
# Orchestrator representation (text-only, no element IDs, full page)
# ===================================================================

def build_orchestrator_representation(
    raw_axtree: dict | None,
    visible_elements: list | None = None,
) -> str:
    """Text-only AXTree distillation for orchestrator planning.

    No element IDs. No footer exclusion. Includes text dedup and
    compression. Cross-checks actionable elements against visibility set.
    """
    if not raw_axtree:
        return "(no AXTree available)"

    visible_names: set[tuple[str, str]] | None = None
    if visible_elements:
        visible_names = set()
        for el in visible_elements:
            role = getattr(el, "role", "") or ""
            name = getattr(el, "name", "") or ""
            visible_names.add((role.lower(), name.lower().strip()))

    lines: list[str] = []
    actions: list[str] = []
    dedup = _TextDedup(window=3)
    _orchestrator_walk(
        raw_axtree, 0, lines, actions, dedup, visible_names=visible_names,
    )

    if actions:
        lines.append("")
        lines.append(f"Available actions: {', '.join(dict.fromkeys(actions))}")

    return "\n".join(lines)


def _orchestrator_walk(
    node: dict,
    depth: int,
    lines: list[str],
    actions: list[str],
    dedup: _TextDedup,
    *,
    max_depth: int = 30,
    visible_names: set[tuple[str, str]] | None = None,
) -> None:
    """Recursive walk for orchestrator's text-only AXTree."""
    if depth > max_depth:
        return
    role = node.get("role", "none")
    name = node.get("name", "")
    children = node.get("children", [])

    if role == "WebArea":
        page_name = name or "Untitled"
        lines.append(f'Page: "{page_name}"')
        lines.append("")
        for child in children:
            _orchestrator_walk(
                child, depth, lines, actions, dedup,
                max_depth=max_depth, visible_names=visible_names,
            )
        return

    if role.lower() in ("none", "generic", "genericcontainer") and not name:
        for child in children:
            _orchestrator_walk(
                child, depth, lines, actions, dedup,
                max_depth=max_depth, visible_names=visible_names,
            )
        return

    if role == "StaticText":
        if name and len(name.strip()) > 2:
            if not dedup.is_duplicate(name):
                compressed = compress_text(name.strip())
                lines.append(f"{'  ' * depth}text \"{compressed}\"")
        return

    indent = "  " * depth

    is_actionable = role.lower() in ("button", "link", "textbox", "searchbox", "combobox")
    not_visible = False
    if is_actionable and name and visible_names is not None:
        key = (role.lower(), name.lower().strip())
        if key not in visible_names:
            not_visible = True

    if role.lower() in ("button", "link") and name and not not_visible:
        actions.append(f'"{name}"')

    # Filter CSS-class names
    display_name = name
    if name and _CSS_CLASS_PATTERN.match(name.strip()):
        display_name = ""

    line = f"{indent}{role}"
    if display_name:
        line += f' "{display_name}"'
    if not_visible:
        line += " [not visible]"
    level = node.get("level")
    if level:
        line += f" \u2014 level {level}"
    value = node.get("value")
    if value and value != name:
        line += f' \u2014 "{value}"'

    lines.append(line)

    for child in children:
        _orchestrator_walk(
            child, depth + 1, lines, actions, dedup,
            max_depth=max_depth, visible_names=visible_names,
        )


# ===================================================================
# Reflector representation — content-focused, no element IDs, card-aware
# ===================================================================

# Roles that indicate navigational chrome (banners, navbars, footers).
# Reflector compresses these to a single-line summary.
_CHROME_ROLES = frozenset({
    "banner", "navigation", "contentinfo", "complementary",
    "toolbar", "menubar", "menu",
})

# Semantic container roles whose direct children may form repeating cards.
_CARD_CONTAINER_ROLES = frozenset({"article", "section", "region"})


def build_reflector_representation(
    raw_axtree: dict,
    *,
    max_chars: int = 10_000,
) -> str:
    """Content-focused page representation for subtask reflection.

    Different goal from CU: the reflector checks whether the intended task
    is visibly done. No element IDs, no Fields/Actionable summary.
    Instead: aggressive chrome compression, card-aware formatting,
    and maximum budget for actual page content.
    """
    if not raw_axtree:
        return "(no AXTree available)"

    lines: list[str] = []
    dedup = _TextDedup(window=3)

    _reflector_walk(raw_axtree, 0, lines, dedup)

    result = "\n".join(lines)
    # Soft truncation at sentence boundary near budget
    if len(result) > max_chars:
        cut = result.rfind("\n", 0, max_chars)
        if cut > max_chars // 2:
            result = result[:cut] + "\n... (truncated)"
        else:
            result = result[:max_chars] + "\n... (truncated)"
    return result


def _reflector_walk(
    node: dict,
    depth: int,
    lines: list[str],
    dedup: _TextDedup,
    *,
    max_depth: int = 25,
) -> None:
    """Recursive walk for reflector: content-dense, chrome-compressed, card-aware."""
    if depth > max_depth:
        return

    role = node.get("role", "none")
    role_lower = role.lower()
    name = node.get("name", "")
    children = node.get("children", [])

    # --- InlineTextBox: always skip ---
    if role_lower == "inlinetextbox":
        return

    # --- WebArea: transparent, recurse ---
    if role_lower == "webarea":
        if name:
            lines.append(f'Page: "{name}"')
            lines.append("")
        for child in children:
            _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)
        return

    # --- Footer: skip entirely ---
    if _is_footer_node(node):
        return

    # --- Chrome: compress to one-line summary ---
    if role_lower in _CHROME_ROLES:
        summary = _summarize_chrome(node)
        if summary:
            lines.append(f"{'  ' * depth}[{role}: {summary}]")
        return

    # --- Noise roles: transparent ---
    if role_lower in _NOISE_ROLES:
        for child in children:
            _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)
        return

    # --- StaticText: dedup and emit ---
    if role_lower == "statictext":
        if name and len(name.strip()) > 1 and not dedup.is_duplicate(name):
            compressed = compress_text(name.strip(), 120)
            lines.append(f"{'  ' * depth}\"{compressed}\"")
        return

    # --- Tables: reuse markdown renderer ---
    if role_lower == "table":
        _render_table_markdown(node, depth, lines)
        return

    # --- Lists with potential card content ---
    if role_lower == "list":
        _reflector_walk_list(node, depth, lines, dedup, max_depth=max_depth)
        return

    # --- Generic/unnamed wrappers: transparent ---
    if role_lower in _GENERIC_ROLES and not name:
        for child in children:
            _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)
        return

    # --- Card containers (article, section with children) ---
    # Handled when encountered as siblings inside _reflector_walk_siblings
    # Single articles are transparent
    if role_lower in _CARD_CONTAINER_ROLES and not name:
        for child in children:
            _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)
        return

    # --- Structural roles: emit with content ---
    if role_lower in _STRUCTURAL_ROLES:
        indent = "  " * depth
        header = f"{indent}{role}"
        if name and not _CSS_CLASS_PATTERN.match(name.strip()):
            header += f' "{compress_text(name.strip(), 80)}"'
        level = node.get("level")
        if level:
            header += f" — level {level}"
        lines.append(header)
        _reflector_walk_siblings(children, depth + 1, lines, dedup, max_depth=max_depth)
        return

    # --- Interactive controls: brief description (no element ID) ---
    if role_lower in ("button", "link", "textbox", "searchbox", "combobox",
                       "checkbox", "radio", "switch", "tab", "menuitem", "slider"):
        indent = "  " * depth
        friendly = _ROLE_MAP.get(role_lower, role_lower)
        display = name if name and not _CSS_CLASS_PATTERN.match(name.strip()) else ""
        if display:
            line = f'{indent}{friendly} "{compress_text(display, 60)}"'
        else:
            line = f"{indent}{friendly}"
        value = node.get("value")
        if value and value != name:
            line += f' — "{compress_text(str(value), 40)}"'
        lines.append(line)
        return

    # --- Other named nodes: emit text, recurse ---
    if name and name.strip():
        if not dedup.is_duplicate(name) and not _CSS_CLASS_PATTERN.match(name.strip()):
            compressed = compress_text(name.strip(), 120)
            lines.append(f"{'  ' * depth}\"{compressed}\"")
    for child in children:
        _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)


def _reflector_walk_siblings(
    children: list[dict],
    depth: int,
    lines: list[str],
    dedup: _TextDedup,
    *,
    max_depth: int = 25,
) -> None:
    """Walk a list of sibling nodes, detecting and formatting repeating cards."""
    if not children:
        return

    groups = _detect_card_groups(children)

    for group_role, group_children in groups:
        if group_role is not None and len(group_children) >= 2:
            # Repeating card group — format compactly
            _format_card_group(group_children, depth, lines, dedup)
        else:
            for child in group_children:
                _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)


def _reflector_walk_list(
    node: dict,
    depth: int,
    lines: list[str],
    dedup: _TextDedup,
    *,
    max_depth: int = 25,
) -> None:
    """Walk a list node, detecting card patterns among listitems."""
    children = node.get("children", [])
    listitems = [c for c in children if c.get("role") == "listitem"]

    if not listitems:
        for child in children:
            _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)
        return

    # Check if listitems contain article (or similar) children with matching structure
    # Extract the semantic child from each listitem for fingerprinting
    semantic_children: list[dict] = []
    for li in listitems:
        li_children = li.get("children", [])
        # Look for the first semantic container child (article, section, etc.)
        container = None
        for c in li_children:
            if c.get("role", "").lower() in _CARD_CONTAINER_ROLES:
                container = c
                break
        semantic_children.append(container or li)

    # Fingerprint: ordered tuple of immediate child roles
    fingerprints = [_role_fingerprint(sc) for sc in semantic_children]

    # Check if majority share same fingerprint (allows minor variations)
    if fingerprints:
        from collections import Counter
        fp_counts = Counter(fingerprints)
        dominant_fp, dominant_count = fp_counts.most_common(1)[0]
        if dominant_count >= 2 and dominant_count >= len(fingerprints) * 0.5:
            # Confirmed repeating cards
            _format_card_group(semantic_children, depth, lines, dedup)
            return

    # Not cards — simple or mixed list, walk individually
    if _is_simple_list(node):
        _render_list_markdown(node, depth, lines)
    else:
        for child in children:
            _reflector_walk(child, depth, lines, dedup, max_depth=max_depth)


def _detect_card_groups(
    children: list[dict],
) -> list[tuple[str | None, list[dict]]]:
    """Detect repeating card patterns among sibling nodes using role fingerprinting.

    Groups consecutive siblings that share the same role AND internal structure.
    Two nodes match if they have the same role and the same fingerprint
    (ordered tuple of their immediate children's roles).
    """
    if not children:
        return []

    groups: list[tuple[str | None, list[dict]]] = []

    def _sig(node: dict) -> tuple[str, tuple[str, ...]]:
        role = node.get("role", "").lower()
        fp = _role_fingerprint(node)
        return (role, fp)

    current_sig = _sig(children[0])
    current_group: list[dict] = [children[0]]

    for child in children[1:]:
        child_sig = _sig(child)
        if child_sig == current_sig and current_sig[0]:
            current_group.append(child)
        else:
            # Only mark as card group if role is a semantic container
            is_card_group = (
                len(current_group) >= 2
                and current_sig[0] in {r.lower() for r in _CARD_CONTAINER_ROLES}
                | {"listitem", "group"}
            )
            groups.append(
                (current_sig[0] if is_card_group else None, current_group)
            )
            current_sig = child_sig
            current_group = [child]

    is_card_group = (
        len(current_group) >= 2
        and current_sig[0] in {r.lower() for r in _CARD_CONTAINER_ROLES}
        | {"listitem", "group"}
    )
    groups.append(
        (current_sig[0] if is_card_group else None, current_group)
    )
    return groups


def _role_fingerprint(node: dict) -> tuple[str, ...]:
    """Compute the ordered role fingerprint of a node's immediate children.

    Two sibling nodes with the same fingerprint have structurally identical
    children layouts — strong signal they're repeating cards.
    """
    children = node.get("children", [])
    if not children:
        return ()
    return tuple(c.get("role", "").lower() for c in children)


def _format_card_group(
    cards: list[dict],
    depth: int,
    lines: list[str],
    dedup: _TextDedup,
) -> None:
    """Format a group of repeating card nodes as compact one-line summaries.

    Each card becomes: "• text1 | text2 | text3 ..."
    Extracts all meaningful text leaves from the card subtree.
    """
    indent = "  " * depth
    lines.append(f"{indent}[{len(cards)} items]")

    for card in cards:
        texts = _extract_card_texts(card)
        if texts:
            # Deduplicate adjacent identical texts within the card
            deduped: list[str] = []
            for t in texts:
                if not deduped or t != deduped[-1]:
                    deduped.append(t)
            card_line = " | ".join(deduped)
            # Truncate very long card lines
            if len(card_line) > 200:
                card_line = card_line[:197] + "..."
            lines.append(f"{indent}  • {card_line}")


def _extract_card_texts(node: dict, max_depth: int = 6) -> list[str]:
    """Extract all meaningful text content from a card subtree.

    Walks depth-first collecting name text from StaticText, headings,
    links, buttons, and other text-bearing nodes. Skips image/decorative nodes.
    """
    if max_depth <= 0:
        return []

    texts: list[str] = []
    role = (node.get("role") or "").lower()
    name = (node.get("name") or "").strip()

    # Skip decorative/image-only nodes
    if role in ("img", "image", "svg", "svgroot", "presentation",
                "inlinetextbox", "linebreak"):
        return []

    # Collect text from text-bearing nodes
    if role in ("statictext", "heading") and name:
        texts.append(compress_text(name, 60))
    elif role in ("link", "button") and name and len(name) > 1:
        texts.append(compress_text(name, 60))
    elif name and role not in _GENERIC_ROLES and role not in _NOISE_ROLES and len(name) > 1:
        # Named non-generic nodes (e.g., combobox with value)
        texts.append(compress_text(name, 60))

    # Recurse into children
    for child in node.get("children", []):
        texts.extend(_extract_card_texts(child, max_depth - 1))

    return texts


def _summarize_chrome(node: dict) -> str:
    """Produce a one-line summary of a chrome section (nav, banner, etc.).

    Extracts just the top-level link/button names to give the reflector
    a sense of what section this is, without wasting budget on structure.
    """
    items: list[str] = []
    _collect_chrome_labels(node, items, max_items=6)
    if items:
        return ", ".join(items)
    name = (node.get("name") or "").strip()
    return name[:80] if name else ""


def _collect_chrome_labels(node: dict, items: list[str], max_items: int = 6) -> None:
    """Collect link/button labels from a chrome subtree (breadth-first-ish)."""
    if len(items) >= max_items:
        return
    role = (node.get("role") or "").lower()
    name = (node.get("name") or "").strip()

    if role in ("link", "button", "tab", "menuitem") and name and len(name) > 1:
        label = compress_text(name, 30)
        if label not in items:
            items.append(label)
        return  # Don't recurse into interactive elements

    for child in node.get("children", []):
        if len(items) >= max_items:
            return
        _collect_chrome_labels(child, items, max_items)


# ===================================================================
# MCP Parameter Context — recipe-based extraction for parameter generation
# ===================================================================

async def build_mcp_parameter_context(
    *,
    recipe: list[dict],
    tool_name: str,
    tool_description: str,
    tool_method: str,
    tool_url_template: str,
    tool_protocol: str,
    tool_examples: list[dict],
    subtask: str,
    session: Any,
    mcp_response_cache: dict[str, Any],
) -> str:
    """Build structured parameter context from a tool's extraction recipe.

    Each ExtractionStep is executed deterministically against the browser/session
    state. Results are formatted for the parameter generation LLM.

    Produces output like:
        TASK: Set delivery location to Indiranagar
        TOOL: set_location (POST /dapi/misc/address-recommend)

        PARAMETERS:
          place_id [chained]:
            From POST /dapi/misc/place-autocomplete response at $.data[0].place_id
            Current value: "ChIJxx..."
          _csrf [ephemeral]:
            Cookie '_csrf'. Changes every session.
            Current value: "xK9m..."
    """
    lines: list[str] = []

    # Header
    lines.append(f"TASK: {subtask}")
    lines.append(f"TOOL: {tool_name} ({tool_method} {tool_url_template})")
    lines.append(f"PROTOCOL: {tool_protocol}")
    lines.append("")

    # Last successful request template
    if tool_examples:
        latest = tool_examples[-1]
        body = latest.get("request_body") if isinstance(latest, dict) else None
        if body:
            lines.append("LAST SUCCESSFUL REQUEST:")
            for k, v in (body if isinstance(body, dict) else {}).items():
                val_str = str(v)[:60]
                lines.append(f'  "{k}": "{val_str}"')
            lines.append(
                f"→ Response: {latest.get('response_status', '?')}"
            )
            lines.append("")

    # Extract current values for each recipe step
    lines.append("PARAMETERS:")
    lines.append("")

    for step in recipe:
        param_path = step.get("param_path", "?")
        param_name = param_path.split(".")[-1]
        classification = step.get("classification", "unknown")
        description = step.get("description", "")
        source_type = step.get("source_type", "")
        config = step.get("source_config", {})

        lines.append(f"  {param_name} [{classification}]:")
        lines.append(f"    {description}")

        # Execute extraction
        extracted = await _execute_extraction_step(
            source_type, config, session, mcp_response_cache,
        )

        if isinstance(extracted, list):
            # dom_list: show options
            lines.append(f"    Options on page ({len(extracted)} found):")
            for item in extracted[:10]:
                val = item.get("value", "")
                label = item.get("label", "")
                lines.append(f'      value="{val}" — {label}')
            if len(extracted) > 10:
                lines.append(f"      ... and {len(extracted) - 10} more")
        elif extracted is not None:
            lines.append(f'    Current value: "{str(extracted)[:80]}"')
        else:
            lines.append("    Current value: NOT FOUND")

        lines.append("")

    # Previous examples
    if tool_examples:
        lines.append("PREVIOUS EXAMPLES:")
        for i, ex in enumerate(tool_examples, 1):
            if not isinstance(ex, dict):
                continue
            body = ex.get("request_body")
            if body:
                import json
                lines.append(f"Example {i}:")
                lines.append(f"  Request: {json.dumps(body, default=str)[:400]}")
                status = ex.get("response_status", "?")
                excerpt = ex.get("response_body_excerpt", "")[:200]
                lines.append(f"  Response: {status} — {excerpt}")
        lines.append("")

    # Current page context
    if session.page:
        lines.append(f"CURRENT URL: {session.page.url}")
        try:
            title = await session.page.title()
            lines.append(f"PAGE: {title}")
        except Exception:
            pass

    return "\n".join(lines)


async def _execute_extraction_step(
    source_type: str,
    config: dict,
    session: Any,
    mcp_response_cache: dict[str, Any],
) -> str | list[dict] | None:
    """Execute a single extraction step against current browser/session state.

    Returns:
    - str: single extracted value
    - list[dict]: for dom_list (options with value + label)
    - None: extraction failed or source not found
    """
    try:
        match source_type:
            case "cookie":
                key = config.get("key", "")
                cookies = await session._context.cookies()
                for c in cookies:
                    if c["name"] == key:
                        return c["value"]

            case "dom_field":
                selector = config.get("selector", "").replace("'", "\\'")
                return await session.page.evaluate(
                    f"() => {{ const el = document.querySelector('{selector}'); "
                    f"return el ? (el.value || el.textContent || '') : null; }}"
                )

            case "dom_list":
                selector = config.get("selector", "")
                value_attr = config.get("value_attr", "value")
                label_attr = config.get("label_attr", "textContent")
                safe_sel = selector.replace("'", "\\'")
                safe_va = value_attr.replace("'", "\\'")
                safe_la = label_attr.replace("'", "\\'")
                items = await session.page.evaluate(f"""() => {{
                    const results = [];
                    for (const el of document.querySelectorAll('{safe_sel}')) {{
                        const val = '{safe_va}' === 'textContent'
                            ? (el.textContent || '').trim()
                            : (el.getAttribute('{safe_va}') || el.value || '');
                        const label = '{safe_la}' === 'textContent'
                            ? (el.textContent || '').trim()
                            : (el.getAttribute('{safe_la}') || '');
                        if (val) results.push({{value: val, label: label.substring(0, 80)}});
                    }}
                    return results;
                }}""")
                return items if items else None

            case "storage":
                storage_type = config.get("storage_type", "localStorage")
                key = config.get("key", "").replace("'", "\\'")
                return await session.page.evaluate(
                    f"() => {storage_type}.getItem('{key}')"
                )

            case "meta_tag":
                selector = config.get("selector", "").replace("'", "\\'")
                return await session.page.evaluate(
                    f"() => {{ const el = document.querySelector('{selector}'); "
                    f"return el ? el.getAttribute('content') : null; }}"
                )

            case "url_component":
                component = config.get("component", "")
                if component == "path_segment":
                    from urllib.parse import urlparse
                    index = config.get("index", 0)
                    segments = (
                        urlparse(session.page.url).path.strip("/").split("/")
                    )
                    return segments[index] if index < len(segments) else None
                elif component == "query_param":
                    from urllib.parse import urlparse, parse_qs
                    key = config.get("key", "")
                    params = parse_qs(urlparse(session.page.url).query)
                    vals = params.get(key, [])
                    return vals[0] if vals else None

            case "prior_api_response":
                endpoint_id = config.get("endpoint_identity", "")
                json_path = config.get("json_path", "")

                # Check MCP response cache first (MCP execution path)
                if endpoint_id in mcp_response_cache:
                    cached = mcp_response_cache[endpoint_id]
                    val = _navigate_json_path(cached, json_path)
                    if val is not None:
                        return str(val)[:200]

                # Fall back to browser captured traffic (CU execution path)
                for req in reversed(session.get_captured_traffic()):
                    if _match_endpoint(
                        req.endpoint_identity or "", endpoint_id,
                    ):
                        if req.response_body_parsed:
                            val = _navigate_json_path(
                                req.response_body_parsed, json_path,
                            )
                            if val is not None:
                                return str(val)[:200]

            case "task_description":
                return None  # LLM derives from task context

    except Exception:
        pass
    return None


def _navigate_json_path(obj: Any, path: str) -> Any | None:
    """Pure structural JSON path navigation.

    Supports: $.key, $.key1.key2, $.arr[0], $.arr[0].field
    No string matching. Walks dict keys and list indices.
    """
    if not path or not obj:
        return None

    # Strip leading "$."
    parts = path.lstrip("$").lstrip(".").split(".")
    current = obj

    for part in parts:
        if not part:
            continue
        # Handle array index: "data[0]" → key="data", index=0
        import re
        arr_match = re.match(r"^(.+?)\[(\d+)\]$", part)
        if arr_match:
            key = arr_match.group(1)
            idx = int(arr_match.group(2))
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None
        elif part.startswith("[") and part.endswith("]"):
            # Bare index like "[0]"
            idx = int(part[1:-1])
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

    return current


def _match_endpoint(concrete: str, template: str) -> bool:
    """Match a concrete endpoint identity against a template.

    Exact match for GraphQL/JSON-RPC operation names.
    Segment-by-segment match for REST where {param} matches anything.
    "POST /restaurants/rest67890/cart" matches "POST /restaurants/{id}/cart"
    """
    if not concrete or not template:
        return False
    if concrete == template:
        return True

    # Split on space: "POST /path" → method + path
    c_parts = concrete.split(" ", 1)
    t_parts = template.split(" ", 1)

    if len(c_parts) != 2 or len(t_parts) != 2:
        # Not REST-style (GraphQL/JSON-RPC) — exact match only
        return concrete == template

    if c_parts[0] != t_parts[0]:
        return False  # Method mismatch

    c_segs = c_parts[1].strip("/").split("/")
    t_segs = t_parts[1].strip("/").split("/")

    if len(c_segs) != len(t_segs):
        return False

    return all(
        ts.startswith("{") and ts.endswith("}") or cs == ts
        for cs, ts in zip(c_segs, t_segs)
    )


# ===================================================================
# Debug helpers (used by CU agent's _save_action_step)
# ===================================================================

def summarize_raw_axtree(node: dict, depth: int = 0) -> dict:
    """Compact summary of raw AXTree for debugging: node counts by role, max depth."""
    from collections import Counter

    role_counts: Counter = Counter()
    max_d = [0]

    def _walk(n: dict, d: int) -> None:
        role_counts[n.get("role", "unknown")] += 1
        if d > max_d[0]:
            max_d[0] = d
        for child in n.get("children", []):
            _walk(child, d + 1)

    _walk(node, 0)
    total = sum(role_counts.values())
    top_roles = dict(role_counts.most_common(15))
    return {
        "total_nodes": total,
        "max_depth": max_d[0],
        "top_roles": top_roles,
        "inlinetextbox_count": role_counts.get("InlineTextBox", 0),
        "statictext_count": role_counts.get("StaticText", 0),
    }


def render_raw_axtree_text(node: dict, max_lines: int = 150) -> str:
    """Render raw AXTree as indented text (for comparison with processed version)."""
    lines: list[str] = []

    def _walk(n: dict, depth: int) -> None:
        if len(lines) >= max_lines:
            return
        role = n.get("role", "")
        name = n.get("name", "")
        indent = "  " * depth
        line = f"{indent}{role}"
        if name:
            line += f' "{name[:60]}"'
        lines.append(line)
        for child in n.get("children", []):
            _walk(child, depth + 1)

    _walk(node, 0)
    if len(lines) >= max_lines:
        lines.append(f"  ... (truncated at {max_lines} lines)")
    return "\n".join(lines)
