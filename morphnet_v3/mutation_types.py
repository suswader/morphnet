"""
morphnet_v3/mutation_types.py — Mutation observer record types + helpers.

Extracted from session_manager.py to break a runtime import cycle:

  session_manager → planner → page_agent → schemas → session_manager
                                                       (for RawMutationRecord)

Schemas needed `RawMutationRecord` + `MutationNodeRef` at module-load time
to re-export them; page_agent needed `summarize_mutations` for its batch
loop. With both definitions sitting in `session_manager.py`, the cycle
closed back through `schemas.py` even after the lazy-import workarounds.

This module owns the types + the two helpers. It has zero morphnet_v3
imports — pure stdlib + Pydantic — so anything in the codebase can import
from here without creating dependency cycles.

Public surface:
- MutationNodeRef          — identity/snapshot of a mutated node
- RawMutationRecord        — one mutation event from the observer
- records_from_raw(...)    — JS dict list → typed records (sm.flush_mutations uses)
- summarize_mutations(...) — compact human-readable summary (page_agent stdout logs)
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict


# ─────────────────────────────────────────────────────────────────
# Mutation observer schemas
# ─────────────────────────────────────────────────────────────────


class MutationNodeRef(BaseModel):
    """Identity + snapshot of a mutated node.

    obs_id is the primary key for grouping. For unstamped elements, it is
    observer-minted (obs-1, obs-2, ...). For stamped elements, obs_id IS the
    aid string (e.g. "aid-55") — ensure_obs_id() returns the aid value directly.
    The aid field is redundant for stamped elements but kept for explicit
    type-checking (aid is not None => stamped, reconcilable).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    obs_id: str
    aid: str | None = None
    parent_obs_id: str | None = None
    parent_aid: str | None = None

    tag: str
    role: str | None = None
    control_kind: str | None = None

    text_preview: str | None = None

    top_layer_hint: bool | None = None
    pointer_blocking_hint: bool | None = None
    interactive_hint: bool | None = None

    z_index: int | None = None


class RawMutationRecord(BaseModel):
    """One mutation event from the observer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    seq: int
    batch_id: str
    step_index: int | None = None
    ts_ms: int

    op: Literal[
        "node_added",
        "node_removed",
        "attr_changed",
        "text_changed",
    ]

    subject: MutationNodeRef

    # For attr_changed
    attr_field: str | None = None
    attr_before: str | None = None
    attr_after: str | None = None

    # For text_changed
    text_before: str | None = None
    text_after: str | None = None

    subtree_size_hint: int | None = None
    removed_ts_ms: int | None = None


# ─────────────────────────────────────────────────────────────────
# Pure-Python helpers
# ─────────────────────────────────────────────────────────────────


def records_from_raw(
    raw_dicts: list[dict],
    batch_id: str | None = None,
) -> list[RawMutationRecord]:
    """Convert raw JS-observer dicts into typed Pydantic RawMutationRecord
    list. Used by `sm.flush_mutations`. Generates a `batch_id` UUID if not
    provided. Skips malformed records silently (same behavior as crawler).
    """
    if not raw_dicts:
        return []
    bid = batch_id or str(uuid4())
    records: list[RawMutationRecord] = []
    for d in raw_dicts:
        subject_data = d.get("subject")
        if not subject_data:
            continue
        try:
            subject = MutationNodeRef(**subject_data)
            rec = RawMutationRecord(
                seq=d.get("seq", 0),
                batch_id=bid,
                step_index=d.get("step_index"),
                ts_ms=d.get("ts_ms", 0),
                op=d.get("op", "node_added"),
                subject=subject,
                attr_field=d.get("attr_field"),
                attr_before=d.get("attr_before"),
                attr_after=d.get("attr_after"),
                text_before=d.get("text_before"),
                text_after=d.get("text_after"),
                subtree_size_hint=d.get("subtree_size_hint"),
                removed_ts_ms=d.get("removed_ts_ms"),
            )
            records.append(rec)
        except (ValueError, TypeError, KeyError):
            continue
    return records


def summarize_mutations(mutations: list[RawMutationRecord]) -> str:
    """Compact summary for stdout/notes logging. Returns empty string when
    there's nothing to report. Used by page_agent's batch loop after every
    mutation flush."""
    if not mutations:
        return ""

    new_seen: set[str] = set()
    new_items: list[str] = []
    changed_items: list[str] = []
    removed_items: list[str] = []
    text_changed_items: list[str] = []

    for m in mutations:
        s = m.subject
        desc = f"<{s.tag}>" + (f' "{s.text_preview}"' if s.text_preview else "")
        if m.op == "node_added":
            label = f"{s.aid or s.obs_id} {desc}"
            if label not in new_seen:
                new_seen.add(label)
                new_items.append(label)
        elif m.op == "attr_changed":
            aid = s.aid or s.obs_id
            changed_items.append(
                f"{aid} {m.attr_field}: {m.attr_before!r} -> {m.attr_after!r}  {desc}"
            )
        elif m.op == "node_removed":
            aid = s.aid or s.obs_id
            removed_items.append(f"{aid}  {desc}")
        elif m.op == "text_changed":
            aid = s.aid or s.obs_id
            text_changed_items.append(f"{aid} text: {m.text_before!r} -> {m.text_after!r}  {desc}")

    lines: list[str] = []
    if new_items:
        lines.append(f"NEW ({len(new_items)}):")
        for item in new_items[:15]:
            lines.append(f"  + {item}")
        if len(new_items) > 15:
            lines.append(f"  ... and {len(new_items) - 15} more")
    if removed_items:
        lines.append(f"REMOVED ({len(removed_items)}):")
        for item in removed_items[:10]:
            lines.append(f"  - {item}")
        if len(removed_items) > 10:
            lines.append(f"  ... and {len(removed_items) - 10} more")
    if changed_items:
        lines.append(f"CHANGED ({len(changed_items)}):")
        for item in changed_items[:10]:
            lines.append(f"  ~ {item}")
        if len(changed_items) > 10:
            lines.append(f"  ... and {len(changed_items) - 10} more")
    if text_changed_items:
        lines.append(f"TEXT_CHANGED ({len(text_changed_items)}):")
        for item in text_changed_items[:10]:
            lines.append(f"  T {item}")
        if len(text_changed_items) > 10:
            lines.append(f"  ... and {len(text_changed_items) - 10} more")

    return "\n".join(lines)