"""Episode builder + final-state reconciliation for mutation-to-model v2.

Takes raw mutation records from the observer (Phase 7), groups them into
per-subject episodes, and reconciles against the post-batch extraction.

Three public functions:
  - build_episodes(records) -> list[SubjectEpisode]
  - apply_persistence_results(episodes, disconnected_ids, final_step_index)
  - reconcile_episodes(episodes, known_aids)
"""

from __future__ import annotations

import logging

from .schemas import (
    AttrDelta,
    RawMutationRecord,
    SubjectEpisode,
    TextDelta,
)

log = logging.getLogger(__name__)


def build_episodes(records: list[RawMutationRecord]) -> list[SubjectEpisode]:
    """Group raw mutation records into per-subject episodes.

    Identity grouping uses ``subject.obs_id`` as the primary key (which IS the
    aid string for stamped elements — see Phase 7 ``ensureObsId``).
    """
    episodes: dict[str, SubjectEpisode] = {}

    for rec in records:
        s = rec.subject
        key = s.obs_id

        if key not in episodes:
            episodes[key] = SubjectEpisode(
                subject_obs_id=s.obs_id,
                subject_aid=s.aid,
                tag=s.tag,
                role=s.role,
            )
        ep = episodes[key]

        if rec.op == "node_added":
            ep.appeared_after_step = rec.step_index
            ep.appeared_ts_ms = rec.ts_ms
            if s.text_preview is not None:
                if ep.text_first is None:
                    ep.text_first = s.text_preview
                ep.text_last = s.text_preview
            # Copy hint fields
            if s.top_layer_hint is not None:
                ep.top_layer_hint = s.top_layer_hint
            if s.pointer_blocking_hint is not None:
                ep.pointer_blocking_hint = s.pointer_blocking_hint
            if s.interactive_hint is not None:
                ep.interactive_hint = s.interactive_hint

        elif rec.op == "node_removed":
            ep.disappeared_after_step = rec.step_index
            ep.disappeared_ts_ms = rec.removed_ts_ms or rec.ts_ms

        elif rec.op == "text_changed":
            ep.text_deltas.append(
                TextDelta(
                    before=rec.text_before,
                    after=rec.text_after,
                    step_index=rec.step_index,
                )
            )
            if rec.text_after is not None:
                ep.text_last = rec.text_after

        elif rec.op == "attr_changed":
            ep.attr_deltas.append(
                AttrDelta(
                    field=rec.attr_field or "",
                    before=rec.attr_before,
                    after=rec.attr_after,
                    step_index=rec.step_index,
                )
            )

    return list(episodes.values())


def apply_persistence_results(
    episodes: list[SubjectEpisode],
    disconnected_ids: list[str],
    final_step_index: int,
) -> None:
    """Mark episodes whose roots are no longer in the DOM.

    Only sets ``disappeared_after_step`` on episodes that have no prior
    ``node_removed`` record — avoids overwriting explicit removal timestamps.
    """
    disconnected = set(disconnected_ids)
    for ep in episodes:
        if ep.subject_obs_id in disconnected and ep.disappeared_after_step is None:
            ep.disappeared_after_step = final_step_index


def reconcile_episodes(
    episodes: list[SubjectEpisode],
    known_aids: set[str],
) -> None:
    """Mark stamped episodes against the post-batch extraction.

    Stamped episodes (``subject_aid`` is not None) are checked:
    - If the aid is in ``known_aids`` → ``present_in_final_extraction=True``.
    - If the aid is NOT in ``known_aids`` → ``present_in_final_extraction=False``.

    Unstamped episodes (``subject_aid`` is None) are skipped (always surfaced).
    """
    stamped_count = 0
    unstamped_count = 0

    for ep in episodes:
        if ep.subject_aid is None:
            unstamped_count += 1
            continue

        stamped_count += 1
        aid = ep.subject_aid
        if aid in known_aids:
            ep.present_in_final_extraction = True
        else:
            ep.present_in_final_extraction = False

    if episodes:
        print(
            f"  [EPISODES] reconciled {stamped_count} stamped episodes,"
            f" {unstamped_count} unstamped (always surfaced)"
        )


def _is_suppressed(ep: SubjectEpisode, *, surface_all_deltas: bool = False) -> bool:
    """Suppress if node is in extraction AND episode has no meaningful deltas.

    When ``surface_all_deltas=True``, any episode with deltas surfaces (A→B included).
    When ``surface_all_deltas=False`` (default), single-delta episodes (A→B) are
    suppressed — the agent already has A from the action result and B from the V5.
    Only intermediate states (A→B→C: B is lost without the episode) surface.
    Ephemeral events (lifecycle pair, not in extraction) always surface regardless.
    """
    if ep.present_in_final_extraction is not True:
        return False
    has_lifecycle_pair = (
        ep.appeared_after_step is not None and ep.disappeared_after_step is not None
    )
    if has_lifecycle_pair:
        return False
    n_deltas = len(ep.text_deltas) + len(ep.attr_deltas)
    if surface_all_deltas:
        return n_deltas == 0
    # Default: suppress single-delta episodes (A→B). Surface only intermediates (A→B→C+).
    return n_deltas <= 1


def _episode_step(ep: SubjectEpisode) -> int | None:
    """Determine the step to group this episode under."""
    if ep.appeared_after_step is not None:
        return ep.appeared_after_step
    # For attr/text-only episodes, use first delta's step
    if ep.attr_deltas:
        return ep.attr_deltas[0].step_index
    if ep.text_deltas:
        return ep.text_deltas[0].step_index
    return None


def format_batch_events(
    episodes: list[SubjectEpisode],
    *,
    surface_all_deltas: bool = False,
) -> str:
    """Format surviving episodes as the ``## During Batch`` text block.

    Returns empty string if no episodes survive reconciliation. Groups
    episodes by step index. Episodes with ``step_index = None`` are grouped
    under ``(unattributed)``.

    When ``surface_all_deltas=True``, any episode with deltas surfaces.
    When ``False`` (default), single-delta episodes are suppressed — the agent
    has the before state from action results and the after state from V5.
    """
    surviving = [
        ep for ep in episodes if not _is_suppressed(ep, surface_all_deltas=surface_all_deltas)
    ]
    if not surviving:
        return ""

    # Group by step
    groups: dict[int | None, list[str]] = {}
    for ep in surviving:
        step = _episode_step(ep)
        if step not in groups:
            groups[step] = []
        # Ephemeral episodes (appeared + disappeared) get no ID — content only.
        # Persisted episodes get their aid so the model can act on them.
        is_ephemeral = ep.appeared_after_step is not None and ep.disappeared_after_step is not None
        eid = (
            ""
            if is_ephemeral
            else (f"`{ep.subject_aid}`" if ep.subject_aid else f"`{ep.subject_obs_id}`")
        )

        # Lifecycle events — show if the episode appeared (even with step=None)
        if ep.appeared_after_step is not None or ep.appeared_ts_ms is not None:
            if is_ephemeral:
                parts = [f"appeared tag=`{ep.tag}`"]
            else:
                parts = [f"{eid} appeared tag=`{ep.tag}`"]
            if ep.role:
                parts[0] += f" role=`{ep.role}`"
            if ep.text_first:
                parts[0] += f' text="{ep.text_first[:80]}"'
            if is_ephemeral:
                parts[0] += " (removed)"
            groups[step].append(parts[0])

        # Attr deltas
        for ad in ep.attr_deltas:
            s = ad.step_index
            if s not in groups:
                groups[s] = []
            prefix = eid + " " if eid else ""
            groups[s].append(f'{prefix}attr `{ad.field}`: "{ad.before}" -> "{ad.after}"')

        # Text deltas
        for td in ep.text_deltas:
            s = td.step_index
            if s not in groups:
                groups[s] = []
            prefix = eid + " " if eid else ""
            groups[s].append(f'{prefix}text changed: "{td.before}" -> "{td.after}"')

    if not any(groups.values()):
        return ""

    lines = ["## During Batch"]
    for step in sorted(groups, key=lambda s: (s is None, s or 0)):
        label = f"after step {step}" if step is not None else "(unattributed)"
        lines.append(f"- {label}:")
        for event_line in groups[step]:
            lines.append(f"  - {event_line}")

    return "\n".join(lines)
