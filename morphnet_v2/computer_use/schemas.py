"""
morphnet_v2/computer_use/schemas.py

CU-side schemas. Mirrors crawler/schemas.py for the entities CU touches:
  - PageFilter entities — re-exported from morphnet_v2/page_filter.py so
    v5_markdown.py can import them via `from .schemas import ...` exactly
    like crawler/master_markdown.py does. Single source of truth still
    lives in page_filter.py.
  - Mutation/episode schemas — defined here, byte-for-byte from crawler.

The planner doesn't see mutations; mutation schemas stay CU-local.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Re-exports from morphnet_v2/page_filter.py — keeps the v5_markdown import
# line byte-identical to crawler's `from .schemas import (Button..., ...)`.
# `X as X` is the explicit re-export syntax — Pylint/Pyright/mypy all honor it,
# unlike `__all__` alone (which Pylint's E0611 doesn't follow).
# Re-export observer schemas from mutation_types (extracted out of
# session_manager in Chunk 3.3 to break a runtime import cycle).
from ..mutation_types import (
    MutationNodeRef as MutationNodeRef,
    RawMutationRecord as RawMutationRecord,
)

from ..page_filter import (
    ActionCandidate as ActionCandidate,
    ButtonEntity as ButtonEntity,
    ContainerEntity as ContainerEntity,
    FormBlockerStatus as FormBlockerStatus,
    FormControl as FormControl,
    FormControlGroup as FormControlGroup,
    FormEntity as FormEntity,
    PageFilterOutput as PageFilterOutput,
    PageSnapshot as PageSnapshot,
    TargetOcclusion as TargetOcclusion,
    ViewportGeometry as ViewportGeometry,
)


# ---------------------------------------------------------------------------
# Mutation-to-model v2 schemas (plan 014, Phase 7+)
# ---------------------------------------------------------------------------


class TextDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    before: str | None = None
    after: str | None = None
    step_index: int | None = None


class AttrDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    field: str
    before: str | None = None
    after: str | None = None
    step_index: int | None = None


class SubjectEpisode(BaseModel):
    """Normalized history for one mutated subject/root."""

    model_config = ConfigDict(extra="forbid", strict=True)

    subject_obs_id: str  # always present — obs-N for unstamped, or aid value for stamped
    subject_aid: str | None = None  # set when subject has data-cdx-aid

    tag: str
    role: str | None = None

    appeared_after_step: int | None = None
    disappeared_after_step: int | None = None
    appeared_ts_ms: int | None = None
    disappeared_ts_ms: int | None = None

    text_first: str | None = None
    text_last: str | None = None
    text_deltas: list[TextDelta] = Field(default_factory=list)
    attr_deltas: list[AttrDelta] = Field(default_factory=list)

    # Hint fields: metadata for process_mutations() and _ax_probe_new_nodes().
    # NOT rendered in the ## During Batch formatter output.
    top_layer_hint: bool | None = None
    pointer_blocking_hint: bool | None = None
    interactive_hint: bool | None = None

    present_in_final_extraction: bool | None = None


# ---------------------------------------------------------------------------
# CU loop schemas (chunk 2.3) — produced by raw_session.py
# ---------------------------------------------------------------------------


class BatchResult(BaseModel):
    """What batch_executor (chunk 2.4) returns to the RawSessionRunner per
    execute_actions tool call. Sent back to Gemini as the function_response
    content + drives V5 rotation + spiral detection.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str                       # joined action results + ## During Batch
    navigated: bool                 # if True, loop exits this batch (page changed)
    new_v5: str | None = None       # post-batch V5 markdown (None when navigated)
    n_actions_executed: int         # count of actions that actually ran
    n_actions_failed: int = 0       # count where ActionResult.success was False
    last_batch_clean: bool = True   # True iff every executed action succeeded


class SessionExit(StrEnum):
    """How RawSessionRunner.run() terminated. Drives SubtaskResult.success."""

    NAVIGATED = "navigated"          # batch caused URL change — hard mechanical signal
    REPORT = "report"                # model called the report tool with a deliverable
    MAX_TURNS = "max_turns"          # exhausted turn budget without nav/report
    SPIRAL = "spiral"                # same (action_type, aid) failed >=3 times
    NO_TOOL_CALL = "no_tool_call"    # model emitted text after the one-shot nudge


class StepResult(BaseModel):
    """Outcome of one CU step (one page-worth of CU activity). Consumed by the
    planner (Phase 3); `success` is the deterministic mechanical signal the
    planner uses for `tree_update.outcome` per architecture rule 11.

    Synthesis rule (also lives as `_synthesize_success` in raw_session.py):
        success = (
            exit_reason == NAVIGATED                                    # URL actually changed
            OR (exit_reason == REPORT AND last_batch_clean)             # report after clean batch
        )
    The REPORT-only branch checks `last_batch_clean` to catch the
    LLM-hallucination case where the model calls `report("done!")` after
    failed clicks. NAVIGATED inherently requires a successful navigating
    action so no extra check needed.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    success: bool
    exit_reason: SessionExit
    turn_count: int
    total_input_tokens: int
    total_output_tokens: int
    final_url: str
    report_message: str | None = None
    action_log: list[str] = Field(default_factory=list)
    total_actions_attempted: int = 0
    total_actions_failed: int = 0
    last_v5: str = ""
