"""
morphnet_v2/computer_use/raw_session.py

The LLM tool loop for one CU step. Faithfully lifted from
`browser-challenge/crawler/master.py:_run_raw_session` + `_update_v5`, with
three v2 adaptations:

  1. **Gemini-native function calling** instead of crawler's LiteLLM-normalized
     tool_calls. The conversation thread uses `google.genai.types.Content` /
     `Part` / `FunctionCall` / `FunctionResponse` objects directly.

  2. **Pure dependency injection.** No `MasterConfig` or `MasterOrchestrator`.
     The runner takes `sm` (for `call_gemini` only) and a `batch_executor`
     callback. Action dispatch + post-batch re-extract live in chunk 2.4.

  3. **Deterministic success rule** (architecture rule 11): the runner
     synthesizes `StepResult.success` from `exit_reason` + `last_batch_clean`.
     No LLM-based verdict.

The V5-deletion fix is preserved: the initial V5 at message index 0 is never
deleted. Without this, Gemini rejects the thread because the very first
`function_call` has no `user` turn before it. Crawler shipped the same fix at
`master.py:1621`.

Spiral detection is lifted verbatim with regex on `BatchResult.text` per
draft.md carve-out #15. The planning tree's `detect_repeated_approaches`
(Phase 3) is the cross-page backup for loop detection.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.genai import types as genai_types
from jinja2 import Template

from morphnet_v2 import notes
from morphnet_v2.computer_use.schemas import BatchResult, SessionExit, StepResult

if TYPE_CHECKING:
    from morphnet_v2.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"
_PAGE_AGENT_TEMPLATE_PATH: Path = _PROMPTS_DIR / "page_agent.j2"

# Spiral threshold lifted from crawler/master.py:2417.
_SPIRAL_THRESHOLD: int = 3

# Regex lifted verbatim from crawler/master.py:2581-2613 (carve-out #15).
# Matches action-result lines emitted by chunk 1.5 action methods, e.g.
#   "click failed aid=aid-12: blocked_by aid-5"
# We don't introduce new regex anywhere else in this module — _parse_response
# uses typed attribute checks on Gemini Part objects (no string matching).
_FAIL_LINE_RE: re.Pattern[str] = re.compile(
    r"(?:type failed|action failed|click failed) aid=(aid-\d+)"
)
_SUCCESS_AID_RE: re.Pattern[str] = re.compile(r"aid=(aid-\d+)")


# ---------------------------------------------------------------------------
# Prompt + tools schema builders (module-level so callers can introspect)
# ---------------------------------------------------------------------------


def render_page_agent_prompt(user_goal: str, dnd_library: str | None) -> str:
    """Render `morphnet_v2/prompts/page_agent.j2` with the two template
    variables. Returns the full system-instruction text.

    `user_goal` is the high-level task (NOT a sliced subtask — morphnet_v2
    does not decompose). From Phase 3 onward, chunk 2.4 prepends the
    planning-tree rendering into `user_goal` before passing it here so CU
    sees prior steps' summaries.

    `dnd_library` gates the `probe_drop_zones` action in the prompt's
    `## Tools` section via the template's `{% if dnd_library == 'html5-native' %}`
    block. `None` is the default (no drag-and-drop library detected).
    """
    template_text = _PAGE_AGENT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return Template(template_text).render(user_goal=user_goal, dnd_library=dnd_library)


def build_tools_schema(dnd_library: str | None) -> list[genai_types.Tool]:
    """Build Gemini function declarations for `execute_actions` + `report`.

    The action-object schema is a single OBJECT with all possible properties
    flattened; only `action` is required and its enum lists every valid action
    name. Gemini fills the fields appropriate to the chosen action. This is
    the same shape crawler's LiteLLM call uses (just expressed in Gemini's
    Schema type instead of OpenAI-style JSON Schema).
    """
    action_names = [
        "click", "type_text", "scroll", "scroll_page", "copy_paste",
        "key_press", "drag", "draw", "hover", "wait_for_page_settle",
        "sleep", "re_extract",
    ]
    if dnd_library == "html5-native":
        action_names.append("probe_drop_zones")

    T = genai_types.Type
    action_schema = genai_types.Schema(
        type=T.OBJECT,
        properties={
            "action":         genai_types.Schema(type=T.STRING, enum=action_names),
            "target_id":      genai_types.Schema(type=T.INTEGER),
            "source_id":      genai_types.Schema(type=T.INTEGER),
            "source_ids":     genai_types.Schema(type=T.ARRAY, items=genai_types.Schema(type=T.INTEGER)),
            "container_ids":  genai_types.Schema(type=T.ARRAY, items=genai_types.Schema(type=T.INTEGER)),
            "text":           genai_types.Schema(type=T.STRING),
            "pixels":         genai_types.Schema(type=T.INTEGER),
            "intent":         genai_types.Schema(type=T.STRING, enum=["navigate", "dismiss"]),
            "drag_mode":      genai_types.Schema(type=T.STRING, enum=["target", "slider", "offset"]),
            "percent":        genai_types.Schema(type=T.NUMBER),
            "offset_x":       genai_types.Schema(type=T.INTEGER),
            "offset_y":       genai_types.Schema(type=T.INTEGER),
            "keys":           genai_types.Schema(type=T.ARRAY, items=genai_types.Schema(type=T.STRING)),
            "strokes": genai_types.Schema(
                type=T.ARRAY,
                items=genai_types.Schema(
                    type=T.ARRAY,
                    items=genai_types.Schema(
                        type=T.ARRAY,
                        items=genai_types.Schema(type=T.NUMBER),
                    ),
                ),
            ),
            "duration":       genai_types.Schema(type=T.INTEGER),
            "max_ms":         genai_types.Schema(type=T.INTEGER),
            "delay_ms":       genai_types.Schema(type=T.INTEGER),
        },
        required=["action"],
    )

    execute_actions_decl = genai_types.FunctionDeclaration(
        name="execute_actions",
        description=(
            "Execute a batch of browser actions in order. Stops on first failure "
            "or navigation. Use this for ALL browser interactions."
        ),
        parameters=genai_types.Schema(
            type=T.OBJECT,
            properties={"actions": genai_types.Schema(type=T.ARRAY, items=action_schema)},
            required=["actions"],
        ),
    )
    report_decl = genai_types.FunctionDeclaration(
        name="report",
        description=(
            "Signal step completion with the final answer/deliverable message. "
            "Call this when the task is fully done."
        ),
        parameters=genai_types.Schema(
            type=T.OBJECT,
            properties={"message": genai_types.Schema(type=T.STRING)},
            required=["message"],
        ),
    )
    return [genai_types.Tool(function_declarations=[execute_actions_decl, report_decl])]


# ---------------------------------------------------------------------------
# Parsed-response container
# ---------------------------------------------------------------------------


@dataclass
class _ParsedResponse:
    """Typed extraction from one Gemini turn. No string matching — every field
    comes from `Part` attribute access (`.function_call`, `.thought`, `.text`)
    or `usage_metadata`."""

    function_calls: list[tuple[str, dict[str, Any]]]
    thinking_text: str
    text_content: str
    parts_to_append: list[genai_types.Part]
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# RawSessionRunner
# ---------------------------------------------------------------------------


class RawSessionRunner:
    """Drive the multi-turn LLM tool loop for ONE CU step.

    A step = one page-worth of CU activity. Loop exits on NAVIGATED, REPORT,
    MAX_TURNS, SPIRAL, or NO_TOOL_CALL. Returns a `StepResult` the planner
    consumes for routing decisions (Phase 3).

    The class is pure-LLM: it does not touch Chrome, does not call PageFilter,
    does not render V5 markdown. All page-side work happens behind the
    `batch_executor` callback that chunk 2.4 wires up.

    Single use: instantiate, call `run()` once, discard. State lives in `run()`
    locals — re-entrant per-call.
    """

    def __init__(
        self,
        *,
        sm: SessionManager,
        model: str,
        max_turns: int,
        batch_executor: Callable[[list[dict[str, Any]]], Awaitable[BatchResult]],
        dnd_library: str | None = None,
        max_output_tokens: int = 8192,
        thinking_budget: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self._sm = sm
        self._model = model
        self._max_turns = max_turns
        self._batch_executor = batch_executor
        self._dnd_library = dnd_library
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget
        self._temperature = temperature
        # Pre-render: tools schema doesn't change mid-step.
        self._tools_schema = build_tools_schema(dnd_library)

    async def run(self, user_goal: str, initial_v5: str) -> StepResult:
        """Drive the loop for one step. Returns when the step terminates."""
        # Per-step state (each call gets fresh state).
        messages: list[genai_types.Content] = [
            genai_types.Content(
                role="user", parts=[genai_types.Part.from_text(text=initial_v5)],
            ),
        ]
        # Initial V5 sits at index 0 — preserved forever (V5-deletion fix).
        v5_message_index = 0
        pending_v5: str | None = None
        latest_v5: str = initial_v5
        spiral_counts: dict[tuple[str, str], int] = {}
        action_log: list[str] = []
        total_actions_attempted = 0
        total_actions_failed = 0
        last_batch_clean = True  # vacuously true if no batches ever run
        report_message: str | None = None
        exit_reason: SessionExit | None = None
        total_input_tokens = 0
        total_output_tokens = 0
        turn_count = 0

        system_instruction = render_page_agent_prompt(user_goal, self._dnd_library)

        for _turn in range(self._max_turns):
            turn_count += 1
            resp = await self._sm.call_gemini(
                model=self._model,
                contents=messages,
                tools=self._tools_schema,
                system_instruction=system_instruction,
                max_output_tokens=self._max_output_tokens,
                thinking_budget=self._thinking_budget,
                temperature=self._temperature,
            )
            parsed = self._parse_response(resp)
            total_input_tokens += parsed.input_tokens
            total_output_tokens += parsed.output_tokens

            # Mirror reasoning to notes — function-calling has no `reasoning`
            # field in the schema, so thinking traces are our only window.
            if parsed.thinking_text:
                notes.log(data_type="thinking", data=parsed.thinking_text, turn=turn_count)

            # Append the assistant turn using the raw Part list (preserves
            # Gemini's thought + function_call ordering in the thread).
            messages.append(genai_types.Content(role="model", parts=parsed.parts_to_append))

            # No function_call: one-shot text nudge if model emitted text; else exit.
            if not parsed.function_calls:
                if parsed.text_content.strip():
                    messages.append(_user_text(
                        "You output text instead of a tool call. NEVER output text. "
                        "Use the execute_actions tool. Try again with a proper tool call."
                    ))
                    continue
                exit_reason = SessionExit.NO_TOOL_CALL
                break

            # Process each function_call in order.
            should_exit = False
            spiral_this_turn = False
            for fn_name, fn_args in parsed.function_calls:
                if fn_name == "report":
                    report_message = str(fn_args.get("message", ""))
                    exit_reason = SessionExit.REPORT
                    should_exit = True
                    break
                if fn_name != "execute_actions":
                    messages.append(_function_response(
                        fn_name, f"ERROR: unknown tool '{fn_name}'",
                    ))
                    continue

                actions = [dict(a) for a in (fn_args.get("actions") or [])]
                batch = await self._batch_executor(actions)
                messages.append(_function_response("execute_actions", batch.text))

                total_actions_attempted += batch.n_actions_executed
                total_actions_failed += batch.n_actions_failed
                last_batch_clean = batch.last_batch_clean
                action_log.extend(_extract_action_lines(batch.text))

                spiral_this_turn |= self._update_spiral_counts(batch.text, spiral_counts)

                if batch.new_v5 is not None:
                    pending_v5 = batch.new_v5
                    latest_v5 = batch.new_v5

                if batch.navigated:
                    exit_reason = SessionExit.NAVIGATED
                    should_exit = True
                    break

            # Apply pending V5 AFTER processing this turn's tool calls.
            if pending_v5 is not None:
                v5_message_index = self._rotate_v5(messages, v5_message_index, pending_v5)
                pending_v5 = None

            if spiral_this_turn:
                messages.append(_user_text(
                    "You have attempted the same action on the same element "
                    "multiple times and it keeps failing. Stop. Re-read the "
                    "current page markdown carefully and try a completely "
                    "different approach."
                ))
                spiral_counts.clear()

            if should_exit:
                break

        if exit_reason is None:
            exit_reason = SessionExit.MAX_TURNS

        success = self._synthesize_success(exit_reason, last_batch_clean)
        final_url = self._sm.page.url
        return StepResult(
            success=success,
            exit_reason=exit_reason,
            turn_count=turn_count,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            final_url=final_url,
            report_message=report_message,
            action_log=action_log,
            total_actions_attempted=total_actions_attempted,
            total_actions_failed=total_actions_failed,
            last_v5=latest_v5,
        )

    # ── private helpers ─────────────────────────────────────────────

    def _parse_response(self, resp: Any) -> _ParsedResponse:
        """Typed extraction — every field comes from `Part` attribute access.
        No string matching, no regex, no JSON parsing of model output."""
        function_calls: list[tuple[str, dict[str, Any]]] = []
        thinking_text = ""
        text_content = ""
        parts_to_append: list[genai_types.Part] = []

        candidate = resp.candidates[0] if resp.candidates else None
        if candidate is not None and candidate.content is not None:
            for part in (candidate.content.parts or []):
                parts_to_append.append(part)
                if part.function_call is not None:
                    fn = part.function_call
                    args = dict(fn.args) if fn.args is not None else {}
                    function_calls.append((fn.name or "", args))
                elif part.thought:
                    if part.text:
                        thinking_text += part.text
                elif part.text is not None:
                    text_content += part.text

        usage = resp.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage is not None else 0
        output_tokens = (usage.candidates_token_count or 0) if usage is not None else 0
        return _ParsedResponse(
            function_calls=function_calls,
            thinking_text=thinking_text,
            text_content=text_content,
            parts_to_append=parts_to_append,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _rotate_v5(
        messages: list[genai_types.Content],
        v5_index: int,
        new_v5: str,
    ) -> int:
        """Append new V5 user message; delete the prior NON-INDEX-0 V5 if any.

        Index 0 (initial V5) is preserved so Gemini's strict validator accepts
        the thread: the very first model `function_call` must be preceded by a
        `user` turn, and index 0 IS that turn. Deleting it leaves the first
        function_call with only the system instruction before it → INVALID_ARGUMENT.

        Two transformations applied at rotation time (Issue 6, May-12 fix):
        - The new V5 gets a demarcation header so the model knows where the
          current page state begins and which AIDs to trust.
        - All prior `function_response` messages have their `aid-N` references
          suffixed with `[stale]` (idempotent) so the model doesn't hallucinate
          a click on an AID from a previous page state.

        Lifted from crawler/master.py:_update_v5 (line 1621). Returns the new
        v5_index (always `len(messages) - 1` after the append).
        """
        header = (
            "---\n"
            "The page markdown below is the CURRENT page state. Use ONLY aid-* values "
            "from this section.\n"
            "AIDs marked [previous-turn-aid] in earlier turns are from prior page states "
            "and may no longer exist.\n"
            "---\n\n"
        )
        new_v5_content = genai_types.Content(
            role="user", parts=[genai_types.Part.from_text(text=header + new_v5)],
        )

        # Mark AIDs in all prior function_response messages with [previous-turn-aid].
        # We walk every index strictly before the old V5 (which is itself about to be
        # deleted). Idempotent: the negative lookahead skips any aid-N already suffixed.
        aid_prev_re = re.compile(r"\baid-(\d+)\b(?!\[previous-turn-aid\])")
        for i in range(v5_index):
            msg = messages[i]
            new_parts: list[genai_types.Part] = []
            changed = False
            for part in msg.parts:
                fr = part.function_response
                if fr is None or fr.response is None:
                    new_parts.append(part)
                    continue
                result_text = fr.response.get("result", "") if isinstance(fr.response, dict) else ""
                if not result_text or "aid-" not in result_text:
                    new_parts.append(part)
                    continue
                new_result = aid_prev_re.sub(r"aid-\1[previous-turn-aid]", result_text)
                if new_result == result_text:
                    new_parts.append(part)
                    continue
                new_parts.append(genai_types.Part.from_function_response(
                    name=fr.name,
                    response={"result": new_result},
                ))
                changed = True
            if changed:
                messages[i] = genai_types.Content(role=msg.role, parts=new_parts)

        if v5_index != 0:
            del messages[v5_index]
        messages.append(new_v5_content)
        return len(messages) - 1

    @staticmethod
    def _update_spiral_counts(
        batch_text: str,
        spiral_counts: dict[tuple[str, str], int],
    ) -> bool:
        """Regex-based spiral detection lifted verbatim from crawler/master.py
        lines 2577-2613 (carve-out #15). Returns True if any `(action, aid)`
        pair has accumulated `_SPIRAL_THRESHOLD` failures.
        """
        for line in batch_text.split("\n"):
            fail_match = _FAIL_LINE_RE.search(line)
            if fail_match:
                fail_aid = fail_match.group(1)
                if "type failed" in line:
                    fail_action = "type_text"
                elif "click failed" in line:
                    fail_action = "click"
                else:
                    fail_action = "action"
                key = (fail_action, fail_aid)
                spiral_counts[key] = spiral_counts.get(key, 0) + 1
            elif "clicked" in line or "typed" in line or "copied" in line:
                ok_match = _SUCCESS_AID_RE.search(line)
                if ok_match:
                    ok_aid = ok_match.group(1)
                    for k in list(spiral_counts):
                        if k[1] == ok_aid:
                            del spiral_counts[k]
        return any(v >= _SPIRAL_THRESHOLD for v in spiral_counts.values())

    @staticmethod
    def _synthesize_success(exit_reason: SessionExit, last_batch_clean: bool) -> bool:
        """Deterministic success rule per architecture rule 11.

        NAVIGATED — URL actually changed; the navigating action must have
        succeeded (mechanically verifiable). Trust.

        REPORT — model called the report tool. The report is the model's
        claim; we trust it only when the LAST batch ran cleanly. Catches the
        hallucination case where the model gives up after failed clicks and
        reports "done" anyway.

        Other exit reasons (MAX_TURNS / SPIRAL / NO_TOOL_CALL) are explicit
        failures.
        """
        if exit_reason == SessionExit.NAVIGATED:
            return True
        if exit_reason == SessionExit.REPORT:
            return last_batch_clean
        return False


# ---------------------------------------------------------------------------
# Tiny content constructors — keep call sites readable
# ---------------------------------------------------------------------------


def _user_text(text: str) -> genai_types.Content:
    """One-line user message constructor."""
    return genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=text)])


def _function_response(name: str, result_text: str) -> genai_types.Content:
    """Build the function_response Content for the message thread. Wraps the
    result string in `{"result": ...}` per Gemini's function-response shape."""
    return genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_function_response(
            name=name, response={"result": result_text},
        )],
    )


def _extract_action_lines(batch_text: str) -> list[str]:
    """Pull action-result lines out of `BatchResult.text` for `StepResult.action_log`.

    Filters out the `## During Batch` heading + its bullet lines (those describe
    mutation events, not actions taken). Mutation events stay in the function
    response sent back to the model, but they're not "actions" in the planner's
    sense.
    """
    out: list[str] = []
    for line in batch_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## ") or stripped.startswith("- "):
            continue
        out.append(line)
    return out