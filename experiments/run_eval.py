"""
experiments/run_eval.py

Top-level entrypoint that drives morphnet_v2's full pipeline against a single
task or a task list. It only imports `SessionManager` — the Orchestrator
(and the PageAgent / ToolExecutor it owns internally) is built lazily on
the first `sm.run_task(...)` call. Per `feedback_dependency_direction`,
nothing above SessionManager is constructed here.

This file is the TOP of the dependency tree: imported by nobody, imports
the bottom-of-tree `SessionManager`.

Task list format (JSON array, one object per task):
    [
      {
        "url": "https://www.swiggy.com",
        "task": "search for La Pino'z Pizza in Kothrud",
        "expected_answer": "garlic bread visible in the cart",
        "label": "swiggy_pune_pizza",
        "site": "swiggy"
      }
    ]
Only `url` and `task` are required by this script. `expected_answer` / `label`
/ `site` are passed-through metadata for downstream graders.

Optional per-task fields:
    headless              per-task headless override (else --headless flag)
    max_steps             per-task step budget override (else --max-steps flag)
    max_turns_per_step    per-task CU turn budget override (else --max-turns-per-step)

Usage:
    # Single ad-hoc task:
    uv run python experiments/run_eval.py --url https://x.com --task "..."

    # Task list:
    uv run python experiments/run_eval.py --tasks experiments/comparison_50_tasks.json
    uv run python experiments/run_eval.py --tasks ... --headless true
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Allow `python experiments/run_eval.py` (no `-m`) by adding repo root to path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Top-of-dependency-tree imports — all below this file.
from morphnet_v2.session_manager import SessionManager  # noqa: E402


DEFAULT_OUT = _REPO_ROOT / "experiments" / "results_v2"


def _str_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes", "y")


async def run_task(
    *,
    url: str,
    task: Optional[str] = None,
    headless: bool = False,
    port: int = 9222,
    results_dir: Path | str | None = None,
    max_steps: int = 10,
    max_turns_per_step: int = 60,
    task_metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Drive one task end-to-end. Just instantiates SessionManager and calls
    `sm.run_task(task)`. The Orchestrator (planner) is built internally by
    SessionManager — this function knows nothing about PageAgent/PageFilter/
    Orchestrator. Returns the PlanningTree, or None for smoke-test mode.

    `task_metadata` (which carries `expected_answer` in eval runs) is a
    WRITE-ONLY trace tag — SessionManager persists it to metadata.json on
    disk; no LLM-facing code path ever reads it back.
    """
    print(f"\n=== {url} ===")
    if task:
        print(f"task: {task}")
    md = dict(task_metadata or {})
    if task:
        md.setdefault("task", task)

    async with SessionManager(
        start_url=url,
        headless=headless,
        port=port,
        results_dir=Path(results_dir) if results_dir else None,
        task_metadata=md or None,
        max_steps=max_steps,
        max_turns_per_step=max_turns_per_step,
    ) as sm:
        ready = await sm.wait_for_page_ready()
        print(f"page settled: {ready}")

        if task is None:
            return None

        tree = await sm.run_task(task)
        print(
            f"\n=== TASK DONE ===\n"
            f"  exit:        {tree.task_exit}\n"
            f"  success:     {tree.success}\n"
            f"  steps:       {tree.step_count}\n"
            f"  tokens:      in={tree.total_input_tokens}, out={tree.total_output_tokens}\n"
            f"  final_url:   {tree.final_url}\n"
            f"  final_answer: {tree.final_answer!r}\n"
        )
        return tree


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Drive morphnet_v2 over a task list (file) or a single ad-hoc task (--url).",
    )
    p.add_argument("--tasks", default=None, help="Path to task-list JSON.")
    p.add_argument("--url", default=None, help="Single ad-hoc URL — pair with --task.")
    p.add_argument("--task", default=None, help="Task description (used with --url).")
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"Results dir (default: {DEFAULT_OUT}).")
    p.add_argument("--headless", type=_str_bool, default=False,
                   help="true|false. Default: false (headed).")
    p.add_argument("--port", type=int, default=9222, help="CDP port (default 9222).")
    p.add_argument("--cooldown", type=float, default=5.0,
                   help="Seconds between tasks (rate-limit guard).")
    p.add_argument("--max-steps", type=int, default=10,
                   help="Planner step budget (max branches per task).")
    p.add_argument("--max-turns-per-step", type=int, default=60,
                   help="Max LLM turns within one CU step.")
    p.add_argument("--max-retries", type=int, default=1,
                   help="Per-task retries on crash before moving on.")
    return p.parse_args()


def _load_tasks(args: argparse.Namespace) -> list[dict]:
    if args.url and args.tasks:
        raise SystemExit("pass --url OR --tasks, not both")
    if args.url:
        return [{"url": args.url, "task": args.task}]
    if args.tasks:
        tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
        if not isinstance(tasks, list) or not tasks:
            raise SystemExit("tasks file must be a non-empty JSON array")
        return tasks
    raise SystemExit("provide --url (with optional --task) or --tasks <file>")


async def _main() -> None:
    args = _parse_args()
    tasks = _load_tasks(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"running {len(tasks)} tasks → {out_dir}")

    for i, task in enumerate(tasks):
        # task_metadata is a WRITE-ONLY trace tag — SessionManager persists it
        # to metadata.json on disk for eval-time grading joins. Nothing in the
        # LLM-facing code path (PageAgent / Planner / call_gemini) reads it
        # back. expected_answer is safe to include here for the same reason.
        # See feedback_task_metadata_write_only memory.
        passthrough_md = {k: v for k, v in task.items() if k in ("label", "site", "expected_answer")}
        for attempt in range(args.max_retries + 1):
            try:
                await run_task(
                    url=task["url"],
                    task=task.get("task"),
                    headless=task.get("headless", args.headless),
                    port=args.port,
                    results_dir=out_dir,
                    max_steps=int(task.get("max_steps", args.max_steps)),
                    max_turns_per_step=int(task.get("max_turns_per_step", args.max_turns_per_step)),
                    task_metadata=passthrough_md,
                )
                break
            except Exception as e:
                print(f"  ✗ task failed (attempt {attempt+1}/{args.max_retries+1}): {e!r}")
                if attempt == args.max_retries:
                    break
        if i < len(tasks) - 1 and args.cooldown > 0:
            print(f"cooldown {args.cooldown}s")
            await asyncio.sleep(args.cooldown)

    print("\ndone")


if __name__ == "__main__":
    asyncio.run(_main())