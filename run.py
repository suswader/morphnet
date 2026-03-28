"""WebArena Verified benchmark runner.

Usage:
  python run.py --tasks verified-4site                    # full 4-site set (678 tasks)
  python run.py --tasks hard-4site                        # hard subset (233 tasks)
  python run.py --tasks hard-4site --reset-mcps           # clear MCPs before run
  python run.py --run-name "baseline_no_mcp"              # label for output files
  python run.py --task-ids 11,12,95,105 --run-name test   # specific task IDs only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import time
from datetime import datetime, timezone

from playwright.async_api import async_playwright

from evolving_interface import config, orchestrator
from evolving_interface.computer_use import start_capture, stop_capture
from evolving_interface.eval_harness import (
    compute_sr_tmpl,
    format_answer,
    get_task_type,
    load_tasks,
    print_results_table,
    resolve_url,
    save_run,
    score_task,
    semantic_score_task,
)
from evolving_interface.mcp_generator import optimize_library

VIEWPORT = {"width": 1440, "height": 900}
OPTIMIZE_EVERY = 25  # optimize MCP library every N tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WebArena Verified benchmark")
    p.add_argument(
        "--tasks", default="verified-4site",
        help="Task file name without .json (default: verified-4site)",
    )
    p.add_argument(
        "--task-ids", default="",
        help="Comma-separated task IDs to run (overrides --tasks filtering)",
    )
    p.add_argument(
        "--reset-mcps", action="store_true",
        help="Clear all discovered MCP tools before this run",
    )
    p.add_argument(
        "--run-name", default="",
        help="Label for this run (used in output filenames)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task ordering (default: 42)",
    )
    p.add_argument(
        "--headless", action="store_true", default=True,
        help="Run browser headless (default: true)",
    )
    p.add_argument(
        "--delay", type=int, default=0,
        help="Seconds to wait between tasks (helps with API rate limits)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{args.tasks}"

    # Load tasks
    tasks = load_tasks(f"{args.tasks}.json")
    if args.task_ids:
        ids = set(int(x) for x in args.task_ids.split(","))
        tasks = [t for t in tasks if t["task_id"] in ids]

    # Shuffle with fixed seed for reproducible ordering
    random.seed(args.seed)
    random.shuffle(tasks)

    # Reset MCPs if requested
    if args.reset_mcps:
        mcps_dir = config.MCPS_DIR
        if mcps_dir.exists():
            for site_dir in mcps_dir.iterdir():
                if site_dir.is_dir():
                    shutil.rmtree(site_dir)
            print(f"Cleared all MCP tools from {mcps_dir}")

    # Print run info
    sites = sorted(set(s for t in tasks for s in t["sites"]))
    print(f"\n{'='*70}")
    print(f"  EVOLVING INTERFACE — WebArena Verified Benchmark")
    print(f"{'='*70}")
    print(f"  Run:         {run_name}")
    print(f"  Task file:   {args.tasks}.json")
    print(f"  Tasks:       {len(tasks)}")
    print(f"  Sites:       {sites}")
    print(f"  MCPs reset:  {args.reset_mcps}")
    print(f"  Router:      {config.GEMINI_REASONING_MODEL}")
    print(f"  CU model:    {config.GEMINI_COMPUTER_USE_MODEL}")
    print(f"  Fast model:  {config.GEMINI_FAST_MODEL}")
    print(f"{'='*70}\n")

    # Launch browser
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=args.headless,
        args=["--window-size=1440,900"],
    )

    eval_results: list[dict] = []
    run_start = time.monotonic()

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        site = task["sites"][0]
        task_type = get_task_type(task)
        intent = task["intent"]
        start_url = resolve_url(task["start_urls"][0])

        print(f"\n{'─'*70}")
        print(
            f"  [{i+1}/{len(tasks)}] Task {task_id} "
            f"[{site}] ({task_type})"
        )
        print(f"  {intent[:90]}")
        print(f"  Start: {start_url}")

        # Fresh browser context per task (clean cookies/state)
        context = await browser.new_context(viewport=VIEWPORT)

        page = await context.new_page()

        try:
            await page.goto(
                start_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )
            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=10000,
                )
            except Exception:
                pass
        except Exception as exc:
            print(f"  SKIP: Could not load {start_url}: {exc}")
            eval_results.append({
                "task_id": task_id,
                "site": site,
                "intent_template_id": task["intent_template_id"],
                "task_type": task_type,
                "passed": False,
                "agent_response": {
                    "action": "retrieve",
                    "status": "UNKNOWN_ERROR",
                    "results": None,
                    "error_details": str(exc),
                },
                "final_answer": None,
                "time_seconds": 0,
            })
            await context.close()
            continue

        # Top-level traffic capture for NetworkEventEvaluator scoring
        capture_ctx = start_capture(page)

        t0 = time.monotonic()
        try:
            task_result = await orchestrator.run_task(
                str(task_id), intent, site, page,
            )
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            task_result = None

        elapsed = time.monotonic() - t0
        all_requests = stop_capture(capture_ctx)

        # Merge step-level captured requests (especially MCP httpx traffic)
        # into the top-level list for NetworkEventEvaluator scoring.
        # MCP tools use httpx directly, bypassing Playwright's network layer.
        if task_result:
            for sr in task_result.steps:
                for cr in sr._captured_requests:
                    all_requests.append(cr)

        # Format the agent's answer into structured JSON
        final_answer = task_result.final_answer if task_result else None
        agent_response = format_answer(task, final_answer, page.url)

        # Score against ground truth (strict + semantic)
        passed = score_task(task, agent_response, all_requests)
        semantic_passed, semantic_reason = semantic_score_task(
            task, agent_response,
        )

        # Build result record
        result = {
            "task_id": task_id,
            "site": site,
            "intent_template_id": task["intent_template_id"],
            "task_type": task_type,
            "passed": passed,
            "semantic_passed": semantic_passed,
            "semantic_reason": semantic_reason,
            "agent_response": agent_response,
            "final_answer": final_answer,
            "strategy": task_result.strategy if task_result else "error",
            "time_seconds": round(elapsed, 1),
            "tokens": (
                (task_result.total_tokens_input + task_result.total_tokens_output)
                if task_result else 0
            ),
            "tools_discovered": (
                task_result.new_tools_discovered if task_result else []
            ),
        }
        eval_results.append(result)

        # Save action logs for this task
        if task_result:
            action_logs_dir = config.RESULTS_DIR / "action_logs"
            action_logs_dir.mkdir(exist_ok=True)
            log_path = action_logs_dir / f"task_{task_id}.jsonl"
            with open(log_path, "w") as f:
                for sr in task_result.steps:
                    for entry in sr.action_log:
                        f.write(json.dumps(entry, default=str) + "\n")

        # Print live result
        status = "PASS" if passed else "FAIL"
        # Debug note: show semantic match only when strict fails
        debug_sem = ""
        if not passed and semantic_passed:
            debug_sem = f"  (debug: semantic match — {semantic_reason[:60]})"
        ans_preview = (final_answer or "(none)")[:50]
        print(
            f"  -> {status}  {elapsed:.1f}s  "
            f"[{agent_response['status']}]  {ans_preview}"
            f"{debug_sem}"
        )

        await context.close()

        # Rate-limit delay between tasks
        if args.delay > 0 and i < len(tasks) - 1:
            print(f"  (waiting {args.delay}s for API quota)")
            await asyncio.sleep(args.delay)

        # Periodic MCP optimization
        if (i + 1) % OPTIMIZE_EVERY == 0:
            for s in set(r["site"] for r in eval_results):
                optimize_library(s)

    total_time = time.monotonic() - run_start

    # Final metrics (strict WebArena Verified evaluation only)
    sr, ci, n_tmpl = compute_sr_tmpl(eval_results)
    passed_count = sum(1 for r in eval_results if r["passed"])

    print(f"\n\n{'='*70}")
    print(f"  RUN COMPLETE: {run_name}")
    print(f"{'='*70}")
    print(f"  Wall time:  {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Passed:     {passed_count}/{len(eval_results)}")
    print()

    # Debug: count near-misses (strict fail, semantic pass)
    near_misses = sum(
        1 for r in eval_results
        if not r["passed"] and r.get("semantic_passed")
    )
    if near_misses:
        print(
            f"  (debug: {near_misses} near-miss tasks — "
            f"semantic match but strict fail)"
        )
        print()

    summary_lines = print_results_table(run_name, eval_results)
    save_run(eval_results, run_name, summary_lines)

    await browser.close()
    await pw.stop()


if __name__ == "__main__":
    asyncio.run(main())
