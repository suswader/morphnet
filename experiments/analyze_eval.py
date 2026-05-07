"""Analyze MorphNet eval results — per-site metrics, diagnostics, failure analysis.

All classification is derived from structured trace fields (outcome, event_type,
module, confidence) — never from substring matching on answer text.

Usage:
    uv run python experiments/analyze_eval.py results/eval_20260416_143000/
    uv run python experiments/analyze_eval.py results/eval_20260416_143000/ --verbose
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(task_dir: Path) -> list[dict]:
    """Load trace entries from a task's trace.jsonl (searches subdirectories too)."""
    trace_files = list(task_dir.glob("**/trace.jsonl"))
    if not trace_files:
        return []
    entries = []
    for tf in trace_files:
        for line in tf.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def extract_task_diagnostics(trace_entries: list[dict]) -> dict:
    """Extract comprehensive diagnostics from structured trace entries.

    All fields come from typed trace events — no text parsing.
    """
    diag: dict = {
        # Core metrics
        "mcp_calls": 0,
        "mcp_successes": 0,
        "mcp_failures": 0,
        "mcp_tools_used": set(),
        "cu_subtasks": 0,
        "cu_subtasks_succeeded": 0,
        "total_actions": 0,
        "actions_succeeded": 0,
        "actions_failed": 0,
        # Timing
        "timestamps": [],
        "duration_seconds": 0,
        # Events
        "popups_dismissed": 0,
        "crashes_recovered": 0,
        "crashes_unrecoverable": 0,
        "pages_reattached": 0,
        "enricher_failures": 0,
        "enricher_disabled": [],
        # Traffic
        "http_errors": 0,  # 4xx/5xx
        "http_requests": 0,
        # Reflector
        "stage1_verdicts": {"success": 0, "failure": 0, "ambiguous": 0},
        "stage3_llm_calls": 0,
        # Confidence tracking
        "action_confidences": [],
        "subtask_confidences": [],
        # Structured outcome from orchestrator
        "task_success_field": None,  # From task_completed event
        "task_confidence": None,
    }

    for entry in trace_entries:
        module = entry.get("module", "")
        event = entry.get("event_type", "")
        outcome = entry.get("outcome")
        detail = entry.get("detail") or {}
        ts = entry.get("timestamp")
        confidence = entry.get("confidence")

        if ts:
            diag["timestamps"].append(ts)

        # MCP
        if module == "mcp_manager" and "execute" in event:
            diag["mcp_calls"] += 1
            tool_name = detail.get("tool_name", "unknown")
            diag["mcp_tools_used"].add(tool_name)
            if outcome == "success":
                diag["mcp_successes"] += 1
            else:
                diag["mcp_failures"] += 1

        # CU subtasks
        if module == "cu_agent" and event == "subtask_started":
            diag["cu_subtasks"] += 1
        if module == "cu_agent" and event == "subtask_completed":
            if detail.get("success"):
                diag["cu_subtasks_succeeded"] += 1
            if confidence is not None:
                diag["subtask_confidences"].append(confidence)

        # Actions
        if module == "cu_agent" and event == "action_selected":
            diag["total_actions"] += 1
            if confidence is not None:
                diag["action_confidences"].append(confidence)

        # Reflector verdicts (Stage 1 deterministic)
        if module == "reflector" and event == "deterministic_signals":
            verdict = detail.get("verdict", "")
            if verdict in diag["stage1_verdicts"]:
                diag["stage1_verdicts"][verdict] += 1
            if verdict == "success":
                diag["actions_succeeded"] += 1
            elif verdict == "failure":
                diag["actions_failed"] += 1

        # Reflector Stage 3 (LLM calls)
        if module == "reflector" and event in ("llm_evaluate_action", "stage3_llm"):
            diag["stage3_llm_calls"] += 1

        # Popups
        if event == "popup_dismissed":
            diag["popups_dismissed"] += 1

        # Crashes
        if event == "crash_recovered":
            diag["crashes_recovered"] += 1
        if event == "crash_unrecoverable":
            diag["crashes_unrecoverable"] += 1
        if event == "page_reattached":
            diag["pages_reattached"] += 1

        # Traffic
        if event == "traffic_captured":
            diag["http_requests"] += 1
            status = detail.get("status_code", 0)
            if isinstance(status, int) and status >= 400:
                diag["http_errors"] += 1

        # Task completion (structured boolean from orchestrator)
        if module == "orchestrator" and event == "task_completed":
            diag["task_success_field"] = detail.get("task_success")
            diag["task_confidence"] = detail.get("confidence") or confidence

    # Compute duration
    if len(diag["timestamps"]) >= 2:
        diag["duration_seconds"] = round(max(diag["timestamps"]) - min(diag["timestamps"]), 1)

    # Clean up non-serializable fields
    diag["mcp_tools_used"] = sorted(diag["mcp_tools_used"])
    del diag["timestamps"]

    return diag


def classify_result(result: dict, diag: dict) -> str:
    """Classify a task result using structured signals only.

    Returns one of: TRUE_SUCCESS, FALSE_SUCCESS, TRUE_FAILURE, CRASH, NOT_RUN.
    """
    exit_code = result.get("exit_code", 0)
    reported_success = result.get("success", False)

    # Process-level crash
    if exit_code != 0:
        return "CRASH"

    # Use the orchestrator's structured task_success field (from Gemini schema)
    task_success = diag.get("task_success_field")
    if task_success is not None:
        if task_success and reported_success:
            return "TRUE_SUCCESS"
        if not task_success and reported_success:
            return "FALSE_SUCCESS"
        if not task_success:
            return "TRUE_FAILURE"

    # Fallback: use reported success + confidence threshold
    # If the model reported success but with low confidence, flag it
    conf = diag.get("task_confidence")
    if reported_success:
        if conf is not None and conf < 0.5:
            return "FALSE_SUCCESS"
        return "TRUE_SUCCESS"

    return "TRUE_FAILURE"


def identify_failure_category(diag: dict) -> str:
    """Categorize failure root cause from structured diagnostics.

    Returns a category string derived from event counts, not text matching.
    """
    if diag["crashes_unrecoverable"] > 0:
        return "browser_crash"
    if diag["mcp_failures"] > 0 and diag["mcp_successes"] == 0 and diag["mcp_calls"] > 3:
        return "mcp_all_failed"
    if diag["http_errors"] > diag["http_requests"] * 0.3 and diag["http_errors"] > 2:
        return "http_errors"
    if diag["total_actions"] == 0 and diag["cu_subtasks"] == 0:
        return "no_actions_taken"
    if diag["actions_failed"] > diag["actions_succeeded"] and diag["total_actions"] > 5:
        return "action_failures_dominant"
    if diag["popups_dismissed"] > 3:
        return "popup_interference"
    return "other"


def analyze(eval_dir: Path, verbose: bool = False) -> dict:
    """Analyze all task results in an eval directory."""
    results_by_site: dict[str, list[dict]] = defaultdict(list)
    all_results: list[dict] = []

    for result_file in sorted(eval_dir.rglob("result.json")):
        try:
            result = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        task_dir = result_file.parent
        result["task_dir"] = str(task_dir)

        # Extract structured diagnostics from trace
        trace_entries = load_trace(task_dir)
        diag = extract_task_diagnostics(trace_entries)
        result["diagnostics"] = diag

        # Classify using structured fields
        result["classification"] = classify_result(result, diag)
        if result["classification"] in ("TRUE_FAILURE", "FALSE_SUCCESS", "CRASH"):
            result["failure_category"] = identify_failure_category(diag)
        else:
            result["failure_category"] = None

        site = result.get("site", "unknown")
        results_by_site[site].append(result)
        all_results.append(result)

    if not all_results:
        print(f"No results found in {eval_dir}")
        return {}

    # === Aggregate Metrics ===
    total = len(all_results)
    by_class = defaultdict(int)
    for r in all_results:
        by_class[r["classification"]] += 1

    total_mcp = sum(r["diagnostics"]["mcp_calls"] for r in all_results)
    total_mcp_ok = sum(r["diagnostics"]["mcp_successes"] for r in all_results)
    total_cu = sum(r["diagnostics"]["cu_subtasks"] for r in all_results)
    total_actions = sum(r["diagnostics"]["total_actions"] for r in all_results)
    total_popups = sum(r["diagnostics"]["popups_dismissed"] for r in all_results)
    total_crashes = sum(r["diagnostics"]["crashes_unrecoverable"] for r in all_results)
    total_recovered = sum(r["diagnostics"]["crashes_recovered"] for r in all_results)
    total_http_errs = sum(r["diagnostics"]["http_errors"] for r in all_results)
    total_duration = sum(r["diagnostics"]["duration_seconds"] for r in all_results)

    all_mcp_tools: set[str] = set()
    for r in all_results:
        all_mcp_tools.update(r["diagnostics"]["mcp_tools_used"])

    # Failure category distribution
    failure_cats = defaultdict(int)
    for r in all_results:
        cat = r.get("failure_category")
        if cat:
            failure_cats[cat] += 1

    # === Print Report ===
    true_success = by_class["TRUE_SUCCESS"]

    print("=" * 72)
    print(f"  MorphNet Eval Results — {eval_dir.name}")
    print("=" * 72)
    print()
    print(f"  Total tasks:        {total}")
    print(f"  True success:       {true_success}/{total} ({true_success/total*100:.0f}%)")
    print(f"  False success:      {by_class['FALSE_SUCCESS']}")
    print(f"  True failure:       {by_class['TRUE_FAILURE']}")
    print(f"  Crashes:            {by_class['CRASH']}")
    print()
    print(f"  Total actions:      {total_actions}")
    print(f"  Total duration:     {total_duration/60:.1f} min")
    print(f"  Popups dismissed:   {total_popups}")
    print(f"  Crashes recovered:  {total_recovered}")
    print(f"  Crashes fatal:      {total_crashes}")
    print(f"  HTTP errors (4xx+): {total_http_errs}")
    print()
    print(f"  MCP calls total:    {total_mcp} (ok: {total_mcp_ok}, fail: {total_mcp - total_mcp_ok})")
    print(f"  CU subtasks total:  {total_cu}")
    if total_mcp > 0:
        print(f"  MCP success rate:   {total_mcp_ok/total_mcp*100:.0f}%")
    print()

    # === Failure Categories ===
    if failure_cats:
        print("  Failure Categories:")
        for cat, count in sorted(failure_cats.items(), key=lambda x: -x[1]):
            print(f"    {cat:<30} {count}")
        print()

    # === Per-Site Breakdown ===
    print("-" * 72)
    hdr = f"  {'Site':<15} {'Tasks':>5} {'True✓':>6} {'False✓':>7} {'Fail':>5} {'Crash':>6} {'MCP':>5} {'CU':>4} {'Time':>6}"
    print(hdr)
    print("-" * 72)

    site_summary = {}
    for site in sorted(results_by_site.keys()):
        sr = results_by_site[site]
        n = len(sr)
        ts = sum(1 for r in sr if r["classification"] == "TRUE_SUCCESS")
        fs = sum(1 for r in sr if r["classification"] == "FALSE_SUCCESS")
        tf = sum(1 for r in sr if r["classification"] == "TRUE_FAILURE")
        cr = sum(1 for r in sr if r["classification"] == "CRASH")
        mc = sum(r["diagnostics"]["mcp_calls"] for r in sr)
        cu = sum(r["diagnostics"]["cu_subtasks"] for r in sr)
        dur = sum(r["diagnostics"]["duration_seconds"] for r in sr)

        pct = f"({ts/n*100:.0f}%)" if n > 0 else ""
        print(f"  {site:<15} {n:>5} {ts:>4}{pct:>5} {fs:>7} {tf:>5} {cr:>6} {mc:>5} {cu:>4} {dur/60:>5.1f}m")

        site_summary[site] = {
            "tasks": n, "true_success": ts, "false_success": fs,
            "true_failure": tf, "crashes": cr,
            "mcp_calls": mc, "cu_subtasks": cu,
            "duration_minutes": round(dur / 60, 1),
            "failure_categories": {},
        }
        for r in sr:
            cat = r.get("failure_category")
            if cat:
                site_summary[site]["failure_categories"][cat] = \
                    site_summary[site]["failure_categories"].get(cat, 0) + 1

    print("-" * 72)
    print()

    # === Per-Task Details (verbose) ===
    if verbose:
        print("=" * 72)
        print("  Per-Task Diagnostics")
        print("=" * 72)
        for site in sorted(results_by_site.keys()):
            print(f"\n  [{site}]")
            for r in results_by_site[site]:
                cls = r["classification"]
                d = r["diagnostics"]
                label = r.get("label", "?")
                cat = r.get("failure_category", "")
                cat_str = f" [{cat}]" if cat else ""
                dur = d["duration_seconds"]

                # Confidence
                conf = d.get("task_confidence")
                conf_str = f" conf={conf:.2f}" if conf is not None else ""

                # Action efficiency
                act_total = d["total_actions"]
                act_ok = d["actions_succeeded"]
                act_fail = d["actions_failed"]

                print(f"    [{cls:>13}] {label:<30}{cat_str}")
                print(f"      actions={act_total} (ok={act_ok} fail={act_fail}) "
                      f"subtasks={d['cu_subtasks']} mcp={d['mcp_calls']} "
                      f"time={dur:.0f}s{conf_str}")

                # Noteworthy events
                notes = []
                if d["popups_dismissed"]:
                    notes.append(f"popups={d['popups_dismissed']}")
                if d["crashes_recovered"]:
                    notes.append(f"crashes_recovered={d['crashes_recovered']}")
                if d["crashes_unrecoverable"]:
                    notes.append(f"CRASH_FATAL={d['crashes_unrecoverable']}")
                if d["http_errors"]:
                    notes.append(f"http_errs={d['http_errors']}")
                if d["pages_reattached"]:
                    notes.append(f"reattached={d['pages_reattached']}")
                if d["stage3_llm_calls"]:
                    notes.append(f"llm_reflections={d['stage3_llm_calls']}")
                if notes:
                    print(f"      events: {', '.join(notes)}")

                # Answer preview
                answer = (r.get("answer") or "")[:100]
                if answer:
                    print(f"      answer: {answer}")
        print()

    # === Save summary JSON ===
    summary = {
        "eval_dir": str(eval_dir),
        "total_tasks": total,
        "classification": dict(by_class),
        "true_success_rate": round(true_success / total * 100, 1) if total > 0 else 0,
        "failure_categories": dict(failure_cats),
        "total_actions": total_actions,
        "total_duration_minutes": round(total_duration / 60, 1),
        "total_mcp_calls": total_mcp,
        "total_mcp_successes": total_mcp_ok,
        "total_cu_subtasks": total_cu,
        "total_popups_dismissed": total_popups,
        "total_crashes_fatal": total_crashes,
        "total_crashes_recovered": total_recovered,
        "total_http_errors": total_http_errs,
        "mcp_tools_used": sorted(all_mcp_tools),
        "per_site": site_summary,
    }

    summary_path = eval_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary saved to: {summary_path}")

    # === Save per-task diagnostics JSON ===
    per_task = []
    for r in all_results:
        per_task.append({
            "label": r.get("label"),
            "site": r.get("site"),
            "classification": r["classification"],
            "failure_category": r.get("failure_category"),
            "success": r.get("success"),
            "exit_code": r.get("exit_code"),
            "answer_preview": (r.get("answer") or "")[:200],
            "diagnostics": r["diagnostics"],
        })
    per_task_path = eval_dir / "eval_per_task.json"
    per_task_path.write_text(json.dumps(per_task, indent=2, default=str))
    print(f"Per-task diagnostics saved to: {per_task_path}")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python experiments/analyze_eval.py <eval_dir> [--verbose]")
        sys.exit(1)

    eval_dir = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv

    if not eval_dir.exists():
        print(f"Directory not found: {eval_dir}")
        sys.exit(1)

    analyze(eval_dir, verbose=verbose)
