"""
End-to-end two-task experiment: CU discovery → graph building → executor replay.

Task 1: Search Guwahati→Dibrugarh trains on ConfirmTkt via CU.
         Observer captures HTTP traffic. Learner builds execution graph.
Task 2: Search Ranchi→Dhanbad trains on ConfirmTkt.
         Orchestrator should find the graph and route to executor.

All logging goes through trace.py. Run in headed mode via session_manager.

Usage:
    uv run python experiments/test_two_task_e2e.py
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphnet.session_manager import SessionManager
from morphnet.morphnet_orchestrator import MorphNetOrchestrator
from morphnet.trace import TaskTrace, Evidence
from morphnet.manifest import list_graphs, load_graph

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PORT = 9444
SITE = "confirmtkt_com"
URL = "https://www.confirmtkt.com"
SITE_DIR = Path(__file__).parent.parent / "morphnet" / "sites" / SITE
RESULTS_BASE = Path(__file__).parent.parent / "results"

# Compute future dates dynamically (3 and 4 days ahead)
_DATE_1 = (datetime.now() + timedelta(days=3)).strftime("%-d %B %Y")
_DATE_2 = (datetime.now() + timedelta(days=4)).strftime("%-d %B %Y")

# Use different city pairs each run to avoid site memorizing previous search.
# Task 1 (CU discovery) and Task 2 (executor replay) use different routes
# so the executor must generalize, not just replay identical parameters.
TASK_1 = (
    "Navigate to the ConfirmTkt home page, clear any prefilled fields, "
    "then type 'Guwahati' in the From field and select 'GHY - Guwahati' from the suggestions. "
    "Then type 'Dibrugarh' in the To field and select 'DBRG - Dibrugarh' from the suggestions. "
    f"Set the departure date to {_DATE_2} and click SEARCH. "
    "Tell me the name of any one train on this route."
)

TASK_2 = (
    f"Search for trains from Ranchi to Dhanbad on {_DATE_1} on ConfirmTkt. "
    "Tell me the name of any one train on this route."
)


# ---------------------------------------------------------------------------
# Chrome launch (mirrors session_manager's CLI)
# ---------------------------------------------------------------------------

def _find_chrome() -> str:
    system = platform.system()
    if system == "Darwin":
        path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if Path(path).exists():
            return path
    elif system == "Linux":
        for name in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
            found = shutil.which(name)
            if found:
                return found
    raise FileNotFoundError("Chrome not found")


def _launch_chrome(port: int) -> subprocess.Popen:
    chrome_bin = _find_chrome()
    project_root = Path(__file__).parent.parent
    tmp_dir = project_root / ".tmp" / "chrome-profiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = tmp_dir / f"chrome-e2e-{port}"
    cmd = [
        chrome_bin,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-default-apps",
        "--disable-component-update",
        "--disable-breakpad",
        "--disable-sync",
        "--metrics-recording-only",
        "--disable-dev-shm-usage",
        "--disable-features=Translate,OptimizationHints,MediaRouter",
        "--disable-blink-features=AutomationControlled",
        "--exclude-switches=enable-automation",
        "--use-gl=angle",
        "--use-angle=default",
        "--window-size=1920,1080",
        "--window-position=0,0",
        "--password-store=basic",
        "--use-mock-keychain",
        "--force-color-profile=srgb",
        "--lang=en-US",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_cdp(port: int, timeout: int = 15) -> None:
    import urllib.request
    url = f"http://localhost:{port}/json/version"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Chrome CDP not responding on port {port}")


# ---------------------------------------------------------------------------
# Site state cleanup
# ---------------------------------------------------------------------------

def clean_site_state(trace: TaskTrace) -> None:
    """Remove old graphs, captures, and tools.json so we start fresh."""
    removed = []
    for subpath in ["tools.json", "graphs", "captures", "bundle", "embeddings.json"]:
        target = SITE_DIR / subpath
        if target.is_file():
            target.unlink()
            removed.append(str(subpath))
        elif target.is_dir():
            shutil.rmtree(target)
            removed.append(str(subpath))

    trace.log(
        "experiment", "site_state_cleaned",
        f"Cleaned {SITE}: removed {removed}",
        detail={"site": SITE, "removed": removed, "preserved": ["profile.json"]},
        outcome="success",
    )


# ---------------------------------------------------------------------------
# Diagnostic: dump what observer captured / learner built
# ---------------------------------------------------------------------------

def dump_graph_diagnostics(trace: TaskTrace) -> int:
    """Inspect graphs built after Task 1. Returns count of graphs found."""
    graphs = list_graphs(SITE)
    trace.log(
        "experiment", "graph_inventory",
        f"Found {len(graphs)} graphs for {SITE}",
        detail={
            "count": len(graphs),
            "graphs": [
                {
                    "id": g.id[:12],
                    "name": g.name,
                    "capability": g.capability_statement,
                    "nodes": len(g.nodes),
                    "edges": len(g.edges),
                    "terminals": len(g.terminal_node_ids),
                    "verified": g.verified,
                    "framework": g.framework_detected,
                }
                for g in graphs
            ],
        },
    )

    for g in graphs:
        # Dump node details
        node_details = []
        for n in g.nodes:
            user_intent_params = [p.name for p in n.core_parameters if p.role == "user_intent"]
            chained_params = [p.name for p in n.core_parameters if p.role == "chained"]
            constant_params = [p.name for p in n.core_parameters if p.role == "captured_constant"]
            node_details.append({
                "id": n.id,
                "fingerprint": n.endpoint_fingerprint,
                "method": n.http_method,
                "url_template": n.url_template[:120],
                "invocation_type": n.invocation.type,
                "user_intent_params": user_intent_params,
                "chained_params": chained_params,
                "constant_params": constant_params,
                "optional_params": [p.name for p in n.optional_parameters],
                "extract_paths": n.response_extract_paths,
                "cu_reasoning": n.cu_reasoning_sample[:200] if n.cu_reasoning_sample else "",
            })

        edge_details = [
            {
                "from": e.from_node_id,
                "to": e.to_node_id,
                "extract": e.from_extract,
                "param": e.to_parameter,
            }
            for e in g.edges
        ]

        trace.log(
            "experiment", "graph_detail",
            f"Graph {g.name}: {len(g.nodes)} nodes, {len(g.edges)} edges",
            detail={
                "graph_id": g.id[:12],
                "name": g.name,
                "description": g.description,
                "capability_statement": g.capability_statement,
                "preconditions": g.preconditions,
                "completion": g.completion,
                "nodes": node_details,
                "edges": edge_details,
                "terminal_ids": g.terminal_node_ids,
            },
        )

    return len(graphs)


def dump_observation_diagnostics(trace: TaskTrace) -> None:
    """Inspect raw observations saved after Task 1."""
    captures_dir = SITE_DIR / "captures"
    if not captures_dir.exists():
        trace.log("experiment", "no_observations", "No captures directory found",
                   outcome="warning")
        return

    for capture_file in sorted(captures_dir.glob("*.json")):
        try:
            data = json.loads(capture_file.read_text())
            trace.log(
                "experiment", "observation_summary",
                f"Observation {capture_file.stem}: {len(data.get('http_requests', []))} HTTP, "
                f"{len(data.get('cu_actions', []))} CU actions",
                detail={
                    "subtask_id": data.get("subtask_id", ""),
                    "description": data.get("subtask_description", "")[:200],
                    "http_request_count": len(data.get("http_requests", [])),
                    "cu_action_count": len(data.get("cu_actions", [])),
                    "dom_snapshot_count": len(data.get("dom_snapshots", [])),
                    "script_count": len(data.get("scripts", {})),
                    "framework_fingerprint": data.get("framework_fingerprint", {}),
                    "bundle_hash": data.get("bundle_hash", "")[:16],
                    "reflector_verdict": data.get("reflector_verdict", ""),
                    "start_url": data.get("start_url", ""),
                    "end_url": data.get("end_url", ""),
                    # Log first 5 HTTP request URLs for quick inspection
                    "http_urls_sample": [
                        {
                            "url": r.get("url", "")[:150],
                            "method": r.get("method", ""),
                            "status": r.get("response_status", 0),
                            "type": r.get("request_type", ""),
                        }
                        for r in data.get("http_requests", [])[:10]
                    ],
                    # Log CU actions for reasoning inspection
                    "cu_actions_sample": [
                        {
                            "type": a.get("action_type", ""),
                            "target_text": a.get("target_text", "")[:80],
                            "typed_value": a.get("typed_value", ""),
                            "reasoning": a.get("cu_reasoning", "")[:200],
                        }
                        for a in data.get("cu_actions", [])[:10]
                    ],
                },
            )
        except Exception as exc:
            trace.log("experiment", "observation_read_error",
                       f"Failed to read {capture_file.stem}: {exc}",
                       error=str(exc))


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

async def run_task(
    task_prompt: str,
    task_label: str,
    trace: TaskTrace,
    port: int,
    max_subtasks: int = 10,
) -> dict:
    """Run one task through the full orchestrator pipeline.

    Returns a summary dict with success, answer, routing info.
    """
    task_trace_id = trace.log(
        "experiment", "task_started",
        f"[{task_label}] {task_prompt[:80]}...",
        detail={"label": task_label, "task": task_prompt, "max_subtasks": max_subtasks},
    )

    session = SessionManager(
        start_url=URL,
        task_prompt=task_prompt,
        headless=False,
        chrome_cdp_url=f"http://localhost:{port}",
        site_name=SITE,
        trace=trace,
    )

    try:
        await session.start()
        trace.log(
            "experiment", "session_started",
            f"[{task_label}] Browser at {session.page.url}",
            parent_id=task_trace_id,
        )

        orchestrator = MorphNetOrchestrator(session=session, trace=trace)
        result = await orchestrator.run_task(task_prompt, max_subtasks=max_subtasks)

        summary = {
            "label": task_label,
            "success": result.success,
            "answer": result.final_answer,
            "subtasks_completed": result.subtasks_completed,
            "total_actions": result.total_actions,
            "total_executor_calls": result.total_executor_calls,
            "planning_tree": result.planning_tree_summary[:500],
        }

        # Determine if executor was used by checking trace entries
        executor_entries = trace.get_entries(module="orchestrator", event_type="executor_success")
        executor_fallback_entries = trace.get_entries(module="orchestrator", event_type="executor_fallback_to_cu")
        summary["executor_used"] = len(executor_entries) > 0
        summary["executor_fallback_count"] = len(executor_fallback_entries)

        trace.log(
            "experiment", "task_completed",
            f"[{task_label}] success={result.success}, "
            f"answer={str(result.final_answer)[:100]}, "
            f"executor_calls={result.total_executor_calls}",
            detail=summary,
            parent_id=task_trace_id,
            outcome="success" if result.success else "failure",
        )

        return summary

    except Exception as exc:
        trace.log(
            "experiment", "task_crashed",
            f"[{task_label}] Crashed: {exc}",
            error=str(exc),
            parent_id=task_trace_id,
            outcome="failure",
        )
        return {
            "label": task_label,
            "success": False,
            "answer": None,
            "error": str(exc),
        }
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    # Create experiment output directory
    now = datetime.now()
    exp_dir = RESULTS_BASE / f"e2e_{now.strftime('%Y-%m-%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Configure Python logging: observer/learner/executor log to console + file
    log_path = exp_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-25s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)

    trace = TaskTrace(
        task_prompt="E2E two-task experiment: CU discovery → executor replay",
        output_dir=exp_dir,
    )

    trace.log("experiment", "experiment_started", "Two-task ConfirmTkt E2E experiment", detail={
        "task_1": TASK_1[:100],
        "task_2": TASK_2[:100],
        "site": SITE,
        "url": URL,
    })

    print(f"\n{'='*60}")
    print("MorphNet E2E Two-Task Experiment")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}\n")

    # Step 0: Clean site state
    print("[0] Cleaning site state...")
    clean_site_state(trace)

    # Step 1: Launch Chrome (headed)
    print("[1] Launching Chrome (headed mode)...")
    chrome_proc = _launch_chrome(PORT)
    _wait_for_cdp(PORT)
    trace.log("experiment", "chrome_launched", f"Chrome ready on port {PORT}")

    try:
        # ── TASK 1: CU Discovery ──────────────────────────────────
        print(f"\n{'─'*60}")
        print("[2] TASK 1: CU Discovery (Guwahati → Dibrugarh)")
        print(f"{'─'*60}")
        t1_start = time.time()

        result_1 = await run_task(
            task_prompt=TASK_1,
            task_label="task1_cu_discovery",
            trace=trace,
            port=PORT,
            max_subtasks=10,
        )

        t1_elapsed = time.time() - t1_start
        print(f"\n    Task 1 completed in {t1_elapsed:.1f}s")
        print(f"    Success: {result_1.get('success')}")
        print(f"    Answer: {str(result_1.get('answer', ''))[:100]}")
        print(f"    Actions: {result_1.get('total_actions', 0)}")

        # ── DIAGNOSTICS: What did observer capture? ────────────────
        print(f"\n{'─'*60}")
        print("[3] Diagnostics: Observations & Graphs")
        print(f"{'─'*60}")

        dump_observation_diagnostics(trace)
        graph_count = dump_graph_diagnostics(trace)

        print(f"    Observations: see trace")
        print(f"    Graphs built: {graph_count}")

        if graph_count == 0:
            trace.log(
                "experiment", "no_graph_built",
                "WARNING: No graph was built after Task 1. "
                "Task 2 will fall back entirely to CU.",
                outcome="warning",
            )
            print("\n    WARNING: No graph built! Task 2 will use CU only.")

        # Restart Chrome between tasks — CDP reconnect fails otherwise
        # (Browser.setDownloadBehavior not supported on re-attach)
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome_proc.kill()
        await asyncio.sleep(1)
        chrome_proc = _launch_chrome(PORT)
        _wait_for_cdp(PORT)
        trace.log("experiment", "chrome_restarted", "Chrome restarted for Task 2")

        # ── TASK 2: Executor Replay ──────────────────────────────
        print(f"\n{'─'*60}")
        print("[4] TASK 2: Executor Replay (Vizag → Tirupati)")
        print(f"{'─'*60}")
        t2_start = time.time()

        result_2 = await run_task(
            task_prompt=TASK_2,
            task_label="task2_executor_replay",
            trace=trace,
            port=PORT,
            max_subtasks=10,
        )

        t2_elapsed = time.time() - t2_start
        print(f"\n    Task 2 completed in {t2_elapsed:.1f}s")
        print(f"    Success: {result_2.get('success')}")
        print(f"    Answer: {str(result_2.get('answer', ''))[:100]}")
        print(f"    Executor used: {result_2.get('executor_used', False)}")
        print(f"    Executor calls: {result_2.get('total_executor_calls', 0)}")
        print(f"    CU fallbacks: {result_2.get('executor_fallback_count', 0)}")

        # ── FINAL REPORT ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*60}")

        trace_summary = trace.summary()

        report = {
            "task_1": result_1,
            "task_2": result_2,
            "graphs_built": graph_count,
            "task_2_used_executor": result_2.get("executor_used", False),
            "task_1_time_s": round(t1_elapsed, 1),
            "task_2_time_s": round(t2_elapsed, 1),
            "trace_entries": trace_summary["total_entries"],
            "trace_path": str(trace._trace_path),
        }

        # Determine overall experiment outcome
        if result_1.get("success") and result_2.get("success") and result_2.get("executor_used"):
            experiment_outcome = "FULL_SUCCESS"
            print("  Outcome: FULL SUCCESS")
            print("  Task 1 discovered via CU, learner built graph,")
            print("  Task 2 executed via graph (no CU needed)")
        elif result_1.get("success") and result_2.get("success"):
            experiment_outcome = "PARTIAL_SUCCESS"
            print("  Outcome: PARTIAL SUCCESS")
            print("  Both tasks succeeded, but Task 2 used CU (executor not used)")
        elif result_1.get("success"):
            experiment_outcome = "TASK2_FAILED"
            print("  Outcome: Task 2 FAILED")
        else:
            experiment_outcome = "TASK1_FAILED"
            print("  Outcome: Task 1 FAILED (no discovery happened)")

        report["experiment_outcome"] = experiment_outcome

        trace.log(
            "experiment", "experiment_completed",
            f"Outcome: {experiment_outcome}",
            detail=report,
            outcome="success" if "SUCCESS" in experiment_outcome else "failure",
        )

        print(f"\n  Task 1: {'PASS' if result_1.get('success') else 'FAIL'} "
              f"({result_1.get('total_actions', 0)} actions, {t1_elapsed:.1f}s)")
        print(f"  Task 2: {'PASS' if result_2.get('success') else 'FAIL'} "
              f"({result_2.get('total_actions', 0)} actions, {t2_elapsed:.1f}s)")
        print(f"  Graphs: {graph_count}")
        print(f"  Trace:  {trace._trace_path}")

        # Save the report as a separate JSON for easy access
        report_path = exp_dir / "experiment_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  Report: {report_path}")
        print(f"  Log:    {log_path}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        trace.log("experiment", "interrupted", "User interrupted experiment")
    except Exception as exc:
        print(f"\nExperiment crashed: {exc}")
        trace.log("experiment", "experiment_crashed", str(exc), error=str(exc), outcome="failure")
        raise
    finally:
        trace.close()
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
