"""experiments/run_tool_test.py — 3× same-task validation.

Runs ONE confirmtkt train-search task three times with different slot values:
  Run 1: empty registry. CU discovers. tool_builder synthesises candidates at
         task end + aggregates into morphnet_v2/sites/confirmtkt/tools.json.
  Run 2: registry pre-loaded. Planner sees invoke_<tool> functions. Executor
         replays. Expected: fewer CU steps, faster, same final_answer.
  Run 3: identical to Run 2. Run AFTER A LONG DELAY (manually) to test that
         the captured cookies / session still let the tool fire — exposes
         cookie-staleness or token-expiration issues.

Each run uses a DIFFERENT task variant (different cities + date) so the tool's
slot values change. If the same captured constants serve all three variants
when only slot values differ, the tool is correctly generalised.

Usage:
    uv run python experiments/run_tool_test.py 1
    uv run python experiments/run_tool_test.py 2
    # (some delay — minutes, hours, day) ...
    uv run python experiments/run_tool_test.py 3
    uv run python experiments/run_tool_test.py report     # summarise the three runs

State persisted across invocations:
  results/tool_test_<datetime>/run{1,2,3}/
  morphnet_v2/sites/confirmtkt/tools.json  (written by Run 1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Three task variants — same structure (a train search), different slot values.
# Each task asks for top 5 trains and their per-class availability so the
# captured response carries the full trainList with availability fields.
_TASK_TEMPLATE = (
    "Search trains from {src} ({src_code}) to {dst} ({dst_code}) on {date}. "
    "For the first 5 trains in the search results (in the order they appear), "
    "report each train's name, number, departure time, and per-class seat "
    "availability (SL / 3A / 2A / 1A / CC / EC etc. — list whichever classes "
    "the page actually shows for that train)."
)
TASK_VARIANTS = [
    {"label": "tool_test_run1_mumbai_delhi",   "url": "https://www.confirmtkt.com",
     "task": _TASK_TEMPLATE.format(src="Mumbai",   src_code="CSTM", dst="New Delhi", dst_code="NDLS", date="5 June 2026")},
    {"label": "tool_test_run2_chennai_bangalore", "url": "https://www.confirmtkt.com",
     "task": _TASK_TEMPLATE.format(src="Chennai",  src_code="MAS",  dst="Bangalore", dst_code="SBC",  date="30 May 2026")},
    {"label": "tool_test_run3_pune_howrah",    "url": "https://www.confirmtkt.com",
     "task": _TASK_TEMPLATE.format(src="Pune",     src_code="PUNE", dst="Howrah",    dst_code="HWH",  date="15 June 2026")},
]


STATE_FILE = _REPO_ROOT / "results" / "tool_test_state.json"
TOOLS_JSON_PATH = _REPO_ROOT / "morphnet_v2" / "sites" / "confirmtkt" / "tools.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _save_state(s: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_one(run_idx: int) -> None:
    """Run variant `run_idx` (1-based). After Run 1, invoke tool_builder + aggregate."""
    variant = TASK_VARIANTS[run_idx - 1]
    state = _load_state()
    if "session_dir" not in state:
        state["session_dir"] = f"results/tool_test_{time.strftime('%Y%m%d_%H%M%S')}"
        _save_state(state)
    out_dir = _REPO_ROOT / state["session_dir"] / f"run{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN {run_idx}/3 — {variant['label']}")
    print(f"{'='*60}")
    print(f"  task: {variant['task']}")
    print(f"  output: {out_dir}")
    if run_idx == 1:
        print(f"  registry: EMPTY (discovery run; will build tools at end)")
    else:
        n_tools = 0
        if TOOLS_JSON_PATH.exists():
            try:
                n_tools = len(json.loads(TOOLS_JSON_PATH.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, ValueError):
                pass
        print(f"  registry: {n_tools} tools from {TOOLS_JSON_PATH.relative_to(_REPO_ROOT)}")
    print()

    t0 = time.time()

    from morphnet_v2.session_manager import SessionManager
    async with SessionManager(
        start_url=variant["url"],
        headless=False,
        port=9400 + run_idx,
        results_dir=out_dir,
        task_metadata={
            "task": variant["task"],
            "label": variant["label"],
            "site": "confirmtkt",
        },
        max_steps=8,
        max_turns_per_step=40,
    ) as sm:
        await sm.wait_for_page_ready()
        tree = await sm.run_task(variant["task"])
        wall_seconds = time.time() - t0
        n_tool_invocations = sum(1 for node in tree._nodes.values() if node.kind == "tool")
        n_cu_steps = sum(1 for node in tree._nodes.values() if node.kind == "cu")

        report = {
            "run_idx": run_idx,
            "variant_label": variant["label"],
            "wall_seconds": round(wall_seconds, 1),
            "n_planner_turns": tree.step_count + 1,  # +1 for terminal turn
            "n_cu_steps": n_cu_steps,
            "n_tool_invocations": n_tool_invocations,
            "task_exit": tree.task_exit,
            "success": tree.success,
            "final_answer": tree.final_answer,
            "final_url": tree.final_url,
            "tokens_in": tree.total_input_tokens,
            "tokens_out": tree.total_output_tokens,
        }
        print(f"\n=== Run {run_idx} report ===")
        for k, v in report.items():
            print(f"  {k}: {v}")
        # persist report
        (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        # Find the task's notes dir (sm wrote it to results_dir/{ts}-{site}/)
        task_notes_dir = next((p for p in out_dir.iterdir() if p.is_dir() and "confirmtkt" in p.name), None)
        if task_notes_dir is None:
            print("WARN: could not locate task notes dir")
            return

        # On Run 1: build candidates + aggregate to site
        if run_idx == 1:
            print(f"\n=== Synthesis (Run 1 only) ===")
            from morphnet_v2 import tool_builder
            candidates = await tool_builder.build_candidates(task_notes_dir, sm)
            kept = [c for c in candidates if c.verdict == "keep"]
            print(f"  candidates: {len(candidates)}; kept: {len(kept)}")
            for c in kept:
                print(f"    [{c.tool_id}] {c.capability_statement[:80]}")
            # Aggregate
            site_dir = task_notes_dir.parent
            out_tools_json = tool_builder.build_site_registry(site_dir, "confirmtkt")
            # Mirror into the project's morphnet_v2/sites/confirmtkt/tools.json for the next run
            project_path = TOOLS_JSON_PATH
            project_path.parent.mkdir(parents=True, exist_ok=True)
            project_path.write_text(out_tools_json.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  wrote tools.json: {project_path.relative_to(_REPO_ROOT)}")


def report_summary() -> None:
    """Cross-run summary."""
    state = _load_state()
    sess = _REPO_ROOT / state.get("session_dir", "")
    if not sess.exists():
        print("no session state found")
        return
    print(f"\n=== 3× tool-test summary — {sess.name} ===\n")
    rows = []
    for i in (1, 2, 3):
        rp = sess / f"run{i}" / "run_report.json"
        if not rp.exists():
            continue
        rows.append(json.loads(rp.read_text(encoding="utf-8")))
    if not rows:
        print("no run reports yet")
        return
    keys = ["run_idx", "wall_seconds", "n_cu_steps", "n_tool_invocations", "task_exit", "success", "final_answer"]
    fmt = "{:>4s}  {:>8s}  {:>8s}  {:>8s}  {:>12s}  {:>7s}  {}"
    print(fmt.format("run", "secs", "cu_steps", "tools", "exit", "success", "answer"))
    print("-" * 100)
    for r in rows:
        ans = (r.get("final_answer") or "")[:60].replace("\n", " ")
        print(fmt.format(
            str(r["run_idx"]),
            str(r["wall_seconds"]),
            str(r["n_cu_steps"]),
            str(r["n_tool_invocations"]),
            str(r["task_exit"]),
            str(r["success"]),
            ans,
        ))

    # Expected behaviour:
    print()
    if len(rows) >= 2:
        r1, r2 = rows[0], rows[1]
        if r2["n_tool_invocations"] > 0:
            print(f"✓ Run 2 invoked {r2['n_tool_invocations']} tool(s) (tools registered after Run 1)")
        else:
            print(f"⚠ Run 2 didn't invoke any tools — planner may not be picking invoke_<tool>")
        if r2["wall_seconds"] < r1["wall_seconds"]:
            speedup = r1["wall_seconds"] / max(1, r2["wall_seconds"])
            print(f"✓ Run 2 was {speedup:.1f}× faster than Run 1 (CU → tools)")
        else:
            print(f"⚠ Run 2 ({r2['wall_seconds']}s) not faster than Run 1 ({r1['wall_seconds']}s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["1", "2", "3", "report"])
    args = p.parse_args()
    if args.cmd == "report":
        report_summary()
        return
    asyncio.run(run_one(int(args.cmd)))


if __name__ == "__main__":
    main()
