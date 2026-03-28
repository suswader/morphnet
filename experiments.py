#!/usr/bin/env python3
"""Experiment harness for WebArena Verified benchmark.

Orchestrates 4 runs for academic evaluation:
  Run 1: baseline_learning  — full benchmark, MCPs build from scratch
  Run 4: hard_baseline      — hard tasks, clean state (SOTA comparison)
  Run 2: transfer_learning  — full benchmark, MCPs retained from Run 1
  Run 3: hard_with_mcps     — hard tasks, richest MCP library

Execution order: 1 → 4 → 2 → 3

Usage:
  python experiments.py                              # all 4 runs
  python experiments.py --dry-run                    # print plan only
  python experiments.py --runs 1,4 --task-limit 5    # mini test
  python experiments.py --resume-from 2              # crash recovery
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

# ── Paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
EVOLVING_DIR = PROJECT_ROOT / "evolving_interface"
MCPS_DIR = EVOLVING_DIR / "mcps"
INTERNAL_RESULTS_DIR = EVOLVING_DIR / "results"
TASKS_DIR = PROJECT_ROOT / "tasks"
EXPERIMENT_DIR = PROJECT_ROOT / "experiment_results"

SITES = ["shopping", "shopping_admin", "reddit", "gitlab"]
HEALTH_PORTS = {
    "shopping": 7770,
    "shopping_admin": 7780,
    "reddit": 9999,
    "gitlab": 8023,
}

log = logging.getLogger("experiments")


# ── Run Configuration ──────────────────────────────────────────────

@dataclass
class RunConfig:
    """Definition of a single experiment run."""

    number: int
    name: str
    task_file: str          # without .json
    seed: int
    reset_mcps: bool
    description: str


RUNS: dict[int, RunConfig] = {
    1: RunConfig(
        number=1,
        name="baseline_learning",
        task_file="verified-4site",
        seed=42,
        reset_mcps=True,
        description=(
            "Full benchmark with MCPs building from scratch. "
            "Establishes baseline CU performance and learning curve."
        ),
    ),
    4: RunConfig(
        number=4,
        name="hard_baseline",
        task_file="hard-4site",
        seed=42,
        reset_mcps=True,
        description=(
            "Hard tasks with no MCPs. Pure computer use baseline "
            "for comparison against published SOTA numbers."
        ),
    ),
    2: RunConfig(
        number=2,
        name="transfer_learning",
        task_file="verified-4site",
        seed=73,
        reset_mcps=False,
        description=(
            "Full benchmark with MCPs retained from Run 1. "
            "Measures improvement from accumulated tool library."
        ),
    ),
    3: RunConfig(
        number=3,
        name="hard_with_mcps",
        task_file="hard-4site",
        seed=42,
        reset_mcps=False,
        description=(
            "Hard tasks with full MCP library from Runs 1+2. "
            "Peak performance; compared against Run 4 (same seed)."
        ),
    ),
}

EXECUTION_ORDER = [1, 4, 2, 3]


# ── Logging ────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> None:
    """Configure dual logging: file (DEBUG) + console (INFO)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment_log.txt"

    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)


# ── Utility Functions ──────────────────────────────────────────────

def fmt_tokens(n: int) -> str:
    """Format token count: 1234567 → '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_time(seconds: float) -> str:
    """Format seconds: 5432 → '1h 30m'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {int(seconds % 60):02d}s"


def git_commit_hash() -> str:
    """Return short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_task_count(task_file: str) -> int:
    """Count tasks in a task JSON file."""
    path = TASKS_DIR / f"{task_file}.json"
    with open(path) as f:
        return len(json.load(f))


def model_versions() -> dict[str, str]:
    """Read model identifiers from config (lazy import)."""
    try:
        from evolving_interface import config as cfg
        return {
            "computer_use": cfg.GEMINI_COMPUTER_USE_MODEL,
            "reasoning": cfg.GEMINI_REASONING_MODEL,
            "fast": cfg.GEMINI_FAST_MODEL,
        }
    except Exception:
        return {"computer_use": "unknown", "reasoning": "unknown", "fast": "unknown"}


# ── Health Check ───────────────────────────────────────────────────

def health_check() -> bool:
    """Verify all 4 WebArena containers respond on their ports."""
    log.info("Checking WebArena containers...")
    all_ok = True
    for site, port in HEALTH_PORTS.items():
        try:
            resp = urlopen(f"http://localhost:{port}", timeout=10)
            log.info(f"  {site:20s} :{port}  HTTP {resp.getcode()}  OK")
        except URLError:
            log.error(f"  {site:20s} :{port}  FAILED (connection refused)")
            all_ok = False
        except Exception as exc:
            log.error(f"  {site:20s} :{port}  FAILED ({exc})")
            all_ok = False
    return all_ok


# ── MCP State Management ──────────────────────────────────────────

def count_mcps() -> dict[str, int]:
    """Count MCP tools per site from tools.json files."""
    counts: dict[str, int] = {}
    for site in SITES:
        tools_file = MCPS_DIR / site / "tools.json"
        if tools_file.exists():
            try:
                with open(tools_file) as f:
                    data = json.load(f)
                counts[site] = len(data)
            except (json.JSONDecodeError, OSError):
                counts[site] = 0
        else:
            counts[site] = 0
    return counts


def reset_mcps() -> None:
    """Delete all MCP tool libraries (all sites)."""
    for site in SITES:
        site_dir = MCPS_DIR / site
        if site_dir.exists():
            shutil.rmtree(site_dir)
    log.info("MCP libraries cleared for all sites")


def backup_mcps(label: str) -> Path:
    """Copy current MCP libraries to a backup directory."""
    backup_dir = EXPERIMENT_DIR / ".mcp_backups" / label
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    for site in SITES:
        src = MCPS_DIR / site
        if src.exists():
            shutil.copytree(src, backup_dir / site)

    counts = count_mcps()
    total = sum(counts.values())
    log.info(f"MCP backup '{label}': {total} tools ({counts})")
    return backup_dir


def restore_mcps(label: str) -> None:
    """Restore MCP libraries from a backup, replacing current state."""
    backup_dir = EXPERIMENT_DIR / ".mcp_backups" / label
    if not backup_dir.exists():
        log.warning(f"MCP backup '{label}' not found — skipping restore")
        return

    # Clear current state
    for site in SITES:
        site_dir = MCPS_DIR / site
        if site_dir.exists():
            shutil.rmtree(site_dir)

    # Copy backup
    for site in SITES:
        src = backup_dir / site
        if src.exists():
            shutil.copytree(src, MCPS_DIR / site)

    counts = count_mcps()
    total = sum(counts.values())
    log.info(f"MCP libraries restored from '{label}': {total} tools ({counts})")


# ── Task ID Loading (for --task-limit) ─────────────────────────────

def load_task_ids(task_file: str, seed: int, limit: int) -> list[int]:
    """Load task IDs, shuffle with seed, return first `limit` IDs."""
    path = TASKS_DIR / f"{task_file}.json"
    with open(path) as f:
        tasks = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return [t["task_id"] for t in tasks[:limit]]


# ── Run Execution ──────────────────────────────────────────────────

def build_run_command(
    run_cfg: RunConfig,
    task_limit: int | None,
) -> list[str]:
    """Build the subprocess command for run.py."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "run.py"),
        "--tasks", run_cfg.task_file,
        "--seed", str(run_cfg.seed),
        "--run-name", run_cfg.name,
        "--headless",
    ]
    # MCP reset is handled by experiments.py — no --reset-mcps needed.
    # This avoids double-reset and keeps state management in one place.

    if task_limit is not None:
        ids = load_task_ids(run_cfg.task_file, run_cfg.seed, task_limit)
        cmd.extend(["--task-ids", ",".join(str(i) for i in ids)])

    return cmd


def execute_run(
    run_cfg: RunConfig,
    task_limit: int | None,
) -> int:
    """Execute run.py as a subprocess, streaming output in real time.
    Returns the process exit code."""
    cmd = build_run_command(run_cfg, task_limit)
    log.info(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line)
        log.debug(line)

    proc.wait()
    return proc.returncode


# ── Results Parsing & Metrics ──────────────────────────────────────

def find_results_file(run_name: str) -> Path | None:
    """Find the most recent JSONL results file matching a run name."""
    tag = run_name.replace(" ", "_").lower()
    matches = sorted(
        INTERNAL_RESULTS_DIR.glob(f"{tag}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def parse_results(jsonl_path: Path) -> list[dict]:
    """Parse a JSONL results file into a list of result dicts."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute all evaluation metrics from parsed results."""
    from evolving_interface.eval_harness import (
        compute_sr_tmpl,
        compute_per_site_sr,
        compute_per_type_sr,
    )

    sr, ci, n_tmpl = compute_sr_tmpl(results)
    per_site = compute_per_site_sr(results)
    per_type = compute_per_type_sr(results)

    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_time = sum(r.get("time_seconds", 0) for r in results)
    n_passed = sum(1 for r in results if r.get("passed"))

    return {
        "task_count": len(results),
        "passed": n_passed,
        "sr_tmpl": round(sr, 2),
        "sr_tmpl_ci": round(ci, 2),
        "sr_tmpl_ci_low": round(max(0, sr - ci), 2),
        "sr_tmpl_ci_high": round(min(100, sr + ci), 2),
        "n_templates": n_tmpl,
        "total_tokens": total_tokens,
        "total_time_seconds": round(total_time, 1),
        "per_site_sr": {
            site: {"sr": round(s, 2), "ci": round(c, 2), "templates": t}
            for site, (s, c, t) in per_site.items()
        },
        "per_type_sr": {
            tt: {"passed": p, "total": tot, "pct": round(pct, 2)}
            for tt, (p, tot, pct) in per_type.items()
        },
    }


# ── Manifest ───────────────────────────────────────────────────────

def save_manifest(
    run_cfg: RunConfig,
    metrics: dict[str, Any],
    mcp_before: dict[str, int],
    mcp_after: dict[str, int],
    start_time: str,
    end_time: str,
    wall_seconds: float,
    jsonl_path: Path,
) -> Path:
    """Save a JSON manifest with full run metadata and metrics."""
    run_dir = EXPERIMENT_DIR / run_cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_cfg.name,
        "run_number": run_cfg.number,
        "task_file": f"{run_cfg.task_file}.json",
        "task_count": metrics["task_count"],
        "seed": run_cfg.seed,
        "mcp_reset": run_cfg.reset_mcps,
        "mcp_tools_before": mcp_before,
        "mcp_tools_after": mcp_after,
        "start_time": start_time,
        "end_time": end_time,
        "wall_clock_seconds": round(wall_seconds, 1),
        "total_tokens": metrics["total_tokens"],
        "sr_tmpl": metrics["sr_tmpl"],
        "sr_tmpl_ci_low": metrics["sr_tmpl_ci_low"],
        "sr_tmpl_ci_high": metrics["sr_tmpl_ci_high"],
        "per_site_sr": metrics["per_site_sr"],
        "per_type_sr": metrics["per_type_sr"],
        "git_commit": git_commit_hash(),
        "python_version": sys.version.split()[0],
        "model_versions": model_versions(),
    }

    # Save manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Copy results + summary into the run directory
    shutil.copy2(jsonl_path, run_dir / "results.jsonl")
    summary_path = jsonl_path.parent / (jsonl_path.stem + "_summary.txt")
    if summary_path.exists():
        shutil.copy2(summary_path, run_dir / "summary.txt")

    log.info(f"Manifest saved: {manifest_path}")
    return manifest_path


# ── Learning Curve (Run 1) ─────────────────────────────────────────

def generate_learning_curve(
    results: list[dict],
    output_dir: Path,
    window: int = 20,
) -> None:
    """Generate rolling-window learning curve data from Run 1.

    For each task in execution order, records performance metrics
    and computes rolling averages over a sliding window. The key
    insight: mcp_tools_available grows as the agent discovers tools,
    and rolling_sr / rolling_tokens should improve in response.
    """
    curve: list[dict[str, Any]] = []
    cumulative_tools = 0

    for i, r in enumerate(results):
        new_tools = len(r.get("tools_discovered", []))
        entry: dict[str, Any] = {
            "task_index": i,
            "task_id": r["task_id"],
            "site": r["site"],
            "strategy": r.get("strategy", "unknown"),
            "tokens": r.get("tokens", 0),
            "time_seconds": r.get("time_seconds", 0),
            "mcp_tools_available": cumulative_tools,
            "new_tools_discovered": new_tools,
            "success": r.get("passed", False),
        }
        cumulative_tools += new_tools
        entry["cumulative_tools"] = cumulative_tools
        curve.append(entry)

    # Rolling windows
    for i, entry in enumerate(curve):
        start = max(0, i - window + 1)
        w = curve[start:i + 1]
        n = len(w)
        entry["rolling_sr"] = round(
            sum(1 for e in w if e["success"]) / n * 100, 2,
        )
        entry["rolling_tokens"] = round(
            sum(e["tokens"] for e in w) / n,
        )
        entry["rolling_time"] = round(
            sum(e["time_seconds"] for e in w) / n, 2,
        )

    path = output_dir / "learning_curve.json"
    with open(path, "w") as f:
        json.dump(curve, f, indent=2)
    log.info(f"Learning curve: {path} ({len(curve)} data points)")


# ── Tool Inventory ─────────────────────────────────────────────────

def generate_tool_inventory(output_dir: Path) -> None:
    """Catalog all discovered MCP tools across all sites."""
    inventory: list[dict[str, Any]] = []

    for site in SITES:
        tools_file = MCPS_DIR / site / "tools.json"
        if not tools_file.exists():
            continue
        try:
            with open(tools_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if isinstance(data, dict):
            for name, tool in data.items():
                http = tool.get("_http", {})
                meta = tool.get("_meta", {})
                inventory.append({
                    "name": name,
                    "site": site,
                    "description": tool.get("description", ""),
                    "http_method": http.get("method", ""),
                    "url_template": http.get("url_template", ""),
                    "status": meta.get("status", "unknown"),
                    "success_count": meta.get("success_count", 0),
                    "failure_count": meta.get("failure_count", 0),
                    "discovered_from_task": meta.get("discovered_from_task", ""),
                })

    path = output_dir / "tool_inventory.json"
    with open(path, "w") as f:
        json.dump(inventory, f, indent=2)
    log.info(f"Tool inventory: {path} ({len(inventory)} tools)")


# ── Comparison Report ──────────────────────────────────────────────

def generate_comparison_report(
    manifests: dict[int, dict],
    output_dir: Path,
) -> None:
    """Generate and print the final cross-run comparison report."""
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 78)
    lines.append("  EXPERIMENT COMPARISON REPORT")
    lines.append("=" * 78)
    lines.append("")

    # ── Summary table ──
    available = [n for n in EXECUTION_ORDER if n in manifests]
    header = (
        f"  {'Run':<25s} {'Tasks':>6s} {'SR_tmpl':>8s} "
        f"{'95% CI':>14s} {'Tokens':>10s} {'Time':>10s}"
    )
    sep = "  " + "-" * 75
    lines.append(header)
    lines.append(sep)

    for run_num in available:
        m = manifests[run_num]
        name = f"{run_num}: {m['run_name']}"
        sr = f"{m['sr_tmpl']:.1f}%"
        ci = f"[{m['sr_tmpl_ci_low']:.1f}, {m['sr_tmpl_ci_high']:.1f}]"
        tokens = fmt_tokens(m["total_tokens"])
        time_s = fmt_time(m["wall_clock_seconds"])
        lines.append(
            f"  {name:<25s} {m['task_count']:>6d} {sr:>8s} "
            f"{ci:>14s} {tokens:>10s} {time_s:>10s}"
        )

    lines.append(sep)
    lines.append("")

    # ── Key deltas ──
    if 1 in manifests and 2 in manifests:
        m1, m2 = manifests[1], manifests[2]
        sr_d = m2["sr_tmpl"] - m1["sr_tmpl"]
        t1 = m1["total_tokens"] / max(m1["task_count"], 1)
        t2 = m2["total_tokens"] / max(m2["task_count"], 1)
        tok_d = (t2 - t1) / max(t1, 1) * 100
        s1 = m1["wall_clock_seconds"] / max(m1["task_count"], 1)
        s2 = m2["wall_clock_seconds"] / max(m2["task_count"], 1)
        time_d = (s2 - s1) / max(s1, 1) * 100

        lines.append("  Transfer learning (Run 2 vs Run 1):")
        lines.append(f"    SR:     {sr_d:+.1f} pp")
        lines.append(f"    Tokens: {tok_d:+.1f}% per task")
        lines.append(f"    Time:   {time_d:+.1f}% per task")
        lines.append("")

    if 3 in manifests and 4 in manifests:
        m4, m3 = manifests[4], manifests[3]
        sr_d = m3["sr_tmpl"] - m4["sr_tmpl"]
        t4 = m4["total_tokens"] / max(m4["task_count"], 1)
        t3 = m3["total_tokens"] / max(m3["task_count"], 1)
        tok_d = (t3 - t4) / max(t4, 1) * 100
        s4 = m4["wall_clock_seconds"] / max(m4["task_count"], 1)
        s3 = m3["wall_clock_seconds"] / max(m3["task_count"], 1)
        time_d = (s3 - s4) / max(s4, 1) * 100

        lines.append("  Hard task improvement (Run 3 vs Run 4):")
        lines.append(f"    SR:     {sr_d:+.1f} pp")
        lines.append(f"    Tokens: {tok_d:+.1f}% per task")
        lines.append(f"    Time:   {time_d:+.1f}% per task")
        lines.append("")

    # ── Per-site breakdown ──
    all_sites = sorted({
        s for m in manifests.values() for s in m.get("per_site_sr", {})
    })
    if all_sites and available:
        col_hdrs = "".join(f" {'Run ' + str(n):>12s}" for n in available)
        lines.append(f"  {'Site':<20s}{col_hdrs}")
        lines.append("  " + "-" * (20 + 13 * len(available)))
        for site in all_sites:
            cols = ""
            for n in available:
                sr = manifests[n].get("per_site_sr", {}).get(
                    site, {},
                ).get("sr", 0)
                cols += f" {sr:>11.1f}%"
            lines.append(f"  {site:<20s}{cols}")
        lines.append("")

    # ── MCP growth ──
    lines.append("  MCP tool growth:")
    for n in available:
        m = manifests[n]
        before = sum(m.get("mcp_tools_before", {}).values())
        after = sum(m.get("mcp_tools_after", {}).values())
        delta = after - before
        lines.append(
            f"    Run {n}: {before} -> {after} ({delta:+d} tools)"
        )
    lines.append("")

    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)

    path = output_dir / "experiment_report.txt"
    path.write_text(report)
    log.info(f"Report saved: {path}")


# ── Dry Run ────────────────────────────────────────────────────────

def print_dry_run(
    run_order: list[int],
    task_limit: int | None,
) -> None:
    """Print execution plan without running anything."""
    print("\n" + "=" * 60)
    print("  DRY RUN — Execution Plan")
    print("=" * 60)

    total_tasks = 0
    for run_num in run_order:
        cfg = RUNS[run_num]
        n = load_task_count(cfg.task_file)
        if task_limit:
            n = min(n, task_limit)
        total_tasks += n
        mcp_state = "RESET" if cfg.reset_mcps else "RETAIN"

        print(f"\n  Run {run_num}: {cfg.name}")
        print(f"    Tasks:    {cfg.task_file}.json ({n} tasks)")
        print(f"    MCP:      {mcp_state}")
        print(f"    Seed:     {cfg.seed}")
        print(f"    Purpose:  {cfg.description}")

        cmd = build_run_command(cfg, task_limit)
        print(f"    Command:  {' '.join(cmd)}")

        if run_num == 1:
            print("    >> After: backup MCPs as 'post_run1'")
        if run_num == 2 and 4 in set(run_order):
            print("    >> Before: restore MCPs from 'post_run1' backup")

    est = total_tasks * 30
    print(f"\n  Total tasks: {total_tasks}")
    print(f"  Estimated time: ~{fmt_time(est)} (at ~30s/task avg)")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment harness for WebArena Verified benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs", default="1,4,2,3",
        help="Comma-separated run numbers (default: 1,4,2,3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print execution plan without running",
    )
    parser.add_argument(
        "--resume-from", type=int, default=None, metavar="N",
        help="Skip runs before N (for crash recovery)",
    )
    parser.add_argument(
        "--task-limit", type=int, default=None, metavar="N",
        help="Limit tasks per run (for testing)",
    )
    args = parser.parse_args()

    run_order = [int(x) for x in args.runs.split(",")]
    for n in run_order:
        if n not in RUNS:
            print(f"Error: unknown run {n}. Valid: {sorted(RUNS)}")
            sys.exit(1)

    # Dry run — no logging, no health check, just print
    if args.dry_run:
        print_dry_run(run_order, args.task_limit)
        return

    # ── Setup ──
    setup_logging(EXPERIMENT_DIR)
    log.info(f"Experiment started. Runs: {run_order}")
    if args.task_limit:
        log.info(f"Task limit: {args.task_limit} per run")

    if not health_check():
        log.error("Container health check FAILED — aborting.")
        sys.exit(1)
    log.info("")

    # ── Execute runs ──
    manifests: dict[int, dict] = {}
    executed: set[int] = set()
    experiment_t0 = time.monotonic()

    for run_num in run_order:
        # Resume logic — skip earlier runs, loading existing manifests
        if args.resume_from and run_num < args.resume_from:
            log.info(f"Skipping Run {run_num} (resuming from {args.resume_from})")
            manifest_path = EXPERIMENT_DIR / RUNS[run_num].name / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifests[run_num] = json.load(f)
                executed.add(run_num)
            continue

        cfg = RUNS[run_num]
        n_tasks = load_task_count(cfg.task_file)
        if args.task_limit:
            n_tasks = min(n_tasks, args.task_limit)

        # ── MCP state transitions ──
        # Before Run 2: restore Run 1's MCPs if Run 4 cleared them
        if run_num == 2 and 4 in executed:
            log.info("Restoring MCP libraries from Run 1 backup...")
            restore_mcps("post_run1")

        # Reset if this run requires it
        if cfg.reset_mcps:
            reset_mcps()

        mcp_before = count_mcps()
        mcp_label = ", ".join(f"{s}={c}" for s, c in mcp_before.items())
        mcp_state = "RESET" if cfg.reset_mcps else "RETAIN"

        # ── Run header ──
        start_ts = now_iso()
        log.info("=" * 60)
        log.info(f"  EXPERIMENT Run {run_num}: {cfg.name}")
        log.info(f"  Tasks: {cfg.task_file}.json ({n_tasks} tasks)")
        log.info(f"  MCP state: {mcp_state} ({mcp_label})")
        log.info(f"  Seed: {cfg.seed}")
        log.info(f"  Started: {start_ts}")
        log.info("=" * 60)

        # ── Execute ──
        wall_t0 = time.monotonic()
        exit_code = execute_run(cfg, args.task_limit)
        wall_seconds = time.monotonic() - wall_t0
        end_ts = now_iso()

        executed.add(run_num)

        if exit_code != 0:
            log.error(
                f"Run {run_num} exited with code {exit_code}. "
                "Attempting to parse partial results..."
            )

        # ── Backup MCPs after Run 1 ──
        if run_num == 1:
            backup_mcps("post_run1")

        mcp_after = count_mcps()

        # ── Parse results ──
        jsonl_path = find_results_file(cfg.name)
        if jsonl_path is None:
            log.error(f"No results file found for '{cfg.name}' — skipping.")
            continue

        results = parse_results(jsonl_path)
        if not results:
            log.error(f"Empty results file for '{cfg.name}' — skipping.")
            continue

        metrics = compute_metrics(results)

        # ── Save manifest ──
        save_manifest(
            cfg, metrics, mcp_before, mcp_after,
            start_ts, end_ts, wall_seconds, jsonl_path,
        )
        manifests[run_num] = {
            "run_name": cfg.name,
            "run_number": cfg.number,
            "task_file": f"{cfg.task_file}.json",
            "task_count": metrics["task_count"],
            "seed": cfg.seed,
            "mcp_reset": cfg.reset_mcps,
            "mcp_tools_before": mcp_before,
            "mcp_tools_after": mcp_after,
            "start_time": start_ts,
            "end_time": end_ts,
            "wall_clock_seconds": round(wall_seconds, 1),
            "total_tokens": metrics["total_tokens"],
            "sr_tmpl": metrics["sr_tmpl"],
            "sr_tmpl_ci_low": metrics["sr_tmpl_ci_low"],
            "sr_tmpl_ci_high": metrics["sr_tmpl_ci_high"],
            "per_site_sr": metrics["per_site_sr"],
            "per_type_sr": metrics["per_type_sr"],
        }

        # ── Post-run summary ──
        log.info("")
        log.info(f"  Run {run_num} complete ({metrics['task_count']}/{n_tasks} tasks):")
        log.info(f"    SR_tmpl: {metrics['sr_tmpl']:.1f}% +/- {metrics['sr_tmpl_ci']:.1f}%")
        log.info(f"    Passed:  {metrics['passed']}/{metrics['task_count']}")
        log.info(f"    Tokens:  {fmt_tokens(metrics['total_tokens'])}")
        log.info(f"    Time:    {fmt_time(wall_seconds)}")
        mcp_delta = sum(mcp_after.values()) - sum(mcp_before.values())
        log.info(f"    MCPs:    {sum(mcp_before.values())} -> {sum(mcp_after.values())} ({mcp_delta:+d})")

        # ETA for remaining runs
        elapsed = time.monotonic() - experiment_t0
        completed_tasks = sum(manifests[n]["task_count"] for n in manifests)
        remaining_nums = [n for n in run_order if n not in executed]
        if remaining_nums and completed_tasks > 0:
            avg_per_task = elapsed / completed_tasks
            remaining_tasks = sum(
                min(load_task_count(RUNS[n].task_file), args.task_limit)
                if args.task_limit
                else load_task_count(RUNS[n].task_file)
                for n in remaining_nums
            )
            eta = avg_per_task * remaining_tasks
            log.info(f"    ETA remaining: ~{fmt_time(eta)}")
        log.info("")

        # Learning curve for Run 1
        if run_num == 1:
            generate_learning_curve(results, EXPERIMENT_DIR)

    # ── Post-experiment outputs ──
    if manifests:
        generate_tool_inventory(EXPERIMENT_DIR)
        generate_comparison_report(manifests, EXPERIMENT_DIR)

    total_time = time.monotonic() - experiment_t0
    log.info(f"All experiments complete. Total time: {fmt_time(total_time)}")


if __name__ == "__main__":
    main()
