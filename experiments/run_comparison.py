#!/usr/bin/env python3
"""Run comparison_50_tasks.json against morphnet, crawler, or both — with parallelism.

Each system gets its own shuffled task queue. A shared site-lock prevents any
two concurrent workers (across either system) from hitting the same site at
the same time — avoids tripping bot detection from duplicated traffic.

Usage:
  # Single task (smoke test)
  uv run python experiments/run_comparison.py \\
      --label wikipedia_alan_turing_birth_death --system both

  # Full 50 with 3+3 parallelism
  uv run python experiments/run_comparison.py --all --system both \\
      --parallel-morphnet 3 --parallel-crawler 3 --shuffle-seed 42

  # One site, one system
  uv run python experiments/run_comparison.py --site wikipedia --system morphnet
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MORPHNET_ROOT = PROJECT_ROOT
CRAWLER_ROOT = PROJECT_ROOT / "browser-challenge"
TASK_FILE = PROJECT_ROOT / "experiments" / "comparison_50_tasks.json"


# ---------------------------------------------------------------------------
# Env / task loading
# ---------------------------------------------------------------------------

def load_env_file() -> dict:
    env = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def subprocess_env() -> dict:
    env = dict(os.environ)
    env.update(load_env_file())
    return env


def load_tasks(label: str | None = None, site: str | None = None) -> list[dict]:
    tasks = json.loads(TASK_FILE.read_text())
    if label:
        tasks = [t for t in tasks if t["label"] == label]
    if site:
        tasks = [t for t in tasks if t["site"] == site]
    return tasks


# ---------------------------------------------------------------------------
# Shuffle: site-distributed so consecutive tasks differ in site
# ---------------------------------------------------------------------------

def shuffle_site_distributed(tasks: list[dict], seed: int) -> list[dict]:
    """Shuffle tasks so adjacent entries are on different sites when possible.

    Round-robin across sites: shuffle each per-site group, then pop one task
    from each group in a (re-shuffled-each-round) random site order. This
    guarantees the first N entries are on N different sites (when N ≤ #sites),
    which is what we want for N parallel workers.
    """
    rng = random.Random(seed)
    by_site: dict[str, list[dict]] = {}
    for t in tasks:
        by_site.setdefault(t["site"], []).append(t)
    for group in by_site.values():
        rng.shuffle(group)
    site_order = list(by_site.keys())
    rng.shuffle(site_order)
    result: list[dict] = []
    while any(by_site.values()):
        for site in site_order:
            if by_site[site]:
                result.append(by_site[site].pop(0))
        rng.shuffle(site_order)
    return result


# ---------------------------------------------------------------------------
# Coordinator: shared site-lock across both systems
# ---------------------------------------------------------------------------

class Coordinator:
    """Hands out tasks to workers, ensuring no two concurrent runs share a site."""

    def __init__(self, queue_morphnet: list[dict], queue_crawler: list[dict]) -> None:
        self.queues: dict[str, list[dict]] = {
            "morphnet": list(queue_morphnet),
            "crawler": list(queue_crawler),
        }
        self.busy_sites: set[str] = set()
        self.lock = asyncio.Lock()

    async def acquire_next(self, system: str) -> dict | None | bool:
        """Pop next task for `system` whose site is not currently busy.

        Returns:
          - dict: the task to run (caller must release_site when done)
          - None: this system's queue is empty — worker can exit
          - False: queue not empty but all candidate sites are busy — caller
                   should sleep briefly and retry
        """
        async with self.lock:
            q = self.queues[system]
            for i, task in enumerate(q):
                if task["site"] not in self.busy_sites:
                    q.pop(i)
                    self.busy_sites.add(task["site"])
                    return task
            return None if not q else False

    async def release(self, site: str) -> None:
        async with self.lock:
            self.busy_sites.discard(site)


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

async def _run_subprocess(cmd: list[str], cwd: Path, env: dict, log_file: Path, timeout_s: float | None = None) -> dict:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    timed_out = False
    with log_file.open("w") as f:
        f.write(f"CMD: {' '.join(cmd)}\nCWD: {cwd}\nSTART: {datetime.now().isoformat()}\nTIMEOUT_S: {timeout_s}\n\n")
        f.flush()
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), env=env,
            stdout=f, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            if timeout_s is not None:
                rc = await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            else:
                rc = await proc.wait()
        except asyncio.TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                rc = -9  # SIGKILL didn't return cleanly
            f.write(f"\n\n[RUNNER] Killed after timeout of {timeout_s}s\n")
    return {
        "exit_code": rc,
        "duration_s": round(time.time() - start, 1),
        "log": str(log_file),
        "timed_out": timed_out,
    }


async def run_morphnet_task(task: dict, output_dir: Path, port: int, env: dict, timeout_s: float | None) -> dict:
    cmd = [
        "uv", "run", "python", "-m", "morphnet.session_manager",
        "--url", task["url"],
        "--task", task["task"],
        "--headless", "false",
        "--port", str(port),
        "--max-subtasks", "15",
        "--output-dir", str(output_dir),
    ]
    return await _run_subprocess(cmd, MORPHNET_ROOT, env, output_dir / "run.log", timeout_s)


async def run_crawler_task(task: dict, output_dir: Path, env: dict, timeout_s: float | None) -> dict:
    # Route via Gemini's OpenAI-compatible endpoint instead of LiteLLM's
    # native gemini/ provider — the gemini/ adapter mis-orders tool-result
    # turns and Gemini rejects them with INVALID_ARGUMENT. The OpenAI-compat
    # endpoint accepts crawler's tool-call sequence as-is.
    crawler_env = dict(env)
    if "GEMINI_API_KEY" in crawler_env:
        crawler_env["OPENAI_API_KEY"] = crawler_env["GEMINI_API_KEY"]
        crawler_env["OPENAI_API_BASE"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    cmd = [
        "uv", "run", "python", "-m", "crawler.main",
        "--url", task["url"],
        "--goal", task["task"],
        "--agent-mode", "raw",
        "--raw-model", "openai/gemini-2.5-flash",
        "--headed",
        "--max-pages", "10",
    ]
    return await _run_subprocess(cmd, CRAWLER_ROOT, crawler_env, output_dir / "run.log", timeout_s)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

async def worker(
    system: str,
    worker_idx: int,
    port_base: int,
    coord: Coordinator,
    out_root: Path,
    results: list[dict],
    results_lock: asyncio.Lock,
    env: dict,
    timeout_s: float | None,
) -> None:
    tag = f"[{system}#{worker_idx}]"
    while True:
        task = await coord.acquire_next(system)
        if task is None:
            print(f"{tag} queue empty — exiting")
            return
        if task is False:
            await asyncio.sleep(2)
            continue
        try:
            out_dir = out_root / system / task["label"]
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"{tag} START {task['label']} ({task['site']})")
            if system == "morphnet":
                r = await run_morphnet_task(task, out_dir, port_base + worker_idx, env, timeout_s)
            else:
                r = await run_crawler_task(task, out_dir, env, timeout_s)
            timeout_marker = " [TIMEOUT]" if r.get("timed_out") else ""
            print(f"{tag} DONE  {task['label']}  exit={r['exit_code']}  {r['duration_s']}s{timeout_marker}")
            async with results_lock:
                results.append({
                    "label": task["label"],
                    "site": task["site"],
                    "system": system,
                    **r,
                })
                (out_root / "summary.json").write_text(json.dumps(results, indent=2))
        finally:
            await coord.release(task["site"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    tasks = load_tasks(label=args.label, site=args.site)
    if not tasks:
        print("No tasks matched.", file=sys.stderr); sys.exit(1)

    # Two independent shuffles — same seed family, different offsets
    queue_morphnet = shuffle_site_distributed(tasks, args.shuffle_seed)
    queue_crawler = shuffle_site_distributed(tasks, args.shuffle_seed + 1000)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output) if args.output else PROJECT_ROOT / "results" / f"comparison_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tasks_used.json").write_text(json.dumps(tasks, indent=2))
    (out_root / "queue_morphnet.json").write_text(
        json.dumps([{"label": t["label"], "site": t["site"]} for t in queue_morphnet], indent=2)
    )
    (out_root / "queue_crawler.json").write_text(
        json.dumps([{"label": t["label"], "site": t["site"]} for t in queue_crawler], indent=2)
    )

    print(f"Tasks: {len(tasks)}  System: {args.system}  Output: {out_root}")
    print(f"Parallelism: morphnet={args.parallel_morphnet}  crawler={args.parallel_crawler}")
    print(f"Shuffle seed: {args.shuffle_seed}")
    print()

    coord = Coordinator(
        queue_morphnet if args.system in ("morphnet", "both") else [],
        queue_crawler if args.system in ("crawler", "both") else [],
    )
    results: list[dict] = []
    results_lock = asyncio.Lock()
    env = subprocess_env()

    workers: list = []
    if args.system in ("morphnet", "both"):
        for i in range(args.parallel_morphnet):
            workers.append(worker("morphnet", i, args.start_port, coord, out_root, results, results_lock, env, args.timeout_s))
    if args.system in ("crawler", "both"):
        for i in range(args.parallel_crawler):
            workers.append(worker("crawler", i, 0, coord, out_root, results, results_lock, env, args.timeout_s))

    await asyncio.gather(*workers)
    print(f"\nDone. Summary: {out_root / 'summary.json'}  ({len(results)} task runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="Run a specific task by label")
    parser.add_argument("--site", help="Filter to one site (e.g. wikipedia)")
    parser.add_argument("--all", action="store_true", help="Run all 50 tasks")
    parser.add_argument("--system", choices=["morphnet", "crawler", "both"], default="both")
    parser.add_argument("--output", default=None, help="Output root dir (auto-timestamped if omitted)")
    parser.add_argument("--start-port", type=int, default=9301, help="Base Chrome CDP port for morphnet workers")
    parser.add_argument("--parallel-morphnet", type=int, default=1, help="Concurrent morphnet workers")
    parser.add_argument("--parallel-crawler", type=int, default=1, help="Concurrent crawler workers")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="Per-task timeout in seconds (kills the subprocess). None = no timeout.")
    args = parser.parse_args()

    if not (args.label or args.site or args.all):
        parser.error("Pass one of: --label, --site, --all")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
