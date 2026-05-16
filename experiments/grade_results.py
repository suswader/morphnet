#!/usr/bin/env python3
"""Grade comparison-run results: pass each task's raw run log to Gemini,
ask it to extract the agent's final answer AND grade it against the
expected_answer criteria. No regex. The grader reads raw logs end-to-end.

Usage:
  uv run python experiments/grade_results.py results/comparison_<ts>/
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from morphnet.session_manager import call_gemini_async  # noqa: E402

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_answer": {
            "type": "string",
            "description": "The agent's final answer as it appears in the log. "
                           "Quote verbatim if there's a clear final-answer statement; "
                           "otherwise summarize what the agent communicated. Empty "
                           "string if the agent never produced an answer.",
        },
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect", "no_answer"]},
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
        "missing_facts": {"type": "array", "items": {"type": "string"}},
        "incorrect_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["extracted_answer", "verdict", "score", "reasoning", "missing_facts", "incorrect_facts"],
}

GRADE_SYSTEM = (
    "You grade a browser agent's answer to a task. You receive the task, the "
    "expected_answer (criteria + literal facts), and the FULL raw run log of "
    "the agent (stdout + stderr from the run).\n\n"
    "Step 1: read the log and find what the agent ultimately reported as its "
    "answer. Look for explicit 'Answer:' / 'Final answer:' lines, journey "
    "summaries, or text outputs near 'RUN COMPLETE'. If the agent errored or "
    "never produced an answer, set extracted_answer to '' and verdict to "
    "'no_answer'.\n\n"
    "Step 2: compare the extracted answer against the expected criteria.\n"
    "  - correct: every required fact present and accurate. Score 1.0.\n"
    "  - partial: some facts right, others missing/wrong. Score 0.3–0.7.\n"
    "  - incorrect: most facts wrong or contradictory. Score 0.0–0.2.\n"
    "  - no_answer: no extractable final answer. Score 0.0.\n\n"
    "For dynamic facts (prices, ratings, fares), accept any plausible value "
    "in the stated range. For static facts (dates, places, piece counts), "
    "require near-exact match."
)


async def grade_one(task: dict, log_text: str) -> dict:
    if not log_text.strip():
        return {
            "extracted_answer": "",
            "verdict": "no_answer",
            "score": 0.0,
            "reasoning": "Run log was empty or missing.",
            "missing_facts": [],
            "incorrect_facts": [],
        }
    # Cap log size at ~120KB to stay within Gemini context. Keep tail —
    # final answers are at the end. Don't truncate the middle, keep the last N.
    if len(log_text) > 120_000:
        log_text = "[... earlier log truncated ...]\n" + log_text[-120_000:]
    prompt = (
        f"TASK:\n{task['task']}\n\n"
        f"EXPECTED ANSWER CRITERIA:\n{task['expected_answer']}\n\n"
        f"AGENT RUN LOG:\n{log_text}"
    )
    return await call_gemini_async(
        model="gemini-2.5-flash",
        contents=[prompt],
        response_schema=GRADE_SCHEMA,
        system_instruction=GRADE_SYSTEM,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )


async def main_async(run_dir: Path, parallel: int) -> None:
    tasks = json.loads((run_dir / "tasks_used.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    by_label = {t["label"]: t for t in tasks}

    sem = asyncio.Semaphore(parallel)

    async def grade_entry(entry):
        async with sem:
            label = entry["label"]
            system = entry["system"]
            task = by_label[label]
            log_path = Path(entry["log"])
            log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            print(f"[{system}] {label}  grading...")
            verdict = await grade_one(task, log_text)
            return {
                "label": label,
                "site": task["site"],
                "system": system,
                "duration_s": entry.get("duration_s"),
                "exit_code": entry.get("exit_code"),
                "log": str(log_path),
                **verdict,
            }

    graded = await asyncio.gather(*(grade_entry(e) for e in summary))

    # Sort for stability
    graded.sort(key=lambda r: (r["site"], r["label"], r["system"]))
    (run_dir / "graded.json").write_text(json.dumps(graded, indent=2))

    # Aggregate
    by_system: dict[str, list[dict]] = {}
    for g in graded:
        by_system.setdefault(g["system"], []).append(g)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for system, rows in sorted(by_system.items()):
        n = len(rows)
        c = sum(1 for r in rows if r["verdict"] == "correct")
        p = sum(1 for r in rows if r["verdict"] == "partial")
        avg_score = sum(r["score"] for r in rows) / n if n else 0
        avg_dur = sum(r.get("duration_s") or 0 for r in rows) / n if n else 0
        print(f"{system}: {c}/{n} correct, {p} partial, avg score {avg_score:.2f}, avg dur {avg_dur:.0f}s")

    print("\nBY SITE (correct/partial/total per system):")
    sites = sorted({g["site"] for g in graded})
    systems = sorted(by_system.keys())
    print(f"  {'site':<14}  " + "  ".join(f"{s:<22}" for s in systems))
    for site in sites:
        cells = []
        for system in systems:
            srows = [g for g in graded if g["site"] == site and g["system"] == system]
            n = len(srows)
            c = sum(1 for r in srows if r["verdict"] == "correct")
            p = sum(1 for r in srows if r["verdict"] == "partial")
            cells.append(f"{c}c/{p}p/{n}t".ljust(22))
        print(f"  {site:<14}  " + "  ".join(cells))

    print(f"\nGraded report: {run_dir / 'graded.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--parallel", type=int, default=8, help="Concurrent Gemini grading calls")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr); sys.exit(1)
    asyncio.run(main_async(run_dir, args.parallel))


if __name__ == "__main__":
    main()
