"""Gemini-driven runner for browser-harness.

Mirrors morphnet_v2.session_manager's CLI (--url, --task, --headless, --port,
--results-dir, --max-steps) so the two stacks can be A/B compared on the same
task. Browser-harness itself is a thin CDP harness with no built-in LLM loop;
this script supplies the loop: screenshot + page_info → Gemini → Python code
snippet → exec via `browser-harness <<PY ... PY`.

Usage:
    uv run python -m experiments.browser_harness_runner \
        --url https://www.swiggy.com \
        --task "add a dominos pizza to my cart and take me to the checkout page. \
I want peppy paneer pizza. I live in kothrud pune." \
        --headless false
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types


# ─────────────────────────────────────────────────────────────────
# Configuration — kept in lock-step with morphnet_v2.session_manager
# so the comparison is fair. If morphnet changes its Chrome path or
# default model, update both.
# ─────────────────────────────────────────────────────────────────

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_FLAGS_BASE = [
    "--remote-allow-origins=*",
    "--no-first-run",
    "--no-default-browser-check",
    "--use-gl=angle",
    "--use-angle=default",
    "--disable-features=IsolateOrigins,site-per-process",
]
HEADLESS_UA_OVERRIDE = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)

# Same flash model morphnet's computer_use uses for action selection.
GEMINI_MODEL = "gemini-3-flash-preview"
THINKING_BUDGET = 4096
MAX_OUTPUT_TOKENS = 8192

BROWSER_HARNESS_BIN = shutil.which("browser-harness") or os.path.expanduser(
    "~/.local/bin/browser-harness"
)

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set. Add to .env.")
_gemini = genai.Client(api_key=_api_key)


# ─────────────────────────────────────────────────────────────────
# Response schema — every Gemini call uses structured output.
# Mirrors the morphnet CU schema: reasoning + evidence_sources +
# confidence are required so the trace stays auditable.
# ─────────────────────────────────────────────────────────────────

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": (
                "Why this action now. Reference what you saw in the screenshot "
                "and page_info — not generic plans."
            ),
        },
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which inputs informed this turn (e.g. screenshot, page_info.url).",
        },
        "confidence": {"type": "number"},
        "action_code": {
            "type": "string",
            "description": (
                "Python snippet that runs inside `browser-harness <<PY ... PY`. "
                "Pre-imported helpers: goto_url, new_tab, click_at_xy(x,y), "
                "type_text(s), fill_input(sel,text), press_key(k), scroll(x,y,dy), "
                "capture_screenshot(), page_info(), js(expr), wait(sec), "
                "wait_for_load(), wait_for_element(sel,timeout=10), "
                "wait_for_network_idle(), list_tabs(), switch_tab(tid), "
                "ensure_real_tab(), upload_file(sel,path), cdp(method,**params). "
                "Use a print() at the end if you want to surface a value back to "
                "this driver. Empty string when done=True."
            ),
        },
        "done": {
            "type": "boolean",
            "description": "True only when the task is fully complete or unrecoverably stuck.",
        },
        "final_answer": {
            "type": "string",
            "description": "User-facing summary of the outcome. Only set when done=True.",
        },
    },
    "required": [
        "reasoning",
        "evidence_sources",
        "confidence",
        "action_code",
        "done",
        "final_answer",
    ],
}


SYSTEM_INSTRUCTION = """\
You are driving a Chrome browser through the `browser-harness` CDP runtime to
complete a user task. Each turn you receive:

  • the task,
  • the latest `page_info()` (url, title, viewport, scroll),
  • a screenshot of the current viewport,
  • a transcript of your previous turns (reasoning + the code you ran + harness stdout/stderr).

You respond with one structured action that the driver pipes verbatim into:

    browser-harness <<'PY'
    <your action_code>
    PY

Helpers are pre-imported — call them directly, no imports needed. The harness
auto-attaches to the running Chrome; you do NOT need to start the daemon.

Working rules:
  1. Prefer coordinate clicks (`click_at_xy(x, y)`) over selector hunts — read
     pixel positions off the screenshot. Drop to `js(...)` only when a target
     has no visible geometry.
  2. After any navigation/click, the next turn's screenshot will tell you if it
     worked. Don't assume.
  3. Sites like Swiggy ask for location first. If a location prompt is up, you
     must satisfy it before any product search.
  4. Use `wait_for_load()` or `wait_for_network_idle()` after page transitions.
  5. Never type credentials. If an auth wall appears, set done=True with a
     final_answer explaining the blocker.
  6. Keep `action_code` minimal — one or two helpers per turn. Smaller turns
     mean tighter feedback loops.
  7. Set `done=True` only when the task is fully complete (e.g. checkout page
     is visibly loaded with the requested item in cart) or unrecoverably stuck.
"""


# ─────────────────────────────────────────────────────────────────
# Chrome lifecycle
# ─────────────────────────────────────────────────────────────────


def _launch_chrome(port: int, headless: bool) -> tuple[subprocess.Popen, Path]:
    user_data_dir = Path(tempfile.mkdtemp(prefix="bh_runner_chrome_"))
    flags = [
        *CHROME_FLAGS_BASE,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--window-size=1920,1080",
    ]
    if headless:
        flags.append("--headless=new")
        flags.append(f"--user-agent={HEADLESS_UA_OVERRIDE}")
    proc = subprocess.Popen(
        [CHROME_PATH, *flags],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc, user_data_dir


def _wait_for_chrome(port: int, timeout: float = 15.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=0.5
            ).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Chrome CDP did not come up on port {port}")


# ─────────────────────────────────────────────────────────────────
# browser-harness invocation
# ─────────────────────────────────────────────────────────────────


def _run_harness(code: str, env: dict[str, str], timeout: float = 60.0) -> dict:
    """Pipe `code` into `browser-harness` and capture stdout/stderr/rc."""
    started = time.time()
    try:
        result = subprocess.run(
            [BROWSER_HARNESS_BIN],
            input=code,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
        return {
            "rc": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.time() - started, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "rc": None,
            "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "") if isinstance(e.stderr, str) else "",
            "duration_s": round(time.time() - started, 3),
            "timed_out": True,
        }


def _screenshot_and_info(env: dict[str, str], path: Path) -> dict:
    code = (
        f"import json\n"
        f"capture_screenshot({json.dumps(str(path))}, max_dim=1800)\n"
        f"print(json.dumps(page_info()))\n"
    )
    out = _run_harness(code, env, timeout=30.0)
    info: dict[str, Any] = {}
    if out["rc"] == 0 and out["stdout"].strip():
        # page_info is the last line; earlier lines could be banners.
        for line in reversed(out["stdout"].splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    info = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    return {"info": info, "harness": out}


# ─────────────────────────────────────────────────────────────────
# Gemini turn
# ─────────────────────────────────────────────────────────────────


def _build_history_text(history: list[dict], cap_chars: int = 4000) -> str:
    """Compact prior turns into the prompt. Each turn shows reasoning, code,
    and a head-and-tail of the harness output so the model can detect loops
    without us slamming the context window."""
    parts: list[str] = []
    for i, t in enumerate(history, start=1):
        out = t.get("harness", {})
        stdout = (out.get("stdout") or "").strip()
        stderr = (out.get("stderr") or "").strip()
        if len(stdout) > cap_chars:
            stdout = stdout[: cap_chars // 2] + "\n…[truncated]…\n" + stdout[-cap_chars // 2 :]
        if len(stderr) > cap_chars:
            stderr = stderr[: cap_chars // 2] + "\n…[truncated]…\n" + stderr[-cap_chars // 2 :]
        parts.append(
            f"--- turn {i} ---\n"
            f"reasoning: {t.get('reasoning','')}\n"
            f"action_code:\n{t.get('action_code','')}\n"
            f"rc={out.get('rc')} timed_out={out.get('timed_out')} duration_s={out.get('duration_s')}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        )
    return "\n".join(parts) if parts else "(no prior turns)"


def _call_gemini(
    *, task: str, page_info: dict, history: list[dict], screenshot_path: Path
) -> dict:
    """Single structured-output Gemini call. Returns the parsed dict."""
    img_bytes = screenshot_path.read_bytes()
    user_text = (
        f"TASK: {task}\n\n"
        f"page_info (current): {json.dumps(page_info)}\n\n"
        f"Prior turns (most recent last):\n{_build_history_text(history)}\n\n"
        "Decide the next action. Return JSON matching the schema."
    )
    contents = [
        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        genai_types.Part.from_text(text=user_text),
    ]
    cfg = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=ACTION_SCHEMA,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.7,
    )
    resp = _gemini.models.generate_content(
        model=GEMINI_MODEL, contents=contents, config=cfg
    )
    text = resp.text or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned non-JSON: {e}\n---\n{text}\n---")
    usage = getattr(resp, "usage_metadata", None)
    return {
        "parsed": parsed,
        "usage": {
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "thinking_tokens": getattr(usage, "thoughts_token_count", None) if usage else None,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────


def _trace_append(trace_path: Path, entry: dict) -> None:
    with trace_path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run(
    *,
    url: str,
    task: str,
    headless: bool,
    port: int,
    results_dir: Path,
    max_turns: int,
    turn_timeout_s: float,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = results_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    trace_path = results_dir / "trace.jsonl"
    result_path = results_dir / "result.json"

    print(f"[bh-runner] results → {results_dir}")
    print(f"[bh-runner] launching Chrome on :{port} (headless={headless})")
    chrome, profile_dir = _launch_chrome(port=port, headless=headless)
    _wait_for_chrome(port)

    bu_name = f"bh-runner-{uuid.uuid4().hex[:8]}"
    env = {
        **os.environ,
        "BU_CDP_URL": f"http://127.0.0.1:{port}",
        "BU_NAME": bu_name,
        "BH_AGENT_WORKSPACE": str(results_dir / "agent_workspace"),
    }
    (Path(env["BH_AGENT_WORKSPACE"])).mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "url": url,
        "task": task,
        "model": GEMINI_MODEL,
        "headless": headless,
        "port": port,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "turns": 0,
        "done": False,
        "final_answer": None,
        "final_url": None,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_thinking_tokens": 0,
        "exit": "unknown",
    }
    _trace_append(trace_path, {"event": "session_start", **summary})

    history: list[dict] = []
    try:
        # Step 0: navigate to start url via a fresh tab so we don't clobber
        # whatever the user had open.
        boot_code = (
            f"new_tab({json.dumps(url)})\n"
            "wait_for_load()\n"
            "print('boot_ok')\n"
        )
        boot = _run_harness(boot_code, env, timeout=45.0)
        _trace_append(
            trace_path,
            {"event": "boot", "code": boot_code, "harness": boot},
        )
        if boot["rc"] != 0:
            print(f"[bh-runner] boot failed: {boot['stderr'][:400]}")
            summary["exit"] = "boot_failed"
            return summary

        for turn in range(1, max_turns + 1):
            print(f"\n[bh-runner] === turn {turn} ===")
            shot_path = steps_dir / f"turn_{turn:02d}.png"
            obs = _screenshot_and_info(env, shot_path)
            info = obs["info"]
            summary["final_url"] = info.get("url", summary["final_url"])

            try:
                gem = _call_gemini(
                    task=task,
                    page_info=info,
                    history=history,
                    screenshot_path=shot_path,
                )
            except Exception as e:
                _trace_append(
                    trace_path,
                    {"event": "gemini_error", "turn": turn, "error": str(e)},
                )
                print(f"[bh-runner] gemini error: {e}")
                summary["exit"] = "gemini_error"
                break

            parsed = gem["parsed"]
            usage = gem["usage"]
            summary["total_input_tokens"] += usage.get("input_tokens") or 0
            summary["total_output_tokens"] += usage.get("output_tokens") or 0
            summary["total_thinking_tokens"] += usage.get("thinking_tokens") or 0

            print(
                f"  reasoning: {parsed.get('reasoning','')[:200]}"
                f"{'…' if len(parsed.get('reasoning',''))>200 else ''}"
            )
            print(f"  done={parsed.get('done')}  confidence={parsed.get('confidence')}")

            if parsed.get("done"):
                summary["done"] = True
                summary["final_answer"] = parsed.get("final_answer")
                summary["exit"] = "completed"
                _trace_append(
                    trace_path,
                    {
                        "event": "turn",
                        "turn": turn,
                        "url": info.get("url"),
                        "screenshot": str(shot_path),
                        "parsed": parsed,
                        "usage": usage,
                        "harness": None,
                    },
                )
                break

            action_code = parsed.get("action_code", "")
            print(f"  action: {action_code[:200]}{'…' if len(action_code)>200 else ''}")
            harness_out = _run_harness(action_code, env, timeout=turn_timeout_s)

            history.append(
                {
                    "reasoning": parsed.get("reasoning", ""),
                    "action_code": action_code,
                    "harness": harness_out,
                }
            )
            _trace_append(
                trace_path,
                {
                    "event": "turn",
                    "turn": turn,
                    "url": info.get("url"),
                    "screenshot": str(shot_path),
                    "parsed": parsed,
                    "usage": usage,
                    "harness": harness_out,
                },
            )
            summary["turns"] = turn

            if harness_out["timed_out"]:
                print(f"  ⚠ harness timed out after {harness_out['duration_s']}s")
            elif harness_out["rc"] != 0:
                print(
                    f"  ⚠ harness rc={harness_out['rc']}: "
                    f"{(harness_out.get('stderr') or '')[:300]}"
                )

        else:
            summary["exit"] = "turn_budget_exhausted"

    finally:
        # Tear down daemon and Chrome cleanly.
        try:
            subprocess.run(
                [BROWSER_HARNESS_BIN, "--reload"],
                env=env,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
        try:
            chrome.terminate()
            chrome.wait(timeout=5)
        except Exception:
            try:
                chrome.kill()
            except Exception:
                pass
        shutil.rmtree(profile_dir, ignore_errors=True)
        summary["ended_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        result_path.write_text(json.dumps(summary, indent=2))
        _trace_append(trace_path, {"event": "session_end", **summary})

    return summary


def _str_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes", "y")


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.browser_harness_runner")
    p.add_argument("--url", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--headless", type=_str_bool, default=False)
    p.add_argument("--port", type=int, default=9222)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--turn-timeout-s", type=float, default=60.0)
    args = p.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = (
            Path(__file__).resolve().parents[1] / "results" / f"bh_{ts}"
        )

    summary = run(
        url=args.url,
        task=args.task,
        headless=args.headless,
        port=args.port,
        results_dir=results_dir,
        max_turns=args.max_turns,
        turn_timeout_s=args.turn_timeout_s,
    )

    print("\n=== bh-runner DONE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
