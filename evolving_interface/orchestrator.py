"""Two-layer orchestrator with structured context management.

Architecture:
  ORCHESTRATOR (single-turn Gemini Pro/Flash calls)
    - Maintains TaskMemory: structured summaries, not raw conversation history
    - Decides next action: MCP tool, CU subtask, or task complete
    - Reflects on results via before/after screenshots
    - Discovers MCP tools from successful CU steps

  CU AGENT (stateless Gemini CU subtask executor)
    - Fresh conversation per subtask — no cross-subtask history
    - Max 10 actions per call with aggressive pruning
    - Returns structured result (answer + actions + traffic)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr
from playwright.async_api import Page

from . import config
from .computer_use import (
    CUResult,
    execute_cu_step,
    start_capture,
    stop_capture,
    _png_to_jpeg,
    MEDIA_RES_LOW,
)
from .mcp_generator import (
    discover_tools,
    execute_tool,
    get_tools_for_agent,
    load_tools,
    update_tool_status,
    verify_tool_result,
)

# ── Gemini clients ──────────────────────────────────────────────
_gemini = genai.Client(api_key=config.GEMINI_API_KEY)
ROUTER_MODEL = config.GEMINI_REASONING_MODEL
FAST_MODEL = config.GEMINI_FAST_MODEL

MAX_ORCHESTRATOR_STEPS = 8
CONFIDENCE_THRESHOLD = 0.5
SITE_LEARNINGS_DIR = config.ROOT_DIR / "site_learnings"
SITE_LEARNINGS_DIR.mkdir(exist_ok=True)
MAX_SITE_LEARNINGS = 20  # cap per site to keep prompt manageable


# ═══════════════════════════════════════════════════════════════════
# Data Models — structured context (never raw conversation history)
# ═══════════════════════════════════════════════════════════════════

class PageSummary(BaseModel):
    url: str
    title: str = ""
    description: str = ""
    key_elements: list[str] = Field(default_factory=list)


class StepSummary(BaseModel):
    step_number: int
    subtask: str
    method: str                           # "mcp" or "computer_use"
    outcome: str = "pending"              # "success" | "partial" | "failed"
    result_summary: str = ""              # 1-2 sentence description
    key_data_extracted: dict[str, str] = Field(default_factory=dict)
    actions_taken: list[str] = Field(default_factory=list)
    final_url: str = ""
    screenshot: bytes | None = None


class TaskMemory(BaseModel):
    """Structured context that the orchestrator accumulates across steps.
    This is LIGHTWEIGHT — text summaries, not raw conversation history."""
    task_description: str
    site_name: str
    site_url: str
    plan: list[str] = Field(default_factory=list)
    steps_completed: list[StepSummary] = Field(default_factory=list)
    pages_observed: list[PageSummary] = Field(default_factory=list)
    current_url: str = ""
    current_page_description: str = ""
    mcp_tools_available: list[str] = Field(default_factory=list)
    steps_remaining: int = MAX_ORCHESTRATOR_STEPS
    key_findings: dict[str, str] = Field(default_factory=dict)
    errors_encountered: list[str] = Field(default_factory=list)


# ── run.py-compatible result models ───────────────────────────────

class StepResult(BaseModel):
    """Per-step result compatible with run.py's logging expectations."""
    step_number: int = 0
    method: str = ""
    tool_name: str | None = None
    success: bool = False
    time_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    new_tools_discovered: list[str] = Field(default_factory=list)
    verified: bool | None = None
    fell_back_to_cu: bool = False
    answer: str | None = None
    result_data: Any = None
    action_log: list[dict] = Field(default_factory=list)
    _captured_requests: list = PrivateAttr(default_factory=list)


class TaskResult(BaseModel):
    """Top-level result compatible with run.py."""
    task_id: str = ""
    site: str = ""
    task_description: str = ""
    strategy: str = ""
    steps: list[StepResult] = Field(default_factory=list)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_time_seconds: float = 0.0
    tools_before: int = 0
    tools_after: int = 0
    new_tools_discovered: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    success: bool = False


# ═══════════════════════════════════════════════════════════════════
# Helper — safe Gemini call with retry
# ═══════════════════════════════════════════════════════════════════

def _call_model(
    model: str,
    prompt: str,
    images: list[tuple[str, bytes]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Single-turn Gemini call. Returns text response."""
    parts: list[types.Part] = [types.Part(text=prompt)]
    if images:
        for mime, data in images:
            parts.append(types.Part(
                inline_data=types.Blob(mime_type=mime, data=data),
                media_resolution=MEDIA_RES_LOW,
            ))

    for attempt in range(3):
        try:
            resp = _gemini.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text or ""
        except Exception as exc:
            if attempt < 2:
                wait = [5, 30, 60][attempt]
                print(f"    Model call error ({exc}), retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Model call failed after 3 attempts: {exc}")
                return ""


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Extract JSON from a model response that may contain markdown."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════
# Orchestrator Steps — each is a single-turn call
# ═══════════════════════════════════════════════════════════════════

def _describe_and_plan(
    memory: TaskMemory, screenshot_jpeg: bytes,
) -> tuple[str, list[str]]:
    """Single API call: describe the current page AND create the plan.

    Returns (page_description, plan_steps).
    Saves one round-trip vs separate _describe_page + _create_plan calls.
    """
    tools_listing = _format_tools(memory.site_name)
    # Include site learnings from prior tasks
    learnings = _load_site_learnings(memory.site_name)
    learnings_text = ""
    if learnings:
        tips = "\n".join(f"  - {tip}" for tip in learnings[-10:])
        learnings_text = f"\nSITE TIPS (from prior tasks):\n{tips}\n\n"
    prompt = (
        f"TASK: {memory.task_description}\n"
        f"WEBSITE: {memory.site_name} ({memory.site_url})\n"
        f"CURRENT URL: {memory.current_url}\n"
        f"{learnings_text}\n"
        f"AVAILABLE MCP TOOLS:\n{tools_listing}\n\n"
        "Look at the screenshot and do TWO things:\n\n"
        "1. PAGE DESCRIPTION: Describe this webpage in 2-3 sentences — "
        "what kind of page it is, key elements visible, any relevant text.\n\n"
        "2. PLAN: Create a plan of 3-6 sequential steps to complete the task.\n"
        "Each step should be specific and achievable in ~5-10 browser actions.\n"
        "Prefer MCP tools when available (10x faster than browser agent).\n\n"
        "TASK TYPE DETECTION:\n"
        "- If this task requires CHANGING something on the website "
        "(submitting a form, posting, adding to cart, deleting, "
        "creating, updating), the plan MUST include a final step "
        "that explicitly performs the submission/confirmation action. "
        "The task is NOT complete until the state change is confirmed.\n"
        "- If this task requires FINDING information (price, count, "
        "name, list), the task is complete when the information is "
        "found and reported.\n\n"
        "For action tasks, the LAST plan step must be something like:\n"
        "  'Click the Submit/Save/Post/Add button and verify the "
        "success confirmation message appears.'\n"
        "Do NOT stop at 'fill in the form' — that is preparation, "
        "not completion.\n\n"
        'Return ONLY valid JSON:\n'
        '{"page_description": "2-3 sentence description...", '
        '"plan": ["Step 1: ...", "Step 2: ...", ...]}'
    )
    raw = _call_model(
        ROUTER_MODEL, prompt,
        images=[("image/jpeg", screenshot_jpeg)],
        max_tokens=1024,
    )
    data = _parse_json(raw)
    if data:
        page_desc = data.get("page_description", "")
        steps = data.get("plan", [])
        if isinstance(steps, list) and steps:
            return page_desc, [str(s) for s in steps]
        # Might have returned just a list (old format)
        if isinstance(data, list):
            return "", [str(s) for s in data]
    # Fallback: try parsing as plain list
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        steps = json.loads(cleaned)
        if isinstance(steps, list):
            return "", [str(s) for s in steps]
    except (json.JSONDecodeError, TypeError):
        pass
    return "", [f"Complete the task: {memory.task_description}"]


def _is_mutate_task(task_description: str) -> bool:
    """Detect if this task requires a state-changing action (not just retrieval)."""
    lower = task_description.lower()
    mutate_signals = (
        "submit", "post", "create", "update", "delete", "add to cart",
        "change", "modify", "set", "write", "comment", "subscribe",
        "unsubscribe", "cancel", "remove", "edit", "save", "publish",
        "merge", "close", "reopen", "assign", "label", "star", "fork",
        "upvote", "downvote", "reply", "send", "order", "purchase",
        "checkout", "register", "sign up", "prepare", "fill out",
    )
    return any(word in lower for word in mutate_signals)


def _load_site_learnings(site_name: str) -> list[str]:
    """Load accumulated navigation/interaction tips for a site."""
    path = SITE_LEARNINGS_DIR / f"{site_name}.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_site_learnings(site_name: str, learnings: list[str]) -> None:
    """Persist site learnings, keeping only the most recent."""
    path = SITE_LEARNINGS_DIR / f"{site_name}.json"
    # Keep most recent learnings
    capped = learnings[-MAX_SITE_LEARNINGS:]
    with open(path, "w") as f:
        json.dump(capped, f, indent=2)


def _generate_site_learning(memory: TaskMemory) -> str | None:
    """After a task completes, extract a generalizable navigation tip.

    Only generates a learning if the task had errors or budget issues
    that future tasks on this site could benefit from knowing about.
    """
    if not memory.errors_encountered and memory.steps_remaining > 1:
        return None  # Task went smoothly, nothing to learn

    # Build a short summary of what happened
    steps_text = "\n".join(
        f"  {s.subtask[:50]} → {s.outcome}: {s.result_summary[:50]}"
        for s in memory.steps_completed
    )
    prompt = (
        f"WEBSITE: {memory.site_name}\n"
        f"TASK: {memory.task_description}\n"
        f"STEPS TAKEN:\n{steps_text}\n"
        f"ERRORS: {'; '.join(memory.errors_encountered[:3])}\n"
        f"STEPS REMAINING: {memory.steps_remaining}\n\n"
        "Extract ONE short, generalizable navigation tip for this website "
        "that would help future tasks. Focus on:\n"
        "- Where to find common features (e.g. 'Admin panel is under Stores > Settings')\n"
        "- Login quirks or required steps\n"
        "- Form interaction patterns that work/fail\n"
        "- Page structure or navigation shortcuts\n\n"
        "Return ONLY the tip as a single sentence. "
        "If there's nothing generalizable to learn, return 'NONE'."
    )
    tip = _call_model(FAST_MODEL, prompt, max_tokens=128)
    tip = tip.strip()
    if not tip or tip.upper() == "NONE":
        return None
    return tip


def _format_tools(site_name: str) -> str:
    tools = get_tools_for_agent(site_name)
    if not tools:
        return "(No MCP tools discovered yet — use computer_use.)"
    lines: list[str] = []
    for t in tools:
        props = t["inputSchema"].get("properties", {})
        param_list = ", ".join(
            f"{k}: {v.get('type', '?')}" for k, v in props.items()
        )
        method = t.get("http_method", "?")
        lines.append(f"- {t['name']} [{method}]: {t['description']}  Params({param_list})")
    return "\n".join(lines)


def _build_memory_prompt(memory: TaskMemory) -> str:
    """Render TaskMemory into a text prompt for the orchestrator."""
    parts: list[str] = []

    parts.append(f"TASK: {memory.task_description}")
    parts.append(f"WEBSITE: {memory.site_name} ({memory.site_url})")
    parts.append(f"CURRENT URL: {memory.current_url}")
    parts.append(f"STEPS REMAINING: {memory.steps_remaining} of {MAX_ORCHESTRATOR_STEPS}")

    # Site learnings from prior tasks
    learnings = _load_site_learnings(memory.site_name)
    if learnings:
        tips = "\n".join(f"  - {tip}" for tip in learnings[-5:])
        parts.append(f"\nSITE TIPS (from prior tasks):\n{tips}")

    # Plan with completion markers
    if memory.plan:
        plan_lines = []
        completed_count = len(memory.steps_completed)
        for i, step in enumerate(memory.plan):
            marker = "✓" if i < completed_count else " "
            plan_lines.append(f"  [{marker}] {step}")
        parts.append(f"\nPLAN:\n" + "\n".join(plan_lines))

    # Completed steps
    if memory.steps_completed:
        step_lines = []
        for s in memory.steps_completed:
            step_lines.append(
                f"  Step {s.step_number}: [{s.method}] {s.subtask[:60]} "
                f"→ {s.outcome}: {s.result_summary[:80]}"
            )
            if s.key_data_extracted:
                for k, v in s.key_data_extracted.items():
                    step_lines.append(f"    Data: {k} = {v}")
        parts.append(f"\nWHAT'S BEEN DONE:\n" + "\n".join(step_lines))

    # Key findings
    if memory.key_findings:
        findings = "\n".join(f"  {k}: {v}" for k, v in memory.key_findings.items())
        parts.append(f"\nKEY FINDINGS:\n{findings}")

    # Pages observed (last 3)
    if memory.pages_observed:
        page_lines = [
            f"  {p.url} — {p.description[:60]}"
            for p in memory.pages_observed[-3:]
        ]
        parts.append(f"\nPAGES OBSERVED:\n" + "\n".join(page_lines))

    # Errors
    if memory.errors_encountered:
        parts.append(f"\nERRORS: {'; '.join(memory.errors_encountered[-3:])}")

    # Tools
    tools_listing = _format_tools(memory.site_name)
    parts.append(f"\nAVAILABLE MCP TOOLS:\n{tools_listing}")

    return "\n".join(parts)


def _decide_next_step(
    memory: TaskMemory,
    screenshots: list[bytes],
) -> dict[str, Any]:
    """Single-turn orchestrator decision: what to do next."""
    memory_text = _build_memory_prompt(memory)

    # Urgency hints based on remaining steps
    urgency = ""
    if memory.steps_remaining <= 2:
        urgency = (
            "\nURGENT: Only {0} step(s) remain. If you have enough "
            "information to answer, use action='complete' NOW. "
            "Do not start new exploration.".format(memory.steps_remaining)
        )
    if memory.steps_remaining == 1:
        urgency = (
            "\nFINAL STEP: Provide your best answer or use one last "
            "targeted action. Prefer answering with partial information."
        )

    prompt = (
        f"{memory_text}\n{urgency}\n\n"
        "Decide the next action. Return ONLY valid JSON:\n"
        "Option 1 — MCP tool: "
        '{"action": "mcp", "tool": "tool_name", '
        '"params": {...}, "purpose": "..."}\n'
        "Option 2 — Browser agent: "
        '{"action": "computer_use", '
        '"subtask": "specific instruction for the browser agent", '
        '"purpose": "..."}\n'
        "Option 3 — Task complete: "
        '{"action": "complete", "answer": "the final answer"}\n\n'
        "BEFORE COMPLETING:\n"
        "If this is an action task (submit, post, create, delete, update):\n"
        "- Has the submit/save/confirm button actually been clicked?\n"
        "- Did you see a success message or confirmation page?\n"
        "- If NO: you are not done. Add a step to perform the final action.\n"
        "If this is a retrieval task (find, count, list, what is):\n"
        "- Do you have the specific answer the task asked for?\n"
        "- Is it a concrete value, not a description of what you did?\n"
        "- If NO: you need one more step to extract the answer.\n"
        "Common mistake: saying 'the form has been filled out' when the "
        "task requires submitting it. Filling ≠ Submitting.\n\n"
        "RULES:\n"
        "- MCP tools are 10x faster — use them when available.\n"
        "- Browser subtasks should be SPECIFIC and achievable in "
        "~5-10 actions. Include the current URL and what the agent "
        "should see.\n"
        "- If a previous approach failed, try something different.\n"
    )

    # Attach screenshots (current + up to 2 prior)
    images: list[tuple[str, bytes]] = []
    for ss in screenshots[:3]:
        jpeg = _png_to_jpeg(ss, quality=50)
        images.append(("image/jpeg", jpeg))

    raw = _call_model(
        ROUTER_MODEL, prompt,
        images=images,
        max_tokens=1024,
    )

    data = _parse_json(raw)
    if data and "action" in data:
        return data

    # Fallback: use CU with the full task
    return {
        "action": "computer_use",
        "subtask": memory.task_description,
        "purpose": "could not parse orchestrator decision",
    }


# ═══════════════════════════════════════════════════════════════════
# Reflection — lightweight before/after comparison
# ═══════════════════════════════════════════════════════════════════

def _reflect_on_step(
    subtask: str,
    result_answer: str | None,
    action_descriptions: list[str],
    screenshot_before: bytes,
    screenshot_after: bytes,
    memory: TaskMemory,
) -> dict[str, Any]:
    """Fast model evaluates step completion via before/after screenshots."""
    actions_text = ", ".join(action_descriptions[:10]) or "none"
    prompt = (
        f"A browser agent was asked to: \"{subtask}\"\n\n"
        f"Actions taken: {actions_text}\n"
        f"Agent reported: \"{result_answer or '(no output)'}\"\n\n"
        f"OVERALL TASK: {memory.task_description}\n\n"
        "[First image = BEFORE, Second image = AFTER]\n\n"
        "Evaluate:\n"
        '1. outcome: "success" | "partial" | "failed"\n'
        "2. summary: One sentence describing what actually happened\n"
        "3. extracted_data: Key-value pairs of any factual data found "
        "(e.g. {\"product_price\": \"$29.99\"}), or {}\n"
        "4. page_description: Brief description of the AFTER page\n\n"
        "ACTION TASK CHECK: If the subtask mentioned submitting, "
        "posting, saving, adding, deleting, or creating something:\n"
        "- Did the browser actually perform the final action?\n"
        "- Is there a success/confirmation message in the AFTER screenshot?\n"
        "- If the form is filled but NOT submitted, outcome is 'partial', "
        "not 'success'. The orchestrator needs to know a submit step "
        "is still needed.\n\n"
        "Return ONLY valid JSON."
    )

    before_jpeg = _png_to_jpeg(screenshot_before, quality=40)
    after_jpeg = _png_to_jpeg(screenshot_after, quality=50)

    raw = _call_model(
        FAST_MODEL, prompt,
        images=[("image/jpeg", before_jpeg), ("image/jpeg", after_jpeg)],
        max_tokens=512,
    )

    data = _parse_json(raw)
    if data:
        return data

    # Fallback: assume success if agent gave an answer
    return {
        "outcome": "success" if result_answer else "failed",
        "summary": result_answer or "No result from agent",
        "extracted_data": {},
        "page_description": "",
    }


# ═══════════════════════════════════════════════════════════════════
# MCP Step Execution
# ═══════════════════════════════════════════════════════════════════

async def _execute_mcp_step(
    page: Page,
    site_name: str,
    tool_name: str,
    params: dict[str, Any],
) -> tuple[StepResult, bool]:
    """Execute an MCP tool. Returns (StepResult, success)."""
    t0 = time.monotonic()
    sr = StepResult(method="mcp", tool_name=tool_name)

    tools = load_tools(site_name)
    tool_def = tools.get(tool_name)
    if tool_def is None:
        sr.answer = f"Tool '{tool_name}' not found"
        return sr, False

    if tool_def.meta.status in ("stale", "deprecated"):
        sr.answer = f"Tool '{tool_name}' is {tool_def.meta.status}"
        return sr, False

    # Start traffic capture
    capture_ctx = start_capture(page)

    print(f"    MCP [{tool_def.meta.status}] {tool_name}({params})")

    try:
        result = await execute_tool(site_name, tool_name, params, page=page)
    except Exception as exc:
        print(f"    MCP exception: {exc}")
        stop_capture(capture_ctx)
        update_tool_status(site_name, tool_name, success=False)
        return sr, False

    sr._captured_requests = stop_capture(capture_ctx)
    # Append httpx-originated CapturedRequest (invisible to Playwright)
    mcp_captured = result.get("captured_request")
    if mcp_captured is not None:
        sr._captured_requests.append(mcp_captured)
    sr.success = result.get("success", False)
    sr.result_data = result.get("body")
    sr.time_ms = int((time.monotonic() - t0) * 1000)

    if not sr.success:
        print(f"    MCP HTTP {result.get('status')}")
        update_tool_status(site_name, tool_name, success=False)
        return sr, False

    # Verify state-changing tools
    is_state_changing = tool_def.http.method in ("POST", "PUT", "PATCH", "DELETE")
    if tool_def.meta.status in ("new", "untrusted") and is_state_changing:
        print(f"    Verifying {tool_name}...")
        await page.reload(wait_until="domcontentloaded")
        verified = await verify_tool_result(
            page, tool_name, tool_def.description, params, result,
        )
        sr.verified = verified
        if not verified:
            print(f"    Verification FAILED")
            update_tool_status(site_name, tool_name, success=False)
            return sr, False
        print(f"    Verified")

    update_tool_status(site_name, tool_name, success=True)

    # Extract answer from API response
    sr.answer = _extract_api_answer(result)
    if sr.answer:
        print(f"    Answer: {sr.answer[:60]}")

    return sr, True


def _extract_api_answer(result: dict[str, Any]) -> str | None:
    """Try to pull a meaningful answer from an MCP tool HTTP response."""
    body = result.get("body")
    if body is None:
        return None

    if isinstance(body, str):
        stripped = body.strip()
        if stripped.startswith("<!") or stripped.startswith("<html"):
            return None
        return stripped[:500] if stripped else None

    if isinstance(body, dict):
        for key in (
            "total", "count", "total_count", "result",
            "price", "value", "answer", "name", "title",
        ):
            if key in body:
                return str(body[key])
        for key in ("items", "results", "products", "data"):
            if key in body and isinstance(body[key], list):
                return str(len(body[key]))

    if isinstance(body, list):
        return str(len(body))

    return None


# ═══════════════════════════════════════════════════════════════════
# MCP Discovery from successful CU steps
# ═══════════════════════════════════════════════════════════════════

def _discover_from_traffic(
    captured_requests: list,
    task: str,
    site_name: str,
) -> list[str]:
    """Discover MCP tools from captured HTTP traffic."""
    new_tools = discover_tools(captured_requests, [], task, site_name)
    names = [t.name for t in new_tools]
    if names:
        print(f"    Discovered {len(names)} tools: {names}")
    return names


# ═══════════════════════════════════════════════════════════════════
# Main Orchestrator Loop
# ═══════════════════════════════════════════════════════════════════

async def _run_orchestrator(
    task: str,
    site_name: str,
    page: Page,
    task_result: TaskResult,
) -> None:
    """Main orchestrator loop with structured context management."""
    site_url = config.SITES.get(site_name, site_name)

    memory = TaskMemory(
        task_description=task,
        site_name=site_name,
        site_url=site_url,
        current_url=page.url,
    )

    # Refresh MCP tool list
    memory.mcp_tools_available = [
        t["name"] for t in get_tools_for_agent(site_name)
    ]

    # Step 0+1: Describe page AND create plan in one API call
    print("  Analyzing page & creating plan...")
    screenshot_png = await page.screenshot(type="png")
    screenshot_jpeg = _png_to_jpeg(screenshot_png, quality=60)
    page_desc, plan = _describe_and_plan(memory, screenshot_jpeg)
    memory.current_page_description = page_desc
    memory.pages_observed.append(PageSummary(
        url=page.url,
        description=page_desc,
    ))
    memory.plan = plan
    if page_desc:
        print(f"    Page: {page_desc[:80]}")
    print(f"    Plan ({len(plan)} steps):")
    for i, step in enumerate(plan):
        print(f"      {i + 1}. {step[:70]}")

    # Determine strategy
    tools_listing = get_tools_for_agent(site_name)
    has_tools = len(tools_listing) > 0
    task_result.strategy = "hybrid" if has_tools else "computer_use"

    # Main loop
    for step_num in range(MAX_ORCHESTRATOR_STEPS):
        memory.steps_remaining = MAX_ORCHESTRATOR_STEPS - step_num
        memory.current_url = page.url

        # Take fresh screenshot
        current_screenshot = await page.screenshot(type="png")

        # Collect recent screenshots for orchestrator context
        recent_screenshots = [current_screenshot]
        for s in memory.steps_completed[-2:]:
            if s.screenshot:
                recent_screenshots.append(s.screenshot)

        # Ask orchestrator: what next?
        print(f"\n  --- Orchestrator step {step_num + 1}/{MAX_ORCHESTRATOR_STEPS} ---")
        decision = _decide_next_step(memory, recent_screenshots)
        action = decision.get("action", "computer_use")
        purpose = decision.get("purpose", "")

        print(f"    Decision: {action}")
        if action == "mcp":
            print(f"    Tool: {decision.get('tool')} params={decision.get('params')}")
        elif action == "computer_use":
            print(f"    Subtask: {decision.get('subtask', '')[:70]}")
        elif action == "complete":
            print(f"    Answer: {decision.get('answer', '')[:70]}")

        # === TASK COMPLETE ===
        if action == "complete":
            answer = decision.get("answer", "")

            # Pre-completion guardrail for MUTATE tasks:
            # Check if any state-changing HTTP request was actually observed
            if _is_mutate_task(task) and memory.steps_remaining > 1:
                has_state_change = any(
                    entry.get("post_requests", 0) > 0
                    for sr in task_result.steps
                    for entry in sr.action_log
                )
                if not has_state_change:
                    print(f"  GUARDRAIL: Mutate task but no POST/PUT detected — forcing submit step")
                    decision = {
                        "action": "computer_use",
                        "subtask": (
                            f"The form/page has been prepared but the final action was NOT performed. "
                            f"Look at the current page and click the Submit/Save/Post/Confirm/Create button. "
                            f"Original task: {task}"
                        ),
                        "purpose": "guardrail: no state-changing request detected",
                    }
                    action = "computer_use"
                    # Fall through to computer_use handler below
                else:
                    task_result.final_answer = answer
                    print(f"  COMPLETE: {answer[:80]}")
                    break
            else:
                task_result.final_answer = answer
                print(f"  COMPLETE: {answer[:80]}")
                break

        # === MCP TOOL ===
        if action == "mcp":
            tool_name = decision.get("tool", "")
            params = decision.get("params", {})
            sr, success = await _execute_mcp_step(
                page, site_name, tool_name, params,
            )
            sr.step_number = step_num + 1
            task_result.steps.append(sr)
            task_result.total_tokens_input += sr.tokens_input
            task_result.total_tokens_output += sr.tokens_output

            # Reflect
            after_screenshot = await page.screenshot(type="png")
            reflection = _reflect_on_step(
                subtask=f"MCP {tool_name}({params})",
                result_answer=sr.answer,
                action_descriptions=[f"called {tool_name}"],
                screenshot_before=current_screenshot,
                screenshot_after=after_screenshot,
                memory=memory,
            )
            outcome = reflection.get("outcome", "failed")
            summary = reflection.get("summary", "")
            extracted = reflection.get("extracted_data", {})
            page_desc = reflection.get("page_description", "")
            print(f"    Reflect: {outcome} — {summary[:60]}")

            # Update memory
            memory.steps_completed.append(StepSummary(
                step_number=step_num + 1,
                subtask=f"MCP {tool_name}",
                method="mcp",
                outcome=outcome,
                result_summary=summary,
                key_data_extracted=extracted if isinstance(extracted, dict) else {},
                actions_taken=[f"called {tool_name}({params})"],
                final_url=page.url,
            ))
            if isinstance(extracted, dict):
                memory.key_findings.update(extracted)
            memory.pages_observed.append(PageSummary(
                url=page.url, description=page_desc,
            ))

            if not success:
                memory.errors_encountered.append(
                    f"MCP {tool_name} failed: {summary[:60]}"
                )

            # Discover tools from MCP traffic
            if success and sr._captured_requests:
                names = _discover_from_traffic(
                    sr._captured_requests, task, site_name,
                )
                sr.new_tools_discovered = names
                task_result.new_tools_discovered += names

            if sr.answer:
                task_result.final_answer = sr.answer

            continue

        # === COMPUTER USE ===
        subtask = decision.get("subtask", task)

        # Append autonomy-level instruction
        if config.TASK_AUTONOMY_LEVEL == "benchmark":
            subtask += (
                "\n\nIMPORTANT: Complete ALL actions including clicking "
                "Submit/Save/Post buttons. Do not stop before the final "
                "confirmation. This is an automated benchmark — proceed "
                "with all actions without hesitation."
            )
        elif config.TASK_AUTONOMY_LEVEL == "cautious":
            subtask += (
                "\n\nBefore clicking any Submit/Save/Post/Delete/Purchase "
                "button, stop and report what you're about to do."
            )

        print(f"  CU: {subtask[:80]}")

        cu_result = await execute_cu_step(page, subtask)

        sr = StepResult(
            step_number=step_num + 1,
            method="computer_use",
            success=cu_result.answer is not None,
            time_ms=int(cu_result.wall_clock_seconds * 1000),
            tokens_input=cu_result.input_tokens,
            tokens_output=cu_result.output_tokens,
            answer=cu_result.answer,
            result_data=cu_result.answer,
            action_log=[e.model_dump() for e in cu_result.action_log],
        )
        sr._captured_requests = cu_result.captured_requests
        task_result.steps.append(sr)
        task_result.total_tokens_input += sr.tokens_input
        task_result.total_tokens_output += sr.tokens_output

        # Reflect on what happened
        after_screenshot = await page.screenshot(type="png")
        reflection = _reflect_on_step(
            subtask=subtask,
            result_answer=cu_result.answer,
            action_descriptions=cu_result.action_descriptions,
            screenshot_before=current_screenshot,
            screenshot_after=after_screenshot,
            memory=memory,
        )
        outcome = reflection.get("outcome", "failed")
        summary = reflection.get("summary", "")
        extracted = reflection.get("extracted_data", {})
        page_desc = reflection.get("page_description", "")
        print(f"    Reflect: {outcome} — {summary[:60]}")

        # Update memory with structured summary
        memory.steps_completed.append(StepSummary(
            step_number=step_num + 1,
            subtask=subtask,
            method="computer_use",
            outcome=outcome,
            result_summary=summary,
            key_data_extracted=extracted if isinstance(extracted, dict) else {},
            actions_taken=cu_result.action_descriptions,
            final_url=page.url,
            screenshot=cu_result.final_screenshot,
        ))
        if isinstance(extracted, dict):
            memory.key_findings.update(extracted)
        memory.pages_observed.append(PageSummary(
            url=page.url, description=page_desc,
        ))

        if outcome == "failed":
            memory.errors_encountered.append(
                f"CU subtask failed: {subtask[:40]} — {summary[:40]}"
            )

        # Discover MCP tools from successful CU traffic
        if outcome == "success" and cu_result.captured_requests:
            names = _discover_from_traffic(
                cu_result.captured_requests, task, site_name,
            )
            sr.new_tools_discovered = names
            task_result.new_tools_discovered += names

        if cu_result.answer:
            task_result.final_answer = cu_result.answer

    # If we exhausted steps without a "complete" decision, synthesize answer
    if task_result.final_answer is None:
        print("  Budget exhausted — synthesizing answer...")
        task_result.final_answer = _synthesize_answer(memory)

    # Save site learning if the task had difficulties
    try:
        tip = _generate_site_learning(memory)
        if tip:
            learnings = _load_site_learnings(site_name)
            # Avoid duplicate tips
            if tip not in learnings:
                learnings.append(tip)
                _save_site_learnings(site_name, learnings)
                print(f"  Learning saved: {tip[:60]}")
    except Exception as exc:
        print(f"  Could not save learning: {exc}")


def _synthesize_answer(memory: TaskMemory) -> str | None:
    """Last resort: synthesize the best answer from accumulated findings."""
    if not memory.key_findings and not memory.steps_completed:
        return None

    findings = json.dumps(memory.key_findings, default=str) if memory.key_findings else "(none)"
    steps = "\n".join(
        f"  {s.subtask[:50]} → {s.outcome}: {s.result_summary[:60]}"
        for s in memory.steps_completed
    ) or "(none)"

    prompt = (
        f"TASK: {memory.task_description}\n\n"
        f"KEY FINDINGS:\n{findings}\n\n"
        f"STEPS TAKEN:\n{steps}\n\n"
        "Based on the above, provide the best possible answer to the task.\n"
        "If the task was to perform an action (create, add, submit), "
        "state whether it was completed.\n"
        "If the task was to retrieve information, provide the data found.\n"
        "Be concise — one sentence or the specific value requested."
    )

    raw = _call_model(FAST_MODEL, prompt, max_tokens=256)
    return raw.strip() if raw.strip() else None


# ═══════════════════════════════════════════════════════════════════
# Entry Point — compatible with run.py
# ═══════════════════════════════════════════════════════════════════

async def run_task(
    task_id: str,
    task: str,
    site_name: str,
    page: Page,
) -> TaskResult:
    """Full orchestrated task execution.

    Interface contract with run.py:
      - Returns TaskResult with .final_answer, .strategy, .steps,
        .total_tokens_input/output, .new_tools_discovered
      - start_capture/stop_capture are handled at run.py level for eval
    """
    t0 = time.monotonic()
    tools_before = len(load_tools(site_name))

    result = TaskResult(
        task_id=task_id,
        site=site_name,
        task_description=task,
        tools_before=tools_before,
    )

    print(f"\n--- Orchestrator ---")

    try:
        await _run_orchestrator(task, site_name, page, result)
    except Exception as exc:
        print(f"  Orchestrator error: {type(exc).__name__}: {exc}")
        # If we have partial results, keep them
        if not result.final_answer:
            # Last-ditch: try a direct CU call with the full task
            print("  Falling back to direct CU...")
            try:
                cu = await execute_cu_step(page, task, max_actions=15)
                sr = StepResult(
                    step_number=len(result.steps) + 1,
                    method="computer_use",
                    success=cu.answer is not None,
                    time_ms=int(cu.wall_clock_seconds * 1000),
                    tokens_input=cu.input_tokens,
                    tokens_output=cu.output_tokens,
                    answer=cu.answer,
                    action_log=[e.model_dump() for e in cu.action_log],
                )
                sr._captured_requests = cu.captured_requests
                result.steps.append(sr)
                result.total_tokens_input += cu.input_tokens
                result.total_tokens_output += cu.output_tokens
                result.final_answer = cu.answer
                result.strategy = "computer_use"
            except Exception as fallback_exc:
                print(f"  Fallback CU also failed: {fallback_exc}")

    result.success = result.final_answer is not None
    result.tools_after = len(load_tools(site_name))
    result.total_time_seconds = time.monotonic() - t0

    # Print summary
    total_tok = result.total_tokens_input + result.total_tokens_output
    print(f"\n  Result: {'PASS' if result.success else 'FAIL'}")
    print(f"  Answer: {result.final_answer or '(none)'}")
    print(f"  Tokens: {total_tok:,} ({result.total_tokens_input:,} in)")
    print(f"  Time:   {result.total_time_seconds:.1f}s")
    print(f"  Steps:  {len(result.steps)}")

    return result
