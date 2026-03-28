"""Stateless browser agent (Gemini Computer Use).

The CU agent is a lightweight, stateless executor. It receives a single
subtask instruction, executes up to MAX_CU_ACTIONS browser actions, and
returns the result. It does NOT maintain history across subtasks — the
orchestrator handles all strategic context.

Key design:
- Fresh conversation per subtask (no history from prior subtasks)
- Max 10 actions per subtask call (configurable)
- Aggressive pruning: only last 2 turns kept in context
- 2s pacing between API calls for rate limit management
- FunctionResponse screenshots MUST be PNG (Gemini CU requirement)
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from datetime import date
from typing import Any
from urllib.parse import urlparse, parse_qs

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image
from pydantic import BaseModel, Field
from playwright.async_api import Page, Request, Response

from . import config
from .config import build_site_context, get_allowed_origins, get_profile_for_url


# ===================================================================
# Traffic Capture -- hooks into Playwright network events
# ===================================================================

CONTAINER_ORIGINS: set[str] = {
    urlparse(url).netloc for url in config.SITES.values()
}

ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

IGNORED_EXTENSIONS = {
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".woff", ".woff2", ".ttf", ".ico", ".map",
}

IGNORED_PATH_FRAGMENTS = {
    "/static/", "/media/", "/pub/static/", "/theme/",
    "/searchtermslog/", "/customer/section/load",
}

IGNORED_ANALYTICS = {
    "google-analytics", "gtag", "facebook", "hotjar",
    "segment", "mixpanel",
}


class CapturedRequest(BaseModel):
    timestamp: float
    method: str
    url: str
    path: str
    query_params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None
    response_status: int = 0
    response_body: Any = None
    response_headers: dict[str, str] = Field(default_factory=dict)


class CaptureContext:
    def __init__(self, page: Page) -> None:
        self.page: Page = page
        self.requests: list[CapturedRequest] = []
        self.pending: dict[str, CapturedRequest] = {}
        self.active: bool = True
        self._req_handler: Any = None
        self._resp_handler: Any = None


def _should_capture(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc not in CONTAINER_ORIGINS:
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in IGNORED_EXTENSIONS):
        return False
    if any(frag in path for frag in IGNORED_PATH_FRAGMENTS):
        return False
    if any(a in url for a in IGNORED_ANALYTICS):
        return False
    return True


def start_capture(page: Page) -> CaptureContext:
    """Attach request/response listeners and return a CaptureContext."""
    ctx = CaptureContext(page)

    def req_handler(request: Request) -> None:
        if not ctx.active:
            return
        if request.method not in ALLOWED_METHODS:
            return
        if not _should_capture(request.url):
            return

        parsed = urlparse(request.url)
        try:
            body = request.post_data
            if body:
                try:
                    body = json.loads(body)
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            body = None

        captured = CapturedRequest(
            timestamp=time.time(),
            method=request.method,
            url=request.url,
            path=parsed.path,
            query_params=dict(parse_qs(parsed.query, keep_blank_values=True)),
            headers={k.lower(): v for k, v in request.headers.items()},
            body=body,
        )
        ctx.pending[request.url + request.method] = captured

    def resp_handler(response: Response) -> None:
        if not ctx.active:
            return
        key = response.url + response.request.method
        captured = ctx.pending.pop(key, None)
        if captured is None:
            return
        captured.response_status = response.status
        captured.response_headers = {
            k.lower(): v for k, v in response.headers.items()
        }
        ctx.requests.append(captured)

    page.on("request", req_handler)
    page.on("response", resp_handler)
    ctx._req_handler = req_handler
    ctx._resp_handler = resp_handler
    return ctx


def stop_capture(ctx: CaptureContext) -> list[CapturedRequest]:
    """Remove event handlers and return all captured requests."""
    if ctx.active:
        ctx.page.remove_listener("request", ctx._req_handler)
        ctx.page.remove_listener("response", ctx._resp_handler)
        ctx.active = False
    for captured in ctx.pending.values():
        ctx.requests.append(captured)
    ctx.pending.clear()
    return ctx.requests


async def get_browser_cookies(page: Page) -> dict[str, str]:
    """Return browser cookies formatted as {"Cookie": "k1=v1; k2=v2"}."""
    cookies = await page.context.cookies()
    if not cookies:
        return {}
    cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
    return {"Cookie": cookie_str}


# ===================================================================
# Computer Use Agent -- stateless subtask executor
# ===================================================================

MODEL = config.GEMINI_COMPUTER_USE_MODEL

# Google's recommended resolution for Computer Use
VIEWPORT = {"width": 1440, "height": 900}

# Per-subtask action budget (narrow subtasks need ~5-10 actions)
MAX_CU_ACTIONS = 10

# Pacing between API calls to manage rate limits
CU_PACING_DELAY = 1.0

# API retry settings
API_MAX_RETRIES = 4
API_BACKOFF_DELAYS = [2, 5, 15, 30]

# Wait 45s on first 429 for TPM window to reset
TPM_RESET_WAIT = 45

# Subtask timeout
SUBTASK_TIMEOUT = 180.0  # 3 minutes per subtask

STUCK_THRESHOLD = 3
STUCK_RECOVERY_MAX = 2

# ── Media resolution for token control (Gemini 3) ──────────────
# HIGH = 1,120 tokens/image — for CU's latest screenshot
# MEDIUM = 560 tokens/image — for degraded mode after 429
# LOW  = 280 tokens/image  — for non-CU models (reflection, verify)
MEDIA_RES_HIGH = types.PartMediaResolution(
    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
)
MEDIA_RES_MEDIUM = types.PartMediaResolution(
    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
)
MEDIA_RES_LOW = types.PartMediaResolution(
    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
)


def _png_to_jpeg(png_bytes: bytes, quality: int = 80) -> bytes:
    """Convert PNG screenshot to JPEG. Reduces payload ~70%."""
    img = Image.open(io.BytesIO(png_bytes))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _norm_to_pixel(x: float, y: float) -> tuple[int, int]:
    """Convert 0-1000 coordinate grid to pixel coordinates."""
    px = int(x / 1000 * VIEWPORT["width"])
    py = int(y / 1000 * VIEWPORT["height"])
    return max(0, min(px, VIEWPORT["width"])), max(0, min(py, VIEWPORT["height"]))


# ── System prompt for CU agent ────────────────────────────────────

def _build_system_prompt() -> str:
    today = date.today().strftime("%B %d, %Y")
    url_lines = "\n".join(
        f"  - {name}: {url}" for name, url in config.SITES.items()
    )
    return (
        f"You are a browser automation agent. Today's date is {today}.\n"
        "Complete the given subtask EXACTLY and FULLY as stated.\n"
        "* Do NOT stop after partial completion — if the subtask says to fill "
        "a form, fill ALL fields AND click submit.\n"
        "* Scroll down to see everything before deciding "
        "something isn't available.\n"
        "* If text is cut off, scroll or click into the element.\n"
        "* When done, state your result as plain text (no tools).\n"
        "* If stuck in a loop, try a different approach.\n"
        "* To replace existing text in a field, use type_text_at with "
        "clear_before_typing=true. Do NOT use control+a then backspace "
        "to clear fields — this selects the entire page, not just the field.\n\n"
        "LOCAL SERVICE URLs (use ONLY these, never external/public URLs):\n"
        f"{url_lines}\n"
        "* IMPORTANT: All websites run locally. NEVER navigate to public "
        "URLs like gitlab.com, reddit.com, amazon.com etc.\n"
    )


def _build_cu_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=1,       # DO NOT change -- values below 1.0 cause looping
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        tools=[
            types.Tool(
                computer_use=types.ComputerUse(
                    environment=types.Environment.ENVIRONMENT_BROWSER,
                ),
            ),
        ],
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        system_instruction=_build_system_prompt(),
    )


_client = genai.Client(api_key=config.GEMINI_API_KEY)
_ALLOWED_ORIGINS: set[str] = get_allowed_origins()


# ── Data models ────────────────────────────────────────────────────

class AgentAction(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0


class ActionLogEntry(BaseModel):
    step: int
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    url_before: str = ""
    url_after: str = ""
    requests_fired: int = 0
    post_requests: int = 0
    timestamp: float = 0.0


class CUResult(BaseModel):
    """Result from a single stateless CU subtask execution."""
    answer: str | None = None
    actions: list[AgentAction] = Field(default_factory=list)
    action_descriptions: list[str] = Field(default_factory=list)
    action_log: list[ActionLogEntry] = Field(default_factory=list)
    captured_requests: list[CapturedRequest] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    final_screenshot: bytes | None = None
    steps: int = 0
    wall_clock_seconds: float = 0.0


# ── Aggressive context pruning ─────────────────────────────────────

def _prune_old_turns(contents: list[types.Content]) -> None:
    """Keep only the initial message + last 2 full turns (4 entries).

    A "turn" = one assistant response + one user FunctionResponse.
    Old turns are replaced with a one-line action summary to maintain
    conversation coherence without the token cost.
    """
    if len(contents) <= 5:
        return

    # Extract action names from turns we're removing
    old_actions: list[str] = []
    for content in contents[1:-4]:
        for part in (content.parts or []):
            if part.function_call:
                old_actions.append(part.function_call.name)

    summary = types.Content(
        role="user",
        parts=[types.Part(
            text=f"[Prior actions taken: {', '.join(old_actions) or 'none'}]",
        )],
    )

    # Replace: [initial, ...old turns..., last 4] → [initial, summary, last 4]
    contents[:] = [contents[0], summary, *contents[-4:]]


# ── API call with retry + rate limit handling ──────────────────────

def _call_gemini(
    contents: list[types.Content],
    cu_config: types.GenerateContentConfig,
) -> tuple[Any, bool]:
    """Call Gemini CU model with retry logic.
    Returns (response, is_malformed).
    On 429: waits 60s for TPM reset then retries.
    """
    for attempt in range(API_MAX_RETRIES):
        delay = API_BACKOFF_DELAYS[min(attempt, len(API_BACKOFF_DELAYS) - 1)]
        try:
            response = _client.models.generate_content(
                model=MODEL, contents=contents, config=cu_config,
            )

            # Check for MALFORMED_FUNCTION_CALL
            if response.candidates:
                candidate = response.candidates[0]
                finish = getattr(candidate, "finish_reason", None)
                if finish and "MALFORMED_FUNCTION_CALL" in str(finish):
                    if attempt < API_MAX_RETRIES - 1:
                        print(f"    MALFORMED_FUNCTION_CALL, retry {attempt + 1}...")
                        time.sleep(delay)
                        continue
                    return response, True

            return response, False

        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            if attempt < API_MAX_RETRIES - 1:
                print(f"    Network error ({exc}), retry in {delay}s...")
                time.sleep(delay)
            else:
                raise
        except APIError as exc:
            if exc.code == 429 and attempt < API_MAX_RETRIES - 1:
                print(f"    API 429, waiting {TPM_RESET_WAIT}s for TPM reset...")
                time.sleep(TPM_RESET_WAIT)
            elif (
                exc.code in (500, 502, 503)
                and attempt < API_MAX_RETRIES - 1
            ):
                print(f"    API {exc.code}, retry in {delay}s...")
                time.sleep(delay)
            else:
                raise

    return None, False


# ── Action execution ──────────────────────────────────────────────

async def _execute_action(page: Page, name: str, args: dict) -> None:
    """Execute a single browser action and wait for the page to settle."""
    if name == "click_at":
        px, py = _norm_to_pixel(args["x"], args["y"])
        await page.mouse.click(px, py)

    elif name == "double_click_at":
        px, py = _norm_to_pixel(args["x"], args["y"])
        await page.mouse.dblclick(px, py)

    elif name == "hover_at":
        px, py = _norm_to_pixel(args["x"], args["y"])
        await page.mouse.move(px, py)

    elif name == "type_text_at":
        px, py = _norm_to_pixel(args["x"], args["y"])
        await page.mouse.click(px, py)
        if args.get("clear_before_typing"):
            # Triple-click selects text within the element (not the whole page)
            await page.mouse.click(px, py, click_count=3)
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.1)
        await page.keyboard.type(args["text"], delay=30)
        if args.get("press_enter"):
            await page.keyboard.press("Enter")

    elif name == "scroll_document":
        direction = args.get("direction", "down")
        if direction in ("up", "down"):
            dy = -400 if direction == "up" else 400
            await page.mouse.wheel(0, dy)
        else:
            dx = -400 if direction == "left" else 400
            await page.mouse.wheel(dx, 0)

    elif name == "scroll_at":
        px, py = _norm_to_pixel(args["x"], args["y"])
        await page.mouse.move(px, py)
        direction = args.get("direction", "down")
        magnitude = args.get("magnitude", 3)
        delta = magnitude * 100
        if direction in ("up", "down"):
            dy = -delta if direction == "up" else delta
            await page.mouse.wheel(0, dy)
        else:
            dx = -delta if direction == "left" else delta
            await page.mouse.wheel(dx, 0)

    elif name == "key_combination":
        keys = args.get("keys", "")
        key_map = {
            "tab": "Tab", "enter": "Enter", "escape": "Escape",
            "backspace": "Backspace", "delete": "Delete",
            "space": "Space", "arrowup": "ArrowUp",
            "arrowdown": "ArrowDown", "arrowleft": "ArrowLeft",
            "arrowright": "ArrowRight",
            "control": "Control", "ctrl": "Control",
            "alt": "Alt", "shift": "Shift", "meta": "Meta",
        }
        parts = keys.split("+")
        parts = [key_map.get(p.strip().lower(), p.strip()) for p in parts]
        keys = "+".join(parts)
        await page.keyboard.press(keys)

    elif name == "navigate":
        url = args.get("url", "")
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc not in _ALLOWED_ORIGINS:
            redirected = False
            url_host = parsed.netloc.lower().replace("www.", "")
            for sname, surl in config.SITES.items():
                clean = sname.replace("_", "")
                if clean in url_host or url_host.split(".")[0] in clean:
                    local_url = f"{surl}{parsed.path}"
                    if parsed.query:
                        local_url += f"?{parsed.query}"
                    print(f"    Redirecting {parsed.netloc} -> {local_url}")
                    await page.goto(
                        local_url, wait_until="domcontentloaded",
                    )
                    redirected = True
                    break
            if not redirected:
                print(f"    Blocked navigate to {parsed.netloc}")
            return
        await page.goto(url, wait_until="domcontentloaded")

    elif name == "go_back":
        await page.go_back(wait_until="domcontentloaded")

    elif name == "go_forward":
        await page.go_forward(wait_until="domcontentloaded")

    elif name == "wait_5_seconds":
        await asyncio.sleep(5)
        return

    elif name in ("search", "open_web_browser"):
        pass

    elif name == "drag_and_drop":
        sx, sy = _norm_to_pixel(args["x"], args["y"])
        dx, dy = _norm_to_pixel(
            args["destination_x"], args["destination_y"],
        )
        await page.mouse.move(sx, sy)
        await page.mouse.down()
        await page.mouse.move(dx, dy, steps=10)
        await page.mouse.up()

    elif name == "select_text":
        sx, sy = _norm_to_pixel(args["start_x"], args["start_y"])
        ex, ey = _norm_to_pixel(args["end_x"], args["end_y"])
        await page.mouse.move(sx, sy)
        await page.mouse.down()
        await page.mouse.move(ex, ey, steps=5)
        await page.mouse.up()

    elif name == "right_click_at":
        px, py = _norm_to_pixel(args["x"], args["y"])
        await page.mouse.click(px, py, button="right")

    else:
        print(f"    Unknown action: {name}, skipping")
        return

    # Post-action wait: only for page-changing actions
    if name in ("navigate", "go_back", "go_forward", "click_at",
                "double_click_at", "right_click_at"):
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        await asyncio.sleep(0.5)
    elif name == "type_text_at" and args.get("press_enter"):
        await asyncio.sleep(0.5)
    else:
        await asyncio.sleep(0.2)


# ── Action loop detection ─────────────────────────────────────────

def _detect_action_loop(actions: list[AgentAction], window: int = 6) -> bool:
    """Detect repetitive action patterns (e.g. ctrl+a/backspace cycles).

    Checks if the last `window` actions form a repeating pattern of
    length 2 or 3. Also detects excessive key_combination sequences.
    """
    if len(actions) < window:
        return False
    recent = [a.action for a in actions[-window:]]
    # Check if same 2-3 action sequence repeats across the window
    for pattern_len in (2, 3):
        if window % pattern_len != 0:
            continue
        pattern = recent[:pattern_len]
        if all(recent[i] == pattern[i % pattern_len] for i in range(window)):
            return True
    # Also detect: 4+ key_combination actions in the last 6
    key_combos = [a for a in actions[-window:] if a.action == "key_combination"]
    if len(key_combos) >= 4:
        return True
    return False


# ── Stuck detection ───────────────────────────────────────────────

def _screenshots_similar(a: bytes, b: bytes) -> bool:
    if not a or not b:
        return False
    ratio = abs(len(a) - len(b)) / max(len(a), len(b))
    return ratio < 0.02


async def _try_unstick(page: Page, attempt: int) -> None:
    if attempt == 0:
        print("    Stuck: pressing Escape")
        await page.keyboard.press("Escape")
    elif attempt == 1:
        print("    Stuck: scrolling down")
        await page.mouse.wheel(0, 400)
    else:
        profile = get_profile_for_url(page.url)
        home = profile.url if profile else page.url.split("/")[0]
        print(f"    Stuck: navigating home ({home})")
        try:
            await page.goto(home, wait_until="domcontentloaded")
        except Exception:
            pass
    await asyncio.sleep(1)


# ── Main entry point: stateless subtask executor ──────────────────

async def execute_cu_step(
    page: Page,
    subtask: str,
    max_actions: int = MAX_CU_ACTIONS,
) -> CUResult:
    """Execute a single subtask using the Gemini CU model.

    This is a STATELESS executor — fresh conversation each call, no
    history from prior subtasks. The orchestrator provides all strategic
    context; this function just does browser interactions.

    Args:
        page: Playwright page (already at the right URL)
        subtask: Specific instruction for the browser agent
        max_actions: Maximum browser actions before stopping

    Returns:
        CUResult with answer, actions taken, captured traffic, etc.
    """
    t0 = time.monotonic()
    result = CUResult()
    capture_ctx = start_capture(page)

    cu_config = _build_cu_config()

    # Build site context (auth hints, navigation tips)
    profile = get_profile_for_url(page.url)
    site_context = build_site_context(profile)

    # Take initial screenshot
    screenshot_png = await page.screenshot(type="png")
    screenshot_jpeg = _png_to_jpeg(screenshot_png, quality=80)
    prev_screenshot = screenshot_png
    same_count = 0
    recovery_attempts = 0

    # Build FRESH, MINIMAL initial context
    task_text = f"SUBTASK: {subtask}\n\nCurrent URL: {page.url}"
    if site_context:
        task_text += f"\n\n{site_context}"

    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=task_text),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg", data=screenshot_jpeg,
                    ),
                    media_resolution=MEDIA_RES_HIGH,
                ),
            ],
        ),
    ]

    try:
        loop_detected = False
        for action_num in range(max_actions):
            elapsed = time.monotonic() - t0
            if elapsed > SUBTASK_TIMEOUT:
                print(f"    Action {action_num + 1}: Timeout ({elapsed:.0f}s)")
                break

            # Rate limit pacing
            if action_num > 0:
                await asyncio.sleep(CU_PACING_DELAY)

            # Prune old turns to keep context small
            _prune_old_turns(contents)

            # Nudge for answer on last action
            if action_num == max_actions - 1:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=(
                            "This is your last action. Based on what you see, "
                            "provide your best answer NOW as plain text."
                        ))],
                    ),
                )

            response, is_malformed = _call_gemini(contents, cu_config)
            if response is None:
                print(f"    Action {action_num + 1}: API returned None, stopping")
                break
            if is_malformed:
                print(f"    Action {action_num + 1}: Malformed, skipping")
                continue

            # Track tokens
            if response.usage_metadata:
                step_in = response.usage_metadata.prompt_token_count or 0
                step_out = response.usage_metadata.candidates_token_count or 0
                result.input_tokens += step_in
                result.output_tokens += step_out
                print(
                    f"    [tokens] in={step_in:,} out={step_out:,} "
                    f"total={result.input_tokens + result.output_tokens:,}"
                )

            if not response.candidates:
                print(f"    Action {action_num + 1}: Empty response, stopping")
                break
            model_content = response.candidates[0].content
            if model_content is None:
                break

            contents.append(model_content)

            # Extract function calls and text
            function_calls: list[Any] = []
            text_response = None
            for part in (model_content.parts or []):
                if part.function_call:
                    function_calls.append(part.function_call)
                if part.text:
                    text_response = (text_response or "") + part.text

            # No function calls → agent is done with subtask
            if not function_calls:
                if text_response:
                    raw = text_response.strip()
                    lines = [l.strip() for l in raw.splitlines() if l.strip()]
                    result.answer = lines[-1] if lines else raw
                    print(f"    ANSWER: {result.answer[:80]}")
                result.actions.append(AgentAction(
                    action="done",
                    args={"answer": result.answer},
                    timestamp=time.time(),
                ))
                break

            # Execute all function calls
            safety_ack = None
            for fc in function_calls:
                action_name = fc.name
                action_args = dict(fc.args or {})

                safety = action_args.pop("safety_decision", None)
                if (
                    safety
                    and isinstance(safety, dict)
                    and safety.get("decision") == "require_confirmation"
                ):
                    safety_ack = "true"

                url_before = page.url
                reqs_before = len(capture_ctx.requests)

                print(f"    {action_name}({action_args})")
                result.actions.append(AgentAction(
                    action=action_name,
                    args=action_args,
                    timestamp=time.time(),
                ))
                desc = action_name
                if action_name == "type_text_at":
                    desc = f"type '{action_args.get('text', '')[:30]}'"
                elif action_name in ("click_at", "double_click_at"):
                    desc = f"{action_name}({action_args.get('x')},{action_args.get('y')})"
                elif action_name == "navigate":
                    desc = f"navigate to {action_args.get('url', '')[:50]}"
                elif action_name == "scroll_document":
                    desc = f"scroll {action_args.get('direction', 'down')}"
                result.action_descriptions.append(desc)

                await _execute_action(page, action_name, action_args)

                url_after = page.url
                reqs_after = len(capture_ctx.requests)
                new_reqs = capture_ctx.requests[reqs_before:reqs_after]
                post_count = sum(1 for r in new_reqs if r.method == "POST")

                result.action_log.append(ActionLogEntry(
                    step=action_num + 1,
                    action=action_name,
                    args=action_args,
                    url_before=url_before,
                    url_after=url_after,
                    requests_fired=reqs_after - reqs_before,
                    post_requests=post_count,
                    timestamp=time.time(),
                ))
                if post_count > 0:
                    print(f"      -> {post_count} POST request(s)")

            # Action loop detection — flag for inclusion in FunctionResponse
            loop_detected = _detect_action_loop(result.actions)
            if loop_detected:
                print(f"    ACTION LOOP detected after {len(result.actions)} actions")
                # Give it 3 more actions with guidance (injected via FunctionResponse)
                max_actions = min(action_num + 4, max_actions)

            # Take screenshot after all actions
            screenshot_png = await page.screenshot(type="png")

            # Stuck detection
            if _screenshots_similar(prev_screenshot, screenshot_png):
                same_count += 1
            else:
                same_count = 0
                recovery_attempts = 0
            prev_screenshot = screenshot_png

            if (
                same_count >= STUCK_THRESHOLD
                and recovery_attempts < STUCK_RECOVERY_MAX
            ):
                print(f"    Stuck ({same_count} similar screenshots)")
                await _try_unstick(page, recovery_attempts)
                recovery_attempts += 1
                same_count = 0
                screenshot_png = await page.screenshot(type="png")
                prev_screenshot = screenshot_png

            # Build FunctionResponse — MUST be PNG for CU model
            fn_response_parts: list[types.Part] = []
            for fc in function_calls:
                response_dict: dict[str, Any] = {"url": page.url}
                if safety_ack:
                    response_dict["safety_acknowledgement"] = safety_ack
                if loop_detected:
                    response_dict["warning"] = (
                        "LOOP DETECTED: You are repeating the same actions. "
                        "To clear a text field, use type_text_at with "
                        "clear_before_typing=true. If the form is filled, "
                        "click the Submit/Save button NOW."
                    )

                fn_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response=response_dict,
                            parts=[
                                types.FunctionResponsePart(
                                    inline_data=types.FunctionResponseBlob(
                                        mime_type="image/png",
                                        data=screenshot_png,
                                    ),
                                ),
                            ],
                        ),
                    ),
                )

            contents.append(
                types.Content(role="user", parts=fn_response_parts),
            )

        result.steps = len(result.actions)
        result.final_screenshot = screenshot_png
    finally:
        result.captured_requests = stop_capture(capture_ctx)

    result.wall_clock_seconds = time.monotonic() - t0
    return result
