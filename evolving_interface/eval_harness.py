"""WebArena Verified evaluation harness.

Task loading, structured answer formatting, scoring (AgentResponseEvaluator
+ NetworkEventEvaluator), and template-macro SR computation with 95% CI.
Follows the protocol in the WebArena Verified paper (NeurIPS 2025 SEA).
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs, unquote_plus

from google import genai
from google.genai import types as genai_types

from . import config

_gemini = genai.Client(api_key=config.GEMINI_API_KEY)
_FAST_MODEL = config.GEMINI_FAST_MODEL

TASKS_DIR = Path(__file__).parent.parent / "tasks"


# ═══════════════════════════════════════════════════════════════════
# 1. Task Loading + URL Resolution
# ═══════════════════════════════════════════════════════════════════

_URL_PLACEHOLDERS = {
    "__SHOPPING__": "shopping",
    "__SHOPPING_ADMIN__": "shopping_admin",
    "__REDDIT__": "reddit",
    "__GITLAB__": "gitlab",
}


def load_tasks(task_file: str = "verified-4site.json") -> list[dict]:
    """Load tasks from a JSON file in the tasks/ directory."""
    path = TASKS_DIR / task_file
    with open(path) as f:
        return json.load(f)


def resolve_url(raw: str) -> str:
    """Replace __SITE__ placeholders with actual URLs from config."""
    for placeholder, site_name in _URL_PLACEHOLDERS.items():
        if placeholder in raw:
            base = config.SITES.get(site_name, "")
            raw = raw.replace(placeholder, base, 1)
    return raw


def get_task_type(task: dict) -> str:
    """Extract task_type from the AgentResponseEvaluator eval entry."""
    for ev in task.get("eval", []):
        if ev.get("evaluator") == "AgentResponseEvaluator":
            return ev["expected"].get("task_type", "RETRIEVE")
    return "RETRIEVE"


def get_expected_status(task: dict) -> str:
    """Get the expected status from the AgentResponseEvaluator."""
    for ev in task.get("eval", []):
        if ev.get("evaluator") == "AgentResponseEvaluator":
            return ev["expected"].get("status", "SUCCESS")
    return "SUCCESS"


# ═══════════════════════════════════════════════════════════════════
# 2. Answer Formatting — agent output → structured JSON
# ═══════════════════════════════════════════════════════════════════

_ACTION_MAP = {
    "RETRIEVE": "retrieve",
    "MUTATE": "mutate",
    "NAVIGATE": "navigate",
}

_FAILURE_PATTERNS = {
    "NOT_FOUND_ERROR": [
        r"not found", r"doesn't exist", r"does not exist",
        r"no results?", r"couldn't find", r"could not find",
        r"no (?:such|matching)", r"0 results",
    ],
    "ACTION_NOT_ALLOWED_ERROR": [
        r"not allowed", r"cannot (?:be |)(?:done|performed|completed)",
        r"not possible", r"not supported", r"disabled",
    ],
    "PERMISSION_DENIED_ERROR": [
        r"permission denied", r"access denied", r"unauthorized",
        r"forbidden", r"not authorized", r"log\s*in required",
    ],
}


def format_answer(
    task: dict,
    final_answer: str | None,
    page_url: str,
) -> dict[str, Any]:
    """Convert the agent's free-text answer into the WebArena Verified
    structured response: {action, status, results, error_details}."""
    task_type = get_task_type(task)
    action = _ACTION_MAP.get(task_type, "retrieve")

    # No answer → error
    if final_answer is None:
        return {
            "action": action,
            "status": "UNKNOWN_ERROR",
            "results": None,
            "error_details": "Agent produced no answer",
        }

    answer = final_answer.strip()

    # Check if the answer indicates a failure
    error_status = _detect_error_status(answer)
    if error_status:
        return {
            "action": action,
            "status": error_status,
            "results": None,
            "error_details": answer[:500],
        }

    # MUTATE: agent says "completed" or similar
    if task_type == "MUTATE":
        return {
            "action": "mutate",
            "status": "SUCCESS",
            "results": None,
            "error_details": None,
        }

    # NAVIGATE: success if we got here
    if task_type == "NAVIGATE":
        return {
            "action": "navigate",
            "status": "SUCCESS",
            "results": None,
            "error_details": None,
        }

    # RETRIEVE: parse the answer into results array
    results_schema = None
    for ev in task.get("eval", []):
        if ev.get("evaluator") == "AgentResponseEvaluator":
            results_schema = ev.get("results_schema")
            break

    results = _parse_results(answer, task, results_schema)
    return {
        "action": "retrieve",
        "status": "SUCCESS",
        "results": results,
        "error_details": None,
    }


def _detect_error_status(answer: str) -> str | None:
    """Check if the agent's answer indicates a failure condition."""
    lower = answer.lower()
    for status, patterns in _FAILURE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower):
                return status
    return None


def _parse_results(
    answer: str,
    task: dict,
    schema: dict | None,
) -> list[Any]:
    """Parse the agent's free-text answer into a results array.
    Uses Gemini Flash for reliable extraction."""
    expected_data = None
    for ev in task.get("eval", []):
        if ev.get("evaluator") == "AgentResponseEvaluator":
            expected_data = ev["expected"].get("retrieved_data")
            break

    # Determine expected structure
    n_expected = len(expected_data) if expected_data else 1
    item_type = "string"
    if schema and "items" in schema:
        item_type = schema["items"].get("type", "string")

    # For single-value answers, try direct parsing first
    if n_expected == 1:
        if item_type == "number":
            val = _try_parse_number(answer)
            if val is not None:
                return [val]
        elif item_type == "string":
            # Strip common prefixes like "The answer is: "
            cleaned = re.sub(
                r'^(?:the\s+)?(?:answer|result|name|price|title)\s+'
                r'(?:is|are|was|were)\s*:?\s*',
                '', answer, flags=re.I,
            ).strip().strip('"\'')
            if cleaned:
                return [cleaned]
            return [answer]

    # Multi-value or complex: use Gemini Flash
    return _extract_results_via_gemini(answer, task, schema, n_expected)


def _try_parse_number(s: str) -> float | int | None:
    """Try to extract a number from a string."""
    cleaned = re.sub(r'[$€£¥,\s]', '', s)
    cleaned = re.sub(r'\b(USD|EUR|GBP)\b', '', cleaned, flags=re.I).strip()
    try:
        val = float(cleaned)
        return int(val) if val == int(val) else val
    except ValueError:
        return None


def _extract_results_via_gemini(
    answer: str,
    task: dict,
    schema: dict | None,
    n_expected: int,
) -> list[Any]:
    """Use Gemini Flash to extract structured results from free text."""
    schema_desc = json.dumps(schema, indent=2) if schema else "array of strings"
    prompt = (
        f"The agent was asked: {task['intent']}\n"
        f"The agent answered: {answer}\n\n"
        f"Extract the answer as a JSON array with {n_expected} element(s).\n"
        f"Expected schema: {schema_desc}\n\n"
        "Return ONLY a valid JSON array. No explanation."
    )
    try:
        resp = _gemini.models.generate_content(
            model=_FAST_MODEL,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=512,
            ),
        )
        text = resp.text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback: return answer as single-element array
    return [answer]


# ═══════════════════════════════════════════════════════════════════
# 3. Scoring
# ═══════════════════════════════════════════════════════════════════

def score_task(
    task: dict,
    agent_response: dict[str, Any],
    captured_requests: list | None = None,
) -> bool:
    """Score a task: all evaluators in the eval block must pass."""
    for ev in task.get("eval", []):
        evaluator = ev.get("evaluator", "")
        if evaluator == "AgentResponseEvaluator":
            if not _score_agent_response(ev, agent_response):
                return False
        elif evaluator == "NetworkEventEvaluator":
            if captured_requests is not None:
                if not _score_network_events(ev, captured_requests):
                    return False
    return True


def semantic_score_task(
    task: dict,
    agent_response: dict[str, Any],
) -> tuple[bool, str]:
    """Semantic evaluation: use Gemini to compare agent output vs expected.
    Returns (passed, reasoning). Runs alongside strict scoring."""
    for ev in task.get("eval", []):
        if ev.get("evaluator") == "AgentResponseEvaluator":
            expected = ev["expected"]
            break
    else:
        return False, "No evaluator found"

    task_type = expected.get("task_type", "RETRIEVE")
    expected_status = expected.get("status", "SUCCESS")

    # Status mismatch
    if agent_response.get("status") != expected_status:
        return False, (
            f"Status: {agent_response.get('status')} "
            f"vs {expected_status}"
        )

    # MUTATE/NAVIGATE: status match is sufficient
    if task_type in ("MUTATE", "NAVIGATE"):
        return (
            agent_response.get("status") == expected_status,
            "Status match",
        )

    # RETRIEVE: semantic comparison via Gemini
    expected_data = expected.get("retrieved_data")
    actual_results = agent_response.get("results")

    if expected_data is None:
        return actual_results is None, "Null check"
    if actual_results is None:
        return False, "No results from agent"

    prompt = (
        f"Task: {task['intent']}\n"
        f"Expected: {json.dumps(expected_data, default=str)}\n"
        f"Agent got: {json.dumps(actual_results, default=str)}\n\n"
        "Are these semantically equivalent? Consider:\n"
        "- Different formats ('$10.00' vs '10', 'March' vs '3')\n"
        "- Equivalent names ('John Smith' vs 'smith, john')\n"
        "- Approximate numbers (within 5%%)\n"
        "- Superset answers (agent returned more data than needed)\n"
        "Reply ONLY 'yes' or 'no' then a brief reason."
    )

    try:
        resp = _gemini.models.generate_content(
            model=_FAST_MODEL,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=prompt)],
            )],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=64,
            ),
        )
        text = resp.text.strip().lower()
        return text.startswith("yes"), resp.text.strip()
    except Exception as exc:
        return False, f"Error: {exc}"


def _score_agent_response(
    ev: dict,
    agent_response: dict[str, Any],
) -> bool:
    """Score against AgentResponseEvaluator."""
    expected = ev["expected"]

    # Action must match task type
    expected_action = _ACTION_MAP.get(expected["task_type"], "retrieve")
    if agent_response.get("action") != expected_action:
        return False

    # Status must match
    if agent_response.get("status") != expected["status"]:
        return False

    # Non-SUCCESS: just need status match, results must be null
    if expected["status"] != "SUCCESS":
        return agent_response.get("results") is None

    # RETRIEVE + SUCCESS: compare results
    if expected["task_type"] == "RETRIEVE":
        return _compare_results(
            expected.get("retrieved_data"),
            agent_response.get("results"),
            ev.get("results_schema"),
            ev.get("ordered", False),
        )

    # MUTATE + SUCCESS or NAVIGATE + SUCCESS: status match suffices
    return True


def _compare_results(
    expected_data: list | None,
    actual_results: list | None,
    schema: dict | None,
    ordered: bool,
) -> bool:
    """Compare expected vs actual results with type-aware normalization."""
    if expected_data is None:
        return actual_results is None
    if actual_results is None or len(actual_results) == 0:
        return False

    # Get format type from schema for normalization
    fmt = ""
    if schema and "items" in schema:
        items = schema["items"]
        fmt = items.get("format", "")
        # Handle object schemas: compare field by field
        if items.get("type") == "object":
            return _compare_object_results(
                expected_data, actual_results, items, ordered,
            )

    # Build normalized expected (with alternatives)
    expected_norm: list[list[str]] = []
    for item in expected_data:
        if isinstance(item, list):
            expected_norm.append([_normalize(v, fmt) for v in item])
        else:
            expected_norm.append([_normalize(item, fmt)])

    actual_norm = [_normalize(v, fmt) for v in actual_results]

    if ordered:
        if len(actual_norm) < len(expected_norm):
            return False
        for exp_alts, act in zip(expected_norm, actual_norm):
            if act not in exp_alts:
                return False
        return True
    else:
        # Unordered: each expected must match some actual
        used: set[int] = set()
        for exp_alts in expected_norm:
            matched = False
            for i, act in enumerate(actual_norm):
                if i in used:
                    continue
                if act in exp_alts:
                    used.add(i)
                    matched = True
                    break
            if not matched:
                return False
        return True


def _compare_object_results(
    expected_data: list,
    actual_results: list,
    items_schema: dict,
    ordered: bool,
) -> bool:
    """Compare results where each element is an object with properties."""
    props = items_schema.get("properties", {})
    if not props:
        return len(actual_results) >= len(expected_data)

    def obj_matches(expected_obj: Any, actual_obj: Any) -> bool:
        if not isinstance(expected_obj, dict) or not isinstance(actual_obj, dict):
            return _normalize(expected_obj) == _normalize(actual_obj)
        for key, prop_schema in props.items():
            fmt = prop_schema.get("format", "")
            exp_val = expected_obj.get(key)
            act_val = actual_obj.get(key)
            if exp_val is None:
                continue
            if isinstance(exp_val, list):
                if _normalize(act_val, fmt) not in [_normalize(v, fmt) for v in exp_val]:
                    return False
            elif _normalize(exp_val, fmt) != _normalize(act_val, fmt):
                return False
        return True

    if ordered:
        if len(actual_results) < len(expected_data):
            return False
        return all(obj_matches(e, a) for e, a in zip(expected_data, actual_results))

    used: set[int] = set()
    for exp in expected_data:
        found = False
        for i, act in enumerate(actual_results):
            if i in used:
                continue
            if obj_matches(exp, act):
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


# ── NetworkEventEvaluator ────────────────────────────────────────

def _score_network_events(
    ev: dict,
    captured_requests: list,
) -> bool:
    """Check captured HTTP traffic against NetworkEventEvaluator spec."""
    expected = ev["expected"]
    should_not_exist = ev.get("should_not_exist", False)
    last_event_only = ev.get("last_event_only", False)

    expected_method = (expected.get("http_method") or "").upper()
    expected_urls = expected.get("url")
    if isinstance(expected_urls, str):
        expected_urls = [expected_urls]
    resolved_urls = [resolve_url(u) for u in (expected_urls or [])]

    expected_status = expected.get("response_status")
    expected_query = expected.get("query_params") or {}
    expected_post = expected.get("post_data") or {}
    expected_headers = expected.get("headers") or {}

    ignored_qp = set(ev.get("ignored_query_params") or [])
    ignored_qp_pats = ev.get("ignored_query_params_patterns") or []

    # When last_event_only, check requests in reverse order and stop
    # at the first that matches the URL/method criteria
    request_list = list(reversed(captured_requests)) if last_event_only else captured_requests

    for req in request_list:
        # Method
        if expected_method and req.method.upper() != expected_method:
            continue

        # URL
        if resolved_urls and not _url_matches(req.url, resolved_urls):
            continue

        # For last_event_only: this is the most recent matching request.
        # Check all remaining criteria against it and return immediately.

        # Response status
        if expected_status is not None and expected_status != -1:
            if req.response_status != expected_status:
                if last_event_only:
                    return should_not_exist
                continue

        # Query params
        if expected_query:
            if not _query_matches(
                req.query_params, expected_query,
                ignored_qp, ignored_qp_pats,
            ):
                if last_event_only:
                    return should_not_exist
                continue

        # Headers (e.g. referer check)
        if expected_headers:
            if not _headers_match(req.headers, expected_headers):
                if last_event_only:
                    return should_not_exist
                continue

        # Post data
        if expected_post:
            if not _post_data_matches(req.body, expected_post):
                if last_event_only:
                    return should_not_exist
                continue

        # Match found
        return not should_not_exist

    # No match
    return should_not_exist


def _headers_match(
    actual_headers: dict[str, str],
    expected_headers: dict[str, str],
) -> bool:
    """Check that actual headers contain all expected header values.
    Expected values are resolved (site placeholders replaced) and
    matched as substrings of the actual header value."""
    for key, exp_val in expected_headers.items():
        if exp_val is None:
            continue
        resolved_val = resolve_url(str(exp_val))
        actual_val = actual_headers.get(key.lower(), "")
        if not actual_val:
            # Try case-insensitive lookup
            for ak, av in actual_headers.items():
                if ak.lower() == key.lower():
                    actual_val = av
                    break
        if resolved_val not in actual_val:
            return False
    return True


def _url_matches(actual_url: str, expected_urls: list[str]) -> bool:
    """Match URL: exact, list-of-alternatives, or regex."""
    actual_path = urlparse(actual_url).path.rstrip("/")
    for exp in expected_urls:
        exp_parsed = urlparse(exp)
        exp_path = exp_parsed.path.rstrip("/")

        # Regex match (starts with ^)
        if exp.startswith("^") or exp_path.startswith("^"):
            pattern = resolve_url(exp)
            # Extract just the path part for regex
            pat_parsed = urlparse(pattern)
            pat_path = pat_parsed.path
            if re.search(pat_path, actual_path):
                return True
            # Also try full URL match
            if re.search(pattern, actual_url):
                return True
            continue

        # Exact path match
        if actual_path == exp_path:
            return True
        # Full URL match
        if actual_url.rstrip("/") == exp.rstrip("/"):
            return True

    return False


def _query_matches(
    actual_qp: dict[str, Any],
    expected_qp: dict[str, Any],
    ignored: set[str],
    ignored_patterns: list[str],
) -> bool:
    """Check that actual query params contain all expected ones."""
    for key, exp_vals in expected_qp.items():
        if key in ignored:
            continue
        if any(re.match(pat, key) for pat in ignored_patterns):
            continue
        if not isinstance(exp_vals, list):
            exp_vals = [str(exp_vals)]
        actual_vals = actual_qp.get(key, [])
        if not isinstance(actual_vals, list):
            actual_vals = [str(actual_vals)]
        # Check that at least one expected value appears in actual
        if not any(str(ev) in [str(av) for av in actual_vals] for ev in exp_vals):
            return False
    return True


def _post_data_matches(
    actual_body: Any,
    expected_post: dict[str, Any],
) -> bool:
    """Check POST body matches expected fields (supports JSONPath keys).
    Handles JSON, URL-encoded form data, and raw string bodies."""
    if actual_body is None:
        return False
    if isinstance(actual_body, str):
        # Try parsing as JSON first
        try:
            actual_body = json.loads(actual_body)
        except (json.JSONDecodeError, TypeError):
            # Try URL-encoded form data (e.g. milestone%5Btitle%5D=code+review)
            try:
                parsed_form = parse_qs(actual_body, keep_blank_values=True)
                if parsed_form:
                    # Flatten single-value lists
                    actual_body = {
                        k: v[0] if len(v) == 1 else v
                        for k, v in parsed_form.items()
                    }
                else:
                    raise ValueError("empty parse")
            except (ValueError, TypeError):
                # Fallback: check as URL-decoded raw string
                decoded = unquote_plus(actual_body)
                for key, val in expected_post.items():
                    if str(val) not in decoded:
                        return False
                return True

    if not isinstance(actual_body, dict):
        return False

    for key, exp_val in expected_post.items():
        # Handle JSONPath-style keys like "$.note.note"
        actual_val = _resolve_jsonpath(actual_body, key)
        if actual_val is None:
            return False
        # Value can be a regex pattern
        exp_str = str(exp_val)
        act_str = str(actual_val)
        if exp_str.startswith("^"):
            if not re.search(exp_str, act_str):
                return False
        elif _normalize(act_str) != _normalize(exp_str):
            return False
    return True


def _resolve_jsonpath(obj: dict, path: str) -> Any:
    """Simple JSONPath resolver for $.a.b style paths."""
    if path.startswith("$."):
        path = path[2:]
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ═══════════════════════════════════════════════════════════════════
# 4. Value Normalization
# ═══════════════════════════════════════════════════════════════════

_MONTHS = {
    "january": "1", "february": "2", "march": "3", "april": "4",
    "may": "5", "june": "6", "july": "7", "august": "8",
    "september": "9", "october": "10", "november": "11", "december": "12",
    "jan": "1", "feb": "2", "mar": "3", "apr": "4",
    "jun": "6", "jul": "7", "aug": "8", "sep": "9",
    "oct": "10", "nov": "11", "dec": "12",
}


def _normalize(value: Any, fmt: str = "") -> str:
    """Normalize a value for comparison based on format type."""
    if value is None:
        return ""
    s = str(value).strip()

    if fmt == "currency":
        cleaned = re.sub(r'[$€£¥,\s]', '', s)
        cleaned = re.sub(r'\b(USD|EUR|GBP|JPY)\b', '', cleaned, flags=re.I).strip()
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            pass

    elif fmt == "month":
        lower = s.lower().strip(".")
        if lower in _MONTHS:
            return _MONTHS[lower]
        try:
            return str(int(s))
        except ValueError:
            pass

    elif fmt == "date":
        # Try common date normalizations
        for month_name, month_num in _MONTHS.items():
            s = re.sub(
                rf'\b{month_name}\b', month_num, s, flags=re.I,
            )
        # Strip ordinal suffixes (1st, 2nd, 3rd)
        s = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', s)

    elif fmt == "url":
        s = s.rstrip("/").lower()
        s = re.sub(r'^https?://', '', s)

    elif fmt == "duration":
        # Normalize time expressions to minutes
        total_min = 0
        hours = re.findall(r'(\d+)\s*(?:hours?|hrs?|h)', s, re.I)
        mins = re.findall(r'(\d+)\s*(?:minutes?|mins?|m(?!onth))', s, re.I)
        for h in hours:
            total_min += int(h) * 60
        for m in mins:
            total_min += int(m)
        if total_min > 0:
            return str(total_min)

    elif fmt == "distance":
        cleaned = re.sub(r'\s*(km|mi|miles?|kilometers?|metres?|meters?|m)\b', '', s, flags=re.I)
        cleaned = cleaned.strip()
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            pass

    # Default: Unicode NFC, lowercase, collapse whitespace
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    # Strip trailing punctuation
    s = s.rstrip(".,;:!?")
    return s


# ═══════════════════════════════════════════════════════════════════
# 5. Metrics — Template-Macro SR with 95% CI
# ═══════════════════════════════════════════════════════════════════

_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021,
    50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
    150: 1.976, 200: 1.972,
}


def _t_critical_975(df: int) -> float:
    """t_{0.975, df} for 95% two-sided CI. Lookup with interpolation."""
    if df >= 200:
        return 1.96
    keys = sorted(_T_CRIT_975.keys())
    for i, k in enumerate(keys):
        if df <= k:
            if df == k:
                return _T_CRIT_975[k]
            if i == 0:
                return _T_CRIT_975[k]
            lo, hi = keys[i - 1], k
            frac = (df - lo) / (hi - lo)
            return _T_CRIT_975[lo] + frac * (_T_CRIT_975[hi] - _T_CRIT_975[lo])
    return 1.96


def compute_sr_tmpl(
    results: list[dict[str, Any]],
) -> tuple[float, float, int]:
    """Compute template-macro success rate with 95% t-interval.

    Returns (sr_tmpl_pct, ci_95_pct, n_templates).
    """
    # Group by intent_template_id
    by_template: dict[int, list[bool]] = {}
    for r in results:
        tid = r["intent_template_id"]
        by_template.setdefault(tid, []).append(r["passed"])

    T = len(by_template)
    if T == 0:
        return 0.0, 0.0, 0

    # Per-template success rates
    sr_per_t = [
        sum(passes) / len(passes)
        for passes in by_template.values()
    ]
    sr_tmpl = sum(sr_per_t) / T

    # Variance and CI
    if T < 2:
        return sr_tmpl * 100, 0.0, T

    variance = sum((p - sr_tmpl) ** 2 for p in sr_per_t) / (T - 1)
    se = math.sqrt(variance / T)

    # t critical value for 95% two-sided
    t_crit = _t_critical_975(T - 1)

    ci = t_crit * se
    return sr_tmpl * 100, ci * 100, T


def compute_per_site_sr(
    results: list[dict[str, Any]],
) -> dict[str, tuple[float, float, int]]:
    """Compute template-macro SR per site."""
    by_site: dict[str, list[dict]] = {}
    for r in results:
        site = r.get("site", r.get("sites", ["unknown"])[0]
                      if isinstance(r.get("sites"), list) else "unknown")
        by_site.setdefault(site, []).append(r)
    return {
        site: compute_sr_tmpl(site_results)
        for site, site_results in sorted(by_site.items())
    }


def compute_per_type_sr(
    results: list[dict[str, Any]],
) -> dict[str, tuple[int, int, float]]:
    """Compute simple pass/total/pct per task type."""
    by_type: dict[str, list[bool]] = {}
    for r in results:
        tt = r.get("task_type", "UNKNOWN")
        by_type.setdefault(tt, []).append(r["passed"])
    return {
        tt: (sum(passes), len(passes), sum(passes) / len(passes) * 100)
        for tt, passes in sorted(by_type.items())
    }


# ═══════════════════════════════════════════════════════════════════
# 6. Output
# ═══════════════════════════════════════════════════════════════════

def print_results_table(
    run_name: str,
    results: list[dict[str, Any]],
) -> list[str]:
    """Print and return a formatted results summary."""
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    sr, ci, n_tmpl = compute_sr_tmpl(results)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    p(f"\n{'='*70}")
    p(f"  {run_name or 'EVALUATION RESULTS'}")
    p(f"{'='*70}")
    p(f"  Tasks:      {passed}/{total} passed ({passed/total*100:.1f}%)")
    p(f"  SR_tmpl:    {sr:.1f}% ± {ci:.1f}%  ({n_tmpl} templates)")
    p()

    # Per-site breakdown
    p("  Per-site SR_tmpl:")
    for site, (s_sr, s_ci, s_t) in compute_per_site_sr(results).items():
        p(f"    {site:20s} {s_sr:5.1f}% ± {s_ci:.1f}%  ({s_t} templates)")
    p()

    # Per task-type breakdown
    p("  Per task-type:")
    for tt, (ok, tot, pct) in compute_per_type_sr(results).items():
        p(f"    {tt:20s} {ok:3d}/{tot:3d}  ({pct:.1f}%)")
    p()

    return lines


def save_run(
    results: list[dict[str, Any]],
    run_name: str,
    summary_lines: list[str],
) -> tuple[str, str]:
    """Save results JSONL + summary text."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = run_name.replace(" ", "_").lower() if run_name else "eval"

    jsonl_path = config.RESULTS_DIR / f"{tag}_{ts}.jsonl"
    txt_path = config.RESULTS_DIR / f"{tag}_{ts}_summary.txt"

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    txt_path.write_text("\n".join(summary_lines))
    print(f"  Results: {jsonl_path}")
    print(f"  Summary: {txt_path}")
    return str(jsonl_path), str(txt_path)
