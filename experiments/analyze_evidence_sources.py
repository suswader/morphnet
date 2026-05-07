"""
Comprehensive analysis of evidence source usage across ALL MorphNet runs.

Analyzes trace.jsonl and steps/*.json files across every site and task to determine:
1. What evidence sources (AXTree, DOM, screenshot, traffic, etc.) are cited in reasoning
2. How often each source is decisive vs. supplementary
3. Differences by site, framework, task type, and module (planner/CU/reflector)
4. Edge cases where specific sources become critical
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

RESULTS_DIR = Path(__file__).parent.parent / "results"

# --- Evidence source classification ---

EVIDENCE_PATTERNS = {
    "axtree": [
        r"\baxtree\b", r"\bax.?tree\b", r"\baccessibility tree\b",
        r"\[\d+\]\s+\w+\"", r"\[\d+\]\s+(btn|txt|dd|lnk|chk|rad|tog|search|menu|tab|slider|num)",
        r"\belement\s+\[\d+\]", r"\[(\d+)\]\s+is\b", r"\[(\d+)\]\s+(button|link|text|input|heading)",
        r"\brole=\"", r"\brole=", r"\bheading\s+\"", r"\bnavigation\s+\"",
        r"\bbanner\b.*\bnavigation\b", r"\blistitem\b", r"\btextbox\b",
        r"\bcombobox\b", r"\bsearchbox\b",
    ],
    "dom": [
        r"\bdom\b", r"\bhtml\b", r"\b<\w+[>\s]", r"\bdata-\w+",
        r"\bclass=\"", r"\bform\b.*\binput\b", r"\binnerHTML\b",
        r"\bdocument\.\w+", r"\bdom summary\b", r"\bdom tree\b",
    ],
    "screenshot": [
        r"\bscreenshot\b", r"\bimage\b", r"\bvisual\b", r"\bSoM\b",
        r"\bset.of.mark\b", r"\bvisually\b", r"\bI can see\b",
        r"\bappears on screen\b", r"\bpage shows\b", r"\bvisible on\b",
    ],
    "url": [
        r"\burl\b", r"\bcurrent url\b", r"\burl changed\b", r"\bnavigat",
        r"https?://\S+", r"\bpath\b.*\bsegment\b",
    ],
    "traffic": [
        r"\btraffic\b", r"\bhttp\b", r"\bapi\b.*\b(call|response|request)\b",
        r"\bstatus\s+\d{3}\b", r"\b(2|4|5)\d{2}\b.*\bstatus\b",
        r"\bfetch\b", r"\bxhr\b", r"\bendpoint\b",
    ],
    "action_history": [
        r"\bstep\s+\d+\b", r"\bprevious\b.*\baction\b", r"\bhistory\b",
        r"\baction\s+\d+\b", r"\btried\b", r"\battempted\b",
        r"\blast\b.*\b(action|step|attempt)\b", r"\bfailed\b.*\bpreviously\b",
    ],
    "aria": [
        r"\baria\b", r"\balert\b", r"\bstatus\s+message\b",
        r"\baria-\w+", r"\brole=\"alert\"", r"\brole=\"status\"",
    ],
    "planning_tree": [
        r"\bplanning\s+tree\b", r"\bplan_\d+", r"\bbranch\b",
        r"\bprun(e|ed|ing)\b", r"\bsubtask\s+\d+\b.*\b(of|remaining)\b",
    ],
    "website_profile": [
        r"\bprofile\b", r"\binsight\b", r"\blearned\b",
        r"\bwebsite\s+(profile|insights?)\b", r"\bprior\b.*\bknowledge\b",
    ],
    "mcp_tools": [
        r"\b(mcp|executor|tool|graph)\b.*\b(available|found|verified|probationary)\b",
        r"\bsearch_trains\b", r"\bcheck_pnr\b",
        r"\b\[verified\]\b", r"\b\[probationary\]\b",
    ],
    "element_state": [
        r"\bvalue\b.*\b(set|match|empty|filled)\b", r"\bchecked\b",
        r"\bexpanded\b", r"\bcollapsed\b", r"\bdisabled\b",
        r"\bfocused\b", r"\brequired\b", r"\bselected\b",
    ],
    "page_content": [
        r"\btext\s+\"", r"\bshowing\b.*\bproducts?\b", r"\bprice\b",
        r"\brating\b", r"\btitle\b", r"\bname\b.*\bshows?\b",
        r"\bresults?\b.*\b(found|showing|listed)\b",
    ],
}


def classify_evidence(text: str) -> dict[str, int]:
    """Classify a text string into evidence source categories."""
    text_lower = text.lower()
    hits = {}
    for category, patterns in EVIDENCE_PATTERNS.items():
        count = 0
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            count += len(matches)
        if count > 0:
            hits[category] = count
    return hits


def extract_site_from_trace(trace_path: Path) -> str:
    """Extract site hostname from a trace.jsonl file."""
    try:
        with open(trace_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    detail = d.get("detail", {})
                    for key in ("url", "start_url"):
                        url = detail.get(key, "")
                        if url and url.startswith("http"):
                            host = urlparse(url).hostname or ""
                            if host.startswith("www."):
                                host = host[4:]
                            if host and "google" not in host and "firebase" not in host and "ampproject" not in host:
                                return host
                    # Check summary for URL
                    summary = d.get("summary", "")
                    urls = re.findall(r"https?://([^\s/]+)", summary)
                    for u in urls:
                        h = u.replace("www.", "")
                        if "google" not in h and "firebase" not in h:
                            return h
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return "unknown"


def extract_task_from_trace(trace_path: Path) -> str:
    """Extract task description from trace."""
    try:
        with open(trace_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    if d.get("event_type") == "task_started":
                        return d.get("detail", {}).get("task", d.get("summary", ""))[:120]
                    if "task" in d.get("detail", {}):
                        return d["detail"]["task"][:120]
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return "unknown"


def extract_framework_from_trace(trace_path: Path) -> str:
    """Extract detected frameworks from trace."""
    frameworks = set()
    try:
        with open(trace_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    detail = d.get("detail", {})
                    if "framework" in str(detail)[:500].lower():
                        fp = detail.get("framework_fingerprint", {})
                        if isinstance(fp, dict):
                            for fw in fp.get("frameworks", []):
                                frameworks.add(fw)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return ",".join(sorted(frameworks)) if frameworks else "unknown"


def analyze_trace(trace_path: Path) -> dict:
    """Analyze a single trace.jsonl file for evidence source usage."""
    results = {
        "site": extract_site_from_trace(trace_path),
        "task": extract_task_from_trace(trace_path),
        "framework": extract_framework_from_trace(trace_path),
        "path": str(trace_path),
        # Per-module evidence counts
        "planner_evidence": Counter(),
        "cu_evidence": Counter(),
        "reflector_evidence": Counter(),
        "overall_evidence": Counter(),
        # Reasoning analysis
        "planner_reasoning_texts": [],
        "cu_reasoning_texts": [],
        "reflector_reasoning_texts": [],
        # Screenshot usage
        "screenshot_provided_count": 0,
        "screenshot_not_provided_count": 0,
        # Event counts
        "total_plan_events": 0,
        "total_action_events": 0,
        "total_reflection_events": 0,
        "total_traffic_events": 0,
        # Task outcome
        "task_success": None,
        "total_subtasks": 0,
        "executor_used": False,
    }

    try:
        with open(trace_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                event_type = d.get("event_type", "")
                module = d.get("module", "")
                detail = d.get("detail", {})
                reasoning = d.get("reasoning") or ""
                evidence_list = d.get("evidence", [])
                summary = d.get("summary", "")

                # Combine all text for evidence classification
                all_text_parts = [reasoning, summary]

                # Extract evidence_sources from plan/action
                if isinstance(detail, dict):
                    plan = detail.get("plan", {})
                    action = detail.get("action", {})
                    if isinstance(plan, dict):
                        es = plan.get("evidence_sources", [])
                        if isinstance(es, list):
                            all_text_parts.extend(es)
                        r = plan.get("reasoning", "")
                        if r:
                            all_text_parts.append(r)
                    if isinstance(action, dict):
                        es = action.get("evidence_sources", [])
                        if isinstance(es, list):
                            all_text_parts.extend(es)
                        r = action.get("reasoning", "")
                        if r:
                            all_text_parts.append(r)

                # Extract from evidence array
                for ev in (evidence_list or []):
                    if isinstance(ev, dict):
                        all_text_parts.append(ev.get("description", ""))
                        all_text_parts.append(ev.get("source", ""))
                        all_text_parts.append(ev.get("raw_excerpt", "") or "")

                combined_text = " ".join(str(t) for t in all_text_parts if t)
                if not combined_text.strip():
                    continue

                hits = classify_evidence(combined_text)

                # Route to module
                if event_type in ("plan_made", "plan_decision", "tree_branch"):
                    results["total_plan_events"] += 1
                    for cat, count in hits.items():
                        results["planner_evidence"][cat] += count
                        results["overall_evidence"][cat] += count
                    if reasoning or (isinstance(plan, dict) and plan.get("reasoning")):
                        results["planner_reasoning_texts"].append(combined_text[:500])

                elif event_type in ("action_selected", "action_executed"):
                    results["total_action_events"] += 1
                    for cat, count in hits.items():
                        results["cu_evidence"][cat] += count
                        results["overall_evidence"][cat] += count
                    if reasoning or (isinstance(detail.get("action", {}), dict) and detail["action"].get("reasoning")):
                        results["cu_reasoning_texts"].append(combined_text[:500])

                elif event_type in ("deterministic_signals", "axtree_diff", "llm_action_verdict",
                                     "llm_action_eval", "subtask_reflection"):
                    results["total_reflection_events"] += 1
                    for cat, count in hits.items():
                        results["reflector_evidence"][cat] += count
                        results["overall_evidence"][cat] += count
                    if combined_text:
                        results["reflector_reasoning_texts"].append(combined_text[:500])

                elif event_type == "traffic_captured":
                    results["total_traffic_events"] += 1
                    results["overall_evidence"]["traffic"] += 1

                elif event_type == "screenshot_taken":
                    results["screenshot_provided_count"] += 1

                # Track outcomes
                if event_type in ("subtask_completed", "subtask_stop"):
                    results["total_subtasks"] += 1

                if event_type in ("task_completed", "task_budget_exhausted"):
                    if isinstance(detail, dict):
                        results["task_success"] = detail.get("success", detail.get("task_success"))

                if event_type in ("executor_success", "executor_fallback_to_cu"):
                    results["executor_used"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def analyze_step_files(run_dir: Path) -> dict:
    """Analyze steps/*.json files for a run."""
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return {}

    results = {
        "total_steps": 0,
        "steps_with_screenshot": 0,
        "steps_without_screenshot": 0,
        "evidence_in_responses": Counter(),
        "reasoning_samples": [],
        "axtree_char_sizes": [],
        "prompt_char_sizes": [],
    }

    for step_file in sorted(steps_dir.glob("action_*.json")):
        if step_file.name.endswith("_raw.jpg") or step_file.name.endswith("_som.jpg"):
            continue
        try:
            with open(step_file) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        results["total_steps"] += 1

        # Screenshot usage
        llm_input = d.get("llm_input", {})
        if llm_input.get("has_screenshot"):
            results["steps_with_screenshot"] += 1
        else:
            results["steps_without_screenshot"] += 1

        # AXTree size
        processed = d.get("processed", {})
        axtree_chars = processed.get("pruned_axtree_chars", 0)
        if axtree_chars:
            results["axtree_char_sizes"].append(axtree_chars)

        # Prompt size
        total_prompt = llm_input.get("total_prompt_chars", 0)
        if total_prompt:
            results["prompt_char_sizes"].append(total_prompt)

        # Evidence in response
        response = d.get("response", {})
        evidence_sources = response.get("evidence_sources", [])
        reasoning = response.get("reasoning", "")

        all_text = " ".join(str(e) for e in evidence_sources) + " " + str(reasoning)
        hits = classify_evidence(all_text)
        for cat, count in hits.items():
            results["evidence_in_responses"][cat] += count

        if reasoning:
            results["reasoning_samples"].append({
                "file": step_file.name,
                "reasoning": reasoning[:300],
                "evidence_sources": evidence_sources[:5],
                "has_screenshot": llm_input.get("has_screenshot", False),
            })

    return results


def classify_task_type(task: str) -> str:
    """Classify task into categories."""
    task_lower = task.lower()
    if any(w in task_lower for w in ["search", "find", "look for", "show me", "list"]):
        return "search"
    if any(w in task_lower for w in ["book", "order", "buy", "purchase", "add to cart"]):
        return "transactional"
    if any(w in task_lower for w in ["navigate", "go to", "open", "visit"]):
        return "navigation"
    if any(w in task_lower for w in ["price", "cost", "cheap", "expensive", "compare"]):
        return "comparison"
    if any(w in task_lower for w in ["check", "status", "verify", "confirm"]):
        return "verification"
    if any(w in task_lower for w in ["sort", "filter", "newest", "highest", "lowest"]):
        return "filtering"
    return "other"


def main():
    print("=" * 80)
    print("MORPHNET EVIDENCE SOURCE ANALYSIS")
    print("Analyzing ALL runs across ALL sites")
    print("=" * 80)

    # Find all trace files
    all_traces = list(RESULTS_DIR.rglob("trace.jsonl"))
    all_traces = [t for t in all_traces if t.stat().st_size > 1024]
    print(f"\nFound {len(all_traces)} trace files with data")

    # Analyze all traces
    all_results = []
    site_counter = Counter()

    for i, trace_path in enumerate(all_traces):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(all_traces)}...", file=sys.stderr)
        result = analyze_trace(trace_path)
        if result["site"] != "unknown" and result["total_plan_events"] + result["total_action_events"] > 0:
            all_results.append(result)
            site_counter[result["site"]] += 1

    print(f"\nAnalyzed {len(all_results)} valid runs")

    # Also analyze step files for a sample of runs
    step_results_by_site = defaultdict(list)
    step_dirs = list(RESULTS_DIR.glob("*/steps"))
    step_dirs = [d.parent for d in step_dirs if (d.parent / "trace.jsonl").exists()]

    print(f"\nAnalyzing {len(step_dirs)} run directories with step files...")
    for i, run_dir in enumerate(step_dirs):
        if i % 50 == 0:
            print(f"  Processing steps {i}/{len(step_dirs)}...", file=sys.stderr)
        site = extract_site_from_trace(run_dir / "trace.jsonl")
        sr = analyze_step_files(run_dir)
        if sr.get("total_steps", 0) > 0:
            sr["site"] = site
            step_results_by_site[site].append(sr)

    # ==================== REPORT ====================
    print("\n" + "=" * 80)
    print("SECTION 1: SITE COVERAGE")
    print("=" * 80)

    for site, count in site_counter.most_common():
        print(f"  {site:40s} {count:4d} runs")
    print(f"  {'TOTAL':40s} {sum(site_counter.values()):4d} runs")

    # --- Per-site evidence breakdown ---
    print("\n" + "=" * 80)
    print("SECTION 2: EVIDENCE SOURCE USAGE BY MODULE (ACROSS ALL SITES)")
    print("=" * 80)

    # Aggregate
    global_planner = Counter()
    global_cu = Counter()
    global_reflector = Counter()
    global_overall = Counter()

    for r in all_results:
        for cat, count in r["planner_evidence"].items():
            global_planner[cat] += count
        for cat, count in r["cu_evidence"].items():
            global_cu[cat] += count
        for cat, count in r["reflector_evidence"].items():
            global_reflector[cat] += count
        for cat, count in r["overall_evidence"].items():
            global_overall[cat] += count

    print("\n--- PLANNER (Orchestrator) Evidence Sources ---")
    total_p = sum(global_planner.values()) or 1
    for cat, count in global_planner.most_common():
        print(f"  {cat:25s} {count:6d} ({100*count/total_p:5.1f}%)")

    print("\n--- CU (Computer Use Agent) Evidence Sources ---")
    total_c = sum(global_cu.values()) or 1
    for cat, count in global_cu.most_common():
        print(f"  {cat:25s} {count:6d} ({100*count/total_c:5.1f}%)")

    print("\n--- REFLECTOR Evidence Sources ---")
    total_r = sum(global_reflector.values()) or 1
    for cat, count in global_reflector.most_common():
        print(f"  {cat:25s} {count:6d} ({100*count/total_r:5.1f}%)")

    print("\n--- OVERALL (All Modules) ---")
    total_o = sum(global_overall.values()) or 1
    for cat, count in global_overall.most_common():
        print(f"  {cat:25s} {count:6d} ({100*count/total_o:5.1f}%)")

    # --- Per-site breakdown ---
    print("\n" + "=" * 80)
    print("SECTION 3: EVIDENCE SOURCE USAGE BY SITE")
    print("=" * 80)

    site_evidence = defaultdict(lambda: Counter())
    site_runs = defaultdict(int)
    site_tasks = defaultdict(list)
    site_success = defaultdict(lambda: {"success": 0, "fail": 0, "unknown": 0})
    site_executor = defaultdict(int)

    for r in all_results:
        site = r["site"]
        site_runs[site] += 1
        site_tasks[site].append(r["task"])
        for cat, count in r["overall_evidence"].items():
            site_evidence[site][cat] += count
        if r["task_success"] is True:
            site_success[site]["success"] += 1
        elif r["task_success"] is False:
            site_success[site]["fail"] += 1
        else:
            site_success[site]["unknown"] += 1
        if r["executor_used"]:
            site_executor[site] += 1

    for site in sorted(site_evidence.keys()):
        runs = site_runs[site]
        ev = site_evidence[site]
        total = sum(ev.values()) or 1
        succ = site_success[site]
        exec_ct = site_executor[site]
        print(f"\n--- {site} ({runs} runs, success={succ['success']}, fail={succ['fail']}, executor_used={exec_ct}) ---")
        for cat, count in ev.most_common(10):
            print(f"  {cat:25s} {count:6d} ({100*count/total:5.1f}%)  [avg {count/runs:.1f}/run]")

    # --- Screenshot analysis ---
    print("\n" + "=" * 80)
    print("SECTION 4: SCREENSHOT USAGE ANALYSIS")
    print("=" * 80)

    for site, step_list in sorted(step_results_by_site.items()):
        total_steps = sum(s["total_steps"] for s in step_list)
        with_ss = sum(s["steps_with_screenshot"] for s in step_list)
        without_ss = sum(s["steps_without_screenshot"] for s in step_list)
        if total_steps == 0:
            continue
        print(f"\n  {site} ({len(step_list)} runs, {total_steps} total steps):")
        print(f"    Steps WITH screenshot:    {with_ss:5d} ({100*with_ss/total_steps:5.1f}%)")
        print(f"    Steps WITHOUT screenshot: {without_ss:5d} ({100*without_ss/total_steps:5.1f}%)")

        # AXTree sizes
        all_ax_sizes = []
        for s in step_list:
            all_ax_sizes.extend(s.get("axtree_char_sizes", []))
        if all_ax_sizes:
            print(f"    AXTree size: min={min(all_ax_sizes)}, median={sorted(all_ax_sizes)[len(all_ax_sizes)//2]}, max={max(all_ax_sizes)}, avg={sum(all_ax_sizes)/len(all_ax_sizes):.0f}")

    # --- Task type analysis ---
    print("\n" + "=" * 80)
    print("SECTION 5: EVIDENCE BY TASK TYPE")
    print("=" * 80)

    task_type_evidence = defaultdict(lambda: Counter())
    task_type_counts = Counter()

    for r in all_results:
        tt = classify_task_type(r["task"])
        task_type_counts[tt] += 1
        for cat, count in r["overall_evidence"].items():
            task_type_evidence[tt][cat] += count

    for tt, count in task_type_counts.most_common():
        ev = task_type_evidence[tt]
        total = sum(ev.values()) or 1
        print(f"\n--- {tt} tasks ({count} runs) ---")
        for cat, ct in ev.most_common(8):
            print(f"  {cat:25s} {ct:6d} ({100*ct/total:5.1f}%)")

    # --- Reasoning deep-dive: what patterns appear in reasoning ---
    print("\n" + "=" * 80)
    print("SECTION 6: REASONING PATTERN ANALYSIS (WHAT DRIVES DECISIONS)")
    print("=" * 80)

    # Collect all reasoning texts by module
    planner_reasons = []
    cu_reasons = []
    reflector_reasons = []

    for r in all_results:
        planner_reasons.extend(r["planner_reasoning_texts"])
        cu_reasons.extend(r["cu_reasoning_texts"])
        reflector_reasons.extend(r["reflector_reasoning_texts"])

    def analyze_reasoning_patterns(reasons, label):
        """Count key phrases in reasoning texts."""
        patterns = {
            "references [N] element ID": r"\[\d+\]",
            "mentions AXTree explicitly": r"\baxtree\b|\bax tree\b|\baccessibility\b",
            "mentions screenshot/visual": r"\bscreenshot\b|\bvisual\b|\bimage\b|\bI can see\b|\bscreen\b",
            "mentions DOM": r"\bdom\b|\bhtml\b|\b<\w+>",
            "mentions URL": r"\burl\b|https?://",
            "mentions previous action/step": r"\bstep\s+\d|\bprevious\b|\blast\b.*\baction\b|\bhistory\b",
            "mentions value/content": r"\bvalue\b|\bcontent\b|\btext\b.*\bshows?\b|\bsays?\b",
            "mentions page state": r"\bpage\b.*\b(state|shows?|has|contains?)\b|\bcurrent\b.*\bpage\b",
            "mentions error/failure": r"\berror\b|\bfail\b|\binvalid\b|\balert\b",
            "mentions form/field": r"\bform\b|\bfield\b|\binput\b|\btextbox\b",
            "mentions navigation": r"\bnavig\b|\burl changed\b|\bredirect\b",
            "mentions API/executor data": r"\bapi\b|\bexecutor\b|\bjson\b|\bresponse\b.*\bdata\b",
            "mentions planning tree": r"\bplan\b|\bbranch\b|\bprun\b|\btree\b",
            "mentions website insight": r"\binsight\b|\bprofile\b|\blearned\b",
        }

        print(f"\n  --- {label} ({len(reasons)} reasoning samples) ---")
        total = len(reasons) or 1
        for desc, pat in patterns.items():
            count = sum(1 for r in reasons if re.search(pat, r, re.IGNORECASE))
            pct = 100 * count / total
            bar = "#" * int(pct / 2)
            print(f"    {desc:45s} {count:5d}/{total} ({pct:5.1f}%) {bar}")

    analyze_reasoning_patterns(planner_reasons, "PLANNER reasoning")
    analyze_reasoning_patterns(cu_reasons, "CU AGENT reasoning")
    analyze_reasoning_patterns(reflector_reasons, "REFLECTOR reasoning")

    # --- Per-site reasoning patterns ---
    print("\n" + "=" * 80)
    print("SECTION 7: PER-SITE REASONING PATTERNS")
    print("=" * 80)

    site_planner_reasons = defaultdict(list)
    site_cu_reasons = defaultdict(list)
    for r in all_results:
        site_planner_reasons[r["site"]].extend(r["planner_reasoning_texts"])
        site_cu_reasons[r["site"]].extend(r["cu_reasoning_texts"])

    for site in sorted(site_planner_reasons.keys()):
        pr = site_planner_reasons[site]
        cr = site_cu_reasons[site]
        if len(pr) < 5 and len(cr) < 5:
            continue

        print(f"\n  === {site} (planner={len(pr)} samples, CU={len(cr)} samples) ===")

        # Key question: does screenshot matter?
        ss_refs_cu = sum(1 for r in cr if re.search(r"\bscreenshot\b|\bvisual\b|\bimage\b|\bI can see\b|\bscreen\b", r, re.IGNORECASE))
        axtree_refs_cu = sum(1 for r in cr if re.search(r"\[\d+\]", r))
        dom_refs_plan = sum(1 for r in pr if re.search(r"\bdom\b|\bhtml\b", r, re.IGNORECASE))
        axtree_refs_plan = sum(1 for r in pr if re.search(r"\[\d+\]|\baxtree\b|\bheading\b|\bnavigation\b", r, re.IGNORECASE))
        url_refs_plan = sum(1 for r in pr if re.search(r"\burl\b|https?://", r, re.IGNORECASE))
        api_refs_plan = sum(1 for r in pr if re.search(r"\bapi\b|\bexecutor\b|\bjson\b", r, re.IGNORECASE))

        total_cu = len(cr) or 1
        total_plan = len(pr) or 1
        print(f"    CU: screenshot refs={ss_refs_cu}/{total_cu} ({100*ss_refs_cu/total_cu:.0f}%), "
              f"element [N] refs={axtree_refs_cu}/{total_cu} ({100*axtree_refs_cu/total_cu:.0f}%)")
        print(f"    Planner: AXTree refs={axtree_refs_plan}/{total_plan} ({100*axtree_refs_plan/total_plan:.0f}%), "
              f"DOM refs={dom_refs_plan}/{total_plan} ({100*dom_refs_plan/total_plan:.0f}%), "
              f"URL refs={url_refs_plan}/{total_plan} ({100*url_refs_plan/total_plan:.0f}%), "
              f"API/executor refs={api_refs_plan}/{total_plan} ({100*api_refs_plan/total_plan:.0f}%)")

    # --- Edge cases and critical moments ---
    print("\n" + "=" * 80)
    print("SECTION 8: EDGE CASES — WHEN SPECIFIC SOURCES BECOME CRITICAL")
    print("=" * 80)

    # Find runs where screenshot references spike
    print("\n  --- Runs where screenshot was heavily referenced ---")
    for r in all_results:
        ss_hits = r["cu_evidence"].get("screenshot", 0)
        total_actions = r["total_action_events"] or 1
        if ss_hits > 5 and ss_hits / total_actions > 0.3:
            print(f"    {r['site']:30s} task='{r['task'][:60]}' screenshot_refs={ss_hits} actions={total_actions}")

    # Find runs where DOM was critical for planner
    print("\n  --- Runs where DOM was heavily referenced by planner ---")
    for r in all_results:
        dom_hits = r["planner_evidence"].get("dom", 0)
        total_plans = r["total_plan_events"] or 1
        if dom_hits > 3 and dom_hits / total_plans > 0.2:
            print(f"    {r['site']:30s} task='{r['task'][:60]}' dom_refs={dom_hits} plans={total_plans}")

    # Find runs where traffic/API was decisive
    print("\n  --- Runs where traffic/API data was heavily used ---")
    for r in all_results:
        traffic_hits = r["overall_evidence"].get("traffic", 0) + r["overall_evidence"].get("mcp_tools", 0)
        if traffic_hits > 10:
            print(f"    {r['site']:30s} task='{r['task'][:60]}' traffic+mcp={traffic_hits}")

    # Find runs where ARIA signals appeared
    print("\n  --- Runs where ARIA signals were present ---")
    for r in all_results:
        aria_hits = r["reflector_evidence"].get("aria", 0)
        if aria_hits > 2:
            print(f"    {r['site']:30s} task='{r['task'][:60]}' aria_refs={aria_hits}")

    # --- Step-level evidence from step files ---
    print("\n" + "=" * 80)
    print("SECTION 9: STEP-LEVEL EVIDENCE FROM ACTION FILES")
    print("=" * 80)

    for site, step_list in sorted(step_results_by_site.items()):
        ev = Counter()
        total_samples = 0
        for s in step_list:
            for cat, count in s.get("evidence_in_responses", {}).items():
                ev[cat] += count
            total_samples += s["total_steps"]
        if total_samples == 0:
            continue
        total = sum(ev.values()) or 1
        print(f"\n  {site} ({total_samples} action steps across {len(step_list)} runs):")
        for cat, count in ev.most_common(8):
            print(f"    {cat:25s} {count:6d} ({100*count/total:5.1f}%)")

    # --- Success correlation ---
    print("\n" + "=" * 80)
    print("SECTION 10: EVIDENCE PATTERNS IN SUCCESSFUL vs FAILED TASKS")
    print("=" * 80)

    success_evidence = Counter()
    fail_evidence = Counter()
    success_count = 0
    fail_count = 0

    for r in all_results:
        if r["task_success"] is True:
            success_count += 1
            for cat, count in r["overall_evidence"].items():
                success_evidence[cat] += count
        elif r["task_success"] is False:
            fail_count += 1
            for cat, count in r["overall_evidence"].items():
                fail_evidence[cat] += count

    print(f"\n  Successful tasks: {success_count}")
    total_s = sum(success_evidence.values()) or 1
    for cat, count in success_evidence.most_common(10):
        print(f"    {cat:25s} {count:6d} ({100*count/total_s:5.1f}%)  avg/run={count/max(success_count,1):.1f}")

    print(f"\n  Failed tasks: {fail_count}")
    total_f = sum(fail_evidence.values()) or 1
    for cat, count in fail_evidence.most_common(10):
        print(f"    {cat:25s} {count:6d} ({100*count/total_f:5.1f}%)  avg/run={count/max(fail_count,1):.1f}")

    # Difference analysis
    print("\n  --- Evidence source differential (success vs failure) ---")
    all_cats = set(list(success_evidence.keys()) + list(fail_evidence.keys()))
    diffs = []
    for cat in all_cats:
        s_pct = 100 * success_evidence.get(cat, 0) / total_s
        f_pct = 100 * fail_evidence.get(cat, 0) / total_f
        diffs.append((cat, s_pct, f_pct, s_pct - f_pct))
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    for cat, s_pct, f_pct, diff in diffs:
        marker = "▲" if diff > 0 else "▼"
        print(f"    {cat:25s} success={s_pct:5.1f}%  fail={f_pct:5.1f}%  diff={diff:+5.1f}% {marker}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
