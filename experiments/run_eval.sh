#!/bin/bash
# MorphNet Eval Runner — 7 websites × 20 tasks = 140 total
#
# Parallelism: 1 task per website at a time, up to --max-parallel websites at once (default 4).
# Each website gets its own Chrome instance on a unique port.
#
# Usage:
#   ./run_eval.sh                          # All 140 tasks
#   ./run_eval.sh --per-site 5             # 5 tasks per site (35 total)
#   ./run_eval.sh --site reddit            # Only Reddit tasks
#   ./run_eval.sh --site reddit --site youtube  # Reddit + YouTube
#   ./run_eval.sh --headless               # Headless mode
#   ./run_eval.sh --resume results/run_...  # Resume from a previous run (skip completed)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TASK_FILE="$SCRIPT_DIR/eval_140_tasks.json"
PER_SITE=20
SITE_FILTERS=()
HEADLESS="false"
MAX_SUBTASKS=15
RESUME_DIR=""
MAX_PARALLEL=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --per-site) PER_SITE="$2"; shift 2 ;;
        --site) SITE_FILTERS+=("$2"); shift 2 ;;
        --headless) HEADLESS="true"; shift ;;
        --max-subtasks) MAX_SUBTASKS="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --task-file) TASK_FILE="$2"; shift 2 ;;
        --resume) RESUME_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Use resume dir or create new
if [ -n "$RESUME_DIR" ]; then
    LOGDIR="$RESUME_DIR"
    echo "=== RESUMING from $LOGDIR ==="
else
    LOGDIR="$PROJECT_DIR/results/eval_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGDIR"
fi

# Build site filter string for Python
FILTER_PY=""
if [ ${#SITE_FILTERS[@]} -gt 0 ]; then
    FILTER_PY=$(printf "'%s'," "${SITE_FILTERS[@]}")
    FILTER_PY="[${FILTER_PY%,}]"
fi

echo "=== MorphNet Eval Runner ==="
echo "Task file:    $TASK_FILE"
echo "Per site:     $PER_SITE"
echo "Site filter:  ${SITE_FILTERS[*]:-all}"
echo "Headless:     $HEADLESS"
echo "Max subtasks: $MAX_SUBTASKS"
echo "Output:       $LOGDIR"
echo ""

# Write per-site task lists and get site names
SITES=$(cd "$PROJECT_DIR" && uv run python -c "
import json, sys
from collections import defaultdict
from pathlib import Path

with open('$TASK_FILE') as f:
    tasks = json.load(f)

site_filters = ${FILTER_PY:-[]}
per_site = $PER_SITE

if site_filters:
    tasks = [t for t in tasks if t['site'] in site_filters]

by_site = defaultdict(list)
for t in tasks:
    by_site[t['site']].append(t)

total = 0
for site, site_tasks in sorted(by_site.items()):
    selected = site_tasks[:per_site]
    total += len(selected)
    task_file = Path('$LOGDIR') / f'.tasks_{site}.json'
    task_file.write_text(json.dumps(selected, indent=2))
    print(site)

print(f'__TOTAL__:{total}')
")

TOTAL=$(echo "$SITES" | grep '__TOTAL__' | cut -d: -f2)
SITES=$(echo "$SITES" | grep -v '__TOTAL__')
SITE_COUNT=$(echo "$SITES" | wc -l | tr -d ' ')

echo "Sites: $SITE_COUNT  |  Total tasks: $TOTAL"
echo "Parallel:     $MAX_PARALLEL"
echo "Strategy: 1 task per website at a time, max $MAX_PARALLEL websites in parallel"
echo ""

# Show task list
for site in $SITES; do
    cd "$PROJECT_DIR" && uv run python -c "
import json, sys
s = sys.argv[1]; d = sys.argv[2]
with open(d + '/.tasks_' + s + '.json') as f:
    tasks = json.load(f)
print(f'  [{s}] {len(tasks)} tasks:')
for i, t in enumerate(tasks):
    print(f'    {i+1}. {t[\"label\"]}: {t[\"task\"][:72]}...')
" "$site" "$LOGDIR"
done
echo ""

# Run all tasks for one website sequentially
run_site() {
    local site=$1
    local port=$2
    local site_tasks="$LOGDIR/.tasks_${site}.json"

    # Load all task data once into a temp file (avoids repeated uv run python calls)
    local task_data
    task_data=$(cd "$PROJECT_DIR" && uv run python -c "
import json, sys
with open(sys.argv[1]) as f:
    tasks = json.load(f)
# Output one line per task: label\turl\ttask (tab-separated)
for t in tasks:
    # Escape newlines/tabs in task text
    task_text = t['task'].replace('\n', ' ').replace('\t', ' ')
    print(f\"{t['label']}\t{t['url']}\t{task_text}\")
" "$site_tasks")

    local count
    count=$(echo "$task_data" | wc -l | tr -d ' ')

    local i=0
    while IFS=$'\t' read -r label url task; do
        i=$((i + 1))
        local task_dir="$LOGDIR/$label"
        local logfile="$task_dir/run.log"

        # Skip if already completed (resume mode)
        if [ -f "$task_dir/result.json" ]; then
            echo "[SKIP] $label (already completed)"
            continue
        fi

        mkdir -p "$task_dir"
        echo "[START] $label ($i/$count for $site, port $port)"

        local start_time
        start_time=$(date +%s)

        # Run with output dir pointed at task_dir (trace + steps land here)
        cd "$PROJECT_DIR" && uv run python -m morphnet.session_manager \
            --url "$url" \
            --task "$task" \
            --headless "$HEADLESS" \
            --port "$port" \
            --max-subtasks "$MAX_SUBTASKS" \
            --output-dir "$task_dir" \
            > "$logfile" 2>&1

        local status=$?
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))

        # Write result.json safely via Python, passing answer through stdin (not shell vars)
        # This avoids all quoting issues with single/double quotes in answers
        cd "$PROJECT_DIR" && {
            grep -m1 "^Success:" "$logfile" 2>/dev/null || echo "Success: Unknown"
            grep -m1 "^Answer:" "$logfile" 2>/dev/null || echo "Answer: "
            grep -m1 "^Subtasks:" "$logfile" 2>/dev/null || echo "Subtasks: 0 | Actions: 0"
            grep -m1 "^Trace:" "$logfile" 2>/dev/null || echo "Trace: "
        } | uv run python -c "
import json, sys

lines = sys.stdin.read().strip().split('\n')
success_line = lines[0] if len(lines) > 0 else ''
answer_line = lines[1] if len(lines) > 1 else ''
stats_line = lines[2] if len(lines) > 2 else ''
trace_line = lines[3] if len(lines) > 3 else ''

# Parse success
success_str = success_line.split(':', 1)[-1].strip() if ':' in success_line else 'Unknown'
success = success_str == 'True'

# Parse answer — everything after 'Answer:  '
answer = answer_line.split(':', 1)[-1].strip() if ':' in answer_line else ''

# Parse subtasks/actions
subtasks = 0
actions = 0
if 'Subtasks:' in stats_line:
    parts = stats_line.split('|')
    for p in parts:
        p = p.strip()
        if p.startswith('Subtasks:'):
            try: subtasks = int(p.split(':')[1].strip())
            except: pass
        elif p.startswith('Actions:'):
            try: actions = int(p.split(':')[1].strip())
            except: pass

# Parse trace dir
trace_dir = trace_line.split(':', 1)[-1].strip() if ':' in trace_line else ''

result = {
    'label': '$label',
    'site': '$site',
    'success': success,
    'answer': answer,
    'exit_code': $status,
    'duration_seconds': $duration,
    'subtasks': subtasks,
    'actions': actions,
    'trace_dir': trace_dir,
}
with open('$task_dir/result.json', 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
"
        if [ $status -eq 0 ]; then
            local success_str
            success_str=$(grep -m1 "^Success:" "$logfile" 2>/dev/null | awk '{print $2}' || echo "Unknown")
            echo "[DONE]  $label — $success_str (${duration}s)"
        else
            echo "[FAIL]  $label (exit $status, ${duration}s)"
        fi

        # Brief pause between tasks on same site
        sleep 3
    done <<< "$task_data"

    echo "[SITE DONE] $site — $count tasks finished"
}

# Launch site runners with max parallelism cap
PORT=9301
PIDS=()
RUNNING=0
for site in $SITES; do
    run_site "$site" "$PORT" &
    PIDS+=($!)
    PORT=$((PORT + 1))
    RUNNING=$((RUNNING + 1))

    # Cap parallel site runners
    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
        # Wait for any child to finish before launching next
        wait -n 2>/dev/null || wait "${PIDS[0]}"
        RUNNING=$((RUNNING - 1))
    fi
done

echo "Launched $SITE_COUNT sites (max $MAX_PARALLEL parallel). Waiting for completion..."
echo ""
wait

echo ""
echo "=== All tasks complete ==="
echo ""

# Run analysis
cd "$PROJECT_DIR" && uv run python experiments/analyze_eval.py "$LOGDIR"
