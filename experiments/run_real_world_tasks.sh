#!/bin/bash
# Run MorphNet tasks from the live-sites-50 task definition file.
#
# Parallelism model: ONE task per website at a time (avoid bot detection),
# but all websites run in parallel. So 5 websites = 5 concurrent Chrome instances.
# Each website's tasks run sequentially.
#
# Usage:
#   ./run_live_tasks.sh                     # Run all 50 tasks (10 per site)
#   ./run_live_tasks.sh --per-site 3        # Run 3 tasks per site (15 total)
#   ./run_live_tasks.sh --site lego         # Run only LEGO tasks
#   ./run_live_tasks.sh --headless          # Run headless (default: visible)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="$SCRIPT_DIR/real_world_tasks.json"
PER_SITE=10
SITE_FILTER=""
HEADLESS="false"
MAX_SUBTASKS=15

while [[ $# -gt 0 ]]; do
    case $1 in
        --per-site) PER_SITE="$2"; shift 2 ;;
        --site) SITE_FILTER="$2"; shift 2 ;;
        --headless) HEADLESS="true"; shift ;;
        --max-subtasks) MAX_SUBTASKS="$2"; shift 2 ;;
        --task-file) TASK_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LOGDIR="$SCRIPT_DIR/results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "=== MorphNet Live-Site Task Runner ==="
echo "Task file:    $TASK_FILE"
echo "Per site:     $PER_SITE"
echo "Site filter:  ${SITE_FILTER:-all}"
echo "Headless:     $HEADLESS"
echo "Log dir:      $LOGDIR"
echo ""
echo "Strategy: 1 task per website at a time, all websites in parallel"
echo ""

# Write per-site task lists to temp files and get site names
SITES=$(uv run python -c "
import json, sys

with open('$TASK_FILE') as f:
    tasks = json.load(f)

site_filter = '$SITE_FILTER'
per_site = $PER_SITE

if site_filter:
    tasks = [t for t in tasks if t['site'] == site_filter]

from collections import defaultdict
by_site = defaultdict(list)
for t in tasks:
    by_site[t['site']].append(t)

for site, site_tasks in by_site.items():
    selected = site_tasks[:per_site]
    with open('$LOGDIR/.tasks_' + site + '.json', 'w') as f:
        json.dump(selected, f)
    print(site)

total = sum(min(len(v), per_site) for v in by_site.values())
print(f'__TOTAL__:{total}')
")

TOTAL=$(echo "$SITES" | grep '__TOTAL__' | cut -d: -f2)
SITES=$(echo "$SITES" | grep -v '__TOTAL__')
SITE_COUNT=$(echo "$SITES" | wc -l | tr -d ' ')

echo "Sites: $SITE_COUNT  |  Total tasks: $TOTAL"
echo ""

# Print all tasks grouped by site
for site in $SITES; do
    uv run python -c "
import json, sys
s = sys.argv[1]
d = sys.argv[2]
with open(d + '/.tasks_' + s + '.json') as f:
    tasks = json.load(f)
print(f'  [{s}] {len(tasks)} tasks:')
for i, t in enumerate(tasks):
    print(f'    {i+1}. {t[\"label\"]}: {t[\"task\"][:75]}...')
" "$site" "$LOGDIR"
done
echo ""

# Run all tasks for one website sequentially, one Chrome instance at a time
run_site() {
    local site=$1
    local port=$2
    local site_tasks="$LOGDIR/.tasks_${site}.json"
    local count

    count=$(uv run python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$site_tasks")

    for ((i=0; i<count; i++)); do
        local label url task
        label=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]['label'])" "$site_tasks" "$i")
        url=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]['url'])" "$site_tasks" "$i")
        task=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]['task'])" "$site_tasks" "$i")

        local logfile="$LOGDIR/${label}.log"

        echo "[START] $label ($((i+1))/$count for $site, port $port)"

        uv run python -m morphnet.session_manager \
            --url "$url" \
            --task "$task" \
            --headless "$HEADLESS" \
            --port "$port" \
            --max-subtasks "$MAX_SUBTASKS" \
            > "$logfile" 2>&1

        local status=$?
        if [ $status -eq 0 ]; then
            echo "[DONE]  $label"
        else
            echo "[FAIL]  $label (exit $status)"
        fi

        # Brief pause between tasks on same site to be polite
        sleep 3
    done

    echo "[SITE COMPLETE] $site — all $count tasks finished"
}

# Launch one runner per website, each on its own port
PORT=9301
PIDS=()
for site in $SITES; do
    run_site "$site" "$PORT" &
    PIDS+=($!)
    PORT=$((PORT + 1))
done

echo "All $SITE_COUNT sites launched (1 task at a time per site). Waiting..."
echo ""
wait

echo ""
echo "=== All tasks complete ==="
echo "Results in: $LOGDIR"
echo ""

# Print summary
PASS=0
FAIL=0
for site in $SITES; do
    echo "--- $site ---"
    # Read task labels from the per-site task file to match log files correctly
    labels=$(uv run python -c "
import json, sys
with open(sys.argv[1]) as f:
    for t in json.load(f):
        print(t['label'])
" "$LOGDIR/.tasks_${site}.json" 2>/dev/null)
    for label in $labels; do
        f="$LOGDIR/${label}.log"
        [ -f "$f" ] || continue
        if grep -q "Success: True" "$f" 2>/dev/null; then
            answer=$(grep "Answer:" "$f" 2>/dev/null | head -1 | cut -c1-120)
            echo "  PASS  $label"
            [ -n "$answer" ] && echo "        $answer"
            PASS=$((PASS + 1))
        else
            error=$(tail -3 "$f" 2>/dev/null | head -1 | cut -c1-120)
            echo "  FAIL  $label"
            [ -n "$error" ] && echo "        $error"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "=== Summary: $PASS passed, $FAIL failed out of $TOTAL ==="
