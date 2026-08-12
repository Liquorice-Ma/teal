#!/bin/bash
# Kill the old xargs queue (current task keeps running and writes its
# CSV row itself), then wait for it to finish and relaunch with the
# reordered (repair-first) task list. Regex classes avoid self-matching.
cd "$(dirname "$0")"

pkill -f "xargs -P [1]" 2>/dev/null
echo "old xargs killed; waiting for the in-flight task to finish..."

while pgrep -f "run_tas[k]" > /dev/null; do sleep 120; done
echo "in-flight task done; relaunching serial batch (repair-first order)"

bash run_scale_check.sh --workers 1 >> scale_batch3.log 2>&1
