#!/bin/bash
# One-shot fix: kill deadlocked relay, clear failed rows, relaunch serial.
cd "$(dirname "$0")"

# kill the deadlocked relay (its wait loop matched its own cmdline)
pkill -f rerun_failed.sh 2>/dev/null
sleep 2

# remove failed rows and their done-marks so the tasks rerun
grep -E '\[OOM\]|\[FAIL\]|,inf$' scale_check.csv | while IFS=, read -r cfg rho seed repair mlu; do
    sed -i "/^2224-$cfg-$rho-$seed-$repair$/d" .scale-done
done
sed -i -E '/\[OOM\]|\[FAIL\]|,inf$/d' scale_check.csv

echo "after cleanup: $(($(wc -l < scale_check.csv)-1)) rows, $(wc -l < .scale-done) done"

nohup bash run_scale_check.sh --workers 1 > scale_batch3.log 2>&1 &
echo "serial relaunch PID=$!"
