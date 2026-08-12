#!/bin/bash
# Kill deadlocked watcher + stray teal, then relaunch serial batch.
cd "$(dirname "$0")"

pkill -f "wait_relaunch" 2>/dev/null
sleep 2
pkill -f "teal.py" 2>/dev/null
sleep 3

# count missing
missing=0
for cfg in ours untrained; do
  for rho in 0.3 0.05; do
    for repair in repair norepair; do
      for seed in 0 1 2; do
        grep -q "^2224-$cfg-$rho-$seed-$repair" .scale-done 2>/dev/null || missing=$((missing+1))
      done
    done
  done
done
echo "missing: $missing cells"

nohup bash run_scale_check.sh --workers 1 > scale_batch4.log 2>&1 &
echo "relaunched, pid=$!"
sleep 10
echo "teal procs: $(pgrep -f teal.py | wc -l)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
