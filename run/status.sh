#!/usr/bin/env bash
# One-screen status of GPU experiments. Run anywhere:
#   ssh autodl 'bash ~/MZN/teal/run/status.sh'

cd "$(dirname "$0")"

echo "======== $(date '+%m-%d %H:%M') 实验状态 ========"
echo
echo "--- 正在跑的训练进程 ---"
pgrep -af 'bin/python teal.py' | sed 's/--slice.*//' || echo "（无）"
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
echo
for f in pilot2.log pilot.log nohup-AB.log nohup-CD.log; do
    [ -f "$f" ] || continue
    echo "--- $f（最新3条 val + 最优）---"
    grep val_obj "$f" | tail -3
    grep val_obj "$f" | sort -k4 -n | head -1 | sed 's/^/best:  /'
    tr '\r' '\n' < "$f" | grep 'Testing: 100' | tail -1 | sed 's/^/test:  /'
    echo
done
if [ -f results-summary.csv ]; then
    n=$(( $(wc -l < results-summary.csv) - 1 ))
    echo "--- 批量实验：已完成 $n 个 run ---"
    tail -3 results-summary.csv
fi
echo "================================================"
