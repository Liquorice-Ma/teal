#!/bin/bash
# Capacity probe under the paper's main protocol (run_overall_valrepair.sh).
#
# Purpose: the paper reports MLU 1.6-2.0 but never states the ISL capacity,
# and the checked-in Starlink2272.json carries 1000 while the demand volume
# implies 2000.  This probe reproduces the recorded untrained reference
# (rho=0.3, seed 0, --repair-input nbr -> 1.644) so that the baseline
# capacity is identified before any sweep is launched.  It also times one
# task and samples GPU memory, since no measurement of either survives.
#
# Usage: bash probe_cap.sh [capacity]   (default: whatever is in the json)

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 --admm-steps 2 --repair-input nbr $S"

if [ -n "$1" ]; then
    $PY prepare_starlink.py --size-x 22 --size-y 72 --capacity "$1" \
        > /dev/null 2>&1
fi
echo "capacity in topology: $($PY -c "
import json
d = json.load(open('../topologies/Starlink2272.json'))
print(sorted({l['capacity'] for l in d['links']}))
")"

# sample GPU memory every 5 s while the task runs
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -l 5 > probe_gpu.log 2>&1 &
SMI=$!

echo "start $(date +%H:%M:%S)"
SECONDS=0
$PY teal.py $COMMON --mask-mode embed --hist-len 3 \
    --obs-ratio 0.3 --seed 0 --epochs 0 --num-restart 1 2>&1 \
    | tr '\r' '\n' | tail -25
echo "elapsed ${SECONDS}s, end $(date +%H:%M:%S)"

kill $SMI 2>/dev/null
echo "peak GPU MiB (whole card, includes other users): $(sort -n probe_gpu.log | tail -1)"
