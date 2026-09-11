#!/bin/bash
# Trained-vs-untrained probe at one enlarged capacity, under the paper's
# main protocol (identical to run_overall_valrepair.sh).
#
# The baseline capacity is 2000 (verified: untrained rho=0.3 seed0 with
# --repair-input nbr reproduces the recorded 1.644).  Enlarging capacity
# rescales MLU by 1/k for a fixed route, so the informative question is
# whether the trained-vs-untrained relation survives once the policy's own
# MLU drops below 1 and the feasibility projection
# util = 1 + (edge_flow/capacity - 1).relu()  stops firing.
#
# Usage: bash probe_bigcap.sh <capacity> [rho] [seed]

set -u
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CAP=${1:?usage: probe_bigcap.sh <capacity> [rho] [seed]}
RHO=${2:-0.3}
SEED=${3:-0}

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 --admm-steps 2 --repair-input nbr $S"
ARCH="--mask-mode embed --hist-len 3"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"

$PY prepare_starlink.py --size-x 22 --size-y 72 --capacity "$CAP" \
    > /dev/null 2>&1
echo "===== capacity $CAP  rho $RHO  seed $SEED ====="
echo "topology capacity: $($PY -c "
import json
d = json.load(open('../topologies/Starlink2272.json'))
print(sorted({l['capacity'] for l in d['links']}))
")"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    -l 10 > probe_gpu_$CAP.log 2>&1 &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

mlu_of() {   # mlu_of <mode args...>
    $PY teal.py $COMMON $ARCH --obs-ratio "$RHO" --seed "$SEED" "$@" 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2
}

SECONDS=0
U=$(mlu_of $UNTRAIN)
T_U=$SECONDS
printf 'untrained  MLU=%-10s  %ss\n' "${U:-FAILED}" "$T_U"

SECONDS=0
T=$(mlu_of $TRAIN)
T_T=$SECONDS
printf 'trained    MLU=%-10s  %ss\n' "${T:-FAILED}" "$T_T"

# LP optimum on the same test slice, for the performance ratio.
SECONDS=0
$PY lp_oracle.py --topo Starlink2272.json --tm-model starlink \
    --prune-demands --slice-test-start 90 --slice-test-stop 101 \
    > /dev/null 2>&1
O=$($PY -c "
import pandas as pd
print('%.4f' % pd.read_csv('lp-oracle-Starlink2272.json.csv').opt_mlu.mean())
" 2>/dev/null)
printf 'LP oracle  MLU=%-10s  %ss\n' "${O:-FAILED}" "$SECONDS"

echo "peak GPU MiB (whole card): $(sort -n probe_gpu_$CAP.log | tail -1)"
echo "done $(date +%H:%M:%S)"
