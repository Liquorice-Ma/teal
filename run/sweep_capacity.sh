#!/bin/bash
# Capacity sweep: find the load level where learning actually helps.
#
# At capacity 2000 an untrained policy already reaches 0.965 satisfied
# demand (uniform splitting over 4 edge-disjoint paths is near-optimal
# once demands are pruned), leaving nothing for the policy to learn.
# This sweep lowers the capacity to create real contention and reports
# untrained vs trained for each level.
#
# Usage: bash sweep_capacity.sh [capacities...]   (default: 500 300 200)

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
SLICES="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--topo Starlink2272.json --tm-model starlink --prune-demands \
--obs-ratio 0.3 --mask-mode embed --hist-len 3 --seed 0 --samples 30 \
--admm-steps 2 $SLICES"

run() {   # run <label> <extra args...>
    local label="$1"; shift
    local out
    out=$($PY teal.py $BASE "$@" 2>&1 | tr '\r' '\n' \
        | grep 'Testing: 100' | tail -1 | grep -oE 'obj=[0-9.]+' \
        | cut -d= -f2)
    printf '%-34s %s\n' "$label" "${out:-FAILED}"
}

CAPS=${@:-500 300 200}
for cap in $CAPS; do
    echo "===== capacity $cap ====="
    $PY prepare_starlink.py --size-x 22 --size-y 72 --capacity "$cap" \
        > /dev/null 2>&1
    $PY lp_oracle.py --topo Starlink2272.json --tm-model starlink \
        --prune-demands --slice-test-start 90 --slice-test-stop 101 \
        > /dev/null 2>&1
    printf '%-34s %s\n' "oracle MLU (LP)" \
        "$($PY -c "import pandas as pd; \
print('%.4f' % pd.read_csv('lp-oracle-Starlink2272.json.csv').opt_mlu.mean())")"

    run "MLU untrained" --obj min_max_link_util --epochs 0
    run "MLU trained (60ep, lr1e-4)" --obj min_max_link_util --epochs 60 \
        --early-stop True --lr 0.0001
    run "flow untrained" --obj total_flow --epochs 0
    run "flow trained (60ep, lr1e-4)" --obj total_flow --epochs 60 \
        --early-stop True --lr 0.0001
    echo
done

# restore the default calibration
$PY prepare_starlink.py --size-x 22 --size-y 72 --capacity 2000 > /dev/null 2>&1
echo "restored capacity 2000"
