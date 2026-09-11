#!/bin/bash
# Trained-vs-untrained across enlarged link capacities, 5 seeds each.
#
# Why this matrix exists.  The MLU objective max_e load_e/c_e is scale
# invariant: every ISL carries the same capacity, so c -> k*c divides the
# optimum by exactly k and leaves the optimal routing untouched (confirmed
# here: the LP oracle at 4000 is 0.3495 = 0.6990/2).  Two places in the code
# break that invariance for the *policy*:
#   lib/teal_env.py:312  obs = concat([capacity, tm])  -- absolute capacity is
#                        a network input, so scaling shifts the input scale;
#   lib/teal_env.py:575  util = 1 + (edge_flow/capacity - 1).relu()  -- the
#                        feasibility projection only fires above capacity.
# The baseline (capacity 2000, MLU ~1.6-1.8) is entirely inside the regime
# where the projection fires on every step, i.e. it silently load-balances for
# whatever the policy emits.  The single-seed probe found the trained/untrained
# gap flipping sign once MLU drops below 1 (2000: 1.6935 vs 1.6441, trained
# worse; 4000: 0.8433 vs 0.8840, trained better).  n=1 cannot settle this --
# the 5-seed baseline spread at rho=0.3 is 1.625-1.830, wider than the effect.
# Hence: 2 capacities x 2 configs x 5 seeds, plus an LP oracle per capacity so
# the performance ratio MLU/oracle can be compared across load levels.
#
# Capacity lives in the shared topologies/Starlink2272.json, so capacities MUST
# run one after another; only the seeds inside a capacity go in parallel.
#
# Usage: bash run_bigcap_matrix.sh [workers]     (default 3)

set -u
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CAPS="4000 8000"
RHO=0.3
SEEDS="0 1 2 3 4"
WORKERS=${1:-3}
BASELINE_CAP=2000          # restored on exit, matches the paper's main table

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 --admm-steps 2 --repair-input nbr $S"
ARCH="--mask-mode embed --hist-len 3"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"

CSV=bigcap_matrix.csv
ORA=bigcap_oracle.csv
DONE=.bigcap-done
LOCK=.bigcap-lock

[ -f "$CSV" ] || echo "capacity,config,rho,seed,mlu" > "$CSV"
[ -f "$ORA" ] || echo "capacity,oracle_mlu" > "$ORA"
touch "$DONE" "$LOCK"

run_one() {
    local spec="$1" cap cfg seed tag mode out
    IFS='|' read -r cap cfg seed <<< "$spec"
    tag="$cap-$cfg-$RHO-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    if [ "$cfg" = untrained ]; then mode="$UNTRAIN"; else mode="$TRAIN"; fi
    local t0=$SECONDS
    out=$($PY teal.py $COMMON $ARCH --obs-ratio "$RHO" --seed "$seed" \
        $mode 2>&1 | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    flock "$LOCK" -c \
        "echo '$cap,$cfg,$RHO,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-26s MLU=%-10s %ss\n' "$tag" "$out" "$((SECONDS - t0))"
}
export -f run_one
export PY COMMON ARCH TRAIN UNTRAIN RHO CSV DONE LOCK

set_capacity() {   # set_capacity <cap>; verifies what actually landed on disk
    $PY prepare_starlink.py --size-x 22 --size-y 72 --capacity "$1" \
        > /dev/null 2>&1
    local got
    got=$($PY -c "
import json
d = json.load(open('../topologies/Starlink2272.json'))
c = {l['capacity'] for l in d['links']}
print(int(c.pop()) if len(c) == 1 else 'MIXED:%s' % c)
")
    [ "$got" = "$1" ] || { echo "CAPACITY MISMATCH: want $1 got $got"; exit 1; }
    echo "capacity on disk is now $got"
}

echo "=== bigcap matrix: caps [$CAPS] rho $RHO seeds [$SEEDS] workers $WORKERS"
echo "=== started $(date '+%m-%d %H:%M:%S')"

for cap in $CAPS; do
    echo "===================== capacity $cap ====================="
    set_capacity "$cap" || exit 1

    # LP oracle for this capacity: overwrites lp-oracle-*.csv, so it must not
    # overlap with another capacity's oracle run.
    if ! grep -q "^$cap," "$ORA"; then
        $PY lp_oracle.py --topo Starlink2272.json --tm-model starlink \
            --prune-demands --slice-test-start 90 --slice-test-stop 101 \
            > /dev/null 2>&1
        o=$($PY -c "
import csv
v = [float(r['opt_mlu']) for r in csv.DictReader(
        open('lp-oracle-Starlink2272.json.csv'))]
print('%.4f' % (sum(v) / len(v)))
" 2>/dev/null)
        if [ -n "$o" ]; then
            cp lp-oracle-Starlink2272.json.csv "lp-oracle-cap$cap.csv"
            echo "$cap,$o" >> "$ORA"
            echo "LP oracle @$cap = $o"
        else
            echo "LP oracle @$cap FAILED"
        fi
    else
        echo "LP oracle @$cap already recorded"
    fi

    TASKS=()
    for cfg in untrained ours; do          # untrained first: 9s, fails fast
        for seed in $SEEDS; do TASKS+=("$cap|$cfg|$seed"); done
    done
    printf '%s\n' "${TASKS[@]}" | xargs -P "$WORKERS" -I{} \
        bash -c 'run_one "{}"'
done

echo "===================== restoring baseline ====================="
set_capacity "$BASELINE_CAP"
echo "ALL DONE $(date '+%m-%d %H:%M:%S') -> $CSV / $ORA"
