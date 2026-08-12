#!/bin/bash
# Serial phase (M3 zero-retraining) of the final matrix, parallelized.
# Replaces the serial loop in run_final_matrix.sh.
#
# Race analysis: path caches (topologies/paths/path-form/*.pkl) are
# generated on first use. Two tasks racing on the SAME missing cache can
# corrupt it. The four configs touch disjoint cache files, so:
#   phase 1: 4 configs x seed 0 in parallel  -> warms every cache
#   phase 2: 4 configs x seed 1,2 in parallel -> all reads, no writes
# Results go to the same final_matrix.csv via the same flock protocol,
# and .final-done tags match run_final_matrix.sh exactly.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 \
--num-restart 3 --warmup-epochs 4"
BASE="--shared-paths --deterministic --topo Starlink2272.json \
--tm-model starlink --prune-demands $S $TRAIN"
CSV=final_matrix.csv
DONE=.final-done
LOCK=.final-lock

run_task() {   # run_task "cfg|rho|seed|repair|obj|extra"
    local line="$1"
    IFS='|' read -r cfg rho seed repair obj extra <<< "$line"
    local tag="$cfg-$rho-$seed-r$repair-$obj"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    local out
    out=$($PY teal.py $BASE --obj min_max_link_util --admm-steps 2 \
        --obs-ratio "$rho" --seed "$seed" --samples 30 $extra 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    flock "$LOCK" bash -c \
        "echo '$cfg,$rho,$seed,$repair,$obj,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-34s %s\n' "$tag" "$out"
}

declare -A EXTRA=(
    [demand-split]="--demand-split"
    [drop5]="--test-topo Starlink2272Drop5.json"
    [drop10]="--test-topo Starlink2272Drop10.json"
    [failures50]="--failures 50"
)

task_of() { echo "$1|0.3|$2|1|mlu|--mask-mode embed --hist-len 3 ${EXTRA[$1]}"; }

echo "===== serial phase (parallelized): warmup 4 configs x seed 0 ====="
for cfg in demand-split drop5 drop10 failures50; do
    run_task "$(task_of "$cfg" 0)" &
done
wait

echo "===== serial phase: seeds 1,2 on warm caches ====="
for cfg in demand-split drop5 drop10 failures50; do
    run_task "$(task_of "$cfg" 1)" &
    run_task "$(task_of "$cfg" 2)" &
done
wait
echo "===== serial phase done (GDP intentionally deferred) ====="
echo "ALL DONE -> $CSV"
