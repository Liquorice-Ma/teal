#!/bin/bash
# Ablation with the repair step disabled (--admm-steps 0).
#
# With repair enabled, ours and no-embed agree to within 0.2% on 5 of 9
# runs (identical on one), while the gate ablation still costs 6.7-9.0%.
# The reading: rebalance_action reshapes the allocation strongly enough to
# absorb input-level differences (the mask embedding only changes feature
# values), whereas the gate changes how messages propagate in the GNN.
# Turning repair off removes that absorption --- on Starlink it moves MLU
# from ~1.21 to ~1.98, i.e. a much wider dynamic range in which module
# contributions can show up.
#
# Detach-safe:
#   setsid nohup ./eval_norepair.sh > norepair.log 2>&1 &

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--obs-ratio 0.3 --hist-len 3 --samples 30 --admm-steps 0 \
--epochs 60 --early-stop True --lr 0.0001 \
--num-restart 3 --warmup-epochs 4 $S"
CSV=norepair.csv
DONE=.norepair-done
[ -f "$CSV" ] || echo "config,seed,mlu" > "$CSV"
touch "$DONE"

run() {   # run <config> <seed> <extra args...>
    local cfg="$1" seed="$2"; shift 2
    local tag="$cfg-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    local out
    out=$($PY teal.py $BASE --seed "$seed" "$@" 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    echo "$cfg,$seed,$out" >> "$CSV"
    echo "$tag" >> "$DONE"
    printf '[done] %-20s %s\n' "$tag" "$out"
}

for seed in 0 1 2; do
    run ours        "$seed" --mask-mode embed
    run no-embed    "$seed" --mask-mode zero
    run no-gate     "$seed" --mask-mode embed --no-gate
    run no-temporal "$seed" --mask-mode embed --hist-len 1
    run zero-fill   "$seed" --mask-mode zero --no-gate --hist-len 1
done
echo "ALL DONE -> $CSV"
