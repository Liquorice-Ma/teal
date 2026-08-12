#!/bin/bash
# Does a per-demand, neighbor-derived fill beat the shared placeholder?
#
# The shared learnable placeholder measured 0.05% against zero-filling
# because every unobserved demand receives the same value, and the policy
# softmaxes over each demand's paths, so a common offset cancels. The 'nbr'
# fill instead estimates each unobserved demand from the observed demands
# sharing its source, which differs across demands and therefore carries
# usable signal.
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--obs-ratio 0.3 --hist-len 3 --samples 30 --admm-steps 2 \
--epochs 60 --early-stop True --lr 0.0001 \
--num-restart 3 --warmup-epochs 4 $S"
CSV=nbr.csv
DONE=.nbr-done
[ -f "$CSV" ] || echo "config,seed,mlu" > "$CSV"
touch "$DONE"

run() {
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
    printf '[done] %-16s %s\n' "$tag" "$out"
}

for seed in 0 1 2; do
    run nbr   "$seed" --mask-mode nbr
    run embed "$seed" --mask-mode embed
    run zero  "$seed" --mask-mode zero
done
echo "ALL DONE -> $CSV"
