#!/bin/bash
# Sparse-robustness check: does the mask-aware design beat the naive
# baselines at the same observability? This is the paper's core claim
# (not "RL beats heuristics", but "we degrade gracefully when the
# traffic matrix is only partially measured").
#
# All runs share capacity 2000, rho=0.3 unless stated, seed 0, and the
# conservation-preserving MLU repair.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
SLICES="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--obj min_max_link_util --topo Starlink2272.json --tm-model starlink \
--prune-demands --hist-len 3 --seed 0 --samples 30 --admm-steps 2 \
--epochs 60 --early-stop True --lr 0.0001 $SLICES"

run() {   # run <label> <extra args...>
    local label="$1"; shift
    local out
    out=$($PY teal.py $BASE "$@" 2>&1 | tr '\r' '\n' \
        | grep 'Testing: 100' | tail -1 | grep -oE 'obj=[0-9.]+' \
        | cut -d= -f2)
    printf '%-40s %s\n' "$label" "${out:-FAILED}"
}

echo "===== upper reference ====="
run "full observation (rho=1.0)" --obs-ratio 1.0 --mask-mode embed

for rho in 0.5 0.3 0.1; do
    echo "===== rho $rho ====="
    run "ours (mask embedding + gate)" --obs-ratio $rho --mask-mode embed
    run "zero-fill (no mask modules)" --obs-ratio $rho --mask-mode zero \
        --no-gate
    run "mean-interp (two-stage)" --obs-ratio $rho --mask-mode mean --no-gate
done
