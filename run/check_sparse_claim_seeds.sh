#!/bin/bash
# Sparse-robustness check with 3 seeds (paper-grade, mean/std computed
# afterwards by summarize_claim.py).
#
# Single-seed runs are not conclusive here: the same configuration was
# observed at 1.2048 / 1.2317 / 1.2718 (5.5% spread) while the gap
# between methods is 2-5%, i.e. the same order as the noise. Every
# configuration is therefore repeated over seeds 0/1/2.
#
# Detach-safe: intended to be launched via
#   setsid nohup ./check_sparse_claim_seeds.sh > claim_seeds.log 2>&1 &
# so it survives SSH disconnects.
#
# Results are appended to claim_seeds.csv as they finish, so partial
# results stay usable if the run is interrupted.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
SLICES="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--obj min_max_link_util --topo Starlink2272.json --tm-model starlink \
--prune-demands --hist-len 3 --samples 30 --admm-steps 2 \
--epochs 60 --early-stop True --lr 0.0001 $SLICES"
CSV=claim_seeds.csv
DONE=.claim-done

[ -f "$CSV" ] || echo "method,rho,seed,mlu" > "$CSV"
touch "$DONE"

run() {   # run <method> <rho> <seed> <extra args...>
    local method="$1" rho="$2" seed="$3"; shift 3
    local tag="$method-$rho-$seed"
    if grep -qxF "$tag" "$DONE"; then
        echo "[skip] $tag"; return
    fi
    local out
    out=$($PY teal.py $BASE --obs-ratio "$rho" --seed "$seed" "$@" 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    if [ -z "$out" ]; then
        echo "[FAIL] $tag"; return
    fi
    echo "$method,$rho,$seed,$out" >> "$CSV"
    echo "$tag" >> "$DONE"
    printf '[done] %-22s %s\n' "$tag" "$out"
}

for seed in 0 1 2; do
    run oracle 1.0 "$seed" --mask-mode embed
    for rho in 0.5 0.3 0.1; do
        run ours      "$rho" "$seed" --mask-mode embed
        run zero-fill "$rho" "$seed" --mask-mode zero --no-gate
        run mean-interp "$rho" "$seed" --mask-mode mean --no-gate
    done
done
echo "ALL DONE -> $CSV"
