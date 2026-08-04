#!/bin/bash
# Denoised paper-grade evaluation.
#
# Lessons from the literature review (C):
#  - TEST reports only 2-3% gains over its baselines; FNC ~8%; ELATE's
#    ablations drop 4.9-5.7%. A few-percent gain is the norm here, so the
#    ~5% seed noise must be suppressed, not the method changed.
#  - TEST/FNC/ELATE all use (k-)shortest paths, never edge-disjoint, so
#    --shared-paths is the correct setting.
#
# Changes vs the earlier check: k-shortest paths, deterministic GPU ops,
# 5 seeds, and an ELATE-style ablation (remove one module at a time) that
# is more robust than "trained vs untrained".
#
# Detach-safe:
#   setsid nohup ./eval_denoised.sh > eval_denoised.log 2>&1 &

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 --epochs 60 --early-stop True --lr 0.0001 $S"
CSV=eval_denoised.csv
DONE=.eval-done
[ -f "$CSV" ] || echo "config,rho,seed,mlu" > "$CSV"
touch "$DONE"

run() {   # run <config> <rho> <seed> <extra args...>
    local cfg="$1" rho="$2" seed="$3"; shift 3
    local tag="$cfg-$rho-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    local out
    out=$($PY teal.py $BASE --obs-ratio "$rho" --seed "$seed" "$@" 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    echo "$cfg,$rho,$seed,$out" >> "$CSV"
    echo "$tag" >> "$DONE"
    printf '[done] %-26s %s\n' "$tag" "$out"
}

# LP oracle for the shared-path setting (PR denominator)
$PY lp_oracle.py --shared-paths --topo Starlink2272.json \
    --tm-model starlink --prune-demands \
    --slice-test-start 90 --slice-test-stop 101 > /dev/null 2>&1
cp lp-oracle-Starlink2272.json.csv lp-oracle-shared.csv 2>/dev/null

for seed in 0 1 2 3 4; do
    run full 1.0 "$seed" --mask-mode embed
    for rho in 0.5 0.3 0.1; do
        # full model and single-module ablations (ELATE-style)
        run ours       "$rho" "$seed" --mask-mode embed
        run no-embed   "$rho" "$seed" --mask-mode zero
        run no-gate    "$rho" "$seed" --mask-mode embed --no-gate
        run no-temporal "$rho" "$seed" --mask-mode embed --hist-len 1
        # external baselines
        run zero-fill  "$rho" "$seed" --mask-mode zero --no-gate --hist-len 1
        run mean-interp "$rho" "$seed" --mask-mode mean --no-gate --hist-len 1
    done
done
echo "ALL DONE -> $CSV"
