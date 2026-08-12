#!/bin/bash
# P0 + P1: complete the imputation-strategy matrix and probe the sparsity
# boundary.
#
# P0 fills the gaps in the central claim's evidence: the imputation
# strategies were only measured at rho=0.3 (nbr) or partially
# (mean-interp), while the claim "imputation does not help, structural
# awareness does" needs the full strategy x observability matrix.
#
# P1 answers the question a reviewer will certainly ask: how far can
# observation be reduced before the "sparse ~= full" result breaks? The
# measured range so far stops at rho=0.1, which never breaks, so the
# boundary lies below it.
#
# Appends to eval_denoised.csv so summarize_eval.py covers everything.
# Detach-safe; resumable via .matrix-done.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 --epochs 60 --early-stop True --lr 0.0001 \
--num-restart 3 --warmup-epochs 4 $S"
CSV=eval_denoised.csv
DONE=.matrix-done
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
    printf '[done] %-24s %s\n' "$tag" "$out"
}

echo "===== P0: complete the imputation matrix ====="
for seed in 0 1 2; do
    for rho in 0.5 0.1; do
        run nbr         "$rho" "$seed" --mask-mode nbr
        run mean-interp "$rho" "$seed" --mask-mode mean --no-gate --hist-len 1
    done
    # global-mean fill WITH the structural modules kept: the missing cell of
    # the fill x structure matrix, and the comparison a reviewer will ask for
    # ('what if your gate is combined with mean interpolation?')
    run mean-gated 0.3 "$seed" --mask-mode mean
done

echo "===== P1: sparsity boundary (rho 0.05 / 0.02) ====="
for seed in 0 1 2; do
    for rho in 0.05 0.02; do
        run ours      "$rho" "$seed" --mask-mode embed
        run zero-fill "$rho" "$seed" --mask-mode zero --no-gate --hist-len 1
    done
done
echo "ALL DONE -> $CSV"
