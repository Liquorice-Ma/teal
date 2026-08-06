#!/bin/bash
# Runs in parallel with eval_matrix.sh (separate CSV/DONE files).
# The workload is CPU-bound, not GPU-bound: a single run occupies ~17 of
# 256 cores while GPU utilization sits at 8%, so concurrent runs are free.
#
# Question 1 (untrained control): ours reaches PR 1.035 at rho=0.02, better
# than PR 1.198 at rho=0.3. Fewer observations cannot genuinely help, so the
# suspicion is degeneration --- with almost everything filled, the input
# approaches a constant, the policy approaches uniform splitting, and
# uniform splitting is already near-optimal here. If an untrained policy
# scores the same at rho=0.02, the low PR reflects degeneration rather than
# learning, and the paper must not advertise "2% observation is enough".
# rho=0.3 is included as a contrast: learning should matter there.
# Note --num-restart 1: restart screening would train warmup epochs and
# stop the arm from being genuinely untrained.
#
# Question 2 (sign flip): the neighbor estimate hurt at rho=0.5 (+7.2%) but
# helped at rho=0.1 (-7.0%). Measuring rho=0.05/0.02 shows whether the flip
# is a real trend (estimation pays off only under extreme sparsity) or noise.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 $S"
CSV=verify.csv
DONE=.verify-done
[ -f "$CSV" ] || echo "config,rho,seed,mlu" > "$CSV"
touch "$DONE"

run() {   # run <config> <rho> <seed> <extra args...>
    local cfg="$1" rho="$2" seed="$3"; shift 3
    local tag="$cfg-$rho-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    local out
    out=$($PY teal.py $COMMON --obs-ratio "$rho" --seed "$seed" "$@" 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    echo "$cfg,$rho,$seed,$out" >> "$CSV"
    echo "$tag" >> "$DONE"
    printf '[done] %-24s %s\n' "$tag" "$out"
}

echo "===== Q1: untrained control (degeneration check) ====="
for seed in 0 1 2; do
    for rho in 0.02 0.05 0.3; do
        run untrained "$rho" "$seed" --mask-mode embed \
            --epochs 0 --num-restart 1
    done
done

echo "===== Q2: neighbor estimate under extreme sparsity ====="
for seed in 0 1 2; do
    for rho in 0.05 0.02; do
        run nbr "$rho" "$seed" --mask-mode nbr \
            --epochs 60 --early-stop True --lr 0.0001 \
            --num-restart 3 --warmup-epochs 4
    done
done
echo "ALL DONE -> $CSV"
