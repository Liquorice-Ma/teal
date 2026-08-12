#!/bin/bash
# Attribution: is the gate's +7-8% a learned policy or a structural prior?
#
# The gate ablation (no-embed vs zero-fill, both zero-filled, both trained)
# showed +6.9% at rho=0.5 and +8.1% at rho=0.3. But an untrained policy just
# measured *better* than a trained one at rho=0.3 (1.5987 vs 1.6490), so the
# gain may not come from learning at all: cutting messages from unobserved
# paths is itself a conservative inductive bias that spreads load more
# evenly, and uniform-ish splitting is already near-optimal on this problem.
#
# This runs the same structural contrast with training removed entirely.
#   untrained-struct   : zero fill + gate + temporal, no training
#   untrained-nostruct : zero fill, no gate, no temporal, no training
# If the gap persists without training, the mechanism is structural and the
# paper must describe it as an architectural prior rather than a learned
# behavior. If the gap vanishes, training is what activates the gate.
#
# --num-restart 1 keeps the untrained arms genuinely untrained (restart
# screening would run warmup epochs).

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 --admm-steps 2 --epochs 0 --num-restart 1 \
--mask-mode zero $S"
CSV=attrib.csv
DONE=.attrib-done
[ -f "$CSV" ] || echo "config,rho,seed,mlu" > "$CSV"
touch "$DONE"

run() {
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
    printf '[done] %-26s %s\n' "$tag" "$out"
}

for seed in 0 1 2; do
    for rho in 0.3 0.5; do
        run untrained-struct   "$rho" "$seed" --hist-len 3
        run untrained-nostruct "$rho" "$seed" --no-gate --hist-len 1
    done
done
echo "ALL DONE -> $CSV"
