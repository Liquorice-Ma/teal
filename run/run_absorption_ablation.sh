#!/bin/bash
# Absorption ablation: which architectural component drives repair absorption?
#
# B1 (run_test_baseline.sh) showed the TEST-style arch (zero + no-gate +
# temporal) does NOT absorb --- trained wins 2/3 under repair at rho=0.3.
# Our model (embed + gate + temporal) DOES absorb (RQ1, 9/25 wins).  This
# script fills the 2x2 by running the two missing architectural cells ---
# no-embed (zero + gate) and no-gate (embed + no-gate) --- plus ours for a
# consistent admm-steps=2 baseline, all under oracle repair at rho=0.3.
#
#   ours       = embed + gate     (expect: ABSORBS, cf. RQ1 / saturation)
#   no-embed   = zero  + gate     (does the gate alone drive absorption?)
#   no-gate    = embed + no-gate  (does the embed alone drive absorption?)
#   test-style = zero  + no-gate  (have: does NOT absorb, B1)
#
# test-style is reused from test_baseline.csv (same admm-steps=2, rho=0.3).
#
# Detach-safe:
#   setsid nohup ./run_absorption_ablation.sh 6 > absorption_ablation.log 2>&1 &

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 --repair-input oracle $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"

CSV=absorption_ablation.csv
DONE=.absorption-ablation-done
LOCK=.absorption-ablation-lock
WORKERS=${1:-6}

[ -f "$CSV" ] || echo "config,train,rho,seed,mlu" > "$CSV"
touch "$DONE" "$LOCK"

run_one() {
    local spec="$1"
    local cfg train rho seed arch train_flag name
    IFS='|' read -r cfg train rho seed <<< "$spec"
    local tag="$cfg-$train-$rho-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }

    case "$cfg" in
        ours)     arch="--mask-mode embed" ;;
        no-embed) arch="--mask-mode zero" ;;
        no-gate)  arch="--mask-mode embed --no-gate" ;;
    esac
    if [ "$train" = "trained" ]; then train_flag="$TRAIN"; else train_flag="$UNTRAIN"; fi

    local out
    out=$($PY teal.py $COMMON $arch --obs-ratio "$rho" --seed "$seed" \
        $train_flag 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }

    flock "$LOCK" -c \
        "echo '$cfg,$train,$rho,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-30s %s\n' "$tag" "$out"
}
export -f run_one
export PY COMMON TRAIN UNTRAIN CSV DONE LOCK

TASKS=()
for cfg in ours no-embed no-gate; do
    for train in trained untrained; do
        for seed in 0 1 2; do
            TASKS+=("$cfg|$train|0.3|$seed")
        done
    done
done

echo "total tasks: ${#TASKS[@]}  workers: $WORKERS"
printf '%s\n' "${TASKS[@]}" \
    | xargs -P "$WORKERS" -I{} bash -c 'run_one "{}"'
echo "ALL DONE -> $CSV"
