#!/bin/bash
# Deployable-repair arm: does training matter once repair is denied the
# ground-truth traffic matrix?
#
# Motivation. Under the MLU objective the repair step is rebalance_action,
# and it computes link utilization from self.obs --- the *true* traffic
# matrix (teal_env.py). That is legitimate in the fully-observed WAN setting
# Teal was built for, but under sparse observation it hands repair an
# information advantage the policy does not have, which is what makes it
# look as though learning contributes nothing.
#
# --repair-input restricts repair to what a controller can actually see:
#   oracle : true TM (original behaviour, kept as the reference arm)
#   nbr    : observation + neighbor estimate for unobserved demands
#   zero   : observation with unobserved demands zero-filled (TEST-like)
#
# Smoke reading at untrained/rho=0.3/seed0, 2 steps:
#   no repair 2.833 | zero 2.193 | nbr 1.644 | oracle 1.477
# so the deployable arms leave real headroom for the learned imputation.
#
# nbr sweeps four rho values first (the headline arm); zero follows at two.
#
# Detach-safe:
#   setsid nohup ./run_deployable_repair.sh 4 > deployable_repair.log 2>&1 &

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 --mask-mode embed $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"

CSV=deployable_repair.csv
DONE=.deployable-repair-done
LOCK=.deployable-repair-lock
WORKERS=${1:-4}

[ -f "$CSV" ] || echo "config,repair_input,rho,seed,mlu" > "$CSV"
touch "$DONE" "$LOCK"

run_one() {
    local spec="$1"
    local cfg ri rho seed
    IFS='|' read -r cfg ri rho seed <<< "$spec"
    local tag="$cfg-$ri-$rho-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }

    local mode
    if [ "$cfg" = "ours" ]; then mode="$TRAIN"; else mode="$UNTRAIN"; fi

    local out
    out=$($PY teal.py $COMMON --repair-input "$ri" --obs-ratio "$rho" \
        --seed "$seed" $mode 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }

    flock "$LOCK" -c \
        "echo '$cfg,$ri,$rho,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-34s %s\n' "$tag" "$out"
}
export -f run_one
export PY COMMON TRAIN UNTRAIN CSV DONE LOCK

TASKS=()
# headline arm: neighbor-filled repair across the rho sweep
for rho in 0.3 0.05 0.1 0.5; do
    for seed in 0 1 2; do
        TASKS+=("ours|nbr|$rho|$seed")
        TASKS+=("untrained|nbr|$rho|$seed")
    done
done
# secondary arm: zero-filled repair at two operating points
for rho in 0.3 0.05; do
    for seed in 0 1 2; do
        TASKS+=("ours|zero|$rho|$seed")
        TASKS+=("untrained|zero|$rho|$seed")
    done
done

echo "total tasks: ${#TASKS[@]}  workers: $WORKERS"
printf '%s\n' "${TASKS[@]}" \
    | xargs -P "$WORKERS" -I{} bash -c 'run_one "{}"'
echo "ALL DONE -> $CSV"
