#!/bin/bash
# Final paper-grade experiment matrix, parallelized.
#
# The workload is CPU-bound (one run uses ~17 of 256 cores, GPU sits at
# ~8%), so NWORKER runs execute concurrently. Tasks are statically sharded
# across workers (no queue races); CSV appends are serialized with flock.
#
# Batches:
#   parallel : M1 main comparison gaps (mean-gated at rho 0.5/0.1)
#              M2 ablations WITHOUT repair (the 2x2 analysis showed repair
#                 masks the learned contribution, so both conditions are
#                 reported) + ours/zero-fill norepair at rho 0.5/0.1
#              M5 total_flow secondary objective
#   serial   : M3 zero-retraining (demand-split / Drop5 / Drop10 /
#                 failures-50) -- rebuilds graphs, kept isolated
#              M4 GDP trace (switches traffic files; LAST and optional:
#                 first-time path generation for ~126K pairs may be long)
#
# Protocol (frozen): k-shortest paths, deterministic, lr 1e-4, restart
# screening x3, 3 seeds, median reporting, PR normalization.
#
# Usage: NWORKER=6 ./run_final_matrix.sh
# Results -> final_matrix.csv (config,rho,seed,repair,obj,mlu)

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
NWORKER=${NWORKER:-6}
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 \
--num-restart 3 --warmup-epochs 4"
BASE="--shared-paths --deterministic --topo Starlink2272.json \
--tm-model starlink --prune-demands $S $TRAIN"
CSV=final_matrix.csv
DONE=.final-done
LOCK=.final-lock
[ -f "$CSV" ] || echo "config,rho,seed,repair,obj,mlu" > "$CSV"
touch "$DONE" "$LOCK"

# ---------- task list: "cfg|rho|seed|repair|obj|extra-args" ----------
TASKS=()
add() { TASKS+=("$1"); }

for seed in 0 1 2; do
    # M1: fill the remaining main-comparison cells
    for rho in 0.5 0.1; do
        add "mean-gated|$rho|$seed|1|mlu|--mask-mode mean"
    done
    # M2: ablations without repair (true module contributions)
    for rho in 0.5 0.3 0.1; do
        add "no-embed|$rho|$seed|0|mlu|--mask-mode zero --hist-len 3"
        add "no-gate|$rho|$seed|0|mlu|--mask-mode embed --no-gate --hist-len 3"
        add "no-temporal|$rho|$seed|0|mlu|--mask-mode embed --hist-len 1"
    done
    for rho in 0.5 0.1; do
        add "ours|$rho|$seed|0|mlu|--mask-mode embed --hist-len 3"
        add "zero-fill|$rho|$seed|0|mlu|--mask-mode zero --no-gate --hist-len 1"
    done
    add "full|1.0|$seed|0|mlu|--mask-mode embed --hist-len 3"
    # M5: secondary objective (total_flow keeps its native ADMM repair)
    add "full|1.0|$seed|1|flow|--mask-mode embed --hist-len 3"
    for rho in 0.5 0.3 0.1; do
        add "ours|$rho|$seed|1|flow|--mask-mode embed --hist-len 3"
        add "zero-fill|$rho|$seed|1|flow|--mask-mode zero --no-gate --hist-len 1"
    done
done

run_task() {   # run_task "cfg|rho|seed|repair|obj|extra"
    local line="$1"
    IFS='|' read -r cfg rho seed repair obj extra <<< "$line"
    local tag="$cfg-$rho-$seed-r$repair-$obj"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    local objarg admm
    if [ "$obj" = flow ]; then objarg="total_flow"; else objarg="min_max_link_util"; fi
    if [ "$repair" = 1 ]; then admm=2; else admm=0; fi
    local out
    out=$($PY teal.py $BASE --obj $objarg --admm-steps $admm \
        --obs-ratio "$rho" --seed "$seed" --samples 30 $extra 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    flock "$LOCK" bash -c \
        "echo '$cfg,$rho,$seed,$repair,$obj,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-34s %s\n' "$tag" "$out"
}
export -f run_task 2>/dev/null

worker() {   # worker <id>
    local id="$1" i
    for i in "${!TASKS[@]}"; do
        [ $((i % NWORKER)) -eq "$id" ] && run_task "${TASKS[$i]}"
    done
}

echo "===== parallel phase: ${#TASKS[@]} tasks on $NWORKER workers ====="
for w in $(seq 0 $((NWORKER-1))); do worker "$w" & done
wait
echo "===== parallel phase done ====="

# ---------- serial phase: graph-rebuilding / data-switching runs ----------
for seed in 0 1 2; do
    run_task "demand-split|0.3|$seed|1|mlu|--mask-mode embed --hist-len 3 --demand-split"
    run_task "drop5|0.3|$seed|1|mlu|--mask-mode embed --hist-len 3 --test-topo Starlink2272Drop5.json"
    run_task "drop10|0.3|$seed|1|mlu|--mask-mode embed --hist-len 3 --test-topo Starlink2272Drop10.json"
    run_task "failures50|0.3|$seed|1|mlu|--mask-mode embed --hist-len 3 --failures 50"
done
echo "===== serial phase done (GDP intentionally deferred) ====="
echo "ALL DONE -> $CSV"
