#!/bin/bash
# Gap-fill batch: completes the imputation matrix, extends the three main
# arms to 5 seeds, and fills the untrained control.
#
# Why these tasks:
#   1. nbr at rho=0.3 and mean-interp / mean-gated at rho=0.05,0.02 are
#      the empty cells of the strategy x observability matrix. nbr already
#      beats ours at rho<=0.1, so the extreme-sparsity comparison must
#      include every imputation strategy, not just zero-fill.
#   2. ours / zero-fill / nbr get seeds 3,4. Seed spread was found to be
#      as large as the between-config differences, so 3 seeds cannot
#      separate the arms; a paired sign test over 5 seeds can.
#   3. untrained at rho=0.5,0.1 completes the control curve that shows
#      ADMM repair, not learning, drives the final MLU.
#   4. untrained on the Drop topologies tells whether drop10 scoring
#      BETTER than the unperturbed reference means good generalization or
#      simply an easier problem instance.
#
# Concurrency: 16. One run needs ~1.07 GB of the 24.5 GB GPU and ~4 of
# 256 cores, so 16 concurrent runs use ~17 GB / ~64 cores. Scheduling is
# dynamic via xargs -P (not static sharding), so no worker idles at the
# tail. Model files are never written (model_save defaults off), so
# concurrent runs of the same config cannot clobber each other.
#
# Appends to the existing CSVs with the existing tag format, so it is
# resumable and re-runnable: completed tags are skipped.
#
# Usage: NWORKER=16 ./run_gap_fill.sh

# absolute path to self, resolved BEFORE cd, since xargs re-invokes it
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
NWORKER=${NWORKER:-16}
LOG=gap_fill.log

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
CORE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
# genuinely untrained: epochs 0 AND num-restart 1, since restart
# screening would train warmup epochs
NOTRAIN="--epochs 0 --num-restart 1"

# ---------------- single task executor (invoked via xargs) ----------------
if [ "$1" = "--task" ]; then
    IFS='|' read -r cfg rho seed target extra <<< "$2"
    case "$target" in
        matrix) CSV=eval_denoised.csv; DONE=.matrix-done; LOCK=.matrix-lock ;;
        verify) CSV=verify.csv;        DONE=.verify-done; LOCK=.verify-lock ;;
    esac
    tag="$cfg-$rho-$seed"
    grep -qxF "$tag" "$DONE" 2>/dev/null && { echo "[skip] $tag"; exit 0; }
    case "$cfg" in
        untrained*) MODE="$NOTRAIN" ;;
        *)          MODE="$TRAIN" ;;
    esac
    out=$($PY teal.py $CORE $MODE --obs-ratio "$rho" --seed "$seed" \
        $extra 2>&1 | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    if [ -z "$out" ]; then
        printf '[FAIL] %-28s (no obj parsed; inf or crash)\n' "$tag"
        exit 0
    fi
    flock "$LOCK" bash -c \
        "echo '$cfg,$rho,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-28s %s\n' "$tag" "$out"
    exit 0
fi

# ---------------- task list ----------------
[ -f eval_denoised.csv ] || echo "config,rho,seed,mlu" > eval_denoised.csv
[ -f verify.csv ] || echo "config,rho,seed,mlu" > verify.csv
touch .matrix-done .verify-done .matrix-lock .verify-lock

OURS="--mask-mode embed"
ZEROFILL="--mask-mode zero --no-gate --hist-len 1"
NBR="--mask-mode nbr"
MEANGATED="--mask-mode mean"
MEANINTERP="--mask-mode mean --no-gate --hist-len 1"

TASKS=()

# 1. empty cells of the strategy x observability matrix (seeds 0-2)
for seed in 0 1 2; do
    TASKS+=("nbr|0.3|$seed|matrix|$NBR")
    for rho in 0.05 0.02; do
        TASKS+=("mean-gated|$rho|$seed|matrix|$MEANGATED")
        TASKS+=("mean-interp|$rho|$seed|matrix|$MEANINTERP")
    done
done

# 2. extend the three main arms to seeds 3,4 for the paired sign test
for seed in 3 4; do
    for rho in 0.5 0.3 0.1 0.05 0.02; do
        TASKS+=("ours|$rho|$seed|matrix|$OURS")
        TASKS+=("zero-fill|$rho|$seed|matrix|$ZEROFILL")
        TASKS+=("nbr|$rho|$seed|matrix|$NBR")
    done
done

# 3. complete the untrained control curve (cheap: no training)
for seed in 0 1 2; do
    for rho in 0.5 0.1; do
        TASKS+=("untrained|$rho|$seed|verify|--mask-mode embed")
    done
done

# 4. untrained on the Drop topologies: is drop10 easier, or is transfer
#    genuinely good? Same rho as the M3 runs.
for seed in 0 1 2; do
    TASKS+=("untrained-drop5|0.3|$seed|verify|\
--mask-mode embed --test-topo Starlink2272Drop5.json")
    TASKS+=("untrained-drop10|0.3|$seed|verify|\
--mask-mode embed --test-topo Starlink2272Drop10.json")
done

echo "===== gap fill: ${#TASKS[@]} tasks on $NWORKER workers =====" | tee -a "$LOG"
date | tee -a "$LOG"

# DRY=1 prints the task list and exits, for verifying tags before a long run
if [ -n "$DRY" ]; then
    printf '%s\n' "${TASKS[@]}"
    echo "TOTAL ${#TASKS[@]}"
    exit 0
fi

# ---------------- warm the path caches serially ----------------
# A missing path pkl is generated on first use; two runs racing on the
# same missing file can corrupt it. Only the Drop topologies are at risk.
# Match edge-disjoint-False specifically: that is what --shared-paths
# uses, and a stale edge-disjoint-True file must not count as a hit.
warm() {   # warm <cfg> <topo-json>
    if ! ls ../topologies/paths/path-form/ 2>/dev/null \
            | grep -q "^$2-4-paths_edge-disjoint-False"; then
        echo "[warmup] building path cache for $2" | tee -a "$LOG"
        bash "$SELF" --task "$1|0.3|0|verify|\
--mask-mode embed --test-topo $2" 2>&1 | tee -a "$LOG"
    fi
}
warm untrained-drop5 Starlink2272Drop5.json
warm untrained-drop10 Starlink2272Drop10.json

# ---------------- dynamic parallel execution ----------------
printf '%s\n' "${TASKS[@]}" \
    | xargs -d '\n' -P "$NWORKER" -I {} bash "$SELF" --task {} 2>&1 \
    | tee -a "$LOG"

echo "===== gap fill done =====" | tee -a "$LOG"
date | tee -a "$LOG"
echo "eval_denoised.csv: $(wc -l < eval_denoised.csv) lines" | tee -a "$LOG"
echo "verify.csv: $(wc -l < verify.csv) lines" | tee -a "$LOG"
