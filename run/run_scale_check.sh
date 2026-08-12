#!/bin/bash
# Batch E: cross-scale mechanism check on Starlink2224 (528 nodes,
# one-third-scale instance of the Shell-1 architecture). Same task
# taxonomy as the main matrix, restricted to the two decisive rho
# points (learning window peak and gradient collapse).
#
#   repair group   (admm 2): rho in {0.3, 0.05}
#   no-repair group(admm 0): rho in {0.3, 0.05}
#   configs: ours (trained, embed) vs untrained; seeds 0-2
#
# NOTE: Starlink2224 has 93 snapshots (vs 101 for Starlink2272), so
# the slice stops differ. Absolute MLU is not comparable across
# topologies; only trained-vs-untrained deltas within 2224 are.
#
# Concurrency: default 8 worked for Starlink2272, but 2224 training
# tasks hold ~9 GB each (denser traffic -> ~2x more demand pairs), so
# trained batches on 2224 must run with --workers 2.
# Usage:
#   DRY=1 bash run_scale_check.sh        # dry run: list tasks only
#   bash run_scale_check.sh              # execute (8 workers)
#   bash run_scale_check.sh --workers 2  # execute (2 workers)
set -u
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NWORKER=8
if [ "${1:-}" = "--workers" ]; then NWORKER="$2"; fi
LOG=scale_check.log
LOGDIR=task-logs

S="--slice-train-start 0 --slice-train-stop 74 --slice-val-start 74 \
--slice-val-stop 84 --slice-test-start 84 --slice-test-stop 93"
CORE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2224.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
NOTRAIN="--epochs 0 --num-restart 1"

CSV=scale_check.csv
DONE=.scale-done
LOCK=.scale-lock
[ -f "$CSV" ] || echo "config,rho,seed,repair,mlu" > "$CSV"
touch "$DONE" "$LOCK"
mkdir -p "$LOGDIR"

EMBED="--mask-mode embed"

run_task() {   # run_task "cfg|rho|seed|repair|obj|extra"
    local line="$1"
    local cfg rho seed repair obj extra
    IFS='|' read -r cfg rho seed repair obj extra <<< "$line"

    local mode
    if [ "$cfg" = untrained ]; then mode="$NOTRAIN"; else mode="$TRAIN"; fi
    if [ "$repair" = norepair ]; then
        mode="$mode --admm-steps 0"
    else
        mode="$mode --admm-steps 2"
    fi
    mode="$mode $extra"

    local tag="2224-$cfg-$rho-$seed-$repair"
    local LOGF="$LOGDIR/$tag.log"

    if grep -qx "$tag" "$DONE"; then
        printf '[skip] %-32s already done\n' "$tag"
        return
    fi
    printf '[run ] %-32s started\n' "$tag"

    $PY teal.py $CORE --obs-ratio "$rho" --seed "$seed" \
        $mode > "$LOGF" 2>&1

    local out
    out=$(tr '\r' '\n' < "$LOGF" | grep 'Testing: 100' \
        | grep -oE 'obj=[0-9.]+' | tail -1 | cut -d= -f2)
    if [ -z "$out" ]; then
        if tr '\r' '\n' < "$LOGF" | grep -q 'obj=inf'; then
            out=inf
        elif grep -q 'OutOfMemoryError' "$LOGF"; then
            out='[OOM]'
        else
            out='[FAIL]'
        fi
    fi

    # serial CSV append
    (
        flock -x 200
        echo "$cfg,$rho,$seed,$repair,$out" >> "$CSV"
        echo "$tag" >> "$DONE"
    ) 200>"$LOCK"
    printf '[done] %-32s mlu=%s\n' "$tag" "$out"
}
export -f run_task
export PY CORE TRAIN NOTRAIN CSV DONE LOCK LOGDIR S

TASKS=()
# repair cells first: they carry the absorption verdict, so they
# finish earliest under serial execution
for seed in 0 1 2; do
    for rho in 0.3 0.05; do
        TASKS+=("ours|$rho|$seed|repair|mlu|$EMBED")
    done
done
for seed in 0 1 2; do
    for rho in 0.3 0.05; do
        TASKS+=("ours|$rho|$seed|norepair|mlu|$EMBED")
        TASKS+=("untrained|$rho|$seed|repair|mlu|$EMBED")
        TASKS+=("untrained|$rho|$seed|norepair|mlu|$EMBED")
    done
done

echo "batch E (scale check): ${#TASKS[@]} tasks, $NWORKER workers"
if [ "${DRY:-0}" = 1 ]; then
    printf '%s\n' "${TASKS[@]}"
    exit 0
fi

printf '%s\n' "${TASKS[@]}" | xargs -P "$NWORKER" -I{} bash -c 'run_task "$@"' _ {} \
    > "$LOG" 2>&1
echo "batch finished; summary:"
grep -c '^\[done\]' "$LOG" || true
tail -30 "$LOG"
