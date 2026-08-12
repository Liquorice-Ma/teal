#!/bin/bash
# Supplement batch (B): close every hole in the three main arms
# (ours / zero-fill / nbr) so each (config, rho) cell has seeds 0-4.
#
# Seed spread is as large as the between-config differences, so the
# paired sign test needs all 5 seeds per cell. Auditing the merged
# eval_denoised.csv + verify.csv shows 20 missing (config, rho, seed)
# points: seed 3 died everywhere in the gap-fill label-loss incident
# and several seed-4 points with it. Filling them is what lets the
# headline comparisons (ours vs zero-fill, ours vs nbr) be judged on
# 20-30 paired observations instead of 9-15.
#
# WHY THIS SCRIPT EXISTS instead of reusing run_gap_fill.sh: that runner
# piped stdout straight into a tqdm-scraping grep chain. Under 16-way
# concurrency (load ~83) the progress line got truncated before 'obj=',
# so 35 tasks that had actually finished computing were recorded as
# [FAIL]. Fixes here:
#   1. output goes to a per-task file, never a pipe;
#   2. obj is taken from ALL 'Testing: 100' lines (last match), not from
#      the last line only, with a fallback to any 'obj=' occurrence;
#   3. obj=inf is recorded as inf rather than silently failing, so a
#      genuine numerical blow-up is distinguishable from a parse failure;
#   4. logs are kept, so any failure can be diagnosed after the fact.
#
# Usage: NWORKER=8 ./run_supplement.sh   (DRY=1 to list tasks)
#
# CONCURRENCY IS 8, NOT 16. Measured per-process GPU use in this batch
# ranges from 0.55 to 3.02 GB (mask-mode mean / nbr need scatter buffers
# and admm-steps 5 keeps more intermediates), far above the 1.07 GB seen
# in the earlier embed-only batch. 16 workers exhausted the 23.5 GB card
# during warmup and 10 tasks died with torch.OutOfMemoryError. 8 workers
# stay under the cap even if every task hits its 3 GB worst case.

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
NWORKER=${NWORKER:-8}
# reduce fragmentation, as advised by the OOM message itself
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=supplement.log
LOGDIR=task-logs

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
CORE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
NOTRAIN="--epochs 0 --num-restart 1"

# ---------------- single task executor (invoked via xargs) ----------------
if [ "$1" = "--task" ]; then
    IFS='|' read -r cfg rho seed target admm extra <<< "$2"
    case "$target" in
        matrix) CSV=eval_denoised.csv; DONE=.matrix-done; LOCK=.matrix-lock
                tag="$cfg-$rho-$seed"
                row="$cfg,$rho,$seed" ;;
        verify) CSV=verify.csv;        DONE=.verify-done;  LOCK=.verify-lock
                tag="$cfg-$rho-$seed"
                row="$cfg,$rho,$seed" ;;
        norepair) CSV=norepair_curve.csv; DONE=.norepair-done; LOCK=.norepair-lock
                tag="$cfg-norepair-$rho-$seed"
                row="$cfg,$rho,$seed" ;;
        sweep)  CSV=repair_sweep.csv;  DONE=.sweep-done;  LOCK=.sweep-lock
                tag="$cfg-admm$admm-$rho-$seed"
                row="$cfg,$admm,$rho,$seed" ;;
    esac
    grep -qxF "$tag" "$DONE" 2>/dev/null && { echo "[skip] $tag"; exit 0; }
    case "$cfg" in
        untrained*) MODE="$NOTRAIN" ;;
        *)          MODE="$TRAIN" ;;
    esac

    LOGF="$LOGDIR/$tag.log"
    $PY teal.py $CORE $MODE --admm-steps "$admm" --obs-ratio "$rho" \
        --seed "$seed" $extra > "$LOGF" 2>&1

    # robust extraction: every 'Testing: 100' line, take the last obj
    out=$(tr '\r' '\n' < "$LOGF" | grep 'Testing: 100' \
        | grep -oE 'obj=[0-9.]+' | tail -1 | cut -d= -f2)
    # fallback: any obj= anywhere (a truncated final line still leaves
    # earlier running averages, which differ only in the last digits)
    [ -z "$out" ] && out=$(tr '\r' '\n' < "$LOGF" \
        | grep -oE 'obj=[0-9.]+' | tail -1 | cut -d= -f2)
    # a real divergence must be recorded, not hidden as a parse failure
    if [ -z "$out" ] && grep -q 'obj=inf' "$LOGF"; then out=inf; fi

    if [ -z "$out" ]; then
        # distinguish resource exhaustion from every other failure, so a
        # rerun immediately shows whether the concurrency is still too high
        if grep -q 'OutOfMemoryError' "$LOGF"; then
            printf '[OOM]  %-30s lower NWORKER\n' "$tag"
        else
            printf '[FAIL] %-30s see %s\n' "$tag" "$LOGF"
        fi
        exit 0
    fi
    flock "$LOCK" bash -c \
        "echo '$row,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-30s %s\n' "$tag" "$out"
    exit 0
fi

# ---------------- setup ----------------
mkdir -p "$LOGDIR"
[ -f eval_denoised.csv ] || echo "config,rho,seed,mlu" > eval_denoised.csv
[ -f repair_sweep.csv ] || echo "config,admm,rho,seed,mlu" > repair_sweep.csv
[ -f verify.csv ] || echo "config,rho,seed,mlu" > verify.csv
[ -f norepair_curve.csv ] || echo "config,rho,seed,mlu" > norepair_curve.csv
touch .matrix-done .sweep-done .verify-done .norepair-done \
      .matrix-lock .sweep-lock .verify-lock .norepair-lock

NBR="--mask-mode nbr"
ZEROFILL="--mask-mode zero --no-gate --hist-len 1"
EMBED="--mask-mode embed"

TASKS=()

# batch D: untrained control WITHOUT repair across the rho curve. The
# no-repair training gain (+22.9% at rho 0.3) is currently a single point;
# the effective-learning-interval claim needs the gain at other rho.
# rho 0.3 seeds 0-2 already exist (2x2.log: 2.8330 / 2.6648 / 3.0377) and
# are appended below rather than re-run. rho 0.05/0.02 probe the known
# gradient-collapse region, completing the boundary picture.
cat >> norepair_curve.csv <<'EOF'
untrained,0.3,0,2.8330
untrained,0.3,1,2.6648
untrained,0.3,2,3.0377
EOF
# de-duplicate if the batch is relaunched (idempotent CSV)
sort -u norepair_curve.csv > .nr.tmp \
    && printf '%s\n' "$(head -1 norepair_curve.csv)" > .nr.hdr \
    && { grep -v '^config,' .nr.tmp; } >> .nr.hdr \
    && mv .nr.hdr norepair_curve.csv && rm -f .nr.tmp
# batch F: TRAINED no-repair cells at rho 0.05/0.02 (the missing
# lower edge of the learning window). The main-topology claim "training
# collapses at rho<=0.05 without repair" has never been directly
# measured -- and batch E on Starlink2224 shows trained no-repair at
# rho=0.05 IMPROVING by 27% over untrained there, i.e. the collapse
# boundary moves with scale. These 6 cells pin down where the boundary
# actually sits on the main topology. Same CSV as batch D so trained
# and untrained pair by (rho, seed).
for seed in 0 1 2; do
    for rho in 0.05 0.02; do
        TASKS+=("ours|$rho|$seed|norepair|0|$EMBED")
    done
done

echo "===== supplement: ${#TASKS[@]} tasks on $NWORKER workers =====" \
    | tee -a "$LOG"
date | tee -a "$LOG"

if [ -n "$DRY" ]; then
    printf '%s\n' "${TASKS[@]}"
    echo "TOTAL ${#TASKS[@]}"
    exit 0
fi

printf '%s\n' "${TASKS[@]}" \
    | xargs -d '\n' -P "$NWORKER" -I {} bash "$SELF" --task {} 2>&1 \
    | tee -a "$LOG"

echo "===== supplement done =====" | tee -a "$LOG"
date | tee -a "$LOG"
echo "eval_denoised.csv: $(wc -l < eval_denoised.csv) lines" | tee -a "$LOG"
echo "repair_sweep.csv:  $(wc -l < repair_sweep.csv) lines" | tee -a "$LOG"
echo "failures: $(grep -c '^\[FAIL\]' "$LOG")" | tee -a "$LOG"
