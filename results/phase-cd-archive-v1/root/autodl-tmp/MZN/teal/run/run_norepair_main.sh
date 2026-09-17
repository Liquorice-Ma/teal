#!/bin/bash
# Reviewer-round batch: isolate the allocator from the post-processing layer.
#
# Motivation. In the deployable main table every method shares the same
# neighbor-assisted rebalancing layer (--admm-steps 2 --repair-input nbr).
# The 2x2 diagnostic showed that this layer absorbs the learned contribution:
# with repair on, methods collapse to within seed noise; with repair off the
# training gain reappears (rho=0.05: 3/3 paired wins, -20.5% mean MLU).
# Phase C therefore reruns the *same six external methods* with the
# post-processing layer disabled, so the allocator itself is the only factor.
#
# Phase D adds node-level metering (--obs-type node), where an unmeasured
# source hides a whole demand row. It is compared against the flow-level
# cells of the deployable main table, so it keeps that protocol
# (--admm-steps 2 --repair-input nbr) rather than Phase C's.
#
# Protocol frozen to match published cells: k-shortest edge-disjoint paths,
# deterministic, lr 1e-4, 60 epochs, restart screening x3, warmup 4.
#
# NOTE the GPU may be shared with unrelated jobs; WORKERS defaults to 2 and
# every task logs OOM separately so a failed cell never enters the CSV.
#
# Usage (detach-safe):
#   setsid nohup ./run_norepair_main.sh 2 > norepair_main.log 2>&1 &

set -u
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
WORKERS=${1:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"
LOGDIR=norepair-main-logs
mkdir -p "$LOGDIR"

# same architecture mapping as the deployable main table
arch_of() {
    case "$1" in
        ours|untrained) echo "--mask-mode embed --hist-len 3" ;;
        test-style)     echo "--mask-mode zero --no-gate --hist-len 3" ;;
        zero-fill)      echo "--mask-mode zero --no-gate --hist-len 1" ;;
        mean-interp)    echo "--mask-mode mean --no-gate --hist-len 1" ;;
        nbr-fill)       echo "--mask-mode nbr --no-gate --hist-len 1" ;;
        *) return 1 ;;
    esac
}
export -f arch_of

run_one() {
    local phase="$1" cfg="$2" rho="$3" seed="$4"
    local csv done lock tag arch mode post extra logf out

    case "$phase" in
        norepair)
            # allocator in isolation: post-processing disabled
            csv=norepair_main.csv
            done=.norepair-main-done
            lock=.norepair-main-lock
            post="--admm-steps 0"
            extra=""
            ;;
        nodeobs)
            # node-level metering under the deployable protocol
            csv=nodeobs.csv
            done=.nodeobs-done
            lock=.nodeobs-lock
            post="--admm-steps 2 --repair-input nbr"
            extra="--obs-type node"
            ;;
        *) echo "[BUG ] unknown phase $phase"; return ;;
    esac

    arch=$(arch_of "$cfg") || { echo "[BUG ] unknown cfg $cfg"; return; }
    if [ "$cfg" = untrained ]; then mode="$UNTRAIN"; else mode="$TRAIN"; fi

    tag="$phase-$cfg-$rho-$seed"
    grep -qxF "$tag" "$done" 2>/dev/null && { echo "[skip] $tag"; return; }
    logf="$LOGDIR/$tag.log"
    echo "[run ] $tag"
    $PY teal.py $BASE $mode $arch $post $extra \
        --obs-ratio "$rho" --seed "$seed" > "$logf" 2>&1
    out=$(tr '\r' '\n' < "$logf" | grep 'Testing: 100' \
        | grep -oE 'obj=[0-9.]+' | tail -1 | cut -d= -f2)
    if [ -z "$out" ]; then
        grep -q 'OutOfMemoryError' "$logf" \
            && echo "[OOM ] $tag" || echo "[FAIL] $tag"
        return
    fi
    flock "$lock" bash -c \
        "echo '$cfg,$rho,$seed,$out' >> '$csv'; echo '$tag' >> '$done'"
    echo "[done] $tag $out"
}
export -f run_one
export PY BASE TRAIN UNTRAIN LOGDIR

[ -f norepair_main.csv ] || echo "config,rho,seed,mlu" > norepair_main.csv
[ -f nodeobs.csv ] || echo "config,rho,seed,mlu" > nodeobs.csv
touch .norepair-main-done .norepair-main-lock .nodeobs-done .nodeobs-lock

# ---- Phase C: six external methods, no post-processing, 5 rho x 5 seeds ----
# untrained cells are cheap (0 epochs) and are scheduled first so that the
# paired training contrast has a complete control column early on.
C=()
for rho in 0.05 0.02 0.1 0.3 0.5; do
    for seed in 0 1 2 3 4; do C+=("norepair|untrained|$rho|$seed"); done
done
for rho in 0.05 0.02 0.1 0.3 0.5; do
    for cfg in ours test-style zero-fill mean-interp nbr-fill; do
        for seed in 0 1 2 3 4; do C+=("norepair|$cfg|$rho|$seed"); done
    done
done
echo "===== phase C: ${#C[@]} no-post-processing cells (workers $WORKERS) ====="
printf '%s\n' "${C[@]}" | xargs -P "$WORKERS" -I{} bash -c \
    'IFS="|" read -r p c r s <<< "$1"; run_one "$p" "$c" "$r" "$s"' _ {}

# ---- Phase D: node-level metering vs the flow-level main-table cells ----
# Only the node-level half is run; the flow-level counterparts already exist
# in overall_valrepair_5seed.csv under the identical protocol.
D=()
for rho in 0.3 0.1 0.5; do
    for cfg in ours zero-fill untrained; do
        for seed in 0 1 2; do D+=("nodeobs|$cfg|$rho|$seed"); done
    done
done
echo "===== phase D: ${#D[@]} node-level cells ====="
printf '%s\n' "${D[@]}" | xargs -P "$WORKERS" -I{} bash -c \
    'IFS="|" read -r p c r s <<< "$1"; run_one "$p" "$c" "$r" "$s"' _ {}

echo "ALL DONE -> norepair_main.csv / nodeobs.csv"
