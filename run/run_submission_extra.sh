#!/bin/bash
# Submission-strengthening batch after validation/test repair alignment.
# Phase A fills low-rho no-postprocessing architecture ablations.
# Phase B extends the deployable main table from 3 to 5 seeds.

set -u
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
WORKERS=${1:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"
LOGDIR=submission-task-logs
mkdir -p "$LOGDIR"

run_one() {
    local phase="$1" cfg="$2" rho="$3" seed="$4"
    local csv done lock tag arch mode admm repair logf out

    if [ "$phase" = ablation ]; then
        csv=ablation_lowrho.csv
        done=.ablation-lowrho-done
        lock=.ablation-lowrho-lock
        admm=0
        repair=""
        case "$cfg" in
            no-embed)    arch="--mask-mode zero --hist-len 3" ;;
            no-gate)     arch="--mask-mode embed --no-gate --hist-len 3" ;;
            no-temporal) arch="--mask-mode embed --hist-len 1" ;;
        esac
        mode="$TRAIN"
    else
        csv=overall_valrepair_5seed.csv
        done=.overall-valrepair-5seed-done
        lock=.overall-valrepair-5seed-lock
        admm=2
        repair="--repair-input nbr"
        case "$cfg" in
            ours|untrained) arch="--mask-mode embed --hist-len 3" ;;
            test-style)     arch="--mask-mode zero --no-gate --hist-len 3" ;;
            zero-fill)      arch="--mask-mode zero --no-gate --hist-len 1" ;;
            mean-interp)    arch="--mask-mode mean --no-gate --hist-len 1" ;;
            nbr-fill)       arch="--mask-mode nbr --no-gate --hist-len 1" ;;
        esac
        if [ "$cfg" = untrained ]; then mode="$UNTRAIN"; else mode="$TRAIN"; fi
    fi

    tag="$phase-$cfg-$rho-$seed"
    grep -qxF "$tag" "$done" 2>/dev/null && { echo "[skip] $tag"; return; }
    logf="$LOGDIR/$tag.log"
    echo "[run ] $tag"
    $PY teal.py $BASE $mode $arch --admm-steps "$admm" $repair \
        --obs-ratio "$rho" --seed "$seed" > "$logf" 2>&1
    out=$(tr '\r' '\n' < "$logf" | grep 'Testing: 100' \
        | grep -oE 'obj=[0-9.]+' | tail -1 | cut -d= -f2)
    if [ -z "$out" ]; then
        grep -q 'OutOfMemoryError' "$logf" && echo "[OOM ] $tag" || echo "[FAIL] $tag"
        return
    fi
    flock "$lock" bash -c \
        "echo '$cfg,$rho,$seed,$out' >> '$csv'; echo '$tag' >> '$done'"
    echo "[done] $tag $out"
}
export -f run_one
export PY BASE TRAIN UNTRAIN LOGDIR

[ -f ablation_lowrho.csv ] || echo "config,rho,seed,mlu" > ablation_lowrho.csv
touch .ablation-lowrho-done .ablation-lowrho-lock
if [ ! -f overall_valrepair_5seed.csv ]; then
    cp overall_valrepair.csv overall_valrepair_5seed.csv
fi
touch .overall-valrepair-5seed-done .overall-valrepair-5seed-lock

# Phase A: complete module ablations at the extreme-sparsity points first.
A=()
for rho in 0.05 0.02; do
    for cfg in no-embed no-gate no-temporal; do
        for seed in 0 1 2; do A+=("ablation|$cfg|$rho|$seed"); done
    done
done
echo "===== phase A: ${#A[@]} low-rho ablation cells ====="
printf '%s\n' "${A[@]}" | xargs -P "$WORKERS" -I{} bash -c \
    'IFS="|" read -r p c r s <<< "$1"; run_one "$p" "$c" "$r" "$s"' _ {}

# Phase B: add seeds 3 and 4 for every main-table method and rho.
B=()
for rho in 0.02 0.05 0.1 0.3 0.5; do
    for cfg in ours test-style zero-fill mean-interp nbr-fill untrained; do
        for seed in 3 4; do B+=("main5|$cfg|$rho|$seed"); done
    done
done
echo "===== phase B: ${#B[@]} main-table extension cells ====="
printf '%s\n' "${B[@]}" | xargs -P "$WORKERS" -I{} bash -c \
    'IFS="|" read -r p c r s <<< "$1"; run_one "$p" "$c" "$r" "$s"' _ {}

echo "ALL DONE"
