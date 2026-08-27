#!/bin/bash
# Pilot: align validation checkpoint selection with the deployed repair
# protocol, then compare every trained allocator fairly at rho=0.3.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 --admm-steps 2 --repair-input nbr $S \
--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4 --obs-ratio 0.3"

CSV=valrepair_pilot.csv
DONE=.valrepair-pilot-done
LOCK=.valrepair-pilot-lock
WORKERS=${1:-4}

[ -f "$CSV" ] || echo "config,rho,seed,mlu" > "$CSV"
touch "$DONE" "$LOCK"

arch_of() {
    case "$1" in
        ours)         echo "--mask-mode embed --hist-len 3" ;;
        ours-no-gate) echo "--mask-mode embed --no-gate --hist-len 3" ;;
        test-style)   echo "--mask-mode zero --no-gate --hist-len 3" ;;
        zero-fill)    echo "--mask-mode zero --no-gate --hist-len 1" ;;
        mean-interp)  echo "--mask-mode mean --no-gate --hist-len 1" ;;
        nbr-fill)     echo "--mask-mode nbr --no-gate --hist-len 1" ;;
    esac
}
export -f arch_of

run_one() {
    local spec="$1" cfg seed tag mode out
    IFS='|' read -r cfg seed <<< "$spec"
    tag="$cfg-0.3-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    mode=$(arch_of "$cfg")
    out=$($PY teal.py $COMMON $mode --seed "$seed" 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    flock "$LOCK" -c \
        "echo '$cfg,0.3,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-30s %s\n' "$tag" "$out"
}
export -f run_one
export PY COMMON CSV DONE LOCK

TASKS=()
for cfg in ours ours-no-gate test-style zero-fill mean-interp nbr-fill; do
    for seed in 0 1 2; do TASKS+=("$cfg|$seed"); done
done

echo "total tasks: ${#TASKS[@]} workers: $WORKERS"
printf '%s\n' "${TASKS[@]}" | xargs -P "$WORKERS" -I{} bash -c 'run_one "{}"'
echo "ALL DONE -> $CSV"
