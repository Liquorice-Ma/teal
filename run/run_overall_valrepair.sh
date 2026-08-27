#!/bin/bash
# Full main-table rerun after aligning validation checkpoint selection with
# the deployed repair protocol. Reuses the rho=0.3 pilot and unaffected
# untrained controls; reruns every trained allocator at the remaining rhos.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--samples 30 --admm-steps 2 --repair-input nbr $S"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
UNTRAIN="--epochs 0 --num-restart 1"

CSV=overall_valrepair.csv
DONE=.overall-valrepair-done
LOCK=.overall-valrepair-lock
WORKERS=${1:-4}

[ -f "$CSV" ] || echo "config,rho,seed,mlu" > "$CSV"
touch "$DONE" "$LOCK"

# Reuse repaired-validation pilot cells at rho=0.3 (excluding no-gate).
if ! grep -q '^ours,0.3,' "$CSV" && [ -f valrepair_pilot.csv ]; then
    awk -F, 'NR>1 && $1!="ours-no-gate" {
        print $0 >> "'"$CSV"'"; print $1"-"$2"-"$3 >> "'"$DONE"'"
    }' valrepair_pilot.csv
fi
# Untrained has no checkpoint selection, so its original cells are valid.
if ! grep -q '^untrained,' "$CSV" && [ -f overall_main.csv ]; then
    awk -F, 'NR>1 && $1=="untrained" {
        print $0 >> "'"$CSV"'"; print $1"-"$2"-"$3 >> "'"$DONE"'"
    }' overall_main.csv
fi

arch_of() {
    case "$1" in
        ours|untrained) echo "--mask-mode embed --hist-len 3" ;;
        test-style)     echo "--mask-mode zero --no-gate --hist-len 3" ;;
        zero-fill)      echo "--mask-mode zero --no-gate --hist-len 1" ;;
        mean-interp)    echo "--mask-mode mean --no-gate --hist-len 1" ;;
        nbr-fill)       echo "--mask-mode nbr --no-gate --hist-len 1" ;;
    esac
}
export -f arch_of

run_one() {
    local spec="$1" cfg rho seed tag mode out
    IFS='|' read -r cfg rho seed <<< "$spec"
    tag="$cfg-$rho-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }
    if [ "$cfg" = untrained ]; then mode="$UNTRAIN"; else mode="$TRAIN"; fi
    out=$($PY teal.py $COMMON $(arch_of "$cfg") --obs-ratio "$rho" \
        --seed "$seed" $mode 2>&1 | tr '\r' '\n' \
        | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }
    flock "$LOCK" -c \
        "echo '$cfg,$rho,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-30s %s\n' "$tag" "$out"
}
export -f run_one
export PY COMMON TRAIN UNTRAIN CSV DONE LOCK

TASKS=()
for rho in 0.1 0.05 0.02 0.5 0.3; do
    for cfg in ours test-style zero-fill mean-interp nbr-fill untrained; do
        for seed in 0 1 2; do TASKS+=("$cfg|$rho|$seed"); done
    done
done

echo "preseeded $(($(wc -l < "$CSV") - 1)) cells"
echo "total grid: ${#TASKS[@]} workers: $WORKERS"
printf '%s\n' "${TASKS[@]}" | xargs -P "$WORKERS" -I{} bash -c 'run_one "{}"'
echo "ALL DONE -> $CSV"
