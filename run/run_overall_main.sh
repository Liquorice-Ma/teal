#!/bin/bash
# Overall comparison under the DEPLOYABLE protocol (paper main table).
#
# Every method gets the identical deployable repair layer: 2 rebalancing
# sweeps whose link-utilization estimate reads only the observation with
# neighbor fill (--repair-input nbr). The only factor that varies across
# methods is the learned allocator:
#   ours        : embed fill + mask-aware gate + temporal (hist 3)
#   test-style  : zero fill + no gate + temporal (hist 3)   [TEST, TMC'26]
#   zero-fill   : zero fill + no gate + hist 1   [full-obs methods' sparse
#                 adaptation: SaTE/Teal-style model fed zero-filled input]
#   mean-interp : mean completion + no gate + hist 1  [two-stage baseline]
#   nbr-fill    : neighbor completion + no gate + hist 1 [two-stage, strong]
#   untrained   : ours architecture, weights frozen at init (ablation)
#
# Grid: rho in {0.3, 0.1, 0.5, 0.05, 0.02} x seeds {0,1,2}.
# Existing cells are pre-seeded from deployable_repair.csv (ours/untrained
# with repair_input=nbr at rho 0.05-0.5) so they are not retrained.
#
# Detach-safe:
#   setsid nohup ./run_overall_main.sh 4 > overall_main.log 2>&1 &

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

CSV=overall_main.csv
DONE=.overall-main-done
LOCK=.overall-main-lock
WORKERS=${1:-4}

[ -f "$CSV" ] || echo "config,rho,seed,mlu" > "$CSV"
touch "$DONE" "$LOCK"

# pre-seed cells already measured in deployable_repair.csv (repair_input=nbr)
if [ -f deployable_repair.csv ] && ! grep -q "^ours," "$CSV"; then
    awk -F, 'NR>1 && $2=="nbr" && ($1=="ours" || $1=="untrained") {
        print $1","$3","$4","$5 >> "'"$CSV"'"
        print $1"-"$3"-"$4 >> "'"$DONE"'"
    }' deployable_repair.csv
    echo "pre-seeded $(grep -c . "$DONE") cells from deployable_repair.csv"
fi

arch_of() {
    case "$1" in
        ours)        echo "--mask-mode embed --hist-len 3" ;;
        untrained)   echo "--mask-mode embed --hist-len 3" ;;
        test-style)  echo "--mask-mode zero --no-gate --hist-len 3" ;;
        zero-fill)   echo "--mask-mode zero --no-gate --hist-len 1" ;;
        mean-interp) echo "--mask-mode mean --no-gate --hist-len 1" ;;
        nbr-fill)    echo "--mask-mode nbr --no-gate --hist-len 1" ;;
    esac
}
export -f arch_of

run_one() {
    local spec="$1"
    local cfg rho seed
    IFS='|' read -r cfg rho seed <<< "$spec"
    local tag="$cfg-$rho-$seed"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }

    local mode
    if [ "$cfg" = "untrained" ]; then mode="$UNTRAIN"; else mode="$TRAIN"; fi

    local out
    out=$($PY teal.py $COMMON $(arch_of "$cfg") --obs-ratio "$rho" \
        --seed "$seed" $mode 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }

    flock "$LOCK" -c \
        "echo '$cfg,$rho,$seed,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-30s %s\n' "$tag" "$out"
}
export -f run_one
export PY COMMON TRAIN UNTRAIN CSV DONE LOCK

# headline operating points first so partial results are usable early
TASKS=()
for rho in 0.3 0.1 0.5 0.05 0.02; do
    for cfg in ours test-style zero-fill mean-interp nbr-fill untrained; do
        for seed in 0 1 2; do
            TASKS+=("$cfg|$rho|$seed")
        done
    done
done

echo "total tasks: ${#TASKS[@]}  workers: $WORKERS"
printf '%s\n' "${TASKS[@]}" \
    | xargs -P "$WORKERS" -I{} bash -c 'run_one "{}"'
echo "ALL DONE -> $CSV"
