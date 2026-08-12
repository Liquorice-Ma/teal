#!/bin/bash
# B1 / reviewer point M2-1: a TEST-style baseline.
#
# TEST (Guo et al., TMC 2026) is the closest prior sparse-input TE system:
# zero-filling for unobserved demands, a Transformer over the sparse
# observation history, and a GCN over the topology --- with no mask-aware
# gating and no learnable placeholder.
#
# None of our existing configs matches that combination:
#   zero-fill : --mask-mode zero --no-gate --hist-len 1  (no Transformer)
#   no-embed  : --mask-mode zero            --hist-len 3  (keeps the gate)
# The missing cell is exactly zero-fill + Transformer + ungated GCN:
#   --mask-mode zero --no-gate --hist-len 3
#
# Two questions are answered at once:
#   (a) how does SiTE compare against the closest prior architecture, and
#   (b) does repair absorption reproduce on a *different* architecture ---
#       i.e. is absorption a property of the repair layer rather than of
#       our particular model?
# Hence the 2x2 of training status x repair budget, at two rho values.
#
# Sampling is left at the study-wide default (--obs-sample uniform) so that
# the only differences from `ours` are the fill strategy and the gate;
# TEST's own top-flow sampling is a separate factor and is not varied here.
#
# Detach-safe:
#   setsid nohup ./run_test_baseline.sh 8 > test_baseline.log 2>&1 &

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 $S"
# TEST-style architecture: zero fill, no mask-aware gate, temporal on.
ARCH="--mask-mode zero --no-gate"
TRAIN="--epochs 60 --early-stop True --lr 0.0001 --num-restart 3 \
--warmup-epochs 4"
# num-restart 1: restart screening trains warmup epochs and would stop the
# arm from being genuinely untrained.
UNTRAIN="--epochs 0 --num-restart 1"

CSV=test_baseline.csv
DONE=.test-baseline-done
LOCK=.test-baseline-lock
WORKERS=${1:-8}

[ -f "$CSV" ] || echo "config,rho,seed,repair,mlu" > "$CSV"
touch "$DONE" "$LOCK"

run_one() {
    local spec="$1"
    local cfg rho seed steps
    IFS='|' read -r cfg rho seed steps <<< "$spec"
    local tag="$cfg-$rho-$seed-r$steps"
    grep -qxF "$tag" "$DONE" && { echo "[skip] $tag"; return; }

    local mode
    if [ "$cfg" = "test-style" ]; then mode="$TRAIN"; else mode="$UNTRAIN"; fi

    local out
    out=$($PY teal.py $COMMON $ARCH --obs-ratio "$rho" --seed "$seed" \
        --admm-steps "$steps" $mode 2>&1 \
        | tr '\r' '\n' | grep 'Testing: 100' | tail -1 \
        | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    [ -z "$out" ] && { echo "[FAIL] $tag"; return; }

    flock "$LOCK" -c \
        "echo '$cfg,$rho,$seed,$steps,$out' >> '$CSV'; echo '$tag' >> '$DONE'"
    printf '[done] %-34s %s\n' "$tag" "$out"
}
export -f run_one
export PY COMMON ARCH TRAIN UNTRAIN CSV DONE LOCK

# repair-on cells first: they carry the absorption verdict.
TASKS=()
for steps in 2 0; do
    for rho in 0.3 0.05; do
        for seed in 0 1 2; do
            TASKS+=("test-style|$rho|$seed|$steps")
            TASKS+=("test-untrained|$rho|$seed|$steps")
        done
    done
done

echo "total tasks: ${#TASKS[@]}  workers: $WORKERS"
printf '%s\n' "${TASKS[@]}" \
    | xargs -P "$WORKERS" -I{} bash -c 'run_one "{}"'
echo "ALL DONE -> $CSV"
