#!/usr/bin/env bash
# GPU experiment matrix for SiTE (run on the 4090 server).
#
# Usage:
#   bash run_experiments.sh            # all batches: A B C D
#   bash run_experiments.sh A B        # selected batches
#   EPOCHS=150 bash run_experiments.sh # override epochs (set after pilot!)
#
# Batches:
#   A  main comparison  : {ours, zero-fill, mean-interp} x {mlu, flow}
#                         x rho{0.7,0.5,0.3,0.1} + full-observation oracle
#   B  ablation (mlu)   : {-temporal, -gate, -embed} x rho sweep
#   C  zero-retraining  : demand-split, Drop5/Drop10 topologies, link failures
#   D  sensitivity      : node/top sampling, hist-len 5/8, capacity 500/2000,
#                         GDP trace  (switches data files; runs last)
#
# Results: one line per run in results-summary.csv; raw logs in logs/;
# per-snapshot rows remain in teal-{obj}-all.csv. Re-running skips
# completed runs recorded in .exp-done.

set -u
cd "$(dirname "$0")"
mkdir -p logs

EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-"0 1 2"}
TOPO=Starlink2272.json
SLICES="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
COMMON="--topo $TOPO --tm-model starlink --prune-demands --admm-steps 2 \
--epochs $EPOCHS $SLICES"
SUMMARY=results-summary.csv
DONE=.exp-done

[ -f "$SUMMARY" ] || echo "tag,final_obj,runtime_s" > "$SUMMARY"
touch "$DONE"

run_one() {
    local tag="$1"; shift
    if grep -qxF "$tag" "$DONE"; then
        echo "[skip] $tag"; return
    fi
    local log="logs/$(echo "$tag" | tr ' /=' '__-').log"
    echo "[run ] $tag"
    if ! python teal.py $COMMON "$@" > "$log" 2>&1; then
        echo "[FAIL] $tag (see $log)"; return
    fi
    local obj rt
    obj=$(grep -o "obj=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
    rt=$(grep -o "runtime=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
    echo "$tag,$obj,$rt" >> "$SUMMARY"
    echo "$tag" >> "$DONE"
}

batch_A() {
    echo "===== batch A: main comparison ====="
    for obj in min_max_link_util total_flow; do
        for s in $SEEDS; do
            # full-observation upper reference (original Teal behavior)
            run_one "A obj=$obj method=oracle rho=1.0 seed=$s" \
                --obj $obj --obs-ratio 1.0 --hist-len 1 --seed $s
            for rho in 0.7 0.5 0.3 0.1; do
                run_one "A obj=$obj method=ours rho=$rho seed=$s" \
                    --obj $obj --obs-ratio $rho --mask-mode embed \
                    --hist-len 3 --seed $s
                run_one "A obj=$obj method=zero rho=$rho seed=$s" \
                    --obj $obj --obs-ratio $rho --mask-mode zero \
                    --hist-len 1 --no-gate --seed $s
                run_one "A obj=$obj method=mean rho=$rho seed=$s" \
                    --obj $obj --obs-ratio $rho --mask-mode mean \
                    --hist-len 1 --no-gate --seed $s
            done
        done
    done
}

batch_B() {
    echo "===== batch B: ablation (mlu) ====="
    for s in $SEEDS; do
        for rho in 0.7 0.5 0.3 0.1; do
            run_one "B ablate=-temporal rho=$rho seed=$s" \
                --obj min_max_link_util --obs-ratio $rho \
                --mask-mode embed --hist-len 1 --seed $s
            run_one "B ablate=-gate rho=$rho seed=$s" \
                --obj min_max_link_util --obs-ratio $rho \
                --mask-mode embed --hist-len 3 --no-gate --seed $s
            run_one "B ablate=-embed rho=$rho seed=$s" \
                --obj min_max_link_util --obs-ratio $rho \
                --mask-mode zero --hist-len 3 --seed $s
        done
    done
}

batch_C() {
    echo "===== batch C: zero-retraining ====="
    for s in $SEEDS; do
        for obj in min_max_link_util total_flow; do
            run_one "C exp=demand-split obj=$obj seed=$s" \
                --obj $obj --obs-ratio 0.3 --mask-mode embed --hist-len 3 \
                --demand-split --seed $s
        done
        for drop in 5 10; do
            run_one "C exp=drop$drop seed=$s" \
                --obj min_max_link_util --obs-ratio 0.3 --mask-mode embed \
                --hist-len 3 --test-topo Starlink2272Drop$drop.json --seed $s
        done
        run_one "C exp=failures50 seed=$s" \
            --obj min_max_link_util --obs-ratio 0.3 --mask-mode embed \
            --hist-len 3 --failures 50 --seed $s
    done
}

batch_D() {
    echo "===== batch D: sensitivity ====="
    for s in $SEEDS; do
        for rho in 0.7 0.5 0.3 0.1; do
            run_one "D exp=node rho=$rho seed=$s" \
                --obj min_max_link_util --obs-ratio $rho --obs-type node \
                --mask-mode embed --hist-len 3 --seed $s
            run_one "D exp=top rho=$rho seed=$s" \
                --obj min_max_link_util --obs-ratio $rho --obs-sample top \
                --mask-mode embed --hist-len 3 --seed $s
        done
        for L in 5 8; do
            run_one "D exp=hist$L seed=$s" \
                --obj min_max_link_util --obs-ratio 0.3 --mask-mode embed \
                --hist-len $L --seed $s
        done
    done

    # capacity sweep: regenerating the topology json changes capacities
    # globally, so this section must not interleave with other batches
    for cap in 500 2000; do
        echo "----- capacity $cap -----"
        python prepare_starlink.py --size-x 22 --size-y 72 --capacity $cap \
            > logs/prepare-cap$cap.log 2>&1
        for s in $SEEDS; do
            run_one "D exp=cap$cap method=ours seed=$s" \
                --obj min_max_link_util --obs-ratio 0.3 --mask-mode embed \
                --hist-len 3 --seed $s
            run_one "D exp=cap$cap method=zero seed=$s" \
                --obj min_max_link_util --obs-ratio 0.3 --mask-mode zero \
                --hist-len 1 --no-gate --seed $s
        done
    done
    python prepare_starlink.py --size-x 22 --size-y 72 --capacity 1000 \
        > logs/prepare-cap1000.log 2>&1   # restore default

    # GDP trace: overwrites the traffic-matrix pkls, restore afterwards
    echo "----- GDP trace -----"
    python prepare_starlink.py --size-x 22 --size-y 72 --capacity 1000 \
        --dataset starlink_gdp > logs/prepare-gdp.log 2>&1
    for s in $SEEDS; do
        run_one "D exp=gdp method=ours seed=$s" \
            --obj min_max_link_util --obs-ratio 0.3 --mask-mode embed \
            --hist-len 3 --seed $s
        run_one "D exp=gdp method=zero seed=$s" \
            --obj min_max_link_util --obs-ratio 0.3 --mask-mode zero \
            --hist-len 1 --no-gate --seed $s
    done
    python prepare_starlink.py --size-x 22 --size-y 72 --capacity 1000 \
        > logs/prepare-restore.log 2>&1   # restore population trace
}

BATCHES=${@:-A B C D}
echo "epochs=$EPOCHS seeds=[$SEEDS] batches=[$BATCHES]"
for b in $BATCHES; do
    batch_$b
done
echo "all done; summary in $SUMMARY"
