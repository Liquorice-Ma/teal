#!/bin/bash
# Reproduce the failures50 crash with a minimal run (1 epoch, few samples).
# The M3 runner captured stdout into a variable, so nothing was logged;
# this re-runs the same config and keeps the full output.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
BASE="--shared-paths --deterministic --topo Starlink2272.json \
--tm-model starlink --prune-demands $S"

$PY teal.py $BASE --epochs 1 --early-stop True --lr 0.0001 \
    --num-restart 1 --warmup-epochs 0 --obj min_max_link_util \
    --admm-steps 2 --obs-ratio 0.3 --seed 0 --samples 5 \
    --mask-mode embed --hist-len 3 --failures 50 2>&1 | tail -40
