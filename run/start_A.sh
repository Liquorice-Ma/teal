#!/bin/bash
# Pilot A: larger reward sample count to reduce COMA gradient variance
# for the MLU objective (see run_experiments.sh for the full matrix).
cd "$(dirname "$0")"
exec /root/autodl-tmp/conda/envs/teal/bin/python teal.py \
  --obj min_max_link_util --topo Starlink2272.json --tm-model starlink \
  --prune-demands --obs-ratio 0.3 --mask-mode embed --hist-len 3 \
  --epochs 100 --early-stop True --seed 0 --lr 0.001 --samples 30 \
  --slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
  --slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101
