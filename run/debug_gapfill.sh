#!/bin/bash
# Diagnose the gap-fill failures. The runner pipes stderr into the obj
# parser, so the actual error was discarded; this re-runs one failed
# config alone and keeps everything.
#
# Short schedule (2 epochs, 1 restart) on purpose: it answers "is the
# config itself broken?" in minutes. If this succeeds, the config is
# fine and the failures came from running 16 of them concurrently.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
CORE="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands --hist-len 3 \
--samples 30 --admm-steps 2 $S"

cfg="${1:-mean-interp}"
rho="${2:-0.05}"
case "$cfg" in
    mean-interp) EXTRA="--mask-mode mean --no-gate --hist-len 1" ;;
    mean-gated)  EXTRA="--mask-mode mean" ;;
    nbr)         EXTRA="--mask-mode nbr" ;;
    ours)        EXTRA="--mask-mode embed" ;;
esac

echo "### $cfg rho=$rho, 2 epochs / 1 restart, full output ###"
$PY teal.py $CORE --epochs 2 --num-restart 1 --early-stop True \
    --lr 0.0001 --obs-ratio "$rho" --seed 0 $EXTRA 2>&1 \
    | grep -vE "UserWarning|warnings.warn|FutureWarning|^\s*$|nx\.node_link|\
The default value|To make this|return F\.linear|proj = linear|q @ k|h_fuse|\
return Variable|^  " \
    | tail -25
