#!/bin/bash
# Quick check after fixing the mask-embedding no-op: does the embed branch
# now make a measurable difference, and at which init scale?
#
# The embedding is additive on the unobserved entries, so its init scale
# decides whether it is distinguishable from zero-filling at all:
#   1     -> below P10 of the demand distribution, ~= zero-fill
#   10    -> P25
#   57    -> median (current default)
#   159   -> arithmetic mean, i.e. what mean-interp effectively fills
# One seed only; this is a scale probe, not the final measurement.

cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
S="--slice-train-start 0 --slice-train-stop 80 --slice-val-start 80 \
--slice-val-stop 90 --slice-test-start 90 --slice-test-stop 101"
B="--shared-paths --deterministic --obj min_max_link_util \
--topo Starlink2272.json --tm-model starlink --prune-demands \
--obs-ratio 0.3 --hist-len 3 --samples 30 --admm-steps 2 \
--epochs 60 --early-stop True --lr 0.0001 --seed 0 $S"

probe() {   # probe <label> <extra args...>
    local label="$1"; shift
    local out
    out=$($PY teal.py $B "$@" 2>&1 | tr '\r' '\n' | grep 'Testing: 100' \
        | tail -1 | grep -oE 'obj=[0-9.]+' | cut -d= -f2)
    printf '%-28s %s\n' "$label" "${out:-FAILED}"
}

echo "oracle MLU = $($PY -c "import pandas as pd; \
print('%.4f' % pd.read_csv('lp-oracle-shared.csv').opt_mlu.mean())" \
2>/dev/null || echo n/a)"
echo
probe "zero-fill (reference)"  --mask-mode zero --no-gate --hist-len 1
probe "mean-interp (reference)" --mask-mode mean --no-gate --hist-len 1
probe "embed init=10"          --mask-mode embed --mask-init 10
probe "embed init=57 (default)" --mask-mode embed --mask-init 57
probe "embed init=159"         --mask-mode embed --mask-init 159
