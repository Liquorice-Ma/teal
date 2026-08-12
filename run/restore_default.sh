#!/bin/bash
# Restore the default calibration (capacity 2000) and recompute the LP
# oracle used as the PR denominator.
cd "$(dirname "$0")"
PY=/root/autodl-tmp/conda/envs/teal/bin/python
$PY prepare_starlink.py --size-x 22 --size-y 72 --capacity 2000 2>&1 | tail -1
$PY lp_oracle.py --topo Starlink2272.json --tm-model starlink \
    --prune-demands --slice-test-start 90 --slice-test-stop 101 2>&1 | tail -1
$PY -c "import pandas as pd; print('oracle MLU = %.4f' % \
pd.read_csv('lp-oracle-Starlink2272.json.csv').opt_mlu.mean())"
