#!/usr/bin/env bash
# Pull experiment results from the GPU server and re-render figures locally.
#
# Usage (on the local Mac, inside run/):
#   bash pull_results.sh                  # default remote: autodl:~/teal
#   REMOTE=autodl REMOTE_DIR=/root/autodl-tmp/teal bash pull_results.sh

set -eu
cd "$(dirname "$0")"

REMOTE=${REMOTE:-autodl}
REMOTE_DIR=${REMOTE_DIR:-"~/teal"}

echo "pulling from $REMOTE:$REMOTE_DIR/run ..."
scp "$REMOTE:$REMOTE_DIR/run/results-summary.csv" . 2>/dev/null \
    || echo "  [warn] results-summary.csv not found on server yet"
scp "$REMOTE:$REMOTE_DIR/run/lp-oracle-*.csv" . 2>/dev/null \
    || echo "  [warn] lp-oracle csv not found on server yet"

if [ -f results-summary.csv ]; then
    echo "rendering figures locally ..."
    python process_results.py
    echo "opening result-figs/ ..."
    open result-figs 2>/dev/null || true
else
    echo "nothing to render yet."
fi
