#!/bin/bash
# Handoff watcher: waits until the original chain (run_final_matrix.sh)
# finishes its parallel phase and enters the serial M3 loop, then kills
# the chain process group and relaunches M3 with run_serial_m3.sh
# (parallelized). Detection key: serial tasks are the only ones using
# --demand-split / --test-topo / --failures.
# Fail-safe: if detection never fires, the original chain just keeps
# running serially; if the chain dies on its own, M3 is relaunched anyway
# (completed tasks are skipped via .final-done).

cd "$(dirname "$0")"
PGID=659161
LOG=serial_handoff.log

launch() {
    nohup bash run_serial_m3.sh >> chain_final.log 2>&1 &
    echo "[handoff] launched run_serial_m3.sh pid $! at $(date)" >> "$LOG"
    exit 0
}

while :; do
    if pgrep -af 'teal\.py' 2>/dev/null \
            | grep -qE -- '--demand-split|--test-topo|--failures 50'; then
        echo "[handoff] serial phase detected at $(date)" >> "$LOG"
        kill -TERM -- -$PGID 2>> "$LOG"
        sleep 15
        kill -KILL -- -$PGID 2> /dev/null
        pkill -9 -g $PGID 2> /dev/null
        sleep 2
        echo "[handoff] chain pgid $PGID killed at $(date)" >> "$LOG"
        launch
    fi
    if ! kill -0 -- -$PGID 2> /dev/null; then
        echo "[handoff] chain pgid gone at $(date), launching M3" >> "$LOG"
        launch
    fi
    sleep 30
done
