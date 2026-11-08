#!/usr/bin/env bash
# Real-process smoke run of the loss/failure-resilience mechanisms: 4 node JVMs + 1 coordinator JVM on 127.0.0.1.
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
JAVA=${JAVA:-/Users/c0rwin/Library/Java/JavaVirtualMachines/corretto-17.0.5/Contents/Home/bin/java}
NODES=4
BASE=${BASE:-7271}
COMMON=(--input "$DIR/points.csv" --algorithm sensitivity --param k=10 --param m=1000 --k 10 --seed 3
    --workers 2 --batch-size 5000 --gather resilient --distortion-queries 10 --producer-id smoke)
PIDS=()

start_nodes() {
    PIDS=()
    for i in $(seq 0 $((NODES - 1))); do
        "$JAVA" -Xmx1g -jar "$DIR/coreset-node.jar" --port $((BASE + i)) --metrics-port -1 \
            >"$DIR/node$i.log" 2>&1 &
        PIDS+=($!)
    done
    for i in $(seq 0 $((NODES - 1))); do
        until grep -q "listening" "$DIR/node$i.log"; do sleep 0.1; done
    done
}

stop_nodes() {
    for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
}

targets() {
    local t=""
    for i in $(seq 0 $((NODES - 1))); do t="$t${t:+,}localhost:$((BASE + i))"; done
    echo "$t"
}

# run_with_kill NAME VICTIM coordinator-args...: SIGKILLs node VICTIM at the coordinator's fault barrier.
run_with_kill() {
    local name=$1 victim=$2
    shift 2
    local barrier="$DIR/$name.go"
    rm -f "$barrier"
    "$JAVA" -Xmx2g -jar "$DIR/coreset-coordinator.jar" --nodes "$(targets)" "${COMMON[@]}" \
        --fault-barrier "$barrier" "$@" >"$DIR/$name.out" 2>&1 &
    local coord=$!
    until grep -q FAULT_BARRIER "$DIR/$name.out" 2>/dev/null; do
        kill -0 "$coord" 2>/dev/null || {
            cat "$DIR/$name.out"
            return 1
        }
        sleep 0.1
    done
    kill -9 "${PIDS[$victim]}"
    echo "$(date +%T) $name: SIGKILL node $victim (pid ${PIDS[$victim]}, port $((BASE + victim)))" >>"$DIR/faults.log"
    sleep 0.5
    touch "$barrier"
    wait "$coord"
}

run() {
    local name=$1
    shift
    "$JAVA" -Xmx2g -jar "$DIR/coreset-coordinator.jar" --nodes "$(targets)" "${COMMON[@]}" "$@" \
        >"$DIR/$name.out" 2>&1
}

trap stop_nodes EXIT
PD=(--packetization per-draw --message-size 1024 --records-per-packet 16)

# A. survivor reweighting after SIGKILL of node 2 (no replicas)
start_nodes
run_with_kill A-none 2 "${PD[@]}" --session A-none --reweight none
stop_nodes
start_nodes
run_with_kill A-weight 2 "${PD[@]}" --session A-weight --reweight weight
stop_nodes

# B. replica failover: each node's frozen message on 1 peer, SIGKILL node 1
start_nodes
run_with_kill B-replica 1 "${PD[@]}" --session B --replicate 1
stop_nodes

# C. per-draw packets under 30% Bernoulli loss vs loss-free, 20 gathers on fresh sampling streams each
start_nodes
run C-lossfree "${PD[@]}" --session C0 --repeat 20
run C-loss30 "${PD[@]}" --session C1 --repeat 20 --loss bernoulli:0.3 --loss-seed 11
run C-loss30-assent "${PD[@]}" --session C2 --repeat 20 --loss bernoulli:0.3 --loss-seed 11 --decoder as-sent
run C-merged-ht --packetization merged --message-size 1024 --records-per-packet 16 --session C3 --repeat 20 \
    --loss bernoulli:0.3 --loss-seed 11 --decoder ht --known-loss 0.3

# D. mean coreset under 30% loss, one record per cell: CF triples vs global offset
CELLS=(--message-size 200 --records-per-packet 1)
run D-cf-assent --packetization mean-cf "${CELLS[@]}" --session D0 --repeat 20 --loss bernoulli:0.3 --decoder as-sent
run D-global-assent --packetization mean-global "${CELLS[@]}" --session D1 --repeat 20 --loss bernoulli:0.3 --decoder as-sent
run D-cf-rescaled --packetization mean-cf "${CELLS[@]}" --session D2 --repeat 20 --loss bernoulli:0.3 --decoder realized-weight
run D-global-rescaled --packetization mean-global "${CELLS[@]}" --session D3 --repeat 20 --loss bernoulli:0.3 --decoder realized-weight
run D-cf-lossfree --packetization mean-cf "${CELLS[@]}" --session D4 --repeat 1
stop_nodes
echo "smoke run finished"
