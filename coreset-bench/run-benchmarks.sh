#!/usr/bin/env bash
# Runs the whole performance suite (JMH kernels, node/gRPC ingestion and end-to-end harnesses, JFR profiles) and writes
# raw results plus a markdown summary, with host, JVM and load average recorded before and after every phase.
#
# usage: coreset-bench/run-benchmarks.sh [--quick] [--skip-build] [--no-jfr] [--jmh REGEX] [--out DIR]
#   --quick       1 fork, 2x1s warmup, 3x1s measurement, smaller harness runs (about 10 minutes instead of 40)
#   --skip-build  reuse coreset-bench/target/benchmarks.jar
#   --no-jfr      skip the JFR profiling phase
#   --jmh REGEX   only JMH benchmarks matching REGEX (default: all)
#   --out DIR     output directory (default: coreset-bench/results/<UTC timestamp>)
# env: JAVA_HOME (JDK 17+); JFR_TOOL optional path to a JDK 21+ `jfr` binary for `jfr view` summaries.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUICK=0
BUILD=1
JFR=1
JMH_REGEX=""
OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1 ;;
        --skip-build) BUILD=0 ;;
        --no-jfr) JFR=0 ;;
        --jmh)
            JMH_REGEX="$2"
            shift
            ;;
        --out)
            OUT="$2"
            shift
            ;;
        *)
            echo "unknown option $1" >&2
            exit 2
            ;;
    esac
    shift
done

JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
JFR_BIN="${JFR_TOOL:-${JAVA_HOME:+$JAVA_HOME/bin/}jfr}"
JAR="$ROOT/coreset-bench/target/benchmarks.jar"
OUT="${OUT:-$ROOT/coreset-bench/results/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
ENV="$OUT/env.md"
HARNESS_JVM=(-Xms8g -Xmx8g -XX:+AlwaysPreTouch)

load() { uptime | sed 's/.*load average[s]*: //'; }

host_info() {
    echo "# Benchmark environment"
    echo
    echo "- date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- git: $(git -C "$ROOT" rev-parse --short HEAD) ($(git -C "$ROOT" rev-parse --abbrev-ref HEAD))$(git -C "$ROOT" diff --quiet || echo ', dirty')"
    echo "- os: $(uname -srm)"
    if command -v sysctl >/dev/null && sysctl -n machdep.cpu.brand_string >/dev/null 2>&1; then
        echo "- cpu: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) logical cpus" \
            "($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?') performance + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0) efficiency)," \
            "$(($(sysctl -n hw.memsize) / 1073741824)) GiB RAM"
    elif [[ -r /proc/cpuinfo ]]; then
        echo "- cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-), $(nproc) logical cpus"
    fi
    echo "- java: $("$JAVA" -version 2>&1 | tr '\n' ' ')"
    echo "- mode: $([[ $QUICK == 1 ]] && echo quick || echo full)"
    echo "- harness JVM flags: ${HARNESS_JVM[*]}"
    echo
    echo "| phase | load avg before (1/5/15 min) | load avg after | seconds |"
    echo "|---|---|---|---|"
}

phase() {
    local name="$1"
    shift
    local before t0
    before="$(load)"
    t0=$(date +%s)
    echo ">> $name (load $before)" >&2
    "$@"
    echo "| $name | $before | $(load) | $(($(date +%s) - t0)) |" >>"$ENV"
}

host_info >"$ENV"

if [[ $BUILD == 1 ]]; then
    phase build mvn -B -q -f "$ROOT/pom.xml" -pl coreset-bench -am package -DskipTests
fi

if [[ $QUICK == 1 ]]; then
    JMH_OPTS=(-f 1 -wi 2 -w 1s -i 3 -r 1s)
    NODE_ARGS=(--points 400000 --repeats 2)
    E2E_ARGS=(--points 50000 --repeats 2)
else
    JMH_OPTS=()
    NODE_ARGS=(--points 1000000 --repeats 3)
    E2E_ARGS=(--points 100000 --repeats 3)
fi

phase jmh "$JAVA" -jar "$JAR" ${JMH_OPTS[@]+"${JMH_OPTS[@]}"} -prof gc -rf json -rff "$OUT/jmh.json" \
    ${JMH_REGEX:+"$JMH_REGEX"} >"$OUT/jmh.txt" 2>&1

phase node-ingest "$JAVA" "${HARNESS_JVM[@]}" -cp "$JAR" io.github.c0rwin.coreset.bench.perf.NodeIngestHarness \
    "${NODE_ARGS[@]}" >"$OUT/node-ingest.md"

phase grpc-e2e "$JAVA" "${HARNESS_JVM[@]}" -cp "$JAR" io.github.c0rwin.coreset.bench.perf.GrpcEndToEndHarness \
    "${E2E_ARGS[@]}" >"$OUT/grpc-e2e.md"

if [[ $JFR == 1 ]]; then
    JFR_DIR="$OUT/jfr"
    mkdir -p "$JFR_DIR"
    # Hot path 1: in-process node ingestion (compress, seeding, merge-and-reduce, metrics).
    phase jfr-node "$JAVA" "${HARNESS_JVM[@]}" \
        "-XX:StartFlightRecording=filename=$JFR_DIR/node-ingest.jfr,settings=profile" \
        -cp "$JAR" io.github.c0rwin.coreset.bench.perf.NodeIngestHarness \
        --points 2000000 --workers 4 --modes single-writer --grpc-nodes 0 --repeats 2 >"$JFR_DIR/node-ingest.out" 2>&1
    # Hot path 2: gRPC ingestion (codec, Netty, node).
    phase jfr-grpc "$JAVA" "${HARNESS_JVM[@]}" \
        "-XX:StartFlightRecording=filename=$JFR_DIR/grpc-ingest.jfr,settings=profile" \
        -cp "$JAR" io.github.c0rwin.coreset.bench.perf.NodeIngestHarness \
        --points 1000000 --workers 1 --modes single-writer --grpc-nodes 2 --repeats 2 >"$JFR_DIR/grpc-ingest.out" 2>&1
    # Hot path 3: k-means++ seeding and Lloyd on a large input (final clustering, mean coreset, candidates).
    # One recording per benchmark: every JMH fork would otherwise overwrite the same file.
    for b in seed lloydIteration; do
        phase "jfr-kmeans-$b" "$JAVA" -jar "$JAR" "KMeansBenchmark.$b\$" -p n=100000 -p k=50 -p d=16 -f 1 -wi 2 -i 5 \
            -jvmArgsAppend "-XX:StartFlightRecording=filename=$JFR_DIR/kmeans-$b.jfr,settings=profile" \
            >"$JFR_DIR/kmeans-$b.out" 2>&1
    done
    if [[ -n "${JFR_TOOL:-}" ]]; then
        for f in "$JFR_DIR"/*.jfr; do
            for v in hot-methods allocation-by-site allocation-by-class gc-pauses; do
                "$JFR_BIN" view --width 140 --truncate beginning "$v" "$f" >"${f%.jfr}.$v.txt" 2>&1 || true
            done
        done
    else
        for f in "$JFR_DIR"/*.jfr; do
            "$JFR_BIN" summary "$f" >"${f%.jfr}.summary.txt" 2>&1 || true
        done
    fi
fi

{
    cat "$ENV"
    echo
    echo "## JMH (score ± 99.9% CI; gc.alloc.rate.norm = bytes allocated per operation)"
    echo
    echo '```'
    awk '/^Benchmark /{p=1} p' "$OUT/jmh.txt" | grep -Ev ':(gc\.alloc\.rate|gc\.count|gc\.time) ' || true
    echo '```'
    echo
    cat "$OUT/node-ingest.md"
    echo
    cat "$OUT/grpc-e2e.md"
} >"$OUT/summary.md"

echo "results in $OUT (summary.md, jmh.json, node-ingest.md, grpc-e2e.md$([[ $JFR == 1 ]] && echo ', jfr/'))" >&2
