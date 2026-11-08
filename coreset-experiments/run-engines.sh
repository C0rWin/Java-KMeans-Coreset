#!/usr/bin/env bash
# Runs an engine-comparison grid: gRPC coreset cluster vs Spark vs Flink (see docs/experiments/engines.md).
#
#   coreset-experiments/run-engines.sh [preset|config.yaml] [--out DIR] [--only REGEX] [--summary-only] [--no-build]
#
# Presets: grid (configs/preset-grid.yaml, the full systems x N x algorithms x datasets grid), smoke-1m
# (configs/smoke-1m.yaml, one 1M x 20 dataset). Writes DIR/results.jsonl, DIR/summary.md, DIR/environment.json and
# per-cell logs under DIR/cells/. Needs JAVA_HOME (Java 17) and Maven; never installs into ~/.m2.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"
preset="${1:-smoke-1m}"
[[ $# -gt 0 ]] && shift

case "$preset" in
    grid) config="$here/configs/preset-grid.yaml" ;;
    smoke-1m) config="$here/configs/smoke-1m.yaml" ;;
    *) config="$(cd "$(dirname "$preset")" && pwd)/$(basename "$preset")" ;;
esac
[[ -f "$config" ]] || {
    echo "no such config: $config" >&2
    exit 2
}

build=1
args=()
for a in "$@"; do
    if [[ "$a" == "--no-build" ]]; then build=0; else args+=("$a"); fi
done

java="${JAVA_HOME:+$JAVA_HOME/bin/}java"
if [[ $build -eq 1 ]]; then
    (cd "$root" && mvn -B -q -pl coreset-experiments -am package -DskipTests)
fi
cp_file="$here/target/runtime-classpath.txt"
[[ -f "$cp_file" ]] || {
    echo "missing $cp_file: run without --no-build first" >&2
    exit 2
}
classpath="$here/target/classes:$(cat "$cp_file")"
# Spark's module opens for Java 17, recorded at build time; the runner passes them on to every cell JVM.
opens="$(sed -n 's/^spark.jvm.args=//p' "$here/target/classes/coreset-experiments.properties")"

echo "config: $config"
echo "host: $(hostname), $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m), load: $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
cd "$root"
# shellcheck disable=SC2086
exec "$java" $opens -Xmx8g -cp "$classpath" io.github.c0rwin.coreset.experiments.run.ExperimentRunner \
    --config "$config" "${args[@]+"${args[@]}"}"
