package io.github.c0rwin.coreset.experiments.config;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One experiment grid, read from YAML or JSON: datasets x systems x parallelism x algorithms, each cell once undisturbed
 * and once per applicable fault. Unknown keys are rejected, so a typo never silently falls back to a default.
 */
public record ExperimentConfig(String name, String output, String cacheDir, Integer warmup, Integer repetitions,
                               Long seed, Integer k, Integer batchSize, Integer sampleMillis, Long cellTimeoutSeconds,
                               List<DatasetConfig> datasets, List<AlgorithmConfig> algorithms,
                               List<SystemConfig> systems, List<Integer> parallelism, SolverConfig solver,
                               QualityConfig quality, MemoryConfig memory, List<String> jvmFlags,
                               List<FaultConfig> faults, LossyConfig lossy, ClusterConfig cluster) {

    public ExperimentConfig {
        Objects.requireNonNull(name, "name");
        output = output != null ? output : "results/engines/" + name;
        cacheDir = cacheDir != null ? cacheDir : Path.of(System.getProperty("user.home"), ".cache",
                "coreset-experiments").toString();
        warmup = warmup != null ? warmup : 1;
        repetitions = repetitions != null ? repetitions : 3;
        seed = seed != null ? seed : 1L;
        Objects.requireNonNull(k, "k");
        batchSize = batchSize != null ? batchSize : 4096;
        sampleMillis = sampleMillis != null ? sampleMillis : 200;
        cellTimeoutSeconds = cellTimeoutSeconds != null ? cellTimeoutSeconds : 3600L;
        datasets = List.copyOf(Objects.requireNonNull(datasets, "datasets"));
        algorithms = algorithms != null ? List.copyOf(algorithms) : List.of();
        systems = List.copyOf(Objects.requireNonNull(systems, "systems"));
        parallelism = List.copyOf(Objects.requireNonNull(parallelism, "parallelism"));
        solver = solver != null ? solver : new SolverConfig(null, null);
        quality = quality != null ? quality : new QualityConfig(null, null, null, null, null, null, null, null);
        memory = memory != null ? memory : new MemoryConfig(null, null);
        jvmFlags = jvmFlags != null ? List.copyOf(jvmFlags) : List.of();
        faults = faults != null ? List.copyOf(faults) : List.of();
        lossy = lossy != null ? lossy : LossyConfig.OFF;
        cluster = cluster != null ? cluster : ClusterConfig.LOCAL;
        if (warmup < 0 || repetitions < 1 || k < 1 || batchSize < 1 || sampleMillis < 10) {
            throw new IllegalArgumentException("need warmup >= 0, repetitions >= 1, k >= 1, batchSize >= 1, "
                    + "sampleMillis >= 10");
        }
        if (parallelism.isEmpty() || parallelism.stream().anyMatch(n -> n < 1)) {
            throw new IllegalArgumentException("parallelism needs at least one value >= 1");
        }
    }

    public static final ObjectMapper JSON = new ObjectMapper()
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    private static final ObjectMapper YAML = new ObjectMapper(new YAMLFactory())
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);

    /** Reads {@code .yaml}/{@code .yml} as YAML, anything else as JSON. */
    public static ExperimentConfig read(Path file) throws IOException {
        String f = file.getFileName().toString();
        ObjectMapper m = f.endsWith(".yaml") || f.endsWith(".yml") ? YAML : JSON;
        return m.readValue(file.toFile(), ExperimentConfig.class);
    }

    /** A dataset: {@code type} plus string parameters, resolved by {@code PointSources}. */
    public record DatasetConfig(String id, String type, Map<String, String> params) {
        public DatasetConfig {
            Objects.requireNonNull(id, "dataset id");
            Objects.requireNonNull(type, "dataset type");
            params = params != null ? Map.copyOf(params) : Map.of();
        }
    }

    /** A coreset algorithm by SPI id; {@code label} defaults to the id. */
    public record AlgorithmConfig(String id, Map<String, String> params, String label) {
        public AlgorithmConfig {
            Objects.requireNonNull(id, "algorithm id");
            params = params != null ? Map.copyOf(params) : Map.of();
            label = label != null ? label : id;
        }
    }

    /**
     * A system under test. {@code type}: {@code grpc}, {@code spark}, {@code spark-mllib} (native k-means|| baseline, no
     * coreset) or {@code flink}. {@code mode}: grpc {@code spawn}|{@code external}; spark {@code local}|
     * {@code local-cluster}|{@code cluster}; flink {@code minicluster}|{@code remote}. {@code options} are
     * system-specific knobs (see docs/experiments/engines.md).
     */
    public record SystemConfig(String type, String mode, Map<String, String> options) {
        public SystemConfig {
            Objects.requireNonNull(type, "system type");
            mode = mode != null ? mode : switch (type) {
                case "grpc" -> "spawn";
                case "spark", "spark-mllib" -> "local";
                case "flink" -> "minicluster";
                default -> throw new IllegalArgumentException("unknown system type " + type
                        + ", expected grpc, spark, spark-mllib or flink");
            };
            options = options != null ? Map.copyOf(options) : Map.of();
        }

        public String label() {
            return type + "/" + mode;
        }

        public boolean usesCoreset() {
            return !type.equals("spark-mllib");
        }

        public int intOption(String key, int dflt) {
            String v = options.get(key);
            return v == null ? dflt : Integer.parseInt(v.trim());
        }

        public long longOption(String key, long dflt) {
            String v = options.get(key);
            return v == null ? dflt : Long.parseLong(v.trim());
        }

        public boolean boolOption(String key, boolean dflt) {
            String v = options.get(key);
            return v == null ? dflt : Boolean.parseBoolean(v.trim());
        }

        public String option(String key, String dflt) {
            return options.getOrDefault(key, dflt);
        }
    }

    /** k-means on the final coreset: best of {@code restarts} k-means++ plus weighted Lloyd runs. */
    public record SolverConfig(Integer restarts, Integer iterations) {
        public SolverConfig {
            restarts = restarts != null ? restarts : 10;
            iterations = iterations != null ? iterations : 50;
        }
    }

    /**
     * Quality evaluation in the orchestrator, after each cell: the reference is the cheapest of {@code referenceRuns}
     * best-of-{@code referenceRestarts} solutions on the full data (cached per dataset and k); distortion uses
     * {@code candidates} solutions per generator. Above {@code maxInMemoryPoints} rows, candidates and the reference are
     * computed on a uniform sample of that many rows, while every cost is still an exact pass over all rows.
     */
    public record QualityConfig(Integer referenceRuns, Integer referenceRestarts, Integer referenceIterations,
                                Integer candidates, Integer candidateIterations, Long maxInMemoryPoints,
                                Integer evalThreads, Boolean distortion) {
        public QualityConfig {
            referenceRuns = referenceRuns != null ? referenceRuns : 3;
            referenceRestarts = referenceRestarts != null ? referenceRestarts : 10;
            referenceIterations = referenceIterations != null ? referenceIterations : 50;
            candidates = candidates != null ? candidates : 2;
            candidateIterations = candidateIterations != null ? candidateIterations : 20;
            maxInMemoryPoints = maxInMemoryPoints != null ? maxInMemoryPoints : 2_000_000L;
            evalThreads = evalThreads != null ? evalThreads : 4;
            distortion = distortion != null ? distortion : true;
        }
    }

    /**
     * Heap budget of the workers of every system: grpc nodes get {@code workersMB / N} each, Spark local-cluster
     * executors the same, Spark local and Flink MiniCluster (one JVM) all of it. Coordinator and driver JVMs of the
     * multi-process modes get {@code driverMB} on top.
     */
    public record MemoryConfig(Integer workersMB, Integer driverMB) {
        public MemoryConfig {
            workersMB = workersMB != null ? workersMB : 8192;
            driverMB = driverMB != null ? driverMB : 2048;
        }
    }

    /**
     * An injected failure. Kinds: {@code kill-nodes} (grpc; SIGKILL {@code count} node processes when a fraction
     * {@code at} of the points was submitted; with {@code snapshotAt} every node is snapshotted at that fraction first and
     * the victims' snapshots are kept), {@code kill-executor} (spark local-cluster; SIGKILL a busy executor
     * {@code at} x the undisturbed mean task time after the first task started), {@code task-failure} (flink; one
     * subtask throws at fraction {@code at} of its rows, after a checkpoint completed). {@code afterKill}: {@code drop}
     * loses the victims' remaining shards, {@code reroute} sends them to survivors. {@code reweight} scales the
     * survivors' coreset to the full input weight. {@code parallelism} and {@code algorithms} (labels) restrict the
     * cells the fault is applied to; empty means all.
     */
    public record FaultConfig(String system, String kind, Double at, Double snapshotAt, Integer count,
                              String afterKill, Boolean reweight, Integer repetitions, Integer baselineRepetitions,
                              List<Integer> parallelism, List<String> algorithms, String label) {
        public FaultConfig {
            Objects.requireNonNull(system, "fault system");
            Objects.requireNonNull(kind, "fault kind");
            at = at != null ? at : 0.5;
            count = count != null ? count : 1;
            afterKill = afterKill != null ? afterKill : "drop";
            reweight = reweight != null ? reweight : false;
            baselineRepetitions = baselineRepetitions != null ? baselineRepetitions : 1;
            parallelism = parallelism != null ? List.copyOf(parallelism) : List.of();
            algorithms = algorithms != null ? List.copyOf(algorithms) : List.of();
            if (!(at > 0.0 && at < 1.0) || (snapshotAt != null && !(snapshotAt > 0.0 && snapshotAt <= at))) {
                throw new IllegalArgumentException("fault needs 0 < snapshotAt <= at < 1");
            }
            if (!afterKill.equals("drop") && !afterKill.equals("reroute")) {
                throw new IllegalArgumentException("afterKill must be drop or reroute: " + afterKill);
            }
            if (baselineRepetitions < 1) {
                throw new IllegalArgumentException("baselineRepetitions must be >= 1");
            }
            label = label != null ? label : kind + "@" + at + (snapshotAt != null ? "/snap@" + snapshotAt : "")
                    + (kind.equals("kill-nodes") ? "/x" + count + "/" + afterKill + (reweight ? "/reweight" : "")
                    : "");
        }

        /**
         * Matches a system by type ({@code spark}) or label ({@code spark/local-cluster}) where the kind can run, within
         * the parallelism and algorithm filters ({@code algorithm} is null for native baselines).
         */
        public boolean appliesTo(SystemConfig s, int n, AlgorithmConfig algorithm) {
            if (!system.equals(s.type()) && !system.equals(s.label())) {
                return false;
            }
            if (!parallelism.isEmpty() && !parallelism.contains(n)) {
                return false;
            }
            if (!algorithms.isEmpty() && (algorithm == null || !algorithms.contains(algorithm.label()))) {
                return false;
            }
            return switch (kind) {
                case "kill-nodes" -> s.type().equals("grpc") && n > count;
                case "kill-executor" -> s.type().equals("spark") && s.mode().equals("local-cluster");
                case "task-failure" -> s.type().equals("flink") && s.mode().equals("minicluster");
                default -> throw new IllegalArgumentException("unknown fault kind " + kind
                        + ", expected kill-nodes, kill-executor or task-failure");
            };
        }
    }

    /**
     * Lossy-transport hook for the grpc system: client-side interceptors drop outgoing messages of {@code methods} per
     * a channel model. Off by default; the current ingest protocol cannot recover a dropped batch, so a lossy run is
     * expected to stall and is reported as failed until loss-resilient ingest exists.
     */
    public record LossyConfig(String model, Double p, Double pGoodToBad, Double pBadToGood, Double lossGood,
                              Double lossBad, List<String> methods, Long seed) {
        public static final LossyConfig OFF = new LossyConfig("none", null, null, null, null, null, null, null);

        public LossyConfig {
            model = model != null ? model : "none";
            methods = methods != null ? List.copyOf(methods) : List.of("Ingest");
            seed = seed != null ? seed : 1L;
        }

        public boolean enabled() {
            return !model.equals("none");
        }
    }

    /**
     * Targets for a real cluster; empty lists mean "spawn locally". The dataset path must resolve on every worker.
     * {@code grpcKillCommand} is a shell template with {@code {host}} and {@code {port}} used to kill remote nodes.
     */
    public record ClusterConfig(List<String> grpcTargets, List<String> grpcMetrics, String grpcKillCommand,
                                String sparkMaster, List<String> sparkJars, String flinkJobManager,
                                List<String> flinkJars) {
        public static final ClusterConfig LOCAL = new ClusterConfig(null, null, null, null, null, null, null);

        public ClusterConfig {
            grpcTargets = grpcTargets != null ? List.copyOf(grpcTargets) : List.of();
            grpcMetrics = grpcMetrics != null ? List.copyOf(grpcMetrics) : List.of();
            sparkJars = sparkJars != null ? List.copyOf(sparkJars) : List.of();
            flinkJars = flinkJars != null ? List.copyOf(flinkJars) : List.of();
        }
    }
}
