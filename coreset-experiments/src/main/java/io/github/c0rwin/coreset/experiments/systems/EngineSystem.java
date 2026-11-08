package io.github.c0rwin.coreset.experiments.systems;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.experiments.config.CellSpec;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** One system under test inside a cell JVM: started once, then run once per repetition. */
public interface EngineSystem extends AutoCloseable {

    /** Brings the engine up (processes, contexts, clusters) and waits until it can take work. */
    void start() throws Exception;

    /**
     * One repetition over the whole source.
     *
     * @param seed  algorithm seed of this repetition
     * @param fault failure to inject, or null; fault cells always run undisturbed repetitions first
     */
    RunOutput run(int rep, long seed, FaultConfig fault) throws Exception;

    /** Roles of this system's processes by pid, for {@code ProcessSampler}; this JVM is labelled by {@link #selfRole}. */
    Map<Long, String> roles();

    String selfRole();

    @Override
    void close();

    /**
     * Result of one repetition: a coreset (the cell solves k-means on it) or, for native baselines, centers.
     *
     * @param buildSeconds  ingest and construction: data read, partial coresets, merges inside the engine
     * @param gatherSeconds collecting the final coreset to the driver when that is a separate step, else NaN
     * @param solveSeconds  native solve time (baselines), else NaN
     * @param metrics       system-specific measurements (bytes, task counts), merged into the record
     * @param fault         what the injected fault did (times, lost weight, recovery), empty if none
     */
    record RunOutput(Coreset coreset, List<SparseVector> centers, double buildSeconds, double gatherSeconds,
                     double solveSeconds, Map<String, Object> metrics, Map<String, Object> fault) {
    }

    /** Creates the system named by the cell. */
    static EngineSystem create(CellSpec cell, PointSource source) {
        return switch (cell.system().type()) {
            case "grpc" -> new GrpcSystem(cell, source);
            case "spark", "spark-mllib" -> new SparkSystem(cell, source);
            case "flink" -> new FlinkSystem(cell, source);
            default -> throw new IllegalArgumentException("unknown system " + cell.system().type());
        };
    }

    static Map<String, Object> map() {
        return new LinkedHashMap<>();
    }
}
