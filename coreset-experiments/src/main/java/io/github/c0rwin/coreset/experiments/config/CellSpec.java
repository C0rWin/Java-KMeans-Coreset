package io.github.c0rwin.coreset.experiments.config;

import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.AlgorithmConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.ClusterConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.LossyConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.MemoryConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.SolverConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.SystemConfig;
import java.util.List;

/**
 * Everything one cell JVM needs: one system at one parallelism on one dataset with one algorithm ({@code null} for
 * the native baseline), undisturbed or with one fault. Written as JSON by the orchestrator.
 */
public record CellSpec(String id, DatasetConfig dataset, String cacheDir, SystemConfig system, int parallelism,
                       AlgorithmConfig algorithm, FaultConfig fault, int warmup, int repetitions, int k,
                       SolverConfig solver, MemoryConfig memory, List<String> jvmFlags, LossyConfig lossy,
                       ClusterConfig cluster, long seed, int batchSize, int sampleMillis, String outDir) {

    public CellSpec {
        jvmFlags = List.copyOf(jvmFlags);
    }

    public String algorithmLabel() {
        return algorithm == null ? "native" : algorithm.label();
    }
}
