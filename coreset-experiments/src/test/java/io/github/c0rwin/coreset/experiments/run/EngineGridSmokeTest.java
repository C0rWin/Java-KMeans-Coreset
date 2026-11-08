package io.github.c0rwin.coreset.experiments.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Runs a tiny grid through the real harness: every system and every fault kind, each cell in its own JVM with real
 * node processes and Spark executors, then checks the records the paper tables are built from.
 */
class EngineGridSmokeTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    @TempDir
    Path dir;

    @Test
    @SuppressWarnings("unchecked")
    void everySystemAndFaultProducesSoundRecords() throws Exception {
        String yaml = """
                name: smoke-test
                warmup: 1
                repetitions: 1
                k: 4
                batchSize: 2000
                sampleMillis: 50
                cellTimeoutSeconds: 300
                datasets:
                  - id: gm
                    type: gaussian-mixture
                    params: {n: 200000, d: 8, clusters: 4, sigma: 2, imbalance: 1, seed: 3}
                algorithms:
                  - {id: uniform, params: {m: 300}}
                systems:
                  - {type: grpc}
                  - {type: spark}
                  - {type: spark-mllib, options: {tol: "1e-3"}}
                  - {type: spark, mode: local-cluster}
                  - {type: flink}
                parallelism: [2]
                solver: {restarts: 2, iterations: 10}
                quality: {referenceRuns: 1, referenceRestarts: 2, referenceIterations: 20, candidates: 1,
                          candidateIterations: 5, evalThreads: 2}
                memory: {workersMB: 1024, driverMB: 512}
                faults:
                  - {system: grpc, kind: kill-nodes, at: 0.5}
                  - {system: spark/local-cluster, kind: kill-executor, at: 0.5}
                  - {system: flink, kind: task-failure, at: 0.5}
                """;
        Path config = dir.resolve("grid.yaml");
        Files.writeString(config, yaml + "output: " + dir.resolve("out") + "\ncacheDir: " + dir.resolve("cache") + "\n");
        ExperimentConfig cfg = ExperimentConfig.read(config);
        int failed = new ExperimentRunner(cfg, dir.resolve("out"), null, false).run();

        List<Map<String, Object>> rows = new ArrayList<>();
        for (String line : Files.readAllLines(dir.resolve("out/results.jsonl"), StandardCharsets.UTF_8)) {
            rows.add(JSON.readValue(line, Map.class));
        }
        List<Map<String, Object>> errors = rows.stream().filter(r -> "error".equals(r.get("phase"))).toList();
        assertTrue(errors.isEmpty(), "errors: " + errors);
        assertEquals(0, failed);

        for (String system : List.of("grpc/spawn", "spark/local", "spark-mllib/local", "spark/local-cluster",
                "flink/minicluster")) {
            Map<String, Object> r = one(rows, x -> system.equals(x.get("system")) && "measure".equals(x.get("phase")));
            double ratio = ((Number) r.get("cost_ratio")).doubleValue();
            assertTrue(ratio > 0.9 && ratio < 1.5, system + " cost ratio " + ratio);
            assertTrue(((Number) r.get("startup_s")).doubleValue() > 0, system);
            assertTrue(((Number) r.get("cpu_s")).doubleValue() > 0, system);
            Map<String, Object> m = (Map<String, Object>) r.get("metrics");
            assertTrue(((Number) m.get("bytes_summary")).doubleValue() > 0, system + " bytes " + m);
            if (!system.startsWith("spark-mllib")) {
                assertTrue(((Number) r.get("distortion_max")).doubleValue() < 1.5, system);
                assertTrue(((Number) r.get("weight_error")).doubleValue() < 1e-9, system);
            }
        }
        Map<String, Object> grpc = one(rows, x -> "grpc/spawn".equals(x.get("system")) && "measure".equals(x.get("phase")));
        assertEquals(3, ((List<?>) grpc.get("processes")).size(), "coordinator and two nodes");
        assertTrue(((Number) ((Map<String, Object>) grpc.get("metrics")).get("bytes_ingest")).doubleValue() > 0);
        Map<String, Object> cluster = one(rows,
                x -> "spark/local-cluster".equals(x.get("system")) && "measure".equals(x.get("phase")));
        assertTrue(((List<Map<String, Object>>) cluster.get("processes")).stream()
                .filter(p -> "executor".equals(p.get("role"))).count() >= 2, "executor processes sampled");

        Map<String, Object> nodeKill = fault(rows, "grpc/spawn");
        double lost = ((Number) nodeKill.get("lost_fraction")).doubleValue();
        assertTrue(lost > 0.05 && lost < 0.95, "lost fraction " + lost);
        assertEquals(1, ((Number) nodeKill.get("survivors")).intValue());
        assertNotNull(nodeKill.get("detect_s_node1"));

        Map<String, Object> executorKill = fault(rows, "spark/local-cluster");
        assertNotNull(executorKill.get("victim_pid"), "executor killed: " + executorKill);
        assertTrue(((Number) executorKill.get("tasks_failed")).intValue() >= 1, "lost task: " + executorKill);
        assertEquals(Boolean.TRUE, faultRecord(rows, "spark/local-cluster").get("equal_to_baseline"));

        Map<String, Object> taskFailure = fault(rows, "flink/minicluster");
        assertNotNull(taskFailure.get("restore_s"), "restored: " + taskFailure);
        assertEquals(Boolean.TRUE, faultRecord(rows, "flink/minicluster").get("equal_to_baseline"));

        String summary = Files.readString(dir.resolve("out/summary.md"));
        assertTrue(summary.contains("## Undisturbed runs") && summary.contains("## Fault injection"), summary);
        assertFalse(summary.contains("## Errors"), summary);
    }

    private static Map<String, Object> one(List<Map<String, Object>> rows, Predicate<Map<String, Object>> p) {
        List<Map<String, Object>> m = rows.stream().filter(p).toList();
        assertEquals(1, m.size(), "matching records: " + m);
        return m.get(0);
    }

    private static Map<String, Object> faultRecord(List<Map<String, Object>> rows, String system) {
        return one(rows, x -> system.equals(x.get("system")) && "fault".equals(x.get("phase")));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> fault(List<Map<String, Object>> rows, String system) {
        return (Map<String, Object>) faultRecord(rows, system).get("fault");
    }
}
