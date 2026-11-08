package io.github.c0rwin.coreset.examples.lightweight;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import io.github.c0rwin.coreset.testbed.run.AlgorithmSpec;
import io.github.c0rwin.coreset.testbed.run.EvaluationSpec;
import io.github.c0rwin.coreset.testbed.run.ModeSpec;
import io.github.c0rwin.coreset.testbed.run.Provenance;
import io.github.c0rwin.coreset.testbed.run.Testbed;
import io.github.c0rwin.coreset.testbed.run.TestbedConfig;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** The testbed grid with the plugged-in id next to a built-in, offline, streamed and on node and gRPC clusters. */
class TestbedPluginTest {

    @Test
    void theTestbedEvaluatesTheRegisteredAlgorithmInEveryMode() throws Exception {
        TestbedConfig config = new TestbedConfig("plugin", "kmeans",
                List.of(new DatasetSpec("gauss", "gaussian-mixture",
                        Map.of("n", "6000", "d", "5", "clusters", "6", "sigma", "1", "imbalance", "1"))),
                List.of(6), List.of(400),
                List.of(new AlgorithmSpec("lightweight", "lightweight", Map.of(), List.of("m"), false),
                        new AlgorithmSpec("uniform")),
                List.of(ModeSpec.offline(), ModeSpec.stream(16), ModeSpec.distributed(2, "node", 16),
                        ModeSpec.distributed(2, "grpc", 16)),
                1, 31L, new EvaluationSpec(3, 1, 10, 3, 2, null, 1.5, 10, true, 2), List.of());
        List<ValidityReport> reports = new ArrayList<>();
        Testbed.Sink sink = new Testbed.Sink() {
            @Override
            public void report(ValidityReport report) {
                reports.add(report);
            }

            @Override
            public void pool(Testbed.PoolInfo pool) {
            }
        };
        try (Testbed testbed = new Testbed(config, new Datasets(Path.of("unused-cache"), false),
                CoresetAlgorithms.load(), Provenance.collect(Path.of("").toAbsolutePath()), line -> { })) {
            testbed.run(sink);
        }
        List<ValidityReport> lightweight = reports.stream()
                .filter(r -> r.config().algorithm().id().equals("lightweight")).toList();
        assertEquals(Set.of("offline", "stream-b16", "distributed-node-n2-b16", "distributed-grpc-n2-b16"),
                lightweight.stream().map(r -> r.config().modeLabel()).collect(Collectors.toSet()));
        for (ValidityReport r : lightweight) {
            System.out.printf("testbed lightweight %s: %d points, distortion max %.4f, weight rel err %.2e,"
                            + " replay deterministic %s%n", r.config().modeLabel(), r.construction().outputSize(),
                    r.distortion().max(), r.integrity().weightRelErr(), r.integrity().replayDeterministic());
            assertEquals(Map.of("m", "400"), r.config().algorithm().params());
            assertTrue(r.construction().outputSize() <= 400, r.config().modeLabel());
            assertTrue(r.distortion().max() < LightweightCoresetTest.MAX_STREAM_DISTORTION,
                    r.config().modeLabel() + " distortion " + r.distortion().max());
            assertTrue(r.integrity().weightRelErr() < LightweightCoresetTest.MAX_WEIGHT_REL_ERR,
                    r.config().modeLabel() + " weight rel err " + r.integrity().weightRelErr());
            // Null when the testbed reused an identical build instead of replaying it.
            assertTrue(!Boolean.FALSE.equals(r.integrity().replayDeterministic()), r.config().modeLabel());
            assertEquals(0, r.integrity().negativeWeights());
        }
    }
}
