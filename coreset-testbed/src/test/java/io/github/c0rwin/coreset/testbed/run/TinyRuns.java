package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A small grid touching every dataset kind, algorithm and mode, executed in-process. */
final class TinyRuns {

    private TinyRuns() {
    }

    static TestbedConfig config(int replicates) {
        return new TestbedConfig("tiny", "kmeans",
                List.of(new DatasetSpec("gauss", "gaussian-mixture",
                                Map.of("n", "3000", "d", "4", "clusters", "5", "sigma", "1", "imbalance", "1")),
                        new DatasetSpec("esa", "esa22", Map.of("k", "3", "alpha", "5")),
                        new DatasetSpec("bow", "sparse-bow",
                                Map.of("n", "800", "d", "600", "topics", "4", "docLength", "12"))),
                List.of(3, 4), List.of(60),
                List.of(new AlgorithmSpec("uniform"),
                        new AlgorithmSpec("sens-b", "sensitivity", Map.of("seeding", "bicriteria"), null, null),
                        new AlgorithmSpec("sensitivity"),
                        new AlgorithmSpec(null, "mean", Map.of("lloydIterations", "3"), null, null)),
                List.of(ModeSpec.offline(), ModeSpec.stream(8), ModeSpec.distributed(2, "node", 8),
                        ModeSpec.distributed(2, "grpc", 8)),
                replicates, 7L,
                new EvaluationSpec(2, 1, 10, 3, 2, null, 1.5, 10, true, 2),
                List.of(new TestbedConfig.Exclude("bow", null, "distributed-grpc-n2-b8", null, null, null)));
    }

    static final class Collector implements Testbed.Sink {
        final List<ValidityReport> reports = new ArrayList<>();
        final List<Testbed.PoolInfo> pools = new ArrayList<>();

        @Override
        public void report(ValidityReport report) {
            reports.add(report);
        }

        @Override
        public void pool(Testbed.PoolInfo pool) {
            pools.add(pool);
        }
    }

    static Collector run(TestbedConfig config) throws Exception {
        Collector c = new Collector();
        try (Testbed t = new Testbed(config, new Datasets(Path.of("unused-cache"), false),
                CoresetAlgorithms.load(), Provenance.collect(Path.of("").toAbsolutePath()), s -> { })) {
            t.run(c);
        }
        return c;
    }
}
