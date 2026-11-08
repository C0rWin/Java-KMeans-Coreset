package io.github.c0rwin.coreset.examples.lightweight;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
import io.github.c0rwin.coreset.flink.CoresetConfig;
import io.github.c0rwin.coreset.flink.CoresetSnapshot;
import io.github.c0rwin.coreset.flink.FlinkCoresets;
import io.github.c0rwin.coreset.grpc.ClientSecurity;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.CoresetNodeServer;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.grpc.NodeServerOptions;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.spark.BuildResult;
import io.github.c0rwin.coreset.spark.SparkCoresets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.spark.api.java.JavaRDD;

/** The code of docs/api.md: each tagged region is a block of that page, checked by ApiDocTest. */
final class ApiDocSnippets {

    private ApiDocSnippets() {
    }

    static KMeansResult embed(List<WeightedPoint> points) {
        // tag::embed[]
        CoresetAlgorithm algorithm = CoresetAlgorithms.load().create("sensitivity", Map.of("k", "8", "m", "400"));
        Coreset coreset = algorithm.compress(Coreset.of(points), new SplittableRandom(7));
        KMeansResult solution = WeightedLloyd.bestOf(coreset.pointSet(), 8, 50, 5, new SplittableRandom(8),
                MetricsSink.NOOP);
        // end::embed[]
        return solution;
    }

    static Coreset stream(List<List<WeightedPoint>> batches) {
        // tag::stream[]
        CoresetAlgorithm algorithm = CoresetAlgorithms.load().create("sensitivity", Map.of("k", "8", "m", "400"));
        MergeReduceTree tree = new MergeReduceTree(algorithm, new SplittableRandom(7));
        for (List<WeightedPoint> batch : batches) {
            tree.add(Coreset.of(batch));
        }
        Coreset summary = tree.compressedSnapshot();
        // end::stream[]
        return summary;
    }

    static Coreset distributed(List<WeightedPoint> points) throws Exception {
        // tag::distributed[]
        try (CoresetNode node = new CoresetNode(CoresetAlgorithms.load());
             CoresetNodeServer server = CoresetNodeServer.start(NodeServerOptions.builder().port(0).build(), node,
                     new RecordingMetrics(), MetricsSink.NOOP);
             CoresetCoordinator coordinator = CoresetCoordinator.connect(List.of("127.0.0.1:" + server.port()),
                     CoordinatorOptions.defaults(), ClientSecurity.plaintext(), MetricsSink.NOOP)) {
            coordinator.createSession("demo", new SessionSpec("lightweight", Map.of("m", "400"), 7L, 2, 16));
            try (IngestStream ingest = coordinator.openIngest("demo")) {
                for (int from = 0; from < points.size(); from += 1_000) {
                    ingest.submit(points.subList(from, Math.min(points.size(), from + 1_000)));
                }
                ingest.flush(Duration.ofMinutes(1));
            }
            GatheredSnapshot gathered = coordinator.gather("demo", Duration.ofMinutes(1));
            return gathered.coreset();
        }
        // end::distributed[]
    }

    static BuildResult spark(JavaRDD<WeightedPoint> points) {
        // tag::spark[]
        BuildResult byId = SparkCoresets.build(points, "lightweight", Map.of("m", "400"), 7L, 2);
        BuildResult byFactory = SparkCoresets.build(points, new LightweightCoresetFactory(400), 7L, 2, 800);
        // end::spark[]
        return byFactory.coreset().equals(byId.coreset()) ? byId : byFactory;
    }

    static DataStream<CoresetSnapshot> flink(DataStream<WeightedPoint> points) {
        // tag::flink[]
        CoresetConfig byId = CoresetConfig.builder("lightweight", Map.of("m", "400"), 7L).build();
        CoresetConfig byFactory = CoresetConfig.builder(new LightweightCoresetFactory(400), 7L).chunkSize(800).build();
        DataStream<CoresetSnapshot> snapshots = FlinkCoresets.stream(points, byId);
        // end::flink[]
        return byFactory.fingerprint() == byId.fingerprint() ? null : snapshots;
    }

    static double[] evaluate(Coreset full, Coreset coreset) {
        // tag::evaluate[]
        double distortion = Distortion.evaluate(full, coreset,
                CandidateSolutions.standard(full, coreset, 8, 10, 3, new SplittableRandom(99))).max();
        double weightRelErr = IntegrityCheck.check(full, coreset).weightRelErr();
        // end::evaluate[]
        return new double[] {distortion, weightRelErr};
    }
}
