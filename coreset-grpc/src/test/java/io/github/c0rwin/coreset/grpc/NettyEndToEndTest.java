package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.cli.NodeCommand;
import io.github.c0rwin.coreset.grpc.proto.StatsResponse;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.github.c0rwin.coreset.node.testing.TestResolver;
import io.grpc.ManagedChannel;
import io.grpc.health.v1.HealthCheckRequest;
import io.grpc.health.v1.HealthCheckResponse;
import io.grpc.health.v1.HealthGrpc;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import picocli.CommandLine;

/** Three real nodes on ephemeral ports (started through the node CLI wiring) driven by the coordinator over Netty. */
class NettyEndToEndTest {

    private static final Duration WAIT = Duration.ofSeconds(60);

    /** Points around three well-separated integer centers, so k=3 clustering is unambiguous. */
    private static List<WeightedPoint> blobs(SplittableRandom rng, int n) {
        double[][] centers = {{0, 0, 0}, {100, 0, 0}, {0, 100, 50}};
        List<WeightedPoint> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double[] c = centers[i % 3];
            double[] v = new double[3];
            for (int d = 0; d < 3; d++) {
                v[d] = c[d] + rng.nextInt(5) - 2;
            }
            out.add(new WeightedPoint(SparseVector.dense(v), 1 + rng.nextInt(3)));
        }
        return out;
    }

    @Test
    void threeNodesThroughTheCoordinator() throws Exception {
        List<NodeCommand.Running> running = new ArrayList<>();
        RecordingMetrics clientMetrics = new RecordingMetrics();
        try {
            for (int i = 0; i < 3; i++) {
                NodeCommand cmd = new NodeCommand();
                new CommandLine(cmd).parseArgs("--port", "0", "--metrics-port", "0", "--shutdown-grace-ms", "500");
                running.add(cmd.start(new TestResolver()));
            }
            List<String> targets = running.stream().map(r -> "localhost:" + r.port()).toList();
            List<WeightedPoint> all = blobs(new SplittableRandom(11), 30_000);
            double weight = all.stream().mapToDouble(WeightedPoint::weight).sum();
            try (CoresetCoordinator coordinator = CoresetCoordinator.connect(targets, CoordinatorOptions.defaults(),
                    clientMetrics)) {
                // merge-equal keeps every node's coreset small (few distinct points) and the cost exact.
                coordinator.createSession("e2e", new SessionSpec("merge-equal", Map.of(), 7L, 3, 16, TreeMode.CAS));
                try (IngestStream stream = coordinator.openIngest("e2e")) {
                    for (int from = 0; from < all.size(); from += 100) {
                        stream.submit(all.subList(from, from + 100));
                    }
                    stream.flush(WAIT);
                    assertEquals(300, stream.acceptedBatches());
                }
                GatheredSnapshot snap = coordinator.gather("e2e", WAIT);
                assertTrue(snap.complete());
                assertEquals(300, snap.appliedBatches());
                assertEquals(weight, snap.coreset().totalWeight());

                KMeansResult km = CoresetCoordinator.cluster(snap.coreset(), 3, 50, 3L, null);
                double rawCost = Cost.cost(all, km.centers());
                double distributed = coordinator.cost("e2e", km.centers());
                assertEquals(rawCost, distributed, 1e-9 * rawCost);
                assertEquals(rawCost, coordinator.cost(km.centers(), all, 5_000), 1e-9 * rawCost);
                // Every point is within distance sqrt(12) of its blob center, so a correct clustering is cheap.
                assertTrue(rawCost <= 12.0 * weight, "cost " + rawCost);

                List<StatsResponse> stats = coordinator.stats();
                for (StatsResponse s : stats) {
                    assertTrue(s.getMetricsJson().contains("\"node.ingest.accepted\":100"), s.getMetricsJson());
                    assertTrue(s.getMetricsJson().contains("grpc.server.Ingest.bytes_in"));
                }
                coordinator.closeSession("e2e");
            }
            assertTrue(clientMetrics.snapshot().counters().get("grpc.client.Ingest.bytes_out") > 0);

            ManagedChannel ch = NettyChannelBuilder.forTarget(targets.get(0)).usePlaintext().build();
            try {
                assertEquals(HealthCheckResponse.ServingStatus.SERVING, HealthGrpc.newBlockingStub(ch)
                        .withDeadlineAfter(10, TimeUnit.SECONDS)
                        .check(HealthCheckRequest.newBuilder().setService("coreset.v1.CoresetNodeService").build())
                        .getStatus());
            } finally {
                ch.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            }

            HttpResponse<String> scrape = HttpClient.newHttpClient().send(HttpRequest.newBuilder(
                    URI.create("http://localhost:" + running.get(0).metricsPort() + "/metrics")).build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(200, scrape.statusCode());
            assertTrue(scrape.body().contains("node_ingest_accepted_total 100.0"), scrape.body());
            assertTrue(scrape.body().contains("grpc_server_Ingest_bytes_in_total"));
        } finally {
            for (NodeCommand.Running r : running) {
                r.close();
            }
        }
    }
}
