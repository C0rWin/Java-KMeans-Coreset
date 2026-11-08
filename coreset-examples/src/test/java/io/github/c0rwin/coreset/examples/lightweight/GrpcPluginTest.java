package io.github.c0rwin.coreset.examples.lightweight;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.CoresetNodeGrpcService;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/** Three nodes behind in-process gRPC and a coordinator, each node resolving algorithms from the class path. */
class GrpcPluginTest {

    private static final Duration WAIT = Duration.ofSeconds(60);
    private static final int NODES = 3;

    @Test
    void nodesAndCoordinatorRunTheRegisteredAlgorithmUnchanged() throws Exception {
        List<WeightedPoint> points = Blobs.points(30_000, 8);
        Coreset full = Coreset.of(points);
        List<CoresetNode> nodes = new ArrayList<>();
        List<CoresetNodeGrpcService> services = new ArrayList<>();
        List<Server> servers = new ArrayList<>();
        List<ManagedChannel> channels = new ArrayList<>();
        try {
            for (int i = 0; i < NODES; i++) {
                CoresetNode node = new CoresetNode(CoresetAlgorithms.load(), MetricsSink.NOOP);
                CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, () -> "{}",
                        CoresetNodeGrpcService.Config.defaults());
                String name = InProcessServerBuilder.generateName();
                servers.add(InProcessServerBuilder.forName(name).addService(service).build().start());
                channels.add(InProcessChannelBuilder.forName(name).build());
                nodes.add(node);
                services.add(service);
            }
            try (CoresetCoordinator coordinator = new CoresetCoordinator(channels, CoordinatorOptions.defaults())) {
                coordinator.createSession("plugin", new SessionSpec("lightweight", Map.of("m", "400"), 17L, 2, 16));
                try (IngestStream stream = coordinator.openIngest("plugin", "producer")) {
                    for (int from = 0; from < points.size(); from += 1_000) {
                        stream.submit(points.subList(from, from + 1_000));
                    }
                    stream.flush(WAIT);
                }
                GatheredSnapshot g = coordinator.gather("plugin", WAIT);
                assertTrue(g.complete(), "applied " + g.appliedBatches() + " of " + g.acceptedBatches());
                assertEquals(30, g.acceptedBatches());

                Coreset union = g.coreset();
                double distortion = Blobs.distortion(full, union);
                double weightErr = Math.abs(union.totalWeight() - full.totalWeight()) / full.totalWeight();
                CoresetAlgorithm finalCompress = CoresetAlgorithms.load().create("lightweight", Map.of("m", "400"));
                KMeansResult solved = CoresetCoordinator.cluster(union, Blobs.K, 50, 5, 3L, finalCompress);
                double solvedCost = Cost.cost(full, solved.centers());
                double reference = Cost.cost(full, CoresetCoordinator.cluster(full, Blobs.K, 50, 5, 3L, null).centers());
                System.out.printf("lightweight over %d gRPC nodes: union %d points, distortion %.4f, weight rel err"
                        + " %.2e, k-means cost ratio vs full-data solve %.4f%n", NODES, union.size(), distortion,
                        weightErr, solvedCost / reference);
                assertTrue(distortion < LightweightCoresetTest.MAX_STREAM_DISTORTION, "distortion " + distortion);
                assertTrue(weightErr < LightweightCoresetTest.MAX_WEIGHT_REL_ERR, "weight rel err " + weightErr);
                assertTrue(solvedCost / reference < 1.05, "cost ratio " + solvedCost / reference);
            }
        } finally {
            for (ManagedChannel ch : channels) {
                ch.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            }
            for (Server s : servers) {
                s.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            }
            services.forEach(CoresetNodeGrpcService::close);
            nodes.forEach(CoresetNode::close);
        }
    }
}
