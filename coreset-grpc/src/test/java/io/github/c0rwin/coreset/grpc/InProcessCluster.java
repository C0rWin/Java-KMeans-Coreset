package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.testing.TestProvider;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.health.v1.HealthCheckResponse.ServingStatus;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.protobuf.services.HealthStatusManager;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** N in-process nodes, each with its own {@link CoresetNode}, recorder, service and channel. */
final class InProcessCluster implements AutoCloseable {

    final List<CoresetNode> nodes = new ArrayList<>();
    final List<RecordingMetrics> metrics = new ArrayList<>();
    final List<ManagedChannel> channels = new ArrayList<>();
    private final List<Server> servers = new ArrayList<>();
    private final List<CoresetNodeGrpcService> services = new ArrayList<>();

    InProcessCluster(int n, Map<String, CoresetAlgorithm> extra, int ingestWindow, ServerInterceptor... faults)
            throws IOException {
        for (int i = 0; i < n; i++) {
            RecordingMetrics m = new RecordingMetrics();
            CoresetNode node = new CoresetNode(TestProvider.registry(extra), m, Duration.ofMillis(200));
            CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, () -> m.snapshot().toJson(),
                    CoresetNodeGrpcService.Config.defaults().withIngest(ingestWindow, Duration.ofMillis(1)));
            HealthStatusManager health = new HealthStatusManager();
            health.setStatus(HealthStatusManager.SERVICE_NAME_ALL_SERVICES, ServingStatus.SERVING);
            String name = InProcessServerBuilder.generateName();
            List<ServerInterceptor> interceptors = new ArrayList<>(List.of(faults));
            interceptors.add(GrpcMetrics.server(m)); // last added runs first
            servers.add(InProcessServerBuilder.forName(name)
                    .addService(ServerInterceptors.intercept(service, interceptors))
                    .addService(health.getHealthService())
                    .build()
                    .start());
            channels.add(InProcessChannelBuilder.forName(name).build());
            nodes.add(node);
            metrics.add(m);
            services.add(service);
        }
    }

    @Override
    public void close() throws InterruptedException {
        for (ManagedChannel ch : channels) {
            ch.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        for (Server s : servers) {
            s.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        for (CoresetNodeGrpcService s : services) {
            s.close();
        }
        for (CoresetNode node : nodes) {
            node.close();
        }
    }
}
