package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.testing.TestProvider;
import io.grpc.Channel;
import io.grpc.ClientInterceptor;
import io.grpc.ClientInterceptors;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Server;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * In-process nodes that store replicas on each other by server name, with crash and straggler injection: a killed
 * node's server is shut down (callers see UNAVAILABLE), a delayed node holds every new call before serving it.
 */
final class ResilienceCluster implements AutoCloseable {

    final List<String> names = new ArrayList<>();
    final List<CoresetNode> nodes = new ArrayList<>();
    private final List<Server> servers = new ArrayList<>();
    private final List<CoresetNodeGrpcService> services = new ArrayList<>();
    private final List<ManagedChannel> channels = new ArrayList<>();
    private final List<ExecutorService> executors = new ArrayList<>();
    private final ConcurrentHashMap<String, ManagedChannel> peerChannels = new ConcurrentHashMap<>();
    private final AtomicLongArray delayMs;

    ResilienceCluster(int n, Map<String, CoresetAlgorithm> extra) throws IOException {
        delayMs = new AtomicLongArray(n);
        for (int i = 0; i < n; i++) {
            CoresetNode node = new CoresetNode(TestProvider.registry(extra), MetricsSink.NOOP, Duration.ofMillis(200));
            CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, () -> "{}",
                    CoresetNodeGrpcService.Config.defaults(), PeerChannels.of(this::peer));
            String name = InProcessServerBuilder.generateName();
            int index = i;
            ExecutorService executor = Executors.newCachedThreadPool(r -> {
                Thread t = new Thread(r, "node-" + index);
                t.setDaemon(true);
                return t;
            });
            executors.add(executor);
            servers.add(InProcessServerBuilder.forName(name).executor(executor)
                    .addService(ServerInterceptors.intercept(service, delayOf(index))).build().start());
            channels.add(InProcessChannelBuilder.forName(name).build());
            names.add(name);
            nodes.add(node);
            services.add(service);
        }
    }

    private ServerInterceptor delayOf(int index) {
        return new ServerInterceptor() {
            @Override
            public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                              ServerCallHandler<Q, R> next) {
                long ms = delayMs.get(index);
                if (ms > 0) {
                    try {
                        Thread.sleep(ms);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        call.close(Status.CANCELLED, new Metadata());
                        return new ServerCall.Listener<>() {
                        };
                    }
                }
                return next.startCall(call, headers);
            }
        };
    }

    private ManagedChannel peer(String name) {
        return peerChannels.computeIfAbsent(name, t -> InProcessChannelBuilder.forName(t).build());
    }

    /** The coordinator's channels, with {@code interceptors} (e.g. a lossy transport) on each. */
    List<Channel> channels(ClientInterceptor... interceptors) {
        List<Channel> out = new ArrayList<>();
        for (ManagedChannel ch : channels) {
            out.add(ClientInterceptors.intercept(ch, interceptors));
        }
        return out;
    }

    /** Crash of node {@code i}: its server and sessions are gone. */
    void kill(int i) throws InterruptedException {
        servers.get(i).shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        nodes.get(i).close();
    }

    void delay(int i, long ms) {
        delayMs.set(i, ms);
    }

    @Override
    public void close() throws InterruptedException {
        for (int i = 0; i < delayMs.length(); i++) {
            delayMs.set(i, 0);
        }
        for (ManagedChannel ch : channels) {
            ch.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        for (ManagedChannel ch : peerChannels.values()) {
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
        for (ExecutorService e : executors) {
            e.shutdownNow();
        }
    }
}
