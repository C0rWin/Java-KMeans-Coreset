package io.github.c0rwin.coreset.grpc;

import com.google.common.util.concurrent.ListenableFuture;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostResponse;
import io.github.c0rwin.coreset.grpc.proto.PointList;
import io.github.c0rwin.coreset.grpc.proto.SnapshotRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.grpc.proto.StatsRequest;
import io.github.c0rwin.coreset.grpc.proto.StatsResponse;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Client side of a coreset cluster: creates sessions on every node, streams batches to them
 * ({@link #openIngest}), gathers and unions their snapshots, clusters the union and evaluates costs in a distributed
 * way. Snapshots may be gathered at any time, including while an {@link IngestStream} is still sending.
 */
public final class CoresetCoordinator implements AutoCloseable {

    private final List<ManagedChannel> ownedChannels;
    private final List<CoresetNodeServiceGrpc.CoresetNodeServiceStub> async;
    private final List<CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub> futures;
    private final CoordinatorOptions options;
    private final ScheduledExecutorService scheduler;

    /** Uses channels owned by the caller (e.g. in-process channels in tests). */
    public CoresetCoordinator(List<? extends Channel> channels, CoordinatorOptions options) {
        this(channels, List.of(), options);
    }

    private CoresetCoordinator(List<? extends Channel> channels, List<ManagedChannel> owned,
                               CoordinatorOptions options) {
        if (channels.isEmpty()) {
            throw new IllegalArgumentException("at least one node is required");
        }
        this.ownedChannels = List.copyOf(owned);
        this.options = options;
        this.async = new ArrayList<>(channels.size());
        this.futures = new ArrayList<>(channels.size());
        for (Channel ch : channels) {
            async.add(CoresetNodeServiceGrpc.newStub(ch).withMaxInboundMessageSize(options.maxMessageBytes())
                    .withMaxOutboundMessageSize(options.maxMessageBytes()));
            futures.add(CoresetNodeServiceGrpc.newFutureStub(ch).withMaxInboundMessageSize(options.maxMessageBytes())
                    .withMaxOutboundMessageSize(options.maxMessageBytes()));
        }
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "coreset-coordinator-retry");
            t.setDaemon(true);
            return t;
        });
    }

    /** Connects to {@code host:port} targets over plaintext Netty channels owned (and closed) by the coordinator. */
    public static CoresetCoordinator connect(List<String> targets, CoordinatorOptions options, MetricsSink metrics) {
        List<ManagedChannel> channels = new ArrayList<>(targets.size());
        for (String target : targets) {
            channels.add(NettyChannelBuilder.forTarget(target)
                    .usePlaintext()
                    .maxInboundMessageSize(options.maxMessageBytes())
                    .intercept(GrpcMetrics.client(metrics))
                    .build());
        }
        return new CoresetCoordinator(channels, channels, options);
    }

    public int nodes() {
        return async.size();
    }

    /** Creates the session on every node (idempotent per node). */
    public void createSession(String sessionId, SessionSpec spec) throws InterruptedException {
        CreateSessionRequest req = CreateSessionRequest.newBuilder()
                .setSessionId(sessionId)
                .setSpec(ProtoCodec.toProto(spec))
                .build();
        List<ListenableFuture<?>> calls = new ArrayList<>();
        for (CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub stub : futures) {
            calls.add(deadline(stub, options.rpcDeadline()).createSession(req));
        }
        awaitAll(calls);
    }

    public void closeSession(String sessionId) throws InterruptedException {
        CloseSessionRequest req = CloseSessionRequest.newBuilder().setSessionId(sessionId).build();
        List<ListenableFuture<?>> calls = new ArrayList<>();
        for (CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub stub : futures) {
            calls.add(deadline(stub, options.rpcDeadline()).closeSession(req));
        }
        awaitAll(calls);
    }

    public IngestStream openIngest(String sessionId) {
        return new IngestStream(sessionId, async, options, scheduler);
    }

    /**
     * Snapshots the session on every node in parallel and unions the coresets.
     *
     * @param awaitApplied how long each node may wait for its accepted batches to be applied first; zero snapshots
     *                     immediately (continuous mode: check {@link GatheredSnapshot#complete()})
     */
    public GatheredSnapshot gather(String sessionId, Duration awaitApplied) throws InterruptedException {
        SnapshotRequest req = SnapshotRequest.newBuilder()
                .setSessionId(sessionId)
                .setAwaitAppliedMs(awaitApplied.toMillis())
                .build();
        List<ListenableFuture<SnapshotResponse>> calls = new ArrayList<>();
        for (CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub stub : futures) {
            calls.add(deadline(stub, options.rpcDeadline().plus(awaitApplied)).snapshot(req));
        }
        List<NodeSnapshot> snaps = new ArrayList<>(calls.size());
        Coreset union = Coreset.of(List.of());
        for (SnapshotResponse r : awaitAll(calls)) {
            NodeSnapshot s = ProtoCodec.fromProto(r);
            snaps.add(s);
            union = Coreset.union(union, s.coreset());
        }
        return new GatheredSnapshot(union, snaps);
    }

    /**
     * Optionally compresses the gathered coreset once more, then runs weighted k-means++ and Lloyd on it.
     *
     * @param finalCompress algorithm for the final reduction, or {@code null} to cluster the union directly
     */
    public static KMeansResult cluster(Coreset coreset, int k, int maxIterations, long seed,
                                       CoresetAlgorithm finalCompress) {
        SplittableRandom rng = new SplittableRandom(seed);
        SplittableRandom compressRng = rng.split();
        SplittableRandom lloydRng = rng.split(); // independent of whether a final compression runs
        Coreset input = finalCompress == null ? coreset : finalCompress.compress(coreset, compressRng);
        return WeightedLloyd.cluster(input.points(), k, maxIterations, lloydRng);
    }

    /** Cost of {@code centers} on the union of the nodes' session coresets (offsets included), summed node-side. */
    public double cost(String sessionId, List<SparseVector> centers) throws InterruptedException {
        PartialCostRequest req = PartialCostRequest.newBuilder()
                .addAllCenters(ProtoCodec.vectorsToProto(centers))
                .setSessionId(sessionId)
                .build();
        List<ListenableFuture<PartialCostResponse>> calls = new ArrayList<>();
        for (CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub stub : futures) {
            calls.add(deadline(stub, options.rpcDeadline()).partialCost(req));
        }
        double total = 0.0;
        for (PartialCostResponse r : awaitAll(calls)) {
            total += r.getCost();
        }
        return total;
    }

    /** Exact cost of {@code centers} on raw {@code points}: shards of {@code shardSize} go round-robin to nodes. */
    public double cost(List<SparseVector> centers, List<WeightedPoint> points, int shardSize)
            throws InterruptedException {
        if (shardSize < 1) {
            throw new IllegalArgumentException("shardSize must be >= 1: " + shardSize);
        }
        List<io.github.c0rwin.coreset.grpc.proto.SparseVector> protoCenters = ProtoCodec.vectorsToProto(centers);
        List<ListenableFuture<PartialCostResponse>> calls = new ArrayList<>();
        for (int from = 0, shard = 0; from < points.size(); from += shardSize, shard++) {
            List<WeightedPoint> part = points.subList(from, Math.min(points.size(), from + shardSize));
            PartialCostRequest req = PartialCostRequest.newBuilder()
                    .addAllCenters(protoCenters)
                    .setPoints(PointList.newBuilder().addAllPoints(ProtoCodec.pointsToProto(part)))
                    .build();
            calls.add(deadline(futures.get(shard % futures.size()), options.rpcDeadline()).partialCost(req));
        }
        double total = 0.0;
        for (PartialCostResponse r : awaitAll(calls)) {
            total += r.getCost();
        }
        return total;
    }

    /** Stats of every node, in node order. */
    public List<StatsResponse> stats() throws InterruptedException {
        List<ListenableFuture<StatsResponse>> calls = new ArrayList<>();
        for (CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub stub : futures) {
            calls.add(deadline(stub, options.rpcDeadline()).stats(StatsRequest.getDefaultInstance()));
        }
        return awaitAll(calls);
    }

    private static CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub deadline(
            CoresetNodeServiceGrpc.CoresetNodeServiceFutureStub stub, Duration d) {
        return stub.withDeadlineAfter(d.toNanos(), TimeUnit.NANOSECONDS);
    }

    /** Waits for every call; rethrows the first gRPC failure as {@link StatusRuntimeException}. */
    private static <T> List<T> awaitAll(List<? extends ListenableFuture<? extends T>> calls)
            throws InterruptedException {
        List<T> out = new ArrayList<>(calls.size());
        try {
            for (ListenableFuture<? extends T> f : calls) {
                out.add(f.get());
            }
        } catch (ExecutionException e) {
            for (ListenableFuture<? extends T> f : calls) {
                f.cancel(true);
            }
            if (e.getCause() instanceof StatusRuntimeException s) {
                throw s;
            }
            throw new IllegalStateException(e.getCause());
        }
        return out;
    }

    @Override
    public void close() {
        scheduler.shutdownNow();
        for (ManagedChannel ch : ownedChannels) {
            ch.shutdown();
        }
        for (ManagedChannel ch : ownedChannels) {
            try {
                if (!ch.awaitTermination(5, TimeUnit.SECONDS)) {
                    ch.shutdownNow();
                }
            } catch (InterruptedException e) {
                ch.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
