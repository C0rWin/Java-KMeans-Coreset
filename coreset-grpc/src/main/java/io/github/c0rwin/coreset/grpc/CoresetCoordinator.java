package io.github.c0rwin.coreset.grpc;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
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
import io.grpc.ChannelCredentials;
import io.grpc.InsecureChannelCredentials;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.TlsChannelCredentials;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Client side of a coreset cluster: creates sessions on every node, streams batches to them
 * ({@link #openIngest}), gathers and unions their snapshots, clusters the union and evaluates costs in a distributed
 * way. Snapshots may be gathered at any time, including while an {@link IngestStream} is still sending.
 */
public final class CoresetCoordinator implements AutoCloseable {

    /** Client keepalive; at least the node's permitted ping interval ({@link NodeServerOptions#PERMIT_KEEPALIVE}). */
    public static final Duration KEEPALIVE_TIME = Duration.ofSeconds(30);
    public static final Duration KEEPALIVE_TIMEOUT = Duration.ofSeconds(10);

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

    /**
     * Connects to {@code host:port} targets over channels owned (and closed) by the coordinator, with keepalive
     * pings so a silently dead node is noticed within about {@link #KEEPALIVE_TIME} plus {@link #KEEPALIVE_TIMEOUT}.
     */
    public static CoresetCoordinator connect(List<String> targets, CoordinatorOptions options, ClientSecurity security,
                                             MetricsSink metrics) throws IOException {
        ChannelCredentials credentials = credentials(security);
        List<ManagedChannel> channels = new ArrayList<>(targets.size());
        try {
            for (String target : targets) {
                NettyChannelBuilder b = NettyChannelBuilder.forTarget(target, credentials)
                        .maxInboundMessageSize(options.maxMessageBytes())
                        .keepAliveTime(KEEPALIVE_TIME.toNanos(), TimeUnit.NANOSECONDS)
                        .keepAliveTimeout(KEEPALIVE_TIMEOUT.toNanos(), TimeUnit.NANOSECONDS)
                        .intercept(GrpcMetrics.client(metrics));
                if (security.authToken() != null) {
                    b.intercept(TokenAuth.client(security.authToken()));
                }
                channels.add(b.build());
            }
        } catch (RuntimeException e) {
            channels.forEach(ManagedChannel::shutdownNow);
            throw e;
        }
        return new CoresetCoordinator(channels, channels, options);
    }

    private static ChannelCredentials credentials(ClientSecurity security) throws IOException {
        if (!security.tls()) {
            return InsecureChannelCredentials.create();
        }
        TlsChannelCredentials.Builder tls = TlsChannelCredentials.newBuilder().trustManager(security.trustCa().toFile());
        if (security.certChain() != null) {
            tls.keyManager(security.certChain().toFile(), security.privateKey().toFile());
        }
        return tls.build();
    }

    public int nodes() {
        return async.size();
    }

    /**
     * Creates the session on every node (idempotent per node). Node {@code i} gets partition {@code i}, so the nodes
     * derive independent random streams from the shared seed.
     */
    public void createSession(String sessionId, SessionSpec spec) throws InterruptedException {
        CreateSessionRequest.Builder req = CreateSessionRequest.newBuilder()
                .setSessionId(sessionId)
                .setSpec(ProtoCodec.toProto(spec));
        List<ListenableFuture<?>> calls = new ArrayList<>(futures.size());
        for (int i = 0; i < futures.size(); i++) {
            calls.add(deadline(futures.get(i), options.rpcDeadline()).createSession(req.setPartition(i).build()));
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

    /** Opens an ingest stream under a fresh random producer id, so it never collides with earlier runs. */
    public IngestStream openIngest(String sessionId) {
        return openIngest(sessionId, UUID.randomUUID().toString());
    }

    /**
     * Opens an ingest stream under {@code producerId}. Batch coresets are a function of
     * {@code (seed, node partition, producer, seq)}, so a fixed id makes a run reproducible. Reusing an id that
     * already sent batches into this session makes the nodes answer DUPLICATE, which fails the stream.
     */
    public IngestStream openIngest(String sessionId, String producerId) {
        Objects.requireNonNull(producerId, "producerId");
        if (producerId.isEmpty()) {
            throw new IllegalArgumentException("producer id must not be empty");
        }
        return new IngestStream(sessionId, producerId, async, options, scheduler);
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
        List<SnapshotResponse> responses = awaitAll(calls);
        List<NodeSnapshot> snaps = new ArrayList<>(responses.size());
        List<Coreset> parts = new ArrayList<>(responses.size());
        for (SnapshotResponse r : responses) {
            NodeSnapshot s = ProtoCodec.fromProto(r, options.maxDim());
            snaps.add(s);
            parts.add(s.coreset());
        }
        // One concatenation instead of pairwise unions, which copied everything gathered so far per node.
        return new GatheredSnapshot(Coreset.union(parts), snaps);
    }

    /**
     * Optionally compresses the gathered coreset once more, then keeps the cheapest of {@code restarts} runs of
     * weighted k-means++ and Lloyd on it ({@link WeightedLloyd#bestOf}); one run often stops in a poor local optimum.
     *
     * @param finalCompress algorithm for the final reduction, or {@code null} to cluster the union directly
     * @throws IllegalArgumentException if {@code restarts < 1}
     */
    public static KMeansResult cluster(Coreset coreset, int k, int maxIterations, int restarts, long seed,
                                       CoresetAlgorithm finalCompress) {
        SplittableRandom rng = new SplittableRandom(seed);
        SplittableRandom compressRng = rng.split();
        SplittableRandom lloydRng = rng.split(); // independent of whether a final compression runs
        Coreset input = finalCompress == null ? coreset : finalCompress.compress(coreset, compressRng);
        return WeightedLloyd.bestOf(input.pointSet(), k, maxIterations, restarts, lloydRng, MetricsSink.NOOP);
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

    /**
     * Exact cost of {@code centers} on raw {@code points}: shards of {@code shardSize} go round-robin to nodes, at
     * most {@link CoordinatorOptions#costShardsPerNode()} in flight per node. Each shard is encoded only when it is
     * sent, so the request side holds a bounded number of encoded shards; partial costs are summed in shard order.
     */
    public double cost(List<SparseVector> centers, List<WeightedPoint> points, int shardSize)
            throws InterruptedException {
        if (shardSize < 1) {
            throw new IllegalArgumentException("shardSize must be >= 1: " + shardSize);
        }
        List<io.github.c0rwin.coreset.grpc.proto.SparseVector> protoCenters = ProtoCodec.vectorsToProto(centers);
        int shards = (int) ((points.size() + (long) shardSize - 1) / shardSize);
        Semaphore[] slots = new Semaphore[futures.size()];
        for (int i = 0; i < slots.length; i++) {
            slots[i] = new Semaphore(options.costShardsPerNode());
        }
        AtomicReference<Throwable> failed = new AtomicReference<>();
        List<ListenableFuture<PartialCostResponse>> calls = new ArrayList<>(shards);
        try {
            for (int shard = 0; shard < shards && failed.get() == null; shard++) {
                int node = shard % futures.size();
                slots[node].acquire();
                int from = shard * shardSize;
                PointList.Builder shardPoints = PointList.newBuilder();
                for (int i = from, end = Math.min(points.size(), from + shardSize); i < end; i++) {
                    shardPoints.addPoints(ProtoCodec.toProto(points.get(i)));
                }
                ListenableFuture<PartialCostResponse> call = deadline(futures.get(node), options.rpcDeadline())
                        .partialCost(PartialCostRequest.newBuilder().addAllCenters(protoCenters)
                                .setPoints(shardPoints).build());
                calls.add(call);
                Futures.addCallback(call, new FutureCallback<>() {
                    @Override
                    public void onSuccess(PartialCostResponse r) {
                        slots[node].release();
                    }

                    @Override
                    public void onFailure(Throwable t) {
                        failed.compareAndSet(null, t);
                        slots[node].release();
                    }
                }, MoreExecutors.directExecutor());
            }
            double total = 0.0;
            for (PartialCostResponse r : awaitAll(calls)) { // call order is shard order
                total += r.getCost();
            }
            return total;
        } catch (InterruptedException | RuntimeException e) {
            calls.forEach(c -> c.cancel(true));
            throw e;
        }
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
