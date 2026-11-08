package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionResponse;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionResponse;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.grpc.proto.PartialCostRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostResponse;
import io.github.c0rwin.coreset.grpc.proto.ReplicateRequest;
import io.github.c0rwin.coreset.grpc.proto.ReplicateResponse;
import io.github.c0rwin.coreset.grpc.proto.SnapshotChunk;
import io.github.c0rwin.coreset.grpc.proto.SnapshotRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.grpc.proto.SnapshotStreamRequest;
import io.github.c0rwin.coreset.grpc.proto.StatsRequest;
import io.github.c0rwin.coreset.grpc.proto.StatsResponse;
import io.github.c0rwin.coreset.grpc.proto.StoreReplicaResponse;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.Messages;
import io.github.c0rwin.coreset.node.NodeLimitException;
import io.github.c0rwin.coreset.node.NodeLimits;
import io.github.c0rwin.coreset.node.NodeMessage;
import io.github.c0rwin.coreset.node.PartialCost;
import io.github.c0rwin.coreset.node.SessionFailedException;
import io.github.c0rwin.coreset.node.UnknownSessionException;
import io.grpc.Context;
import io.grpc.Deadline;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * gRPC adapter over a {@link CoresetNode}. Handlers never block on ingestion: {@code ingest} is a lock-free offer.
 *
 * <p>Ingest flow control: inbound auto-request is disabled. The server requests {@code ingestWindow} batches up
 * front and one more after every non-BUSY ack; after a BUSY ack it waits {@code busyReadDelay} before requesting the
 * next batch, so a saturated node stops reading and HTTP/2 flow control pushes back on the sender instead of the
 * server buffering batches.
 *
 * <p>Calls that wait or scan (session create and close, snapshot, cost) run on a bounded executor of their own, so
 * they can neither starve the ingest callbacks nor grow threads without bound; when its queue is full they fail with
 * RESOURCE_EXHAUSTED.
 */
public final class CoresetNodeGrpcService extends CoresetNodeServiceGrpc.CoresetNodeServiceImplBase
        implements AutoCloseable {

    /**
     * @param ingestWindow       batches requested ahead on each ingest stream
     * @param busyReadDelay      pause before reading on after a BUSY ack
     * @param blockingThreads    threads for blocking calls
     * @param blockingQueue      blocking calls queued beyond the threads
     * @param sessionIdsInStats  list session ids in Stats; only safe when every caller is authenticated
     */
    public record Config(int ingestWindow, Duration busyReadDelay, int blockingThreads, int blockingQueue,
                         boolean sessionIdsInStats) {

        public Config {
            Objects.requireNonNull(busyReadDelay, "busyReadDelay");
            if (ingestWindow < 1 || blockingThreads < 1 || blockingQueue < 1) {
                throw new IllegalArgumentException("ingestWindow, blockingThreads and blockingQueue must be >= 1");
            }
        }

        public static Config defaults() {
            return new Config(16, Duration.ofMillis(1), 4, 256, false);
        }

        public Config withIngest(int window, Duration busyDelay) {
            return new Config(window, busyDelay, blockingThreads, blockingQueue, sessionIdsInStats);
        }

        public Config withBlocking(int threads, int queue) {
            return new Config(ingestWindow, busyReadDelay, threads, queue, sessionIdsInStats);
        }

        public Config withSessionIdsInStats(boolean expose) {
            return new Config(ingestWindow, busyReadDelay, blockingThreads, blockingQueue, expose);
        }
    }

    private static final int MAX_BUSY_SHIFT = 6;
    /** Peers one Replicate call may copy to. */
    static final int MAX_PEERS = 64;
    /** Replication deadline when the caller set none. */
    static final Duration REPLICATION_DEADLINE = Duration.ofSeconds(60);

    private final CoresetNode node;
    private final NodeLimits limits;
    private final Supplier<String> statsJson;
    private final Config config;
    private final long busyReadDelayNanos;
    private final ScheduledExecutorService scheduler;
    private final ThreadPoolExecutor blocking;
    private final PeerChannels peers;

    /** A node that cannot replicate to peers; see the four-argument constructor. */
    public CoresetNodeGrpcService(CoresetNode node, Supplier<String> statsJson, Config config) {
        this(node, statsJson, config, PeerChannels.none());
    }

    /**
     * @param statsJson source of {@code StatsResponse.metrics_json}
     * @param peers     channels for {@code Replicate}; closed with this service
     */
    public CoresetNodeGrpcService(CoresetNode node, Supplier<String> statsJson, Config config, PeerChannels peers) {
        this.peers = Objects.requireNonNull(peers, "peers");
        this.node = Objects.requireNonNull(node, "node");
        this.limits = node.limits();
        this.statsJson = Objects.requireNonNull(statsJson, "statsJson");
        this.config = Objects.requireNonNull(config, "config");
        this.busyReadDelayNanos = config.busyReadDelay().toNanos();
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "coreset-grpc-ingest-backoff");
            t.setDaemon(true);
            return t;
        });
        AtomicInteger n = new AtomicInteger();
        this.blocking = new ThreadPoolExecutor(config.blockingThreads(), config.blockingThreads(), 60, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(config.blockingQueue()), r -> {
                    Thread t = new Thread(r, "coreset-grpc-blocking-" + n.incrementAndGet());
                    t.setDaemon(true);
                    return t;
                });
        this.blocking.allowCoreThreadTimeOut(true);
    }

    @Override
    public void createSession(CreateSessionRequest request, StreamObserver<CreateSessionResponse> response) {
        offload(response, () -> CreateSessionResponse.newBuilder()
                .setCreated(node.createSession(request.getSessionId(),
                        ProtoCodec.fromProto(request.getSpec(), limits), Integer.toUnsignedLong(request.getPartition())))
                .build());
    }

    @Override
    public void closeSession(CloseSessionRequest request, StreamObserver<CloseSessionResponse> response) {
        offload(response, () -> CloseSessionResponse.newBuilder()
                .setClosed(node.closeSession(request.getSessionId()))
                .build());
    }

    @Override
    public StreamObserver<Batch> ingest(StreamObserver<Ack> responseObserver) {
        ServerCallStreamObserver<Ack> call = (ServerCallStreamObserver<Ack>) responseObserver;
        call.disableAutoRequest();
        call.request(config.ingestWindow());
        return new StreamObserver<>() {
            private boolean rejected; // inbound callbacks are serialised by gRPC
            private int busyStreak;

            @Override
            public void onNext(Batch batch) {
                if (rejected) {
                    return;
                }
                if (batch.getSeq() < 0) { // uint64 seq >= 2^63 wraps negative in Java; the node only takes longs >= 0
                    rejected = true;
                    call.onError(Status.INVALID_ARGUMENT.withDescription("seq " + Long.toUnsignedString(batch.getSeq())
                            + " exceeds 2^63-1").asRuntimeException());
                    return;
                }
                // Outcomes known from the header skip the proto-to-core conversion, the bulk of the per-batch heap.
                IngestResult r = node.admit(batch.getSessionId(), batch.getProducerId(), batch.getSeq());
                if (r == null) {
                    try {
                        limits.checkPoints(batch.getPointsCount());
                        // Decoded straight into the batch's CSR block, which the worker then compresses as is.
                        Coreset points = Coreset.of(ProtoCodec.blockFromProto(batch.getPointsList(), limits.maxDim()));
                        r = node.ingest(batch.getSessionId(), batch.getProducerId(), batch.getSeq(), points);
                    } catch (IllegalArgumentException e) {
                        r = IngestResult.failed("invalid batch: " + e.getMessage());
                    }
                }
                call.onNext(ProtoCodec.ack(batch.getSeq(), r));
                if (r.status() == IngestResult.Status.BUSY) {
                    // Back off harder while the node stays saturated: 1x, 2x, ... 64x the base delay.
                    long delay = busyReadDelayNanos << Math.min(busyStreak++, MAX_BUSY_SHIFT);
                    scheduler.schedule(() -> {
                        if (!call.isCancelled()) {
                            call.request(1);
                        }
                    }, delay, TimeUnit.NANOSECONDS);
                } else {
                    busyStreak = 0;
                    call.request(1);
                }
            }

            @Override
            public void onError(Throwable t) {
                // Client cancelled or the transport broke; unacknowledged batches are resent by the client.
            }

            @Override
            public void onCompleted() {
                if (!rejected) {
                    call.onCompleted();
                }
            }
        };
    }

    @Override
    public void snapshot(SnapshotRequest request, StreamObserver<SnapshotResponse> response) {
        offload(response, () -> {
            awaitApplied(request.getSessionId(), request.getAwaitAppliedMs());
            return ProtoCodec.toProto(node.snapshot(request.getSessionId()));
        });
    }

    /**
     * Waits for applied batches if {@code ms != 0} (capped by the node limit); a failed session is not an error
     * here, the reply reports it.
     */
    private void awaitApplied(String sessionId, long ms) {
        if (ms == 0) { // uint64: values above 2^63 arrive negative
            return;
        }
        long cap = limits.maxAwait().toMillis();
        Duration wait = Duration.ofMillis(ms < 0 || ms > cap ? cap : ms);
        Context ctx = Context.current();
        Thread waiter = Thread.currentThread();
        Context.CancellationListener wake = c -> LockSupport.unpark(waiter);
        ctx.addListener(wake, Runnable::run);
        try {
            node.awaitApplied(sessionId, wait, ctx::isCancelled);
        } catch (SessionFailedException e) {
            // Reported through the reply.
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw Status.CANCELLED.withDescription("interrupted").asRuntimeException();
        } finally {
            ctx.removeListener(wake);
        }
    }

    /**
     * Streams the session's message: the reliable header, then each packet in seq order unless only the header was
     * asked for. With {@code replica} the stored copy of {@code owner}'s message is served instead.
     */
    @Override
    public void snapshotStream(SnapshotStreamRequest request, StreamObserver<SnapshotChunk> response) {
        ServerCallStreamObserver<SnapshotChunk> call = (ServerCallStreamObserver<SnapshotChunk>) response;
        offloadCall(response, () -> streaming(response, () -> {
            String id = request.getSessionId();
            NodeMessage m;
            if (request.getReplica()) {
                m = node.replica(id, request.getOwner());
                if (m == null) {
                    throw Status.NOT_FOUND.withDescription("no replica of node " + Long.toUnsignedString(
                            request.getOwner()) + " in session " + Messages.bounded(id)).asRuntimeException();
                }
            } else {
                awaitApplied(id, request.getAwaitAppliedMs());
                m = node.message(id, request.getStream());
            }
            call.onNext(SnapshotChunk.newBuilder()
                    .setHeader(MessageCodec.header(m, request.getOwner(), request.getReplica())).build());
            if (!request.getHeaderOnly()) {
                for (int p = 0, n = m.packets(); p < n && !call.isCancelled(); p++) {
                    call.onNext(SnapshotChunk.newBuilder().setPacket(MessageCodec.packet(m, p)).build());
                }
            }
            call.onCompleted();
        }));
    }

    /**
     * Freezes this node's message and stores it on every peer, all copies in flight at once; answers once every peer
     * answered or the call's deadline (default {@link #REPLICATION_DEADLINE}) passed.
     */
    @Override
    public void replicate(ReplicateRequest request, StreamObserver<ReplicateResponse> response) {
        offload(response, () -> {
            if (request.getPeersCount() > MAX_PEERS) {
                throw new IllegalArgumentException(request.getPeersCount() + " peers exceed " + MAX_PEERS);
            }
            String id = request.getSessionId();
            awaitApplied(id, request.getAwaitAppliedMs());
            NodeMessage m = node.freeze(id, request.getStream());
            SnapshotChunk header = SnapshotChunk.newBuilder()
                    .setHeader(MessageCodec.header(m, request.getOwner(), false).setSessionId(id)).build();
            SnapshotChunk[] packets = new SnapshotChunk[m.packets()];
            for (int p = 0; p < packets.length; p++) { // encoded once for all peers
                packets[p] = SnapshotChunk.newBuilder().setPacket(MessageCodec.packet(m, p)).build();
            }
            Deadline deadline = Context.current().getDeadline();
            if (deadline == null) {
                deadline = Deadline.after(REPLICATION_DEADLINE.toNanos(), TimeUnit.NANOSECONDS);
            }
            CountDownLatch done = new CountDownLatch(request.getPeersCount());
            Queue<String> failed = new ConcurrentLinkedQueue<>();
            for (String peer : request.getPeersList()) {
                StreamObserver<SnapshotChunk> out = CoresetNodeServiceGrpc.newStub(peers.channel(peer))
                        .withDeadline(deadline)
                        .storeReplica(new StreamObserver<>() {
                            @Override
                            public void onNext(StoreReplicaResponse r) {
                            }

                            @Override
                            public void onError(Throwable t) {
                                failed.add(Messages.bounded(peer + ": " + StatusText.of(t)));
                                done.countDown();
                            }

                            @Override
                            public void onCompleted() {
                                done.countDown();
                            }
                        });
                out.onNext(header);
                for (SnapshotChunk p : packets) {
                    out.onNext(p);
                }
                out.onCompleted();
            }
            try {
                if (!done.await(deadline.timeRemaining(TimeUnit.NANOSECONDS), TimeUnit.NANOSECONDS)) {
                    throw Status.DEADLINE_EXCEEDED.withDescription("peers did not confirm in time").asRuntimeException();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw Status.CANCELLED.withDescription("interrupted").asRuntimeException();
            }
            return ReplicateResponse.newBuilder()
                    .setStored(request.getPeersCount() - failed.size())
                    .setRecords(m.recordCount())
                    .addAllFailedPeers(failed)
                    .build();
        });
    }

    /** Receives another node's message (header with session id, then every packet) and keeps it as a replica. */
    @Override
    public StreamObserver<SnapshotChunk> storeReplica(StreamObserver<StoreReplicaResponse> response) {
        return new StreamObserver<>() {
            private MessageHeader header; // inbound callbacks are serialised by gRPC
            private final List<Packet> packets = new ArrayList<>();
            private boolean rejected;

            @Override
            public void onNext(SnapshotChunk chunk) {
                if (rejected) {
                    return;
                }
                if (chunk.hasHeader() && header == null) {
                    header = chunk.getHeader();
                    // Bounded before any packet is buffered; uint32 above 2^31 arrives negative.
                    if (header.getPackets() < 0 || header.getRecords() < 0
                            || header.getRecords() > limits.maxSnapshotPoints()) {
                        reject("replica of " + Integer.toUnsignedString(header.getRecords())
                                + " records exceeds the limit of " + limits.maxSnapshotPoints());
                    }
                } else if (chunk.hasPacket() && header != null && packets.size() < header.getPackets()) {
                    packets.add(chunk.getPacket());
                } else {
                    reject("expected one header, then at most its packet count of packets");
                }
            }

            private void reject(String why) {
                rejected = true;
                packets.clear();
                response.onError(Status.INVALID_ARGUMENT.withDescription(why).asRuntimeException());
            }

            @Override
            public void onError(Throwable t) {
                packets.clear();
            }

            @Override
            public void onCompleted() {
                if (rejected) {
                    return;
                }
                if (header == null) {
                    reject("no header");
                    return;
                }
                offload(response, () -> {
                    NodeMessage m = MessageCodec.decode(header, packets, limits.maxDim(), limits.maxSnapshotPoints());
                    node.storeReplica(header.getSessionId(), header.getOwner(), m);
                    return StoreReplicaResponse.newBuilder().setRecords(m.recordCount()).build();
                });
            }
        };
    }

    @Override
    public void partialCost(PartialCostRequest request, StreamObserver<PartialCostResponse> response) {
        offload(response, () -> {
            BooleanSupplier cancelled = Context.current()::isCancelled;
            PartialCost cost = switch (request.getSourceCase()) {
                case POINTS -> {
                    // Shape and work budget come from the counts, so an oversized request is refused undecoded.
                    int n = request.getPoints().getPointsCount();
                    limits.checkPoints(n);
                    limits.checkCost(n, request.getCentersCount());
                    yield node.partialCost(ProtoCodec.vectorsFromProto(request.getCentersList(), limits.maxDim()),
                            ProtoCodec.pointsFromProto(request.getPoints().getPointsList(), limits.maxDim()),
                            cancelled);
                }
                case SESSION_ID -> {
                    limits.checkCost(0, request.getCentersCount());
                    yield node.partialCost(request.getSessionId(),
                            ProtoCodec.vectorsFromProto(request.getCentersList(), limits.maxDim()), cancelled);
                }
                default -> throw new IllegalArgumentException("source (points or session_id) is required");
            };
            return PartialCostResponse.newBuilder()
                    .setCost(cost.cost())
                    .setWeight(cost.weight())
                    .setPoints(cost.points())
                    .build();
        });
    }

    @Override
    public void stats(StatsRequest request, StreamObserver<StatsResponse> response) {
        unary(response, () -> StatsResponse.newBuilder()
                .setMetricsJson(statsJson.get())
                .addAllSessionIds(config.sessionIdsInStats() ? new TreeSet<>(node.sessionIds()) : Set.of())
                .build());
    }

    private interface Handler<T> {
        T handle();
    }

    /** Runs {@code handler} on the blocking executor with the caller's context (deadline, cancellation). */
    private <T> void offload(StreamObserver<T> response, Handler<T> handler) {
        offloadCall(response, () -> unary(response, handler));
    }

    /** Runs {@code body}, which completes {@code response} itself, on the blocking executor. */
    private void offloadCall(StreamObserver<?> response, Runnable body) {
        try {
            blocking.execute(Context.current().wrap(body));
        } catch (RejectedExecutionException e) {
            response.onError(Status.RESOURCE_EXHAUSTED.withDescription("node busy: too many blocking calls queued")
                    .asRuntimeException());
        }
    }

    private static <T> void unary(StreamObserver<T> response, Handler<T> handler) {
        T result;
        try {
            result = handler.handle();
        } catch (RuntimeException | OutOfMemoryError e) {
            response.onError(statusOf(e));
            return;
        }
        response.onNext(result);
        response.onCompleted();
    }

    /** Streams through {@code body}; a failure before or while streaming ends the call with its status. */
    private static void streaming(StreamObserver<?> response, Runnable body) {
        try {
            body.run();
        } catch (RuntimeException | OutOfMemoryError e) {
            response.onError(statusOf(e));
        }
    }

    private static StatusRuntimeException statusOf(Throwable e) {
        if (e instanceof StatusRuntimeException s) {
            return s;
        }
        if (e instanceof UnknownSessionException) {
            return status(Status.NOT_FOUND, e);
        }
        if (e instanceof IllegalArgumentException) {
            return status(Status.INVALID_ARGUMENT, e);
        }
        if (e instanceof CancellationException) {
            return status(Status.CANCELLED, e);
        }
        if (e instanceof IllegalStateException) {
            return status(Status.FAILED_PRECONDITION, e);
        }
        if (e instanceof NodeLimitException) {
            return status(Status.RESOURCE_EXHAUSTED, e);
        }
        if (e instanceof OutOfMemoryError) {
            // The request is dropped and its garbage reclaimable; the node keeps serving other sessions.
            return Status.RESOURCE_EXHAUSTED.withDescription("out of memory").asRuntimeException();
        }
        return Status.INTERNAL.withDescription("internal error").withCause(e).asRuntimeException();
    }

    private static StatusRuntimeException status(Status status, Throwable e) {
        return status.withDescription(Messages.bounded(e.getMessage())).asRuntimeException();
    }

    /** Stops the backoff scheduler and the blocking executor and closes the peer channels; the node is the caller's. */
    @Override
    public void close() {
        scheduler.shutdownNow();
        blocking.shutdownNow();
        peers.close();
    }
}
