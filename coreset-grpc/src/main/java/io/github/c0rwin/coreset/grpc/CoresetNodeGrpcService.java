package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionResponse;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionResponse;
import io.github.c0rwin.coreset.grpc.proto.PartialCostRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostResponse;
import io.github.c0rwin.coreset.grpc.proto.SnapshotRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.grpc.proto.StatsRequest;
import io.github.c0rwin.coreset.grpc.proto.StatsResponse;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.Messages;
import io.github.c0rwin.coreset.node.NodeLimitException;
import io.github.c0rwin.coreset.node.NodeLimits;
import io.github.c0rwin.coreset.node.PartialCost;
import io.github.c0rwin.coreset.node.SessionFailedException;
import io.github.c0rwin.coreset.node.UnknownSessionException;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CancellationException;
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

    private final CoresetNode node;
    private final NodeLimits limits;
    private final Supplier<String> statsJson;
    private final Config config;
    private final long busyReadDelayNanos;
    private final ScheduledExecutorService scheduler;
    private final ThreadPoolExecutor blocking;

    /** @param statsJson source of {@code StatsResponse.metrics_json} */
    public CoresetNodeGrpcService(CoresetNode node, Supplier<String> statsJson, Config config) {
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
                        ProtoCodec.fromProto(request.getSpec(), limits)))
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
                IngestResult r;
                try {
                    limits.checkPoints(batch.getPointsCount()); // before decoding: refused batches cost no heap
                    List<WeightedPoint> points = ProtoCodec.pointsFromProto(batch.getPointsList(), limits.maxDim());
                    r = node.ingest(batch.getSessionId(), batch.getProducerId(), batch.getSeq(), points);
                } catch (IllegalArgumentException e) {
                    r = IngestResult.failed("invalid batch: " + e.getMessage());
                }
                call.onNext(ProtoCodec.ack(batch.getSeq(), r));
                if (r.status() == IngestResult.Status.BUSY) {
                    scheduler.schedule(() -> {
                        if (!call.isCancelled()) {
                            call.request(1);
                        }
                    }, busyReadDelayNanos, TimeUnit.NANOSECONDS);
                } else {
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
            long ms = request.getAwaitAppliedMs(); // uint64: values above 2^63 arrive negative
            if (ms != 0) {
                long cap = limits.maxAwait().toMillis();
                Duration wait = Duration.ofMillis(ms < 0 || ms > cap ? cap : ms);
                Context ctx = Context.current();
                Thread waiter = Thread.currentThread();
                Context.CancellationListener wake = c -> LockSupport.unpark(waiter);
                ctx.addListener(wake, Runnable::run);
                try {
                    node.awaitApplied(request.getSessionId(), wait, ctx::isCancelled);
                } catch (SessionFailedException e) {
                    // Reported through the snapshot's state and failure fields.
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw Status.CANCELLED.withDescription("interrupted").asRuntimeException();
                } finally {
                    ctx.removeListener(wake);
                }
            }
            return ProtoCodec.toProto(node.snapshot(request.getSessionId()));
        });
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
        try {
            blocking.execute(Context.current().wrap(() -> unary(response, handler)));
        } catch (RejectedExecutionException e) {
            response.onError(Status.RESOURCE_EXHAUSTED.withDescription("node busy: too many blocking calls queued")
                    .asRuntimeException());
        }
    }

    private static <T> void unary(StreamObserver<T> response, Handler<T> handler) {
        T result;
        try {
            result = handler.handle();
        } catch (StatusRuntimeException e) {
            response.onError(e);
            return;
        } catch (UnknownSessionException e) {
            response.onError(status(Status.NOT_FOUND, e));
            return;
        } catch (IllegalArgumentException e) {
            response.onError(status(Status.INVALID_ARGUMENT, e));
            return;
        } catch (CancellationException e) {
            response.onError(status(Status.CANCELLED, e));
            return;
        } catch (IllegalStateException e) {
            response.onError(status(Status.FAILED_PRECONDITION, e));
            return;
        } catch (NodeLimitException e) {
            response.onError(status(Status.RESOURCE_EXHAUSTED, e));
            return;
        } catch (OutOfMemoryError e) {
            // The request is dropped and its garbage reclaimable; the node keeps serving other sessions.
            response.onError(Status.RESOURCE_EXHAUSTED.withDescription("out of memory").asRuntimeException());
            return;
        } catch (RuntimeException e) {
            response.onError(Status.INTERNAL.withDescription("internal error").withCause(e).asRuntimeException());
            return;
        }
        response.onNext(result);
        response.onCompleted();
    }

    private static StatusRuntimeException status(Status status, Throwable e) {
        return status.withDescription(Messages.bounded(e.getMessage())).asRuntimeException();
    }

    /** Stops the backoff scheduler and the blocking executor; the node is owned by the caller. */
    @Override
    public void close() {
        scheduler.shutdownNow();
        blocking.shutdownNow();
    }
}
