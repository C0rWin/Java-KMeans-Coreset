package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
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
import java.util.TreeSet;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
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
 */
public final class CoresetNodeGrpcService extends CoresetNodeServiceGrpc.CoresetNodeServiceImplBase
        implements AutoCloseable {

    public static final int DEFAULT_INGEST_WINDOW = 16;
    public static final Duration DEFAULT_BUSY_READ_DELAY = Duration.ofMillis(1);

    private final CoresetNode node;
    private final NodeLimits limits;
    private final Supplier<String> statsJson;
    private final int ingestWindow;
    private final long busyReadDelayNanos;
    private final ScheduledExecutorService scheduler;

    public CoresetNodeGrpcService(CoresetNode node, RecordingMetrics stats) {
        this(node, () -> stats.snapshot().toJson(), DEFAULT_INGEST_WINDOW, DEFAULT_BUSY_READ_DELAY);
    }

    /** @param statsJson source of {@code StatsResponse.metrics_json} */
    public CoresetNodeGrpcService(CoresetNode node, Supplier<String> statsJson, int ingestWindow,
                                  Duration busyReadDelay) {
        if (ingestWindow < 1) {
            throw new IllegalArgumentException("ingestWindow must be >= 1: " + ingestWindow);
        }
        this.node = Objects.requireNonNull(node, "node");
        this.limits = node.limits();
        this.statsJson = Objects.requireNonNull(statsJson, "statsJson");
        this.ingestWindow = ingestWindow;
        this.busyReadDelayNanos = busyReadDelay.toNanos();
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "coreset-grpc-ingest-backoff");
            t.setDaemon(true);
            return t;
        });
    }

    @Override
    public void createSession(CreateSessionRequest request, StreamObserver<CreateSessionResponse> response) {
        unary(response, () -> CreateSessionResponse.newBuilder()
                .setCreated(node.createSession(request.getSessionId(),
                        ProtoCodec.fromProto(request.getSpec(), node.limits())))
                .build());
    }

    @Override
    public void closeSession(CloseSessionRequest request, StreamObserver<CloseSessionResponse> response) {
        unary(response, () -> CloseSessionResponse.newBuilder()
                .setClosed(node.closeSession(request.getSessionId()))
                .build());
    }

    @Override
    public StreamObserver<Batch> ingest(StreamObserver<Ack> responseObserver) {
        ServerCallStreamObserver<Ack> call = (ServerCallStreamObserver<Ack>) responseObserver;
        call.disableAutoRequest();
        call.request(ingestWindow);
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
        unary(response, () -> {
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
        unary(response, () -> {
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
                .addAllSessionIds(new TreeSet<>(node.sessionIds()))
                .build());
    }

    private interface Handler<T> {
        T handle();
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

    /** Stops the backoff scheduler; the node is owned by the caller. */
    @Override
    public void close() {
        scheduler.shutdownNow();
    }
}
