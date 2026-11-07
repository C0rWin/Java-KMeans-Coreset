package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
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
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionFailedException;
import io.github.c0rwin.coreset.node.UnknownSessionException;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.TreeSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
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
                .setCreated(node.createSession(request.getSessionId(), ProtoCodec.fromProto(request.getSpec())))
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
            @Override
            public void onNext(Batch batch) {
                IngestResult r;
                try {
                    List<WeightedPoint> points = ProtoCodec.pointsFromProto(batch.getPointsList());
                    r = node.ingest(batch.getSessionId(), batch.getSeq(), points);
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
                call.onCompleted();
            }
        };
    }

    @Override
    public void snapshot(SnapshotRequest request, StreamObserver<SnapshotResponse> response) {
        unary(response, () -> {
            if (request.getAwaitAppliedMs() > 0) {
                try {
                    node.awaitApplied(request.getSessionId(), Duration.ofMillis(request.getAwaitAppliedMs()));
                } catch (SessionFailedException e) {
                    // Reported through the snapshot's state and failure fields.
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw Status.CANCELLED.withDescription("interrupted").asRuntimeException();
                }
            }
            return ProtoCodec.toProto(node.snapshot(request.getSessionId()));
        });
    }

    @Override
    public void partialCost(PartialCostRequest request, StreamObserver<PartialCostResponse> response) {
        unary(response, () -> {
            List<SparseVector> centers = ProtoCodec.vectorsFromProto(request.getCentersList());
            switch (request.getSourceCase()) {
                case POINTS -> {
                    List<WeightedPoint> points = ProtoCodec.pointsFromProto(request.getPoints().getPointsList());
                    double weight = 0.0;
                    for (WeightedPoint p : points) {
                        weight += p.weight();
                    }
                    return PartialCostResponse.newBuilder()
                            .setCost(node.partialCost(centers, points))
                            .setWeight(weight)
                            .setPoints(points.size())
                            .build();
                }
                case SESSION_ID -> {
                    NodeSnapshot s = node.snapshot(request.getSessionId());
                    Coreset c = s.coreset();
                    return PartialCostResponse.newBuilder()
                            .setCost(Cost.cost(c, centers))
                            .setWeight(c.totalWeight())
                            .setPoints(c.size())
                            .build();
                }
                default -> throw new IllegalArgumentException("source (points or session_id) is required");
            }
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
            response.onError(Status.NOT_FOUND.withDescription(e.getMessage()).asRuntimeException());
            return;
        } catch (IllegalArgumentException e) {
            response.onError(Status.INVALID_ARGUMENT.withDescription(e.getMessage()).asRuntimeException());
            return;
        } catch (IllegalStateException e) {
            response.onError(Status.FAILED_PRECONDITION.withDescription(e.getMessage()).asRuntimeException());
            return;
        }
        response.onNext(result);
        response.onCompleted();
    }

    /** Stops the backoff scheduler; the node is owned by the caller. */
    @Override
    public void close() {
        scheduler.shutdownNow();
    }
}
