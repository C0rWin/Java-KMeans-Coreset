package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hosts coreset ingestion sessions; transport-agnostic (the gRPC service is a thin adapter).
 *
 * <p><b>Ingestion.</b> A session runs {@code W} worker threads. Batch {@code seq} is deduplicated (lock-free, see
 * {@code SeqDeduplicator}), then offered to worker {@code seq % W}'s bounded MPSC ring buffer; a full ring yields
 * {@link IngestResult.Status#BUSY} instead of blocking, so memory per session is bounded by
 * {@code W * queueCapacity} batches. Producers never take a lock: the hot path is a map insert, one CAS on the ring's
 * producer index and a volatile read to decide whether to unpark the worker.
 *
 * <p><b>Determinism.</b> Batch {@code seq} is compressed with a generator derived only from {@code (seed, seq)}, so a
 * replayed or retried batch yields the identical batch coreset regardless of worker or timing.
 *
 * <p><b>Tree modes.</b> {@link TreeMode#SINGLE_WRITER}: each worker owns its merge-reduce tree and after every
 * insertion publishes the references of its occupied levels (O(log batches), no points copied), so {@link #snapshot}
 * reads W volatile fields wait-free, unions lazily and is always consistent; the price is W partial trees (W times the levels of a single tree in the union).
 * {@link TreeMode#CAS}: all workers share one tree of CAS slots, giving a single tree of {@code log2(batches)} levels
 * and letting any idle worker perform carries; the price is that a carry holds summaries outside the slots, so a
 * snapshot is validated optimistically and may be returned flagged inconsistent under sustained load, and
 * {@link #awaitApplied} additionally needs a moment with no carry in flight.
 *
 * <p><b>Failures.</b> An exception in a worker (e.g. from the algorithm) marks the session
 * {@link SessionState#FAILED}: later ingests return FAILED with the cause, snapshots report it and
 * {@link #awaitApplied} throws {@link SessionFailedException}. No worker dies silently.
 */
public final class CoresetNode implements AutoCloseable {

    /** Default time a closing session waits for a busy worker before interrupting it. */
    public static final Duration DEFAULT_SHUTDOWN_GRACE = Duration.ofSeconds(10);

    private final AlgorithmResolver resolver;
    private final Duration shutdownGrace;
    private final MetricsSink metrics;
    private final NodeLimits limits;
    private final ConcurrentHashMap<String, Session> sessions = new ConcurrentHashMap<>();
    private volatile boolean closed;

    public CoresetNode(AlgorithmResolver resolver) {
        this(resolver, MetricsSink.NOOP);
    }

    public CoresetNode(AlgorithmResolver resolver, MetricsSink metrics) {
        this(resolver, metrics, DEFAULT_SHUTDOWN_GRACE);
    }

    /** @param shutdownGrace how long closing waits for a worker inside the algorithm before interrupting it */
    public CoresetNode(AlgorithmResolver resolver, MetricsSink metrics, Duration shutdownGrace) {
        this(resolver, metrics, shutdownGrace, NodeLimits.defaults());
    }

    /** @param limits resource bounds enforced before anything is allocated for a caller */
    public CoresetNode(AlgorithmResolver resolver, MetricsSink metrics, Duration shutdownGrace, NodeLimits limits) {
        this.resolver = Objects.requireNonNull(resolver, "resolver");
        this.metrics = Objects.requireNonNull(metrics, "metrics");
        this.shutdownGrace = Objects.requireNonNull(shutdownGrace, "shutdownGrace");
        this.limits = Objects.requireNonNull(limits, "limits");
    }

    /**
     * Creates and starts a session. Idempotent: repeating the call with an equal spec returns {@code false}.
     *
     * @throws IllegalStateException    if the id exists with a different spec, or the node is closed
     * @throws IllegalArgumentException if the id is blank or the resolver rejects the algorithm
     */
    public boolean createSession(String sessionId, SessionSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("session id must not be blank");
        }
        requireOpen();
        Session existing = sessions.get(sessionId);
        if (existing != null) {
            return requireSameSpec(existing, spec);
        }
        CoresetAlgorithm algorithm = resolver.resolve(spec.algorithmId(), spec.params());
        if (algorithm == null) {
            throw new IllegalStateException("resolver returned null for " + spec.algorithmId());
        }
        Session session = new Session(sessionId, spec, algorithm, metrics, limits);
        Session raced = sessions.putIfAbsent(sessionId, session);
        if (raced != null) {
            return requireSameSpec(raced, spec);
        }
        session.start();
        if (closed) {
            // If close() already removed the session it has closed it too; either way it is dead.
            if (sessions.remove(sessionId, session)) {
                session.close(shutdownGrace);
            }
            throw new IllegalStateException("node closed");
        }
        metrics.count(NodeMetricNames.SESSIONS_CREATED, 1);
        return true;
    }

    private static boolean requireSameSpec(Session existing, SessionSpec spec) {
        if (!existing.spec.equals(spec)) {
            throw new IllegalStateException("session " + existing.id + " exists with a different spec");
        }
        return false;
    }

    /** Stops the session's workers (dropping queued batches) and forgets it; false if it did not exist. */
    public boolean closeSession(String sessionId) {
        Session s = sessions.remove(sessionId);
        if (s == null) {
            return false;
        }
        s.close(shutdownGrace);
        metrics.count(NodeMetricNames.SESSIONS_CLOSED, 1);
        return true;
    }

    /**
     * Offers one batch of the default producer; see {@link #ingest(String, String, long, List)}.
     */
    public IngestResult ingest(String sessionId, long seq, List<WeightedPoint> batch) {
        return ingest(sessionId, "", seq, batch);
    }

    /**
     * Offers one batch. Never blocks. Idempotent by {@code (producerId, seq)}: {@code seq} is non-negative and unique
     * per batch within the producer (dense seqs from 0 keep the dedupe state small), and a pair that was already
     * accepted returns DUPLICATE. Independent producers, e.g. successive coordinator runs into one session, must use
     * distinct ids; the empty id is the default producer.
     */
    public IngestResult ingest(String sessionId, String producerId, long seq, List<WeightedPoint> batch) {
        Objects.requireNonNull(producerId, "producerId");
        long start = metrics == MetricsSink.NOOP ? 0L : System.nanoTime();
        Session s = sessions.get(sessionId);
        IngestResult r = s == null ? IngestResult.failed("unknown session: " + sessionId)
                : s.ingest(producerId, seq, batch);
        if (metrics != MetricsSink.NOOP) {
            metrics.timeNanos(NodeMetricNames.INGEST_LATENCY, System.nanoTime() - start);
            switch (r.status()) {
                case ACCEPTED -> {
                    metrics.count(NodeMetricNames.INGEST_ACCEPTED, 1);
                    metrics.count(NodeMetricNames.INGEST_ACCEPTED_POINTS, r.acceptedPoints());
                }
                case DUPLICATE -> metrics.count(NodeMetricNames.INGEST_DUPLICATES, 1);
                case BUSY -> metrics.count(NodeMetricNames.INGEST_BUSY, 1);
                case FAILED -> metrics.count(NodeMetricNames.INGEST_FAILED, 1);
            }
        }
        return r;
    }

    /** Union of the session's published summaries plus completeness counters; wait-free in single-writer mode. */
    public NodeSnapshot snapshot(String sessionId) {
        return session(sessionId).snapshot();
    }

    /**
     * Waits until every batch accepted before this call is reflected in {@link #snapshot}.
     *
     * @return false on timeout
     * @throws SessionFailedException if the session failed
     */
    public boolean awaitApplied(String sessionId, Duration timeout) throws InterruptedException {
        return session(sessionId).awaitApplied(timeout);
    }

    /** {@code sum_p w_p * min_c ||p - c||^2} over the given points: one shard of a distributed cost evaluation. */
    public double partialCost(List<SparseVector> centers, List<WeightedPoint> points) {
        return Cost.cost(points, centers);
    }

    /** Cost of {@code centers} on the session's current snapshot, including its additive offset. */
    public double partialCost(String sessionId, List<SparseVector> centers) {
        return Cost.cost(snapshot(sessionId).coreset(), centers);
    }

    public SessionSpec spec(String sessionId) {
        return session(sessionId).spec;
    }

    public Set<String> sessionIds() {
        return Set.copyOf(sessions.keySet());
    }

    public MetricsSink metrics() {
        return metrics;
    }

    /** Closes every session and joins all worker threads. */
    @Override
    public void close() {
        closed = true;
        for (String id : sessions.keySet()) {
            closeSession(id);
        }
    }

    Session session(String sessionId) {
        Session s = sessions.get(sessionId);
        if (s == null) {
            throw new UnknownSessionException(sessionId);
        }
        return s;
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("node closed");
        }
    }
}
