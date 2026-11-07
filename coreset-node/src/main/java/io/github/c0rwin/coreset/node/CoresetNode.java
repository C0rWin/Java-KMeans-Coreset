package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

/**
 * Hosts coreset ingestion sessions; transport-agnostic (the gRPC service is a thin adapter).
 *
 * <p><b>Ingestion.</b> A session runs {@code W} worker threads. Batch {@code (producer, seq)} is deduplicated
 * (lock-free, see {@code SeqDeduplicator}), then offered to one worker's bounded MPSC ring buffer, chosen by
 * {@code seq} rotated by the producer; a full ring yields {@link IngestResult.Status#BUSY} instead of blocking, so
 * memory per session is bounded by {@code W * queueCapacity} batches. Producers never take a lock: the hot path is a
 * map insert, one CAS on the ring's producer index and a volatile read to decide whether to unpark the worker. An
 * ACCEPTED batch is enqueued, not yet applied: closing the session drops what is still queued, and an ingest that
 * overlaps a close or a failure is answered FAILED rather than ACCEPTED.
 *
 * <p><b>Determinism.</b> Batch {@code seq} of a producer is compressed with a generator derived only from
 * {@code (seed, partition, producer, seq)}, so a replayed or retried batch yields the identical batch coreset
 * regardless of worker or timing, while nodes created with distinct partitions draw independent streams. That is the
 * only reproducibility guarantee in {@link TreeMode#CAS}, where the worker performing a merge, and so the merge
 * generator, depends on timing. In {@link TreeMode#SINGLE_WRITER} the snapshot is reproducible
 * too, provided every worker receives its batches in the same order (BUSY retries can change that order).
 *
 * <p><b>Tree modes.</b> {@link TreeMode#SINGLE_WRITER}: each worker owns its merge-reduce tree and after every
 * insertion publishes the references of its occupied levels (O(log batches), no points copied), so {@link #snapshot}
 * reads W volatile fields wait-free, unions lazily and is always consistent; the price is W partial trees (W times the levels of a single tree in the union).
 * {@link TreeMode#CAS}: all workers share one tree of CAS slots, giving a single tree of {@code log2(batches)} levels
 * and letting any idle worker perform carries; the price is that a carry holds summaries outside the slots, so a
 * snapshot is validated optimistically. Readers ({@link #snapshot}, {@link #awaitApplied}) hold off new carries while
 * the ones in flight finish, so sustained ingest cannot starve them; the wait is bounded by
 * {@link NodeLimits#casSnapshotWait()} (snapshot) or the caller's timeout (await), after which a snapshot is
 * returned flagged inconsistent.
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
    private final AtomicInteger openSessions = new AtomicInteger();
    private final AtomicInteger openWorkers = new AtomicInteger();
    private final ScheduledExecutorService reaper;
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
        long idle = limits.sessionIdleTimeout().toNanos();
        if (idle > 0) {
            reaper = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "coreset-node-reaper");
                t.setDaemon(true);
                return t;
            });
            long period = Math.max(1_000_000L, Math.min(idle / 4, TimeUnit.MINUTES.toNanos(1)));
            reaper.scheduleWithFixedDelay(() -> expireIdle(idle), period, period, TimeUnit.NANOSECONDS);
        } else {
            reaper = null;
        }
    }

    private void expireIdle(long idleNanos) {
        long now = System.nanoTime();
        for (Map.Entry<String, Session> e : sessions.entrySet()) {
            if (e.getValue().idleFor(idleNanos, now) && sessions.remove(e.getKey(), e.getValue())) {
                e.getValue().close(shutdownGrace);
                release(e.getValue().spec.workers());
                metrics.count(NodeMetricNames.SESSIONS_EXPIRED, 1);
            }
        }
    }

    /** Creates and starts a session in partition 0; see {@link #createSession(String, SessionSpec, long)}. */
    public boolean createSession(String sessionId, SessionSpec spec) {
        return createSession(sessionId, spec, 0);
    }

    /**
     * Creates and starts a session. Idempotent: repeating the call with an equal spec returns {@code false} (the
     * partition of the first call stays). Every {@link NodeLimits} bound is checked before the algorithm is resolved
     * or any worker queue is allocated.
     *
     * @param partition this node's share of a session spread over several nodes; every generator of the session is
     *                  derived from {@code (seed, partition)}, so give each node a distinct value to keep their random
     *                  streams independent. 0 derives from the seed alone.
     * @throws IllegalStateException    if the id exists with a different spec, or the node is closed
     * @throws IllegalArgumentException if the id or spec is outside the limits, the partition is negative, or the
     *                                  resolver rejects the algorithm
     * @throws NodeLimitException       if the node has no room for another session or for its workers
     */
    public boolean createSession(String sessionId, SessionSpec spec, long partition) {
        Objects.requireNonNull(spec, "spec");
        if (partition < 0) {
            throw new IllegalArgumentException("partition must be >= 0: " + partition);
        }
        limits.checkSessionId(sessionId);
        limits.checkAlgorithm(spec.algorithmId(), spec.params());
        if (spec.queueCapacity() > limits.maxQueueCapacity()) {
            throw new IllegalArgumentException("queueCapacity " + spec.queueCapacity() + " exceeds the node limit of "
                    + limits.maxQueueCapacity());
        }
        requireOpen();
        Session existing = sessions.get(sessionId);
        if (existing != null) {
            return requireSameSpec(existing, spec);
        }
        reserve(spec.workers());
        Session session;
        try {
            CoresetAlgorithm algorithm = resolver.resolve(spec.algorithmId(), spec.params());
            if (algorithm == null) {
                throw new IllegalStateException("resolver returned null for " + Messages.bounded(spec.algorithmId()));
            }
            session = new Session(sessionId, spec, partition, algorithm, metrics, limits);
        } catch (Throwable t) {
            release(spec.workers());
            throw asLimit(t);
        }
        Session raced = sessions.putIfAbsent(sessionId, session);
        if (raced != null) {
            release(spec.workers());
            return requireSameSpec(raced, spec);
        }
        // From here on, whoever removes the session from the map closes it and releases its reservation.
        try {
            session.start();
        } catch (Throwable t) {
            discard(sessionId, session);
            throw asLimit(t);
        }
        if (closed) {
            discard(sessionId, session); // a no-op if close() already took it
            throw new IllegalStateException("node closed");
        }
        metrics.count(NodeMetricNames.SESSIONS_CREATED, 1);
        return true;
    }

    private void reserve(int workers) {
        if (openSessions.incrementAndGet() > limits.maxSessions()) {
            openSessions.decrementAndGet();
            throw new NodeLimitException("node already hosts the maximum of " + limits.maxSessions() + " sessions");
        }
        if (openWorkers.addAndGet(workers) > limits.maxTotalWorkers()) {
            openWorkers.addAndGet(-workers);
            openSessions.decrementAndGet();
            throw new NodeLimitException(workers + " more workers would exceed the node limit of "
                    + limits.maxTotalWorkers());
        }
    }

    private void release(int workers) {
        openWorkers.addAndGet(-workers);
        openSessions.decrementAndGet();
    }

    private void discard(String sessionId, Session session) {
        if (sessions.remove(sessionId, session)) {
            session.close(shutdownGrace);
            release(session.spec.workers());
        }
    }

    /** Out of heap or native threads while setting a session up is the node's capacity, not a bug: map it. */
    private static RuntimeException asLimit(Throwable t) {
        if (t instanceof VirtualMachineError) {
            return new NodeLimitException("not enough memory or threads for the session: " + Messages.bounded(t), t);
        }
        if (t instanceof RuntimeException r) {
            return r;
        }
        if (t instanceof Error e) {
            throw e;
        }
        return new IllegalStateException(t);
    }

    private static boolean requireSameSpec(Session existing, SessionSpec spec) {
        if (!existing.spec.equals(spec)) {
            throw new IllegalStateException("session " + Messages.bounded(existing.id) + " exists with a different spec");
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
        release(s.spec.workers());
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
        IngestResult r = s == null ? IngestResult.failed("unknown session: " + Messages.bounded(sessionId))
                : s.ingest(producerId, seq, batch);
        record(r, start);
        return r;
    }

    /**
     * Answers a batch from its header alone when the outcome is already known (unknown, failed or closed session,
     * committed seq, full target queue), so a transport can skip decoding it. Returns null when the batch must be
     * decoded and passed to {@link #ingest(String, String, long, List)}, which re-checks everything. Never blocks.
     */
    public IngestResult admit(String sessionId, String producerId, long seq) {
        long start = metrics == MetricsSink.NOOP ? 0L : System.nanoTime();
        Session s = sessions.get(sessionId);
        IngestResult r = s == null ? IngestResult.failed("unknown session: " + Messages.bounded(sessionId))
                : s.admit(producerId, seq);
        if (r != null) {
            record(r, start);
        }
        return r;
    }

    private void record(IngestResult r, long start) {
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
    }

    /** Union of the session's published summaries plus completeness counters; wait-free in single-writer mode. */
    public NodeSnapshot snapshot(String sessionId) {
        Session s = session(sessionId);
        s.touch();
        return s.snapshot();
    }

    /**
     * Waits until every batch accepted before this call is reflected in {@link #snapshot}; never cancelled.
     *
     * @see #awaitApplied(String, Duration, BooleanSupplier)
     */
    public boolean awaitApplied(String sessionId, Duration timeout) throws InterruptedException {
        return awaitApplied(sessionId, timeout, () -> false);
    }

    /**
     * Waits until every batch accepted before this call is reflected in {@link #snapshot}, at most
     * {@code min(timeout, limits.maxAwait())}.
     *
     * @param cancelled polled whenever the waiting thread wakes; whoever flips it should unpark that thread
     * @return false on timeout
     * @throws SessionFailedException                       if the session failed
     * @throws java.util.concurrent.CancellationException once {@code cancelled} returns true
     */
    public boolean awaitApplied(String sessionId, Duration timeout, BooleanSupplier cancelled)
            throws InterruptedException {
        Session s = session(sessionId);
        s.touch();
        return s.awaitApplied(timeout, cancelled);
    }

    /** Cost of {@code centers} on raw {@code points}; never cancelled. */
    public PartialCost partialCost(List<SparseVector> centers, List<WeightedPoint> points) {
        return partialCost(centers, points, () -> false);
    }

    /**
     * {@code sum_p w_p * min_c ||p - c||^2} over the given points: one shard of a distributed cost evaluation, in one
     * pass with no allocation.
     *
     * @throws IllegalArgumentException                     if the shard or the center count exceeds the limits
     * @throws NodeLimitException                           if points times centers exceeds the work budget
     * @throws java.util.concurrent.CancellationException if {@code cancelled} turns true midway
     */
    public PartialCost partialCost(List<SparseVector> centers, List<WeightedPoint> points, BooleanSupplier cancelled) {
        limits.checkPoints(points.size());
        limits.checkCost(points.size(), centers.size());
        return PartialCost.of(points, centers, 0.0, cancelled);
    }

    /** Cost of {@code centers} on the session's current snapshot; never cancelled. */
    public PartialCost partialCost(String sessionId, List<SparseVector> centers) {
        return partialCost(sessionId, centers, () -> false);
    }

    /** Cost of {@code centers} on the session's current snapshot, including its additive offset. */
    public PartialCost partialCost(String sessionId, List<SparseVector> centers, BooleanSupplier cancelled) {
        Coreset c = snapshot(sessionId).coreset();
        limits.checkCost(c.size(), centers.size());
        return PartialCost.of(c.points(), centers, c.offset(), cancelled);
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

    public NodeLimits limits() {
        return limits;
    }

    /** Closes every session and joins all worker threads; sessions are stopped together, not one after another. */
    @Override
    public void close() {
        closed = true;
        if (reaper != null) {
            reaper.shutdownNow();
        }
        List<Session> taken = new ArrayList<>();
        for (String id : sessions.keySet()) {
            Session s = sessions.remove(id);
            if (s != null) {
                s.signalStop();
                taken.add(s);
            }
        }
        for (Session s : taken) {
            s.close(shutdownGrace);
            release(s.spec.workers());
            metrics.count(NodeMetricNames.SESSIONS_CLOSED, 1);
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
