package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;
import java.util.random.RandomGenerator;

/** One ingestion session: dedupe, routing to workers, failure state and snapshots. Lock-free throughout. */
final class Session {

    static final int MAX_PRODUCER_ID_LENGTH = 128;
    /** Power of two: the O(points) snapshot size estimate is recorded for one snapshot in this many. */
    private static final int BYTES_SAMPLE_EVERY = 16;

    final String id;
    final SessionSpec spec;
    final CoresetAlgorithm algorithm;
    final MetricsSink metrics;
    final CasMergeReduceTree casTree;
    private final NodeLimits limits;
    private final Worker[] workers;
    private final Producer defaultProducer;
    private final ConcurrentHashMap<String, Producer> producers = new ConcurrentHashMap<>();
    private final AtomicInteger producerCount = new AtomicInteger();
    private final AtomicInteger snapshots = new AtomicInteger();
    private final DoubleAdder acceptedWeight = new DoubleAdder();
    private final AtomicInteger dim = new AtomicInteger(-1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final ConcurrentLinkedQueue<Thread> waiters = new ConcurrentLinkedQueue<>();
    private volatile boolean closed;
    /** Last read or observed ingest progress (nanoTime); written by readers and the reaper, never by ingest. */
    private volatile long lastActiveNanos = System.nanoTime();
    private long reaperProgress = -1; // reaper thread only

    Session(String id, SessionSpec spec, CoresetAlgorithm algorithm, MetricsSink metrics, NodeLimits limits) {
        this.id = id;
        this.spec = spec;
        this.algorithm = algorithm;
        this.metrics = metrics;
        this.limits = limits;
        this.defaultProducer = new Producer(Producer.DEFAULT_ID, limits.maxSeqWindow());
        this.casTree = spec.treeMode() == TreeMode.CAS ? new CasMergeReduceTree(algorithm, metrics) : null;
        RandomGenerator[] rngs = Seeds.workers(spec.seed(), spec.workers());
        this.workers = new Worker[spec.workers()];
        for (int i = 0; i < workers.length; i++) {
            workers[i] = new Worker(this, i, rngs[i]);
        }
    }

    void start() {
        for (Worker w : workers) {
            w.thread.start();
        }
    }

    boolean stopping() {
        return closed || failure.get() != null;
    }

    SessionState state() {
        if (failure.get() != null) {
            return SessionState.FAILED;
        }
        return closed ? SessionState.CLOSED : SessionState.ACTIVE;
    }

    /**
     * One pass over the batch (dimension check fused with the weight sum); the caller's list is copied once unless
     * it already is an immutable {@code List.of}/{@code List.copyOf} list, which is taken by reference.
     */
    IngestResult ingest(String producerId, long seq, List<WeightedPoint> batch) {
        Throwable f = failure.get();
        if (f != null) {
            return IngestResult.failed("session failed: " + Messages.bounded(f));
        }
        if (closed) {
            return IngestResult.failed("session closed");
        }
        if (seq < 0) {
            return IngestResult.failed("seq must be >= 0: " + seq);
        }
        if (batch.size() > limits.maxPointsPerBatch()) {
            return IngestResult.failed(batch.size() + " points exceed the per-batch limit of "
                    + limits.maxPointsPerBatch());
        }
        List<WeightedPoint> points;
        try {
            points = List.copyOf(batch);
        } catch (NullPointerException e) {
            return IngestResult.failed("batch contains null points");
        }
        double weight = 0.0;
        if (!points.isEmpty()) {
            int d = points.get(0).vector().dim();
            if (d > limits.maxDim()) {
                return IngestResult.failed("dim " + d + " exceeds the node limit of " + limits.maxDim());
            }
            for (WeightedPoint p : points) {
                if (p.vector().dim() != d) {
                    return IngestResult.failed("batch mixes dimensions " + d + " and " + p.vector().dim());
                }
                weight += p.weight();
            }
            int sessionDim = dim.get();
            if (sessionDim < 0 && !dim.compareAndSet(-1, d)) {
                sessionDim = dim.get();
            }
            if (sessionDim >= 0 && sessionDim != d) {
                return IngestResult.failed("batch dimension " + d + " differs from session dimension " + sessionDim);
            }
        }
        Producer producer = producer(producerId);
        if (producer == null) {
            return IngestResult.failed(producerId.length() > MAX_PRODUCER_ID_LENGTH
                    ? "producer id longer than " + MAX_PRODUCER_ID_LENGTH + " chars"
                    : "session has the maximum of " + limits.maxProducersPerSession() + " producers");
        }
        SeqDeduplicator dedupe = producer.dedupe;
        switch (dedupe.claim(seq)) {
            case DUPLICATE:
                return IngestResult.duplicate();
            case IN_FLIGHT:
                return IngestResult.busy("seq " + seq + " is being enqueued by another call");
            case OUT_OF_WINDOW:
                return IngestResult.failed("seq " + seq + " is outside the dedupe window of " + dedupe.window()
                        + " above the producer's lowest unacknowledged seq " + dedupe.floor()
                        + "; number batches densely per producer");
            case CLAIMED:
                break;
            default:
                throw new AssertionError();
        }
        Worker w = workers[producer.worker(seq, workers.length)];
        // Counted before the offer so a snapshot never sees the batch applied but not accepted.
        acceptedWeight.add(weight);
        boolean offered = false;
        try {
            offered = w.queue.offer(new Worker.PendingBatch(seq, producer.salt, points, weight));
        } finally {
            if (offered) {
                dedupe.commit(seq);
            } else {
                // Also on an Error between claim and offer: an abandoned claim would answer BUSY forever.
                dedupe.release(seq);
                acceptedWeight.add(-weight);
            }
        }
        if (!offered) {
            return IngestResult.busy("worker queue full");
        }
        w.wake();
        if (stopping()) {
            // close() or fail() began after the checks above: the worker may never take this batch.
            return IngestResult.failed("session " + (closed ? "closed" : "failed") + " while the batch was enqueued");
        }
        if (metrics != MetricsSink.NOOP) {
            metrics.record(NodeMetricNames.QUEUE_DEPTH, w.queue.size());
        }
        return IngestResult.accepted(points.size());
    }

    /** The producer's state, created on first use; null if the id is too long or the session has too many. */
    private Producer producer(String id) {
        if (id.isEmpty()) {
            return defaultProducer;
        }
        Producer p = producers.get(id);
        if (p != null) {
            return p;
        }
        if (id.length() > MAX_PRODUCER_ID_LENGTH) {
            return null;
        }
        if (producerCount.incrementAndGet() > limits.maxProducersPerSession()) {
            producerCount.decrementAndGet();
            return producers.get(id);
        }
        Producer fresh = new Producer(id, limits.maxSeqWindow());
        Producer raced = producers.putIfAbsent(id, fresh);
        if (raced != null) {
            producerCount.decrementAndGet();
            return raced;
        }
        return fresh;
    }

    NodeSnapshot snapshot() {
        List<Coreset> parts = new ArrayList<>();
        long appliedBatches = 0;
        double appliedWeight = 0.0;
        boolean consistent = true;
        if (casTree == null) {
            for (Worker w : workers) {
                Worker.View v = w.view();
                parts.addAll(v.levels());
                appliedBatches += v.batches();
                appliedWeight += v.inputWeight();
            }
        } else {
            long waitNanos = stopping() ? 0L : limits.casSnapshotWait().toNanos();
            CasMergeReduceTree.Read read = casTree.read(waitNanos, this::stopping);
            for (Summary s : read.summaries()) {
                parts.add(s.coreset());
                appliedBatches += s.batches();
                appliedWeight += s.inputWeight();
            }
            consistent = read.consistent();
            if (metrics != MetricsSink.NOOP) {
                metrics.record(NodeMetricNames.SNAPSHOT_CAS_ATTEMPTS, read.attempts());
                metrics.record(NodeMetricNames.CAS_IN_FLIGHT_WEIGHT, casTree.inFlightWeight());
                if (!consistent) {
                    metrics.count(NodeMetricNames.SNAPSHOT_INCONSISTENT, 1);
                }
            }
        }
        // Accepted is read after the parts, so it can never be below what they report as applied.
        long accepted = 0;
        for (Worker w : workers) {
            accepted += w.accepted();
        }
        double acceptedW = acceptedWeight.sum();
        // The union is built here, on demand, rather than on every insertion: one reference copy into a pre-sized
        // array, plus the one List.of makes (Coreset keeps an immutable list as is).
        long size = 0;
        for (Coreset c : parts) {
            size += c.size();
        }
        if (size > limits.maxSnapshotPoints()) {
            throw new NodeLimitException("snapshot of session " + Messages.bounded(id) + " has " + size
                    + " points, above the limit of " + limits.maxSnapshotPoints());
        }
        WeightedPoint[] points = new WeightedPoint[(int) size];
        int at = 0;
        double offset = 0.0;
        for (Coreset c : parts) {
            for (WeightedPoint p : c.points()) {
                points[at++] = p;
            }
            offset += c.offset();
        }
        Coreset coreset = new Coreset(List.of(points), offset);
        if (metrics != MetricsSink.NOOP) {
            metrics.record(NodeMetricNames.SNAPSHOT_POINTS, coreset.size());
            if ((snapshots.getAndIncrement() & (BYTES_SAMPLE_EVERY - 1)) == 0) {
                metrics.record(NodeMetricNames.SNAPSHOT_BYTES, coreset.estimatedBytes()); // O(points): sampled
            }
        }
        Throwable f = failure.get();
        return new NodeSnapshot(coreset, state(), f == null ? "" : Messages.bounded(f), accepted, appliedBatches,
                acceptedW, appliedWeight, consistent);
    }

    /** Marks a read (snapshot, await, cost) as activity for the idle reaper. */
    void touch() {
        lastActiveNanos = System.nanoTime();
    }

    /**
     * Called by the single reaper thread. Ingest progress is detected from the workers' own counters, so producers
     * pay nothing for idle tracking.
     */
    boolean idleFor(long timeoutNanos, long now) {
        long progress = 0;
        for (Worker w : workers) {
            progress += w.accepted() + w.applied();
        }
        if (progress != reaperProgress) {
            reaperProgress = progress;
            lastActiveNanos = now;
            return false;
        }
        return now - lastActiveNanos > timeoutNanos;
    }

    /** Worker {@code i}'s last published view (tests). */
    Worker.View workerView(int i) {
        return workers[i].view();
    }

    /**
     * Waits until every batch accepted before this call is reflected in the trees. Per-worker FIFO order makes
     * "worker i applied at least as many batches as had been enqueued to it" sufficient; in CAS mode new carries are
     * then held off until the ones in flight are placed, so other producers cannot starve the wait.
     *
     * @param cancelled polled on every wake-up; whoever flips it must unpark the waiting thread
     * @throws CancellationException once {@code cancelled} returns true
     */
    boolean awaitApplied(Duration timeout, BooleanSupplier cancelled) throws InterruptedException {
        long[] targets = new long[workers.length];
        for (int i = 0; i < workers.length; i++) {
            targets[i] = workers[i].accepted();
        }
        Duration wait = timeout.compareTo(limits.maxAwait()) > 0 ? limits.maxAwait() : timeout;
        long deadline = System.nanoTime() + wait.toNanos();
        BooleanSupplier stop = () -> stopping() || cancelled.getAsBoolean();
        Thread me = Thread.currentThread();
        waiters.add(me);
        try {
            for (;;) {
                Throwable f = failure.get();
                if (f != null) {
                    throw new SessionFailedException(id, f);
                }
                if (closed) {
                    throw new IllegalStateException("session " + Messages.bounded(id) + " closed");
                }
                if (cancelled.getAsBoolean()) {
                    throw new CancellationException("awaitApplied cancelled");
                }
                if (reached(targets, deadline, stop)) {
                    return true;
                }
                long remaining = deadline - System.nanoTime();
                if (remaining <= 0) {
                    return false;
                }
                LockSupport.parkNanos(this, remaining);
                if (Thread.interrupted()) {
                    throw new InterruptedException();
                }
            }
        } finally {
            waiters.remove(me);
        }
    }

    private boolean reached(long[] targets, long deadline, BooleanSupplier stop) {
        for (int i = 0; i < workers.length; i++) {
            if (workers[i].applied() < targets[i]) {
                return false;
            }
        }
        // A carry in progress holds applied batches outside every slot.
        return casTree == null || casTree.awaitQuiescent(deadline, stop);
    }

    void onApplied() {
        if (!waiters.isEmpty()) {
            wakeWaiters();
        }
    }

    private void wakeWaiters() {
        for (Thread t : waiters) {
            LockSupport.unpark(t);
        }
    }

    void fail(Throwable cause) {
        if (closed || !failure.compareAndSet(null, cause)) {
            return;
        }
        metrics.count(NodeMetricNames.SESSION_FAILURES, 1);
        for (Worker w : workers) {
            w.wakeForStop();
        }
        wakeWaiters();
    }

    /** Marks the session closed and wakes its workers and waiters without waiting for them. */
    void signalStop() {
        closed = true;
        for (Worker w : workers) {
            w.wakeForStop();
        }
        wakeWaiters();
    }

    /**
     * Stops all workers (batches still queued are dropped) and joins them in parallel: every worker gets the same
     * {@code grace} deadline, then all still busy are interrupted and get one more {@code grace}. So closing takes at
     * most {@code 2 * grace} whatever the worker count. Returns whether every worker thread terminated.
     */
    boolean close(Duration grace) {
        signalStop();
        boolean interrupted = !joinAll(System.nanoTime() + grace.toNanos());
        boolean anyAlive = false;
        for (Worker w : workers) {
            if (w.thread.isAlive()) {
                w.thread.interrupt();
                anyAlive = true;
            }
        }
        if (anyAlive) {
            interrupted |= !joinAll(System.nanoTime() + grace.toNanos());
        }
        boolean allDone = true;
        for (Worker w : workers) {
            allDone &= !w.thread.isAlive();
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        return allDone;
    }

    /** Joins every worker until {@code deadline} (nanoTime); false if the calling thread was interrupted. */
    private boolean joinAll(long deadline) {
        for (Worker w : workers) {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) {
                return true;
            }
            try {
                w.thread.join(Math.max(1L, remaining / 1_000_000L));
            } catch (InterruptedException e) {
                return false;
            }
        }
        return true;
    }
}
