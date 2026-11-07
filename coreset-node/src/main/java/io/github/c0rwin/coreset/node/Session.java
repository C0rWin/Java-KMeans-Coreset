package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.locks.LockSupport;
import java.util.random.RandomGenerator;

/** One ingestion session: dedupe, routing to workers, failure state and snapshots. Lock-free throughout. */
final class Session {

    static final int MAX_PRODUCER_ID_LENGTH = 128;

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
    private final DoubleAdder acceptedWeight = new DoubleAdder();
    private final AtomicInteger dim = new AtomicInteger(-1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final ConcurrentLinkedQueue<Thread> waiters = new ConcurrentLinkedQueue<>();
    private volatile boolean closed;

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

    IngestResult ingest(String producerId, long seq, List<WeightedPoint> batch) {
        Throwable f = failure.get();
        if (f != null) {
            return IngestResult.failed("session failed: " + f);
        }
        if (closed) {
            return IngestResult.failed("session closed");
        }
        if (seq < 0) {
            return IngestResult.failed("seq must be >= 0: " + seq);
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
        // The union is built here, on demand, rather than on every insertion.
        int size = 0;
        for (Coreset c : parts) {
            size += c.size();
        }
        List<WeightedPoint> points = new ArrayList<>(size);
        double offset = 0.0;
        for (Coreset c : parts) {
            points.addAll(c.points());
            offset += c.offset();
        }
        Coreset coreset = new Coreset(points, offset);
        if (metrics != MetricsSink.NOOP) {
            metrics.record(NodeMetricNames.SNAPSHOT_POINTS, coreset.size());
            metrics.record(NodeMetricNames.SNAPSHOT_BYTES, coreset.estimatedBytes());
        }
        Throwable f = failure.get();
        return new NodeSnapshot(coreset, state(), f == null ? "" : f.toString(), accepted, appliedBatches, acceptedW,
                appliedWeight, consistent);
    }

    /** Worker {@code i}'s last published view (tests). */
    Worker.View workerView(int i) {
        return workers[i].view();
    }

    /**
     * Waits until every batch accepted before this call is reflected in the trees. Per-worker FIFO order makes
     * "worker i applied at least as many batches as had been enqueued to it" sufficient; in CAS mode new carries are
     * then held off until the ones in flight are placed, so other producers cannot starve the wait.
     */
    boolean awaitApplied(Duration timeout) throws InterruptedException {
        long[] targets = new long[workers.length];
        for (int i = 0; i < workers.length; i++) {
            targets[i] = workers[i].accepted();
        }
        long deadline = System.nanoTime() + timeout.toNanos();
        Thread me = Thread.currentThread();
        waiters.add(me);
        try {
            for (;;) {
                Throwable f = failure.get();
                if (f != null) {
                    throw new SessionFailedException(id, f);
                }
                if (closed) {
                    throw new IllegalStateException("session " + id + " closed");
                }
                if (reached(targets, deadline)) {
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

    private boolean reached(long[] targets, long deadline) {
        for (int i = 0; i < workers.length; i++) {
            if (workers[i].applied() < targets[i]) {
                return false;
            }
        }
        // A carry in progress holds applied batches outside every slot.
        return casTree == null || casTree.awaitQuiescent(deadline, this::stopping);
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

    /**
     * Stops all workers (batches still queued are dropped) and joins them, interrupting any that is still busy after
     * {@code grace}. Returns whether every worker thread terminated.
     */
    boolean close(Duration grace) {
        closed = true;
        for (Worker w : workers) {
            w.wakeForStop();
        }
        wakeWaiters();
        boolean interrupted = false;
        boolean allDone = true;
        for (Worker w : workers) {
            try {
                w.thread.join(grace.toMillis());
                if (w.thread.isAlive()) {
                    w.thread.interrupt();
                    w.thread.join(grace.toMillis());
                }
            } catch (InterruptedException e) {
                interrupted = true;
            }
            if (w.thread.isAlive()) {
                allDone = false;
            } else {
                w.queue.clear(); // consumer-side op, safe once the consumer thread is gone
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        return allDone;
    }
}
