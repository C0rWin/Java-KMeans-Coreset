package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.locks.LockSupport;
import java.util.random.RandomGenerator;

/** One ingestion session: dedupe, routing to workers, failure state and snapshots. Lock-free throughout. */
final class Session {

    private static final int CAS_READ_ATTEMPTS = 1024;

    final String id;
    final SessionSpec spec;
    final CoresetAlgorithm algorithm;
    final MetricsSink metrics;
    final CasMergeReduceTree casTree;
    private final Worker[] workers;
    private final SeqDeduplicator dedupe = new SeqDeduplicator();
    private final DoubleAdder acceptedWeight = new DoubleAdder();
    private final AtomicInteger dim = new AtomicInteger(-1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final ConcurrentLinkedQueue<Thread> waiters = new ConcurrentLinkedQueue<>();
    private volatile boolean closed;

    Session(String id, SessionSpec spec, CoresetAlgorithm algorithm, MetricsSink metrics) {
        this.id = id;
        this.spec = spec;
        this.algorithm = algorithm;
        this.metrics = metrics;
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

    IngestResult ingest(long seq, List<WeightedPoint> batch) {
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
        switch (dedupe.claim(seq)) {
            case DUPLICATE:
                return IngestResult.duplicate();
            case IN_FLIGHT:
                return IngestResult.busy("seq " + seq + " is being enqueued by another call");
            case CLAIMED:
                break;
            default:
                throw new AssertionError();
        }
        Worker w = workers[(int) (seq % workers.length)];
        if (!w.queue.offer(new Worker.PendingBatch(seq, points, weight))) {
            dedupe.release(seq);
            return IngestResult.busy("worker queue full");
        }
        dedupe.commit(seq);
        acceptedWeight.add(weight);
        w.wake();
        if (metrics != MetricsSink.NOOP) {
            metrics.record(NodeMetricNames.QUEUE_DEPTH, w.queue.size());
        }
        return IngestResult.accepted(points.size());
    }

    NodeSnapshot snapshot() {
        List<Summary> parts;
        boolean consistent = true;
        if (casTree == null) {
            parts = new ArrayList<>(workers.length);
            for (Worker w : workers) {
                parts.add(w.view());
            }
        } else {
            CasMergeReduceTree.Read read = casTree.read(CAS_READ_ATTEMPTS);
            parts = read.summaries();
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
        int size = 0;
        for (Summary s : parts) {
            size += s.coreset().size();
        }
        List<WeightedPoint> points = new ArrayList<>(size);
        double offset = 0.0;
        long appliedBatches = 0;
        double appliedWeight = 0.0;
        for (Summary s : parts) {
            points.addAll(s.coreset().points());
            offset += s.coreset().offset();
            appliedBatches += s.batches();
            appliedWeight += s.inputWeight();
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

    /**
     * Waits until every batch accepted before this call is reflected in the trees. Per-worker FIFO order makes
     * "worker i applied at least as many batches as had been enqueued to it" sufficient.
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
                if (reached(targets)) {
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

    private boolean reached(long[] targets) {
        for (int i = 0; i < workers.length; i++) {
            if (workers[i].applied() < targets[i]) {
                return false;
            }
        }
        // A carry in progress holds applied batches outside every slot.
        return casTree == null || casTree.inFlightBatches() == 0;
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
