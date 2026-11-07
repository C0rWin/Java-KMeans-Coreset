package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.List;
import java.util.concurrent.locks.LockSupport;
import java.util.random.RandomGenerator;
import org.jctools.queues.MpscArrayQueue;

/**
 * Single consumer of one bounded MPSC queue. In {@link TreeMode#SINGLE_WRITER} it exclusively owns a
 * {@link MergeReduceTree} and its generator, so the tree needs no synchronisation; after each insertion it publishes
 * an immutable {@link View} of the tree's level references through a volatile field (O(levels), no point copies). In {@link TreeMode#CAS} it inserts into the session's shared
 * {@link CasMergeReduceTree} instead.
 */
final class Worker implements Runnable {

    /** Short spin before parking: cheap when batches arrive back to back, no CPU burn when idle. */
    private static final int SPINS = 64;

    record PendingBatch(long seq, long producerSalt, List<WeightedPoint> points, double weight) {
    }

    /** Published tree state: occupied levels (shared immutable coresets) and the input they summarise. */
    record View(List<Coreset> levels, long batches, double inputWeight) {
        static final View EMPTY = new View(List.of(), 0, 0.0);
    }

    final MpscArrayQueue<PendingBatch> queue;
    final Thread thread;
    private final Session session;
    private final RandomGenerator treeRng;
    private final MergeReduceTree tree;
    private final String depthMetric;
    private volatile View view = View.EMPTY;
    private volatile long applied;
    private volatile boolean parked;

    Worker(Session session, int index, RandomGenerator treeRng) {
        this.session = session;
        this.treeRng = treeRng;
        this.queue = new MpscArrayQueue<>(session.spec.queueCapacity());
        this.tree = session.casTree == null ? new MergeReduceTree(session.algorithm, treeRng, session.metrics) : null;
        this.depthMetric = NodeMetricNames.workerTreeDepth(index);
        this.thread = new Thread(this, "coreset-node-" + session.id + "-w" + index);
        this.thread.setDaemon(true);
    }

    @Override
    public void run() {
        try {
            while (!session.stopping()) {
                PendingBatch batch = queue.relaxedPoll();
                if (batch == null) {
                    idle();
                } else {
                    process(batch);
                }
            }
        } catch (Throwable t) {
            session.fail(t);
        }
    }

    private void idle() {
        for (int i = 0; i < SPINS; i++) {
            if (!queue.isEmpty() || session.stopping()) {
                return;
            }
            Thread.onSpinWait();
        }
        // Dekker handshake with wake(): publish "parked" before re-checking, producers offer before reading it.
        parked = true;
        if (queue.isEmpty() && !session.stopping()) {
            LockSupport.park(this);
        }
        parked = false;
    }

    private void process(PendingBatch batch) {
        MetricsSink metrics = session.metrics;
        Coreset input = Coreset.of(batch.points());
        Coreset compressed;
        if (input.isEmpty()) {
            compressed = input;
        } else {
            try (Span ignored = metrics.span(NodeMetricNames.BATCH_COMPRESS)) {
                compressed = session.algorithm.compress(input,
                        Seeds.batch(session.spec.seed(), batch.producerSalt(), batch.seq()), metrics);
            }
            if (compressed == null) {
                throw new IllegalStateException(session.algorithm.getClass().getName() + " returned null");
            }
        }
        int height;
        if (tree != null) {
            tree.add(compressed);
            View prev = view;
            view = new View(tree.levelsView(), prev.batches() + 1, prev.inputWeight() + batch.weight());
            height = 64 - Long.numberOfLeadingZeros(tree.batchesAdded());
        } else {
            session.casTree.add(new Summary(compressed, 1, batch.weight()), treeRng);
            height = session.casTree.height();
        }
        applied = applied + 1; // single writer: no read-modify-write race
        if (metrics != MetricsSink.NOOP) {
            metrics.record(depthMetric, height);
        }
        session.onApplied();
    }

    /** Called by producers after a successful offer. */
    void wake() {
        if (parked) {
            LockSupport.unpark(thread);
        }
    }

    void wakeForStop() {
        LockSupport.unpark(thread);
    }

    /** Last published tree view ({@link TreeMode#SINGLE_WRITER} only). */
    View view() {
        return view;
    }

    /** Batches this worker has fully inserted into its tree. */
    long applied() {
        return applied;
    }

    /** Batches ever enqueued here; monotonic, counts an offer as soon as it claims its slot. */
    long accepted() {
        return queue.currentProducerIndex();
    }
}
