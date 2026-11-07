package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.random.RandomGenerator;

/**
 * Merge-and-reduce binary counter shared by many writer threads without locks.
 *
 * <p>Slot {@code i} holds at most one summary. To insert, a writer tries {@code CAS(slot, null, carry)}; if the slot
 * is occupied it takes the occupant with {@code getAndSet(slot, null)}, compresses {@code union(occupant, carry)}
 * outside any atomic operation and carries the result to the next slot. A take that finds the slot already emptied
 * by a competing writer just retries the place at the same level.
 *
 * <p>Snapshot tradeoff: while a writer holds a taken occupant, that input is in no slot, and a carry can move a
 * summary from slot {@code i} to {@code i+1} between two slot reads. {@link #read} therefore validates a read
 * seqlock-style: every writer counts its in-hand batches in {@code inFlight} before mutating and bumps
 * {@code version} after each mutation, and a read is consistent iff {@code inFlight == 0} (read first) and
 * {@code version} did not change. Under sustained contention a consistent read may never be observed; the caller
 * then gets the last best-effort read, flagged inconsistent.
 */
final class CasMergeReduceTree {

    /** Level {@code i} holds {@code 2^i} insertions, so 64 slots cover any {@code long} count of batches. */
    static final int SLOTS = 64;

    private final AtomicReferenceArray<Summary> slots = new AtomicReferenceArray<>(SLOTS);
    private final AtomicLong version = new AtomicLong();
    private final AtomicLong inFlightBatches = new AtomicLong();
    private final DoubleAdder inFlightWeight = new DoubleAdder();
    private final CoresetAlgorithm algorithm;
    private final MetricsSink metrics;

    record Read(List<Summary> summaries, boolean consistent, int attempts) {
    }

    CasMergeReduceTree(CoresetAlgorithm algorithm, MetricsSink metrics) {
        this.algorithm = algorithm;
        this.metrics = metrics;
    }

    /**
     * Inserts {@code batch}; merges use {@code rng}, which must be owned by the calling thread. Returns the level
     * where the final carry was placed. If compress throws, the batches in hand stay counted as in flight, so later
     * reads are flagged inconsistent (the session fails anyway).
     */
    int add(Summary batch, RandomGenerator rng) {
        inFlightBatches.addAndGet(batch.batches());
        inFlightWeight.add(batch.inputWeight());
        Summary carry = batch;
        try (Span ignored = metrics.span(MetricNames.MRTREE_ADD)) {
            for (int level = 0; level < SLOTS; ) {
                if (slots.compareAndSet(level, null, carry)) {
                    // Order matters for read(): mutate, bump version, then stop counting as in flight.
                    version.incrementAndGet();
                    inFlightWeight.add(-carry.inputWeight());
                    inFlightBatches.addAndGet(-carry.batches());
                    return level;
                }
                Summary occupant = slots.getAndSet(level, null);
                if (occupant == null) {
                    metrics.count(NodeMetricNames.CAS_RETRIES, 1);
                    continue;
                }
                version.incrementAndGet();
                inFlightBatches.addAndGet(occupant.batches());
                inFlightWeight.add(occupant.inputWeight());
                carry = merge(occupant, carry, level, rng);
                level++;
            }
        }
        throw new IllegalStateException("CAS tree overflow");
    }

    private Summary merge(Summary older, Summary newer, int level, RandomGenerator rng) {
        try (Span ignored = metrics.span(MetricNames.MRTREE_MERGE)) {
            Coreset merged = algorithm.compress(Coreset.union(older.coreset(), newer.coreset()), rng, metrics);
            if (merged == null) {
                throw new IllegalStateException(algorithm.getClass().getName() + " returned null");
            }
            if (metrics != MetricsSink.NOOP) {
                metrics.count(MetricNames.MRTREE_MERGES, 1);
                metrics.count(MetricNames.mrtreeLevelMerges(level), 1);
            }
            return new Summary(merged, older.batches() + newer.batches(), older.inputWeight() + newer.inputWeight());
        }
    }

    /** Optimistic read of all slots, retried up to {@code maxAttempts} times until validated. */
    Read read(int maxAttempts) {
        List<Summary> last = List.of();
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            long v1 = version.get();
            List<Summary> got = new ArrayList<>();
            for (int i = 0; i < SLOTS; i++) {
                Summary s = slots.get(i);
                if (s != null) {
                    got.add(s);
                }
            }
            // inFlight before version: a writer that mutated during the scan either still counts as in flight
            // here, or has already bumped version (it bumps before leaving).
            boolean quiet = inFlightBatches.get() == 0;
            if (quiet && version.get() == v1) {
                return new Read(got, true, attempt);
            }
            last = got;
            Thread.onSpinWait();
        }
        return new Read(last, false, maxAttempts);
    }

    long inFlightBatches() {
        return inFlightBatches.get();
    }

    double inFlightWeight() {
        return inFlightWeight.sum();
    }

    /** Highest occupied level + 1 (racy, for metrics). */
    int height() {
        for (int i = SLOTS - 1; i >= 0; i--) {
            if (slots.get(i) != null) {
                return i + 1;
            }
        }
        return 0;
    }
}
