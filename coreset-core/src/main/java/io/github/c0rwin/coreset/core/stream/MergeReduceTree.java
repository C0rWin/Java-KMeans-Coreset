package io.github.c0rwin.coreset.core.stream;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Streaming merge-and-reduce over coresets, organised as a binary counter (Bentley-Saxe): level {@code i} holds at
 * most one coreset summarising {@code 2^i} batches. Adding a batch places it at level 0; while the target level is
 * occupied, the two coresets are merged ({@code compress(union(older, newer))}) and carried one level up. After
 * {@code b} batches the occupied levels are exactly the set bits of {@code b}, and each batch has been compressed at
 * most {@code floor(log2 b)} times.
 *
 * <p>Not thread-safe and holds no locks: use it from a single writer thread (or serialise all
 * calls externally); the {@link RandomGenerator} is used unsynchronised. Inputs are immutable {@link Coreset}s and
 * are never modified.
 */
public final class MergeReduceTree {

    private final CoresetAlgorithm algorithm;
    private final RandomGenerator rng;
    private final MetricsSink metrics;
    private final List<Coreset> levels = new ArrayList<>(); // null marks an empty level
    private long batches;
    private int occupied;

    public MergeReduceTree(CoresetAlgorithm algorithm, RandomGenerator rng) {
        this(algorithm, rng, MetricsSink.NOOP);
    }

    public MergeReduceTree(CoresetAlgorithm algorithm, RandomGenerator rng, MetricsSink metrics) {
        this.algorithm = Objects.requireNonNull(algorithm, "algorithm");
        this.rng = Objects.requireNonNull(rng, "rng");
        this.metrics = Objects.requireNonNull(metrics, "metrics");
    }

    /** Inserts a batch coreset at level 0 and carries merges upward. */
    public void add(Coreset batchCoreset) {
        Objects.requireNonNull(batchCoreset, "batchCoreset");
        try (Span ignored = metrics.span(MetricNames.MRTREE_ADD)) {
            Coreset carry = batchCoreset;
            int level = 0;
            int chain = 0;
            while (level < levels.size() && levels.get(level) != null) {
                carry = merge(levels.get(level), carry, level);
                levels.set(level, null);
                occupied--;
                level++;
                chain++;
            }
            if (level == levels.size()) {
                levels.add(carry);
            } else {
                levels.set(level, carry);
            }
            occupied++;
            batches++;
            if (metrics != MetricsSink.NOOP) {
                metrics.record(MetricNames.MRTREE_CARRY_CHAIN_LENGTH, chain);
                metrics.record(MetricNames.mrtreeLevelPoints(level), carry.size());
            }
        }
    }

    private Coreset merge(Coreset older, Coreset newer, int level) {
        try (Span ignored = metrics.span(MetricNames.MRTREE_MERGE)) {
            Coreset union = Coreset.union(older, newer);
            Coreset merged = algorithm.compress(union, rng, metrics);
            if (metrics != MetricsSink.NOOP) {
                metrics.count(MetricNames.MRTREE_MERGES, 1);
                metrics.count(MetricNames.mrtreeLevelMerges(level), 1);
                double in = union.totalWeight();
                double drift = in > 0.0 ? (merged.totalWeight() - in) / in : 0.0;
                metrics.record(MetricNames.MRTREE_MERGE_RELATIVE_WEIGHT_DRIFT, drift);
            }
            return merged;
        }
    }

    /** Union of all levels without further compression; an empty tree yields an empty coreset. */
    public Coreset snapshot() {
        int size = 0;
        double offset = 0.0;
        for (Coreset c : levels) {
            if (c != null) {
                size += c.size();
                offset += c.offset();
            }
        }
        List<WeightedPoint> points = new ArrayList<>(size);
        for (Coreset c : levels) {
            if (c != null) {
                points.addAll(c.points());
            }
        }
        Coreset snapshot = new Coreset(points, offset);
        if (metrics != MetricsSink.NOOP) {
            metrics.record(MetricNames.MRTREE_SNAPSHOT_POINTS, snapshot.size());
            metrics.record(MetricNames.MRTREE_SNAPSHOT_BYTES, snapshot.estimatedBytes());
        }
        return snapshot;
    }

    /**
     * Occupied levels, lowest first, as an immutable list of references to the tree's own (immutable) coresets. Costs
     * O(levels) and copies no points, so a writer can publish it after every add; later adds never change it.
     */
    public List<Coreset> levelsView() {
        List<Coreset> out = new ArrayList<>(occupied);
        for (Coreset c : levels) {
            if (c != null) {
                out.add(c);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** {@link #snapshot()} compressed once by the tree's algorithm; an empty tree yields an empty coreset. */
    public Coreset compressedSnapshot() {
        try (Span ignored = metrics.span(MetricNames.MRTREE_COMPRESSED_SNAPSHOT)) {
            Coreset snapshot = snapshot();
            if (snapshot.isEmpty() && snapshot.offset() == 0.0) {
                return snapshot;
            }
            Coreset compressed = algorithm.compress(snapshot, rng, metrics);
            if (metrics != MetricsSink.NOOP) {
                metrics.record(MetricNames.MRTREE_COMPRESSED_SNAPSHOT_POINTS, compressed.size());
                metrics.record(MetricNames.MRTREE_COMPRESSED_SNAPSHOT_BYTES, compressed.estimatedBytes());
            }
            return compressed;
        }
    }

    /** Number of occupied levels; always {@code Long.bitCount(batchesAdded())}. */
    public int levels() {
        return occupied;
    }

    public long batchesAdded() {
        return batches;
    }
}
