package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A partial coreset travelling through the aggregation tree, with the counters of the subtree it summarises. The
 * coreset is kept encoded: it is shipped far more often than it is read, and the byte count is the metric. Each
 * partial is encoded once by its producer and decoded once by its consumer.
 */
final class Partial implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Seed stream of the aggregation rounds; partition trees use their partition index (>= 0). */
    static final long REDUCE_STREAM = -1;
    /** Per stage: partial count, points, encoded bytes, estimated heap bytes. */
    static final int STAT_WIDTH = 4;

    /** Position among the partials of its stage; partition index at stage 0, group index after a round. */
    final int index;
    final int stage;
    final int points;
    final int depth;
    final int treeLevels;
    final long reduceOps;
    final long inputPoints;
    final double inputWeight;
    /** Encoded coreset; null only for the driver-side final merge, which keeps {@link #value} instead. */
    final byte[] coreset;
    final transient Coreset value;
    /** {@link #STAT_WIDTH} totals per stage {@code 0..stage}, summed over the subtree. */
    final long[] stats;

    private Partial(int index, int stage, int points, int depth, int treeLevels, long reduceOps, long inputPoints,
                    double inputWeight, byte[] coreset, Coreset value, long[] stats) {
        this.index = index;
        this.stage = stage;
        this.points = points;
        this.depth = depth;
        this.treeLevels = treeLevels;
        this.reduceOps = reduceOps;
        this.inputPoints = inputPoints;
        this.inputWeight = inputWeight;
        this.coreset = coreset;
        this.value = value;
        this.stats = stats;
    }

    Coreset coreset() {
        return value != null ? value : CoresetCodec.decode(coreset);
    }

    private static void addOwnStage(long[] stats, int stage, Coreset c, long encodedBytes) {
        int at = stage * STAT_WIDTH;
        stats[at] += 1;
        stats[at + 1] += c.size();
        stats[at + 2] += encodedBytes;
        stats[at + 3] += c.estimatedBytes();
    }

    /**
     * Stage-0 partial: one pass over the partition, feeding a merge-reduce tree in chunks of {@code chunkSize}
     * points, then one compression of the tree's snapshot. Every generator is derived from
     * {@code (seed, partition, level, chunk)}. Each point's entries are appended once to its chunk's block, presized
     * from the previous chunk; no per-chunk list.
     */
    static Partial ofPartition(int partition, Iterator<WeightedPoint> points, CoresetAlgorithm algorithm, long seed,
                               int chunkSize) {
        MergeReduceTree tree = new MergeReduceTree(algorithm, MergeRandomness.derived(seed, partition),
                MetricsSink.NOOP);
        PointBlock.Builder chunk = null;
        long chunkEntries = 0; // entries of the previous full chunk, to presize the next
        long n = 0;
        double weight = 0.0;
        int dim = -1;
        while (points.hasNext()) {
            WeightedPoint p = points.next();
            SparseVector v = p.vector();
            if (dim < 0) {
                dim = v.dim();
            } else if (v.dim() != dim) {
                throw new IllegalArgumentException(
                        "partition " + partition + ": point " + n + " has dim " + v.dim() + ", expected " + dim);
            }
            if (chunk == null) {
                long expected = chunkEntries > 0 ? chunkEntries : (long) v.nnz() * chunkSize;
                chunk = PointBlock.builder(dim, chunkSize, expected);
            }
            chunk.add(v, p.weight());
            n++;
            weight += p.weight();
            if (chunk.size() == chunkSize) {
                PointBlock block = chunk.build();
                chunkEntries = block.storedEntries();
                tree.add(Coreset.of(block));
                chunk = null;
            }
        }
        if (chunk != null) {
            tree.add(Coreset.of(chunk.build()));
        }
        long batches = tree.batchesAdded();
        int levels = 64 - Long.numberOfLeadingZeros(batches);
        Coreset out = tree.compressedSnapshot();
        long ops = batches == 0 ? 0 : batches - Long.bitCount(batches) + 1;
        byte[] encoded = CoresetCodec.encode(out);
        long[] stats = new long[STAT_WIDTH];
        addOwnStage(stats, 0, out, encoded.length);
        // The first batch is merged once per level above it (levels - 1), then compressed by the snapshot.
        return new Partial(partition, 0, out.size(), levels, levels, ops, n, weight, encoded, null, stats);
    }

    /**
     * Merges one group of partials of the previous stage: decodes each, in index order, straight into one block and
     * unions them without copying (one segment per partial), then compresses once with the generator derived from
     * {@code (seed, REDUCE_STREAM, stage, index)}. Sorting makes the result independent of the order in which a
     * shuffle delivered the group. With {@code encode} false (the driver's final merge) the result is kept decoded.
     */
    static Partial merge(Iterable<Partial> group, int index, int stage, CoresetAlgorithm algorithm, long seed,
                         boolean encode) {
        List<Partial> parts = new ArrayList<>();
        for (Partial p : group) {
            parts.add(p);
        }
        parts.sort(Comparator.comparingInt(p -> p.index));
        List<Coreset> decoded = new ArrayList<>(parts.size());
        int depth = 0;
        int treeLevels = 0;
        long ops = 0;
        long inputPoints = 0;
        double inputWeight = 0.0;
        long[] stats = new long[(stage + 1) * STAT_WIDTH];
        for (Partial p : parts) {
            if (p.stage != stage - 1) {
                throw new IllegalStateException("partial of stage " + p.stage + " merged into stage " + stage);
            }
            decoded.add(CoresetCodec.decode(p.coreset));
            depth = Math.max(depth, p.depth);
            treeLevels = Math.max(treeLevels, p.treeLevels);
            ops += p.reduceOps;
            inputPoints += p.inputPoints;
            inputWeight += p.inputWeight;
            for (int i = 0; i < p.stats.length; i++) {
                stats[i] += p.stats[i];
            }
        }
        Coreset out = Coreset.union(decoded);
        if (!out.isEmpty() || out.offset() != 0.0) {
            out = algorithm.compress(out,
                    new SplittableRandom(MergeRandomness.derive(seed, REDUCE_STREAM, stage, index)));
            ops++;
            depth++;
        }
        byte[] encoded = encode ? CoresetCodec.encode(out) : null;
        addOwnStage(stats, stage, out, encode ? encoded.length : CoresetCodec.encodedLength(out));
        return new Partial(index, stage, out.size(), depth, treeLevels, ops, inputPoints, inputWeight, encoded,
                encode ? null : out, stats);
    }
}
