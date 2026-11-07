package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Chunk buffer feeding a {@link MergeReduceTree}, cut into epochs: {@link #takePartial} summarises the epoch's records
 * and starts the next epoch with an empty tree. Records are buffered by reference in one reusable array, so the hot
 * path allocates nothing and touches no Flink state; each full chunk is copied once into an immutable batch.
 *
 * <p>Chunks are numbered across epochs. The merge carrying out of {@code level} when the epoch's {@code s}-th chunk
 * arrives uses {@code derive(seed, stream, level, firstChunk + s)}, and the epoch's final reduce uses level
 * {@link MergeRandomness#FINAL_LEVEL}, so every compression is a pure function of its position in the stream. A level
 * {@code i} coreset of the tree has been compressed exactly {@code i} times.
 *
 * <p>Single-threaded, like the tree.
 */
final class ChunkedTree {

    /** Summary of one epoch. */
    record Partial(Coreset coreset, long records, double weight, int depth) {
    }

    private final CoresetAlgorithm algorithm;
    private final long seed;
    private final long stream;
    private final WeightedPoint[] chunk;
    private final LevelShadow shadow = new LevelShadow();
    private int buffered;
    private long firstChunk;
    private MergeReduceTree tree;
    private long records;
    private double weight;
    private long peakStateBytes;

    ChunkedTree(CoresetAlgorithm algorithm, long seed, long stream, int chunkSize) {
        this.algorithm = algorithm;
        this.seed = seed;
        this.stream = stream;
        this.chunk = new WeightedPoint[chunkSize];
        this.tree = newTree(0);
    }

    private MergeReduceTree newTree(long first) {
        firstChunk = first;
        return new MergeReduceTree(algorithm, randomness(first), MetricsSink.NOOP);
    }

    private MergeRandomness randomness(long first) {
        final long s = seed;
        final long st = stream;
        return (level, sequence) -> new SplittableRandom(MergeRandomness.derive(s, st, level, first + sequence));
    }

    /** Buffers {@code p}; true when the chunk is full and {@link #flushChunk} must be called. */
    boolean add(WeightedPoint p) {
        chunk[buffered++] = p;
        records++;
        weight += p.weight();
        return buffered == chunk.length;
    }

    /**
     * Inserts the buffered points, if any, into the tree as one batch. State only grows between flushes, so the state
     * size just before each flush, sized from the batch's one encoding pass, is tracked as {@link #peakStateBytes()}.
     */
    void flushChunk() {
        if (buffered == 0) {
            return;
        }
        // Coreset copies the view once; the array is then reused.
        Coreset batch = Coreset.of(Arrays.asList(chunk).subList(0, buffered));
        long batchBytes = PointCodec.coresetLength(batch);
        // A buffered point carries its own dim varint; inside a coreset one dim header serves all points.
        int dimBytes = PointCodec.varintLength(chunk[0].vector().dim());
        long bufferBytes = 4 + batchBytes - 8 - PointCodec.varintLength(buffered) - dimBytes
                + (long) buffered * dimBytes;
        peakStateBytes = Math.max(peakStateBytes,
                OperatorStates.Builder.FIXED_BYTES + 4 + shadow.totalBytes() + bufferBytes);
        tree.add(batch);
        Coreset carry = tree.levelsView().get(0);
        shadow.onAdd(tree.batchesAdded(), 0, carry == batch ? batchBytes : PointCodec.coresetLength(carry));
        buffered = 0; // stale references are overwritten by the next chunk and never serialized
    }

    /** True if the epoch has seen no record. */
    boolean isEmpty() {
        return records == 0;
    }

    /**
     * Flushes the buffer and returns the epoch's summary: its only level, or the levels' union compressed once. Starts
     * a new epoch. Returns null for an epoch without records.
     */
    Partial takePartial() {
        flushChunk();
        long batches = tree.batchesAdded();
        if (batches == 0) {
            return null;
        }
        boolean compress = tree.levels() > 1;
        Coreset c = compress ? tree.compressedSnapshot() : tree.levelsView().get(0);
        int depth = shadow.maxDepth(batches) + (compress ? 1 : 0);
        Partial p = new Partial(c, records, weight, depth);
        tree = newTree(firstChunk + batches);
        shadow.clear();
        records = 0;
        weight = 0.0;
        return p;
    }

    /** Largest encoded size {@link #state()} reached (just before a flush), restored values included. */
    long peakStateBytes() {
        return peakStateBytes;
    }

    void restorePeak(long bytes) {
        peakStateBytes = Math.max(peakStateBytes, bytes);
    }

    /** Encoded size of {@link #state()}, computed from cached level sizes plus one pass over the buffer. */
    long stateBytes() {
        long buffer = 4;
        for (int i = 0; i < buffered; i++) {
            buffer += PointCodec.pointLength(chunk[i]);
        }
        return OperatorStates.Builder.FIXED_BYTES + 4 + shadow.totalBytes() + buffer;
    }

    OperatorStates.Builder state() {
        return new OperatorStates.Builder(firstChunk, tree.batchesAdded(), tree.levelsView(),
                Arrays.asList(chunk).subList(0, buffered), records, weight);
    }

    void restore(OperatorStates.Builder s) {
        if (s.buffer().size() > chunk.length) {
            throw new IllegalStateException("checkpoint buffers " + s.buffer().size()
                    + " points but the chunk size is " + chunk.length + "; restore with the original configuration");
        }
        tree = MergeReduceTree.restore(algorithm, randomness(s.firstChunk()), MetricsSink.NOOP, s.levels(),
                s.batches());
        firstChunk = s.firstChunk();
        shadow.restore(s.batches(), s.levels(), null);
        Arrays.fill(chunk, null);
        buffered = s.buffer().size();
        for (int i = 0; i < buffered; i++) {
            chunk[i] = s.buffer().get(i);
        }
        records = s.records();
        weight = s.weight();
    }

    /** Records of the current epoch, buffered ones included. */
    long records() {
        return records;
    }

    /** Chunks inserted across all epochs. */
    long chunks() {
        return firstChunk + tree.batchesAdded();
    }

    int levels() {
        return tree.levels();
    }

    /** Compressions on the deepest path of the current epoch's tree. */
    int depth() {
        return shadow.maxDepth(tree.batchesAdded());
    }

    List<Coreset> levelsView() {
        return tree.levelsView();
    }
}
