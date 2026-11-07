package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.algo.Algorithms;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;
import org.apache.spark.api.java.function.VoidFunction2;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Structured Streaming sink for {@code writeStream().foreachBatch(sink)}: each micro-batch is reduced to a coreset by
 * a distributed {@link SparkCoresets#build} and inserted into a driver-side {@link MergeReduceTree}, whose state is
 * checkpointed after every batch.
 *
 * <p><b>Exactly once per batchId.</b> The checkpoint is written before {@code foreachBatch} returns, i.e. before
 * Spark commits the batch. A batchId at or below the last applied one is skipped, so a batch Spark replays after a
 * failure between the two commits is not counted twice, and a restarted query resumes the tree where it stopped. Batch
 * {@code b} is built with the seed derived from {@code (seed, -2, 0, b)} and tree merges use
 * {@code MergeRandomness.derived(seed, -3)}, so a resumed sink produces bit-identically the state an uninterrupted
 * one would have (given the same batch contents).
 *
 * <p>Batches are applied under the sink's monitor (Spark calls {@code foreachBatch} sequentially anyway). Readers
 * never block: every batch publishes an immutable view that {@link #snapshot()} and friends read.
 */
public final class CoresetStreamSink implements VoidFunction2<Dataset<Row>, Long> {

    private static final long serialVersionUID = 1L;
    static final long BATCH_STREAM = -2;
    static final long TREE_STREAM = -3;

    private final transient StreamCheckpoint checkpoint;
    private final transient CoresetAlgorithm algorithm;
    private final transient MergeRandomness randomness;
    private final transient MergeReduceTree tree;
    private final PointColumns columns;
    private final String algorithmId;
    private final TreeMap<String, String> params;
    private final long seed;
    private final int treeDepth;
    private final int chunkSize;
    private long lastBatchId;
    /** Set when a batch failed after the tree started changing; the in-memory tree is then untrusted. */
    private boolean broken;
    private volatile View view;

    /** Immutable state published after every applied or skipped batch. */
    private record View(long lastBatchId, long batches, List<Coreset> levels, long skipped, BuildMetrics lastBuild) {
    }

    private CoresetStreamSink(StreamCheckpoint checkpoint, PointColumns columns, String algorithmId,
                              Map<String, String> params, long seed, int treeDepth, int chunkSize) throws IOException {
        this.checkpoint = checkpoint;
        this.columns = Objects.requireNonNull(columns, "columns");
        this.algorithmId = algorithmId;
        this.params = new TreeMap<>(params);
        this.seed = seed;
        this.treeDepth = treeDepth;
        this.chunkSize = chunkSize;
        this.algorithm = Algorithms.create(algorithmId, params);
        this.randomness = MergeRandomness.derived(seed, TREE_STREAM);
        Optional<StreamCheckpoint.State> state = checkpoint.latest();
        if (state.isPresent()) {
            StreamCheckpoint.State s = state.get();
            if (!s.sameConfig(seed, algorithmId, params, treeDepth, chunkSize)) {
                throw new IllegalStateException("checkpoint was written by a sink with algorithm " + s.algorithmId()
                        + " " + s.params() + ", seed " + s.seed() + ", treeDepth " + s.treeDepth() + ", chunkSize "
                        + s.chunkSize() + "; resuming with a different configuration would mix incompatible state");
            }
            tree = MergeReduceTree.restore(algorithm, randomness, MetricsSink.NOOP, s.levels(), s.batches());
            lastBatchId = s.lastBatchId();
        } else {
            tree = new MergeReduceTree(algorithm, randomness, MetricsSink.NOOP);
            lastBatchId = -1;
        }
        view = new View(lastBatchId, tree.batchesAdded(), tree.levelsView(), 0, null);
    }

    /** {@link #open(SparkSession, String, PointColumns, String, Map, long, int, int)} with the default chunk size. */
    public static CoresetStreamSink open(SparkSession spark, String checkpointDir, PointColumns columns,
                                         String algorithmId, Map<String, String> params, long seed, int treeDepth)
            throws IOException {
        return open(spark, checkpointDir, columns, algorithmId, params, seed, treeDepth,
                SparkCoresets.defaultChunkSize(params));
    }

    /**
     * Opens a sink on {@code checkpointDir} (any Hadoop file system URI), resuming from its newest state if any.
     *
     * @throws IllegalStateException if the directory holds state written with a different configuration
     * @throws IOException           if the newest state cannot be read or is corrupt
     */
    public static CoresetStreamSink open(SparkSession spark, String checkpointDir, PointColumns columns,
                                         String algorithmId, Map<String, String> params, long seed, int treeDepth,
                                         int chunkSize) throws IOException {
        if (treeDepth < 1 || chunkSize < 1) {
            throw new IllegalArgumentException("treeDepth and chunkSize must be >= 1: " + treeDepth + ", " + chunkSize);
        }
        StreamCheckpoint checkpoint = new StreamCheckpoint(checkpointDir, spark.sparkContext().hadoopConfiguration());
        return new CoresetStreamSink(checkpoint, columns, algorithmId, params, seed, treeDepth, chunkSize);
    }

    @Override
    public void call(Dataset<Row> batch, Long batchId) throws IOException {
        apply(batch, batchId);
    }

    /**
     * Applies one micro-batch unless {@code batchId} was already applied.
     *
     * @return false if the batch was skipped as a replay
     * @throws IllegalStateException if an earlier batch failed after the tree started changing; reopen the sink from
     *                               its checkpoint
     */
    public synchronized boolean apply(Dataset<Row> batch, long batchId) throws IOException {
        if (broken) {
            throw new IllegalStateException("an earlier batch failed mid-update; reopen the sink from its checkpoint");
        }
        View v = view;
        if (batchId <= lastBatchId) {
            view = new View(v.lastBatchId(), v.batches(), v.levels(), v.skipped() + 1, v.lastBuild());
            return false;
        }
        BuildResult built = SparkCoresets.build(batch, columns, algorithmId, params,
                MergeRandomness.derive(seed, BATCH_STREAM, 0, batchId), treeDepth, chunkSize);
        Coreset c = built.coreset();
        try {
            if (!c.isEmpty() || c.offset() != 0.0) {
                tree.add(c);
            }
            checkpoint.write(new StreamCheckpoint.State(seed, algorithmId, params, treeDepth, chunkSize, batchId,
                    tree.batchesAdded(), tree.levelsView()));
        } catch (Throwable t) {
            broken = true; // a carry chain may be half done and the batch is not durable
            throw t;
        }
        lastBatchId = batchId;
        view = new View(batchId, tree.batchesAdded(), tree.levelsView(), v.skipped(), built.metrics());
        return true;
    }

    /** Union of the tree's levels after the last applied batch, uncompressed; empty before the first batch. */
    public Coreset snapshot() {
        return restoredView().snapshot();
    }

    /** {@link #snapshot()} compressed once, exactly as the tree's own {@code compressedSnapshot()} would. */
    public Coreset compressedSnapshot() {
        return restoredView().compressedSnapshot();
    }

    private MergeReduceTree restoredView() {
        View v = view;
        return MergeReduceTree.restore(algorithm, randomness, MetricsSink.NOOP, v.levels(), v.batches());
    }

    /** Last applied batchId, -1 before the first. */
    public long lastBatchId() {
        return view.lastBatchId();
    }

    /** Non-empty batch coresets inserted into the tree, including those restored from the checkpoint. */
    public long batchesAdded() {
        return view.batches();
    }

    /** Batches skipped as replays since this sink was opened. */
    public long skippedBatches() {
        return view.skipped();
    }

    /** Metrics of the last batch built by this sink instance, or null if none yet. */
    public BuildMetrics lastBuildMetrics() {
        return view.lastBuild();
    }
}
