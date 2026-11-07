package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.Algorithms;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.TreeMap;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import scala.Tuple2;

/**
 * Distributed coreset construction and coreset-based k-means on Spark.
 *
 * <p>A build runs in three steps. (1) Every partition streams its points through a {@code MergeReduceTree} in chunks
 * and compresses the tree's snapshot once, giving one partial coreset per partition. (2) While more than
 * {@code fanIn = max(2, ceil(partitions^(1/treeDepth)))} partials remain, executors merge groups of {@code fanIn}
 * consecutive partials (union, then one compression). (3) The driver merges the rest the same way. This is the shape
 * of {@code treeAggregate}, built from {@code groupByKey} so that each group is merged in partition order: compression
 * is not associative, and folding values in shuffle arrival order would make results depend on timing.
 *
 * <p>Every generator is derived from {@code (seed, partition or round, level, sequence)} with
 * {@link MergeRandomness#derive}, never from task attempt, thread or time. As long as the input RDD recomputes the
 * same partition contents in the same order (Spark's own requirement for deterministic lineage), retried and
 * speculative tasks produce bit-identical partials and the build is reproducible.
 */
public final class SparkCoresets {

    /** Seed stream of solver restarts; partition trees use streams >= 0 and reduce rounds {@code -1}. */
    static final long SOLVER_STREAM = -4;
    private static final int COST_CHUNK = 4096;

    private SparkCoresets() {
    }

    /** {@link #build(JavaRDD, String, Map, long, int, int)} with {@link #defaultChunkSize(Map)}. */
    public static BuildResult build(JavaRDD<WeightedPoint> points, String algorithmId, Map<String, String> params,
                                    long seed, int treeDepth) {
        return build(points, algorithmId, params, seed, treeDepth, defaultChunkSize(params));
    }

    /**
     * Builds one coreset of {@code points}: one pass over the input; every partial is encoded once by its task and
     * decoded once by its consumer, and the driver holds at most {@code fanIn} partials plus the result.
     *
     * @param algorithmId {@code uniform}, {@code sensitivity} or {@code mean}; see {@link Algorithms#create}
     * @param params      algorithm parameters, validated on the driver before any job starts
     * @param seed        global seed; equal inputs, partitioning and seed give an identical coreset
     * @param treeDepth   depth of the aggregation tree, as in {@code RDD.treeAggregate}: 1 merges all partition
     *                    coresets on the driver, larger values add executor-side rounds
     * @param chunkSize   points per level-0 batch of each partition's merge-reduce tree
     * @throws IllegalArgumentException for invalid parameters, a point dimension mismatch within a partition, or
     *                                  partitions of different dimensions
     */
    public static BuildResult build(JavaRDD<WeightedPoint> points, String algorithmId, Map<String, String> params,
                                    long seed, int treeDepth, int chunkSize) {
        Objects.requireNonNull(points, "points");
        if (treeDepth < 1) {
            throw new IllegalArgumentException("treeDepth must be >= 1: " + treeDepth);
        }
        if (chunkSize < 1) {
            throw new IllegalArgumentException("chunkSize must be >= 1: " + chunkSize);
        }
        CoresetAlgorithm algorithm = Algorithms.create(algorithmId, params);
        TreeMap<String, String> p = new TreeMap<>(params);
        long start = System.nanoTime();
        int partitions = points.getNumPartitions();
        int fanIn = Math.max(2, (int) Math.ceil(Math.pow(partitions, 1.0 / treeDepth)));

        JavaRDD<Partial> current = points.mapPartitionsWithIndex((index, it) -> List.of(
                Partial.ofPartition(index, it, Algorithms.create(algorithmId, p), seed, chunkSize)).iterator(), false);
        int count = partitions;
        int stage = 0;
        while (count > fanIn) {
            stage++;
            int groups = (count + fanIn - 1) / fanIn;
            final int s = stage;
            current = current.mapToPair(x -> new Tuple2<>(x.index / fanIn, x))
                    .groupByKey(groups)
                    .map(g -> Partial.merge(g._2(), g._1(), s, Algorithms.create(algorithmId, p), seed, true));
            count = groups;
        }
        Partial last = Partial.merge(current.collect(), 0, stage + 1, algorithm, seed, false);
        Coreset coreset = last.coreset();
        return new BuildResult(coreset, metrics(last, partitions, chunkSize, fanIn, stage, System.nanoTime() - start));
    }

    /** {@link #build(JavaRDD, String, Map, long, int)} over the points in {@code columns} of {@code data}. */
    public static BuildResult build(Dataset<Row> data, PointColumns columns, String algorithmId,
                                    Map<String, String> params, long seed, int treeDepth) {
        return build(points(data, columns), algorithmId, params, seed, treeDepth);
    }

    /** {@link #build(JavaRDD, String, Map, long, int, int)} over the points in {@code columns} of {@code data}. */
    public static BuildResult build(Dataset<Row> data, PointColumns columns, String algorithmId,
                                    Map<String, String> params, long seed, int treeDepth, int chunkSize) {
        return build(points(data, columns), algorithmId, params, seed, treeDepth, chunkSize);
    }

    /**
     * Twice the target coreset size ({@code m}, or {@code maxM} for adaptive {@code mean}), so a partition tree's
     * first merge halves two raw chunks' worth of points down to m.
     *
     * @throws IllegalArgumentException if neither parameter is present or parsable
     */
    public static int defaultChunkSize(Map<String, String> params) {
        String m = params.containsKey("m") ? params.get("m") : params.get("maxM");
        if (m == null) {
            throw new IllegalArgumentException("params carry neither m nor maxM; pass chunkSize explicitly");
        }
        try {
            return Math.toIntExact(2L * Integer.parseInt(m.trim()));
        } catch (NumberFormatException | ArithmeticException e) {
            throw new IllegalArgumentException("cannot derive chunk size from m/maxM '" + m + "'", e);
        }
    }

    /**
     * Points of {@code data}, decoded from Spark's internal rows (no external {@code Row}, no boxing); column types
     * are checked here, on the driver. Partitions and their order are those of {@code data}'s physical plan.
     */
    public static JavaRDD<WeightedPoint> points(Dataset<Row> data, PointColumns columns) {
        PointColumns.RowDecoder decoder = columns.decoder(data.schema());
        return data.queryExecution().toRdd().toJavaRDD().map(decoder::decode);
    }

    /**
     * Full-data cost of {@code centers} in one pass, centers broadcast once; per partition only a reused buffer of
     * point references is allocated. Partition sums are added in partition order, so the result is reproducible.
     *
     * @throws IllegalArgumentException if {@code centers} is empty
     */
    public static FullDataCost cost(JavaRDD<WeightedPoint> points, List<SparseVector> centers) {
        if (centers.isEmpty()) {
            throw new IllegalArgumentException("centers must not be empty");
        }
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(points.context());
        Broadcast<ArrayList<SparseVector>> shared = jsc.broadcast(new ArrayList<>(centers));
        try {
            List<PartitionCost> parts = new ArrayList<>(points.mapPartitionsWithIndex(
                    (index, it) -> List.of(PartitionCost.of(index, it, shared.value())).iterator(), false).collect());
            parts.sort(Comparator.comparingInt(PartitionCost::index));
            double cost = 0.0;
            double weight = 0.0;
            long count = 0;
            for (PartitionCost pc : parts) {
                cost += pc.cost();
                weight += pc.weight();
                count += pc.count();
            }
            return new FullDataCost(cost, weight, count);
        } finally {
            shared.destroy();
        }
    }

    /**
     * Builds a coreset, keeps the best of {@code restarts} k-means++ plus weighted Lloyd runs on it (by coreset cost)
     * and evaluates the winner on the full data. Restart {@code i} uses the generator derived from
     * {@code (seed, SOLVER_STREAM, 0, i)}. The input is read twice (build, evaluation); cache it if that is costly.
     *
     * @throws IllegalArgumentException if {@code k < 1}, {@code restarts < 1}, {@code maxIterations < 0} or the input
     *                                  has no positive-weight point
     */
    public static CoresetKMeans kMeans(JavaRDD<WeightedPoint> points, String algorithmId, Map<String, String> params,
                                       long seed, int treeDepth, int k, int restarts, int maxIterations) {
        if (k < 1 || restarts < 1 || maxIterations < 0) {
            throw new IllegalArgumentException(
                    "need k >= 1, restarts >= 1, maxIterations >= 0: " + k + ", " + restarts + ", " + maxIterations);
        }
        BuildResult built = build(points, algorithmId, params, seed, treeDepth);
        Coreset coreset = built.coreset();
        long solveStart = System.nanoTime();
        KMeansResult best = null;
        double bestCost = Double.POSITIVE_INFINITY;
        int bestRestart = -1;
        for (int i = 0; i < restarts; i++) {
            KMeansResult r = WeightedLloyd.cluster(coreset.pointSet(), k, maxIterations,
                    new SplittableRandom(MergeRandomness.derive(seed, SOLVER_STREAM, 0, i)), MetricsSink.NOOP);
            if (r.centers().isEmpty()) {
                throw new IllegalArgumentException("input has no positive-weight point");
            }
            double c = Cost.cost(coreset, r.centers());
            if (c < bestCost) {
                best = r;
                bestCost = c;
                bestRestart = i;
            }
        }
        long solveNanos = System.nanoTime() - solveStart;
        long evaluateStart = System.nanoTime();
        FullDataCost full = cost(points, best.centers());
        return new CoresetKMeans(best.centers(), bestCost, full, bestRestart, restarts, built, solveNanos,
                System.nanoTime() - evaluateStart);
    }

    private static BuildMetrics metrics(Partial last, int partitions, int chunkSize, int fanIn, int rounds,
                                        long wallNanos) {
        int width = Partial.STAT_WIDTH;
        List<StageStats> stages = new ArrayList<>(last.stage + 1);
        long shippedEncoded = 0;
        long shippedEstimated = 0;
        for (int s = 0; s <= last.stage; s++) {
            int at = s * width;
            String name = s == 0 ? "partition" : s == last.stage ? "final" : "round-" + s;
            StageStats st = new StageStats(s, name, last.stats[at], last.stats[at + 1], last.stats[at + 2],
                    last.stats[at + 3]);
            stages.add(st);
            if (s < last.stage) {
                shippedEncoded += st.encodedBytes();
                shippedEstimated += st.estimatedBytes();
            }
        }
        return new BuildMetrics(partitions, chunkSize, fanIn, rounds, last.treeLevels, last.depth, last.reduceOps,
                last.inputPoints, last.inputWeight, stages, shippedEncoded, shippedEstimated, wallNanos);
    }

    private record PartitionCost(int index, double cost, double weight, long count) implements Serializable {

        static PartitionCost of(int index, Iterator<WeightedPoint> it, List<SparseVector> centers) {
            List<WeightedPoint> chunk = new ArrayList<>(COST_CHUNK);
            double cost = 0.0;
            double weight = 0.0;
            long count = 0;
            while (it.hasNext()) {
                WeightedPoint p = it.next();
                chunk.add(p);
                weight += p.weight();
                count++;
                if (chunk.size() == COST_CHUNK) {
                    cost += Cost.cost(chunk, centers);
                    chunk.clear();
                }
            }
            cost += Cost.cost(chunk, centers); // 0 for an empty remainder
            return new PartitionCost(index, cost, weight, count);
        }
    }
}
