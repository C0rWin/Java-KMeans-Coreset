package io.github.c0rwin.coreset.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.Algorithms;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.core.validate.ErrorBudget;
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.apache.spark.SparkJobInfo;
import org.apache.spark.SparkStageInfo;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkStatusTracker;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class SparkCoresetsTest {

    private static final int K = 6;
    private static final int D = 8;
    private static final int PARTITIONS = 8;
    private static final int PER_PARTITION = 10_000;
    private static final long SEED = 7;
    private static final Map<String, String> SENSITIVITY = Map.of("k", "6", "m", "400");

    /**
     * Max empirical distortion accepted for sensitivity(k=6, m=400) after six compounding levels (measured: 1.14, vs
     * 1.06 for one compression of all data on one machine).
     */
    private static final double MAX_DISTORTION = 1.20;
    /** Coreset k-means must land within this factor of MLlib k-means|| on the full data. */
    private static final double MAX_COST_FACTOR_VS_MLLIB = 1.05;

    private static SparkSession spark;
    private static JavaRDD<WeightedPoint> data;
    private static Coreset full;

    @BeforeAll
    static void setUp() {
        spark = LocalSpark.session();
        double[][] centers = Mixture.centers(K, D, 10.0, SEED);
        data = Mixture.rdd(LocalSpark.jsc(), centers, PARTITIONS, PER_PARTITION, SEED, true).cache();
        full = Coreset.of(data.collect());
    }

    @AfterAll
    static void tearDown() {
        data.unpersist();
    }

    private static List<Candidate> candidates(Coreset coreset) {
        return CandidateSolutions.standard(full, coreset, K, 10, 3, new SplittableRandom(99));
    }

    @Test
    void sensitivityCoresetHasLowDistortionAndConservesWeight() {
        BuildResult r = SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 2);
        Coreset c = r.coreset();
        Distortion.Report report = Distortion.evaluate(full, c, candidates(c));
        IntegrityCheck.Report integrity = IntegrityCheck.check(full, c);
        // Same algorithm and m applied once to all data on one machine: the floor the distributed tree compounds on.
        Coreset oneShot = Algorithms.create("sensitivity", SENSITIVITY).compress(full, new SplittableRandom(SEED));
        double oneShotMax = Distortion.evaluate(full, oneShot, candidates(oneShot)).max();
        System.out.printf("sensitivity spark coreset: %d points, distortion max %.4f (%s) mean %.4f; one-shot max"
                        + " %.4f; weight rel err %.2e, depth %d, metrics %s%n", c.size(), report.max(),
                report.argmaxGenerator(), report.mean(), oneShotMax, integrity.weightRelErr(),
                r.metrics().compressionDepth(), r.metrics());

        assertTrue(c.size() <= 400);
        assertTrue(report.max() < MAX_DISTORTION, "distortion " + report.max());
        // No worse than per-level errors of the one-shot size compounding over the tree (research notes §6a item 6).
        assertTrue(report.max() <= Math.pow(oneShotMax, r.metrics().compressionDepth()));
        assertTrue(integrity.weightRelErr() < 0.05, "weight rel err " + integrity.weightRelErr());
        assertEquals(PARTITIONS * PER_PARTITION, r.metrics().inputPoints());
        assertEquals(full.totalWeight(), r.metrics().inputWeight(), 1e-9 * full.totalWeight());
    }

    @Test
    void uniformAndMeanCoresetsConserveWeightExactly() {
        Coreset uniform = SparkCoresets.build(data, "uniform", Map.of("m", "400"), SEED, 2).coreset();
        assertTrue(IntegrityCheck.check(full, uniform).weightRelErr() < 1e-9);

        Coreset mean = SparkCoresets.build(data, "mean", Map.of("m", "200", "lloydIterations", "5"), SEED, 2)
                .coreset();
        IntegrityCheck.Report r = IntegrityCheck.check(full, mean);
        assertTrue(r.weightRelErr() < 1e-9, "weight " + r.weightRelErr());
        assertTrue(r.firstMomentRelErr() < 1e-9, "first moment " + r.firstMomentRelErr());
        assertTrue(r.secondMomentRelErr() < 1e-9, "second moment " + r.secondMomentRelErr());
        List<List<SparseVector>> queries = new ArrayList<>();
        for (Candidate cand : candidates(mean)) {
            queries.add(cand.centers());
        }
        // Barger-Feldman Lemma 1 survives every distributed merge: the mean coreset never underestimates.
        assertEquals(0, IntegrityCheck.oneSidedViolations(full, mean, queries, 1e-9));
    }

    @Test
    void retriedTasksReproduceIdenticalCoresets() throws InterruptedException {
        BuildResult first = SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 3);
        BuildResult again = SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 3);
        // Even partitions fail their first attempt after reading some points; Spark reruns them.
        JavaRDD<WeightedPoint> flaky = data.map(p -> {
            TaskContext tc = TaskContext.get();
            if (tc.attemptNumber() == 0 && tc.partitionId() % 2 == 0 && p.weight() > 1.9) {
                throw new IllegalStateException("injected failure in partition " + tc.partitionId());
            }
            return p;
        });
        LocalSpark.jsc().setJobGroup("flaky-build", "first attempts of even partitions fail", false);
        BuildResult retried;
        try {
            retried = SparkCoresets.build(flaky, "sensitivity", SENSITIVITY, SEED, 3);
        } finally {
            LocalSpark.jsc().clearJobGroup();
        }
        assertEquals(PARTITIONS / 2, failedTasks("flaky-build"));

        assertEquals(first.coreset(), again.coreset());
        assertEquals(first.coreset(), retried.coreset());
        assertEquals(withoutTime(first.metrics()), withoutTime(retried.metrics()));
        assertNotEquals(first.coreset(), SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED + 1, 3).coreset());
    }

    /** Failed task attempts of a job group; the status store is fed asynchronously, so wait for it briefly. */
    private static int failedTasks(String group) throws InterruptedException {
        JavaSparkStatusTracker tracker = LocalSpark.jsc().statusTracker();
        long deadline = System.nanoTime() + 10_000_000_000L;
        int failed = 0;
        while (System.nanoTime() < deadline) {
            failed = 0;
            for (int job : tracker.getJobIdsForGroup(group)) {
                SparkJobInfo info = tracker.getJobInfo(job);
                for (int stage : info == null ? new int[0] : info.stageIds()) {
                    SparkStageInfo s = tracker.getStageInfo(stage);
                    failed += s == null ? 0 : s.numFailedTasks();
                }
            }
            if (failed >= PARTITIONS / 2) {
                break;
            }
            Thread.sleep(20);
        }
        return failed;
    }

    private static BuildMetrics withoutTime(BuildMetrics m) {
        return new BuildMetrics(m.partitions(), m.chunkSize(), m.fanIn(), m.reduceRounds(), m.partitionTreeLevels(),
                m.compressionDepth(), m.reduceOps(), m.inputPoints(), m.inputWeight(), m.stages(),
                m.shippedEncodedBytes(), m.shippedEstimatedBytes(), 0);
    }

    @Test
    void aggregationTreeMetricsAreConsistent() {
        BuildMetrics deep = SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 3).metrics();
        assertEquals(2, deep.fanIn()); // ceil(8^(1/3))
        assertEquals(2, deep.reduceRounds());
        assertEquals(List.of(8L, 4L, 2L, 1L), deep.stages().stream().map(StageStats::partials).toList());
        assertEquals(List.of("partition", "round-1", "round-2", "final"),
                deep.stages().stream().map(StageStats::name).toList());
        assertEquals(deep.partitionTreeLevels() + 3, deep.compressionDepth());
        long shipped = 0;
        for (StageStats s : deep.stages().subList(0, 3)) {
            shipped += s.encodedBytes();
            assertTrue(s.points() <= s.partials() * 400);
        }
        assertEquals(shipped, deep.shippedEncodedBytes());
        assertTrue(deep.shippedEstimatedBytes() > deep.shippedEncodedBytes());
        // 10k points per partition in chunks of 800: 13 batches -> 13 - bitCount(13) merges + 1 final reduce
        assertEquals(800, deep.chunkSize());
        assertEquals(4, deep.partitionTreeLevels());
        assertEquals(PARTITIONS * (13 - 3 + 1) + 4 + 2 + 1, deep.reduceOps());
        ErrorBudget budget = deep.errorBudget(0.2, 0.01);
        assertEquals(deep.compressionDepth(), budget.levels());
        assertEquals(Math.pow(1.2, 1.0 / deep.compressionDepth()) - 1, budget.epsPerLevel(), 1e-12);

        BuildMetrics flat = SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 1).metrics();
        assertEquals(0, flat.reduceRounds());
        assertEquals(List.of(8L, 1L), flat.stages().stream().map(StageStats::partials).toList());
        assertEquals(flat.stages().get(0).encodedBytes(), flat.shippedEncodedBytes());
        System.out.printf("tree depth 3 shipped %d B (heap %d B); depth 1 shipped %d B; stages %s%n",
                deep.shippedEncodedBytes(), deep.shippedEstimatedBytes(), flat.shippedEncodedBytes(), deep.stages());
    }

    @Test
    void datasetInputsMatchRddInput() {
        Coreset expected = SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 2).coreset();

        StructType sparseSchema = new StructType()
                .add("idx", DataTypes.createArrayType(DataTypes.IntegerType))
                .add("val", DataTypes.createArrayType(DataTypes.DoubleType))
                .add("dim", DataTypes.IntegerType)
                .add("w", DataTypes.DoubleType);
        Dataset<Row> sparse = spark.createDataFrame(data.map(p -> {
            SparseVector v = p.vector();
            List<Integer> idx = new ArrayList<>();
            List<Double> val = new ArrayList<>();
            for (int k = 0; k < v.nnz(); k++) {
                idx.add(v.indexAt(k));
                val.add(v.valueAt(k));
            }
            return RowFactory.create(idx.toArray(), val.toArray(), v.dim(), p.weight());
        }), sparseSchema);
        assertEquals(expected, SparkCoresets.build(sparse, PointColumns.sparse("idx", "val", "dim").withWeight("w"),
                "sensitivity", SENSITIVITY, SEED, 2).coreset());

        StructType arraySchema = new StructType()
                .add("features", DataTypes.createArrayType(DataTypes.DoubleType))
                .add("w", DataTypes.DoubleType);
        Dataset<Row> dense = spark.createDataFrame(data.map(p -> {
            double[] x = p.vector().toDense();
            Double[] boxed = new Double[x.length];
            for (int j = 0; j < x.length; j++) {
                boxed[j] = x[j];
            }
            return RowFactory.create(boxed, p.weight());
        }), arraySchema);
        assertEquals(expected, SparkCoresets.build(dense, PointColumns.vector("features").withWeight("w"),
                "sensitivity", SENSITIVITY, SEED, 2).coreset());

        assertEquals(expected, SparkCoresets.build(mlFrame(), PointColumns.vector("features").withWeight("weight"),
                "sensitivity", SENSITIVITY, SEED, 2).coreset());

        StructType mlSparseSchema = new StructType()
                .add("features", SQLDataTypes.VectorType())
                .add("w", DataTypes.DoubleType);
        Dataset<Row> mlSparse = spark.createDataFrame(data.map(p -> {
            SparseVector v = p.vector();
            int[] idx = new int[v.nnz()];
            double[] val = new double[v.nnz()];
            for (int k = 0; k < idx.length; k++) {
                idx[k] = v.indexAt(k);
                val[k] = v.valueAt(k);
            }
            return RowFactory.create(Vectors.sparse(v.dim(), idx, val), p.weight());
        }), mlSparseSchema);
        assertEquals(expected, SparkCoresets.build(mlSparse, PointColumns.vector("features").withWeight("w"),
                "sensitivity", SENSITIVITY, SEED, 2).coreset());

        assertThrows(IllegalArgumentException.class, () -> SparkCoresets.points(dense, PointColumns.vector("w")));
        Dataset<Row> withNull = spark.createDataFrame(
                List.of(RowFactory.create(new Double[] {1.0, null}, 1.0)), arraySchema);
        Exception e = assertThrows(Exception.class,
                () -> SparkCoresets.points(withNull, PointColumns.vector("features")).collect());
        assertTrue(String.valueOf(e.getMessage()).contains("null array element"), e.getMessage());
    }

    /** {@code data} as (features: ML vector, weight: double), same partitions and order. */
    private static Dataset<Row> mlFrame() {
        StructType schema = new StructType()
                .add("features", SQLDataTypes.VectorType())
                .add("weight", DataTypes.DoubleType);
        return spark.createDataFrame(
                data.map(p -> RowFactory.create(Vectors.dense(p.vector().toDense()), p.weight())), schema);
    }

    @Test
    void coresetKMeansIsCompetitiveWithMllib() {
        CoresetKMeans ours = SparkCoresets.kMeans(data, "sensitivity", SENSITIVITY, SEED, 2, K, 10, 30);

        Dataset<Row> frame = mlFrame().cache();
        long mllibStart = System.nanoTime();
        KMeansModel model = new KMeans().setK(K).setSeed(SEED).setMaxIter(30).setWeightCol("weight").fit(frame);
        long mllibNanos = System.nanoTime() - mllibStart;
        frame.unpersist();
        List<SparseVector> mllibCenters = new ArrayList<>();
        for (Vector v : model.clusterCenters()) {
            mllibCenters.add(SparseVector.dense(v.toArray()));
        }
        FullDataCost mllib = SparkCoresets.cost(data, mllibCenters);

        System.out.printf("k-means on %d points: coreset cost %.6e (build %.0f ms, solve %.0f ms, eval %.0f ms,"
                        + " deployed distortion %.4f) vs MLlib k-means|| cost %.6e (fit %.0f ms); ratio %.4f%n",
                ours.full().count(), ours.full().cost(), ours.build().metrics().wallNanos() / 1e6,
                ours.solveNanos() / 1e6, ours.evaluateNanos() / 1e6, ours.deployedDistortion(), mllib.cost(),
                mllibNanos / 1e6, ours.full().cost() / mllib.cost());

        assertEquals(K, ours.centers().size());
        assertEquals(PARTITIONS * PER_PARTITION, ours.full().count());
        assertEquals(full.totalWeight(), ours.full().weight(), 1e-9 * full.totalWeight());
        assertTrue(ours.full().cost() <= MAX_COST_FACTOR_VS_MLLIB * mllib.cost(),
                "coreset " + ours.full().cost() + " vs mllib " + mllib.cost());
        assertTrue(ours.deployedDistortion() < MAX_DISTORTION);
        assertEquals(ours.centers(),
                SparkCoresets.kMeans(data, "sensitivity", SENSITIVITY, SEED, 2, K, 10, 30).centers());
    }

    @Test
    void emptyInputsGiveEmptyCoresets() {
        BuildResult none = SparkCoresets.build(LocalSpark.jsc().<WeightedPoint>emptyRDD(), "uniform", Map.of("m", "5"),
                SEED, 2);
        assertTrue(none.coreset().isEmpty());
        assertEquals(0, none.metrics().partitions());

        BuildResult emptyPartitions = SparkCoresets.build(
                LocalSpark.jsc().parallelize(List.<WeightedPoint>of(), 3), "sensitivity", SENSITIVITY, SEED, 2);
        assertTrue(emptyPartitions.coreset().isEmpty());
        assertEquals(0, emptyPartitions.metrics().inputPoints());
        assertEquals(0, emptyPartitions.metrics().reduceOps());
    }

    @Test
    void invalidArgumentsFailOnTheDriver() {
        assertThrows(IllegalArgumentException.class,
                () -> SparkCoresets.build(data, "sensitivity", SENSITIVITY, SEED, 0));
        assertThrows(IllegalArgumentException.class,
                () -> SparkCoresets.build(data, "nope", Map.of("m", "3"), SEED, 2));
        assertThrows(IllegalArgumentException.class,
                () -> SparkCoresets.build(data, "uniform", Map.of("m", "3"), SEED, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> SparkCoresets.defaultChunkSize(Map.of("k", "3")));
        assertEquals(64, SparkCoresets.defaultChunkSize(Map.of("k", "3", "eps", "0.1", "maxM", "32")));
        assertThrows(IllegalArgumentException.class,
                () -> SparkCoresets.kMeans(data, "uniform", Map.of("m", "3"), SEED, 2, 0, 1, 1));
    }
}
