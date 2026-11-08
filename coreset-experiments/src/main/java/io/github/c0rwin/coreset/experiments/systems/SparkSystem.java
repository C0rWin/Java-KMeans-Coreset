package io.github.c0rwin.coreset.experiments.systems;

import io.github.c0rwin.coreset.experiments.Jvm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.experiments.config.CellSpec;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.SystemConfig;
import io.github.c0rwin.coreset.experiments.data.ChunkReader;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import io.github.c0rwin.coreset.spark.BuildMetrics;
import io.github.c0rwin.coreset.spark.BuildResult;
import io.github.c0rwin.coreset.spark.SparkCoresets;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
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
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Spark in {@code local[N]} (one JVM, N task threads), {@code local-cluster[N,1,MB]} (a standalone master and workers
 * in this JVM, N single-core executor processes, so executors can be killed) or against a cluster master. Partition
 * {@code i} reads shard {@code i} of the source inside its task. {@code spark} builds the coreset with
 * {@link SparkCoresets#build}; {@code spark-mllib} fits MLlib's k-means (k-means|| init, weight column) on the full data
 * as a coreset-free baseline.
 */
final class SparkSystem implements EngineSystem {

    private static final String EXECUTOR_MAIN = "CoarseGrainedExecutorBackend";

    private final CellSpec cell;
    private final PointSource src;
    private final SystemConfig sys;
    private final int n;
    private final boolean mllib;
    private SparkSession spark;
    private JavaSparkContext jsc;
    private SparkStats stats;
    private double baselineTaskMillis = Double.NaN; // mean task run time of the last undisturbed repetition

    SparkSystem(CellSpec cell, PointSource src) {
        this.cell = cell;
        this.src = src;
        this.sys = cell.system();
        this.n = cell.parallelism();
        this.mllib = sys.type().equals("spark-mllib");
    }

    @Override
    public String selfRole() {
        return "driver";
    }

    private int executorMB() {
        return Math.max(512, cell.memory().workersMB() / n);
    }

    private int partitions() {
        return n * sys.intOption("partitionsPerWorker", 1);
    }

    @Override
    public void start() throws Exception {
        SparkSession.Builder b = SparkSession.builder().appName(cell.id())
                .config("spark.ui.enabled", "false")
                .config("spark.default.parallelism", Integer.toString(partitions()))
                .config("spark.sql.shuffle.partitions", Integer.toString(partitions()))
                .config("spark.task.maxFailures", "4");
        String extraJava = String.join(" ", cell.jvmFlags());
        switch (sys.mode()) {
            case "local" -> b.master("local[" + n + "]");
            case "local-cluster" -> {
                String home = System.getenv("SPARK_HOME");
                if (home == null || !new File(home, "jars").isDirectory()) {
                    throw new IllegalStateException("local-cluster needs SPARK_HOME with a jars/ directory (the"
                            + " orchestrator provides an empty one); executors use this JVM's class path");
                }
                b.master("local-cluster[" + n + ",1," + executorMB() + "]")
                        .config("spark.executor.memory", executorMB() + "m")
                        .config("spark.executor.extraClassPath", Jvm.classPath())
                        .config("spark.executor.extraJavaOptions", extraJava);
            }
            case "cluster" -> {
                if (cell.cluster().sparkMaster() == null) {
                    throw new IllegalArgumentException("spark mode cluster needs cluster.sparkMaster");
                }
                b.master(cell.cluster().sparkMaster())
                        .config("spark.executor.instances", Integer.toString(n))
                        .config("spark.executor.cores", "1")
                        .config("spark.cores.max", Integer.toString(n))
                        .config("spark.executor.memory", executorMB() + "m")
                        .config("spark.executor.extraJavaOptions", extraJava)
                        .config("spark.jars", String.join(",", cell.cluster().sparkJars()));
            }
            default -> throw new IllegalArgumentException("spark mode must be local, local-cluster or cluster: "
                    + sys.mode());
        }
        if (!sys.mode().equals("cluster")) {
            b.config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1");
        }
        spark = b.getOrCreate();
        jsc = JavaSparkContext.fromSparkContext(spark.sparkContext());
        jsc.setLogLevel("WARN");
        stats = new SparkStats();
        spark.sparkContext().addSparkListener(stats);
        awaitExecutors(180);
    }

    /** Waits until N executors are registered (no-op in local mode); returns the seconds waited. */
    private double awaitExecutors(int timeoutSeconds) throws InterruptedException {
        if (sys.mode().equals("local")) {
            return 0.0;
        }
        long t0 = System.nanoTime();
        long deadline = t0 + TimeUnit.SECONDS.toNanos(timeoutSeconds);
        while (spark.sparkContext().getExecutorMemoryStatus().size() < n + 1) { // the driver is listed too
            if (System.nanoTime() > deadline) {
                throw new IllegalStateException("only " + (spark.sparkContext().getExecutorMemoryStatus().size() - 1)
                        + " of " + n + " executors registered");
            }
            Thread.sleep(20);
        }
        return (System.nanoTime() - t0) / 1e9;
    }

    @Override
    public Map<Long, String> roles() {
        Map<Long, String> r = new LinkedHashMap<>();
        executors().forEach(p -> r.put(p.pid(), "executor"));
        return r;
    }

    private static List<ProcessHandle> executors() {
        List<ProcessHandle> out = new ArrayList<>();
        ProcessHandle.current().descendants().forEach(p -> {
            Optional<String> cmd = p.info().commandLine();
            if (cmd.isPresent() && cmd.get().contains(EXECUTOR_MAIN)) {
                out.add(p);
            }
        });
        return out;
    }

    /** Partition {@code i} streams shard {@code i}; the source is serialized into the task, not the rows. */
    private JavaRDD<WeightedPoint> points() {
        int p = partitions();
        List<long[]> ranges = new ArrayList<>(p);
        for (int i = 0; i < p; i++) {
            ranges.add(new long[] {PointSource.shardStart(src.size(), i, p), PointSource.shardStart(src.size(), i + 1, p)});
        }
        return jsc.parallelize(ranges, p).mapPartitions(new ReadRanges(src, cell.batchSize(),
                sys.longOption("readPauseMicrosPerChunk", 0)));
    }

    @Override
    public RunOutput run(int rep, long seed, FaultConfig fault) throws Exception {
        double executorWait = awaitExecutors(180); // a killed executor's replacement must be up before timing
        String group = cell.id() + "-rep" + rep;
        jsc.setJobGroup(group, group, false);
        Map<String, Object> f = new java.util.concurrent.ConcurrentHashMap<>();
        long start = System.currentTimeMillis();
        if (fault != null) {
            if (!fault.kind().equals("kill-executor") || !sys.mode().equals("local-cluster")) {
                throw new IllegalArgumentException("spark supports kill-executor in local-cluster mode only");
            }
            if (Double.isNaN(baselineTaskMillis)) {
                throw new IllegalStateException("kill-executor needs an undisturbed repetition first");
            }
            // timed from the first task start, so it lands inside a task even with one partition per executor
            long delay = (long) (fault.at() * baselineTaskMillis);
            f.put("kill_delay_ms", delay);
            stats.armKill(group, delay, (executor, done) -> killExecutor(executor, done, f, start));
        }
        Map<String, Object> m = EngineSystem.map();
        m.put("partitions", partitions());
        m.put("master", spark.sparkContext().master());
        m.put("executor_wait_s", executorWait);
        try {
            long t0 = System.nanoTime();
            if (mllib) {
                KMeansModel model = new KMeans().setK(cell.k()).setMaxIter(cell.solver().iterations()).setSeed(seed)
                        .setInitMode("k-means||").setTol(Double.parseDouble(sys.option("tol", "1e-4")))
                        .setFeaturesCol("features").setWeightCol("weight").fit(frame());
                double fit = (System.nanoTime() - t0) / 1e9;
                List<SparseVector> centers = new ArrayList<>();
                for (Vector c : model.clusterCenters()) {
                    centers.add(SparseVector.dense(c.toArray()));
                }
                m.put("mllib_iterations", model.summary().numIter());
                return finish(group, null, centers, Double.NaN, fit, m, f, fault != null);
            }
            int depth = sys.intOption("treeDepth", 2);
            int chunk = sys.intOption("chunkSize", SparkCoresets.defaultChunkSize(cell.algorithm().params()));
            BuildResult built = SparkCoresets.build(points(), cell.algorithm().id(), cell.algorithm().params(), seed,
                    depth, chunk);
            double build = (System.nanoTime() - t0) / 1e9;
            BuildMetrics bm = built.metrics();
            m.put("tree_depth", depth);
            m.put("chunk_size", chunk);
            m.put("fan_in", bm.fanIn());
            m.put("reduce_rounds", bm.reduceRounds());
            m.put("compression_depth", bm.compressionDepth());
            m.put("bytes_partials_encoded", bm.shippedEncodedBytes());
            return finish(group, built.coreset(), null, build, Double.NaN, m, f, fault != null);
        } finally {
            stats.disarm();
        }
    }

    private RunOutput finish(String group, io.github.c0rwin.coreset.core.Coreset coreset, List<SparseVector> centers,
                             double build, double solve, Map<String, Object> m, Map<String, Object> f,
                             boolean armed)
            throws java.util.concurrent.TimeoutException, InterruptedException {
        spark.sparkContext().listenerBus().waitUntilEmpty();
        SparkStats.Agg a = stats.take(group);
        m.put("tasks", a.tasks);
        m.put("failed_tasks", a.failedTasks);
        m.put("stages", a.stages.size());
        m.put("executor_cpu_s", a.executorCpuNanos / 1e9);
        m.put("executor_run_s", a.executorRunMillis / 1e3);
        m.put("gc_s", a.gcMillis / 1e3);
        m.put("bytes_shuffle_write", a.shuffleWrite);
        m.put("bytes_shuffle_read", a.shuffleRead);
        m.put("bytes_result", a.resultSize);
        m.put("bytes_summary", a.shuffleRead + a.resultSize);
        m.put("bytes_ingest", 0L); // tasks read their shard from storage
        if (armed) {
            stats.disarm();
            for (int i = 0; i < 100 && stats.fired() && !f.containsKey("kill_epoch_ms") && !f.containsKey("kill_error");
                 i++) {
                java.util.concurrent.locks.LockSupport.parkNanos(50_000_000L); // kill action still recording
            }
            Long killed = (Long) f.get("kill_epoch_ms");
            if (killed != null) {
                awaitExecutors(120);
                spark.sparkContext().listenerBus().waitUntilEmpty();
                stats.executorEventsAfter(killed).forEach(f::put);
                f.put("tasks_failed", a.failedTasks);
                if (a.failedTasks == 0) { // the victim's task finished before the signal landed
                    f.put("fault_missed", true);
                }
            } else {
                f.put("fault_missed", true);
                f.putIfAbsent("kill_skipped", "no task of the repetition was running at the kill time");
            }
            f.remove("kill_epoch_ms");
        } else if (coreset != null && a.tasks > 0) {
            baselineTaskMillis = (double) a.executorRunMillis / a.tasks;
        }
        return new RunOutput(coreset, centers, build, Double.NaN, solve, m, f);
    }

    /** SIGKILL the executor process with this id (its command line carries {@code --executor-id <id>}). */
    private void killExecutor(String executorId, long finishedTasks, Map<String, Object> f, long start) {
        ProcessHandle victim = null;
        for (ProcessHandle p : executors()) {
            if (p.info().commandLine().orElse("").contains("--executor-id " + executorId + " ")) {
                victim = p;
            }
        }
        if (victim == null) {
            f.put("kill_error", "no process found for executor " + executorId);
            return;
        }
        long now = System.currentTimeMillis();
        victim.destroyForcibly();
        f.put("kill_epoch_ms", now);
        f.put("kill_at_s", (now - start) / 1e3);
        f.put("kill_after_tasks", finishedTasks);
        f.put("victim_executor", executorId);
        f.put("victim_pid", victim.pid());
    }

    private Dataset<Row> frame() {
        StructType schema = new StructType(new StructField[] {
                new StructField("features", SQLDataTypes.VectorType(), false, Metadata.empty()),
                new StructField("weight", DataTypes.DoubleType, false, Metadata.empty())});
        return spark.createDataFrame(points().map(new ToRow()), schema);
    }

    @Override
    public void close() {
        if (spark != null) {
            spark.stop();
        }
    }

    /**
     * Streams the row ranges of one partition. {@code pauseMicros > 0} sleeps after every chunk: a test-only knob
     * that makes tasks long enough for a kill to land inside one on a loaded host.
     */
    static final class ReadRanges implements FlatMapFunction<Iterator<long[]>, WeightedPoint> {
        private static final long serialVersionUID = 1L;
        private final PointSource src;
        private final int chunkRows;
        private final long pauseMicros;

        ReadRanges(PointSource src, int chunkRows, long pauseMicros) {
            this.src = src;
            this.chunkRows = chunkRows;
            this.pauseMicros = pauseMicros;
        }

        @Override
        public Iterator<WeightedPoint> call(Iterator<long[]> ranges) {
            return new Iterator<>() {
                private Iterator<WeightedPoint> current = java.util.Collections.emptyIterator();
                private long emitted;

                @Override
                public boolean hasNext() {
                    while (!current.hasNext() && ranges.hasNext()) {
                        long[] r = ranges.next();
                        try {
                            current = ChunkReader.points(src.open(r[0], r[1], chunkRows));
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    }
                    return current.hasNext();
                }

                @Override
                public WeightedPoint next() {
                    hasNext();
                    if (pauseMicros > 0 && ++emitted % chunkRows == 0) {
                        java.util.concurrent.locks.LockSupport.parkNanos(pauseMicros * 1000);
                    }
                    return current.next();
                }
            };
        }
    }

    /** Point to an MLlib row: dense features plus weight. */
    static final class ToRow implements Function<WeightedPoint, Row> {
        private static final long serialVersionUID = 1L;

        @Override
        public Row call(WeightedPoint p) {
            SparseVector v = p.vector();
            if (2L * v.nnz() >= v.dim()) {
                return RowFactory.create(Vectors.dense(v.toDense()), p.weight());
            }
            int[] idx = new int[v.nnz()];
            double[] val = new double[v.nnz()];
            for (int e = 0; e < idx.length; e++) {
                idx[e] = v.indexAt(e);
                val[e] = v.valueAt(e);
            }
            return RowFactory.create(Vectors.sparse(v.dim(), idx, val), p.weight());
        }
    }
}
