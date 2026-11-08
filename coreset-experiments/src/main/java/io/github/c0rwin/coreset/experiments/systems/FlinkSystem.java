package io.github.c0rwin.coreset.experiments.systems;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.experiments.config.CellSpec;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.SystemConfig;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import io.github.c0rwin.coreset.flink.CoresetConfig;
import io.github.c0rwin.coreset.flink.CoresetSnapshot;
import io.github.c0rwin.coreset.flink.CoresetTypes;
import io.github.c0rwin.coreset.flink.FlinkCoresets;
import java.net.URI;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.configuration.CheckpointingOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.PipelineOptions;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.configuration.RestartStrategyOptions;
import org.apache.flink.core.execution.JobClient;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;

/**
 * Flink 2.x: a MiniCluster in this JVM (one task manager with N slots) or a remote JobManager. Each repetition is one
 * bounded streaming job: N source subtasks read their shards, {@link FlinkCoresets#stream} builds the coreset, and the
 * final snapshot ({@code last = true}) is collected. Jobs are submitted through the REST endpoint in both modes, so
 * the single-host and cluster paths are the same code. There is no Flink-native k-means baseline: Flink ML's last
 * release (2.3.0) supports Flink 1.15 to 1.17 only.
 */
final class FlinkSystem implements EngineSystem {

    private static final String REPORTER = "experiments";

    private final CellSpec cell;
    private final PointSource src;
    private final SystemConfig sys;
    private final int n;
    private MiniCluster mini;
    private String host;
    private int port;

    FlinkSystem(CellSpec cell, PointSource src) {
        this.cell = cell;
        this.src = src;
        this.sys = cell.system();
        this.n = cell.parallelism();
    }

    @Override
    public String selfRole() {
        return sys.mode().equals("minicluster") ? "minicluster" : "client";
    }

    @Override
    public void start() throws Exception {
        switch (sys.mode()) {
            case "minicluster" -> {
                Configuration conf = new Configuration();
                conf.set(RestOptions.ADDRESS, "127.0.0.1");
                conf.set(RestOptions.BIND_ADDRESS, "127.0.0.1");
                conf.set(RestOptions.BIND_PORT, "0");
                conf.setString("metrics.reporter." + REPORTER + ".factory.class",
                        FlinkBytesReporter.Factory.class.getName());
                largeResults(conf);
                mini = new MiniCluster(new MiniClusterConfiguration.Builder().setConfiguration(conf)
                        .setNumTaskManagers(1).setNumSlotsPerTaskManager(n).build());
                mini.start();
                URI rest = mini.getRestAddress().get();
                host = rest.getHost();
                port = rest.getPort();
            }
            case "remote" -> {
                String jm = cell.cluster().flinkJobManager();
                if (jm == null) {
                    throw new IllegalArgumentException("flink mode remote needs cluster.flinkJobManager host:port");
                }
                host = jm.substring(0, jm.lastIndexOf(':'));
                port = Integer.parseInt(jm.substring(jm.lastIndexOf(':') + 1));
            }
            default -> throw new IllegalArgumentException("flink mode must be minicluster or remote: " + sys.mode());
        }
    }

    @Override
    public Map<Long, String> roles() {
        return Map.of();
    }

    private StreamExecutionEnvironment env(boolean checkpoints) {
        Configuration conf = new Configuration();
        conf.set(PipelineOptions.GENERIC_TYPES, false);
        largeResults(conf);
        long interval = sys.longOption("checkpointMs", 0);
        if (checkpoints && interval == 0) {
            interval = 200; // a fault needs checkpoints to restore from
        }
        if (interval > 0) {
            conf.set(CheckpointingOptions.CHECKPOINTING_INTERVAL, Duration.ofMillis(interval));
        }
        conf.set(RestartStrategyOptions.RESTART_STRATEGY, "fixed-delay");
        conf.set(RestartStrategyOptions.RESTART_STRATEGY_FIXED_DELAY_ATTEMPTS, 3);
        conf.set(RestartStrategyOptions.RESTART_STRATEGY_FIXED_DELAY_DELAY,
                Duration.ofMillis(sys.longOption("restartDelayMs", 100)));
        String[] jars = mini != null ? new String[0] : cell.cluster().flinkJars().toArray(new String[0]);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.createRemoteEnvironment(host, port, conf, jars);
        env.setParallelism(n);
        return env;
    }

    @Override
    public RunOutput run(int rep, long seed, FaultConfig fault) throws Exception {
        if (fault != null && !fault.kind().equals("task-failure")) {
            throw new IllegalArgumentException("flink supports fault kind task-failure, not " + fault.kind());
        }
        if (fault != null && mini == null) {
            throw new IllegalArgumentException("task-failure timing needs the in-JVM MiniCluster");
        }
        // Fault cells checkpoint in every repetition, so undisturbed and failed runs pay the same overhead.
        boolean checkpoints = cell.fault() != null;
        StreamExecutionEnvironment env = env(checkpoints);
        String run = cell.id() + "-rep" + rep;
        FlinkRows.CLOCK.remove(run);
        int failSubtask = fault == null ? -1 : n - 1;
        DataStream<WeightedPoint> points = env.fromSource(new FlinkRows.Indices(src.size(), n),
                        WatermarkStrategy.noWatermarks(), "rows")
                .map(new FlinkRows.Reader(src, cell.batchSize(), run, failSubtask, fault == null ? 0 : fault.at()))
                .returns(CoresetTypes.POINT)
                .name("read-rows");
        CoresetConfig.Builder cb = CoresetConfig.builder(cell.algorithm().id(), cell.algorithm().params(), seed)
                .chunksPerPartial(sys.intOption("chunksPerPartial", 8))
                .snapshotEveryPartials(sys.intOption("snapshotEveryPartials", 0))
                .compressSnapshots(sys.boolOption("compressSnapshots", false));
        if (sys.options().containsKey("chunkSize")) {
            cb.chunkSize(sys.intOption("chunkSize", 0));
        }
        CloseableIterator<CoresetSnapshot> it = FlinkCoresets.stream(points, cb.build()).filter(new LastOnly())
                .collectAsync();
        long t0 = System.nanoTime();
        JobClient client = env.executeAsync(run);
        CoresetSnapshot last = null;
        try (it) {
            while (it.hasNext()) {
                last = it.next();
            }
        }
        JobExecutionResult result = client.getJobExecutionResult().get();
        double seconds = (System.nanoTime() - t0) / 1e9;
        if (last == null) {
            throw new IllegalStateException("job " + result.getJobID() + " emitted no final snapshot");
        }
        boolean compressed = sys.boolOption("compressSnapshots", false) && last.compressed() != null;
        Coreset coreset = compressed ? last.compressed() : last.coreset();

        Map<String, Object> m = EngineSystem.map();
        m.put("job_id", result.getJobID().toString());
        m.put("records", last.records());
        m.put("partials", last.partials());
        m.put("levels", last.levels());
        m.put("compression_depth", last.depth() + (compressed ? 1 : 0));
        m.put("checkpoints", checkpoints || sys.longOption("checkpointMs", 0) > 0);
        if (mini != null) {
            m.put("bytes_out", FlinkBytesReporter.bytesOut(result.getJobID().toString()));
            m.put("bytes_summary", m.get("bytes_out"));
        } else {
            m.put("bytes_available", false);
        }
        m.put("bytes_ingest", 0L); // subtasks read their shard from storage

        Map<String, Object> f = new LinkedHashMap<>();
        if (fault != null) {
            Map<String, Long> c = FlinkRows.CLOCK.getOrDefault(run, Map.of());
            if (c.containsKey("failNanos")) {
                f.put("fail_at_s", (c.get("failNanos") - t0) / 1e9);
                f.put("fail_row", c.get("failIndex"));
                if (c.containsKey("resumeNanos")) {
                    f.put("restore_s", (c.get("resumeNanos") - c.get("failNanos")) / 1e9);
                    f.put("resume_row", c.get("resumeIndex"));
                    f.put("replayed_rows", c.get("failIndex") - c.get("resumeIndex"));
                }
            } else {
                f.put("fault_missed", true);
                f.put("fail_skipped", "no checkpoint completed before the subtask reached its failure point");
            }
        }
        return new RunOutput(coreset, null, seconds, Double.NaN, Double.NaN, m, f);
    }

    /**
     * The final snapshot is one record of up to tens of MB (union of the merge stage's levels); the collect sink,
     * REST and RPC limits default to 2, 100 and 10 MB.
     */
    private static void largeResults(Configuration conf) {
        conf.setString("collect-sink.batch-size.max", "1gb");
        conf.setString("rest.client.max-content-length", "1073741824");
        conf.setString("rest.server.max-content-length", "1073741824");
        conf.setString("pekko.framesize", "1073741824b");
    }

    /** Keeps the final snapshot only. */
    static final class LastOnly implements FilterFunction<CoresetSnapshot> {
        private static final long serialVersionUID = 1L;

        @Override
        public boolean filter(CoresetSnapshot s) {
            return s.last();
        }
    }

    @Override
    public void close() {
        if (mini != null) {
            try {
                mini.close();
            } catch (Exception e) {
                throw new IllegalStateException("MiniCluster did not shut down cleanly", e);
            }
        }
    }
}
