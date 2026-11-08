package io.github.c0rwin.coreset.experiments.systems;

import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import org.apache.flink.metrics.Counter;
import org.apache.flink.metrics.Metric;
import org.apache.flink.metrics.MetricConfig;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.reporter.MetricReporter;
import org.apache.flink.metrics.reporter.MetricReporterFactory;

/**
 * In-JVM reporter for a MiniCluster: sums the task-level {@code numBytesOut} counters (bytes serialized into network
 * buffers, i.e. leaving a task; chained operators exchange no bytes) per job. A task's counter is read when Flink
 * removes it at task end, and live counters are added on query.
 */
public final class FlinkBytesReporter implements MetricReporter {

    private static final Map<String, LongAdder> FINISHED = new ConcurrentHashMap<>();
    private static final Map<Counter, String> LIVE = new ConcurrentHashMap<>();

    /** Bytes out of every task of {@code jobId} so far. */
    public static long bytesOut(String jobId) {
        long sum = FINISHED.getOrDefault(jobId, new LongAdder()).sum();
        for (Map.Entry<Counter, String> e : LIVE.entrySet()) {
            if (e.getValue().equals(jobId)) {
                sum += e.getKey().getCount();
            }
        }
        return sum;
    }

    private static String taskJob(String name, Metric metric, MetricGroup group) {
        Map<String, String> v = group.getAllVariables();
        if (!"numBytesOut".equals(name) || !(metric instanceof Counter) || !v.containsKey("<task_name>")
                || v.containsKey("<operator_name>")) {
            return null;
        }
        return v.get("<job_id>");
    }

    @Override
    public void open(MetricConfig config) {
    }

    @Override
    public void close() {
    }

    @Override
    public void notifyOfAddedMetric(Metric metric, String name, MetricGroup group) {
        String job = taskJob(name, metric, group);
        if (job != null) {
            LIVE.put((Counter) metric, job);
        }
    }

    @Override
    public void notifyOfRemovedMetric(Metric metric, String name, MetricGroup group) {
        String job = taskJob(name, metric, group);
        if (job != null && LIVE.remove(metric) != null) {
            FINISHED.computeIfAbsent(job, j -> new LongAdder()).add(((Counter) metric).getCount());
        }
    }

    /** Registered through META-INF/services; select with {@code metrics.reporter.<name>.factory.class}. */
    public static final class Factory implements MetricReporterFactory {
        @Override
        public MetricReporter createMetricReporter(Properties properties) {
            return new FlinkBytesReporter();
        }
    }
}
