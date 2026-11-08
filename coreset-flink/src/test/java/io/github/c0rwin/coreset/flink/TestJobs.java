package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.connector.sink2.Sink;
import org.apache.flink.api.connector.sink2.SinkWriter;
import org.apache.flink.api.connector.sink2.WriterInitContext;
import org.apache.flink.configuration.CheckpointingOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.PipelineOptions;
import org.apache.flink.configuration.RestartStrategyOptions;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

/** Job plumbing shared by the MiniCluster tests: environment, deterministic input, failure injection, collection. */
final class TestJobs {

    static final Map<String, Queue<Object>> SINKS = new ConcurrentHashMap<>();
    static final Map<String, AtomicInteger> FAILURES = new ConcurrentHashMap<>();
    /** Per run, the first record index each subtask processed in a restarted attempt. */
    static final Map<String, Map<Integer, Long>> FIRST_AFTER_RESTART = new ConcurrentHashMap<>();

    private TestJobs() {
    }

    /**
     * Environment with Kryo disabled (any generic type fails job construction), 50 ms exactly-once checkpoints and a
     * fixed-delay restart strategy.
     */
    static StreamExecutionEnvironment env(int parallelism) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        Configuration conf = new Configuration();
        conf.set(PipelineOptions.GENERIC_TYPES, false);
        conf.set(CheckpointingOptions.CHECKPOINTING_INTERVAL, Duration.ofMillis(50));
        conf.set(RestartStrategyOptions.RESTART_STRATEGY, "fixed-delay");
        conf.set(RestartStrategyOptions.RESTART_STRATEGY_FIXED_DELAY_ATTEMPTS, 3);
        conf.set(RestartStrategyOptions.RESTART_STRATEGY_FIXED_DELAY_DELAY, Duration.ofMillis(50));
        env.configure(conf);
        env.setParallelism(parallelism);
        return env;
    }

    /**
     * Records {@code 0 .. count - 1} of {@code mixture}, split evenly over the subtasks; with
     * {@code eventTimeModulo > 0} record {@code i} has event time {@code i % eventTimeModulo} ms and watermarks follow
     * it. Subtask {@code failSubtask} throws on its {@code failAt}-th record of the first attempt (never if
     * {@code failAt < 0}); readers pause after {@code pauseAfter} records until a checkpoint
     * completes.
     */
    static DataStream<WeightedPoint> points(StreamExecutionEnvironment env, Mixture mixture, long count,
                                            long pauseAfter, String run, int failSubtask, long failAt,
                                            long eventTimeModulo) {
        WatermarkStrategy<Long> watermarks = eventTimeModulo > 0
                ? WatermarkStrategy.<Long>forMonotonousTimestamps()
                        .withTimestampAssigner((i, ts) -> i % eventTimeModulo)
                : WatermarkStrategy.noWatermarks();
        return env.fromSource(new DeterministicSource(count, pauseAfter), watermarks, "mixture")
                .map(new Generator(mixture, run, failSubtask, failAt))
                .returns(CoresetTypes.POINT);
    }

    static <T> List<T> collected(String run, Class<T> type) {
        return SINKS.getOrDefault(run, new ConcurrentLinkedQueue<>()).stream().map(type::cast)
                .collect(Collectors.toList());
    }

    /** Index -> point, with optional failure injection on the first attempt. */
    static final class Generator extends RichMapFunction<Long, WeightedPoint> {
        private static final long serialVersionUID = 1L;
        private final Mixture mixture;
        private final String run;
        private final int failSubtask;
        private final long failAt;
        private transient long seen;
        private transient int subtask;
        private transient int attempt;

        Generator(Mixture mixture, String run, int failSubtask, long failAt) {
            this.mixture = mixture;
            this.run = run;
            this.failSubtask = failSubtask;
            this.failAt = failAt;
        }

        @Override
        public void open(OpenContext openContext) {
            subtask = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
            attempt = getRuntimeContext().getTaskInfo().getAttemptNumber();
        }

        @Override
        public WeightedPoint map(Long index) {
            seen++;
            if (attempt == 0 && subtask == failSubtask && seen == failAt) {
                FAILURES.computeIfAbsent(run, r -> new AtomicInteger()).incrementAndGet();
                throw new IllegalStateException("injected failure after " + failAt + " records");
            }
            if (attempt > 0 && seen == 1) {
                FIRST_AFTER_RESTART.computeIfAbsent(run, r -> new ConcurrentHashMap<>()).put(subtask, index);
            }
            return mixture.point(index);
        }
    }

    /** Appends every value to {@link #SINKS} under the run name; replays after a restart append again. */
    static final class CollectSink<T> implements Sink<T> {
        private static final long serialVersionUID = 1L;
        private final String run;

        CollectSink(String run) {
            this.run = run;
        }

        @Override
        public SinkWriter<T> createWriter(WriterInitContext context) {
            Queue<Object> queue = SINKS.computeIfAbsent(run, r -> new ConcurrentLinkedQueue<>());
            return new SinkWriter<>() {
                @Override
                public void write(T element, Context context) {
                    queue.add(element);
                }

                @Override
                public void flush(boolean endOfInput) {
                }

                @Override
                public void close() {
                }
            };
        }
    }
}
