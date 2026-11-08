package io.github.c0rwin.coreset.flink;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.flink.api.common.JobID;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.connector.source.lib.NumberSequenceSource;
import org.apache.flink.api.java.typeutils.GenericTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.execution.JobClient;
import org.apache.flink.core.memory.DataOutputSerializer;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.metrics.Metric;
import org.apache.flink.runtime.testutils.InMemoryReporter;
import org.apache.flink.runtime.testutils.MiniClusterResourceConfiguration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.junit5.MiniClusterExtension;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * End-to-end jobs on a MiniCluster (1 task manager, 4 slots) with Kryo disabled: validity on a Gaussian mixture,
 * exactly-once recovery from an injected failure, tumbling windows, metrics and state-size bounds.
 */
class FlinkCoresetsTest {

    private static final int P = 4;
    private static final int PER = 20_000;
    private static final int N = P * PER;
    private static final int K = 6;
    private static final long SEED = 7;
    private static final int M = 400;
    private static final Map<String, String> SENSITIVITY = Map.of("k", "6", "m", "400");
    /**
     * Max empirical distortion accepted for sensitivity(k=6, m=400) on this mixture: the coreset-spark threshold for
     * the same compression depth (6). Measured values are printed by the tests.
     */
    private static final double MAX_DISTORTION = 1.20;
    /** Encoded bytes of one dense d = 8 point: dim + weight + nnz + 8 index deltas + 8 doubles. */
    private static final long POINT_BYTES = 1 + 8 + 1 + 8 + 64;

    private static final Mixture MIX = new Mixture(K, 8, 10.0, SEED, true);
    private static final InMemoryReporter REPORTER = InMemoryReporter.createWithRetainedMetrics();

    @RegisterExtension
    static final MiniClusterExtension CLUSTER = new MiniClusterExtension(
            new MiniClusterResourceConfiguration.Builder()
                    .setNumberTaskManagers(1)
                    .setNumberSlotsPerTaskManager(P)
                    .setConfiguration(REPORTER.addToConfiguration(new Configuration()))
                    .build());

    private static Coreset full;

    @BeforeAll
    static void data() {
        full = MIX.coreset(0, N);
    }

    private static CoresetConfig.Builder sensitivity() {
        return CoresetConfig.builder("sensitivity", SENSITIVITY, SEED).snapshotEveryPartials(4).compressSnapshots(true);
    }

    private record Run(JobID job, List<CoresetSnapshot> snapshots) {
        CoresetSnapshot last() {
            List<CoresetSnapshot> last = snapshots.stream().filter(CoresetSnapshot::last).toList();
            assertEquals(1, last.size(), "final snapshots");
            return last.get(0);
        }
    }

    private static Run stream(String run, CoresetConfig config, long pauseAfter, int failSubtask, long failAt)
            throws Exception {
        StreamExecutionEnvironment env = TestJobs.env(P);
        DataStream<WeightedPoint> points = TestJobs.points(env, MIX, N, pauseAfter, run, failSubtask, failAt, 0);
        FlinkCoresets.stream(points, config).sinkTo(new TestJobs.CollectSink<>(run)).setParallelism(1);
        JobClient client = env.executeAsync(run);
        client.getJobExecutionResult().get();
        return new Run(client.getJobID(), TestJobs.collected(run, CoresetSnapshot.class));
    }

    private static List<Candidate> candidates(Coreset data, Coreset coreset) {
        return CandidateSolutions.standard(data, coreset, K, 10, 3, new SplittableRandom(99));
    }

    private static double distortion(Coreset data, Coreset coreset) {
        return Distortion.evaluate(data, coreset, candidates(data, coreset)).max();
    }

    /** Gauge values of {@code metric} for every subtask of {@code operator}, by subtask index. */
    private static List<Long> gauges(JobID job, String operator, String metric) {
        Pattern p = Pattern.compile("\\." + Pattern.quote(operator) + "\\.(\\d+)\\.coreset\\." + metric + "$");
        TreeMap<Integer, Long> bySubtask = new TreeMap<>();
        for (Map.Entry<String, Metric> e : REPORTER.findMetrics(job, p.pattern()).entrySet()) {
            Matcher m = p.matcher(e.getKey());
            assertTrue(m.find(), e.getKey());
            bySubtask.put(Integer.parseInt(m.group(1)), ((Number) ((Gauge<?>) e.getValue()).getValue()).longValue());
        }
        return new ArrayList<>(bySubtask.values());
    }

    private static byte[] bytes(CoresetSnapshot s) throws Exception {
        DataOutputSerializer out = new DataOutputSerializer(1 << 16);
        CoresetSnapshotSerializer.INSTANCE.serialize(s, out);
        return out.getCopyOfBuffer();
    }

    @Test
    void sensitivityStreamIsValidAndExposesExactMetrics() throws Exception {
        Run r = stream("sensitivity", sensitivity().build(), -1, -1, -1);
        CoresetSnapshot fin = r.last();
        // 25 chunks of 800 per subtask: partials of 8, 8, 8 and 1 chunks; snapshots after every 4 partials.
        assertEquals(List.of(4L, 8L, 12L, 16L, 16L), r.snapshots().stream().map(CoresetSnapshot::partials).toList());
        assertEquals(N, fin.records());
        assertEquals(full.totalWeight(), fin.weight(), 1e-9 * full.totalWeight());
        assertEquals(P, fin.levels()); // 4 partials per subtask: one level-2 coreset each
        assertEquals(5, fin.depth());
        assertTrue(fin.coreset().size() <= P * M);
        assertTrue(fin.compressed().size() <= M);

        double union = distortion(full, fin.coreset());
        double compressed = distortion(full, fin.compressed());
        double oneShot = distortion(full, Algorithms.create("sensitivity", SENSITIVITY).compress(full,
                new SplittableRandom(SEED)));
        System.out.printf("flink sensitivity: union %d points distortion %.4f (depth %d), compressed %d points"
                        + " %.4f (depth %d), one-shot %.4f, weight rel err %.2e%n", fin.coreset().size(), union,
                fin.depth(), fin.compressed().size(), compressed, fin.depth() + 1, oneShot,
                IntegrityCheck.check(full, fin.compressed()).weightRelErr());
        assertTrue(union < MAX_DISTORTION, "union distortion " + union);
        assertTrue(compressed < MAX_DISTORTION, "compressed distortion " + compressed);
        assertTrue(compressed <= Math.pow(oneShot, fin.depth() + 1), "compressed " + compressed + " vs " + oneShot);

        assertEquals(List.of((long) PER, (long) PER, (long) PER, (long) PER), gauges(r.job(), "coreset-local",
                "recordsIn"));
        assertEquals(List.of(4L, 4L, 4L, 4L), gauges(r.job(), "coreset-local", "partialsEmitted"));
        assertEquals(List.of(25L, 25L, 25L, 25L), gauges(r.job(), "coreset-local", "chunks"));
        assertEquals(List.of(0L, 0L, 0L, 0L), gauges(r.job(), "coreset-local", "treeLevels"));
        assertEquals(List.of((long) N), gauges(r.job(), "coreset-merge", "recordsIn"));
        assertEquals(List.of(16L), gauges(r.job(), "coreset-merge", "partialsMerged"));
        assertEquals(List.of(5L), gauges(r.job(), "coreset-merge", "snapshotsEmitted"));
        assertEquals(List.of((long) P), gauges(r.job(), "coreset-merge", "treeLevels"));
        assertEquals(List.of(5L), gauges(r.job(), "coreset-merge", "treeDepth"));
        assertEquals(List.of((long) fin.coreset().size()), gauges(r.job(), "coreset-merge", "snapshotPoints"));

        // Local state peaks with a full buffer next to levels 0..2 of an 8-chunk epoch: O(chunk + m log(chunks)),
        // independent of the stream length. The bound below is that worst case in encoded bytes.
        int chunk = CoresetConfig.defaultChunkSize(SENSITIVITY);
        long localBound = 256 + POINT_BYTES * (2L * chunk + M * 3L);
        List<Long> localPeak = gauges(r.job(), "coreset-local", "peakStateBytes");
        List<Long> localLast = gauges(r.job(), "coreset-local", "stateBytes");
        for (int s = 0; s < P; s++) {
            assertTrue(localPeak.get(s) <= localBound, "local peak " + localPeak.get(s) + " > " + localBound);
            assertTrue(localPeak.get(s) >= POINT_BYTES * (chunk + M), "peak " + localPeak.get(s) + " too small");
            assertTrue(localLast.get(s) <= localPeak.get(s), "checkpointed " + localLast + " vs peak " + localPeak);
        }
        // Merge stage: per subtask at most floor(log2 4) + 1 levels of at most max(m, chunk) points.
        long mergeBound = 256 + P * (64 + 3L * chunk * POINT_BYTES);
        long mergePeak = gauges(r.job(), "coreset-merge", "peakStateBytes").get(0);
        System.out.printf("state bytes: local peak %s (bound %d), merge peak %d (bound %d), input %d%n", localPeak,
                localBound, mergePeak, mergeBound, N * POINT_BYTES);
        assertTrue(mergePeak <= mergeBound, "merge peak " + mergePeak + " > " + mergeBound);
    }

    @Test
    void meanCoresetConservesMomentsAndNeverUnderestimates() throws Exception {
        CoresetConfig c = CoresetConfig.builder("mean", Map.of("m", "200", "lloydIterations", "5"), SEED)
                .compressSnapshots(true).build();
        CoresetSnapshot fin = stream("mean", c, -1, -1, -1).last();
        for (Coreset coreset : List.of(fin.coreset(), fin.compressed())) {
            IntegrityCheck.Report r = IntegrityCheck.check(full, coreset);
            assertTrue(r.weightRelErr() < 1e-9, "weight " + r.weightRelErr());
            assertTrue(r.firstMomentRelErr() < 1e-9, "first moment " + r.firstMomentRelErr());
            assertTrue(r.secondMomentRelErr() < 1e-9, "second moment " + r.secondMomentRelErr());
            List<List<SparseVector>> queries = new ArrayList<>();
            for (Candidate cand : candidates(full, coreset)) {
                queries.add(cand.centers());
            }
            // Barger-Feldman Lemma 1 survives every streaming merge: the mean coreset never underestimates.
            assertEquals(0, IntegrityCheck.oneSidedViolations(full, coreset, queries, 1e-9));
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"sensitivity", "uniform"})
    void recoveryFromAnInjectedFailureIsBitIdentical(String algorithm) throws Exception {
        CoresetConfig c = algorithm.equals("uniform")
                ? CoresetConfig.builder("uniform", Map.of("m", "400"), SEED).snapshotEveryPartials(4)
                        .compressSnapshots(true).build()
                : sensitivity().build();
        String failed = "failed-" + algorithm;
        // Readers pause after 6,000 records until a checkpoint taken during the pause completes; subtask 1 throws on
        // its 12,000th record of the first attempt, so the job restores real mid-stream state.
        Run clean = stream("clean-" + algorithm, c, 6_000, -1, -1);
        Run recovered = stream(failed, c, 6_000, 1, 12_000);

        assertEquals(1, TestJobs.FAILURES.get(failed).get());
        Map<Integer, Long> restartedAt = TestJobs.FIRST_AFTER_RESTART.get(failed);
        // Subtasks that finished before the last checkpoint may stay finished; subtask 1 cannot have.
        long resumed = restartedAt.get(1);
        assertTrue(resumed >= PER + 6_000 && resumed < PER + 12_000, "subtask 1 resumed at record " + resumed);

        CoresetSnapshot expected = clean.last();
        CoresetSnapshot actual = recovered.last();
        assertEquals(expected, actual);
        assertArrayEquals(bytes(expected), bytes(actual));
        assertEquals(N, actual.records());
        assertEquals(full.totalWeight(), actual.weight(), 1e-9 * full.totalWeight());
        if (algorithm.equals("uniform")) {
            // Uniform sampling preserves total weight at every merge, so recovery must too.
            assertEquals(full.totalWeight(), actual.coreset().totalWeight(), 1e-9 * full.totalWeight());
            assertEquals(full.totalWeight(), actual.compressed().totalWeight(), 1e-9 * full.totalWeight());
        } else {
            // Same input, configuration and seed without pauses: timing never leaks into the result.
            assertEquals(expected, stream("unpaused", c, -1, -1, -1).last());
            assertNotEquals(expected.compressed(),
                    stream("other-seed", CoresetConfig.builder("sensitivity", SENSITIVITY, SEED + 1)
                            .snapshotEveryPartials(4).compressSnapshots(true).build(), -1, -1, -1).last().compressed());
        }
    }

    private static List<WindowCoreset> windows(String run, CoresetConfig c) throws Exception {
        StreamExecutionEnvironment env = TestJobs.env(P);
        // Event time i % PER: every subtask covers [0, PER) ms, so each 5 s window gets a partial from all four.
        DataStream<WeightedPoint> points = TestJobs.points(env, MIX, N, -1, run, -1, -1, PER);
        FlinkCoresets.tumblingWindows(points, c, Duration.ofSeconds(5)).sinkTo(new TestJobs.CollectSink<>(run))
                .setParallelism(1);
        JobClient client = env.executeAsync(run);
        client.getJobExecutionResult().get();
        List<WindowCoreset> out = new ArrayList<>(TestJobs.collected(run, WindowCoreset.class));
        out.sort(Comparator.comparingLong(WindowCoreset::windowStart));
        assertEquals(List.of((long) PER, (long) PER, (long) PER, (long) PER),
                gauges(client.getJobID(), "coreset-window-local", "recordsIn"));
        assertEquals(List.of(0L, 0L, 0L, 0L), gauges(client.getJobID(), "coreset-window-local", "lateRecordsDropped"));
        assertEquals(List.of(4L, 4L, 4L, 4L), gauges(client.getJobID(), "coreset-window-local", "partialsEmitted"));
        return out;
    }

    @Test
    void tumblingWindowsSummariseEveryWindowDeterministically() throws Exception {
        CoresetConfig c = CoresetConfig.builder("sensitivity", SENSITIVITY, SEED).build();
        List<WindowCoreset> out = windows("windows", c);
        assertEquals(4, out.size());
        for (int j = 0; j < 4; j++) {
            WindowCoreset w = out.get(j);
            assertEquals(5_000L * j, w.windowStart());
            assertEquals(5_000L * (j + 1), w.windowEnd());
            assertEquals(4 * 5_000, w.records());
            assertEquals(4, w.partials());
            assertEquals(4, w.depth()); // 7 chunks per subtask: 3 levels, reduced once; then the window merge
            assertTrue(w.coreset().size() <= M);
            List<WeightedPoint> inWindow = new ArrayList<>();
            for (int s = 0; s < P; s++) {
                long from = (long) s * PER + 5_000L * j;
                inWindow.addAll(MIX.coreset(from, from + 5_000).points());
            }
            Coreset data = Coreset.of(inWindow);
            assertEquals(data.totalWeight(), w.weight(), 1e-9 * data.totalWeight());
            double d = distortion(data, w.coreset());
            System.out.printf("flink window [%d, %d): %d points, distortion %.4f%n", w.windowStart(), w.windowEnd(),
                    w.coreset().size(), d);
            assertTrue(d < MAX_DISTORTION, "window distortion " + d);
        }
        assertEquals(out, windows("windows-again", c));
    }

    @Test
    void rejectsInputThatWouldBeSerializedWithKryo() {
        StreamExecutionEnvironment env = TestJobs.env(P);
        DataStream<WeightedPoint> kryo = env.fromSource(new NumberSequenceSource(0, 9),
                        WatermarkStrategy.noWatermarks(), "seq")
                .map(MIX::point)
                .returns(new GenericTypeInfo<>(WeightedPoint.class));
        CoresetConfig c = sensitivity().build();
        assertThrows(IllegalArgumentException.class, () -> FlinkCoresets.stream(kryo, c));
        assertThrows(IllegalArgumentException.class, () -> FlinkCoresets.tumblingWindows(kryo, c,
                Duration.ofSeconds(1)));
        // The test environment really forbids Kryo, so every job above ran without a generic type anywhere.
        kryo.sinkTo(new TestJobs.CollectSink<>("kryo"));
        Exception e = assertThrows(Exception.class, () -> env.executeAsync("kryo"));
        assertMessage(e, "Generic types have been disabled");
    }

    private static void assertMessage(Throwable t, String fragment) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (c.getMessage() != null && c.getMessage().contains(fragment)) {
                return;
            }
        }
        throw new AssertionError("no cause mentions '" + fragment + "'", t);
    }
}
