package io.github.c0rwin.coreset.core.metrics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class RecordingMetricsTest {

    @Test
    void concurrentCountsAndRecordsAreExact() throws Exception {
        RecordingMetrics m = new RecordingMetrics();
        int threads = 8;
        int perThread = 20_000;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                futures.add(pool.submit(() -> {
                    start.await();
                    for (int i = 0; i < perThread; i++) {
                        m.count("c", 1);
                        m.count("c2", 2);
                        m.record("d", 1.0);
                        m.record("v", i % 1000);
                    }
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> f : futures) {
                f.get(60, TimeUnit.SECONDS);
            }
        } finally {
            pool.shutdownNow();
        }
        MetricsSnapshot s = m.snapshot();
        assertEquals((long) threads * perThread, s.counters().get("c"));
        assertEquals(2L * threads * perThread, s.counters().get("c2"));
        Distribution d = s.distributions().get("d");
        assertEquals((long) threads * perThread, d.count());
        assertEquals((double) threads * perThread, d.sum());
        assertEquals(1.0, d.min());
        assertEquals(1.0, d.max());
        assertEquals(1.0, d.p99());
        Distribution v = s.distributions().get("v");
        assertEquals((long) threads * perThread, v.count());
        assertEquals(threads * (perThread / 1000) * (999.0 * 1000 / 2), v.sum());
        assertEquals(0.0, v.min());
        assertEquals(999.0, v.max());
        assertRelative(499, v.p50()); // each of 0..999 appears n/1000 times; rank n/2 is the last 499
    }

    private static void assertRelative(double expected, double actual) {
        assertTrue(Math.abs(actual - expected) <= RecordingMetrics.QUANTILE_RELATIVE_ERROR * Math.abs(expected),
                () -> "expected " + expected + " within relative error, got " + actual);
    }

    @Test
    void everyBucketMidpointIsWithinRelativeErrorOfItsMembers() {
        SplittableRandom rnd = new SplittableRandom(1);
        for (int i = 0; i < 100_000; i++) {
            double x = Math.scalb(1.0 + rnd.nextDouble(), rnd.nextInt(RecordingMetrics.MIN_EXPONENT,
                    RecordingMetrics.MAX_EXPONENT + 1));
            assertRelative(x, RecordingMetrics.midpointOf(RecordingMetrics.bucketOf(x)));
        }
    }

    @Test
    void quantilesAreWithinDocumentedRelativeError() {
        RecordingMetrics m = new RecordingMetrics();
        for (int i = 100; i >= 1; i--) {
            m.record("x", i);
        }
        Distribution d = m.snapshot().distributions().get("x");
        assertEquals(100, d.count());
        assertEquals(5050.0, d.sum());
        assertEquals(1.0, d.min());
        assertEquals(100.0, d.max());
        assertEquals(50.5, d.mean());
        // Nearest rank over 1..100: p-quantile is the value 100p.
        assertRelative(50, d.p50());
        assertRelative(95, d.p95());
        assertRelative(99, d.p99());
    }

    @Test
    void singleSampleQuantiles() {
        RecordingMetrics m = new RecordingMetrics();
        m.timeNanos("t", 42);
        Distribution d = m.snapshot().distributions().get("t");
        assertEquals(new Distribution(1, 42, 42, 42, 42, 42, 42, 42), d);
    }

    @Test
    void wideRangeOfMagnitudesAndSigns() {
        RecordingMetrics m = new RecordingMetrics();
        for (int i = 0; i < 60; i++) {
            m.record("w", -3.7e4);
        }
        for (int i = 0; i < 20; i++) {
            m.record("w", 0.0);
        }
        for (int i = 0; i < 14; i++) {
            m.record("w", 1e-300); // below the tracked range: counted as 0
        }
        for (int i = 0; i < 5; i++) {
            m.record("w", 1.234e15);
        }
        m.timeNanos("w", Long.MAX_VALUE);
        Distribution d = m.snapshot().distributions().get("w");
        assertEquals(100, d.count());
        assertEquals(-3.7e4, d.min());
        assertEquals((double) Long.MAX_VALUE, d.max());
        assertRelative(-3.7e4, d.p50());
        assertRelative(1.234e15, d.p95()); // ranks 61..94 are zeros, 95..99 are 1.234e15
        assertRelative(1.234e15, d.p99());

        for (double x = 1e-9; x < 1e18; x *= 3.3) {
            RecordingMetrics single = new RecordingMetrics();
            single.record("a", x);
            single.record("a", x * 1.0001);
            assertRelative(x, single.snapshot().distributions().get("a").p50());
        }
    }

    @Test
    void largeDistributionKeepsExactAggregatesAndBoundedQuantiles() {
        RecordingMetrics m = new RecordingMetrics();
        int n = 40_960;
        for (int i = 0; i < n; i++) {
            m.record("u", i);
        }
        Distribution d = m.snapshot().distributions().get("u");
        assertEquals(n, d.count());
        assertEquals((double) n * (n - 1) / 2, d.sum());
        assertEquals(0.0, d.min());
        assertEquals(n - 1.0, d.max());
        assertRelative(n / 2 - 1, d.p50());
        assertRelative(Math.ceil(0.99 * n) - 1, d.p99());
    }

    /** Review M1: instrumentation must not fail the data path, so a NaN sample is counted instead of thrown. */
    @Test
    void nanIsCountedSeparatelyAndResetClears() {
        RecordingMetrics m = new RecordingMetrics();
        m.record("x", 2.0);
        m.record("x", Double.NaN);
        m.record("x", Double.NaN);
        MetricsSnapshot withNan = m.snapshot();
        assertEquals(2L, withNan.counters().get("x.nan_samples"));
        assertEquals(1, withNan.distributions().get("x").count());
        assertEquals(2.0, withNan.distributions().get("x").sum());
        m.count("a", 3);
        m.record("b", 1);
        m.reset();
        MetricsSnapshot s = m.snapshot();
        assertTrue(s.counters().isEmpty());
        assertTrue(s.distributions().isEmpty());
    }

    @Test
    void jsonIsExactAndEscaped() {
        MetricsSnapshot s = new MetricsSnapshot(
                Map.of("b", 2L, "a\"q\\", 1L),
                Map.of("lat", new Distribution(2, 3.0, 1.0, 2.0, 1.5, 1.5, 1.95, 1.99),
                        "inf", new Distribution(1, Double.POSITIVE_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
        assertEquals("{\"counters\":{\"a\\\"q\\\\\":1,\"b\":2},\"distributions\":{"
                + "\"inf\":{\"count\":1,\"sum\":null,\"min\":0.0,\"max\":0.0,\"mean\":0.0,\"p50\":0.0,\"p95\":0.0,"
                + "\"p99\":0.0},"
                + "\"lat\":{\"count\":2,\"sum\":3.0,\"min\":1.0,\"max\":2.0,\"mean\":1.5,\"p50\":1.5,\"p95\":1.95,"
                + "\"p99\":1.99}}}", s.toJson());
        assertEquals("{\"counters\":{},\"distributions\":{}}", new MetricsSnapshot(Map.of(), Map.of()).toJson());
        assertEquals("{\"counters\":{\"\\u0001\\n\":0},\"distributions\":{}}",
                new MetricsSnapshot(Map.of("\u0001\n", 0L), Map.of()).toJson());
    }

    @Test
    void spanRecordsNonNegativeWallAndCpu() {
        RecordingMetrics m = new RecordingMetrics();
        double sink = 0;
        try (Span ignored = m.span("work")) {
            for (int i = 1; i < 200_000; i++) {
                sink += Math.sqrt(i);
            }
        }
        assertTrue(sink > 0);
        Map<String, Distribution> d = m.snapshot().distributions();
        Distribution wall = d.get("work" + MetricNames.WALL_NS_SUFFIX);
        Distribution cpu = d.get("work" + MetricNames.CPU_NS_SUFFIX);
        assertNotNull(wall);
        assertNotNull(cpu, "thread CPU time is supported on HotSpot");
        assertEquals(1, wall.count());
        assertTrue(wall.min() >= 0);
        assertTrue(cpu.min() >= 0);
        Distribution alloc = d.get("work" + MetricNames.ALLOC_BYTES_SUFFIX);
        if (alloc != null) {
            assertTrue(alloc.min() >= 0);
        }
    }

    @Test
    void spanClosesOnceAndNoopSpanIsShared() {
        RecordingMetrics m = new RecordingMetrics();
        Span s = m.span("x");
        s.close();
        s.close();
        assertEquals(1, m.snapshot().distributions().get("x" + MetricNames.WALL_NS_SUFFIX).count());
        assertSame(MetricsSink.NOOP.span("a"), Span.start(MetricsSink.NOOP, "b"));
    }
}
