package io.github.c0rwin.coreset.experiments.metrics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ProcessSamplerTest {

    @Test
    void parsesPsCpuTimes() {
        assertEquals(1.23, ProcessSampler.cpuTime("0:01.23"), 1e-9);
        assertEquals(62.5, ProcessSampler.cpuTime("1:02.50"), 1e-9);
        assertEquals(3723.0, ProcessSampler.cpuTime("01:02:03"), 1e-9);
        assertEquals(86400 + 61.0, ProcessSampler.cpuTime("1-00:01:01"), 1e-9);
    }

    @Test
    void reportsThisJvmWithCpuAndRss() throws InterruptedException {
        try (ProcessSampler s = new ProcessSampler(20, Map::of)) {
            s.start();
            long until = System.nanoTime() + 500_000_000L;
            double x = 0;
            while (System.nanoTime() < until) {
                x += Math.sqrt(x + 1);
            }
            List<ProcessSampler.Usage> u = s.stop();
            ProcessSampler.Usage self = u.stream().filter(p -> p.pid() == ProcessHandle.current().pid()).findFirst()
                    .orElseThrow();
            assertEquals("self", self.role());
            assertTrue(self.peakRssMB() > 10, "rss " + self.peakRssMB() + " (x=" + x + ")");
            assertTrue(self.cpuSeconds() > 0.05, "cpu " + self.cpuSeconds());
        }
    }

    @Test
    void failingEnumerationIsCountedAndFallsBackToKnownProcesses() {
        long self = ProcessHandle.current().pid();
        java.util.concurrent.atomic.AtomicInteger calls = new java.util.concurrent.atomic.AtomicInteger();
        try (ProcessSampler s = new ProcessSampler(10_000, () -> Map.of(self, "driver"), () -> {
            calls.incrementAndGet();
            throw new RuntimeException("Cannot allocate memory");
        })) {
            s.start();
            List<ProcessSampler.Usage> u = s.stop();
            assertEquals(2, calls.get(), "start and stop each sampled once");
            assertEquals(2, s.errors());
            assertEquals(1, u.size(), "the named root is still sampled");
            assertEquals("driver", u.get(0).role());
            assertTrue(u.get(0).peakRssMB() > 10);
        }
        try (ProcessSampler s = new ProcessSampler(10_000, () -> {
            throw new IllegalStateException("roles unavailable");
        }, List::of)) {
            s.start();
            assertEquals(1, s.stop().size(), "this JVM alone");
            assertEquals(2, s.errors());
        }
    }
}
