package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.metrics.Span;
import io.micrometer.prometheusmetrics.PrometheusConfig;
import io.micrometer.prometheusmetrics.PrometheusMeterRegistry;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class MicrometerMetricsSinkTest {

    @Test
    void mapsCountsSamplesAndTimings() {
        PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        MicrometerMetricsSink sink = new MicrometerMetricsSink(registry);
        sink.count("node.ingest.busy", 2);
        sink.count("node.ingest.busy", 3);
        sink.record("node.queue.depth", 4);
        sink.record("node.queue.depth", 6);
        sink.timeNanos("node.ingest.latency_ns", 2_000_000);
        try (Span ignored = sink.span("node.batch.compress")) {
            // timed block
        }
        assertEquals(5.0, registry.get("node.ingest.busy").counter().count());
        assertEquals(10.0, registry.get("node.queue.depth").summary().totalAmount());
        assertEquals(2.0, registry.get("node.ingest.latency_ns").timer().totalTime(TimeUnit.MILLISECONDS));
        assertEquals(1, registry.get("node.batch.compress.wall_ns").timer().count());
        String scrape = registry.scrape();
        assertTrue(scrape.contains("node_ingest_busy_total 5.0"), scrape);
        assertTrue(scrape.contains("node_queue_depth_count 2"), scrape);
    }
}
