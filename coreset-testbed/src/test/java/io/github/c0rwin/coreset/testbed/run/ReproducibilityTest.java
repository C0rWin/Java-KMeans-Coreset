package io.github.c0rwin.coreset.testbed.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ReproducibilityTest {

    private static TinyRuns.Collector first;

    @BeforeAll
    static void runOnce() throws Exception {
        first = TinyRuns.run(TinyRuns.config(2));
    }

    @Test
    void sameSeedGivesIdenticalReportsExceptTimings() throws Exception {
        TinyRuns.Collector second = TinyRuns.run(TinyRuns.config(2));
        assertEquals(first.reports.size(), second.reports.size());
        for (int i = 0; i < first.reports.size(); i++) {
            assertEquals(first.reports.get(i).withoutTimings(), second.reports.get(i).withoutTimings(),
                    "report " + i);
        }
    }

    @Test
    void gridIsCompleteAndEveryRebuildWasDeterministic() {
        // 3 datasets x 2 k x 4 algorithms x 4 modes x 2 replicates, minus grpc on bow (2 k x 4 alg x 2 rep).
        assertEquals(3 * 2 * 4 * 4 * 2 - 16, first.reports.size());
        assertEquals(6, first.pools.size());
        for (ValidityReport r : first.reports) {
            if (!r.construction().reused()) {
                assertEquals(Boolean.TRUE, r.integrity().replayDeterministic(), r.config().toString());
            }
            assertTrue(r.distortion().max() >= 1.0);
            assertEquals(r.config().algorithm().id().equals("mean") ? Integer.valueOf(0) : null,
                    r.integrity().oneSidedViolations(), "mean never underestimates; others are not one-sided");
        }
    }

    @Test
    void directAndGrpcTransportsBuildTheSameCoreset() {
        Map<String, String> byKey = new HashMap<>();
        int compared = 0;
        for (ValidityReport r : first.reports) {
            if (!"distributed".equals(r.config().mode())) {
                continue;
            }
            String key = r.dataset().id() + "/" + r.config().algorithm().label() + "/" + r.config().k() + "/"
                    + r.config().replicate();
            String other = byKey.putIfAbsent(key, r.integrity().replayHash());
            if (other != null) {
                assertEquals(other, r.integrity().replayHash(), key);
                compared++;
            }
            if ("grpc".equals(r.config().transport())) {
                assertTrue(r.construction().bytesOnWire() > r.construction().bytesSerialized(), key);
            } else {
                assertNull(r.construction().bytesOnWire());
            }
            assertEquals(2, r.construction().perNode().size());
        }
        assertEquals(2 * 2 * 4 * 2, compared);
    }

    @Test
    void kIndependentAlgorithmsAreBuiltOncePerGridPoint() {
        for (ValidityReport r : first.reports) {
            boolean takesK = r.config().algorithm().params().containsKey("k");
            assertEquals(!takesK && r.config().k() == 4, r.construction().reused(), r.config().toString());
        }
    }

    @Test
    void streamReportsCarryPerLevelDistortionAndBenchmarkGetsTheDeltaSweep() {
        for (ValidityReport r : first.reports) {
            List<ValidityReport.Level> levels = r.construction().perLevel();
            if ("stream".equals(r.config().mode())) {
                assertEquals(4, levels.size()); // 8 blocks: levels 0..3
                for (int l = 0; l < levels.size(); l++) {
                    assertEquals(l, levels.get(l).level());
                    assertNotNull(levels.get(l).distortionMax());
                }
                assertEquals(r.dataset().n(), levels.get(3).inputPoints());
            } else {
                assertTrue(levels.isEmpty());
            }
            assertEquals(r.dataset().id().equals("esa"), r.distortion().benchmarkDeltaSweep() != null);
            boolean sampledOffline = "offline".equals(r.config().mode())
                    && "sensitivity".equals(r.config().algorithm().id());
            assertEquals(sampledOffline, r.distortion().ciCoverage() != null, r.config().toString());
            assertFalse(r.verdict().notes() == null);
        }
    }
}
