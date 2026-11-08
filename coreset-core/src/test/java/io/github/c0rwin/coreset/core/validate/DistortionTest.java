package io.github.c0rwin.coreset.core.validate;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import java.util.List;
import org.junit.jupiter.api.Test;

class DistortionTest {

    private static Candidate at(String name, double... coords) {
        return new Candidate(name, List.of(SparseVector.dense(coords)));
    }

    @Test
    void reportsRatiosDistortionsAndArgmax() {
        Coreset full = Coreset.of(List.of(point(1, 0), point(1, 2)));
        // Coreset: one point of weight 2 at 0. Q = {1}: full 2, coreset 2. Q = {0}: full 4, coreset 0. Q = {2}: 4 vs 8.
        Coreset coreset = Coreset.of(List.of(point(2, 0)));
        Distortion.Report r = Distortion.evaluate(full, coreset,
                List.of(at("mid", 1), at("left", 0), at("right", 2)));
        assertEquals(1.0, r.entries().get(0).ratio());
        assertEquals(0.0, r.entries().get(1).ratio());
        assertEquals(Double.POSITIVE_INFINITY, r.entries().get(1).distortion());
        assertEquals(2.0, r.entries().get(2).ratio());
        assertEquals(2.0, r.entries().get(2).distortion());
        assertEquals(Double.POSITIVE_INFINITY, r.max());
        assertEquals("left", r.argmaxGenerator());
    }

    @Test
    void underestimateCountsAsInverseRatio() {
        Coreset full = new Coreset(List.of(point(1, 0)), 0.0);
        Distortion.Report r = Distortion.evaluate(full, Coreset.of(List.of(point(0.5, 0))), List.of(at("q", 3)));
        assertEquals(0.5, r.entries().get(0).ratio());
        assertEquals(2.0, r.max());
        assertEquals(2.0, r.mean());
    }

    @Test
    void zeroFullCostIsOneOnlyWhenCoresetCostIsZero() {
        Coreset full = Coreset.of(List.of(point(1, 4)));
        Distortion.Report r = Distortion.evaluate(full, full, List.of(at("on", 4)));
        assertEquals(1.0, r.max());
        Distortion.Report off = Distortion.evaluate(full, new Coreset(List.of(point(1, 4)), 1.0), List.of(at("on", 4)));
        assertEquals(Double.POSITIVE_INFINITY, off.entries().get(0).ratio());
    }

    @Test
    void rejectsEmptyCandidateList() {
        Coreset full = Coreset.of(List.of(point(1, 0)));
        assertThrows(IllegalArgumentException.class, () -> Distortion.evaluate(full, full, List.of()));
    }
}
