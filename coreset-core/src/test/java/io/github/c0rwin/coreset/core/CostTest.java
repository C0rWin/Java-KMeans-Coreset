package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class CostTest {

    private static SparseVector v(double... c) {
        return SparseVector.dense(c);
    }

    @Test
    void weightedCostUsesNearestCenterAndOffset() {
        List<WeightedPoint> pts = List.of(
                new WeightedPoint(v(0, 0), 2.0),   // nearest (1,0): d2 = 1
                new WeightedPoint(v(4, 0), 0.5),   // nearest (5,0): d2 = 1
                new WeightedPoint(v(5, 3), 3.0),   // nearest (5,0): d2 = 9
                new WeightedPoint(v(100, 100), 0.0));
        List<SparseVector> centers = List.of(v(1, 0), v(5, 0));

        assertEquals(2.0 + 0.5 + 27.0, Cost.cost(pts, centers));
        assertEquals(29.5 + 4.25, Cost.cost(new Coreset(pts, 4.25), centers));
    }

    @Test
    void weightTwoEqualsTwoCopies() {
        List<SparseVector> centers = List.of(v(1, 1), v(-2, 0.5));
        SparseVector p = v(0.3, -0.7);
        SparseVector q = v(-1, 2);
        double weighted = Cost.cost(List.of(new WeightedPoint(p, 2.0), new WeightedPoint(q, 1.0)), centers);
        double copies = Cost.cost(List.of(new WeightedPoint(p, 1.0), new WeightedPoint(p, 1.0),
                new WeightedPoint(q, 1.0)), centers);
        assertEquals(copies, weighted, 1e-12);
    }

    @Test
    void nearestBreaksTiesByLowestIndex() {
        List<SparseVector> centers = List.of(v(2, 0), v(-2, 0), v(0, 0.5));
        assertEquals(2, Cost.nearest(v(0, 0), centers));
        assertEquals(0, Cost.nearest(v(0, -10), List.of(v(1, 0), v(-1, 0))));
    }

    @Test
    void emptyInputs() {
        assertEquals(0.0, Cost.cost(List.of(), List.of()));
        assertEquals(1.5, Cost.cost(new Coreset(List.of(), 1.5), List.of()));
        List<WeightedPoint> pts = List.of(new WeightedPoint(v(1), 1.0));
        assertThrows(IllegalArgumentException.class, () -> Cost.cost(pts, List.of()));
        assertThrows(IllegalArgumentException.class, () -> Cost.nearest(v(1), List.of()));
    }
}
