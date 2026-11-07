package io.github.c0rwin.coreset.core.kmeans;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class KMeansPlusPlusTest {

    private static WeightedPoint wp(double w, double... c) {
        return new WeightedPoint(SparseVector.dense(c), w);
    }

    @Test
    void fewerPointsThanK() {
        List<WeightedPoint> pts = List.of(wp(1, 0, 0), wp(1, 1, 0), wp(1, 0, 1));
        List<SparseVector> centers = KMeansPlusPlus.seed(pts, 10, new SplittableRandom(1));
        assertEquals(3, centers.size());
        assertEquals(3, new HashSet<>(centers).size());
    }

    @Test
    void duplicatesLimitCentersToDistinctPoints() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            pts.add(wp(1, 1, 1));
            pts.add(wp(2, 5, 5));
        }
        List<SparseVector> centers = KMeansPlusPlus.seed(pts, 4, new SplittableRandom(2));
        assertEquals(2, centers.size());
        assertEquals(new HashSet<>(List.of(SparseVector.dense(new double[] {1, 1}),
                SparseVector.dense(new double[] {5, 5}))), new HashSet<>(centers));
    }

    @Test
    void allIdenticalPointsTerminateWithOneCenter() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            pts.add(wp(1, 3, 4));
        }
        List<SparseVector> centers = assertTimeout(() -> KMeansPlusPlus.seed(pts, 5, new SplittableRandom(3)));
        assertEquals(List.of(SparseVector.dense(new double[] {3, 4})), centers);
    }

    @Test
    void zeroWeightPointsAreNeverChosen() {
        List<WeightedPoint> pts = List.of(wp(0, 100, 100), wp(1, 0, 0), wp(0, -50, 3), wp(0, 7, 7));
        for (int seed = 0; seed < 20; seed++) {
            List<SparseVector> centers = KMeansPlusPlus.seed(pts, 3, new SplittableRandom(seed));
            assertEquals(List.of(SparseVector.dense(new double[] {0, 0})), centers);
        }
        List<WeightedPoint> zeros = List.of(wp(0, 1, 1), wp(0, 2, 2));
        assertTrue(assertTimeout(() -> KMeansPlusPlus.seed(zeros, 2, new SplittableRandom(4))).isEmpty());
        assertTrue(KMeansPlusPlus.seed(List.of(), 2, new SplittableRandom(4)).isEmpty());
    }

    @Test
    void seedingPicksOnePointPerWellSeparatedGroup() {
        SplittableRandom rnd = new SplittableRandom(5);
        List<WeightedPoint> pts = new ArrayList<>();
        double[][] means = {{0, 0}, {1000, 0}, {0, 1000}};
        for (double[] m : means) {
            for (int i = 0; i < 30; i++) {
                pts.add(wp(1, m[0] + rnd.nextDouble(), m[1] + rnd.nextDouble()));
            }
        }
        List<SparseVector> centers = KMeansPlusPlus.seed(pts, 3, new SplittableRandom(6));
        HashSet<Integer> groups = new HashSet<>();
        for (SparseVector c : centers) {
            groups.add((int) Math.round(c.get(0) / 1000) + 2 * (int) Math.round(c.get(1) / 1000));
        }
        assertEquals(3, groups.size());
    }

    @Test
    void distanceEvaluationsMatchFormula() {
        SplittableRandom rnd = new SplittableRandom(7);
        List<WeightedPoint> pts = new ArrayList<>();
        int n = 40;
        for (int i = 0; i < n; i++) {
            pts.add(wp(1 + rnd.nextDouble(), rnd.nextDouble(), rnd.nextDouble(), rnd.nextDouble()));
        }
        pts.add(wp(0, 9, 9, 9)); // zero weight: not part of n
        RecordingMetrics m = new RecordingMetrics();
        int k = 5;
        assertEquals(k, KMeansPlusPlus.seed(pts, k, new SplittableRandom(8), m).size());
        assertEquals((long) n * (k - 1), m.snapshot().counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS));
        assertEquals(1, m.snapshot().distributions().get(MetricNames.KMEANSPP_SEED + MetricNames.WALL_NS_SUFFIX)
                .count());

        // Runs out of distinct points: one update pass per chosen center.
        RecordingMetrics m2 = new RecordingMetrics();
        List<WeightedPoint> dup = List.of(wp(1, 1), wp(1, 1), wp(1, 2));
        assertEquals(2, KMeansPlusPlus.seed(dup, 5, new SplittableRandom(9), m2).size());
        assertEquals(3L * 2, m2.snapshot().counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS));
    }

    @Test
    void sameSeedSameCenters() {
        SplittableRandom rnd = new SplittableRandom(10);
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            pts.add(wp(rnd.nextDouble(), rnd.nextDouble(), rnd.nextDouble()));
        }
        assertEquals(KMeansPlusPlus.seed(pts, 7, new SplittableRandom(11)),
                KMeansPlusPlus.seed(pts, 7, new SplittableRandom(11)));
    }

    @Test
    void invalidK() {
        assertThrows(IllegalArgumentException.class,
                () -> KMeansPlusPlus.seed(List.of(wp(1, 1)), 0, new SplittableRandom(1)));
    }

    private static <T> T assertTimeout(org.junit.jupiter.api.function.ThrowingSupplier<T> s) {
        return org.junit.jupiter.api.Assertions.assertTimeoutPreemptively(Duration.ofSeconds(10), s);
    }
}
