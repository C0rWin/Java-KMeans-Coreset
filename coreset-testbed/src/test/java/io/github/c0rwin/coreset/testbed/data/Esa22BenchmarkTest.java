package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class Esa22BenchmarkTest {

    private static Centers clustering(Esa22Benchmark b, int a) {
        List<SparseVector> cs = new ArrayList<>();
        for (int j = 0; j < b.k(); j++) {
            cs.add(b.center(a, j));
        }
        return Centers.of(cs);
    }

    @Test
    void matchesThePaperFigureForKEquals2AndAlpha3() {
        // ESA'22 Figure 1 (left): rows are all sign patterns of (+-1/2, -+1/2) in each of the 3 column blocks.
        Esa22Benchmark b = Esa22Benchmark.of(2, 3);
        List<WeightedPoint> pts = b.points();
        assertEquals(8, pts.size());
        assertEquals(6, b.dim());
        assertEquals(SparseVector.dense(new double[] {0.5, -0.5, 0.5, -0.5, 0.5, -0.5}), pts.get(0).vector());
        assertEquals(SparseVector.dense(new double[] {-0.5, 0.5, 0.5, -0.5, 0.5, -0.5}), pts.get(1).vector());
        assertEquals(SparseVector.dense(new double[] {-0.5, 0.5, -0.5, 0.5, -0.5, 0.5}), pts.get(7).vector());
    }

    @Test
    void clusteringsAreBalancedMutuallyIndependentAndOfEqualCost() {
        int k = 4;
        int alpha = 3;
        Esa22Benchmark b = Esa22Benchmark.of(k, alpha);
        List<WeightedPoint> pts = b.points();
        assertEquals(64, b.n());
        assertEquals(12, b.dim());
        assertEquals(64, pts.size());

        double[] mean = new double[b.dim()];
        for (WeightedPoint p : pts) {
            p.vector().addTo(mean, 1.0 / pts.size());
            assertEquals(alpha * (k - 1.0) / k, p.vector().squaredNorm(), 1e-12);
        }
        for (double m : mean) {
            assertEquals(0.0, m, 1e-12);
        }

        double expectedCost = b.n() * (alpha - 1.0) * (k - 1.0) / k;
        double[] scratch = new double[k];
        for (int a = 0; a < alpha; a++) {
            Centers c = clustering(b, a);
            int[] sizes = new int[k];
            for (int r = 0; r < b.n(); r++) {
                int j = b.clusterOf(a, r);
                sizes[j]++;
                assertEquals(j, c.nearest(pts.get(r).vector(), scratch), "row " + r + " clustering " + a);
            }
            for (int s : sizes) {
                assertEquals(b.clusterSize(), s);
            }
            assertEquals(expectedCost, Problem.kMeans().cost(Coreset.of(pts), c), 1e-9 * expectedCost);
            for (int a2 = a + 1; a2 < alpha; a2++) {
                int[][] overlap = new int[k][k];
                for (int r = 0; r < b.n(); r++) {
                    overlap[b.clusterOf(a, r)][b.clusterOf(a2, r)]++;
                }
                for (int[] row : overlap) {
                    for (int o : row) {
                        assertEquals(b.n() / (k * k), o); // k^(alpha-2): maximally dissimilar clusterings
                    }
                }
            }
        }
    }

    @Test
    void closedFormFullCostMatchesNumericCostForEveryDeficientCount() {
        Esa22Benchmark b = Esa22Benchmark.of(3, 3);
        Coreset full = Coreset.of(b.points());
        for (int deficient = 0; deficient < b.k(); deficient++) {
            List<SparseVector> kept = new ArrayList<>();
            for (int j = deficient; j < b.k(); j++) {
                kept.add(b.center(1, j));
            }
            double numeric = Problem.kMeans().cost(full, Centers.of(kept));
            assertEquals(numeric, b.fullCost(deficient), 1e-9 * numeric);
        }
    }

    @Test
    void deltaSweepIsExactOnTheFullDataAndDetectsAMissingCluster() {
        Esa22Benchmark b = Esa22Benchmark.of(4, 3);
        List<WeightedPoint> pts = b.points();
        double[] grid = Esa22Benchmark.deltaGrid(20);

        Esa22Benchmark.DeltaSweep exact = b.deltaSweep(Coreset.of(pts), grid).orElseThrow();
        assertEquals(1.0, exact.distortion(), 1e-12);
        assertEquals(0, exact.deficientClusters());

        // Drop cluster 0 of clustering 2 entirely: with weight-preserving coresets the remaining mass is misplaced.
        List<WeightedPoint> missing = new ArrayList<>();
        for (int r = 0; r < b.n(); r++) {
            if (b.clusterOf(2, r) != 0) {
                missing.add(pts.get(r));
            }
        }
        Esa22Benchmark.DeltaSweep sweep = b.deltaSweep(Coreset.of(missing), grid).orElseThrow();
        assertEquals(2, sweep.clustering());
        assertEquals(1, sweep.deficientClusters());
        // Full cost with one deficient cluster vs coreset cost that simply lacks those points.
        double full = b.fullCost(1);
        double onCoreset = (b.n() - b.clusterSize()) * b.pointCostToOwnCenter();
        assertEquals(full / onCoreset, sweep.distortion(), 1e-9);
        assertTrue(sweep.distortion() > 1.3);
    }

    @Test
    void rejectsDegenerateOrOversizedInstances() {
        assertThrows(IllegalArgumentException.class, () -> Esa22Benchmark.of(1, 3));
        assertThrows(IllegalArgumentException.class, () -> Esa22Benchmark.of(3, 1));
        assertThrows(IllegalArgumentException.class, () -> Esa22Benchmark.of(100, 5));
    }
}
