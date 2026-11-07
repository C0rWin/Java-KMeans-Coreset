package io.github.c0rwin.coreset.core.kmeans;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
        PointSet set = PointSet.of(PointBlock.of(pts));
        assertEquals(k, KMeansPlusPlus.seed(set, k, 1, new SplittableRandom(8), m).size());
        assertEquals((long) n * (k - 1), m.snapshot().counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS));
        assertEquals(1, m.snapshot().distributions().get(MetricNames.KMEANSPP_SEED + MetricNames.WALL_NS_SUFFIX)
                .count());

        // Greedy: n for the first center, then L candidates in one pass plus the kept one per step, each either
        // computed or pruned; without pruning all are computed.
        RecordingMetrics g = new RecordingMetrics();
        int trials = KMeansPlusPlus.defaultLocalTrials(k);
        assertEquals(3, trials);
        assertEquals(k, KMeansPlusPlus.seed(pts, k, new SplittableRandom(8), g).size());
        long greedyFormula = (long) n + (long) n * (k - 1) * (trials + 1);
        Map<String, Long> gc = g.snapshot().counters();
        assertEquals(greedyFormula, gc.get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS)
                + gc.get(MetricNames.KMEANSPP_PRUNED_DISTANCES));
        RecordingMetrics u = new RecordingMetrics();
        assertEquals(KMeansPlusPlus.seed(pts, k, new SplittableRandom(8)),
                KMeansPlusPlus.seed(set, k, trials, new SplittableRandom(8), u, false));
        assertEquals(greedyFormula, u.snapshot().counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS));
        assertEquals(0L, u.snapshot().counters().get(MetricNames.KMEANSPP_PRUNED_DISTANCES));

        // Runs out of distinct points: one update pass per chosen center.
        RecordingMetrics m2 = new RecordingMetrics();
        List<WeightedPoint> dup = List.of(wp(1, 1), wp(1, 1), wp(1, 2));
        assertEquals(2, KMeansPlusPlus.seed(PointSet.of(PointBlock.of(dup)), 5, 1, new SplittableRandom(9), m2).size());
        assertEquals(3L * 2, m2.snapshot().counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS));
        assertEquals(2, KMeansPlusPlus.seed(dup, 5, new SplittableRandom(9)).size());
    }

    /**
     * Triangle pruning changes nothing: same centers as evaluating every distance, on dense and sparse (coordinate
     * map) input, with duplicates (D = 0), zero weights and several segments; on separated clusters it skips most
     * distances.
     */
    @Test
    void prunedGreedySeedingIsBitIdenticalAndSkipsMostDistances() {
        for (boolean sparse : new boolean[] {false, true}) {
            SplittableRandom rnd = new SplittableRandom(sparse ? 31 : 30);
            int dim = sparse ? 5000 : 6;
            List<WeightedPoint> pts = new ArrayList<>();
            for (int c = 0; c < 25; c++) {
                double[] mean = new double[dim];
                int[] support = new int[6];
                for (int j = 0; j < support.length; j++) {
                    support[j] = sparse ? rnd.nextInt(dim) : j;
                    mean[support[j]] = 200 * rnd.nextDouble() - 100;
                }
                for (int i = 0; i < 80; i++) {
                    double[] x = new double[dim];
                    for (int j : support) {
                        x[j] = mean[j] + rnd.nextGaussian();
                    }
                    double w = i % 17 == 0 ? 0.0 : 0.5 + rnd.nextDouble();
                    pts.add(new WeightedPoint(SparseVector.dense(x), w)); // keeps only the non-zeros
                    if (i % 9 == 0) {
                        pts.add(pts.get(pts.size() - 1)); // exact duplicate
                    }
                }
            }
            PointBlock block = PointBlock.of(pts);
            int half = block.size() / 2;
            PointSet split = PointSet.concat(List.of(PointSet.of(block.slice(0, half)),
                    PointSet.of(block.slice(half, block.size()))));
            long pruned = 0;
            long computed = 0;
            for (int seed = 0; seed < 6; seed++) {
                for (int k : new int[] {2, 25, 60}) {
                    RecordingMetrics m = new RecordingMetrics();
                    List<SparseVector> fast = KMeansPlusPlus.seed(split, k, new SplittableRandom(seed), m);
                    List<SparseVector> full = KMeansPlusPlus.seed(PointSet.of(block), k,
                            KMeansPlusPlus.defaultLocalTrials(k), new SplittableRandom(seed), MetricsSink.NOOP,
                            false);
                    assertEquals(full, fast, "sparse=" + sparse + " seed=" + seed + " k=" + k);
                    pruned += m.snapshot().counters().get(MetricNames.KMEANSPP_PRUNED_DISTANCES);
                    computed += m.snapshot().counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS);
                }
            }
            assertTrue(pruned > computed, "sparse=" + sparse + " pruned " + pruned + " computed " + computed);
        }
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

    /**
     * Why greedy is the default: on 20 well-separated clusters of geometric sizes (25% down to 0.26%), plain D^2
     * seeding leaves some cluster without a seed in most runs, while greedy seeding covers all of them.
     */
    @Test
    void greedySeedingCoversSmallWellSeparatedClusters() {
        int k = 20;
        int d = 20;
        SplittableRandom rnd = new SplittableRandom(7);
        double[][] means = new double[k][d];
        for (double[] mean : means) {
            for (int j = 0; j < d; j++) {
                mean[j] = (2 * rnd.nextDouble() - 1) * 60.0;
            }
        }
        double ratio = Math.pow(0.0026 / 0.25, 1.0 / (k - 1));
        double norm = (1 - Math.pow(ratio, k)) / (1 - ratio);
        List<WeightedPoint> pts = new ArrayList<>();
        Map<SparseVector, Integer> label = new HashMap<>();
        int n = 4000;
        for (int c = 0; c < k; c++) {
            long count = c == k - 1 ? n - pts.size() : Math.round(n * Math.pow(ratio, c) / norm);
            for (int t = 0; t < count; t++) {
                double[] v = new double[d];
                for (int j = 0; j < d; j++) {
                    v[j] = means[c][j] + rnd.nextGaussian();
                }
                SparseVector p = SparseVector.dense(v);
                pts.add(new WeightedPoint(p, 1.0));
                label.put(p, c);
            }
        }
        PointSet set = PointSet.of(PointBlock.of(pts));
        int greedy = 0;
        int plain = 0;
        for (int run = 0; run < 10; run++) {
            greedy += covered(label, k, KMeansPlusPlus.seed(set, k, new SplittableRandom(100 + run), MetricsSink.NOOP));
            plain += covered(label, k,
                    KMeansPlusPlus.seed(set, k, 1, new SplittableRandom(100 + run), MetricsSink.NOOP));
        }
        assertTrue(greedy >= 9, "greedy covered every cluster in " + greedy + "/10 runs");
        assertTrue(plain <= 5, "plain D^2 covered every cluster in " + plain + "/10 runs");
    }

    private static int covered(Map<SparseVector, Integer> label, int k, List<SparseVector> seeds) {
        Set<Integer> hit = new HashSet<>();
        for (SparseVector s : seeds) {
            hit.add(label.get(s));
        }
        return hit.size() == k ? 1 : 0;
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
