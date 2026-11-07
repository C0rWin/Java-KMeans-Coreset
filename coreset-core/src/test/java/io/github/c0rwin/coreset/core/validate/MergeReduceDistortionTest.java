package io.github.c0rwin.coreset.core.validate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.TestData;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/**
 * Streams 32 batches of a 5-component Gaussian mixture through a merge-and-reduce tree and measures, after 2^L batches
 * (when the tree holds a single coreset that went through L reduce levels), the distortion over the standard
 * candidate sets against all points seen so far.
 */
class MergeReduceDistortionTest {

    private static final int K = 5;
    private static final int DIM = 4;
    private static final int BATCH = 500;
    private static final int LEVELS = 5;

    /** Distortion after each level, index L-1; the generators and seeds are fixed, so the values are reproducible. */
    private static double[] distortionPerLevel(CoresetAlgorithm algo, boolean checkOneSided) {
        SplittableRandom data = new SplittableRandom(1);
        double[][] centers = TestData.mixtureCenters(K, DIM, 10.0, data);
        MergeReduceTree tree = new MergeReduceTree(algo, new SplittableRandom(2));
        List<WeightedPoint> seen = new ArrayList<>();
        double[] out = new double[LEVELS];
        for (int b = 1; b <= 1 << LEVELS; b++) {
            List<WeightedPoint> batch = TestData.mixture(centers, BATCH, true, data);
            seen.addAll(batch);
            tree.add(Coreset.of(batch));
            if (Integer.bitCount(b) != 1 || b == 1) {
                continue;
            }
            int level = Integer.numberOfTrailingZeros(b);
            assertEquals(1, tree.levels());
            Coreset full = Coreset.of(seen);
            Coreset coreset = tree.snapshot();
            List<Candidate> candidates =
                    CandidateSolutions.standard(full, coreset, K, 10, 3, new SplittableRandom(100 + level));
            out[level - 1] = Distortion.evaluate(full, coreset, candidates).max();
            if (checkOneSided) {
                List<List<SparseVector>> queries = new ArrayList<>();
                for (Candidate c : candidates) {
                    queries.add(c.centers());
                }
                assertEquals(0, IntegrityCheck.oneSidedViolations(full, coreset, queries, 1e-12), "level " + level);
            }
        }
        return out;
    }

    private static void assertBelow(double threshold, double[] perLevel, String what) {
        for (int l = 0; l < perLevel.length; l++) {
            assertTrue(perLevel[l] < threshold,
                    what + " level " + (l + 1) + " distortion " + perLevel[l] + " >= " + threshold);
        }
    }

    /**
     * m = 500 = 100 k. Measured maxima per level: uniform 1.121, sensitivity with k-means++ 1.098, with bicriteria
     * 1.200 (no growth with depth); the 1.3 bound leaves margin over those values.
     */
    @Test
    void samplingCoresetsStayAccurateAcrossLevels() {
        assertBelow(1.3, distortionPerLevel(new UniformCoreset(BATCH), false), "uniform");
        assertBelow(1.3, distortionPerLevel(new SensitivityCoreset(K, BATCH, Seeding.kMeansPlusPlus(0)), false),
                "sensitivity/kmeans++");
        assertBelow(1.3, distortionPerLevel(new SensitivityCoreset(K, BATCH, Seeding.bicriteria()), false),
                "sensitivity/bicriteria");
    }

    /** m = 100 = 20 k. One-sided at every level; measured maximum distortion 1.043. */
    @Test
    void meanCoresetNeverUnderestimatesAcrossLevels() {
        assertBelow(1.1, distortionPerLevel(new MeanCoreset(100, 10), true), "mean");
    }
}
