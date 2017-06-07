package univ.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;

import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.KmeansPlusPlusSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseKmeansCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseUniformCoreset;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

@RunWith(Parameterized.class)
public class CoresetAlgorithmsTest {

    @Parameterized.Parameter(0)
    public int N;

    @Parameterized.Parameter(1)
    public int D;

    @Parameterized.Parameter(2)
    public SparseCoresetAlgorithm algorithm;

    @Parameterized.Parameter(3)
    public int sampleSize;

    private List<SparseWeightableVector> dataset;

    @Parameterized.Parameters
    public static Iterable<Object[]> data() {
        final KmeansPlusPlusSeedingAlgorithm seeding = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(10));
        return Arrays.asList(new Object[][]{
                {1_000, 3, new SparseUniformCoreset(100), 100},
                {1_000, 3, new SparseUniformCoreset(200), 200},
                {1_000, 3, new SparseUniformCoreset(500), 500},
                {1_000, 3, new SparseUniformCoreset(1000), 1000},
                {1_000, 3, new SparseUniformCoreset(1100), 1100},
                {1_000, 3, new SparseNonUniformCoreset(seeding, 100), 100},
                {1_000, 3, new SparseNonUniformCoreset(seeding, 200), 200},
                {1_000, 3, new SparseNonUniformCoreset(seeding, 500), 500},
                {1_000, 3, new SparseNonUniformCoreset(seeding, 1000), 1000},
                {1_000, 3, new SparseNonUniformCoreset(seeding, 1100), 1100},
                {1_000, 3, new SparseKmeansCoresetAlgorithm(100), 100},
                {1_000, 3, new SparseKmeansCoresetAlgorithm(200), 200},
                {1_000, 3, new SparseKmeansCoresetAlgorithm(500), 500},
                {1_000, 3, new SparseKmeansCoresetAlgorithm(1000), 1000},
                {1_000, 3, new SparseKmeansCoresetAlgorithm(1100), 1100},
        });
    }

    @Before
    public void setup() {
        dataset = new ArrayList<>();
        final Random rng = new Random(System.currentTimeMillis());
        for (int i = 0; i < N; i++) {
            Map<Integer, Double> coords = new HashMap<>();
            for (int j = 0; j < D; j++) {
                coords.put(j, rng.nextDouble());
            }
            dataset.add(new SparseWeightableVector(coords, 1d, D));
        }
    }

    @Test
    public void coresetAlgorithmTest() {
        // Make sure we generating expected number of random points.
        Assert.assertEquals(N, dataset.size());
        final List<SparseWeightableVector> sample = algorithm.takeSample(dataset);
        // Next we need to make sure we able to sample require amount of points.
        // Invariant: sample size should be equal or less than sample size
        Assert.assertTrue(sampleSize >= sample.size());
        Assert.assertTrue(algorithm.getClass().getName() + "-" + sampleSize,
                Iterables.all(dataset, new Predicate<SparseWeightableVector>() {
                    @Override
                    public boolean apply(SparseWeightableVector vector) {
                        return vector.getWeight() == 1d;
                    }
                }));
    }
}
