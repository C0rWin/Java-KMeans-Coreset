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
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.KMeansPlusPlusSeed;

@RunWith(Parameterized.class)
public class KMeansPlusPlusSeedTest {

    @Parameter
    public int k;

    @Parameter(1)
    public int N;

    @Parameter(2)
    public int D;

    @Parameter(3)
    public int maxBound;

    @Parameter(4)
    public int minBound;

    public List<SparseWeightableVector> dataset;

    @Parameters
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {  2,   500_000, 100, 500_000, 1_000_000},
                {  5,   500_000, 100, 500_000, 1_000_000},
                { 10,   500_000, 100, 500_000, 1_000_000},
                {100,   500_000, 100, 500_000, 1_000_000},
        });
    }

    @Before
    public void setup() {
        dataset = new ArrayList<>(N);
        final Random rng = new Random();
        for (int i = 0; i < N; i++) {
            final Map<Integer, Double> coords = new HashMap<>();
            for (int j = 0; j < D; j++) {
                coords.put(j, minBound + (maxBound - minBound) * rng.nextDouble());
            }
            dataset.add(new SparseWeightableVector(coords, minBound + (maxBound - minBound) * rng.nextDouble(), D));
        }
    }

    @Test
    public void kmeansSeeding() throws Exception {
        final KMeansPlusPlusSeed seed = new KMeansPlusPlusSeed(k);
        final List<SparseCentroidCluster> clusters = seed.seed(dataset);

        Assert.assertEquals(k, clusters.size());
    }
}
