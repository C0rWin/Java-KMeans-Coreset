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

import com.google.common.collect.Lists;

import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseKmeansCoresetAlgorithm;
import univ.ml.sparse.algorithm.streaming.StreamingAlgorithm;

@RunWith(Parameterized.class)
public class SparseKmeansCoresetAlgorithmTest {

    @Parameter
    public int K;

    @Parameter(1)
    public int N;

    @Parameter(2)
    public int D;

    @Parameter(3)
    public int sampleSize;

    @Parameter(4)
    public int batchSize;

    private List<SparseWeightableVector> dataset;

    @Parameters
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {10, 100_000, 1_000, 500, 20_000}
        });
    }

    @Before
    public void setup() {
        final Random rng = new Random(System.currentTimeMillis());
        dataset = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            Map<Integer, Double> coords = new HashMap<>();
            for (int j = 0; j < D; j++) {
                coords.put(j, rng.nextDouble());
            }
            final SparseWeightableVector point = new SparseWeightableVector(coords, 1, D);
            point.setProbability(rng.nextDouble());
            dataset.add(point);
        }
    }

    @Test
    public void testKmeansCoresetAlgorithm() {
        final SparseCoresetAlgorithm algorithm = new SparseKmeansCoresetAlgorithm(sampleSize);
        final StreamingAlgorithm streaming = new StreamingAlgorithm(algorithm);

        final List<List<SparseWeightableVector>> chunks = Lists.partition(dataset, batchSize);

        for (final List<SparseWeightableVector> chunk : chunks) {
            streaming.addDataset(chunk);
        }

        final List<SparseWeightableVector> coreset = streaming.getTotalCoreset();

        double totalWeight = 0d;
        for (SparseWeightableVector x : coreset) {
            totalWeight += x.getWeight();
        }

        Assert.assertEquals(1d * N, totalWeight, 10E-6);
    }

}
