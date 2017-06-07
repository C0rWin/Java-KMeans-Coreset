package univ.ml;

import static org.junit.runners.Parameterized.Parameter;
import static org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import univ.ml.sparse.RandomSampleAlgorithm;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;

@RunWith(Parameterized.class)
public class RandomSampleTest {

    private List<SparseWeightableVector> dataset;

    @Parameter
    public double[] distr;

    @Parameter(1)
    public int D;

    @Parameter(2)
    public int sampleSize;

    @Parameter(3)
    public int trials;

    @Parameters
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][]{
                {new double[]{.5d, .5d}, 2, 1, 1_00},
                {new double[]{.1d, .4d, .3d, .05d, .15d}, 2, 1, 100_000},
                {new double[]{.07d, .13d, .35d, .05d, .25d, .1d, .05d}, 2, 1, 1_000_000}
        });
    }

    @Before
    public void setup() {
        dataset = new ArrayList<>();
        final Random rng = new Random(System.currentTimeMillis());
        for (double p : distr) {
            final Map<Integer, Double> coords = new HashMap<>();
            for (int i = 0; i < D; i++) {
                coords.put(i, rng.nextDouble());
            }
            final SparseWeightableVector vector = new SparseWeightableVector(coords, 1d, D);
            vector.setProbability(p);
            dataset.add(vector);
        }
    }

    @Test
    public void randomSample() {
        final RandomSampleAlgorithm randomSample = new SparseRandomSample();
        final Map<SparseWeightableVector, Integer> freq = new HashMap<>();
        for (final SparseWeightableVector vector : dataset) {
            freq.put(vector, 0);
        }

        for (int n = 0; n < trials; n++) {
            for (int i = 0; i < dataset.size(); i++) {
                dataset.get(i).setProbability(distr[i]);
                dataset.get(i).setWeight(1d);
            }

            final List<SparseWeightableVector> samples = randomSample.getSampleOfSize(dataset, sampleSize);
            for (SparseWeightableVector vector : samples) {
                final Integer val = freq.get(vector);
                freq.put(vector, val + (int)vector.getWeight());
            }
        }

        final double[] expected = new double[distr.length];
        final long[] observed = new long[distr.length];

        for (int i = 0; i < dataset.size(); i++) {
            expected[i] = trials * sampleSize * distr[i];
            observed[i] = freq.get(dataset.get(i));
        }

        final ChiSquareTest chiSquareTest = new ChiSquareTest();
        Assert.assertFalse(chiSquareTest.chiSquareTest(expected, observed, 0.01));
    }

}
