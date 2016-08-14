package univ.ml;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.util.Precision;
import org.junit.Assert;
import org.junit.Test;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;

public class RandomSamplingTest extends RandomizedTest {

    public static class RandomSampleLinear {

        /**
         * Sample k points with given distribution
         *
         * @param probabilities points probabilities, assuming that overall sum is 1
         * @param k amount of points to sample
         * @return indexes of sampled points
         */
        public int[] sample(final double[] probabilities, int k) {
            int[] result = new int[k];

            final Random rng = new Random(System.nanoTime());

            for (int i = 0; i < k; i++) {
                double p = rng.nextDouble();
                int p_i = 0;
                for (double sum = 0; sum < p; ) {
                    sum += probabilities[p_i++];
                }
                result[i] = p_i - 1;
            }
            return result;
        }

    }

    @Test
    @Repeat(iterations = 1_000)
    public void samplingChiSquareTest() {
        double[] probabilities = new double[] {0.1, 0.2, 0.05, 0.01, 0.31, 0.03, 0.11, 0.09, 0.04, 0.06};

        long[] observation = new long[probabilities.length];

        final RandomSampleLinear sampler = new RandomSampleLinear();

        final int TRIALS = 1_000;

        final int K = 5;

        for (int i = 0; i < TRIALS; i++) {
            final int[] sample = sampler.sample(probabilities, K);
            for (int j : sample) {
                observation[j]++;
            }
        }

        double[] frequencies = new double[probabilities.length];
        for (int i = 0; i < observation.length; i++) {
            frequencies[i] = Precision.round(observation[i] / (double)(TRIALS * K), 2);
        }

        final ChiSquareTest chiSquareTest = new ChiSquareTest();
        System.out.println("Estimated: \t" + Arrays.toString(probabilities));
        System.out.println("Observed: \t" + Arrays.toString(frequencies));
        final double chiSquare = chiSquareTest.chiSquare(probabilities, observation);
        System.out.println(chiSquare);
        Assert.assertTrue(chiSquare < 16.919);
    }
}
