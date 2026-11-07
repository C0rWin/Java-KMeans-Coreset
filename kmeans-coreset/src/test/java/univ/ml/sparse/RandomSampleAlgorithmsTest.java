package univ.ml.sparse;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The samplers use unseeded randomness, so every statistical assertion below states its failure probability
 * for a correct sampler.
 */
class RandomSampleAlgorithmsTest {

    static Stream<Arguments> samplers() {
        return Stream.of(
                Arguments.of("SparseRandomSample", (Supplier<RandomSampleAlgorithm>) SparseRandomSample::new),
                Arguments.of("CTRandomSample", (Supplier<RandomSampleAlgorithm>) CTRandomSample::new),
                Arguments.of("BinarySearchRandomSample", (Supplier<RandomSampleAlgorithm>) BinarySearchRandomSample::new));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("samplers")
    void frequenciesMatchUnnormalizedProbabilities(final String name, final Supplier<RandomSampleAlgorithm> sampler) {
        // Sums to 20, so q = {.1, .4, .3, .05, .15}.
        final double[] probabilities = {2, 8, 6, 1, 3};
        final int draws = 1_000_000;
        final double[] counts = drawCounts(sampler.get(), probabilities, draws);

        // Hoeffding + union bound: P(any |freq - q| >= 0.005) <= 5 * 2 * exp(-2 * 1e6 * 0.005^2) < 1e-20.
        for (int i = 0; i < probabilities.length; i++) {
            final double expected = probabilities[i] / 20;
            final double observed = counts[i] / draws;
            assertEquals(expected, observed, 0.005,
                    name + " point " + i + ": observed frequency " + observed + ", expected " + expected
                            + " (counts " + Arrays.toString(counts) + ")");
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("samplers")
    void zeroProbabilityPointsAreNeverDrawn(final String name, final Supplier<RandomSampleAlgorithm> sampler) {
        final double[] probabilities = {0, 1, 0, 3, 0};
        final double[] counts = drawCounts(sampler.get(), probabilities, 5_000_000);

        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] == 0) {
                assertEquals(0d, counts[i],
                        name + " drew zero-probability point " + i + " (counts " + Arrays.toString(counts) + ")");
            }
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("samplers")
    void pointLighterThanOneCtBucketIsStillDrawn(final String name, final Supplier<RandomSampleAlgorithm> sampler) {
        // q_0 = 9e-6 is below CTRandomSample's bucket width 1e-5.
        final double[] probabilities = {9e-6, 1 - 9e-6};
        final double[] counts = drawCounts(sampler.get(), probabilities, 5_000_000);

        // Expected 45 draws of point 0; P(none) = (1 - 9e-6)^5e6 < e^-44 < 1e-19.
        assertTrue(counts[0] > 0, name + " never drew point 0 (counts " + Arrays.toString(counts) + ")");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("samplers")
    void manyTinyProbabilitiesNeitherThrowNorLoseDraws(final String name, final Supplier<RandomSampleAlgorithm> sampler) {
        // Tiny masses around a dominant one are absorbed by rounding in the running sum.
        final int tiny = 20_000;
        final double[] probabilities = new double[2 * tiny + 2];
        Arrays.fill(probabilities, 1e-17);
        probabilities[tiny] = 1;
        probabilities[probabilities.length - 1] = 0;

        assertDoesNotThrow(() -> drawCounts(sampler.get(), probabilities, 10_000), name);
    }

    /**
     * Aggregating samplers return each drawn point once with its weight multiplied by its draw count, while
     * BinarySearchRandomSample returns one entry per draw; with unit input weights both read as draw counts.
     */
    private static double[] drawCounts(final RandomSampleAlgorithm sampler, final double[] probabilities,
                                       final int draws) {
        final List<SparseWeightableVector> dataset = new ArrayList<>();
        final Map<SparseWeightableVector, Integer> index = new IdentityHashMap<>();
        for (int i = 0; i < probabilities.length; i++) {
            final SparseWeightableVector vector = new SparseWeightableVector(new double[]{i}, 1d);
            vector.setProbability(probabilities[i]);
            dataset.add(vector);
            index.put(vector, i);
        }

        final double[] counts = new double[probabilities.length];
        for (SparseWeightableVector vector : sampler.getSampleOfSize(dataset, draws)) {
            counts[index.get(vector)] += vector.getWeight();
        }
        assertEquals(draws, Arrays.stream(counts).sum(), "every draw must be accounted for");
        return counts;
    }
}
