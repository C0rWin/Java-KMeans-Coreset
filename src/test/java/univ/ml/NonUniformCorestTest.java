package univ.ml;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;
import com.google.common.collect.Lists;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

public class NonUniformCorestTest extends RandomizedTest {

    @Test
    @Repeat(iterations = 100)
    public void sampleSizeSmallerThanOverAllPointsShouldReturnAllPoints() {
        final int k = randomIntBetween(5, 20);
        final int sampleSize = randomIntBetween(250, 300);
        final int dimension = randomIntBetween(10, 200);
        final int datasetSize = randomIntBetween(50, 200);

        final NonUniformCoreset<WeightedDoublePoint> alg = new NonUniformCoreset<>(k, sampleSize);
        final RandomDataGenerator rnd = new RandomDataGenerator();

        List<WeightedDoublePoint> points = generateRandomSet(dimension, datasetSize, rnd);
        final List<WeightedDoublePoint> sample = alg.takeSample(points);

        Assert.assertEquals(datasetSize, sample.size());
    }

    @Test
    @Repeat(iterations = 100)
    public void sampleSizeGreaterThanOverAllPointsShouldReturnSampleSize() {
        final int k = randomIntBetween(5, 20);
        final int sampleSize = randomIntBetween(50, 300);
        final int dimension = randomIntBetween(10, 200);
        final int datasetSize = randomIntBetween(250, 1_000);

        final NonUniformCoreset<WeightedDoublePoint> alg = new NonUniformCoreset<>(k, sampleSize);
        final RandomDataGenerator rnd = new RandomDataGenerator();

        List<WeightedDoublePoint> points = generateRandomSet(dimension, datasetSize, rnd);
        final List<WeightedDoublePoint> sample = alg.takeSample(points);

        Assert.assertTrue(sampleSize>=sample.size());
    }

    private List<WeightedDoublePoint> generateRandomSet(int dimension, int datasetSize, RandomDataGenerator rnd) {
        List<WeightedDoublePoint> points = Lists.newArrayList();

        for (int j = 0; j < datasetSize; j++) {
            double[] coord = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                coord[i] = rnd.nextGaussian(5, 2);
            }
            points.add(new WeightedDoublePoint(coord, 1, ""));
        }
        return points;
    }

}
