package univ.ml;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;
import com.google.common.collect.Lists;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.Assert;
import org.junit.Test;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;

import java.util.List;

public class NonUniformCoresetTest extends RandomizedTest {

    @Test
    @Repeat(iterations = 100)
    public void sampleSizeSmallerThanOverAllPointsShouldReturnAllPoints() {
        final int k = randomIntBetween(5, 20);
        final int sampleSize = randomIntBetween(250, 300);
        final int dimension = randomIntBetween(10, 200);
        final int datasetSize = randomIntBetween(50, 200);

        final SparseNonUniformCoreset alg = new SparseNonUniformCoreset(k, sampleSize);

        List<SparseWeightableVector> points = generateRandomSet(dimension, datasetSize);
        final List<SparseWeightableVector> sample = alg.takeSample(points);

        Assert.assertEquals(datasetSize, sample.size());
    }

    @Test
    @Repeat(iterations = 100)
    public void sampleSizeGreaterThanOverAllPointsShouldReturnSampleSize() {
        final int k = randomIntBetween(5, 20);
        final int sampleSize = randomIntBetween(50, 300);
        final int dimension = randomIntBetween(10, 200);
        final int datasetSize = randomIntBetween(250, 1_000);

        final SparseNonUniformCoreset alg = new SparseNonUniformCoreset(k, sampleSize);

        List<SparseWeightableVector> points = generateRandomSet(dimension, datasetSize);
        final List<SparseWeightableVector> sample = alg.takeSample(points);

        Assert.assertTrue(sampleSize>=sample.size());
    }

    @Test
    @Repeat(iterations = 100)
    public void emptyDataset() {
        final int k = randomIntBetween(5, 20);
        final int sampleSize = randomIntBetween(50, 300);
        final int dimension = randomIntBetween(10, 200);
        final int datasetSize = randomIntBetween(250, 1_000);

        final SparseNonUniformCoreset alg = new SparseNonUniformCoreset(k, sampleSize);

        List<SparseWeightableVector> points = generateRandomSet(dimension, datasetSize);
        final List<SparseWeightableVector> sample = alg.takeSample(points);

        Assert.assertTrue(sampleSize>=sample.size());
    }

    private List<SparseWeightableVector> generateRandomSet(int dimension, int datasetSize) {
        final RandomDataGenerator rnd = new RandomDataGenerator();
        List<SparseWeightableVector> points = Lists.newArrayList();

        for (int j = 0; j < datasetSize; j++) {
            double[] coord = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                coord[i] = rnd.nextGaussian(5, 2);
            }
            points.add(new SparseWeightableVector(coord, randomIntBetween(10, 1000)));
        }
        return points;
    }

}
