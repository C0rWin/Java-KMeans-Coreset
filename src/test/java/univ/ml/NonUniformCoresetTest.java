package univ.ml;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.Assert;
import org.junit.Test;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.SparseWeightedKMeansPlusPlus;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;

import java.util.List;
import java.util.Map;

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
    public void weightedVector() throws Exception {
        Map<Integer, Double> values1 = Maps.newHashMap();
        Map<Integer, Double> values2 = Maps.newHashMap();
        for(int i = 0; i < 10; ++i) {
            values1.put(i, 1.0);
            values2.put(i, 2.0);
        }
        SparseWeightableVector v1 = new SparseWeightableVector(values1, 10);
        SparseWeightableVector v2 = new SparseWeightableVector(values2, 10);

        v1.combineToSelf(1, 3, v2);

        for (Double v : v1.toArray()) {
            System.out.println(v);
        }
    }

    @Test
    public void sparseKmeans() throws Exception {
        int dimension = 50_000;
        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(20);

        final List<SparseWeightableVector> points = Lists.newArrayList();

        for (int i = 0; i < 1000; i++) {
            Map<Integer, Double> v = Maps.newHashMap();
            for (int j = 0; j < dimension; j++) {
                if (randomBoolean())
                    v.put(j, randomIntBetween(10, 100)*randomDouble());
            }
            points.add(new SparseWeightableVector(v, randomDouble(), dimension));
        }
        final List<SparseCentroidCluster> clusters = kmeans.cluster(points);

        final SparseWSSE wsse = new SparseWSSE();

        final double cost = wsse.getCost(clusters);
        System.out.println("Total cost: " + cost);
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
