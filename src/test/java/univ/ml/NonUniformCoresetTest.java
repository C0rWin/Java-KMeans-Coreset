package univ.ml;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.Assert;
import org.junit.Test;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.Repeat;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.SparseWeightedKMeansPlusPlus;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseUniformCoreset;
import univ.ml.sparse.algorithm.streaming.StreamingAlgorithm;

public class NonUniformCoresetTest extends RandomizedTest {

    @Test
//    @Repeat(iterations = 100)
    public void sampleSizeSmallerThanOverAllPointsShouldReturnAllPoints() {
        final int k = randomIntBetween(10, 25);
        final int sampleSize = randomIntBetween(128, 512);
        final int dimension = 1_000;
        final int datasetSize = randomIntBetween(10_000, 20_000);

        final SparseNonUniformCoreset alg = new SparseNonUniformCoreset(k, sampleSize);

        List<SparseWeightableVector> points = generateRandomSet(dimension, datasetSize);
        double totalWeight = 0.0;
        for (SparseWeightableVector vector : points) {
            totalWeight += vector.getWeight();
        }
        System.out.println("Total Weight (before): " + totalWeight);


        final List<SparseWeightableVector> sample = alg.takeSample(points);

        Assert.assertEquals(sampleSize, sample.size());

        totalWeight = 0.0;
        for (SparseWeightableVector vector : sample) {
            totalWeight += vector.getWeight();
        }
        System.out.println("Total Weight (after): " + totalWeight);

        final SparseWeightedKMeansPlusPlus transform = new SparseWeightedKMeansPlusPlus(k);
        final List<SparseCentroidCluster> clusters = transform.cluster(sample);
        Assert.assertEquals(k, clusters.size());
    }

    @Test
    public void streamingCoreset() throws IOException {
        int k = 10;
        int sampleSize = 500;

        final MNISTReader reader = new MNISTReader("/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv");
        final List<SparseWeightableVector> pointSet = reader.readPoints(20_000);

        final Iterable<List<SparseWeightableVector>> partitions = Iterables.partition(pointSet, 10_000);

        final StreamingAlgorithm streamingNonUniform = new StreamingAlgorithm(new SparseNonUniformCoreset(k, sampleSize));
        final StreamingAlgorithm streamingUniform = new StreamingAlgorithm(new SparseUniformCoreset(sampleSize));

        List<SparseWeightableVector> dataset = Lists.newArrayList();

        for (List<SparseWeightableVector> each : partitions) {
            dataset.addAll(each);
            streamingNonUniform.addDataset(each);
            streamingUniform.addDataset(each);
        }

        final List<SparseWeightableVector> nonUniformSample = streamingNonUniform.getTotalCoreset();
        double totalNonUniformWeight = 0;
        for (SparseWeightableVector each : nonUniformSample) {
            totalNonUniformWeight += each.getWeight();
        }
        System.out.println("Non uniform weight: " + totalNonUniformWeight);


        final List<SparseWeightableVector> uniformSample = streamingUniform.getTotalCoreset();
        double totalUniformWeight = 0;
        for (SparseWeightableVector each : uniformSample) {
            totalUniformWeight += each.getWeight();
        }
        System.out.println("Uniform weight: " + totalUniformWeight);
//        Assert.assertEquals(sampleSize, sample.size());

        final SparseWeightedKMeansPlusPlus transform = new SparseWeightedKMeansPlusPlus(k);
        final SparseWSSE wsse = new SparseWSSE();
        final List<SparseCentroidCluster> nonUniformClusters = transform.cluster(nonUniformSample);
        final List<SparseCentroidCluster> uniformClusters = transform.cluster(uniformSample);

        final List<SparseWeightableVector> nonUniformCenters = Lists.newArrayList();
        for (SparseCentroidCluster cluster : nonUniformClusters) {
            nonUniformCenters.add(new SparseWeightableVector(cluster.getCenter().getVector()));
        }

        final List<SparseWeightableVector> uniformCenters = Lists.newArrayList();
        for (SparseCentroidCluster cluster : uniformClusters) {
            uniformCenters.add(new SparseWeightableVector(cluster.getCenter().getVector()));
        }

        double nonUniformEnergy = wsse.getCost(nonUniformCenters, dataset);
        double uniformEnergy = wsse.getCost(uniformCenters, dataset);

        final List<SparseCentroidCluster> realClusters = transform.cluster(dataset);
        double realEnergy = wsse.getCost(realClusters);

        Assert.assertEquals(k, nonUniformClusters.size());

        System.out.println("Uniform Coreset Energy: \t\t" + uniformEnergy);
        System.out.println("Non Uniform Coreset Energy: \t" + nonUniformEnergy);
        System.out.println("Real Energy: \t\t" + realEnergy);
    }

    @Test
    public void testRandomSample() {
        final int k = 10;
        final int sampleSize = 256;
        final int dimension = 1000;
        final int datasetSize = 40000;

        List<SparseWeightableVector> points = generateRandomSet(dimension, datasetSize);

        SparseRandomSample randomSample = new SparseRandomSample(points);

        final List<SparseWeightableVector> sample = randomSample.getSampleOfSize(sampleSize);
        Assert.assertEquals(256, sample.size());

        final Set<SparseWeightableVector> set = new HashSet(sample);
        Assert.assertEquals(256, set.size());
    }

//    @Test
//    public void readRoysFile( ) throws IOException {
//        final List<SparseWeightableVector> vectors = new ArrayList<>();
//        try (final BufferedReader reader = new BufferedReader(new InputStreamReader(getClass().getClassLoader().getResourceAsStream("vectors1.txt")))) {
//            String line = null;
//            while( (line=reader.readLine()) != null) {
//                try {
//                    final Map<String, String> keyVectors = Splitter.on(";").withKeyValueSeparator("=").split(line);
//                    final Double weight = Double.valueOf(keyVectors.get("w"));
//                    final Double probability = Double.valueOf(keyVectors.get("p"));
//                    final String vector = keyVectors.get("vec");
//
//                    final String cleanedVector = vector
//                            .replaceAll("\\[\\(", "")
//                            .replaceAll("\\)\\],", "")
//                            .replaceAll("\\)\\]", "")
//                            .replaceAll("\\),\\(", ";");
//
//                    final Map<String, String> coords = Splitter.on(";").withKeyValueSeparator(",").split(cleanedVector);
//                    final Map<Integer, Double> points = new HashMap<>();
//
//                    for (Map.Entry<String, String> each : coords.entrySet()) {
//                        points.put(Integer.valueOf(each.getKey()), Double.valueOf(each.getValue()));
//                    }
//                    final SparseWeightableVector v = new SparseWeightableVector(points, weight, 100_000);
//                    v.setProbability(probability);
//                    vectors.add(v);
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//
//        final Set<SparseWeightableVector> set = new HashSet<>(vectors);
//        System.out.println("Sample size = [" + set.size() + "]");
//
//        final SparseWeightedKMeansPlusPlus transform = new SparseWeightedKMeansPlusPlus(10);
//        final List<SparseCentroidCluster> clusters = transform.cluster(vectors);
//        Assert.assertEquals(10, clusters.size());
//    }

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

        Assert.assertTrue(sampleSize >= sample.size());
    }

    @Test
    public void weightedVector() throws Exception {
        Map<Integer, Double> values1 = Maps.newHashMap();
        Map<Integer, Double> values2 = Maps.newHashMap();
        for (int i = 0; i < 10; ++i) {
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
                    v.put(j, randomIntBetween(10, 100) * randomDouble());
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

        Assert.assertTrue(sampleSize >= sample.size());
    }

    private List<SparseWeightableVector> generateRandomSet(int dimension, int datasetSize) {
        final RandomDataGenerator rnd = new RandomDataGenerator();
        List<SparseWeightableVector> points = Lists.newArrayList();

        for (int j = 0; j < datasetSize; j++) {
            double[] coord = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                coord[i] = rnd.nextGaussian(5, 2);
            }
            final SparseWeightableVector point = new SparseWeightableVector(coord, randomIntBetween(10, 100));
            point.setProbability(1.0 / datasetSize);
            points.add(point);
//            points.add(new SparseWeightableVector(coord, randomIntBetween(10, 100)));
        }
        return points;
    }

}
