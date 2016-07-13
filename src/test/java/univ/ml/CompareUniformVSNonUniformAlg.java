package univ.ml;

import com.google.common.collect.Lists;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.util.MathArrays;
import org.junit.BeforeClass;
import org.junit.Test;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.SparseWeightedKMeansPlusPlus;
import univ.ml.sparse.algorithm.BiCriteriaAlgorithm;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseUniformCoreset;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public class CompareUniformVSNonUniformAlg {

    private static SparseWeightedKMeansPlusPlus transform;

    private static SparseWSSE wsse;

    private static List<SparseWeightableVector> pointSet;

    private static int k = 15;

    private static int pointsToRead = 10_000;

    private static int iterations = 10;

    private int sampleSize = 20_000;

    private int batchSize = 10_000;

    @BeforeClass
    public static void setup() throws IOException {
        final MNISTReader reader = new MNISTReader("/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv");
//        pointSet = generateRandomSet(1_000, pointsToRead, 0);
        pointSet = reader.readPoints(pointsToRead);
        transform = new SparseWeightedKMeansPlusPlus(k, iterations);
    }

    @Test
    public void compare() throws IOException {

        // Compute baseline on entire dataset
        final List<SparseCentroidCluster> realClusters = transform.cluster(pointSet);

        wsse = new SparseWSSE();
        double realEnergy = wsse.getCost(realClusters);


        double totalWeight = 0d;
        for (SparseWeightableVector vector : pointSet) {
            totalWeight += vector.getWeight();
        }

        System.out.println("Total weight = " + totalWeight);

        System.out.println("===");
        System.out.println("Real energy: " + realEnergy);
        System.out.println();

        // NonUniform coreset
        final SparseCoresetEvaluator evaluator = new SparseCoresetEvaluator(transform);
        final double nonUniformEnergy = evaluator.evaluate(new SparseNonUniformCoreset(k, sampleSize), pointSet, batchSize);

        System.out.println("===");
        System.out.println("Non-Uniform Coreset energy: " + nonUniformEnergy);
        System.out.println();

        // Uniform coreset
        final double uniformEnergy = evaluator.evaluate(new SparseUniformCoreset(sampleSize), pointSet, batchSize);

        System.out.println("===");
        System.out.println("Uniform Coreset energy: " + uniformEnergy);
        System.out.println();
    }

    @Test
    public void compareKmeans() {
        List<DoublePoint> dPoints = Lists.newArrayList();

        for (SparseWeightableVector vector : pointSet) {
            double[] coord = new double[vector.getVector().getDimension()];
            for (int i = 0; i < vector.getDimension(); i++) {
                coord[i] = vector.getVector().getEntry(i);
            }
            dPoints.add(new DoublePoint(coord));
        }

        final KMeansPlusPlusClusterer kmeans = new KMeansPlusPlusClusterer(k, iterations);
        final List<CentroidCluster<DoublePoint>> clusters = kmeans.cluster(dPoints);

        double cost = 0d;
        for (CentroidCluster<DoublePoint> cluster : clusters) {
            final double[] centerCoord = cluster.getCenter().getPoint();
            for (DoublePoint point : cluster.getPoints()) {
                final double[] pointCoord = point.getPoint();
                final double d = MathArrays.distance(centerCoord, pointCoord);
                cost += d * d;
            }
        }

        System.out.println("Original (kmeans) energy: " + cost);

        List<WeightedDoublePoint> weightedPoints = Lists.newArrayList();

        for (SparseWeightableVector vector : pointSet) {
            double[] coord = new double[vector.getVector().getDimension()];
            for (int i = 0; i < vector.getDimension(); i++) {
                coord[i] = vector.getVector().getEntry(i);
            }
            weightedPoints.add(new WeightedDoublePoint(coord, 1, "1"));
        }

        final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> weightedKmeans = new WeightedKMeansPlusPlusClusterer<>(k, iterations);
        final List<CentroidCluster<WeightedDoublePoint>> weightedClusters = weightedKmeans.cluster(weightedPoints);

        cost = 0d;
        for (CentroidCluster<WeightedDoublePoint> cluster : weightedClusters) {
            final double[] centerCoord = cluster.getCenter().getPoint();
            for (WeightedDoublePoint point : cluster.getPoints()) {
                final double[] pointCoord = point.getPoint();
                final double d = MathArrays.distance(centerCoord, pointCoord);
                cost += d * d;
            }
        }

        System.out.println("Weighted (kmeans) energy: " + cost);

        // Compute baseline on entire dataset
        final List<SparseCentroidCluster> realClusters = transform.cluster(pointSet);

        wsse = new SparseWSSE();
        double realEnergy = wsse.getCost(realClusters);

        System.out.println("Sparse (kmeans) energy: " + realEnergy);

        final BiCriteriaAlgorithm biCriteria = new BiCriteriaAlgorithm(k, .5);
        final List<SparseCentroidCluster> biCriteriaResults = biCriteria.takeSample(pointSet);

        System.out.println("BiCriteria Size = " + biCriteriaResults.size());

        cost = 0d;
        for (int i = 0; i < pointSet.size(); i++) {
            final RealVector v1 = pointSet.get(i).getVector();

            double minDist = Double.MAX_VALUE;

            for (SparseCentroidCluster cluster : biCriteriaResults) {
                final RealVector v2 = cluster.getCenter().getVector();

                final double d = v2.getDistance(v1);
                if (d < minDist) {
                    minDist = d;
                }
            }
            cost += minDist * minDist;
        }

        System.out.println("BiCriteria energy: " + cost);

        cost = 0d;
        final Map<Integer, SparseCentroidCluster> centers = transform.chooseInitialCenters(pointSet);
        for (int i = 0; i < pointSet.size(); i++) {
            final RealVector v1 = pointSet.get(i).getVector();

            double minDist = Double.MAX_VALUE;

            for (SparseCentroidCluster cluster : centers.values()) {
                final RealVector v2 = cluster.getCenter().getVector();

                final double d = v2.getDistance(v1);
                if (d < minDist) {
                    minDist = d;
                }
            }
            cost += minDist * minDist;
        }

        System.out.println("Kmeans seed energy: " + cost);
    }

//    @Test
//    public void testNonUniformStreaming() {
//        final List<SparseWeightableVector> points = new ArrayList<>(pointSet.size());
//        for (SparseWeightableVector each : pointSet) {
//            points.add(new SparseWeightableVector(each, each.getWeight()));
//        }
//
//        int trial = 1;
//        int batchSize = points.size() / 1;
//
//        final Iterable<List<SparseWeightableVector>> partition = Iterables.partition(points, batchSize);
//
//        final SparseCoresetAlgorithm coresetAlg = new SparseNonUniformCoreset(k, sampleSize);
//        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(k);
//
//        for (List<SparseWeightableVector> each : partition) {
//            final List<SparseWeightableVector> sample = coresetAlg.takeSample(each);
//            final List<SparseCentroidCluster> clusters = kmeans.cluster(sample);
//
//            final List<SparseCentroidCluster> opt = kmeans.cluster(each);
//
//            final List<SparseWeightableVector> centers = Lists.newArrayList();
//
//            for (final SparseCentroidCluster cluster : clusters) {
//                centers.add(new SparseWeightableVector(cluster.getCenter().getVector()));
//            }
//
//            final double coresetEnergy = wsse.getCost(centers, each);
//            final double optEnergy = wsse.getCost(opt);
//
//            double totalWeight = 0d;
//
//            for (SparseWeightableVector vector : sample) {
//                totalWeight += vector.getWeight();
//            }
//
//            System.out.println("#" + trial + ", optEnergy = " + optEnergy +
//                    ", coreset energy = " + coresetEnergy +
//                    ", total weight = " + totalWeight +
//                    ", error = " + (coresetEnergy - optEnergy)/optEnergy);
//
//            trial++;
//        }
//    }

//    private static List<SparseWeightableVector> generateRandomSet(int dimension, int datasetSize) {
//        final RandomDataGenerator rnd = new RandomDataGenerator();
//
//        final RandomVectorGenerator vectorGenerator =
//                new UncorrelatedRandomVectorGenerator(dimension, new GaussianRandomGenerator(new JDKRandomGenerator()));
//
//        List<SparseWeightableVector> points = Lists.newArrayList();
//
//        for (int j = 0; j < datasetSize; j++) {
//            double[] coord = vectorGenerator.nextVector();
//            final SparseWeightableVector point = new SparseWeightableVector(coord, rnd.nextUniform(10, 100));
//            point.setProbability(1.0 / datasetSize);
//            points.add(point);
//        }
//        return points;
//    }

    private static List<SparseWeightableVector> generateRandomSet(int dimension, int datasetSize, double mean) {
        final RandomDataGenerator rnd = new RandomDataGenerator();
        List<SparseWeightableVector> points = Lists.newArrayList();

        for (int j = 0; j < datasetSize; j++) {
            double[] coord = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                coord[i] = rnd.nextGaussian(mean, 2);
            }
            final SparseWeightableVector point = new SparseWeightableVector(coord, 1);
//            final SparseWeightableVector point = new SparseWeightableVector(coord, rnd.nextInt(10, 100));
            point.setProbability(1.0 / datasetSize);
            points.add(point);
        }
        return points;
    }


}
