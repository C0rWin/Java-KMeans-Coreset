package univ.ml;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.UnitSphereRandomVectorGenerator;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.VectorialMean;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;
import org.junit.BeforeClass;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.BiCriteriaSeedingAlgorithm;
import univ.ml.sparse.algorithm.KMeansPlusPlusSeed;
import univ.ml.sparse.algorithm.KmeansPlusPlusSeedingAlgorithm;
import univ.ml.sparse.algorithm.SensitivityFunction;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseUniformCoreset;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

public class CompareUniformVSNonUniformAlg {

    private static SparseWeightedKMeansPlusPlus transform;

    private static SparseWSSE wsse;

    private static List<SparseWeightableVector> pointSet;

    private static int k = 15;

    private static int pointsToRead = 50_000;

    private static int iterations = 10;

    private int sampleSize = 1_500;

    private int batchSize = 50_000;

    @BeforeClass
    public static void setup() throws IOException {
        final MNISTReader reader = new MNISTReader("/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv");
        pointSet = reader.readPoints(pointsToRead);
//        pointSet = generateRandomSet(1_0, pointsToRead, 0);
        transform = new SparseWeightedKMeansPlusPlus(k, iterations);
    }


    @Test
    public void simpleTest() {
        int D = 3;
        int N = 50_000;
        int trials = 10;

        List<SparseWeightableVector> points = getNormalizedVectors(D, N);

        double totalVariance = 0d;
        final int STEP = 100;

        int START = 0;

        int END = 100;

        final VectorialMean mean = new VectorialMean(D);
        for (SparseWeightableVector point : points) {
            double v[] = new double[point.getDimension()];
            for (int i = 0; i < point.getDimension(); i++) {
                v[i] = point.getEntry(i);
            }
            totalVariance += point.getWeight() * FastMath.pow(MathArrays.safeNorm(v), 2);
            MathArrays.scaleInPlace(point.getWeight(), v);
            mean.increment(v);
        }

        System.out.println("Mean = " + MathArrays.safeNorm(mean.getResult()));
        System.out.println("Total vectors variance = " + totalVariance);

        Map<Integer, Mean> weightsMeans = Maps.newTreeMap();
        Map<Integer, Mean> norms = Maps.newTreeMap();

        for (int i = START; i < END; i++) {
            weightsMeans.put(STEP * (i + 1), new Mean());
            norms.put(STEP * (i + 1), new Mean());
        }

        for (int i = 0; i < trials; i++) {
            for (Map.Entry<Integer, Mean> weights : weightsMeans.entrySet()) {
                final int sampleSize = weights.getKey();

                // Create seeding algorithm for (alpha, beta) - approximation based on k-means++.
                final SparseSeedingAlgorithm seedingAlgorithm = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(1));

                // Initialize coreset algorithm
                final SparseNonUniformCoreset coreset = new SparseNonUniformCoreset(seedingAlgorithm, sampleSize,
                                                                        new SensitivityFunction(), new SparseRandomSample());

                List<SparseWeightableVector> copy = Lists.newArrayList();
                for (SparseWeightableVector point : points) {
                    copy.add(new SparseWeightableVector(point, point.getWeight()));
                }

                final List<SparseWeightableVector> sample = coreset.takeSample(copy);

                double totalWeight = 0d;
                for (SparseWeightableVector s : sample) {
                    double v[] = new double[s.getDimension()];
                    for (int j = 0; j < s.getVector().getDimension(); j++) {
                        v[j] = s.getEntry(j);
                    }
                    totalWeight += s.getWeight();
                }

                weights.getValue().increment(totalWeight);

                final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(1);
                final List<SparseCentroidCluster> clusters = kmeans.cluster(sample);

                norms.get(weights.getKey()).increment(clusters.get(0).getCenter().getVector().getNorm());
            }
        }
        System.out.println("Size;Weight;Error;Size*Error");

        for (Map.Entry<Integer, Mean> each : weightsMeans.entrySet()) {
            final double norm = FastMath.pow(norms.get(each.getKey()).getResult(), 2);
            System.out.println(each.getKey() + ";" + each.getValue().getResult()
                    + ";" + norm
                    + ";" + (norm * each.getKey()));
        }
    }

    @Test
    public void checkKmeans() {
        final List<SparseWeightableVector> vectors = getNormalizedVectors(10, 10);
        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(1);
        final List<SparseCentroidCluster> clusters = kmeans.cluster(vectors);

        System.out.println("Center = " + clusters.get(0).getCenter().getVector());
    }

    @Test
    public void bitwise() {
        int  v = 1;

        System.out.println( (Integer.MAX_VALUE >> 30));
    }

    @Test
    public void compare() {

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
        final SparseCoresetEvaluator evaluator = new SparseCoresetEvaluator(transform, pointSet, batchSize);
        final SparseSeedingAlgorithm seedingAlgorithm = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(k));
//        final SparseSeedingAlgorithm seedingAlgorithm = new BiCriteriaSeedingAlgorithm(k, 45);

        final double nonUniformEnergy = evaluator.evaluate(new SparseNonUniformCoreset(seedingAlgorithm, sampleSize));

        System.out.println("===");
        System.out.println("Non-Uniform Coreset energy: " + nonUniformEnergy);
        System.out.println();

        // Uniform coreset
        final double uniformEnergy = evaluator.evaluate(new SparseUniformCoreset(sampleSize));

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

        final BiCriteriaSeedingAlgorithm biCriteria = new BiCriteriaSeedingAlgorithm(k, 50);
        final List<SparseCentroidCluster> biCriteriaResults = biCriteria.seed(pointSet);

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

        final SparseSeedingAlgorithm seedAlg = new KMeansPlusPlusSeed(k);
        final List<SparseCentroidCluster> seedAlgResults = seedAlg.seed(pointSet);

        System.out.println("KMeans++ Seed Size = " + seedAlgResults.size());

        cost = 0d;
        for (int i = 0; i < pointSet.size(); i++) {
            final RealVector v1 = pointSet.get(i).getVector();

            double minDist = Double.MAX_VALUE;

            for (SparseCentroidCluster cluster : seedAlgResults) {
                final RealVector v2 = cluster.getCenter().getVector();

                final double d = v2.getDistance(v1);
                if (d < minDist) {
                    minDist = d;
                }
            }
            cost += minDist * minDist;
        }

        System.out.println("KMeans++ seed energy: " + cost);

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

    private List<SparseWeightableVector> getNormalizedVectors(int d, int n) {
        List<SparseWeightableVector> points = Lists.newArrayList();

        final UnitSphereRandomVectorGenerator rnd = new UnitSphereRandomVectorGenerator(d, new JDKRandomGenerator());

        final VectorialMean vectorMean = new VectorialMean(d);
        final double[][] tempMatrix = new double[n][];

        final double weight = 1d/n;

        for (int i = 0; i < n; i++) {
            tempMatrix[i] = rnd.nextVector();
            vectorMean.increment(tempMatrix[i]);
        }

        for (int i = 0; i < n; i++) {
            tempMatrix[i] = MathArrays.ebeSubtract(tempMatrix[i], vectorMean.getResult());
        }

        double totalVariance = 0d;
        for (int i = 0; i < n; i++) {
            totalVariance += weight * FastMath.pow(MathArrays.safeNorm(tempMatrix[i]), 2);
        }

        System.out.println("Variance BEFORE " + totalVariance);
        for (int i = 0; i < n; i++) {
            MathArrays.scaleInPlace(1d / FastMath.sqrt(totalVariance), tempMatrix[i]);
        }

        totalVariance = 0d;
        for (int i = 0; i < n; i++) {
            totalVariance += weight * FastMath.pow(MathArrays.safeNorm(tempMatrix[i]), 2);
        }
        System.out.println("Variance AFTER " + totalVariance);

        for (int i = 0; i < n; i++) {
            Map<Integer, Double> coord = Maps.newHashMap();
            for (int j = 0; j < d; j++) {
                coord.put(j, tempMatrix[i][j]);
            }
            points.add(new SparseWeightableVector(coord, weight, d));
        }
        return points;
    }

}
