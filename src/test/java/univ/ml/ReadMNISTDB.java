package univ.ml;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Files;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.StatUtils;
import org.junit.Test;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseCoresetEvaluator;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.SparseWeightedKMeansPlusPlus;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseUniformCoreset;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class ReadMNISTDB {

    public static final String MNIST_FILE = "/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv";

    @Test
    public void readFile() throws IOException {
        List<String> strings = Files.readLines(new File("/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv"), Charset.defaultCharset());
        System.out.println(strings.size());
        System.out.println(Iterables.size(Splitter.on(",").split(strings.get(0))));
    }

    @Test
    public void artificialClustersK2() {
        final List<WeightedDoublePoint> pointSet = Lists.newArrayList();

        pointSet.add(new WeightedDoublePoint(new double[]{1, 3, 1}, 2, "One"));
        pointSet.add(new WeightedDoublePoint(new double[]{0, 1, 1}, 3, "Two"));
        pointSet.add(new WeightedDoublePoint(new double[]{1, 3, 0}, 1, "Three"));

        WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer = new WeightedKMeansPlusPlusClusterer<WeightedDoublePoint>(1);
        List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(pointSet);

        for (CentroidCluster<WeightedDoublePoint> each : clusters) {
            List<WeightedDoublePoint> points = each.getPoints();
            double cost = 0;
            EuclideanDistance distance = new EuclideanDistance();
            for (WeightedDoublePoint point : points) {
                cost += point.getWeight() * Math.pow(distance.compute(point.getPoint(), each.getCenter().getPoint()), 2);
            }

            System.out.println(Arrays.toString(each.getCenter().getPoint()));
            System.out.println("Cost: " + cost);
        }
    }

    @Test
    public void artificialClustersNotWeightedK2() {
        final Collection<DoublePoint> pointSet = Lists.newArrayList();

        pointSet.add(new DoublePoint(new double[]{1, 3, 1}));
        pointSet.add(new DoublePoint(new double[]{1, 3, 1}));
        pointSet.add(new DoublePoint(new double[]{0, 1, 1}));
        pointSet.add(new DoublePoint(new double[]{0, 1, 1}));
        pointSet.add(new DoublePoint(new double[]{0, 1, 1}));
        pointSet.add(new DoublePoint(new double[]{1, 3, 0}));

        KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<DoublePoint>(1);
        List<CentroidCluster<DoublePoint>> clusters = clusterer.cluster(pointSet);

        for (CentroidCluster<DoublePoint> each : clusters) {
            List<DoublePoint> points = each.getPoints();
            double cost = 0;
            EuclideanDistance distance = new EuclideanDistance();
            for (DoublePoint point : points) {
                cost += Math.pow(distance.compute(point.getPoint(), each.getCenter().getPoint()), 2);
            }

            System.out.println(Arrays.toString(each.getCenter().getPoint()));
            System.out.println("Cost: " + cost);
        }
    }

    @Test
    public void testSampleMethod() {
        final double[] weights = new double[]{0.12, 2, 0.23, 0.4, 0.15, 0.6, 7, 0.48, 0.19, 0.10, 0.17, 0.17};
        final double total = StatUtils.sum(weights);

        Map<Integer, AtomicInteger> validation = Maps.newHashMap();
        for (int i = 0; i < weights.length; i++) {
            validation.put(i, new AtomicInteger(0));
        }

        for (int j = 0; j < 1_000_000; j++) {
            Random rnd = new Random();
            double distr = total * rnd.nextDouble();
            double sum = 0;
            int i = 0;
            for (; sum < distr; i++) {
                sum += weights[i];
            }
            validation.get(i - 1).incrementAndGet();
        }

        for (Map.Entry<Integer, AtomicInteger> each : validation.entrySet()) {
            System.out.println(each.getKey() + " = " + total * each.getValue().get() / 1_000_000);
        }
    }

    @Test
    public void testCoresets() throws Exception {
        final List<String> lines = Files.readLines(new File(MNIST_FILE), Charset.defaultCharset());
        final List<WeightedDoublePoint> pointSet = Lists.newArrayList();

        for (int j = 0; j < lines.size(); ++j) {
            List<String> coordinates = Splitter.on(',').splitToList(lines.get(j));
            final double[] _coords = new double[coordinates.size() - 1];
            for (int i = 1; i < coordinates.size(); i++) {
                _coords[i - 1] = Double.valueOf(coordinates.get(i));
            }
            pointSet.add(new WeightedDoublePoint(_coords, 1, String.valueOf(coordinates.get(0))));
        }

        int _K = 30;
        final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer = new WeightedKMeansPlusPlusClusterer<>(_K);
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(pointSet);
        final WSSE wsse = new WSSE(new EuclideanDistance());
        final double optCost = wsse.getCost(clusters);

        final List<Integer> sampleSizes = Lists.newArrayList(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
        //        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
        //        15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000
        );

        System.out.println("Sample size;\t\tUniform Error;\t\tNon Uniform Error;\t\tKmeans Error");
        for (final Integer sampleSize : sampleSizes) {

            CoresetEvaluator evaluator = new CoresetEvaluator(clusterer);

            double nonUniformCost = evaluator.evalute(new NonUniformCoreset<>(_K, sampleSize), pointSet);
            double uniformCost = evaluator.evalute(new UniformCoreset<>(sampleSize), Collections.unmodifiableList(pointSet));
//            System.out.println("Uniform cost: " + uniformCost);
//            double kmeansCost = evaluator.evalute(new KmeansCoreset(sampleSize), pointSet);

            System.out.println(sampleSize + ";\t\t" + (uniformCost / optCost - 1) +
                    ";\t\t" + (nonUniformCost / optCost - 1)
//                    + ";\t\t" + (kmeansCost / optCost - 1)
            );
        }
    }

    @Test
    public void testSparseCoresets() throws Exception {
        final List<String> lines = Files.readLines(new File(MNIST_FILE), Charset.defaultCharset());
        final List<SparseWeightableVector> pointSet = Lists.newArrayList();

        System.out.println("Reading data");
        for (int j = 0; j < lines.size(); ++j) {
            List<String> coordinates = Splitter.on(',').splitToList(lines.get(j));
            final double[] _coords = new double[coordinates.size() - 1];
            for (int i = 1; i < coordinates.size(); i++) {
                _coords[i - 1] = Double.valueOf(coordinates.get(i));
            }
            pointSet.add(new SparseWeightableVector(_coords, 1));
        }

        System.out.println("Running KMeans++");
        int _K = 30;
        final SparseWeightedKMeansPlusPlus clusterer = new SparseWeightedKMeansPlusPlus(_K, 10);
        final List<SparseCentroidCluster> clusters = clusterer.cluster(pointSet);
        final SparseWSSE wsse = new SparseWSSE();
        final double optCost = wsse.getCost(clusters);

        final List<Integer> sampleSizes = Lists.newArrayList(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
                //        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                //        15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000
        );

        System.out.println("Evaluating Coresets construction");
        System.out.println("Sample size;\t\tUniform Error;\t\tNon Uniform Error;\t\tKmeans Error");
        for (final Integer sampleSize : sampleSizes) {

            SparseCoresetEvaluator evaluator = new SparseCoresetEvaluator(clusterer);

            double nonUniformCost = evaluator.evalute(new SparseNonUniformCoreset(_K, sampleSize), pointSet);
            double uniformCost = evaluator.evalute(new SparseUniformCoreset(sampleSize), Collections.unmodifiableList(pointSet));
//            System.out.println("Uniform cost: " + uniformCost);
//            double kmeansCost = evaluator.evalute(new KmeansCoreset(sampleSize), pointSet);

            System.out.println(sampleSize + ";\t\t" + (uniformCost / optCost - 1) +
                    ";\t\t" + (nonUniformCost / optCost - 1));
        }
    }


    @Test
    public void svdCoreset() {

        final int N = 5_000;
        final int d = 1000;
        final int j = 10;
        final double epsilon = 0.1;

        final int m = Math.min(j + (int)Math.ceil(j/epsilon) - 1, d - 1);

        final BlockRealMatrix A = new BlockRealMatrix(N, d);
        for (int i = 0; i < N; i++) {
            final Random rnd = new Random();
            A.setRow(i, IntStream.range(0, d).mapToDouble(x -> rnd.nextDouble()).toArray());
        }

        SingularValueDecomposition svd = new SingularValueDecomposition(A);
//        final RealMatrix U = svd.getU().getSubMatrix(0, 1000, 0, 10);
        final RealMatrix S = svd.getS().getSubMatrix(0, m, 0, m);
        final RealMatrix VT = svd.getV().getSubMatrix(0, d - 1, 0, m).transpose();

        final double c = Math.pow(A.getFrobeniusNorm(), 2) - Math.pow(S.getFrobeniusNorm(), 2);
        final RealMatrix C = S.multiply(VT);


        final BlockRealMatrix Q = new BlockRealMatrix(d, d);
        for (int i = 0; i < d; i++) {
            final Random rnd = new Random();
            Q.setRow(i, IntStream.range(0, d).mapToDouble(x -> rnd.nextDouble()).toArray());
        }

        final SingularValueDecomposition svd1 = new SingularValueDecomposition(Q);
        final RealMatrix tst = svd1.getV().getSubMatrix(0, d - 1, 0, d - 1 - j);

        final double costA = Math.pow(A.multiply(tst).getFrobeniusNorm(), 2);
        final double costC = c + Math.pow(C.multiply(tst).getFrobeniusNorm(), 2);

        System.out.println(Math.abs(costC / costA - 1));
    }
}