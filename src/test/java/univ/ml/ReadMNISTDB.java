package univ.ml;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Files;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.junit.Test;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class ReadMNISTDB implements Serializable {

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

        pointSet.add(new WeightedDoublePoint(new double[] {1, 3, 1}, 2, "One"));
        pointSet.add(new WeightedDoublePoint(new double[] {0, 1, 1}, 3, "Two"));
        pointSet.add(new WeightedDoublePoint(new double[] {1, 3, 0}, 1, "Three"));

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

        pointSet.add(new DoublePoint(new double[] {1, 3, 1}));
        pointSet.add(new DoublePoint(new double[] {1, 3, 1}));
        pointSet.add(new DoublePoint(new double[] {0, 1, 1}));
        pointSet.add(new DoublePoint(new double[] {0, 1, 1}));
        pointSet.add(new DoublePoint(new double[] {0, 1, 1}));
        pointSet.add(new DoublePoint(new double[] {1, 3, 0}));

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
        final double[] weights= new double[] {0.12, 2, 0.23, 0.4, 0.15, 0.6, 7, 0.48, 0.19, 0.10, 0.17, 0.17};
        final double total = StatUtils.sum(weights);

        Map<Integer, AtomicInteger> validation = Maps.newHashMap();
        for (int i = 0; i < weights.length; i++) {
            validation.put(i, new AtomicInteger(0));
        }

        for (int j = 0; j < 1_000_000; j++) {
            Random rnd = new Random();
            double distr = total * rnd.nextDouble();
            double sum = 0;
            int i  = 0;
            for (; sum < distr; i++) {
                sum += weights[i];
            }
            validation.get(i - 1).incrementAndGet();
        }

        for (Map.Entry<Integer, AtomicInteger> each : validation.entrySet()) {
            System.out.println(each.getKey() + " = " + total*each.getValue().get()/1_000_000);
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
        WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer = new WeightedKMeansPlusPlusClusterer<>(_K, 1000);
        List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(pointSet);
        final DistanceMeasure measure = new EuclideanDistance();

        final double optCost = clusters.stream()
                .map(cluster ->
                    cluster.getPoints()
                            .stream()
                            .map(point ->
                                    // Compute squared distance
                                    Math.pow(measure.compute(point.getPoint(), cluster.getCenter().getPoint()), 2)
                            ).collect(Collectors.summingDouble(x->x))
                ).collect(Collectors.summingDouble(x->x));

        List<Integer> sampleSizes = Lists.newArrayList(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000);

        System.out.println("Sample size;\t\tUniform Error;\t\tNon Uniform Error;\t\tKmeans Error");
        for (Integer sampleSize : sampleSizes) {

            CoresetEvaluator evaluator = new CoresetEvaluator(clusterer);

            double nonUniformCost = evaluator.evalute(new NonUniformCoreset<>(_K, sampleSize), pointSet);
            double uniformCost = evaluator.evalute(new UniformCoreset<>(sampleSize), pointSet);
            double kmeansCost = evaluator.evalute(new KmeansCoreset(sampleSize), pointSet);

            System.out.println(sampleSize + ";\t\t" + (uniformCost / optCost - 1) +
                    ";\t\t" + (nonUniformCost / optCost - 1) +
                    ";\t\t" + (kmeansCost / optCost - 1));
        }
    }

    @Test
    public void testSparkStreamingContext() {
        SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("WordCound");
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(60));

        JavaReceiverInputDStream<String> lines = jssc.socketTextStream("localhost", 9999);


        JavaDStream<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterable<String> call(String s) throws Exception {
                return Arrays.asList(s.split(" "));
            }
        });

        JavaPairDStream<String, Integer> pairs = words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String s) throws Exception {
                return new Tuple2<String, Integer>(s, 1);
            }
        });

        JavaPairDStream<String, Integer> wordsCount = pairs.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer i1, Integer i2) throws Exception {
                return i1 + i2;
            }
        });

        wordsCount.print();

        jssc.start();
        jssc.awaitTermination();

    }

}