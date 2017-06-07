package univ.ml.distributed;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.junit.ClassRule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import univ.ml.distributed.coreset.ClientConfig;
import univ.ml.distributed.coreset.CoresetAlgorithm;
import univ.ml.distributed.coreset.CoresetAlgorithmFactory;
import univ.ml.distributed.coreset.CoresetClient;
import univ.ml.distributed.coreset.CoresetUtils;
import univ.ml.distributed.coreset.CoresetWeightedPoint;
import univ.ml.distributed.coreset.RandomPointsProvider;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;
import univ.ml.sparse.algorithm.streaming.StreamingAlgorithm;

public class CoresetClientServerTest {

    final private static Logger log = LoggerFactory.getLogger(CoresetClientServerTest.class);

    @ClassRule
    public static CoresetExternalServer server = CoresetExternalServer.create(9191);

/*
    @ClassRule
    public static CoresetExternalServer server = CoresetExternalServer.create(9191, 9192, 9193, 9194, 9195);

*/
    private final static int k = 25;

    private final static int sampleSize = 1000;

    private final static int batchSize = 10_000;

    private final static int N = 100_000;

    private final static int dim = 10;

    @Test
    public void startNonUniformCoresetServerTest() throws InterruptedException {
        final ClientConfig clientConfig = ClientConfig.init()
                .k(k)
                .coresetAlgName(CoresetAlgorithm.NON_UNIFORM)
                .sampleSize(sampleSize)
                .batchSize(batchSize)
                .hosts("localhost:9191")
//                .hosts("localhost:9191,localhost:9192,localhost:9193,localhost:9194,localhost:9195")
                .create();

        log.info("Starting client");
        final CoresetClient client = new CoresetClient(clientConfig, new RandomPointsProvider(N, dim));
        client.initialize();
        client.distributeAllPoints();
        // Close connections to all servers
        log.info("XXXXX: Waiting for all servers to finish computations.");
        while (!client.isAllDone()) {
            TimeUnit.MILLISECONDS.sleep(100);
        }
        log.info("XXXXX: Total energy = " + client.computeTotalEnergy());
        client.closeAll();
    }

    @Test
    public void startNonUniformCoresetStandaloneTest() {

        final StreamingAlgorithm streaming = new StreamingAlgorithm(
                CoresetAlgorithmFactory.createNonUniformAlgorithm(k, sampleSize));

        final RandomPointsProvider provider = new RandomPointsProvider(N, dim);

        while (true) {
            final List<CoresetWeightedPoint> points = provider.nextBatchOfCoresetPoints(batchSize);
            if (points.isEmpty()) {
                break;
            }
            streaming.addDataset(CoresetUtils.toSparseWeightableVectors(points));
        }

        final List<SparseWeightableVector> coreset = streaming.getTotalCoreset();

        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(k);
        final List<SparseWeightableVector> centers = kmeans.cluster(coreset)
                .stream()
                .map(c -> new SparseWeightableVector(c.getCenter().getVector()))
                .collect(Collectors.toList());
        // Starting over again
        provider.reset();

        double totalEnergy = 0d;

        while (true) {
            final List<SparseWeightableVector> points = CoresetUtils.toSparseWeightableVectors(provider.nextBatchOfCoresetPoints(batchSize));
            if (points.isEmpty()) {
                break;
            }

            totalEnergy += CoresetUtils.getEnergy(centers, points);
        }
        System.out.println("XXXXX: Total energy = " + totalEnergy);
    }

    @Test
    public void testNonUniformCoresetNoStreaming() {

        final SparseCoresetAlgorithm algorithm = CoresetAlgorithmFactory.createNonUniformAlgorithm(20, 1_000);

        final RandomPointsProvider pointsProvider = new RandomPointsProvider(50_000, 3);

        final List<SparseWeightableVector> points = CoresetUtils.toSparseWeightableVectors(
                pointsProvider.nextBatchOfCoresetPoints(50_000));

        System.out.println("1. Total weight = " + points
                .stream()
                .map(x -> x.getWeight())
                .collect(Collectors.summingDouble(x -> x)));

        final List<SparseWeightableVector> sample = algorithm.takeSample(points);

        System.out.println("2. Total weight = " + sample
                .stream()
                .map(x -> x.getWeight())
                .collect(Collectors.summingDouble(x -> x)));
    }

}
