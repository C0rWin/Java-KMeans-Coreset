package univ.ml;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.KmeansPlusPlusSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseUniformCoreset;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.MessageFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public class CompareParamsTest {

    private static int pointsToRead = 50_000;

    private static List<SparseWeightableVector> pointSet;

    private static double realEnergy;

    @Parameter
    public int k;

    @Parameter(1)
    public int sampleSize;

    @Parameter(2)
    public int batchSize;

    @Parameters
    public static Collection<Object[]> data() throws Exception {
        final int K = 30;

        final String pointsFileName = "pointset_" + pointsToRead + ".dat";
        if (!new File(pointsFileName).exists()) {
            final MNISTReader reader = new MNISTReader("/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv");
            pointSet = reader.readPoints(pointsToRead);

            try (FileOutputStream fout = new FileOutputStream(pointsFileName)) {
                ObjectOutputStream oos = new ObjectOutputStream(fout);
                oos.writeObject(pointSet);
            }
        } else {
            try (final FileInputStream fis = new FileInputStream(pointsFileName)) {
                final ObjectInputStream ois = new ObjectInputStream(fis);
                pointSet = (List<SparseWeightableVector>) ois.readObject();
            }
        }

        List<SparseCentroidCluster> realClusters;
        final String clustersFileName = "clusters_" + pointsToRead + "_" + K + ".dat";
        if (!new File(clustersFileName).exists()) {
            final SparseWeightedKMeansPlusPlus transform = new SparseWeightedKMeansPlusPlus(K);
            // Compute baseline on entire dataset
            realClusters = transform.cluster(pointSet);

            try (FileOutputStream fout = new FileOutputStream(clustersFileName)) {
                ObjectOutputStream oos = new ObjectOutputStream(fout);
                oos.writeObject(realClusters);
            }
        } else {
            try (final FileInputStream fis = new FileInputStream(clustersFileName)) {
                final ObjectInputStream ois = new ObjectInputStream(fis);
                realClusters = (List<SparseCentroidCluster>) ois.readObject();
            }
        }

        SparseWSSE wsse = new SparseWSSE();
        realEnergy = wsse.getCost(realClusters);

        System.out.println("Coreset size;Real Energy; Non-Uniform Energy; Uniform Energy");
        System.out.println();

        final int batchSize = pointsToRead / 1;
        return Arrays.asList(new Object[][]{
                {K, 50, batchSize},
                {K, 75, batchSize},
                {K, 100, batchSize},
                {K, 125, batchSize},
                {K, 150, batchSize},
                {K, 175, batchSize},
                {K, 200, batchSize},
                {K, 225, batchSize},
                {K, 250, batchSize},
                {K, 275, batchSize},
                {K, 300, batchSize},
                {K, 325, batchSize},
                {K, 350, batchSize},
                {K, 375, batchSize},
                {K, 400, batchSize},
                {K, 425, batchSize},
                {K, 450, batchSize},
                {K, 475, batchSize},
                {K, 500, batchSize},
                {K, 1000, batchSize},
                {K, 1250, batchSize},
                {K, 1500, batchSize},
                {K, 1750, batchSize},
                {K, 2000, batchSize},
                {K, 2250, batchSize},
                {K, 2500, batchSize},
                {K, 2750, batchSize},
                {K, 3000, batchSize},
        });
    }

    @Test
    public void compare() {

        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(k);
        final SparseCoresetEvaluator evaluator = new SparseCoresetEvaluator(kmeans, pointSet, batchSize);
        final SparseSeedingAlgorithm seedingAlgorithm = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(k, 5));

        // NonUniform coreset
        final double nonUniformEnergy = evaluator.evaluate(new SparseNonUniformCoreset(seedingAlgorithm, sampleSize));

        // Uniform coreset
        final double uniformEnergy = evaluator.evaluate(new SparseUniformCoreset(sampleSize));

        System.out.println(MessageFormat.format("{0,number,#};{1,number,#};{2,number,#};{3,number,#}",
                sampleSize, realEnergy, nonUniformEnergy, uniformEnergy));
    }

}