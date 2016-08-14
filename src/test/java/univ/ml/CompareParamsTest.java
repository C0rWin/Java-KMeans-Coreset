package univ.ml;

import static org.junit.runners.Parameterized.Parameters;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.MessageFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

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

@RunWith(Parameterized.class)
public class CompareParamsTest {

    private static int pointsToRead = 20_000;

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
        final int K = 15;

        if (!new File("pointset.dat").exists()) {
            final MNISTReader reader = new MNISTReader("/Users/bartem/sandbox/coreset/src/main/resources/mnist_train.csv");
            pointSet = reader.readPoints(pointsToRead);

            FileOutputStream fout = new FileOutputStream("pointset.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fout);
            oos.writeObject(pointSet);
        } else {
            final FileInputStream fis = new FileInputStream("pointset.dat");
            final ObjectInputStream ois = new ObjectInputStream(fis);
            pointSet = (List<SparseWeightableVector>)ois.readObject();
        }

        List<SparseCentroidCluster> realClusters;
        if (!new File("clusters.dat").exists()) {
            final SparseWeightedKMeansPlusPlus transform = new SparseWeightedKMeansPlusPlus(K);
            // Compute baseline on entire dataset
            realClusters = transform.cluster(pointSet);

            FileOutputStream fout = new FileOutputStream("clusters.dat");
            ObjectOutputStream oos = new ObjectOutputStream(fout);
            oos.writeObject(realClusters);
        } else {
            final FileInputStream fis = new FileInputStream("clusters.dat");
            final ObjectInputStream ois = new ObjectInputStream(fis);
            realClusters = (List<SparseCentroidCluster>)ois.readObject();
        }

        SparseWSSE wsse = new SparseWSSE();
        realEnergy = wsse.getCost(realClusters);

        System.out.println("Coreset size;Real Energy; Non-Uniform Energy; Uniform Energy");
        System.out.println();

        final int batchSize = pointsToRead / 5;
        return Arrays.asList(new Object[][]{
                {K, 500, batchSize},
                {K, 1000, batchSize},
                {K, 1500, batchSize},
                {K, 2000, batchSize},
                {K, 2500, batchSize},
                {K, 3000, batchSize}
        });
    }

    @Test
    public void compare() {
        // NonUniform coreset
        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(k);
        final SparseCoresetEvaluator evaluator = new SparseCoresetEvaluator(kmeans);
//        final SparseSeedingAlgorithm seedingAlgorithm = new BiCriteriaSeedingAlgorithm(k, 50);
        final SparseSeedingAlgorithm seedingAlgorithm = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(k, 2));

        final double nonUniformEnergy = evaluator.evaluate(new SparseNonUniformCoreset(seedingAlgorithm, sampleSize), pointSet, batchSize);

        // Uniform coreset
        final double uniformEnergy = evaluator.evaluate(new SparseUniformCoreset(sampleSize), pointSet, batchSize);

        System.out.println(MessageFormat.format("{0,number,#};{1,number,#};{2,number,#};{3,number,#}",
                sampleSize, realEnergy, nonUniformEnergy, uniformEnergy));

    }

}
