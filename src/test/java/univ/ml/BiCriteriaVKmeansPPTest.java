package univ.ml;

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
import org.junit.runners.Parameterized.Parameters;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.BiCriteriaSeedingAlgorithm;
import univ.ml.sparse.algorithm.KmeansPlusPlusSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

@RunWith(Parameterized.class)
public class BiCriteriaVKmeansPPTest {

    private static List<SparseWeightableVector> pointSet;

    private static int pointsToRead = 50_000;

    @Parameter
    public int k;

    @Parameter(1)
    public int percintile;

    @Parameters
    static public Collection<Object[]> setup() throws Exception {
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

        System.out.println("BiCriteria;KMeans Seed");
        System.out.println();

        return Arrays.asList(new Object[][] {
                {10, 50},
                {15, 50},
                {20, 50},
                {25, 50},
                {30, 50},
                {35, 50}
        });
    }

    @Test
    public void biCriteriaVKmeansSeed() {
        final SparseSeedingAlgorithm bicriteria = new BiCriteriaSeedingAlgorithm(k, percintile);

        final List<SparseCentroidCluster> seed1 = bicriteria.seed(pointSet);

        final SparseSeedingAlgorithm kmeansSeeding = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(k));

        final List<SparseCentroidCluster> seed2 = kmeansSeeding.seed(pointSet);

        final SparseWSSE wsse = new SparseWSSE();

        final double cost1 = wsse.getCost(seed1);
        final double cost2 = wsse.getCost(seed2);

        System.out.println(MessageFormat.format("{0,number,#};{1,number,#}", cost1, cost2));

    }
}
