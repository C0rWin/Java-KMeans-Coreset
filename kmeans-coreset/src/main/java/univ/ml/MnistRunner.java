package univ.ml;

import java.io.BufferedReader;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.KmeansPlusPlusSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseCoresetUtility;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseSeedingAlgorithm;
import univ.ml.sparse.algorithm.SparseUniformCoreset;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

public class MnistRunner {

    public static void main(String[] args) throws Exception {
        final URL mnistResource = MnistRunner.class.getClassLoader().getResource("mnist_train.csv");
        final Path mnistTrain = Paths.get(mnistResource.toURI());
        final List<SparseWeightableVector> pointset = Lists.newArrayList();
        try (final BufferedReader reader = Files.newBufferedReader(mnistTrain, Charset.defaultCharset())) {
            String line = null;

            while ((line = reader.readLine())!= null) {
                final List<String> point = Splitter.on(",").splitToList(line);
                final Map<Integer, Double> _coords = Maps.newHashMap();
                for (int i = 1; i < point.size(); i++) {
                    _coords.put(i - 1, Double.valueOf(point.get(i)));
                }
                pointset.add(new SparseWeightableVector(_coords, 1d, point.size()));
            }

            System.out.println("k;sampleSize;opt;nonUniform;Uniform;eps_nonUniform;eps_uniform");

            int K = 25;

            List<Integer> sampleSizes = Lists.newArrayList(50, 100);
//            List<Integer> sampleSizes = Lists.newArrayList(50, 100, 400, 500, 1000, 1500);

            for (final Integer sampleSize : sampleSizes) {
                final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(K);

                final SparseSeedingAlgorithm seeding = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(K, 2));

                final SparseCoresetAlgorithm nonUniformCoreset = new SparseNonUniformCoreset(seeding, sampleSize);
                final SparseCoresetAlgorithm uniformCoreset = new SparseUniformCoreset(sampleSize);

                final List<SparseWeightableVector> nonUniformSample = nonUniformCoreset.takeSample(pointset);
                final List<SparseWeightableVector> uniformSample = uniformCoreset.takeSample(pointset);

                final List<SparseCentroidCluster> optC = kmeans.cluster(pointset);
                final List<SparseCentroidCluster> nonUniformC = kmeans.cluster(nonUniformSample);
                final List<SparseCentroidCluster> uniformC = kmeans.cluster(uniformSample);

                final double optEnergy = SparseCoresetUtility.getEnergy(optC, pointset);
                final double nonUniformEnergy = SparseCoresetUtility.getEnergy(nonUniformC, pointset);
                final double uniformEnergy = SparseCoresetUtility.getEnergy(uniformC, pointset);

                System.out.println(K + ";" + sampleSize + ";" + optEnergy + ";" + nonUniformEnergy +";" + uniformEnergy +";"
                        + (nonUniformEnergy/optEnergy - 1) + ";" + (uniformEnergy/optEnergy - 1) + ";");
            }
        }
    }
}
