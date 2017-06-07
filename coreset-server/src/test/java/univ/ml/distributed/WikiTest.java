package univ.ml.distributed;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import com.google.common.base.Splitter;

import univ.ml.distributed.coreset.CoresetAlgorithmFactory;
import univ.ml.distributed.coreset.CoresetUtils;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.NormalizeWeightsDecorator;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

@RunWith(Parameterized.class)
public class WikiTest {

    @Parameterized.Parameters
    public static Iterable<Object> data() {
        return Arrays.asList(new Object[][]{
                {10, 32}, {10, 64}, {10, 128}, {10, 256},
                {16, 32}, {16, 64}, {16, 128}, {16, 256},
                {32, 64}, {32, 128}, {32, 256},
                {64, 128}, {64, 256},
        });
    }

    @Parameterized.Parameter
    public int K;

    @Parameterized.Parameter(1)
    public int SAMPLE_SIZE = 128;

    @Test
    public void wikiDataset() throws IOException {
        System.out.println("K;Sample Size;NonUniform;Uniform;KMeans;t1;t2;t3");


        for (int k = 1; k <= 3; k++) {
            final Path path = Paths.get("/Users/bartem/sandbox/roy_coreset/wiki_large" + k + ".txt");
            final BufferedReader reader = Files.newBufferedReader(path);
            String line = null;

            final List<SparseWeightableVector> d1 = new ArrayList<>();
            final List<SparseWeightableVector> d2 = new ArrayList<>();
            final List<SparseWeightableVector> d3 = new ArrayList<>();
            final List<SparseWeightableVector> o = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                final List<String> data = Splitter.on(" ").splitToList(line);
                final Map<Integer, Double> coord = new HashMap<>();
                for (int i = 2; i < data.size(); i += 2) {
                    coord.put(Integer.valueOf(data.get(i)), Double.valueOf(data.get(i + 1)));
                }
                d1.add(new SparseWeightableVector(coord, 1d, Integer.valueOf(data.get(1))));
                d2.add(new SparseWeightableVector(coord, 1d, Integer.valueOf(data.get(1))));
                d3.add(new SparseWeightableVector(coord, 1d, Integer.valueOf(data.get(1))));
                o.add(new SparseWeightableVector(coord, 1d, Integer.valueOf(data.get(1))));
            }

            final SparseCoresetAlgorithm nonUniformAlgorithm = new NormalizeWeightsDecorator(CoresetAlgorithmFactory.createNonUniformAlgorithm(K, SAMPLE_SIZE));
            final SparseCoresetAlgorithm uniformAlgorithm = CoresetAlgorithmFactory.createUniformAlgorithm(SAMPLE_SIZE);
            final SparseCoresetAlgorithm kmeansAlgorithm = CoresetAlgorithmFactory.createKmeansAlgorithm(SAMPLE_SIZE);

            long t = System.currentTimeMillis();
            final List<SparseWeightableVector> sample1 = nonUniformAlgorithm.takeSample(d1);
            long t1 = System.currentTimeMillis() - t;
            t = System.currentTimeMillis();
            final List<SparseWeightableVector> sample2 = uniformAlgorithm.takeSample(d2);
            long t2 = System.currentTimeMillis() - t;
            t = System.currentTimeMillis();
            final List<SparseWeightableVector> sample3 = kmeansAlgorithm.takeSample(d3);
            long t3 = System.currentTimeMillis() - t;

            final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(K);

            final List<SparseCentroidCluster> c1 = kmeans.cluster(sample1);
            final List<SparseCentroidCluster> c2 = kmeans.cluster(sample2);
            final List<SparseCentroidCluster> c3 = kmeans.cluster(sample3);

            final double nonUniformEnergy = CoresetUtils.getEnergy(c1.stream()
                            .map(x ->
                                    new SparseWeightableVector(x.getCenter().getVector())
                            ).collect(Collectors.toList()),
                    o);

            final double uniformEnergy = CoresetUtils.getEnergy(c2.stream()
                            .map(x ->
                                    new SparseWeightableVector(x.getCenter().getVector())
                            ).collect(Collectors.toList()),
                    o);

            final double kmeansEnergy = CoresetUtils.getEnergy(c3.stream()
                            .map(x ->
                                    new SparseWeightableVector(x.getCenter().getVector())
                            ).collect(Collectors.toList()),
                    o);


            System.out.println(K + ";" + SAMPLE_SIZE + ";" + nonUniformEnergy + ";" + uniformEnergy + ";"
                    + kmeansEnergy + ";" + t1 + ";" + t2 + ";" + t3);
        }
        System.out.println("<<<===>>>");
    }
}
