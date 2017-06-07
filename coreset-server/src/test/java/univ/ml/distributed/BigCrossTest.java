package univ.ml.distributed;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.junit.Test;

import com.google.common.base.Splitter;
import com.google.common.collect.Maps;

import univ.ml.distributed.coreset.CoresetAlgorithmFactory;
import univ.ml.distributed.coreset.CoresetUtils;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

public class BigCrossTest {

    @Test
    public void bigCrossData() throws IOException {
        final Path path = Paths.get("/Users/bartem/sandbox/bigcross/test.data");
        final BufferedReader reader = Files.newBufferedReader(path);

        List<SparseWeightableVector> points1 = new ArrayList<>();
        List<SparseWeightableVector> points2 = new ArrayList<>();
        List<SparseWeightableVector> o = new ArrayList<>();
        String line;

        while ((line = reader.readLine()) != null) {
            final List<String> data = Splitter.on(",").splitToList(line);
            final Map<Integer, Double> coords = Maps.newHashMap();
            for (int i = 0; i < data.size(); i++) {
                coords.put(i, Double.valueOf(data.get(i)));
            }
            points1.add(new SparseWeightableVector(coords, 1d, coords.size()));
            points2.add(new SparseWeightableVector(coords, 1d, coords.size()));
            o.add(new SparseWeightableVector(coords, 1d, coords.size()));
        }

        final SparseCoresetAlgorithm nonUniformAlgorithm = CoresetAlgorithmFactory.createNonUniformAlgorithm(20, 15000);
        final SparseCoresetAlgorithm uniformAlgorithm = CoresetAlgorithmFactory.createUniformAlgorithm(15000);

        final List<SparseWeightableVector> sample1 = nonUniformAlgorithm.takeSample(points1);
        final List<SparseWeightableVector> sample2 = uniformAlgorithm.takeSample(points2);

        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(20);

        final List<SparseCentroidCluster> c1 = kmeans.cluster(sample1);
        final List<SparseCentroidCluster> c2 = kmeans.cluster(sample2);

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


        System.out.println(nonUniformEnergy + ";" + uniformEnergy);
    }
}
