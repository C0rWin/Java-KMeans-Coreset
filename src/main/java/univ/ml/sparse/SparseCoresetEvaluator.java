package univ.ml.sparse;

import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

import java.util.List;
import java.util.stream.Collectors;

public class SparseCoresetEvaluator {

    private final SparseWeightedKMeansPlusPlus clusterer;

    private final SparseWSSE costFunction = new SparseWSSE();

    public SparseCoresetEvaluator(final SparseWeightedKMeansPlusPlus clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final SparseCoresetAlgorithm algorithm, final List<SparseWeightableVector> pointSet) {
        final List<SparseCentroidCluster> clusters = clusterer.cluster(algorithm.takeSample(pointSet));

        final List<SparseWeightableVector> centers = clusters.stream()
                .map(cluster -> new SparseWeightableVector(cluster.getCenter().getVector(), 1))
                .collect(Collectors.toList());

        return costFunction.getCost(centers, pointSet);
    }

}
