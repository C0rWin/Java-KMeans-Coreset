package univ.ml.sparse;

import java.util.List;

import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

import com.google.common.collect.Lists;

public class SparseCoresetEvaluator {

    private final SparseWeightedKMeansPlusPlus clusterer;

    private final SparseWSSE costFunction = new SparseWSSE();

    public SparseCoresetEvaluator(final SparseWeightedKMeansPlusPlus clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final SparseCoresetAlgorithm algorithm, final List<SparseWeightableVector> pointSet) {
        final List<SparseCentroidCluster> clusters = clusterer.cluster(algorithm.takeSample(pointSet));

        final List<SparseWeightableVector> centers = Lists.newArrayList();

        for (SparseCentroidCluster cluster : clusters) {
            centers.add(new SparseWeightableVector(cluster.getCenter().getVector(), 1));
        }

        return costFunction.getCost(centers, pointSet);
    }

}
