package univ.ml.sparse;

import java.io.Serializable;
import java.util.List;

import com.google.common.collect.Lists;

import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

public class SparseCoresetEvaluator implements Serializable {

    private static final long serialVersionUID = -7660109554383619879L;

    private final SparseWeightedKMeansPlusPlus clusterer;

    private final SparseWSSE costFunction = new SparseWSSE();

    public SparseCoresetEvaluator(final SparseWeightedKMeansPlusPlus clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final SparseCoresetAlgorithm algorithm, final List<SparseWeightableVector> pointSet) {
        final List<SparseCentroidCluster> clusters = clusterer.cluster(algorithm.takeSample(pointSet));

        final List<SparseWeightableVector> centers = Lists.newArrayList();

//        for (SparseCentroidCluster cluster : clusters) {
//            centers.add(new SparseWeightableVector(cluster.getCenter().getVector(), 1));
//        }

        return costFunction.getCost(centers, pointSet);
    }

}
