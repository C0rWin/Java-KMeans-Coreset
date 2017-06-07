
package univ.ml.distributed.coreset;

import univ.ml.sparse.algorithm.KmeansPlusPlusSeedingAlgorithm;
import univ.ml.sparse.algorithm.NormalizeWeightsDecorator;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseKmeansCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseNonUniformCoreset;
import univ.ml.sparse.algorithm.SparseUniformCoreset;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

public class CoresetAlgorithmFactory {

    public static SparseCoresetAlgorithm createNonUniformAlgorithm(final int k, final int sampleSize) {
        final KmeansPlusPlusSeedingAlgorithm seed = new KmeansPlusPlusSeedingAlgorithm(new SparseWeightedKMeansPlusPlus(k));
        return new NormalizeWeightsDecorator(new SparseNonUniformCoreset(seed, sampleSize));
    }

    public static SparseCoresetAlgorithm createUniformAlgorithm(final int sampleSize) {
        return new SparseUniformCoreset(sampleSize);
    }

    public static SparseCoresetAlgorithm createKmeansAlgorithm(final int sampleSize) {
        return new SparseKmeansCoresetAlgorithm(sampleSize);
    }
}
