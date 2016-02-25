package univ.ml.sparse;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

import java.util.List;
import java.util.Set;

public class SparseSVDCoreset implements SparseCoresetAlgorithm {
    private final int j;

    private final int size;

    public SparseSVDCoreset(final int j, final int size) {
        this.j = j;
        this.size = size;
    }

    @Override
    public List<SparseWeightableVector> takeSample(List<SparseWeightableVector> pointset) {
        if (size >= pointset.size())
            return pointset;

        int d = pointset.size();

        final OpenMapRealMatrix A = new OpenMapRealMatrix(pointset.size(), d);
        int idx = 0;

        // Create block matrix so SVD algorithm will be able to work with it.
        for (SparseWeightableVector each : pointset) {
            A.setRowVector(idx++, each.getVector());
        }

        SingularValueDecomposition svd = new SingularValueDecomposition(A);
        final RealMatrix S = svd.getS().getSubMatrix(0, size, 0, size);
        final RealMatrix VT = svd.getV().getSubMatrix(0, d - 1, 0, size).transpose();

        // If point set is just a concat of two or more data chunks, need to
        // go over all points and collect distinct weights values, then to sum
        // everything into a new weight.
        double totalWeight = 0;
        Set<Double> weights = Sets.newHashSet();

        for (SparseWeightableVector vector : pointset) {
            weights.add(vector.getWeight());
        }

        for (Double weight : weights) {
            totalWeight += weight;
        }

        // If totalWeight is zero that means no one actually computed weights for these points,
        // hence they are new, therefore totalWeight computed according to the formula.
        if (totalWeight == 0)
            totalWeight = Math.pow(A.getFrobeniusNorm(), 2) - Math.pow(S.getFrobeniusNorm(), 2);

        // Get the coreset.
        final RealMatrix C = S.multiply(VT);

        // Copy into final list.
        final List<SparseWeightableVector> results = Lists.newArrayList();
        for (int i = 0; i < C.getRowDimension(); ++i) {
            results.add(new SparseWeightableVector(C.getRow(i), totalWeight));
        }
        return results;
    }
}
