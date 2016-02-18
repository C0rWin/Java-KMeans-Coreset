package univ.ml;

import com.clearspring.analytics.util.Lists;
import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.SingularValueDecomposition;
import org.apache.commons.math.linear.SingularValueDecompositionImpl;

import java.util.List;
import java.util.stream.Collectors;

public class SVDCoreset extends BaseCoreset<WeightedDoublePoint> {

    private final int j;

    private final int size;

    public SVDCoreset(final int j, final int size) {
        this.j = j;
        this.size = size;
    }

    @Override
    public List<WeightedDoublePoint> takeSample(List<WeightedDoublePoint> pointset) {
        int d = pointset.size();

        final BlockRealMatrix A = new BlockRealMatrix(pointset.size(), d);
        int idx = 0;

        // Create block matrix so SVD algorithm will be able to work with it.
        for (WeightedDoublePoint each : pointset) {
            A.setRow(idx++, each.getPoint());
        }

        SingularValueDecomposition svd = new SingularValueDecompositionImpl(A);
        final RealMatrix S = svd.getS().getSubMatrix(0, size, 0, size);
        final RealMatrix VT = svd.getV().getSubMatrix(0, d - 1, 0, size).transpose();

        // If point set is just a concat of two or more data chunks, need to
        // go over all points and collect distinct weights values, then to sum
        // everything into a new weight.
        Double totalWeight = pointset.stream()
                .map(p -> p.getWeight())
                .distinct()
                .collect(Collectors.summingDouble(x -> x));

        // If totalWeight is zero that means no one actually computed weights for these points,
        // hence they are new, therefore totalWeight computed according to the formula.
        if (totalWeight == 0)
            totalWeight = Math.pow(A.getFrobeniusNorm(), 2) - Math.pow(S.getFrobeniusNorm(), 2);

        // Get the coreset.
        final RealMatrix C = S.multiply(VT);

        // Copy into final list.
        final List<WeightedDoublePoint> results = Lists.newArrayList();
        for (int i = 0; i < C.getRowDimension(); ++i) {
            results.add(new WeightedDoublePoint(C.getRow(i), totalWeight, ""));
        }
        return results;
    }
}
