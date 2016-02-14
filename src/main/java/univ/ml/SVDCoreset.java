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

    private final double epsilon;

    private final int d;

    private final int m;

    public SVDCoreset(final int d, final int j, final double epsilon) {
        this.j = j;
        this.epsilon = epsilon;
        this.d = d;

        m = Math.min(j + (int)Math.ceil(j/epsilon) - 1, d - 1);

    }

    @Override
    public List<WeightedDoublePoint> reduce(List<WeightedDoublePoint> pointset) {
        final BlockRealMatrix A = new BlockRealMatrix(pointset.size(), d);
        int idx = 0;

        // Create block matrix so SVD algorithm will be able to work with it.
        for (WeightedDoublePoint each : pointset) {
            A.setRow(idx++, each.getPoint());
        }

        SingularValueDecomposition svd = new SingularValueDecompositionImpl(A);
        final RealMatrix S = svd.getS().getSubMatrix(0, m, 0, m);
        final RealMatrix VT = svd.getV().getSubMatrix(0, d - 1, 0, m).transpose();

        // If point set is just a merge of two or more data chunks, need to
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
