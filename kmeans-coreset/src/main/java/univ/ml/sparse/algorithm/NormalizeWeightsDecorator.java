package univ.ml.sparse.algorithm;

import java.util.List;

import univ.ml.sparse.SparseWeightableVector;

public class NormalizeWeightsDecorator implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = -6832466693319204731L;

    private final SparseCoresetAlgorithm delegator;

    public NormalizeWeightsDecorator(final SparseCoresetAlgorithm delegator) {
        this.delegator = delegator;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        double before = 0d;
        for (final SparseWeightableVector each : pointset) {
            before += each.getWeight();
        }

        final List<SparseWeightableVector> sample = delegator.takeSample(pointset);

        double after = 0d;
        for (final SparseWeightableVector each : sample) {
            after += each.getWeight();
        }

        // Normalize points weights to sum up into original weighted sum.
        double ratio = before / after;

        for (SparseWeightableVector point : sample) {
            point.setWeight(point.getWeight() * ratio);
        }

        return sample;
    }
}
