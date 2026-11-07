package univ.ml.sparse.algorithm.streaming;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Test;

import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

public class StreamingAlgorithmTest {

    /**
     * Like the repo's algorithms, returns its input list itself when it is small enough; otherwise
     * collapses it into one point at the weighted mean carrying the total weight.
     */
    private static final class CollapseToMean implements SparseCoresetAlgorithm {
        private static final long serialVersionUID = 1L;
        private final int sampleSize;

        CollapseToMean(final int sampleSize) {
            this.sampleSize = sampleSize;
        }

        @Override
        public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
            if (pointset.size() <= sampleSize) {
                return pointset;
            }
            double weight = 0;
            double weightedSum = 0;
            for (SparseWeightableVector p : pointset) {
                weight += p.getWeight();
                weightedSum += p.getWeight() * p.getEntry(0);
            }
            return new ArrayList<>(Arrays.asList(point(weightedSum / weight, weight)));
        }
    }

    private static SparseWeightableVector point(final double x, final double weight) {
        return new SparseWeightableVector(new double[] {x}, weight);
    }

    private static double totalWeight(final List<SparseWeightableVector> points) {
        return points.stream().mapToDouble(SparseWeightableVector::getWeight).sum();
    }

    private static List<String> describe(final List<SparseWeightableVector> points) {
        return points.stream().map(p -> p.getEntry(0) + "@" + p.getWeight()).sorted().collect(Collectors.toList());
    }

    @Test
    public void addDatasetDoesNotModifyCallerList() {
        final StreamingAlgorithm streaming = new StreamingAlgorithm(new CollapseToMean(4));
        final List<SparseWeightableVector> first = new ArrayList<>(Arrays.asList(point(1, 1)));
        final List<SparseWeightableVector> second = new ArrayList<>(Arrays.asList(point(2, 1)));

        streaming.addDataset(first);
        streaming.addDataset(second);

        assertEquals(Arrays.asList("1.0@1.0"), describe(first));
        assertEquals(Arrays.asList("2.0@1.0"), describe(second));
    }

    @Test
    public void treeKeepsNoReferenceToCallerList() {
        final StreamingAlgorithm streaming = new StreamingAlgorithm(new CollapseToMean(4));
        final List<SparseWeightableVector> batch = new ArrayList<>(Arrays.asList(point(1, 1)));

        streaming.addDataset(batch);
        batch.clear();

        assertEquals(Arrays.asList("1.0@1.0"), describe(streaming.getTotalCoreset()));
    }

    @Test
    public void unmodifiableInputIsAccepted() {
        final StreamingAlgorithm streaming = new StreamingAlgorithm(new CollapseToMean(4));

        streaming.addDataset(List.of(point(1, 1)));
        streaming.addDataset(List.of(point(2, 1)));

        assertEquals(Arrays.asList("1.0@1.0", "2.0@1.0"), describe(streaming.getTotalCoreset()));
    }

    @Test
    public void totalCoresetIsUnionOfTreeLevelsWithoutExtraCompression() {
        final StreamingAlgorithm streaming = new StreamingAlgorithm(new CollapseToMean(2));

        // Level 1 holds {a, b} (merged size 2 needs no compression), level 0 holds {c}.
        streaming.addDataset(new ArrayList<>(Arrays.asList(point(1, 1))));
        streaming.addDataset(new ArrayList<>(Arrays.asList(point(2, 2))));
        streaming.addDataset(new ArrayList<>(Arrays.asList(point(5, 3))));

        assertEquals(Arrays.asList("1.0@1.0", "2.0@2.0", "5.0@3.0"), describe(streaming.getTotalCoreset()));
    }

    @Test
    public void totalWeightIsConservedThroughTheTree() {
        final StreamingAlgorithm streaming = new StreamingAlgorithm(new CollapseToMean(2));
        double inputWeight = 0;
        for (int i = 0; i < 13; i++) {
            final List<SparseWeightableVector> batch = new ArrayList<>(Arrays.asList(point(i, i + 1), point(-i, 0.5)));
            inputWeight += totalWeight(batch);
            streaming.addDataset(batch);
        }

        final List<SparseWeightableVector> total = streaming.getTotalCoreset();

        assertEquals(inputWeight, totalWeight(total), 1e-9);
    }

    @Test
    public void emptyTreeYieldsEmptyCoreset() {
        final SparseCoresetAlgorithm rejectsEmpty = pointset -> {
            throw new AssertionError("takeSample must not be called on an empty tree");
        };

        assertTrue(new StreamingAlgorithm(rejectsEmpty).getTotalCoreset().isEmpty());
    }
}
