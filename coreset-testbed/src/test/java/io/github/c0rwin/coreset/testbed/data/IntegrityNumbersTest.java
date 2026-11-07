package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Integrity numbers on inputs small enough to compute by hand, via the dataset's moment summary. */
class IntegrityNumbersTest {

    private static WeightedPoint wp(double w, double... c) {
        return new WeightedPoint(SparseVector.dense(c), w);
    }

    // P = {(0,0) w1, (2,0) w1, (0,2) w2}: W = 4, sum w p = (2,4), sum w ||p||^2 = 0 + 4 + 8 = 12.
    private static final List<WeightedPoint> P = List.of(wp(1, 0, 0), wp(1, 2, 0), wp(2, 0, 2));

    @Test
    void momentSummaryCarriesTheInputMoments() {
        Dataset ds = Dataset.of("tiny", P, "", "test", "inline", List.of(), Optional.empty());
        assertEquals(4.0, ds.weightSum());
        assertEquals(2, ds.nnz()); // zeros are not stored: (0,0) has none, the others one each
        Coreset m = ds.moments();
        assertEquals(1, m.size());
        assertEquals(4.0, m.points().get(0).weight());
        assertEquals(SparseVector.dense(new double[] {0.5, 1.0}), m.points().get(0).vector());
        assertEquals(12.0 - 4.0 * 1.25, m.offset(), 1e-12); // spread around the mean
    }

    @Test
    void handComputedRelativeErrorsAgreeWithTheFullInputCheck() {
        // C = {(1,0) w2, (0,2) w1.5} + offset 1: W = 3.5, sum w p = (2,3), second moment 1 + 2 + 6 = 9.
        Coreset c = new Coreset(List.of(wp(2, 1, 0), wp(1.5, 0, 2)), 1.0);
        Dataset ds = Dataset.of("tiny", P, "", "test", "inline", List.of(), Optional.empty());

        IntegrityCheck.Report viaSummary = IntegrityCheck.check(ds.moments(), c);
        assertEquals(0.5 / 4.0, viaSummary.weightRelErr(), 1e-12);
        assertEquals(1.0 / Math.sqrt(48.0), viaSummary.firstMomentRelErr(), 1e-12); // |(0,-1)| / sqrt(W * E)
        assertEquals(3.0 / 12.0, viaSummary.secondMomentRelErr(), 1e-12);
        assertEquals(0, viaSummary.negativeWeights());
        assertEquals(0, viaSummary.nonFinite());

        IntegrityCheck.Report direct = IntegrityCheck.check(Coreset.of(P), c);
        assertEquals(direct.weightRelErr(), viaSummary.weightRelErr(), 1e-12);
        assertEquals(direct.firstMomentRelErr(), viaSummary.firstMomentRelErr(), 1e-12);
        assertEquals(direct.secondMomentRelErr(), viaSummary.secondMomentRelErr(), 1e-12);
    }

    @Test
    void anExactCoresetHasZeroErrors() {
        Dataset ds = Dataset.of("tiny", P, "", "test", "inline", List.of(), Optional.empty());
        IntegrityCheck.Report r = IntegrityCheck.check(ds.moments(), ds.moments());
        assertEquals(0.0, r.weightRelErr());
        assertEquals(0.0, r.firstMomentRelErr());
        assertEquals(0.0, r.secondMomentRelErr());
    }
}
