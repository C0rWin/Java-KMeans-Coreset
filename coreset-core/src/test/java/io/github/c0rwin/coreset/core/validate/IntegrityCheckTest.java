package io.github.c0rwin.coreset.core.validate;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.algo.TestData;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class IntegrityCheckTest {

    @Test
    void meanCoresetPassesEveryCheck() {
        Coreset p = new Coreset(TestData.mixture(500, 4, 3, true, 1).points(), 1.5);
        Coreset c = new MeanCoreset(20, 10).compress(p, new SplittableRandom(2));
        IntegrityCheck.Report r = IntegrityCheck.check(p, c);
        assertTrue(r.weightRelErr() < 1e-12, r.toString());
        assertTrue(r.firstMomentRelErr() < 1e-12, r.toString());
        assertTrue(r.secondMomentRelErr() < 1e-12, r.toString());
        assertEquals(0, r.negativeWeights());
        assertEquals(0, r.nonFinite());
        assertEquals(0, IntegrityCheck.oneSidedViolations(p, c, queries(200, 3), 1e-12));
    }

    /** The legacy mean coreset emitted cluster means without the additive term; the oracle must flag that. */
    @Test
    void meansWithoutAdditiveOffsetAreFlagged() {
        Coreset p = TestData.mixture(500, 4, 3, true, 3);
        Coreset c = new MeanCoreset(20, 10).compress(p, new SplittableRandom(4));
        Coreset withoutOffset = Coreset.of(c.points());
        // Q = the means themselves costs 0 on the offset-free coreset but not on P: always a violation.
        List<SparseVector> means = new ArrayList<>();
        for (WeightedPoint x : c.points()) {
            means.add(x.vector());
        }
        assertEquals(1, IntegrityCheck.oneSidedViolations(p, withoutOffset, List.of(means), 1e-12));
        assertTrue(IntegrityCheck.oneSidedViolations(p, withoutOffset, queries(200, 3), 1e-12) > 0);
        IntegrityCheck.Report r = IntegrityCheck.check(p, withoutOffset);
        assertTrue(r.weightRelErr() < 1e-12 && r.firstMomentRelErr() < 1e-12);
        assertTrue(r.secondMomentRelErr() > 1e-3, r.toString());
    }

    @Test
    void relativeErrorsAgainstKnownValues() {
        Coreset input = new Coreset(List.of(point(1, 1, 0), point(1, 3, 0)), 0.0);
        // Weight 3 instead of 2, first moment (6,0) instead of (4,0), second 12 + 1 instead of 10.
        Coreset coreset = new Coreset(List.of(point(3, 2, 0)), 1.0);
        IntegrityCheck.Report r = IntegrityCheck.check(input, coreset);
        assertEquals(0.5, r.weightRelErr(), 1e-15);
        assertEquals(0.5, r.firstMomentRelErr(), 1e-15);
        assertEquals(0.3, r.secondMomentRelErr(), 1e-15);
    }

    @Test
    void zeroReferenceReportsAbsoluteError() {
        Coreset input = Coreset.of(List.of(point(1, -1), point(1, 1)));
        Coreset coreset = Coreset.of(List.of(point(2, 0.25)));
        assertEquals(0.5, IntegrityCheck.check(input, coreset).firstMomentRelErr(), 1e-15);
    }

    @Test
    void overflowingSecondMomentIsNonFinite() {
        Coreset c = Coreset.of(List.of(new WeightedPoint(SparseVector.dense(new double[] {1e200}), 1.0)));
        assertEquals(1, IntegrityCheck.check(c, c).nonFinite());
    }

    @Test
    void rejectsDimensionMismatch() {
        assertThrows(IllegalArgumentException.class, () -> IntegrityCheck.check(
                Coreset.of(List.of(point(1, 0))), Coreset.of(List.of(point(1, 0, 0)))));
    }

    private static List<List<SparseVector>> queries(int count, int dim) {
        SplittableRandom rnd = new SplittableRandom(99);
        List<List<SparseVector>> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            out.add(TestData.randomCenters(1 + rnd.nextInt(6), dim, 15.0, rnd));
        }
        return out;
    }
}
