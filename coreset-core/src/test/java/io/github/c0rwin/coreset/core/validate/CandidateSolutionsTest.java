package io.github.c0rwin.coreset.core.validate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.algo.TestData;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class CandidateSolutionsTest {

    @Test
    void standardSetIsSeededAndLabelled() {
        Coreset p = TestData.mixture(400, 3, 2, false, 1);
        Coreset c = new UniformCoreset(60).compress(p, new SplittableRandom(2));
        List<Candidate> a = CandidateSolutions.standard(p, c, 3, 5, 2, new SplittableRandom(3));
        List<Candidate> b = CandidateSolutions.standard(p, c, 3, 5, 2, new SplittableRandom(3));
        assertEquals(a, b);
        assertEquals(List.of("kmeans++@P#0", "kmeans++@P#1", "lloyd@C#0", "lloyd@C#1", "lloyd@P#0", "lloyd@P#1",
                "convex@B#0", "convex@B#1"), a.stream().map(Candidate::generator).toList());
        for (Candidate x : a) {
            assertEquals(3, x.centers().size(), x.generator());
        }
    }

    /** Convex combinations stay inside the bounding box of the data, whose bicriteria centers are input points. */
    @Test
    void convexCombinationsLieInsideTheData() {
        Coreset p = TestData.mixture(300, 4, 2, true, 4);
        double[] lo = {Double.MAX_VALUE, Double.MAX_VALUE};
        double[] hi = {-Double.MAX_VALUE, -Double.MAX_VALUE};
        p.points().forEach(x -> {
            for (int d = 0; d < 2; d++) {
                lo[d] = Math.min(lo[d], x.vector().get(d));
                hi[d] = Math.max(hi[d], x.vector().get(d));
            }
        });
        for (Candidate x : CandidateSolutions.convexCombinationsOfBicriteria(p, 4, 20, new SplittableRandom(5))) {
            for (SparseVector v : x.centers()) {
                for (int d = 0; d < 2; d++) {
                    assertTrue(v.get(d) >= lo[d] - 1e-9 && v.get(d) <= hi[d] + 1e-9, v.toString());
                }
            }
        }
    }
}
