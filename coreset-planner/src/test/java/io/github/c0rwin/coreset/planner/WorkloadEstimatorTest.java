package io.github.c0rwin.coreset.planner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.testbed.data.PointFileWriter;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WorkloadEstimatorTest {

    @Test
    void pilotExtrapolatesKappaPtToTheFullInput() {
        PointSet data = Fixtures.mixture();
        WorkloadProfile w = Fixtures.mixtureProfile();
        assertEquals(data.size(), w.n());
        for (int q = 0; q < w.candidateCount(); q++) {
            List<SparseVector> centers = w.tables().centers().get(q);
            double total = 0.0;
            double sq = 0.0;
            for (int i = 0; i < data.size(); i++) {
                SparseVector v = data.vector(i);
                double c = v.squaredDistance(centers.get(Cost.nearest(v, centers)));
                total += c;
                sq += c * c;
            }
            double exact = sq / (total * total);
            double est = w.candidates().get(q).kappaPt();
            assertEquals(1.0, est / exact, 0.15, w.candidates().get(q).name());
        }
    }

    @Test
    void sensitivityFamilyTotalIsTheStandardBound() {
        WorkloadProfile w = Fixtures.mixtureProfile();
        double alpha = io.github.c0rwin.coreset.core.algo.SensitivityBound.kMeansPlusPlusAlpha(Fixtures.K);
        assertEquals(6 * alpha + 4 * Fixtures.K, w.families().get(Family.SENSITIVITY).sTotal(), 1e-6);
        for (Family f : new Family[] {Family.UNIFORM, Family.SENSITIVITY, Family.LIGHTWEIGHT}) {
            WorkloadProfile.FamilyStats s = w.families().get(f);
            for (int q = 0; q < w.candidateCount(); q++) {
                // sum c^2/q / C^2 <= max c/(qC) <= S: the bound dominates the instance variance
                assertTrue(s.sigma2()[q] + 1.0 <= s.range()[q] + 1e-9, f + " candidate " + q);
                assertTrue(s.range()[q] <= s.sTotal() + 1e-6, f + " candidate " + q);
            }
        }
    }

    @Test
    void fileAndInMemoryPassesAgree(@TempDir Path dir) throws Exception {
        PointSet data = PointSet.of(io.github.c0rwin.coreset.core.PointBlock.of(
                io.github.c0rwin.coreset.testbed.data.SyntheticData.gaussianMixture(5_000, 3, 3, 1.0, 0.0, 10.0, 1L)));
        Path f = dir.resolve("d.cbin");
        try (PointFileWriter wr = PointFileWriter.dense(f, 3, 0)) {
            for (int i = 0; i < data.size(); i++) {
                wr.writeRow(data.vector(i).toDense());
            }
            wr.finish();
        }
        WorkloadEstimator est = new WorkloadEstimator(1_000, 3, 5L, false);
        WorkloadProfile a = est.estimate("x", f);
        WorkloadProfile b = est.estimate("x", data);
        assertEquals(a.n(), b.n());
        assertEquals(a.kmeansppCost(), b.kmeansppCost(), 1e-9 * a.kmeansppCost());
        assertEquals(a.candidates(), b.candidates());
    }
}
