package io.github.c0rwin.coreset.planner;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.planner.model.Calibration;
import io.github.c0rwin.coreset.planner.model.PaperResilienceModel;
import io.github.c0rwin.coreset.testbed.data.SyntheticData;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** Small synthetic workloads shared by the planner tests; profiles are memoised. */
final class Fixtures {

    static final int K = 5;
    private static PointSet mixture;
    private static WorkloadProfile mixtureProfile;
    private static PointSet heavy;
    private static WorkloadProfile heavyProfile;

    private Fixtures() {
    }

    static synchronized PointSet mixture() {
        if (mixture == null) {
            mixture = PointSet.of(PointBlock.of(SyntheticData.gaussianMixture(100_000, 8, K, 1.0, 0.5, 20.0, 7L)));
        }
        return mixture;
    }

    static synchronized WorkloadProfile mixtureProfile() {
        if (mixtureProfile == null) {
            mixtureProfile = new WorkloadEstimator(10_000, K, 11L, true).estimate("mixture", mixture());
        }
        return mixtureProfile;
    }

    /** A mixture plus 20 far outliers: heavy points that merged packetization sends as single records. */
    static synchronized PointSet heavy() {
        if (heavy == null) {
            List<WeightedPoint> pts = new ArrayList<>(SyntheticData.gaussianMixture(100_000, 8, K, 1.0, 0.5, 20.0, 9L));
            SplittableRandom rnd = new SplittableRandom(3L);
            for (int i = 0; i < 20; i++) {
                double[] x = new double[8];
                for (int j = 0; j < 8; j++) {
                    x[j] = 400.0 + 50.0 * rnd.nextDouble();
                }
                pts.add(new WeightedPoint(SparseVector.dense(x), 1.0));
            }
            heavy = PointSet.of(PointBlock.of(pts));
        }
        return heavy;
    }

    static synchronized WorkloadProfile heavyProfile() {
        if (heavyProfile == null) {
            heavyProfile = new WorkloadEstimator(100_020, K, 13L, true).estimate("heavy", heavy());
        }
        return heavyProfile;
    }

    static Planner planner() {
        return new Planner(new PaperResilienceModel(Calibration.bundled()));
    }

    static EnvironmentProfile env(double rho, double burst, int nodes, double failPost) {
        return new EnvironmentProfile(rho, burst, nodes, 0.0, failPost, EnvironmentProfile.Partition.RANDOM, 0, 1.0,
                5e-4, 125e6, 1400, false);
    }
}
