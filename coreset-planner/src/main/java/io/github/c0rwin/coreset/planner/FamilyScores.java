package io.github.c0rwin.coreset.planner;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.algo.Clustering;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityBound;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.planner.Configuration.Family;
import java.util.random.RandomGenerator;

/**
 * Sampling distributions q of the three sampling families on a weighted point set, plus the standard sensitivity
 * bound s that every family's importance ratio s/q is measured against.
 * <ul>
 *   <li>uniform: {@code q_p = w_p / W};</li>
 *   <li>sensitivity: {@code q_p = s_p / S} with {@link SensitivityBound#standard} on a k-means++ clustering;</li>
 *   <li>lightweight (Bachem, Lucic, Krause, KDD 2018): {@code q_p = w_p/(2W) + w_p d(p, mu)^2 / (2 sum w d(., mu)^2)}.</li>
 * </ul>
 */
public final class FamilyScores {

    /** Sensitivity bound per point (sums to S = 6 alpha + 4 k'). */
    public final double[] sensitivity;
    /** Total of {@link #sensitivity}. */
    public final double sensitivityTotal;
    /** The k-means++ clustering the bound uses. */
    public final Clustering clustering;
    private final PointSet points;

    private FamilyScores(PointSet points, Clustering clustering, double[] sensitivity) {
        this.points = points;
        this.clustering = clustering;
        this.sensitivity = sensitivity;
        double s = 0.0;
        for (double v : sensitivity) {
            s += v;
        }
        this.sensitivityTotal = s;
    }

    /** One k-means++ seeding (greedy) of {@code points} and the standard bound with alpha = 16 (log2 k + 2). */
    public static FamilyScores of(PointSet points, int k, RandomGenerator rng) {
        Clustering b = Seeding.kMeansPlusPlus(0).cluster(points, k, rng, MetricsSink.NOOP);
        double[] s = SensitivityBound.standard(SensitivityBound.kMeansPlusPlusAlpha(k)).sensitivities(points, b);
        return new FamilyScores(points, b, s);
    }

    /** Sampling probabilities of {@code family} (sum 1); {@link Family#MEAN} is not a sampling family. */
    public double[] q(Family family) {
        int n = points.size();
        double[] q = new double[n];
        switch (family) {
            case UNIFORM -> {
                double w = points.totalWeight();
                for (int i = 0; i < n; i++) {
                    q[i] = points.weight(i) / w;
                }
            }
            case SENSITIVITY -> {
                for (int i = 0; i < n; i++) {
                    q[i] = sensitivity[i] / sensitivityTotal;
                }
            }
            case LIGHTWEIGHT -> {
                double[] d2 = new double[n];
                double[] mean = weightedMean(points);
                double meanNorm = 0.0;
                for (double v : mean) {
                    meanNorm += v * v;
                }
                double w = points.totalWeight();
                double spread = 0.0;
                for (int i = 0; i < n; i++) {
                    d2[i] = points.weight(i) * points.vector(i).squaredDistance(mean, meanNorm);
                    spread += d2[i];
                }
                for (int i = 0; i < n; i++) {
                    q[i] = points.weight(i) / (2.0 * w) + (spread > 0.0 ? d2[i] / (2.0 * spread) : points.weight(i)
                            / (2.0 * w));
                }
            }
            case MEAN -> throw new IllegalArgumentException("mean coreset has no sampling distribution");
        }
        return q;
    }

    static double[] weightedMean(PointSet points) {
        double[] acc = new double[points.dim()];
        double w = 0.0;
        for (int t = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0; i < b.size(); i++) {
                double wi = b.weight(i);
                SparseVector v = b.vector(i);
                v.addTo(acc, wi);
                w += wi;
            }
        }
        for (int j = 0; j < acc.length; j++) {
            acc[j] /= w;
        }
        return acc;
    }
}
