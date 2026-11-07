package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.List;

/**
 * Turns a rough clustering B into per-point bounds {@code s_p} on the sensitivity
 * {@code sigma_p = sup_Q w_p d(p, Q)^2 / cost(P, Q)}, as used by {@link SensitivityCoreset}.
 *
 * <p>Instances are immutable and thread-safe.
 */
public abstract class SensitivityBound {

    private SensitivityBound() {
    }

    /**
     * The bound of Bachem, Lucic and Krause, "Practical Coreset Constructions for Machine Learning" (2017), Lemma 2.2,
     * in weighted form. For p in cluster B_i of B, with W_i its weight and {@code cost(B_i)} its cost:
     * <pre>  s_p = 2 alpha w_p d(p, B)^2 / cost(B) + 4 alpha w_p cost(B_i) / (W_i cost(B)) + 4 w_p / W_i</pre>
     * (only the last term when {@code cost(B) = 0}). It follows from {@code d(p, Q)^2 <= 2 d(p, b_i)^2 + 2 d(b_i, Q)^2},
     * averaging the same inequality over B_i for {@code d(b_i, Q)^2}, and {@code cost(P, Q) >= cost(B) / alpha}, so
     * {@code s_p >= sigma_p} holds whenever {@code cost(B) <= alpha * OPT_k}. The total is
     * {@code S = 6 alpha + 4 * (non-empty clusters)}.
     *
     * @throws IllegalArgumentException unless {@code alpha} is finite and {@code >= 1}
     */
    public static SensitivityBound standard(double alpha) {
        if (!(alpha >= 1.0) || Double.isInfinite(alpha)) {
            throw new IllegalArgumentException("alpha must be finite and >= 1: " + alpha);
        }
        return new Standard(alpha);
    }

    /**
     * {@code 16 (log2 k + 2)}, the approximation factor Bachem et al. (Section 2.4) derive for k-means++ seeding:
     * Arthur and Vassilvitskii bound its expected cost by {@code 8 (log2 k + 2) OPT_k}, so by Markov's inequality one
     * seeding is within this factor with probability at least 1/2. Lloyd steps and the extra centers of bicriteria
     * seeding (whose first round is a k-means++ seeding) only lower cost(B), so the factor stays valid for them.
     *
     * @throws IllegalArgumentException if {@code k < 1}
     */
    public static double kMeansPlusPlusAlpha(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        return 16.0 * (Math.log(k) / Math.log(2.0) + 2.0);
    }

    /**
     * The formula used before {@link #standard}: {@code s_p = 8 w_p / W_i + 2 w_p d(p, B)^2 / cost(B)}, total
     * {@code 8 * clusters + 2}. It omits B's approximation factor and bounds {@code sigma_p} only when B is optimal,
     * so Hoeffding intervals and a-priori sizes built on it carry no guarantee. Kept for comparison.
     */
    public static SensitivityBound legacy() {
        return Legacy.INSTANCE;
    }

    /**
     * Per-point bounds for {@code points} under the clustering {@code b} of the same points (zero for zero-weight
     * points). Two passes: cluster weights and costs, then the bounds.
     */
    public final double[] sensitivities(List<WeightedPoint> points, Clustering b) {
        final int n = points.size();
        if (b.assignment.length != n) {
            throw new IllegalArgumentException("clustering has " + b.assignment.length + " points, input " + n);
        }
        final int clusters = b.centers().size();
        double[] clusterWeight = new double[clusters];
        double[] clusterCost = new double[clusters];
        double[] weights = new double[n];
        for (int i = 0; i < n; i++) {
            double w = points.get(i).weight();
            weights[i] = w;
            if (w > 0.0) {
                int c = b.assignment[i];
                clusterWeight[c] += w;
                clusterCost[c] += w * b.squaredDistances[i];
            }
        }
        double[] s = weights; // overwritten in place: one array per call
        for (int i = 0; i < n; i++) {
            double w = weights[i];
            if (w > 0.0) {
                int c = b.assignment[i];
                s[i] = bound(w, b.squaredDistances[i], clusterWeight[c], clusterCost[c], b.cost());
            }
        }
        return s;
    }

    /** {@code s_p} for weight w, squared distance d2 to B, its cluster's weight and cost, and cost(B). */
    abstract double bound(double w, double d2, double clusterWeight, double clusterCost, double costB);

    private static final class Standard extends SensitivityBound {
        private final double alpha;

        Standard(double alpha) {
            this.alpha = alpha;
        }

        @Override
        double bound(double w, double d2, double clusterWeight, double clusterCost, double costB) {
            double s = 4.0 * w / clusterWeight;
            if (costB > 0.0) {
                s += 2.0 * alpha * w * d2 / costB + 4.0 * alpha * w * clusterCost / (clusterWeight * costB);
            }
            return s;
        }

        @Override
        public String toString() {
            return "standard(alpha=" + alpha + ")";
        }
    }

    private static final class Legacy extends SensitivityBound {
        static final Legacy INSTANCE = new Legacy();

        @Override
        double bound(double w, double d2, double clusterWeight, double clusterCost, double costB) {
            double s = 8.0 * w / clusterWeight;
            if (costB > 0.0) {
                s += 2.0 * w * d2 / costB;
            }
            return s;
        }

        @Override
        public String toString() {
            return "legacy";
        }
    }
}
