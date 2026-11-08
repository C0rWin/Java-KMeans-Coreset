package io.github.c0rwin.coreset.planner;

import com.fasterxml.jackson.annotation.JsonIgnore;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.planner.Configuration.Family;
import java.util.List;
import java.util.Map;

/**
 * What the planner knows about the data, estimated from one bounded pilot sample ({@link WorkloadEstimator}).
 * Quantities marked "full scale" are extrapolated from the pilot to the {@code n} input points with the factor
 * {@code r = n / pilotSize}: a pilot point stands for r input points of the same cost, so
 * {@code kappa_full = kappa_pilot / r} and per-point maxima shrink by r, while ratios of the form
 * {@code sum c^2/q / C^2} are scale free.
 *
 * @param datasetId label of the input
 * @param n number of input points
 * @param d dimension
 * @param density stored entries / (n d)
 * @param pilotSize points in the pilot sample
 * @param k number of centers
 * @param kmeansppCost cost of one k-means++ seeding of the pilot, scaled to the full input
 * @param balancedness smallest / largest cluster size of that seeding (1 = balanced)
 * @param candidates the fixed query family of the instance-dependent guarantee
 * @param families per sampling family: sensitivity totals and per-candidate variance factors
 * @param mean mean-coreset measurements on the pilot, or null if skipped
 * @param tables per-point pilot arrays for the merged/split terms and for validation (not serialised)
 */
public record WorkloadProfile(String datasetId, long n, int d, double density, int pilotSize, int k,
                              double kmeansppCost, double balancedness, List<CandidateStats> candidates,
                              Map<Family, FamilyStats> families, MeanStats mean,
                              @JsonIgnore PilotTables tables) {

    /**
     * @param name generator label
     * @param kappaPt full-scale {@code sum_p cost(p,Q)^2 / cost(P,Q)^2}, the concentration of kappa_pt law
     * @param costShareMax full-scale {@code max_p cost(p,Q) / cost(P,Q)} (the Bernstein range of the node-loss term)
     */
    public record CandidateStats(String name, double kappaPt, double costShareMax) {
    }

    /**
     * @param sTotal {@code max_p s_p / q_p} with s the standard sensitivity bound: the S that makes q a valid
     *     sampling distribution in ass:sampling (for the sensitivity family this is {@code sum s = 6 alpha + 4 k'})
     * @param sMax full-scale {@code max_p s_p}
     * @param sigma2 per candidate, {@code sum_p cost(p,Q)^2 / q_p / cost(P,Q)^2 - 1}: m times the relative variance
     *     of one draw (fixed Q, exact for the multinomial estimator)
     * @param range per candidate, {@code max_p cost(p,Q) / (q_p cost(P,Q))}: m times the Bernstein range of a draw
     */
    public record FamilyStats(double sTotal, double sMax, double[] sigma2, double[] range) {
    }

    /**
     * Mean coreset (Barger and Feldman) with {@code cells[j]} k-means++ cells on a pilot subsample.
     *
     * @param cells grid of cell counts
     * @param bias {@code bias[j][q]}: {@code cost(MC,Q)/cost(P,Q) - 1}, non-negative by their Lemma 1
     * @param kappaCells {@code kappaCells[j][q]}: {@code sum_g cost(P_g,Q)^2 / cost(P,Q)^2}, the cell-loss analogue
     *     of kappa_pt
     */
    public record MeanStats(int[] cells, double[][] bias, double[][] kappaCells) {
    }

    /**
     * Pilot-scale arrays: {@code q[family][i]} (sums to 1 over the pilot), {@code share[q][i] = c_i / C_pilot}, the
     * candidate centers, the scale factor r, and per family a log-binned histogram of q for fast merged/split terms.
     */
    public record PilotTables(Map<Family, double[]> q, double[][] share, List<List<SparseVector>> centers,
                              double scale, Map<Family, QBins> bins) {
    }

    /**
     * Pilot points grouped by q into bins of ratio {@value #BIN_RATIO}: bin b holds {@code count[b]} points with mean
     * probability {@code q[b]} and, per candidate, {@code share2[cand][b] = sum share_i^2}. Using the bin mean for
     * lambda changes lambda by at most the bin ratio.
     */
    public record QBins(double[] q, double[] count, double[][] share2) {
        public static final double BIN_RATIO = 1.05;

        public static QBins of(double[] q, double[][] share) {
            java.util.TreeMap<Long, double[]> acc = new java.util.TreeMap<>();
            int nq = share.length;
            double logRatio = Math.log(BIN_RATIO);
            for (int i = 0; i < q.length; i++) {
                if (q[i] <= 0.0) {
                    continue;
                }
                long key = (long) Math.floor(Math.log(q[i]) / logRatio);
                double[] a = acc.computeIfAbsent(key, x -> new double[2 + nq]);
                a[0] += 1.0;
                a[1] += q[i];
                for (int c = 0; c < nq; c++) {
                    a[2 + c] += share[c][i] * share[c][i];
                }
            }
            int nb = acc.size();
            double[] bq = new double[nb];
            double[] cnt = new double[nb];
            double[][] s2 = new double[nq][nb];
            int b = 0;
            for (double[] a : acc.values()) {
                cnt[b] = a[0];
                bq[b] = a[1] / a[0];
                for (int c = 0; c < nq; c++) {
                    s2[c][b] = a[2 + c];
                }
                b++;
            }
            return new QBins(bq, cnt, s2);
        }
    }

    public int candidateCount() {
        return candidates.size();
    }
}
