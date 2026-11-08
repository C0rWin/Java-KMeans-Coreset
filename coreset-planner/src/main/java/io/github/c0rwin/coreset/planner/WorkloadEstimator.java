package io.github.c0rwin.coreset.planner;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.planner.WorkloadProfile.CandidateStats;
import io.github.c0rwin.coreset.planner.WorkloadProfile.FamilyStats;
import io.github.c0rwin.coreset.planner.WorkloadProfile.MeanStats;
import io.github.c0rwin.coreset.planner.WorkloadProfile.PilotTables;
import io.github.c0rwin.coreset.testbed.data.PointFile;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Builds a {@link WorkloadProfile} from one pass over the input and computations on a bounded pilot sample.
 *
 * <p>Cost: the pass reads every row once and keeps a uniform reservoir of {@code pilotSize} rows (Vitter's
 * algorithm R), so it is O(n) row reads, O(pilotSize * nnz) memory. On the pilot: one greedy k-means++ seeding for the
 * sensitivity bound, {@value #KMEANSPP_CANDIDATES} k-means++ and {@value #LLOYD_CANDIDATES} Lloyd candidates
 * ({@value #LLOYD_ITERATIONS} iterations) and one k/2 candidate, i.e. O(pilotSize * k * nnz * 50); and, for the mean
 * coreset, plain D^2 seedings with {@code cells} centers on at most {@value #MEAN_PILOT} points for each grid value,
 * O(MEAN_PILOT * sum(cells) * nnz). Nothing scales with n beyond the pass.
 */
public final class WorkloadEstimator {

    public static final int DEFAULT_PILOT = 20_000;
    static final int KMEANSPP_CANDIDATES = 4;
    static final int LLOYD_CANDIDATES = 2;
    static final int LLOYD_ITERATIONS = 20;
    static final int MEAN_PILOT = 10_000;
    static final int MEAN_MAX_CELLS = 2048;

    private final int pilotSize;
    private final int k;
    private final long seed;
    private final boolean measureMean;

    public WorkloadEstimator(int pilotSize, int k, long seed, boolean measureMean) {
        if (pilotSize < 2 * k || k < 1) {
            throw new IllegalArgumentException("need k >= 1 and pilot >= 2k: " + pilotSize + ", " + k);
        }
        this.pilotSize = pilotSize;
        this.k = k;
        this.seed = seed;
        this.measureMean = measureMean;
    }

    /** Streams a binary point file once. */
    public WorkloadProfile estimate(String id, Path file) throws IOException {
        Reservoir res = new Reservoir(pilotSize, seed);
        long nnz = 0;
        int d;
        try (PointFile.Reader r = PointFile.open(file, PointFile.DEFAULT_CHUNK_ROWS)) {
            d = r.header().d();
            nnz = r.header().nnz();
            while (r.hasNext()) {
                res.offer(r.next());
            }
        }
        return profile(id, res, d, nnz);
    }

    /** Same estimate for an in-memory point set (unit or general weights are both read as rows). */
    public WorkloadProfile estimate(String id, PointSet points) {
        Reservoir res = new Reservoir(pilotSize, seed);
        for (int t = 0; t < points.segmentCount(); t++) {
            res.offer(points.segment(t));
        }
        return profile(id, res, points.dim(), points.storedEntries());
    }

    private WorkloadProfile profile(String id, Reservoir res, int d, long nnz) {
        PointSet pilot = PointSet.of(PointBlock.of(res.rows()));
        long n = res.seen;
        int ns = pilot.size();
        double scale = (double) n / ns;
        SplittableRandom rng = new SplittableRandom(seed ^ 0x5DEECE66DL);

        FamilyScores scores = FamilyScores.of(pilot, k, rng.split());
        int[] sizes = new int[scores.clustering.centers().size()];
        for (int a : scores.clustering.assignment()) {
            if (a >= 0) {
                sizes[a]++;
            }
        }
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int s : sizes) {
            min = Math.min(min, s);
            max = Math.max(max, s);
        }
        double balancedness = max == 0 ? 1.0 : (double) min / max;
        double kmeansppCost = scores.clustering.cost() * scale;

        List<String> names = new ArrayList<>();
        List<List<SparseVector>> centers = new ArrayList<>();
        for (int j = 0; j < KMEANSPP_CANDIDATES; j++) {
            names.add("kmeans++#" + j);
            centers.add(KMeansPlusPlus.seed(pilot, k, rng.split(), MetricsSink.NOOP));
        }
        for (int j = 0; j < LLOYD_CANDIDATES; j++) {
            names.add("lloyd#" + j);
            centers.add(WeightedLloyd.cluster(pilot, k, LLOYD_ITERATIONS, rng.split(), MetricsSink.NOOP).centers());
        }
        if (k >= 2) {
            names.add("kmeans++(k/2)");
            centers.add(KMeansPlusPlus.seed(pilot, k / 2, rng.split(), MetricsSink.NOOP));
        }

        int nq = centers.size();
        double[][] share = new double[nq][];
        List<CandidateStats> cands = new ArrayList<>(nq);
        for (int q = 0; q < nq; q++) {
            share[q] = costShares(pilot, centers.get(q));
            double kappa = 0.0;
            double cmax = 0.0;
            for (double c : share[q]) {
                kappa += c * c;
                cmax = Math.max(cmax, c);
            }
            cands.add(new CandidateStats(names.get(q), kappa / scale, cmax / scale));
        }

        Map<Family, double[]> qs = new EnumMap<>(Family.class);
        Map<Family, FamilyStats> fams = new EnumMap<>(Family.class);
        for (Family f : Family.values()) {
            if (f.sampling()) {
                double[] q = scores.q(f);
                qs.put(f, q);
                fams.put(f, familyStats(scores.sensitivity, q, share, scale));
            }
        }

        MeanStats mean = measureMean ? meanStats(pilot, centers, rng.split()) : null;
        double density = (double) nnz / ((double) n * d);
        Map<Family, WorkloadProfile.QBins> bins = new EnumMap<>(Family.class);
        qs.forEach((f, q) -> bins.put(f, WorkloadProfile.QBins.of(q, share)));
        return new WorkloadProfile(id, n, d, density, ns, k, kmeansppCost, balancedness, List.copyOf(cands), fams,
                mean, new PilotTables(qs, share, centers, scale, bins));
    }

    /** {@code c_i / sum c} with {@code c_i = w_i d(p_i, Q)^2}. */
    static double[] costShares(PointSet pilot, List<SparseVector> centers) {
        int ns = pilot.size();
        double[] c = new double[ns];
        double total = 0.0;
        for (int i = 0; i < ns; i++) {
            SparseVector v = pilot.vector(i);
            c[i] = pilot.weight(i) * v.squaredDistance(centers.get(Cost.nearest(v, centers)));
            total += c[i];
        }
        for (int i = 0; i < ns; i++) {
            c[i] = total > 0.0 ? c[i] / total : 0.0;
        }
        return c;
    }

    static FamilyStats familyStats(double[] sensitivity, double[] q, double[][] share, double scale) {
        double sTotal = 0.0;
        double sMax = 0.0;
        for (int i = 0; i < q.length; i++) {
            if (q[i] > 0.0) {
                sTotal = Math.max(sTotal, sensitivity[i] / q[i]);
            }
            sMax = Math.max(sMax, sensitivity[i]);
        }
        int nq = share.length;
        double[] sigma2 = new double[nq];
        double[] range = new double[nq];
        for (int j = 0; j < nq; j++) {
            double acc = 0.0;
            double rg = 0.0;
            for (int i = 0; i < q.length; i++) {
                double c = share[j][i];
                if (c > 0.0) {
                    acc += c * c / q[i];
                    rg = Math.max(rg, c / q[i]);
                }
            }
            sigma2[j] = Math.max(0.0, acc - 1.0);
            range[j] = rg;
        }
        return new FamilyStats(sTotal, sMax / scale, sigma2, range);
    }

    private MeanStats meanStats(PointSet pilot, List<List<SparseVector>> centers, SplittableRandom rng) {
        PointSet sub = pilot.size() <= MEAN_PILOT ? pilot : pilot.slice(0, MEAN_PILOT);
        int ns = sub.size();
        int nq = centers.size();
        double[][] cost = new double[nq][ns];
        double[] full = new double[nq];
        for (int q = 0; q < nq; q++) {
            for (int i = 0; i < ns; i++) {
                SparseVector v = sub.vector(i);
                cost[q][i] = sub.weight(i) * v.squaredDistance(centers.get(q).get(Cost.nearest(v, centers.get(q))));
                full[q] += cost[q][i];
            }
        }
        List<Integer> grid = new ArrayList<>();
        for (int m = k; m <= Math.min(MEAN_MAX_CELLS, ns / 8); m *= 2) {
            grid.add(m);
        }
        int[] cells = grid.stream().mapToInt(Integer::intValue).toArray();
        double[][] bias = new double[cells.length][nq];
        double[][] kappa = new double[cells.length][nq];
        for (int j = 0; j < cells.length; j++) {
            List<SparseVector> seeds = KMeansPlusPlus.seed(sub, cells[j], 1, rng.split(), MetricsSink.NOOP);
            int[] assign = new int[ns];
            for (int i = 0; i < ns; i++) {
                assign[i] = Cost.nearest(sub.vector(i), seeds);
            }
            Coreset mc = WeightedLloyd.meansOf(sub, assign, seeds.size());
            for (int q = 0; q < nq; q++) {
                double[] cellCost = new double[seeds.size()];
                for (int i = 0; i < ns; i++) {
                    cellCost[assign[i]] += cost[q][i];
                }
                double kap = 0.0;
                for (double c : cellCost) {
                    kap += (c / full[q]) * (c / full[q]);
                }
                kappa[j][q] = kap;
                bias[j][q] = Math.abs(Cost.cost(mc, centers.get(q)) / full[q] - 1.0);
            }
        }
        return new MeanStats(cells, bias, kappa);
    }

    /** Uniform reservoir of row copies (rows are copied so no input chunk stays reachable). */
    private static final class Reservoir {
        private final WeightedPoint[] rows;
        private final SplittableRandom rnd;
        long seen;

        Reservoir(int capacity, long seed) {
            rows = new WeightedPoint[capacity];
            rnd = new SplittableRandom(seed);
        }

        void offer(PointBlock block) {
            for (int i = 0, n = block.size(); i < n; i++) {
                long slot = seen < rows.length ? seen : rnd.nextLong(seen + 1);
                seen++;
                if (slot < rows.length) {
                    rows[(int) slot] = copy(block, i);
                }
            }
        }

        List<WeightedPoint> rows() {
            int n = (int) Math.min(seen, rows.length);
            List<WeightedPoint> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(rows[i]);
            }
            return out;
        }

        private static WeightedPoint copy(PointBlock b, int i) {
            SparseVector v = b.vector(i);
            int nnz = v.nnz();
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int e = 0; e < nnz; e++) {
                idx[e] = v.indexAt(e);
                val[e] = v.valueAt(e);
            }
            return new WeightedPoint(SparseVector.ofOwned(v.dim(), idx, val), b.weight(i));
        }
    }
}
