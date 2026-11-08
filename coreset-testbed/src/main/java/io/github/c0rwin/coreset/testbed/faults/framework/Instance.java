package io.github.c0rwin.coreset.testbed.faults.framework;

import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The shared mixture: 8 k-center queries (six D^2 seedings, two random k-subsets; query 0 is the fixed query), cost
 * tables and full-data proposals for z = 2 and z = 1.
 */
final class Instance {

    static final int QUERIES = 8;

    final Mixture data;
    final int k;
    final List<double[]> queries;
    private final long seed;
    private final double[][] tables = new double[3][];
    private final CostSpace[] full = new CostSpace[3];

    Instance(Mixture data, int k, long seed) {
        this.data = data;
        this.k = k;
        this.seed = seed;
        int[] rows = CostSpace.identity(data.n);
        double[] ones = new double[data.n];
        Arrays.fill(ones, 1.0);
        SplittableRandom qr = Seeds.rng(seed, "queries", 0, 0);
        this.queries = new ArrayList<>();
        for (int j = 0; j < 6; j++) {
            queries.add(data.seed(rows, ones, k, 2.0, qr));
        }
        for (int j = 6; j < QUERIES; j++) {
            double[] c = new double[k * data.d];
            for (int t = 0; t < k; t++) {
                System.arraycopy(data.x, qr.nextInt(data.n) * data.d, c, t * data.d, data.d);
            }
            queries.add(c);
        }
    }

    synchronized double[] table(double z) {
        int i = index(z);
        if (tables[i] == null) {
            tables[i] = data.table(queries, k, z == 2.0 ? Problem.kMeans() : Problem.kMedian());
        }
        return tables[i];
    }

    /** All rows with unit weight, the proposal of {@link Mixture#proposal} for this z, classes = components. */
    synchronized CostSpace full(double z) {
        int i = index(z);
        if (full[i] == null) {
            int[] rows = CostSpace.identity(data.n);
            double[] ones = new double[data.n];
            Arrays.fill(ones, 1.0);
            double[] s = data.proposal(rows, ones, k, z, Seeds.rng(seed, "proposal", i, 0));
            full[i] = new CostSpace(rows, ones, s, table(z), QUERIES, data.label, data.components);
        }
        return full[i];
    }

    private static int index(double z) {
        if (z == 2.0) {
            return 2;
        }
        if (z == 1.0) {
            return 1;
        }
        throw new IllegalArgumentException("z must be 1 or 2: " + z);
    }
}
