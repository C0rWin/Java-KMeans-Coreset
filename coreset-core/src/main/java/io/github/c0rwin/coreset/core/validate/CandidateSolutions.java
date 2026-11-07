package io.github.c0rwin.coreset.core.validate;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Seeded generators of candidate center sets for {@link Distortion}, following the ESA'22 evaluation of
 * Schwiegelshohn and Sheikh-Omar plus the solutions a user would actually deploy: k-means++ on P, Lloyd on the coreset
 * C, Lloyd on P, and random convex combinations of bicriteria centers of P. All randomness comes from {@code rng}.
 */
public final class CandidateSolutions {

    /** Bicriteria centers mixed into one convex-combination center, at most. */
    static final int MAX_COMBINED = 3;

    private CandidateSolutions() {
    }

    /** {@code count} k-means++ seedings on P, labelled {@code kmeans++@P#i}. */
    public static List<Candidate> kMeansPlusPlusOnFull(Coreset full, int k, int count, RandomGenerator rng) {
        List<Candidate> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            out.add(new Candidate("kmeans++@P#" + i, KMeansPlusPlus.seed(full.block(), k, rng, MetricsSink.NOOP)));
        }
        return out;
    }

    /** {@code count} Lloyd runs on the coreset, i.e. the solutions deployed from it; labelled {@code lloyd@C#i}. */
    public static List<Candidate> lloydOnCoreset(Coreset coreset, int k, int iterations, int count,
                                                 RandomGenerator rng) {
        return lloyd(coreset, "lloyd@C#", k, iterations, count, rng);
    }

    /** {@code count} Lloyd runs on P, labelled {@code lloyd@P#i}. */
    public static List<Candidate> lloydOnFull(Coreset full, int k, int iterations, int count, RandomGenerator rng) {
        return lloyd(full, "lloyd@P#", k, iterations, count, rng);
    }

    private static List<Candidate> lloyd(Coreset data, String label, int k, int iterations, int count,
                                         RandomGenerator rng) {
        List<Candidate> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            out.add(new Candidate(label + i,
                    WeightedLloyd.cluster(data.block(), k, iterations, rng, MetricsSink.NOOP).centers()));
        }
        return out;
    }

    /**
     * {@code count} sets of k centers, each center a convex combination of 1 to {@value #MAX_COMBINED} distinct
     * bicriteria centers of P with uniform (flat Dirichlet) coefficients; labelled {@code convex@B#i}. One bicriteria
     * clustering is shared by all candidates.
     */
    public static List<Candidate> convexCombinationsOfBicriteria(Coreset full, int k, int count, RandomGenerator rng) {
        List<SparseVector> b = Seeding.bicriteria().cluster(full.block(), k, rng, MetricsSink.NOOP).centers();
        if (b.isEmpty()) {
            throw new IllegalArgumentException("input has no positive-weight point");
        }
        final int dim = b.get(0).dim();
        int[] order = new int[b.size()];
        double[] acc = new double[dim];
        double[] lambda = new double[MAX_COMBINED];
        List<Candidate> out = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            List<SparseVector> centers = new ArrayList<>(k);
            for (int j = 0; j < k; j++) {
                int arity = 1 + rng.nextInt(Math.min(MAX_COMBINED, b.size()));
                for (int t = 0; t < order.length; t++) {
                    order[t] = t;
                }
                double total = 0.0;
                for (int t = 0; t < arity; t++) {
                    int pick = t + rng.nextInt(order.length - t);
                    int tmp = order[t];
                    order[t] = order[pick];
                    order[pick] = tmp;
                    lambda[t] = -Math.log(1.0 - rng.nextDouble()); // Exp(1); normalised below
                    total += lambda[t];
                }
                if (total == 0.0) { // every draw was nextDouble() == 0.0: Exp(1) gave -0.0, so mix equally
                    Arrays.fill(lambda, 0, arity, 1.0);
                    total = arity;
                }
                Arrays.fill(acc, 0.0);
                for (int t = 0; t < arity; t++) {
                    b.get(order[t]).addTo(acc, lambda[t] / total);
                }
                centers.add(SparseVector.dense(acc));
            }
            out.add(new Candidate("convex@B#" + i, centers));
        }
        return out;
    }

    /** All four generators, {@code count} candidates each, in the order listed in the class description. */
    public static List<Candidate> standard(Coreset full, Coreset coreset, int k, int lloydIterations, int count,
                                           RandomGenerator rng) {
        List<Candidate> out = new ArrayList<>(4 * count);
        out.addAll(kMeansPlusPlusOnFull(full, k, count, rng));
        out.addAll(lloydOnCoreset(coreset, k, lloydIterations, count, rng));
        out.addAll(lloydOnFull(full, k, lloydIterations, count, rng));
        out.addAll(convexCombinationsOfBicriteria(full, k, count, rng));
        return out;
    }
}
