package io.github.c0rwin.coreset.testbed.problem;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.List;
import java.util.Locale;

/**
 * A center-based clustering objective {@code cost(P, C) = sum_p w_p * dist(p, C)^z}: k-means is {@code z = 2},
 * k-median {@code z = 1}. Implementations must be stateless and thread-safe.
 */
public interface Problem {

    String KMEANS = "kmeans";
    String KMEDIAN = "kmedian";

    String id();

    double z();

    /** {@code dist(p, C)^z} for one unit-weight point; {@code scratch} holds at least {@code centers.size()}. */
    double pointCost(SparseVector p, Centers centers, double[] scratch);

    /**
     * The coreset's additive constant under this objective. The mean coreset's offset is a sum of squared distances,
     * so it only means something for {@code z = 2}.
     *
     * @throws IllegalArgumentException if {@code z != 2} and the coreset carries a positive offset
     */
    double offset(Coreset c);

    /** {@code sum_p w_p dist(p, C)^z + offset}, single-threaded; zero-weight points are skipped. */
    default double cost(Coreset c, Centers centers) {
        return cost(c.points(), centers) + offset(c);
    }

    default double cost(List<WeightedPoint> points, Centers centers) {
        double[] scratch = new double[centers.size()];
        double s = 0.0;
        for (int i = 0, n = points.size(); i < n; i++) {
            WeightedPoint p = points.get(i);
            if (p.weight() > 0.0) {
                s += p.weight() * pointCost(p.vector(), centers, scratch);
            }
        }
        return s;
    }

    static Problem kMeans() {
        return PowerCost.KMEANS_INSTANCE;
    }

    static Problem kMedian() {
        return PowerCost.KMEDIAN_INSTANCE;
    }

    /** @throws IllegalArgumentException for an id other than {@value #KMEANS} or {@value #KMEDIAN} */
    static Problem of(String id) {
        switch (id.toLowerCase(Locale.ROOT)) {
            case KMEANS:
                return kMeans();
            case KMEDIAN:
                return kMedian();
            default:
                throw new IllegalArgumentException("unknown problem '" + id + "', expected " + KMEANS + " or "
                        + KMEDIAN);
        }
    }

    /** {@code dist^z} objective. */
    record PowerCost(String id, double z) implements Problem {

        static final PowerCost KMEANS_INSTANCE = new PowerCost(KMEANS, 2.0);
        static final PowerCost KMEDIAN_INSTANCE = new PowerCost(KMEDIAN, 1.0);

        public PowerCost {
            if (!(z > 0.0) || Double.isInfinite(z)) {
                throw new IllegalArgumentException("z must be positive and finite: " + z);
            }
        }

        @Override
        public double pointCost(SparseVector p, Centers centers, double[] scratch) {
            double d2 = centers.minSquaredDistance(p, scratch);
            if (z == 2.0) {
                return d2;
            }
            return z == 1.0 ? Math.sqrt(d2) : Math.pow(d2, z / 2.0);
        }

        @Override
        public double offset(Coreset c) {
            if (z != 2.0 && c.offset() > 0.0) {
                throw new IllegalArgumentException(id + ": a coreset with a squared-distance offset ("
                        + c.offset() + ") cannot be evaluated for z=" + z);
            }
            return c.offset();
        }
    }
}
