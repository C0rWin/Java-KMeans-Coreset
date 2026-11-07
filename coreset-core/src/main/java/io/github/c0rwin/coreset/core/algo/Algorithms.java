package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Builds coreset algorithms from string ids and parameters (as carried over RPC or configuration).
 *
 * <ul>
 *   <li>{@code uniform}: {@code m} (required) — {@link UniformCoreset}.</li>
 *   <li>{@code sensitivity}: {@code k}, {@code m} (required); {@code seeding} = {@code kmeans++} (default) or
 *       {@code bicriteria}; {@code lloydIterations} (default 0, only with {@code kmeans++}); {@code bound} =
 *       {@code standard} (default, {@link SensitivityBound#standard}) or {@code legacy}
 *       ({@link SensitivityBound#legacy}, for comparison only); {@code alpha} (only with {@code standard}, default
 *       {@link SensitivityBound#kMeansPlusPlusAlpha}{@code (k)}) — {@link SensitivityCoreset}.</li>
 *   <li>{@code mean}: either {@code m} (fixed number of means) or {@code k}, {@code eps}, {@code maxM} (adaptive
 *       search, {@link MeanCoreset#adaptive}); {@code lloydIterations} (default 20) — {@link MeanCoreset}.</li>
 * </ul>
 *
 * <p>Parameters also have upper bounds ({@link Limits}, {@link Limits#DEFAULT} unless given), since they arrive from
 * remote callers and set how much work and memory a compression takes. The algorithms themselves stop with a
 * {@link java.util.concurrent.CancellationException} when their thread is interrupted.
 */
public final class Algorithms {

    public static final String UNIFORM = "uniform";
    public static final String SENSITIVITY = "sensitivity";
    public static final String MEAN = "mean";

    public static final String SEEDING_KMEANS_PLUS_PLUS = "kmeans++";
    public static final String SEEDING_BICRITERIA = "bicriteria";

    public static final String BOUND_STANDARD = "standard";
    public static final String BOUND_LEGACY = "legacy";

    /** Lloyd steps for {@code mean} when {@code lloydIterations} is absent; Lloyd stops earlier on convergence. */
    public static final int DEFAULT_MEAN_LLOYD_ITERATIONS = 20;

    /** Default upper bound on {@code m} (draws or means) and {@code maxM}. */
    public static final int DEFAULT_MAX_M = 1_000_000;
    /** Default upper bound on {@code k}. */
    public static final int DEFAULT_MAX_K = 100_000;
    /** Default upper bound on {@code lloydIterations}. */
    public static final int DEFAULT_MAX_LLOYD_ITERATIONS = 1_000;

    /**
     * Largest accepted parameter values.
     *
     * @param maxM               bound on {@code m} and {@code maxM}
     * @param maxK               bound on {@code k}
     * @param maxLloydIterations bound on {@code lloydIterations}
     */
    public record Limits(int maxM, int maxK, int maxLloydIterations) {

        public static final Limits DEFAULT = new Limits(DEFAULT_MAX_M, DEFAULT_MAX_K, DEFAULT_MAX_LLOYD_ITERATIONS);

        /** @throws IllegalArgumentException if {@code maxM < 1}, {@code maxK < 1} or {@code maxLloydIterations < 0} */
        public Limits {
            if (maxM < 1 || maxK < 1 || maxLloydIterations < 0) {
                throw new IllegalArgumentException("limits must be maxM >= 1, maxK >= 1, maxLloydIterations >= 0: "
                        + maxM + ", " + maxK + ", " + maxLloydIterations);
            }
        }
    }

    private Algorithms() {
    }

    /** {@link #create(String, Map, Limits)} with {@link Limits#DEFAULT}. */
    public static CoresetAlgorithm create(String id, Map<String, String> params) {
        return create(id, params, Limits.DEFAULT);
    }

    /**
     * @throws IllegalArgumentException for an unknown id, an unknown or missing parameter, an unparsable value, a value
     *     above {@code limits} or a value the algorithm rejects
     */
    public static CoresetAlgorithm create(String id, Map<String, String> params, Limits limits) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(params, "params");
        Objects.requireNonNull(limits, "limits");
        switch (id) {
            case UNIFORM:
                allow(id, params, "m");
                return new UniformCoreset(bounded(id, "m", intParam(id, params, "m"), limits.maxM()));
            case SENSITIVITY: {
                allow(id, params, "k", "m", "seeding", "lloydIterations", "bound", "alpha");
                int k = bounded(id, "k", intParam(id, params, "k"), limits.maxK());
                int m = bounded(id, "m", intParam(id, params, "m"), limits.maxM());
                SensitivityBound bound = sensitivityBound(id, params, k);
                String seeding = params.getOrDefault("seeding", SEEDING_KMEANS_PLUS_PLUS);
                switch (seeding) {
                    case SEEDING_KMEANS_PLUS_PLUS:
                        return new SensitivityCoreset(k, m, Seeding.kMeansPlusPlus(bounded(id, "lloydIterations",
                                intParam(id, params, "lloydIterations", 0), limits.maxLloydIterations())), bound);
                    case SEEDING_BICRITERIA:
                        if (params.containsKey("lloydIterations")) {
                            throw new IllegalArgumentException(
                                    id + ": lloydIterations applies only to seeding=" + SEEDING_KMEANS_PLUS_PLUS);
                        }
                        return new SensitivityCoreset(k, m, Seeding.bicriteria(), bound);
                    default:
                        throw new IllegalArgumentException(id + ": unknown seeding '" + seeding + "', expected "
                                + SEEDING_KMEANS_PLUS_PLUS + " or " + SEEDING_BICRITERIA);
                }
            }
            case MEAN: {
                allow(id, params, "m", "k", "eps", "maxM", "lloydIterations");
                int iterations = bounded(id, "lloydIterations",
                        intParam(id, params, "lloydIterations", DEFAULT_MEAN_LLOYD_ITERATIONS),
                        limits.maxLloydIterations());
                boolean adaptive = params.containsKey("eps") || params.containsKey("k") || params.containsKey("maxM");
                if (adaptive == params.containsKey("m")) {
                    throw new IllegalArgumentException(
                            id + ": give either m, or k, eps and maxM (adaptive), got " + new TreeSet<>(params.keySet()));
                }
                return adaptive
                        ? MeanCoreset.adaptive(bounded(id, "k", intParam(id, params, "k"), limits.maxK()),
                                doubleParam(id, params, "eps"),
                                bounded(id, "maxM", intParam(id, params, "maxM"), limits.maxM()), iterations)
                        : new MeanCoreset(bounded(id, "m", intParam(id, params, "m"), limits.maxM()), iterations);
            }
            default:
                throw new IllegalArgumentException(
                        "unknown coreset algorithm '" + id + "', expected one of " + UNIFORM + ", " + SENSITIVITY
                                + ", " + MEAN);
        }
    }

    private static void allow(String id, Map<String, String> params, String... names) {
        Set<String> unknown = new TreeSet<>(params.keySet());
        unknown.removeAll(Set.of(names));
        if (!unknown.isEmpty()) {
            throw new IllegalArgumentException(
                    id + ": unknown parameter(s) " + unknown + ", allowed " + new TreeSet<>(Set.of(names)));
        }
    }

    private static int bounded(String id, String name, int value, int max) {
        if (value > max) {
            throw new IllegalArgumentException(id + ": " + name + " must be <= " + max + ", got " + value);
        }
        return value;
    }

    private static SensitivityBound sensitivityBound(String id, Map<String, String> params, int k) {
        String bound = params.getOrDefault("bound", BOUND_STANDARD);
        switch (bound) {
            case BOUND_STANDARD:
                return SensitivityBound.standard(params.containsKey("alpha")
                        ? doubleParam(id, params, "alpha")
                        : SensitivityBound.kMeansPlusPlusAlpha(k));
            case BOUND_LEGACY:
                if (params.containsKey("alpha")) {
                    throw new IllegalArgumentException(id + ": alpha applies only to bound=" + BOUND_STANDARD);
                }
                return SensitivityBound.legacy();
            default:
                throw new IllegalArgumentException(
                        id + ": unknown bound '" + bound + "', expected " + BOUND_STANDARD + " or " + BOUND_LEGACY);
        }
    }

    private static int intParam(String id, Map<String, String> params, String name) {
        String v = params.get(name);
        if (v == null) {
            throw new IllegalArgumentException(id + ": missing required parameter " + name);
        }
        return parseInt(id, name, v);
    }

    private static int intParam(String id, Map<String, String> params, String name, int fallback) {
        String v = params.get(name);
        return v == null ? fallback : parseInt(id, name, v);
    }

    private static int parseInt(String id, String name, String v) {
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(id + ": parameter " + name + " must be an integer, got '" + v + "'", e);
        }
    }

    private static double doubleParam(String id, Map<String, String> params, String name) {
        String v = params.get(name);
        if (v == null) {
            throw new IllegalArgumentException(id + ": missing required parameter " + name);
        }
        try {
            return Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(id + ": parameter " + name + " must be a number, got '" + v + "'", e);
        }
    }
}
