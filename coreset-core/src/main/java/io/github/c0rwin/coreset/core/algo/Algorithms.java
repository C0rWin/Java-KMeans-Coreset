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

    private Algorithms() {
    }

    /**
     * @throws IllegalArgumentException for an unknown id, an unknown or missing parameter, an unparsable value or a
     *     value the algorithm rejects
     */
    public static CoresetAlgorithm create(String id, Map<String, String> params) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(params, "params");
        switch (id) {
            case UNIFORM:
                allow(id, params, "m");
                return new UniformCoreset(intParam(id, params, "m"));
            case SENSITIVITY: {
                allow(id, params, "k", "m", "seeding", "lloydIterations", "bound", "alpha");
                int k = intParam(id, params, "k");
                int m = intParam(id, params, "m");
                SensitivityBound bound = sensitivityBound(id, params, k);
                String seeding = params.getOrDefault("seeding", SEEDING_KMEANS_PLUS_PLUS);
                switch (seeding) {
                    case SEEDING_KMEANS_PLUS_PLUS:
                        return new SensitivityCoreset(k, m,
                                Seeding.kMeansPlusPlus(intParam(id, params, "lloydIterations", 0)), bound);
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
                int iterations = intParam(id, params, "lloydIterations", DEFAULT_MEAN_LLOYD_ITERATIONS);
                boolean adaptive = params.containsKey("eps") || params.containsKey("k") || params.containsKey("maxM");
                if (adaptive == params.containsKey("m")) {
                    throw new IllegalArgumentException(
                            id + ": give either m, or k, eps and maxM (adaptive), got " + new TreeSet<>(params.keySet()));
                }
                return adaptive
                        ? MeanCoreset.adaptive(intParam(id, params, "k"), doubleParam(id, params, "eps"),
                                intParam(id, params, "maxM"), iterations)
                        : new MeanCoreset(intParam(id, params, "m"), iterations);
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
