package io.github.c0rwin.coreset.planner;

import java.util.Objects;

/**
 * What the plan must achieve and what it should minimise.
 *
 * @param eps target relative cost error
 * @param delta allowed failure probability, split by the model into sampling, channel and node budgets
 * @param k number of centers of the k-means queries
 * @param guarantee {@link Guarantee#INSTANCE}: error over the workload's finite candidate family (fixed-query laws
 *     plus a union bound); {@link Guarantee#UNIFORM}: every k-center query, worst-case sampling-lemma form
 * @param mode which prediction must not exceed {@code eps}
 * @param objective what the cheapest plan minimises
 * @param bytesWeightPerMB weight of one megabyte in {@link Objective#WEIGHTED}
 * @param timeWeightPerSecond weight of one second in {@link Objective#WEIGHTED}
 */
public record Requirement(double eps, double delta, int k, Guarantee guarantee, Mode mode, Objective objective,
                          double bytesWeightPerMB, double timeWeightPerSecond) {

    public enum Guarantee { INSTANCE, UNIFORM }

    /**
     * CERTIFIED: the conservative bound (Bernstein per fixed query, union bound over the finite candidate set, upper
     * calibration constants) must meet eps; CALIBRATED: the point prediction (kappa-law and Gaussian quantile with
     * the calibrated constants) must.
     */
    public enum Mode { CERTIFIED, CALIBRATED }

    public enum Objective { MIN_BYTES, MIN_TIME, WEIGHTED }

    public Requirement {
        Objects.requireNonNull(guarantee, "guarantee");
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(objective, "objective");
        if (!(eps > 0.0 && eps <= 1.0) || !(delta > 0.0 && delta < 1.0) || k < 1 || !(bytesWeightPerMB >= 0.0)
                || !(timeWeightPerSecond >= 0.0)) {
            throw new IllegalArgumentException("invalid requirement: " + this);
        }
    }

    public static Requirement of(double eps, double delta, int k, Objective objective) {
        return new Requirement(eps, delta, k, Guarantee.INSTANCE, Mode.CERTIFIED, objective, 1.0, 1.0);
    }

    public Requirement withMode(Mode m) {
        return new Requirement(eps, delta, k, guarantee, m, objective, bytesWeightPerMB, timeWeightPerSecond);
    }

    /** The scalar the planner minimises. */
    public double cost(double bytes, double seconds) {
        return switch (objective) {
            case MIN_BYTES -> bytes;
            case MIN_TIME -> seconds;
            case WEIGHTED -> bytes / 1e6 * bytesWeightPerMB + seconds * timeWeightPerSecond;
        };
    }
}
