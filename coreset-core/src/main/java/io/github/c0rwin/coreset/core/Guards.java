package io.github.c0rwin.coreset.core;

import java.util.concurrent.CancellationException;

/** Checks shared by the algorithms, meant to be fused into a pass they already make over their input. */
public final class Guards {

    private Guards() {
    }

    /**
     * Rejects an input whose total weight or weighted second moment {@code sum_p w_p ||p||^2} is not finite. Every
     * weight and coordinate is finite on its own, but such sums overflow: the moment bounds every cost the algorithms
     * compute, and a zero-weight point with an infinite squared norm makes it NaN. Callers accumulate both sums in
     * their first pass (over all points, zero weights included) and call this before any other work.
     *
     * @throws IllegalArgumentException if either sum is not finite
     */
    public static void requireSummable(double totalWeight, double secondMoment) {
        if (!Double.isFinite(totalWeight) || !Double.isFinite(secondMoment)) {
            throw new IllegalArgumentException("input overflows double: total weight " + totalWeight
                    + ", sum of w * ||p||^2 " + secondMoment + "; rescale the weights or coordinates");
        }
    }

    /**
     * Cooperative cancellation point for long loops (seeding, Lloyd, sampling): throws if the current thread is
     * interrupted. The interrupt flag stays set, so the caller's own handling still sees it.
     *
     * @throws CancellationException naming {@code operation} if the thread is interrupted
     */
    public static void checkInterrupted(String operation) {
        if (Thread.currentThread().isInterrupted()) {
            throw new CancellationException("interrupted during " + operation);
        }
    }
}
