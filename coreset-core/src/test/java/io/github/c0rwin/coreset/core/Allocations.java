package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.management.ManagementFactory;
import java.util.function.Supplier;

/** Bytes allocated by the current thread while running an action, for allocation-bound guard tests. */
public final class Allocations {

    private static final com.sun.management.ThreadMXBean THREADS =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();

    private Allocations() {
    }

    /**
     * Fewest bytes allocated over a few runs of {@code action} (the first runs also warm up class loading and
     * lambdas); its result is kept reachable until measured so it is not optimised away. Skips the test when the JVM
     * cannot count allocations.
     */
    public static long bytes(Supplier<?> action) {
        assumeTrue(THREADS.isThreadAllocatedMemorySupported() && THREADS.isThreadAllocatedMemoryEnabled());
        long best = Long.MAX_VALUE;
        Object sink = null;
        for (int run = 0; run < 5; run++) {
            long before = THREADS.getCurrentThreadAllocatedBytes();
            Object r = action.get();
            long after = THREADS.getCurrentThreadAllocatedBytes();
            sink = r;
            best = Math.min(best, after - before);
        }
        if (sink == THREADS) {
            throw new AssertionError(); // keeps sink live
        }
        return best;
    }
}
