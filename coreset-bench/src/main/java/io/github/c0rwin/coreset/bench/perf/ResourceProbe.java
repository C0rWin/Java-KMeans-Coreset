package io.github.c0rwin.coreset.bench.perf;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;

/**
 * JVM-wide resource deltas around a measured run: bytes allocated by all live threads, GC count and time, and peak
 * heap. Allocation is summed over threads alive at {@link #stop}, so call it before worker threads exit.
 */
final class ResourceProbe {

    private static final com.sun.management.ThreadMXBean THREADS =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();

    private final long allocStart;
    private final long gcCountStart;
    private final long gcTimeStart;
    private final long wallStart;
    private final long baselineHeap;

    /** Result of one probe window; {@code baselineHeapBytes} is heap in use after a full GC at the start. */
    record Usage(double seconds, long allocatedBytes, long gcCount, long gcMillis, long peakHeapBytes,
                 long baselineHeapBytes) {
        double allocRateMBs() {
            return allocatedBytes / seconds / (1024.0 * 1024.0);
        }
    }

    private ResourceProbe() {
        System.gc(); // so the baseline is live data (e.g. the pre-generated workload), not garbage
        baselineHeap = heapUsed();
        resetPeaks();
        allocStart = allocatedByAllThreads();
        gcCountStart = gcCount();
        gcTimeStart = gcMillis();
        wallStart = System.nanoTime();
    }

    static ResourceProbe start() {
        return new ResourceProbe();
    }

    /**
     * Ends the window. Threads that died inside the window are missing from the allocation sum, threads started inside
     * it count fully.
     */
    Usage stop() {
        double seconds = (System.nanoTime() - wallStart) / 1e9;
        return new Usage(seconds, allocatedByAllThreads() - allocStart, gcCount() - gcCountStart,
                gcMillis() - gcTimeStart, peakHeap(), baselineHeap);
    }

    static long allocatedByAllThreads() {
        long[] ids = THREADS.getAllThreadIds();
        long sum = 0;
        for (long a : THREADS.getThreadAllocatedBytes(ids)) {
            if (a > 0) {
                sum += a;
            }
        }
        return sum;
    }

    private static long heapUsed() {
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }

    private static void resetPeaks() {
        for (MemoryPoolMXBean p : ManagementFactory.getMemoryPoolMXBeans()) {
            if (p.getType() == MemoryType.HEAP) {
                p.resetPeakUsage();
            }
        }
    }

    /** Sum of per-pool peaks: an upper bound on the true peak, since pools need not peak simultaneously. */
    private static long peakHeap() {
        long sum = 0;
        for (MemoryPoolMXBean p : ManagementFactory.getMemoryPoolMXBeans()) {
            if (p.getType() == MemoryType.HEAP) {
                sum += p.getPeakUsage().getUsed();
            }
        }
        return sum;
    }

    private static long gcCount() {
        long sum = 0;
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) {
            sum += Math.max(0, g.getCollectionCount());
        }
        return sum;
    }

    private static long gcMillis() {
        long sum = 0;
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) {
            sum += Math.max(0, g.getCollectionTime());
        }
        return sum;
    }
}
