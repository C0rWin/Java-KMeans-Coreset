package io.github.c0rwin.coreset.testbed.run;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.lang.management.ThreadMXBean;
import java.util.List;
import java.util.concurrent.locks.LockSupport;

/**
 * Brackets a block of work with JVM measurements: wall time, CPU and allocated bytes of the calling thread (or of all
 * live threads for multi-threaded work), GC count and time, and optionally the heap peak. The heap peak is the larger
 * of the summed per-pool peaks (which collectors such as G1 only update at GC time) and the maximum of the used heap
 * polled every millisecond by a sampler thread; it is only meaningful when nothing else runs in the JVM.
 */
final class ResourceProbe {

    private static final ThreadMXBean THREADS = ManagementFactory.getThreadMXBean();
    private static final com.sun.management.ThreadMXBean ALLOC =
            THREADS instanceof com.sun.management.ThreadMXBean b && b.isThreadAllocatedMemorySupported() ? b : null;
    private static final com.sun.management.OperatingSystemMXBean OS =
            ManagementFactory.getOperatingSystemMXBean() instanceof com.sun.management.OperatingSystemMXBean o
                    ? o : null;

    /** Whose CPU and allocations are counted. */
    enum Scope { THREAD, PROCESS }

    record Sample(double wallMs, double cpuMs, Double processCpuMs, long allocBytes, Long peakHeapBytes,
                  Long baselineHeapBytes, long gcCount, long gcMs) {
    }

    private final Scope scope;
    private final boolean exclusive;
    private final long wall0;
    private final long cpu0;
    private final long processCpu0;
    private final long alloc0;
    private final long gcCount0;
    private final long gcMs0;
    private final Long baseline;
    private final HeapSampler sampler;

    private ResourceProbe(Scope scope, boolean exclusive) {
        this.scope = scope;
        this.exclusive = exclusive;
        if (exclusive) {
            System.gc(); // a clean baseline makes the peak mostly this run's working set
            long used = 0;
            for (MemoryPoolMXBean p : heapPools()) {
                p.resetPeakUsage();
                used += p.getUsage().getUsed();
            }
            baseline = used;
            sampler = new HeapSampler();
            sampler.start();
        } else {
            baseline = null;
            sampler = null;
        }
        long[] gc = gc();
        gcCount0 = gc[0];
        gcMs0 = gc[1];
        alloc0 = allocated();
        processCpu0 = processCpu();
        cpu0 = THREADS.getCurrentThreadCpuTime();
        wall0 = System.nanoTime();
    }

    /** @param exclusive this is the only run in the JVM, so heap and process CPU are attributable to it */
    static ResourceProbe start(Scope scope, boolean exclusive) {
        return new ResourceProbe(scope, exclusive);
    }

    Sample stop() {
        long wall = System.nanoTime() - wall0;
        long cpu = THREADS.getCurrentThreadCpuTime() - cpu0;
        long processCpu = processCpu() - processCpu0;
        long alloc = allocated() - alloc0;
        long[] gc = gc();
        Long peak = null;
        if (exclusive) {
            long sum = 0;
            for (MemoryPoolMXBean p : heapPools()) {
                sum += p.getPeakUsage().getUsed();
            }
            peak = Math.max(sum, sampler.finish());
        }
        double cpuMs = scope == Scope.PROCESS && exclusive && processCpu0 >= 0 ? processCpu / 1e6 : cpu / 1e6;
        return new Sample(wall / 1e6, cpuMs, exclusive && processCpu0 >= 0 ? processCpu / 1e6 : null,
                Math.max(alloc, 0), peak, baseline, gc[0] - gcCount0, gc[1] - gcMs0);
    }

    private long allocated() {
        if (ALLOC == null) {
            return 0;
        }
        if (scope == Scope.THREAD) {
            return ALLOC.getCurrentThreadAllocatedBytes();
        }
        // Threads that start and die between start and stop are missed; the node's workers outlive the probe.
        long sum = 0;
        for (long b : ALLOC.getThreadAllocatedBytes(THREADS.getAllThreadIds())) {
            sum += Math.max(b, 0);
        }
        return sum;
    }

    /** Polls the used heap until finished; daemon so an abandoned probe never blocks JVM exit. */
    private static final class HeapSampler extends Thread {
        private volatile boolean running = true;
        private volatile long max;

        HeapSampler() {
            super("testbed-heap-sampler");
            setDaemon(true);
        }

        @Override
        public void run() {
            var memory = ManagementFactory.getMemoryMXBean();
            while (running) {
                long used = memory.getHeapMemoryUsage().getUsed();
                if (used > max) {
                    max = used;
                }
                LockSupport.parkNanos(1_000_000L);
            }
        }

        long finish() {
            running = false;
            LockSupport.unpark(this);
            try {
                join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return Math.max(max, ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed());
        }
    }

    private static long processCpu() {
        return OS == null ? -1 : OS.getProcessCpuTime();
    }

    private static List<MemoryPoolMXBean> heapPools() {
        return ManagementFactory.getMemoryPoolMXBeans().stream().filter(p -> p.getType() == MemoryType.HEAP)
                .toList();
    }

    private static long[] gc() {
        long count = 0;
        long ms = 0;
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) {
            count += Math.max(g.getCollectionCount(), 0);
            ms += Math.max(g.getCollectionTime(), 0);
        }
        return new long[] {count, ms};
    }
}
