package io.github.c0rwin.coreset.core.metrics;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Measures a block of work on the current thread. On {@link #close()} it emits {@code <name>.wall_ns} (wall-clock
 * nanoseconds) and, when the sink opts in with {@link MetricsSink#threadProbes()}, {@code <name>.cpu_ns} (thread CPU
 * nanoseconds, when the JVM supports it) and {@code <name>.alloc_bytes} (bytes allocated by the thread, when
 * {@code com.sun.management.ThreadMXBean} supports it). A span must be closed on the thread that started it; closing
 * twice has no effect.
 *
 * <p>Spans on {@link MetricsSink#NOOP} are a shared inert instance and never touch clocks or MXBeans.
 */
public final class Span implements AutoCloseable {

    private static final Span INERT = new Span(null, null, 0L, -1L, -1L);
    /** Suffixed names per span name, so closing a span builds no strings; bounded against unbounded name sets. */
    private static final ConcurrentHashMap<String, String[]> NAMES = new ConcurrentHashMap<>();
    private static final int MAX_CACHED_NAMES = 4096;

    private final MetricsSink sink;
    private final String name;
    private final long wallStart;
    private final long cpuStart;
    private final long allocStart;
    private boolean closed;

    private Span(MetricsSink sink, String name, long wallStart, long cpuStart, long allocStart) {
        this.sink = sink;
        this.name = name;
        this.wallStart = wallStart;
        this.cpuStart = cpuStart;
        this.allocStart = allocStart;
    }

    public static Span start(MetricsSink sink, String name) {
        if (sink == MetricsSink.NOOP) {
            return INERT;
        }
        if (!sink.threadProbes()) {
            return new Span(sink, name, System.nanoTime(), -1L, -1L);
        }
        long alloc = Probes.allocatedBytes();
        long cpu = Probes.cpuNanos();
        return new Span(sink, name, System.nanoTime(), cpu, alloc);
    }

    @Override
    public void close() {
        if (sink == null || closed) {
            return;
        }
        closed = true;
        long wall = System.nanoTime() - wallStart;
        long cpuEnd = cpuStart >= 0 ? Probes.cpuNanos() : -1L;
        long allocEnd = allocStart >= 0 ? Probes.allocatedBytes() : -1L;
        String[] names = names(name);
        sink.timeNanos(names[0], wall);
        if (cpuEnd >= 0) {
            sink.timeNanos(names[1], cpuEnd - cpuStart);
        }
        if (allocEnd >= 0) {
            sink.record(names[2], allocEnd - allocStart);
        }
    }

    private static String[] names(String name) {
        String[] n = NAMES.get(name);
        if (n == null) {
            n = new String[] {name + MetricNames.WALL_NS_SUFFIX, name + MetricNames.CPU_NS_SUFFIX,
                name + MetricNames.ALLOC_BYTES_SUFFIX};
            if (NAMES.size() < MAX_CACHED_NAMES) {
                NAMES.putIfAbsent(name, n);
            }
        }
        return n;
    }

    /** Lazily initialised so programs without thread probes never load management classes. */
    private static final class Probes {
        private static final ThreadMXBean THREADS = ManagementFactory.getThreadMXBean();
        private static final boolean CPU_SUPPORTED =
                THREADS.isCurrentThreadCpuTimeSupported() && THREADS.isThreadCpuTimeEnabled();
        private static final com.sun.management.ThreadMXBean ALLOC = allocBean();

        private static com.sun.management.ThreadMXBean allocBean() {
            try {
                if (THREADS instanceof com.sun.management.ThreadMXBean bean
                        && bean.isThreadAllocatedMemorySupported()
                        && bean.isThreadAllocatedMemoryEnabled()) {
                    return bean;
                }
            } catch (LinkageError | UnsupportedOperationException e) {
                // jdk.management absent or disabled: allocation tracking is optional.
            }
            return null;
        }

        static long cpuNanos() {
            return CPU_SUPPORTED ? THREADS.getCurrentThreadCpuTime() : -1L;
        }

        static long allocatedBytes() {
            return ALLOC != null ? ALLOC.getCurrentThreadAllocatedBytes() : -1L;
        }
    }
}
