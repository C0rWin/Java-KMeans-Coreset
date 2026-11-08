package io.github.c0rwin.coreset.experiments.metrics;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Samples RSS and CPU time of this JVM and every descendant process (gRPC nodes, Spark executors) at a fixed period.
 * Linux reads {@code /proc/<pid>/stat}; elsewhere (macOS) one {@code ps} call per sample covers all processes. A window
 * reports per process the peak RSS seen and the CPU seconds used inside it; a process killed mid-window keeps its last
 * sample. Peaks between samples are missed, so peak RSS is a lower bound at the sampling period.
 */
public final class ProcessSampler implements AutoCloseable {

    private static final boolean PROC = Files.isDirectory(Path.of("/proc/self"));
    private static final long PAGE_BYTES = 4096; // /proc rss is in pages; 4 KiB on x86-64 and most aarch64 kernels

    /** Usage of one process in a window. */
    public record Usage(long pid, String role, double peakRssMB, double cpuSeconds) {
    }

    private record Sample(long rssKb, double cpuSeconds) {
    }

    private static final class Track {
        final String role;
        final double cpuStart;
        double cpuLast;
        long peakRssKb;

        Track(String role, double cpuStart, long rssKb) {
            this.role = role;
            this.cpuStart = cpuStart;
            this.cpuLast = cpuStart;
            this.peakRssKb = rssKb;
        }
    }

    private final ScheduledExecutorService exec = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "process-sampler");
        t.setDaemon(true);
        return t;
    });
    private final Supplier<Map<Long, String>> roles;
    private final Supplier<List<Long>> descendants;
    private final Map<Long, Track> window = new ConcurrentHashMap<>();
    private final java.util.concurrent.atomic.AtomicInteger errors = new java.util.concurrent.atomic.AtomicInteger();
    private volatile boolean open;

    /** @param roles current role per pid (e.g. {@code node-0}); unnamed descendants are {@code child}, this JVM {@code self} */
    public ProcessSampler(int periodMillis, Supplier<Map<Long, String>> roles) {
        this(periodMillis, roles, () -> Processes.descendants().stream().map(ProcessHandle::pid).toList());
    }

    /** As above with the descendant enumeration supplied (tests inject failures through it). */
    ProcessSampler(int periodMillis, Supplier<Map<Long, String>> roles, Supplier<List<Long>> descendants) {
        this.roles = roles;
        this.descendants = descendants;
        exec.scheduleAtFixedRate(this::sampleLocked, periodMillis, periodMillis, TimeUnit.MILLISECONDS);
    }

    /** Starts a window: CPU is counted from now; processes that appear later are counted from zero. Never throws. */
    public synchronized void start() {
        window.clear();
        errors.set(0);
        open = true;
        sample(true);
    }

    /** Ends the window with one last sample. Never throws. */
    public synchronized List<Usage> stop() {
        sample(false);
        open = false;
        List<Usage> out = new ArrayList<>();
        window.forEach((pid, t) -> out.add(new Usage(pid, t.role, t.peakRssKb / 1024.0,
                Math.max(0.0, t.cpuLast - t.cpuStart))));
        out.sort((a, b) -> a.role().compareTo(b.role()));
        return out;
    }

    /** Samples in the current window that failed in part (enumeration, roles or reading), each only lowering resolution. */
    public int errors() {
        return errors.get();
    }

    private synchronized void sampleLocked() {
        if (open) {
            sample(false);
        }
    }

    /** One sample; failures are counted, never thrown, since metrics must not fail a measurement. */
    private void sample(boolean first) {
        try {
            long self = ProcessHandle.current().pid();
            Map<Long, String> names = new LinkedHashMap<>();
            names.put(self, "self");
            boolean failed = false;
            try {
                descendants.get().forEach(pid -> names.put(pid, "child"));
            } catch (RuntimeException e) {
                failed = true; // fall back to the processes already known: tracked ones and named roles
                window.forEach((pid, t) -> names.putIfAbsent(pid, t.role));
            }
            try {
                names.putAll(roles.get());
            } catch (RuntimeException e) {
                failed = true;
            }
            if (failed) {
                errors.incrementAndGet();
            }
            Map<Long, Sample> now = read(names.keySet());
            now.forEach((pid, s) -> {
                Track t = window.get(pid);
                if (t == null) {
                    // a process seen at the window start counts from its CPU then; later arrivals from zero
                    window.put(pid, new Track(names.getOrDefault(pid, "child"), first ? s.cpuSeconds : 0.0, s.rssKb));
                    t = window.get(pid);
                }
                t.cpuLast = Math.max(t.cpuLast, s.cpuSeconds);
                t.peakRssKb = Math.max(t.peakRssKb, s.rssKb);
            });
        } catch (RuntimeException e) {
            errors.incrementAndGet();
        }
    }

    static Map<Long, Sample> read(Iterable<Long> pids) {
        Map<Long, Sample> out = new LinkedHashMap<>();
        if (PROC) {
            for (long pid : pids) {
                try {
                    String stat = Files.readString(Path.of("/proc", Long.toString(pid), "stat"));
                    String[] f = stat.substring(stat.lastIndexOf(')') + 2).split(" ");
                    // fields after "(comm) ": state=0, utime=11, stime=12, rss=21 (proc(5) numbering minus 3)
                    double cpu = (Long.parseLong(f[11]) + Long.parseLong(f[12])) / 100.0;
                    out.put(pid, new Sample(Long.parseLong(f[21]) * PAGE_BYTES / 1024, cpu));
                } catch (IOException | RuntimeException e) {
                    // process exited between listing and reading
                }
            }
            return out;
        }
        String list = java.util.stream.StreamSupport.stream(pids.spliterator(), false).map(String::valueOf)
                .collect(Collectors.joining(","));
        try {
            Process ps = new ProcessBuilder("ps", "-o", "pid=,rss=,time=", "-p", list).redirectErrorStream(true)
                    .start();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(ps.getInputStream(),
                    StandardCharsets.UTF_8))) {
                for (String line; (line = r.readLine()) != null; ) {
                    String[] f = line.trim().split("\\s+");
                    if (f.length == 3 && f[0].chars().allMatch(Character::isDigit)) {
                        out.put(Long.parseLong(f[0]), new Sample(Long.parseLong(f[1]), cpuTime(f[2])));
                    }
                }
            }
            ps.waitFor(5, TimeUnit.SECONDS);
        } catch (IOException e) {
            // ps unavailable: no per-process numbers
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return out;
    }

    /** Parses ps {@code time}: {@code [dd-][hh:]mm:ss[.cc]}. */
    static double cpuTime(String s) {
        double days = 0;
        int dash = s.indexOf('-');
        if (dash >= 0) {
            days = Double.parseDouble(s.substring(0, dash));
            s = s.substring(dash + 1);
        }
        String[] parts = s.split(":");
        double secs = 0;
        for (String p : parts) {
            secs = secs * 60 + Double.parseDouble(p);
        }
        return days * 86400 + secs;
    }

    @Override
    public void close() {
        exec.shutdownNow();
    }
}
