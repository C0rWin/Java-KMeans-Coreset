package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Shared context of a fault-injection run: prepared instances (cached), the worker pool, seeds, recorders and the
 * report (markdown sections plus the verdict table) written to {@code summary.md} and {@code verdicts.csv}.
 */
final class FaultLab implements AutoCloseable {

    enum Scale { FULL, QUICK }

    /** PASS / FAIL / INCONCLUSIVE against a numeric prediction of research section 5.5. */
    record Verdict(String id, String prediction, String measured, String status) {
    }

    static final List<String> EXPERIMENTS = List.of("F1", "F2", "F3", "F4", "F5", "F6");

    final Path out;
    final Scale scale;
    final long seed;
    final ForkJoinPool pool;
    private final Datasets datasets;
    private final PrintWriter log;
    private final Map<String, Instance> instances = new HashMap<>();
    private final List<Verdict> verdicts = new ArrayList<>();
    private final StringBuilder report = new StringBuilder();
    private final long t0 = System.nanoTime();

    FaultLab(Path out, Scale scale, long seed, int threads, Datasets datasets, PrintWriter log) throws IOException {
        this.out = out;
        this.scale = scale;
        this.seed = seed;
        this.pool = new ForkJoinPool(threads);
        this.datasets = datasets;
        this.log = log;
        Files.createDirectories(out);
    }

    boolean quick() {
        return scale == Scale.QUICK;
    }

    int seeds(int full, int quick) {
        return quick() ? quick : full;
    }

    void run(List<String> experiments) throws IOException {
        for (String e : experiments) {
            long s = System.nanoTime();
            log("%s: start", e);
            switch (e.toUpperCase(Locale.ROOT)) {
                case "F1" -> F1NodeLoss.run(this);
                case "F2" -> F2PostLoss.run(this);
                case "F3" -> F3Erasure.run(this);
                case "F4" -> F4Stragglers.run(this);
                case "F5" -> F5Depth.run(this);
                case "F6" -> F6Tree.run(this);
                default -> throw new IllegalArgumentException("unknown experiment " + e + ", expected " + EXPERIMENTS);
            }
            log("%s: done in %.1f s", e, (System.nanoTime() - s) / 1e9);
        }
        writeReport();
    }

    Instance instance(String id) {
        synchronized (instances) {
            Instance inst = instances.get(id);
            if (inst == null) {
                long s = System.nanoTime();
                Object[] spec = spec(id);
                DatasetSpec ds = (DatasetSpec) spec[0];
                int k = (Integer) spec[1];
                inst = Instance.prepare(id, datasets.load(ds), k, quick() ? 3 : 10, quick() ? 10 : 30,
                        quick() ? 2 : 5, LabSeeds.of(seed, "instance", LabSeeds.hash(id), 0), pool);
                instances.put(id, inst);
                log("instance %s: n=%d d=%d k=%d smax/S=%.3g S=%.1f smallest cluster=%d (%.1f s)", id, inst.n,
                        inst.dim(), k, inst.sMax, inst.sensTotal, inst.clusterSize[inst.smallestCluster],
                        (System.nanoTime() - s) / 1e9);
            }
            return inst;
        }
    }

    /** {DatasetSpec, k} per instance id; QUICK shrinks every instance for tests. */
    private Object[] spec(String id) {
        boolean q = quick();
        String n = switch (id) {
            case "zipf", "zipf-heavy" -> q ? "4000" : "200000";
            case "zipf-100k" -> q ? "2000" : "100000";
            case "zipf-400k" -> q ? "8000" : "400000";
            case "esa22" -> "";
            default -> throw new IllegalArgumentException("unknown instance " + id);
        };
        if (id.equals("esa22")) {
            int k = q ? 4 : 10;
            return new Object[] {new DatasetSpec(id, "esa22", Map.of("k", Integer.toString(k), "alpha", "5")), k};
        }
        int clusters = q ? 6 : 20;
        Map<String, String> p = Map.of("n", n, "d", q ? "6" : "20", "clusters", Integer.toString(clusters),
                "sigma", q ? "1" : "3", "imbalance", id.equals("zipf-heavy") ? "2.5" : "1.5", "box", "100",
                "seed", "1");
        return new Object[] {new DatasetSpec(id, "gaussian-mixture", p), clusters};
    }

    /** Runs {@code task(0..count-1)} on the pool; results in index order. Tasks must not use the pool themselves. */
    <T> List<T> parallel(int count, IntFunction<T> task) {
        try {
            return pool.submit(() -> IntStream.range(0, count).parallel().mapToObj(task).toList()).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            throw c instanceof RuntimeException re ? re : new IllegalStateException(c);
        }
    }

    Recorder recorder(String experiment) {
        try {
            return new Recorder(out, experiment);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    void verdict(String id, String prediction, String measured, String status) {
        verdicts.add(new Verdict(id, prediction, measured, status));
        log("VERDICT %s: %s | %s", id, status, measured);
    }

    void section(String markdown) {
        report.append(markdown).append("\n\n");
    }

    void log(String fmt, Object... args) {
        log.printf(Locale.ROOT, "[%7.1fs] " + fmt + "%n", prepend((System.nanoTime() - t0) / 1e9, args));
        log.flush();
    }

    private static Object[] prepend(Object first, Object[] rest) {
        Object[] a = new Object[rest.length + 1];
        a[0] = first;
        System.arraycopy(rest, 0, a, 1, rest.length);
        return a;
    }

    List<Verdict> verdicts() {
        return List.copyOf(verdicts);
    }

    private void writeReport() throws IOException {
        StringBuilder md = new StringBuilder();
        md.append("# Fault-injection results (scale ").append(scale.name().toLowerCase(Locale.ROOT))
                .append(", seed ").append(seed).append(")\n\n## Verdicts\n\n")
                .append("| id | prediction | measured | verdict |\n|---|---|---|---|\n");
        StringBuilder csv = new StringBuilder("id,status,prediction,measured\n");
        for (Verdict v : verdicts) {
            md.append("| ").append(v.id()).append(" | ").append(v.prediction()).append(" | ").append(v.measured())
                    .append(" | **").append(v.status()).append("** |\n");
            csv.append(v.id()).append(',').append(v.status()).append(",\"").append(v.prediction().replace("\"", "'"))
                    .append("\",\"").append(v.measured().replace("\"", "'")).append("\"\n");
        }
        md.append("\n").append(report);
        md.append("## Instances\n\n| id | n | d | k | s_max/S | S | smallest cluster | prepare s |\n"
                + "|---|---|---|---|---|---|---|---|\n");
        for (Instance i : instances.values().stream().sorted((a, b) -> a.id.compareTo(b.id)).toList()) {
            md.append(String.format(Locale.ROOT, "| %s | %d | %d | %d | %.3g | %.1f | %d | %.1f |%n", i.id, i.n,
                    i.dim(), i.k, i.sMax, i.sensTotal, i.clusterSize[i.smallestCluster], i.prepareMs / 1e3));
        }
        Files.writeString(out.resolve("summary.md"), md.toString(), StandardCharsets.UTF_8);
        Files.writeString(out.resolve("verdicts.csv"), csv.toString(), StandardCharsets.UTF_8);
    }

    @Override
    public void close() {
        pool.shutdownNow();
    }

    static String f(String fmt, Object... args) {
        return String.format(Locale.ROOT, fmt, args);
    }
}
