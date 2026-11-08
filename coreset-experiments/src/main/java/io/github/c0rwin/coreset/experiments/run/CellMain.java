package io.github.c0rwin.coreset.experiments.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.experiments.config.CellSpec;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import io.github.c0rwin.coreset.experiments.data.PointSources;
import io.github.c0rwin.coreset.experiments.metrics.Environment;
import io.github.c0rwin.coreset.experiments.metrics.ProcessSampler;
import io.github.c0rwin.coreset.experiments.quality.QualityEvaluator;
import io.github.c0rwin.coreset.experiments.systems.EngineSystem;
import io.github.c0rwin.coreset.experiments.systems.EngineSystem.RunOutput;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Runs one cell: starts the system (timed as startup), runs the warm-up repetitions (not recorded) and the measured
 * ones, solves k-means on every coreset with the same solver, and writes one JSON line per repetition to
 * {@code <outDir>/records.jsonl} plus the coreset to {@code rep-<i>.coreset} for the orchestrator's quality pass.
 * Usually its own JVM ({@code java ... CellMain cell.json}), so systems never share a heap or JIT state.
 */
public final class CellMain {

    private static final ObjectMapper JSON = ExperimentConfig.JSON;

    private CellMain() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("usage: CellMain <cell.json>");
            System.exit(2);
        }
        CellSpec cell = JSON.readValue(Path.of(args[0]).toFile(), CellSpec.class);
        java.util.logging.Logger.getLogger("").setLevel(java.util.logging.Level.WARNING);
        int status = run(cell) ? 0 : 1;
        System.exit(status); // engine threads (Spark, Flink, gRPC) may linger as non-daemons
    }

    /** Runs the cell; false if a repetition failed (its error is recorded). */
    public static boolean run(CellSpec cell) throws IOException {
        Path out = Path.of(cell.outDir());
        Files.createDirectories(out);
        Path records = out.resolve("records.jsonl");
        PointSource src = PointSources.open(cell.dataset(), Path.of(cell.cacheDir()));
        Map<String, Object> base = base(cell, src);
        try (BufferedWriter w = Files.newBufferedWriter(records, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
             EngineSystem system = EngineSystem.create(cell, src)) {
            long pid = ProcessHandle.current().pid();
            ProcessSampler sampler = new ProcessSampler(cell.sampleMillis(), () -> {
                Map<Long, String> r = new LinkedHashMap<>(system.roles());
                r.put(pid, system.selfRole());
                return r;
            });
            try (sampler) {
                long t0 = System.nanoTime();
                try {
                    system.start();
                } catch (Exception e) {
                    write(w, error(base, "start", e));
                    return false;
                }
                base.put("startup_s", (System.nanoTime() - t0) / 1e9);
                return repetitions(cell, src, system, sampler, base, w, out);
            }
        }
    }

    private static boolean repetitions(CellSpec cell, PointSource src, EngineSystem system, ProcessSampler sampler,
                                       Map<String, Object> base, BufferedWriter w, Path out) throws IOException {
        FaultConfig fault = cell.fault();
        int rep = 0;
        try {
            for (int i = 0; i < cell.warmup(); i++) {
                RunOutput o = system.run(-1 - i, cell.seed(), null);
                solve(cell, o, cell.seed());
            }
            if (fault == null) {
                for (int i = 0; i < cell.repetitions(); i++, rep++) {
                    long seed = cell.seed() + i; // independent coresets, so quality spread is measured too
                    write(w, measure(cell, src, system, sampler, base, rep, seed, null, "measure", null, out));
                }
                return true;
            }
            String fingerprint = null;
            for (int i = 0; i < fault.baselineRepetitions(); i++, rep++) {
                Map<String, Object> r = measure(cell, src, system, sampler, base, rep, cell.seed(), null, "baseline",
                        null, out);
                write(w, r);
                fingerprint = (String) r.get("coreset_fingerprint");
            }
            int faultReps = fault.repetitions() != null ? fault.repetitions() : cell.repetitions();
            for (int i = 0; i < faultReps; i++, rep++) {
                write(w, measure(cell, src, system, sampler, base, rep, cell.seed(), fault, "fault", fingerprint,
                        out));
            }
            return true;
        } catch (Exception e) {
            write(w, error(base, "rep " + rep, e));
            return false;
        }
    }

    private static Map<String, Object> measure(CellSpec cell, PointSource src, EngineSystem system,
                                               ProcessSampler sampler, Map<String, Object> base, int rep, long seed,
                                               FaultConfig fault, String phase, String baselineFingerprint,
                                               Path out) throws Exception {
        List<Double> loadBefore = Environment.loadAverage();
        sampler.start();
        RunOutput o;
        Solved s;
        List<ProcessSampler.Usage> usage;
        try {
            o = system.run(rep, seed, fault);
            s = solve(cell, o, seed);
        } finally {
            usage = sampler.stop();
        }
        Map<String, Object> r = new LinkedHashMap<>(base);
        r.put("phase", phase);
        r.put("rep", rep);
        r.put("seed", seed);
        r.put("fault_label", fault == null ? (phase.equals("baseline") ? cell.fault().label() : "none") : fault.label());
        r.put("build_s", o.buildSeconds());
        r.put("gather_s", o.gatherSeconds());
        r.put("solve_s", s.seconds);
        double total = nz(o.buildSeconds()) + nz(o.gatherSeconds()) + s.seconds;
        r.put("total_s", total);
        double ingestTime = Double.isNaN(o.buildSeconds()) ? s.seconds : o.buildSeconds();
        r.put("points_per_s", src.size() / ingestTime);
        double cpu = 0;
        double peak = 0;
        double rssSum = 0;
        List<Map<String, Object>> procs = new ArrayList<>();
        for (ProcessSampler.Usage u : usage) {
            cpu += u.cpuSeconds();
            peak = Math.max(peak, u.peakRssMB());
            rssSum += u.peakRssMB();
            Map<String, Object> p = new LinkedHashMap<>();
            p.put("role", u.role());
            p.put("pid", u.pid());
            p.put("peak_rss_mb", u.peakRssMB());
            p.put("cpu_s", u.cpuSeconds());
            procs.add(p);
        }
        r.put("cpu_s", cpu);
        r.put("peak_rss_mb_max", peak);
        r.put("peak_rss_mb_sum", rssSum);
        r.put("processes", procs);
        if (o.coreset() != null) {
            r.put("coreset_size", o.coreset().size());
            String fp = fingerprint(o.coreset());
            r.put("coreset_fingerprint", fp);
            if (baselineFingerprint != null) {
                r.put("equal_to_baseline", fp.equals(baselineFingerprint));
            }
            Path file = out.resolve("rep-" + rep + ".coreset");
            try (ObjectOutputStream os = new ObjectOutputStream(Files.newOutputStream(file))) {
                os.writeObject(o.coreset());
            }
            r.put("coreset_file", file.getFileName().toString());
        }
        r.put("centers", QualityEvaluator.encode(s.centers));
        r.put("metrics", o.metrics());
        r.put("fault", o.fault());
        r.put("load_before", loadBefore);
        r.put("load_after", Environment.loadAverage());
        return r;
    }

    private record Solved(List<SparseVector> centers, double seconds) {
    }

    /** Best of the configured restarts on the coreset (the solver every coreset system shares), or native centers. */
    private static Solved solve(CellSpec cell, RunOutput o, long seed) {
        if (o.coreset() == null) {
            return new Solved(o.centers(), o.solveSeconds());
        }
        long t0 = System.nanoTime();
        PointSet ps = o.coreset().pointSet();
        KMeansResult r = WeightedLloyd.bestOf(ps, cell.k(), cell.solver().iterations(), cell.solver().restarts(),
                new SplittableRandom(seed), MetricsSink.NOOP);
        return new Solved(r.centers(), (System.nanoTime() - t0) / 1e9);
    }

    private static double nz(double v) {
        return Double.isNaN(v) ? 0.0 : v;
    }

    private static Map<String, Object> base(CellSpec cell, PointSource src) {
        Map<String, Object> b = new LinkedHashMap<>();
        b.put("cell", cell.id());
        b.put("dataset", src.id());
        b.put("n", src.size());
        b.put("d", src.dim());
        b.put("system", cell.system().label());
        b.put("system_type", cell.system().type());
        b.put("system_mode", cell.system().mode());
        b.put("system_options", cell.system().options());
        b.put("parallelism", cell.parallelism());
        b.put("algorithm", cell.algorithmLabel());
        b.put("algorithm_params", cell.algorithm() == null ? Map.of() : cell.algorithm().params());
        b.put("k", cell.k());
        b.put("solver_restarts", cell.solver().restarts());
        b.put("solver_iterations", cell.solver().iterations());
        b.put("warmup", cell.warmup());
        b.put("cell_jvm_args", ManagementFactory.getRuntimeMXBean().getInputArguments());
        b.put("worker_jvm_flags", cell.jvmFlags());
        b.put("memory_workers_mb", cell.memory().workersMB());
        b.put("memory_driver_mb", cell.memory().driverMB());
        b.put("dataset_provenance", src.provenance());
        return b;
    }

    private static Map<String, Object> error(Map<String, Object> base, String where, Exception e) {
        Map<String, Object> r = new LinkedHashMap<>(base);
        r.put("phase", "error");
        StringWriter sw = new StringWriter();
        e.printStackTrace(new PrintWriter(sw));
        r.put("error", where + ": " + e);
        r.put("stack", sw.toString());
        return r;
    }

    private static void write(BufferedWriter w, Map<String, Object> record) throws IOException {
        w.write(JSON.writeValueAsString(record));
        w.newLine();
        w.flush();
    }

    /** SHA-256 over every entry, weight and the offset, in order: equal coresets have equal fingerprints. */
    static String fingerprint(Coreset c) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            ByteBuffer b = ByteBuffer.allocate(16);
            PointSet ps = c.pointSet();
            for (int i = 0; i < ps.size(); i++) {
                SparseVector v = ps.vector(i);
                for (int e = 0; e < v.nnz(); e++) {
                    b.clear();
                    b.putLong(v.indexAt(e)).putLong(Double.doubleToLongBits(v.valueAt(e)));
                    md.update(b.array(), 0, 16);
                }
                b.clear();
                b.putLong(-1).putLong(Double.doubleToLongBits(ps.weight(i)));
                md.update(b.array(), 0, 16);
            }
            b.clear();
            b.putLong(Double.doubleToLongBits(c.offset()));
            md.update(b.array(), 0, 8);
            return HexFormat.of().formatHex(md.digest(), 0, 16);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
