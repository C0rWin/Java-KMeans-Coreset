package io.github.c0rwin.coreset.experiments.run;

import io.github.c0rwin.coreset.experiments.Jvm;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.experiments.config.CellSpec;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.AlgorithmConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.SystemConfig;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import io.github.c0rwin.coreset.experiments.data.PointSources;
import io.github.c0rwin.coreset.experiments.metrics.Environment;
import io.github.c0rwin.coreset.experiments.quality.QualityEvaluator;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Runs an experiment grid: materialises each dataset once, computes (or loads) its reference solution, runs every
 * cell in a fresh JVM, one at a time, then evaluates the quality of each repetition's output here and appends it to
 * {@code results.jsonl}; finally writes {@code summary.md}.
 *
 * <pre>
 * ExperimentRunner --config grid.yaml [--out dir] [--only regex] [--in-process] [--summary-only]
 * </pre>
 * {@code --in-process} runs cells in this JVM (no local-cluster; used by the smoke test's fast path).
 */
public final class ExperimentRunner {

    private static final ObjectMapper JSON = ExperimentConfig.JSON;

    private final ExperimentConfig cfg;
    private final Path out;
    private final Pattern only;
    private final boolean inProcess;

    ExperimentRunner(ExperimentConfig cfg, Path out, Pattern only, boolean inProcess) {
        this.cfg = cfg;
        this.out = out;
        this.only = only;
        this.inProcess = inProcess;
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> a = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--in-process", "--summary-only" -> a.put(args[i], "true");
                case "--config", "--out", "--only" -> {
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException(args[i] + " needs a value");
                    }
                    a.put(args[i], args[++i]);
                }
                default -> throw new IllegalArgumentException("unknown argument " + args[i]
                        + "; usage: --config file [--out dir] [--only regex] [--in-process] [--summary-only]");
            }
        }
        if (!a.containsKey("--config")) {
            throw new IllegalArgumentException("--config is required");
        }
        ExperimentConfig cfg = ExperimentConfig.read(Path.of(a.get("--config")));
        Path out = Path.of(a.getOrDefault("--out", cfg.output()));
        if (a.containsKey("--summary-only")) {
            Summary.write(out);
            return;
        }
        Pattern only = a.containsKey("--only") ? Pattern.compile(a.get("--only")) : null;
        int failed = new ExperimentRunner(cfg, out, only, a.containsKey("--in-process")).run();
        System.out.println("results: " + out.resolve("results.jsonl") + ", summary: " + out.resolve("summary.md")
                + (failed > 0 ? ", " + failed + " cell(s) failed" : ""));
        System.exit(failed > 0 ? 1 : 0);
    }

    /** Runs every cell; returns the number of failed cells. */
    int run() throws Exception {
        Files.createDirectories(out.resolve("spark-home").resolve("jars"));
        Map<String, Object> env = Environment.capture();
        env.put("config", cfg);
        JSON.writerWithDefaultPrettyPrinter().writeValue(out.resolve("environment.json").toFile(), env);
        Path results = out.resolve("results.jsonl");
        int failed = 0;
        try (BufferedWriter w = Files.newBufferedWriter(results, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            for (DatasetConfig ds : cfg.datasets()) {
                List<CellSpec> cells = cells(ds);
                if (cells.isEmpty()) {
                    continue;
                }
                log("dataset " + ds.id() + ": materialising");
                PointSource src = PointSources.open(ds, Path.of(cfg.cacheDir()));
                log("dataset " + ds.id() + ": " + src.size() + " x " + src.dim() + ", reference and candidates");
                QualityEvaluator quality = new QualityEvaluator(src, cfg.k(), cfg.quality(), Path.of(cfg.cacheDir()),
                        cfg.seed());
                log("reference cost " + quality.reference().cost() + " " + quality.reference().info());
                for (CellSpec cell : cells) {
                    log("cell " + cell.id());
                    boolean ok = runCell(cell);
                    failed += ok ? 0 : 1;
                    for (Map<String, Object> r : readRecords(cell)) {
                        w.write(JSON.writeValueAsString(withQuality(r, cell, src, quality)));
                        w.newLine();
                    }
                    w.flush();
                }
            }
        }
        Summary.write(out);
        return failed;
    }

    List<CellSpec> cells(DatasetConfig ds) {
        List<CellSpec> cells = new ArrayList<>();
        for (SystemConfig sys : cfg.systems()) {
            List<AlgorithmConfig> algos = sys.usesCoreset() ? cfg.algorithms() : Arrays.asList((AlgorithmConfig) null);
            for (int n : cfg.parallelism()) {
                for (AlgorithmConfig algo : algos) {
                    add(cells, ds, sys, n, algo, null);
                    for (FaultConfig f : cfg.faults()) {
                        if (f.appliesTo(sys, n, algo)) {
                            add(cells, ds, sys, n, algo, f);
                        }
                    }
                }
            }
        }
        return cells;
    }

    private void add(List<CellSpec> cells, DatasetConfig ds, SystemConfig sys, int n, AlgorithmConfig algo,
                     FaultConfig fault) {
        if (ds.excludes(sys, algo)) {
            return;
        }
        String id = (ds.id() + "__" + sys.label() + "__N" + n + "__" + (algo == null ? "native" : algo.label())
                + (fault == null ? "" : "__" + fault.label())).replaceAll("[^A-Za-z0-9._-]+", "-");
        if (only != null && !only.matcher(id).find()) {
            return;
        }
        cells.add(new CellSpec(id, ds, cfg.cacheDir(), sys, n, algo, fault, cfg.warmup(), cfg.repetitions(), cfg.k(),
                cfg.solver(), cfg.memory(), cfg.jvmFlags(), cfg.lossy(), cfg.cluster(), cfg.seed(), cfg.batchSize(),
                cfg.sampleMillis(), out.resolve("cells").resolve(id).toAbsolutePath().toString()));
    }

    /** Heap of the cell JVM: it holds the workers' state only in single-JVM modes. */
    static int cellHeapMB(CellSpec c) {
        SystemConfig s = c.system();
        boolean singleJvm = (s.type().startsWith("spark") && s.mode().equals("local"))
                || (s.type().equals("flink") && s.mode().equals("minicluster"));
        return singleJvm ? c.memory().workersMB() : c.memory().driverMB();
    }

    private boolean runCell(CellSpec cell) throws Exception {
        Path dir = Path.of(cell.outDir());
        Files.createDirectories(dir);
        Path spec = dir.resolve("cell.json");
        JSON.writerWithDefaultPrettyPrinter().writeValue(spec.toFile(), cell);
        if (inProcess) {
            return CellMain.run(cell);
        }
        List<String> cmd = new ArrayList<>();
        cmd.add(Jvm.java());
        cmd.addAll(cell.jvmFlags());
        cmd.addAll(Arrays.asList(Environment.build("spark.jvm.args").trim().split("\\s+")));
        cmd.add("-Xmx" + cellHeapMB(cell) + "m");
        cmd.add("-cp");
        cmd.add(Jvm.classPath());
        cmd.add(CellMain.class.getName());
        cmd.add(spec.toString());
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true)
                .redirectOutput(dir.resolve("cell.log").toFile());
        // Executors of local-cluster are launched like a Spark distribution's: an (empty) jars/ dir and the Scala
        // version, the class path itself comes from spark.executor.extraClassPath.
        pb.environment().put("SPARK_HOME", out.resolve("spark-home").toAbsolutePath().toString());
        pb.environment().put("SPARK_SCALA_VERSION", Environment.build("scala.binary.version"));
        if (cell.cluster().sparkMaster() == null) {
            // local-cluster masters otherwise bind to whatever interface the hostname resolves to
            pb.environment().put("SPARK_LOCAL_IP", "127.0.0.1");
            pb.environment().put("SPARK_LOCAL_HOSTNAME", "localhost");
        }
        Process p = pb.start();
        if (!p.waitFor(cfg.cellTimeoutSeconds(), TimeUnit.SECONDS)) {
            p.toHandle().descendants().forEach(ProcessHandle::destroyForcibly);
            p.destroyForcibly().waitFor();
            appendError(cell, "cell timed out after " + cfg.cellTimeoutSeconds() + " s");
            return false;
        }
        return p.exitValue() == 0;
    }

    private void appendError(CellSpec cell, String error) throws IOException {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("cell", cell.id());
        r.put("system", cell.system().label());
        r.put("parallelism", cell.parallelism());
        r.put("algorithm", cell.algorithmLabel());
        r.put("phase", "error");
        r.put("error", error);
        Files.writeString(Path.of(cell.outDir()).resolve("records.jsonl"), JSON.writeValueAsString(r) + "\n",
                StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static List<Map<String, Object>> readRecords(CellSpec cell) throws IOException {
        Path f = Path.of(cell.outDir()).resolve("records.jsonl");
        List<Map<String, Object>> out = new ArrayList<>();
        if (!Files.exists(f)) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("cell", cell.id());
            r.put("phase", "error");
            r.put("error", "cell wrote no records, see " + Path.of(cell.outDir()).resolve("cell.log"));
            out.add(r);
            return out;
        }
        for (String line : Files.readAllLines(f, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                @SuppressWarnings("unchecked")
                Map<String, Object> r = JSON.readValue(line, Map.class);
                out.add(r);
            }
        }
        return out;
    }

    /** Adds the quality fields and drops the bulky per-rep artifacts. */
    private static Map<String, Object> withQuality(Map<String, Object> r, CellSpec cell, PointSource src,
                                                   QualityEvaluator quality) throws IOException {
        if (!r.containsKey("centers")) {
            return r;
        }
        Coreset coreset = null;
        Object file = r.remove("coreset_file");
        if (file != null) {
            Path f = Path.of(cell.outDir()).resolve(file.toString());
            try (ObjectInputStream in = new ObjectInputStream(Files.newInputStream(f))) {
                coreset = (Coreset) in.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
            Files.delete(f);
        }
        Map<String, Object> q = quality.evaluate(coreset, QualityEvaluator.decode(r.get("centers"), src.dim()));
        r.putAll(q);
        return r;
    }

    private static void log(String s) {
        System.out.println("[" + java.time.LocalTime.now().withNano(0) + "] " + s);
    }
}
