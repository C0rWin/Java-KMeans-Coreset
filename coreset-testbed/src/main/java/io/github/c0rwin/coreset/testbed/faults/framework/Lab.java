package io.github.c0rwin.coreset.testbed.faults.framework;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import java.util.stream.IntStream;
import java.util.zip.GZIPOutputStream;

/**
 * Run context: scale, seeds, pool, shared instance, verdicts and report. Writes summary, verdicts and table CSVs to
 * the output directory, raw gzip CSVs to the raw directory, and the raw SHA-256 list to both.
 */
final class Lab implements AutoCloseable {

    enum Scale { FULL, QUICK }

    record Verdict(String id, String prediction, String measured, String status) {
    }

    static final List<String> EXPERIMENTS = List.of("A", "B", "C", "D", "E");

    final Path out;
    final Path raw;
    final Scale scale;
    final long seed;
    private final ForkJoinPool pool;
    private final PrintWriter log;
    private final List<Verdict> verdicts = new ArrayList<>();
    private final StringBuilder report = new StringBuilder();
    private final List<Table> tables = new ArrayList<>();
    private final List<Path> rawFiles = new ArrayList<>();
    private final long t0 = System.nanoTime();
    private Instance instance;

    Lab(Path out, Path raw, Scale scale, long seed, int threads, PrintWriter log) throws IOException {
        this.out = out;
        this.raw = raw;
        this.scale = scale;
        this.seed = seed;
        this.pool = new ForkJoinPool(threads);
        this.log = log;
        Files.createDirectories(out);
        Files.createDirectories(raw);
    }

    boolean quick() {
        return scale == Scale.QUICK;
    }

    int pick(int full, int quick) {
        return quick() ? quick : full;
    }

    long pick(long full, long quick) {
        return quick() ? quick : full;
    }

    void run(List<String> experiments) throws IOException {
        for (String e : experiments) {
            long s = System.nanoTime();
            log("%s: start", e);
            switch (e.toUpperCase(Locale.ROOT)) {
                case "A" -> ExpA.run(this);
                case "B" -> ExpB.run(this);
                case "C" -> ExpC.run(this);
                case "D" -> ExpD.run(this);
                case "E" -> ExpE.run(this);
                default -> throw new IllegalArgumentException("unknown experiment " + e + ", expected " + EXPERIMENTS);
            }
            log("%s: done in %.1f s", e, (System.nanoTime() - s) / 1e9);
        }
        writeReport();
    }

    /** n = 20000 (quick: 2000) points in d = 5, 10 Zipf-sized components (exponent 1), sigma 4, box 100; k = 10. */
    synchronized Instance instance() {
        if (instance == null) {
            long s = System.nanoTime();
            Mixture data = Mixture.gaussian(pick(20000, 2000), 5, 10, 4.0, 100.0, 1.0, Seeds.of(seed, "data", 0, 0));
            instance = new Instance(data, 10, seed);
            log("instance: n=%d d=%d components=%d k=%d (%.1f s)", data.n, data.d, data.components, instance.k,
                    (System.nanoTime() - s) / 1e9);
        }
        return instance;
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

    Table table(String name, String... columns) {
        Table t = new Table(name, columns);
        tables.add(t);
        return t;
    }

    /** Appends a markdown block to the report. */
    void section(String markdown) {
        report.append(markdown).append("\n\n");
    }

    void section(String heading, String text, Table table) {
        section("### " + heading + "\n\n" + text + "\n\n" + table.markdown());
    }

    void verdict(String id, String prediction, String measured, boolean pass) {
        verdict(id, prediction, measured, pass ? "PASS" : "FAIL");
    }

    void verdict(String id, String prediction, String measured, String status) {
        verdicts.add(new Verdict(id, prediction, measured, status));
        log("VERDICT %s: %s | %s", id, status, measured);
    }

    List<Verdict> verdicts() {
        return List.copyOf(verdicts);
    }

    /** A gzip CSV in the raw directory; gzip writes no timestamp, so equal content gives equal bytes. */
    Writer raw(String name, String header) {
        Path p = raw.resolve(name + ".csv.gz");
        synchronized (rawFiles) {
            rawFiles.add(p);
        }
        try {
            Writer w = new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(p)),
                    StandardCharsets.UTF_8));
            w.write(header);
            w.write('\n');
            return w;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    void log(String fmt, Object... args) {
        Object[] a = new Object[args.length + 1];
        a[0] = (System.nanoTime() - t0) / 1e9;
        System.arraycopy(args, 0, a, 1, args.length);
        log.printf(Locale.ROOT, "[%7.1fs] " + fmt + "%n", a);
        log.flush();
    }

    private void writeReport() throws IOException {
        long pass = verdicts.stream().filter(v -> v.status().startsWith("PASS")).count();
        StringBuilder md = new StringBuilder();
        md.append("# Framework simulations (scale ").append(scale.name().toLowerCase(Locale.ROOT)).append(", seed ")
                .append(seed).append(")\n\n## Verdicts\n\n").append(verdicts.size()).append(" checks, ").append(pass)
                .append(" PASS.\n\n| id | prediction | measured | verdict |\n|---|---|---|---|\n");
        StringBuilder csv = new StringBuilder("id,status,prediction,measured\n");
        for (Verdict v : verdicts) {
            md.append("| ").append(v.id()).append(" | ").append(Table.cell(v.prediction())).append(" | ")
                    .append(Table.cell(v.measured())).append(" | **").append(v.status()).append("** |\n");
            csv.append(Table.csv(v.id())).append(',').append(Table.csv(v.status())).append(',')
                    .append(Table.csv(v.prediction())).append(',').append(Table.csv(v.measured())).append('\n');
        }
        md.append("\n## Results\n\n").append(report);
        Files.writeString(out.resolve("summary.md"), md.toString(), StandardCharsets.UTF_8);
        Files.writeString(out.resolve("verdicts.csv"), csv.toString(), StandardCharsets.UTF_8);
        for (Table t : tables) {
            Files.writeString(out.resolve(t.name + ".csv"), t.csv(), StandardCharsets.UTF_8);
        }
    }

    private void writeChecksums() throws IOException {
        StringBuilder sums = new StringBuilder();
        for (Path p : rawFiles.stream().sorted().toList()) {
            sums.append(sha256(p)).append("  ").append(p.getFileName()).append('\n');
        }
        Files.writeString(out.resolve("raw-SHA256SUMS"), sums.toString(), StandardCharsets.UTF_8);
        Files.writeString(raw.resolve("SHA256SUMS"), sums.toString(), StandardCharsets.UTF_8);
    }

    static String sha256(Path p) throws IOException {
        try (InputStream in = Files.newInputStream(p)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 16];
            for (int r = in.read(buf); r >= 0; r = in.read(buf)) {
                md.update(buf, 0, r);
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void close() throws IOException {
        pool.shutdownNow();
        writeChecksums();
    }

    static String f(String fmt, Object... args) {
        return String.format(Locale.ROOT, fmt, args);
    }
}
