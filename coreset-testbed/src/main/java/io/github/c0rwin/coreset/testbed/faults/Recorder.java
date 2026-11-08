package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.testbed.report.Json;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Predicate;

/**
 * Writes one JSON line per run ({@code <experiment>.jsonl.gz}) and keeps every metric per cell in memory for the
 * verdicts and for {@code <experiment>-cells.csv} (mean, sd, p50, p90, max per metric).
 */
final class Recorder implements AutoCloseable {

    /** One run: its cell (the factor levels), seed, scalar metrics and optionally the per-candidate r(Q) vector. */
    record Row(Map<String, Object> cell, int seed, Map<String, Double> metrics, double[] r) {
    }

    static final class Cell {
        final Map<String, Object> params;
        final Map<String, DoubleList> metrics = new LinkedHashMap<>();
        final List<double[]> rs = new ArrayList<>();
        final List<Integer> rSeeds = new ArrayList<>();

        Cell(Map<String, Object> params) {
            this.params = params;
        }

        double[] values(String metric) {
            DoubleList l = metrics.get(metric);
            return l == null ? new double[0] : l.toArray();
        }
    }

    static final class DoubleList {
        private double[] a = new double[16];
        private int n;

        void add(double v) {
            if (n == a.length) {
                a = Arrays.copyOf(a, 2 * n);
            }
            a[n++] = v;
        }

        double[] toArray() {
            return Arrays.copyOf(a, n);
        }
    }

    private final String experiment;
    private final Path dir;
    private final Writer jsonl;
    private final Map<String, Cell> cells = new LinkedHashMap<>();
    private long rows;

    Recorder(Path dir, String experiment) throws IOException {
        this.experiment = experiment;
        this.dir = dir;
        this.jsonl = Json.jsonLinesGzip(dir.resolve(experiment + ".jsonl.gz"));
    }

    static Map<String, Object> cell(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put((String) kv[i], kv[i + 1]);
        }
        return m;
    }

    static Map<String, Double> metrics(Object... kv) {
        Map<String, Double> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            Object v = kv[i + 1];
            if (v != null) {
                m.put((String) kv[i], ((Number) v).doubleValue());
            }
        }
        return m;
    }

    static String key(Map<String, Object> cell) {
        return new TreeMap<>(cell).toString();
    }

    void addAll(List<Row> batch) {
        for (Row r : batch) {
            add(r);
        }
    }

    void add(Row row) {
        Map<String, Object> line = new LinkedHashMap<>();
        line.put("experiment", experiment);
        line.putAll(row.cell());
        line.put("seed", row.seed());
        line.putAll(row.metrics());
        if (row.r() != null) {
            float[] f = new float[row.r().length];
            for (int j = 0; j < f.length; j++) {
                f[j] = (float) row.r()[j];
            }
            line.put("r", f);
        }
        try {
            jsonl.write(Json.toJsonLine(line));
            jsonl.write('\n');
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        Cell c = cells.computeIfAbsent(key(row.cell()), k -> new Cell(row.cell()));
        for (Map.Entry<String, Double> e : row.metrics().entrySet()) {
            c.metrics.computeIfAbsent(e.getKey(), k -> new DoubleList()).add(e.getValue());
        }
        if (row.r() != null) {
            c.rs.add(row.r());
            c.rSeeds.add(row.seed());
        }
        rows++;
    }

    Cell get(Map<String, Object> cell) {
        Cell c = cells.get(key(cell));
        if (c == null) {
            throw new IllegalArgumentException(experiment + ": no runs in cell " + cell);
        }
        return c;
    }

    boolean has(Map<String, Object> cell) {
        return cells.containsKey(key(cell));
    }

    double[] values(Map<String, Object> cell, String metric) {
        return get(cell).values(metric);
    }

    List<Cell> cells(Predicate<Map<String, Object>> filter) {
        List<Cell> out = new ArrayList<>();
        for (Cell c : cells.values()) {
            if (filter.test(c.params)) {
                out.add(c);
            }
        }
        return out;
    }

    long rows() {
        return rows;
    }

    @Override
    public void close() throws IOException {
        jsonl.close();
        Set<String> params = new LinkedHashSet<>();
        Set<String> metrics = new LinkedHashSet<>();
        for (Cell c : cells.values()) {
            params.addAll(c.params.keySet());
            metrics.addAll(c.metrics.keySet());
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.join(",", params)).append(",runs");
        for (String m : metrics) {
            for (String s : new String[] {"mean", "sd", "p50", "p90", "max"}) {
                sb.append(',').append(m).append('_').append(s);
            }
        }
        sb.append('\n');
        for (Cell c : cells.values()) {
            List<String> row = new ArrayList<>();
            for (String p : params) {
                Object v = c.params.get(p);
                row.add(v == null ? "" : v.toString());
            }
            int runs = 0;
            for (DoubleList l : c.metrics.values()) {
                runs = Math.max(runs, l.n);
            }
            row.add(Integer.toString(runs));
            for (String m : metrics) {
                double[] x = c.values(m);
                for (double v : new double[] {Stats.mean(x), Stats.sd(x), Stats.quantile(x, 0.5),
                    Stats.quantile(x, 0.9), x.length == 0 ? Double.NaN : Stats.max(x)}) {
                    row.add(Double.isNaN(v) ? "" : String.format(Locale.ROOT, "%.6g", v));
                }
            }
            sb.append(String.join(",", row)).append('\n');
        }
        Files.writeString(dir.resolve(experiment + "-cells.csv"), sb.toString(), StandardCharsets.UTF_8);
    }
}
