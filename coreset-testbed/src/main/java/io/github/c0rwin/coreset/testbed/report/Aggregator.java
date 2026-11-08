package io.github.c0rwin.coreset.testbed.report;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.ToDoubleFunction;

/**
 * Groups reports by (dataset, mode, algorithm, k, size) and summarises each group the ESA'22 way: the mean over runs
 * of each run's maximum distortion, with its standard deviation and the maximum over runs for tail risk. Taking the
 * maximum over runs alone would overstate typical behaviour, which is why the mean of maxima is the headline number.
 */
public final class Aggregator {

    private Aggregator() {
    }

    /**
     * One group's summary; {@code NaN} marks a statistic that does not apply (e.g. stream levels in offline mode).
     *
     * @param distortionMeanOfMax ESA'22 headline: mean over runs of the per-run maximum distortion
     * @param levelFirst          stream: mean distortion of the level-0 summary (one compressed block)
     * @param levelLast           stream: mean distortion of the highest level formed
     * @param heapMb              mean heap peak minus pre-run baseline (sequential runs only)
     */
    public record Row(String dataset, String mode, String algorithm, int k, int m, int runs,
                      double distortionMeanOfMax, double distortionSd, double distortionMax, String worstGenerator,
                      double levelFirst, double levelLast, double deltaSweep, double ciCoverage,
                      double weightRelErr, double outputSize, double relativeCost, double buildMs, double buildCpuMs,
                      double allocMb, double heapMb, double distanceEvals, double bytesSerialized,
                      double validationMs, double passRate, double loadAverage) {
    }

    private record Key(String dataset, String mode, String algorithm, int k, int m) {
    }

    public static List<Row> aggregate(List<ValidityReport> reports) {
        Map<Key, List<ValidityReport>> groups = new LinkedHashMap<>();
        for (ValidityReport r : reports) {
            Key key = new Key(r.dataset().id(), r.config().modeLabel(), r.config().algorithm().label(),
                    r.config().k(), r.config().targetSize());
            groups.computeIfAbsent(key, x -> new ArrayList<>()).add(r);
        }
        List<Row> rows = new ArrayList<>(groups.size());
        for (Map.Entry<Key, List<ValidityReport>> e : groups.entrySet()) {
            Key key = e.getKey();
            List<ValidityReport> g = e.getValue();
            double[] maxima = g.stream().mapToDouble(r -> r.distortion().max()).toArray();
            double mean = mean(maxima);
            double sd = 0.0;
            for (double v : maxima) {
                sd += (v - mean) * (v - mean);
            }
            sd = maxima.length > 1 ? Math.sqrt(sd / (maxima.length - 1)) : 0.0;
            double max = 0.0;
            for (double v : maxima) {
                max = Math.max(max, v);
            }
            rows.add(new Row(key.dataset(), key.mode(), key.algorithm(), key.k(), key.m(), g.size(), mean, sd, max,
                    mostFrequent(g),
                    meanOf(g, r -> levelDistortion(r, true)), meanOf(g, r -> levelDistortion(r, false)),
                    meanOf(g, r -> r.distortion().benchmarkDeltaSweep() == null ? Double.NaN
                            : r.distortion().benchmarkDeltaSweep().distortion()),
                    meanOf(g, r -> r.distortion().ciCoverage() == null ? Double.NaN : r.distortion().ciCoverage()),
                    meanOf(g, r -> r.integrity().weightRelErr()),
                    meanOf(g, r -> r.construction().outputSize()),
                    meanOf(g, r -> r.downstream().relativeCost()),
                    meanOf(g, r -> r.construction().wallMs()),
                    meanOf(g, r -> r.construction().cpuMs()),
                    meanOf(g, r -> r.construction().allocBytes() / 1048576.0),
                    meanOf(g, r -> r.construction().peakHeapBytes() == null
                            || r.construction().baselineHeapBytes() == null ? Double.NaN
                            : (r.construction().peakHeapBytes() - r.construction().baselineHeapBytes()) / 1048576.0),
                    meanOf(g, r -> r.construction().distanceEvals()),
                    meanOf(g, r -> r.construction().bytesSerialized()),
                    meanOf(g, r -> r.validation().wallMs()),
                    meanOf(g, r -> r.verdict().pass() ? 1.0 : 0.0),
                    meanOf(g, r -> r.run().loadAverage() == null ? Double.NaN : r.run().loadAverage())));
        }
        rows.sort(Comparator.comparing(Row::dataset).thenComparing(Row::mode).thenComparingInt(Row::k)
                .thenComparingInt(Row::m).thenComparing(Row::algorithm));
        return rows;
    }

    private static double levelDistortion(ValidityReport r, boolean first) {
        List<ValidityReport.Level> levels = r.construction().perLevel();
        if (levels == null || levels.isEmpty()) {
            return Double.NaN;
        }
        Double d = levels.get(first ? 0 : levels.size() - 1).distortionMax();
        return d == null ? Double.NaN : d;
    }

    private static String mostFrequent(List<ValidityReport> g) {
        Map<String, Integer> counts = new HashMap<>();
        String best = null;
        for (ValidityReport r : g) {
            String gen = r.distortion().argmaxGenerator();
            String family = gen == null ? "?" : gen.replaceAll("#\\d+$", "");
            int c = counts.merge(family, 1, Integer::sum);
            if (best == null || c > counts.get(best) || (c == counts.get(best) && family.compareTo(best) < 0)) {
                best = family;
            }
        }
        return Objects.requireNonNullElse(best, "?");
    }

    /** Mean over the values that are not NaN; NaN when none is. */
    private static double meanOf(List<ValidityReport> g, ToDoubleFunction<ValidityReport> f) {
        double s = 0.0;
        int n = 0;
        for (ValidityReport r : g) {
            double v = f.applyAsDouble(r);
            if (!Double.isNaN(v)) {
                s += v;
                n++;
            }
        }
        return n == 0 ? Double.NaN : s / n;
    }

    private static double mean(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        return v.length == 0 ? Double.NaN : s / v.length;
    }

    public static final List<String> CSV_HEADER = List.of("dataset", "mode", "algorithm", "k", "m", "runs",
            "distortion_mean_of_max", "distortion_sd", "distortion_max", "worst_generator", "level_first",
            "level_last", "delta_sweep", "ci_coverage", "weight_rel_err", "output_size", "relative_cost",
            "build_ms", "build_cpu_ms", "alloc_mb", "heap_mb", "distance_evals", "bytes_serialized",
            "validation_ms", "pass_rate", "load_average");

    public static String csv(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.join(",", CSV_HEADER)).append('\n');
        for (Row r : rows) {
            sb.append(String.join(",", r.dataset(), r.mode(), r.algorithm(), Integer.toString(r.k()),
                    Integer.toString(r.m()), Integer.toString(r.runs()), num(r.distortionMeanOfMax()),
                    num(r.distortionSd()), num(r.distortionMax()), r.worstGenerator(), num(r.levelFirst()),
                    num(r.levelLast()), num(r.deltaSweep()), num(r.ciCoverage()), num(r.weightRelErr()),
                    num(r.outputSize()), num(r.relativeCost()), num(r.buildMs()), num(r.buildCpuMs()),
                    num(r.allocMb()), num(r.heapMb()), num(r.distanceEvals()), num(r.bytesSerialized()),
                    num(r.validationMs()), num(r.passRate()), num(r.loadAverage()))).append('\n');
        }
        return sb.toString();
    }

    private static String num(double v) {
        return Double.isNaN(v) ? "" : String.format(Locale.ROOT, "%.6g", v);
    }

    /** One markdown table per dataset; columns that are empty for the whole dataset are left out. */
    public static String markdown(List<Row> rows) {
        StringBuilder sb = new StringBuilder();
        Map<String, List<Row>> byDataset = new LinkedHashMap<>();
        for (Row r : rows) {
            byDataset.computeIfAbsent(r.dataset(), x -> new ArrayList<>()).add(r);
        }
        for (Map.Entry<String, List<Row>> e : byDataset.entrySet()) {
            List<Row> g = e.getValue();
            boolean levels = g.stream().anyMatch(r -> !Double.isNaN(r.levelLast()));
            boolean sweep = g.stream().anyMatch(r -> !Double.isNaN(r.deltaSweep()));
            boolean heap = g.stream().anyMatch(r -> !Double.isNaN(r.heapMb()));
            sb.append("### ").append(e.getKey()).append("\n\n");
            List<String> head = new ArrayList<>(List.of("mode", "k", "m", "algorithm", "runs", "distortion",
                    "sd", "max", "worst candidate"));
            if (levels) {
                head.addAll(List.of("level 0", "top level"));
            }
            if (sweep) {
                head.add("δ-sweep");
            }
            head.addAll(List.of("rel. cost", "size", "build ms", "build cpu ms", "alloc MB"));
            if (heap) {
                head.add("heap MB");
            }
            head.addAll(List.of("eval ms", "pass"));
            sb.append("| ").append(String.join(" | ", head)).append(" |\n");
            sb.append("|").append("---|".repeat(head.size())).append('\n');
            for (Row r : g) {
                List<String> cells = new ArrayList<>(List.of(r.mode(), Integer.toString(r.k()),
                        Integer.toString(r.m()), r.algorithm(), Integer.toString(r.runs()),
                        fmt(r.distortionMeanOfMax(), 3), fmt(r.distortionSd(), 3), fmt(r.distortionMax(), 3),
                        r.worstGenerator()));
                if (levels) {
                    cells.add(fmt(r.levelFirst(), 3));
                    cells.add(fmt(r.levelLast(), 3));
                }
                if (sweep) {
                    cells.add(fmt(r.deltaSweep(), 3));
                }
                cells.add(String.format(Locale.ROOT, "%+.3f", r.relativeCost()));
                cells.add(String.format(Locale.ROOT, "%.0f", r.outputSize()));
                cells.add(String.format(Locale.ROOT, "%.0f", r.buildMs()));
                cells.add(String.format(Locale.ROOT, "%.0f", r.buildCpuMs()));
                cells.add(String.format(Locale.ROOT, "%.1f", r.allocMb()));
                if (heap) {
                    cells.add(Double.isNaN(r.heapMb()) ? "" : String.format(Locale.ROOT, "%.1f", r.heapMb()));
                }
                cells.add(String.format(Locale.ROOT, "%.0f", r.validationMs()));
                cells.add(String.format(Locale.ROOT, "%.0f%%", 100 * r.passRate()));
                sb.append("| ").append(String.join(" | ", cells)).append(" |\n");
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    private static String fmt(double v, int digits) {
        if (Double.isNaN(v)) {
            return "";
        }
        if (Double.isInfinite(v)) {
            return "∞";
        }
        return String.format(Locale.ROOT, "%." + digits + "f", v);
    }
}
