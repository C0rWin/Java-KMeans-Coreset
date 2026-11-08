package io.github.c0rwin.coreset.experiments.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;

/**
 * {@code summary.md} from {@code results.jsonl} and {@code environment.json}: one table of undisturbed cells (median
 * [min-max] over repetitions), one of fault cells against their in-cell baseline, and every error.
 */
public final class Summary {

    private static final ObjectMapper JSON = new ObjectMapper();

    private Summary() {
    }

    @SuppressWarnings("unchecked")
    public static void write(Path dir) throws IOException {
        List<Map<String, Object>> rows = new ArrayList<>();
        Path results = dir.resolve("results.jsonl");
        if (Files.exists(results)) {
            for (String line : Files.readAllLines(results, StandardCharsets.UTF_8)) {
                if (!line.isBlank()) {
                    rows.add(JSON.readValue(line, Map.class));
                }
            }
        }
        Map<String, Object> env = Files.exists(dir.resolve("environment.json"))
                ? JSON.readValue(dir.resolve("environment.json").toFile(), Map.class) : Map.of();
        StringBuilder sb = new StringBuilder();
        sb.append("# Engine comparison: ").append(name(env)).append("\n\n");
        sb.append("Environment: ").append(env.get("host")).append(", ").append(env.get("os")).append(", ")
                .append(env.get("cpu")).append(", ").append(env.get("cpus")).append(" cpus, ")
                .append(gib(env.get("memoryBytes"))).append(" GiB; ").append(env.get("java")).append("; Spark ")
                .append(env.get("spark")).append(", Flink ").append(env.get("flink")).append(", gRPC ")
                .append(env.get("grpc")).append("; commit ").append(env.get("gitCommit"))
                .append(Boolean.TRUE.equals(env.get("gitDirty")) ? " (dirty)" : "").append(".\n\n");
        sb.append("Load average (1/5/15 min) at start: ").append(env.get("loadAverage")).append("; per cell 1-min load ")
                .append(loadRange(rows)).append(".\n\n");
        sb.append("Times are seconds, median [min-max] over the measured repetitions (warm-up excluded); startup is"
                + " reported separately and not part of total. total = build + gather + solve. CPU s and RSS cover"
                + " every process of the system (sampled). Bytes: ingest = raw points shipped to workers, summary ="
                + " partial coresets / task results shipped. cost ratio = full-data cost of the deployed centers /"
                + " reference cost.\n\n");
        undisturbed(sb, rows);
        faults(sb, rows);
        errors(sb, rows);
        Files.writeString(dir.resolve("summary.md"), sb.toString(), StandardCharsets.UTF_8);
    }

    private static String name(Map<String, Object> env) {
        Object c = env.get("config");
        return c instanceof Map<?, ?> m ? String.valueOf(m.get("name")) : "?";
    }

    private static String key(Map<String, Object> r) {
        return r.get("dataset") + "|" + r.get("system") + "|" + r.get("parallelism") + "|" + r.get("algorithm") + "|"
                + r.get("fault_label");
    }

    private static Map<String, List<Map<String, Object>>> group(List<Map<String, Object>> rows, String phase) {
        Map<String, List<Map<String, Object>>> g = new LinkedHashMap<>();
        for (Map<String, Object> r : rows) {
            if (phase.equals(r.get("phase"))) {
                g.computeIfAbsent(key(r), k -> new ArrayList<>()).add(r);
            }
        }
        return g;
    }

    private static void undisturbed(StringBuilder sb, List<Map<String, Object>> rows) {
        Map<String, List<Map<String, Object>>> g = group(rows, "measure");
        if (g.isEmpty()) {
            return;
        }
        sb.append("## Undisturbed runs\n\n");
        sb.append("| dataset | system | N | algorithm | reps | startup | build | gather | solve | total | points/s"
                + " | CPU s | peak RSS max MB | RSS sum MB | bytes ingest | bytes summary | coreset pts | cost ratio"
                + " | distortion max | weight err |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (List<Map<String, Object>> rs : g.values()) {
            Map<String, Object> f = rs.get(0);
            sb.append("| ").append(f.get("dataset")).append(" | ").append(f.get("system")).append(" | ")
                    .append(f.get("parallelism")).append(" | ").append(f.get("algorithm")).append(" | ")
                    .append(rs.size()).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("startup_s"))))).append(" | ")
                    .append(spread(col(rs, r -> r.get("build_s")))).append(" | ")
                    .append(spread(col(rs, r -> r.get("gather_s")))).append(" | ")
                    .append(spread(col(rs, r -> r.get("solve_s")))).append(" | ")
                    .append(spread(col(rs, r -> r.get("total_s")))).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("points_per_s"))))).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("cpu_s"))))).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("peak_rss_mb_max"))))).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("peak_rss_mb_sum"))))).append(" | ")
                    .append(bytes(Stats.median(col(rs, r -> metric(r, "bytes_ingest"))))).append(" | ")
                    .append(bytes(Stats.median(col(rs, r -> metric(r, "bytes_summary"))))).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("coreset_size"))))).append(" | ")
                    .append(spread(col(rs, r -> r.get("cost_ratio")))).append(" | ")
                    .append(num(Stats.median(col(rs, r -> r.get("distortion_max"))))).append(" | ")
                    .append(sci(Stats.median(col(rs, r -> r.get("weight_error"))))).append(" |\n");
        }
        sb.append('\n');
    }

    private static void faults(StringBuilder sb, List<Map<String, Object>> rows) {
        Map<String, List<Map<String, Object>>> g = group(rows, "fault");
        if (g.isEmpty()) {
            return;
        }
        Map<String, List<Map<String, Object>>> base = group(rows, "baseline");
        sb.append("## Fault injection\n\n");
        sb.append("Baseline = undisturbed repetition(s) in the same cell and seed (with the same checkpointing). "
                + "recovery = node failure detection (grpc), executor replacement registered (spark) or task restore"
                + " from checkpoint (flink), in seconds after the failure.\n\n");
        sb.append("| dataset | system | N | algorithm | fault | reps | baseline build | fault build | slowdown |"
                + " recovery | lost fraction | equal to baseline | baseline cost ratio | fault cost ratio |\n");
        sb.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (Map.Entry<String, List<Map<String, Object>>> e : g.entrySet()) {
            List<Map<String, Object>> rs = e.getValue();
            List<Map<String, Object>> bs = base.getOrDefault(e.getKey(), List.of());
            Map<String, Object> f = rs.get(0);
            double bb = Stats.median(col(bs, r -> r.get("build_s")));
            double fb = Stats.median(col(rs, r -> r.get("build_s")));
            long equal = rs.stream().filter(r -> Boolean.TRUE.equals(r.get("equal_to_baseline"))).count();
            sb.append("| ").append(f.get("dataset")).append(" | ").append(f.get("system")).append(" | ")
                    .append(f.get("parallelism")).append(" | ").append(f.get("algorithm")).append(" | ")
                    .append(f.get("fault_label")).append(" | ").append(rs.size()).append(" | ")
                    .append(num(bb)).append(" | ").append(spread(col(rs, r -> r.get("build_s")))).append(" | ")
                    .append(num(fb / bb)).append("x | ")
                    .append(spread(col(rs, Summary::recovery))).append(" | ")
                    .append(spread(col(rs, r -> fault(r, "lost_fraction")))).append(" | ")
                    .append(equal).append('/').append(rs.size()).append(" | ")
                    .append(spread(col(bs, r -> r.get("cost_ratio")))).append(" | ")
                    .append(spread(col(rs, r -> r.get("cost_ratio")))).append(" |\n");
        }
        sb.append('\n');
    }

    private static Object recovery(Map<String, Object> r) {
        Object f = r.get("fault");
        if (!(f instanceof Map<?, ?> m)) {
            return null;
        }
        for (Object k : m.keySet()) {
            String s = k.toString();
            if (s.equals("restore_s") || s.equals("replacement_registered_s") || s.startsWith("detect_s")) {
                return m.get(k);
            }
        }
        return m.get("executor_removed_s");
    }

    private static void errors(StringBuilder sb, List<Map<String, Object>> rows) {
        List<Map<String, Object>> errs = rows.stream().filter(r -> "error".equals(r.get("phase"))).toList();
        if (errs.isEmpty()) {
            return;
        }
        sb.append("## Errors\n\n");
        for (Map<String, Object> r : errs) {
            sb.append("- `").append(r.get("cell")).append("`: ").append(String.valueOf(r.get("error"))
                    .replace('\n', ' ')).append('\n');
        }
        sb.append('\n');
    }

    private static Object metric(Map<String, Object> r, String k) {
        return r.get("metrics") instanceof Map<?, ?> m ? m.get(k) : null;
    }

    private static Object fault(Map<String, Object> r, String k) {
        return r.get("fault") instanceof Map<?, ?> m ? m.get(k) : null;
    }

    private static List<Double> col(List<Map<String, Object>> rs, Function<Map<String, Object>, Object> f) {
        List<Double> out = new ArrayList<>();
        for (Map<String, Object> r : rs) {
            Object v = f.apply(r);
            out.add(v instanceof Number x ? x.doubleValue() : Double.NaN);
        }
        return out;
    }

    private static String loadRange(List<Map<String, Object>> rows) {
        List<Double> l = new ArrayList<>();
        for (Map<String, Object> r : rows) {
            for (String k : List.of("load_before", "load_after")) {
                if (r.get(k) instanceof List<?> x && !x.isEmpty() && x.get(0) instanceof Number v) {
                    l.add(v.doubleValue());
                }
            }
        }
        return l.isEmpty() ? "n/a" : num(Stats.min(l)) + "-" + num(Stats.max(l));
    }

    private static String spread(List<Double> xs) {
        double med = Stats.median(xs);
        if (Double.isNaN(med)) {
            return "-";
        }
        double lo = Stats.min(xs);
        double hi = Stats.max(xs);
        return lo == hi ? num(med) : num(med) + " [" + num(lo) + "-" + num(hi) + "]";
    }

    static String num(double v) {
        if (Double.isNaN(v)) {
            return "-";
        }
        double a = Math.abs(v);
        if (a >= 1e5) {
            return String.format(Locale.ROOT, "%.3g", v);
        }
        return String.format(Locale.ROOT, a >= 100 ? "%.0f" : a >= 1 ? "%.2f" : "%.3f", v);
    }

    private static String sci(double v) {
        return Double.isNaN(v) ? "-" : String.format(Locale.ROOT, "%.1e", v);
    }

    private static String bytes(double v) {
        if (Double.isNaN(v)) {
            return "-";
        }
        return v >= 1 << 20 ? String.format(Locale.ROOT, "%.1f MiB", v / (1 << 20))
                : v >= 1 << 10 ? String.format(Locale.ROOT, "%.1f KiB", v / (1 << 10))
                : String.format(Locale.ROOT, "%.0f B", v);
    }

    private static String gib(Object bytes) {
        return bytes instanceof Number n ? String.format(Locale.ROOT, "%.0f", n.doubleValue() / (1L << 30)) : "?";
    }
}
