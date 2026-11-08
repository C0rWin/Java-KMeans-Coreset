package io.github.c0rwin.coreset.bench.perf;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Argument parsing, environment banner, logging and statistics shared by the harness entry points. */
final class HarnessSupport {

    private HarnessSupport() {
    }

    /** Parses {@code --key value} pairs; unknown keys are rejected so typos do not silently use defaults. */
    static Map<String, String> args(String[] argv, Map<String, String> defaults) {
        Map<String, String> out = new HashMap<>(defaults);
        for (int i = 0; i < argv.length; i++) {
            String a = argv[i];
            if (!a.startsWith("--") || i + 1 >= argv.length) {
                throw new IllegalArgumentException("expected --key value, got " + a);
            }
            String key = a.substring(2);
            if (!defaults.containsKey(key)) {
                throw new IllegalArgumentException("unknown option --" + key + ", known " + defaults.keySet());
            }
            out.put(key, argv[++i]);
        }
        return out;
    }

    static int[] ints(String csv) {
        return Arrays.stream(csv.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
    }

    /** gRPC and Netty log through java.util.logging; INFO lines would interleave with the markdown output. */
    static void quietLogs() {
        java.util.logging.Logger.getLogger("").setLevel(java.util.logging.Level.WARNING);
    }

    static void banner(String title) {
        Runtime rt = Runtime.getRuntime();
        System.out.printf("# %s%n", title);
        System.out.printf("# java %s (%s), %d cpus, max heap %d MiB, load avg %.2f%n",
                System.getProperty("java.version"), System.getProperty("java.vm.name"), rt.availableProcessors(),
                rt.maxMemory() >> 20, ManagementFactory.getOperatingSystemMXBean().getSystemLoadAverage());
        System.out.printf("# jvm args %s%n", ManagementFactory.getRuntimeMXBean().getInputArguments());
    }

    static double median(List<Double> xs) {
        List<Double> s = new ArrayList<>(xs);
        s.sort(Double::compare);
        int n = s.size();
        return n % 2 == 1 ? s.get(n / 2) : (s.get(n / 2 - 1) + s.get(n / 2)) / 2.0;
    }

    static double min(List<Double> xs) {
        return xs.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
    }

    static double max(List<Double> xs) {
        return xs.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
    }

    static String mib(long bytes) {
        return String.format("%.0f", bytes / (1024.0 * 1024.0));
    }
}
