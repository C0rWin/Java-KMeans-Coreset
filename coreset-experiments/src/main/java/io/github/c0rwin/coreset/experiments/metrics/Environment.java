package io.github.c0rwin.coreset.experiments.metrics;

import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

/** Host, JVM and engine versions, recorded once per experiment, and load averages, recorded per cell. */
public final class Environment {

    private static final Properties BUILD = new Properties();

    static {
        try (InputStream in = Environment.class.getResourceAsStream("/coreset-experiments.properties")) {
            if (in != null) {
                BUILD.load(in);
            }
        } catch (IOException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private Environment() {
    }

    /** Build-time property (engine versions, the JVM flags Spark needs on Java 17). */
    public static String build(String key) {
        return BUILD.getProperty(key, "");
    }

    public static Map<String, Object> capture() {
        Map<String, Object> e = new LinkedHashMap<>();
        e.put("timestamp", Instant.now().toString());
        e.put("host", command("hostname")); // InetAddress.getLocalHost can stall on macOS DNS
        e.put("os", System.getProperty("os.name") + " " + System.getProperty("os.version") + " "
                + System.getProperty("os.arch"));
        e.put("cpu", command("sysctl", "-n", "machdep.cpu.brand_string"));
        e.put("cpus", Runtime.getRuntime().availableProcessors());
        e.put("memoryBytes", totalMemory());
        e.put("java", System.getProperty("java.vendor") + " " + System.getProperty("java.runtime.version") + " ("
                + System.getProperty("java.vm.name") + ")");
        e.put("orchestratorJvmArgs", ManagementFactory.getRuntimeMXBean().getInputArguments());
        e.put("spark", build("spark.version"));
        e.put("flink", build("flink.version"));
        e.put("grpc", build("grpc.version"));
        e.put("project", build("project.version"));
        e.put("gitCommit", command("git", "rev-parse", "--short", "HEAD"));
        e.put("gitDirty", !command("git", "status", "--porcelain", "--untracked-files=no").isEmpty());
        e.put("loadAverage", loadAverage());
        return e;
    }

    /** 1, 5 and 15 minute load averages (only the 1-minute value where the OS offers no more). */
    public static List<Double> loadAverage() {
        try {
            if (Files.isReadable(Path.of("/proc/loadavg"))) {
                String[] f = Files.readString(Path.of("/proc/loadavg")).trim().split("\\s+");
                return List.of(Double.parseDouble(f[0]), Double.parseDouble(f[1]), Double.parseDouble(f[2]));
            }
            String s = command("sysctl", "-n", "vm.loadavg").replace("{", "").replace("}", "").trim();
            if (!s.isEmpty()) {
                String[] f = s.split("\\s+");
                return List.of(Double.parseDouble(f[0]), Double.parseDouble(f[1]), Double.parseDouble(f[2]));
            }
        } catch (IOException | RuntimeException ignored) {
            // fall through to the JVM's 1-minute value
        }
        return List.of(ManagementFactory.getOperatingSystemMXBean().getSystemLoadAverage());
    }

    private static long totalMemory() {
        String mac = command("sysctl", "-n", "hw.memsize");
        if (!mac.isEmpty()) {
            return Long.parseLong(mac);
        }
        try {
            for (String line : Files.readAllLines(Path.of("/proc/meminfo"))) {
                if (line.startsWith("MemTotal:")) {
                    return Long.parseLong(line.replaceAll("\\D+", "")) * 1024;
                }
            }
        } catch (IOException | RuntimeException ignored) {
            // unknown
        }
        return -1;
    }

    /** First line of a command's output, empty if it fails. */
    static String command(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.DISCARD).start();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            if (!p.waitFor(10, TimeUnit.SECONDS) || p.exitValue() != 0) {
                return "";
            }
            int nl = out.indexOf('\n');
            return nl < 0 ? out : out.substring(0, nl);
        } catch (IOException e) {
            return "";
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "";
        }
    }
}
