package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/** Where and on what a run executed: git state, JVM, host. Collected once per process. */
public final class Provenance {

    private final String gitCommit;
    private final Boolean gitDirty;
    private final ValidityReport.Jvm jvm;
    private final ValidityReport.Host host;
    private final ValidityReport.Engine engine;

    private Provenance(String gitCommit, Boolean gitDirty, ValidityReport.Jvm jvm, ValidityReport.Host host,
                       ValidityReport.Engine engine) {
        this.gitCommit = gitCommit;
        this.gitDirty = gitDirty;
        this.jvm = jvm;
        this.host = host;
        this.engine = engine;
    }

    /** Probes git in {@code workDir} (null fields when it is not a checkout or git is missing). */
    public static Provenance collect(Path workDir) {
        String commit = command(workDir, "git", "rev-parse", "HEAD");
        String status = commit == null ? null : command(workDir, "git", "status", "--porcelain");
        Boolean dirty = status == null ? null : !status.isBlank();
        Runtime rt = Runtime.getRuntime();
        ValidityReport.Jvm jvm = new ValidityReport.Jvm(System.getProperty("java.vendor"),
                System.getProperty("java.version"),
                List.copyOf(ManagementFactory.getRuntimeMXBean().getInputArguments()));
        long mem = ManagementFactory.getOperatingSystemMXBean() instanceof com.sun.management.OperatingSystemMXBean o
                ? o.getTotalMemorySize() : 0L;
        ValidityReport.Host host = new ValidityReport.Host(
                System.getProperty("os.name") + " " + System.getProperty("os.version"),
                System.getProperty("os.arch"), cpuModel(workDir), rt.availableProcessors(), mem);
        String version = Provenance.class.getPackage().getImplementationVersion();
        return new Provenance(commit, dirty, jvm, host,
                new ValidityReport.Engine("local", version == null ? "dev" : version));
    }

    ValidityReport.Run run(int concurrentRuns) {
        double load = ManagementFactory.getOperatingSystemMXBean().getSystemLoadAverage();
        return new ValidityReport.Run(UUID.randomUUID().toString(), Instant.now().toString(), gitCommit, gitDirty,
                jvm, host, engine, concurrentRuns, load < 0 ? null : load);
    }

    public String gitCommit() {
        return gitCommit;
    }

    public Boolean gitDirty() {
        return gitDirty;
    }

    public ValidityReport.Host host() {
        return host;
    }

    public ValidityReport.Jvm jvm() {
        return jvm;
    }

    private static String cpuModel(Path workDir) {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("mac")) {
            String s = command(workDir, "sysctl", "-n", "machdep.cpu.brand_string");
            if (s != null && !s.isBlank()) {
                return s.trim();
            }
        }
        Path cpuinfo = Path.of("/proc/cpuinfo");
        if (Files.isReadable(cpuinfo)) {
            try {
                for (String line : Files.readAllLines(cpuinfo, StandardCharsets.UTF_8)) {
                    if (line.startsWith("model name")) {
                        return line.substring(line.indexOf(':') + 1).trim();
                    }
                }
            } catch (IOException ignored) {
                // fall through to the generic answer
            }
        }
        return System.getProperty("os.arch");
    }

    /** Trimmed stdout of a successful command, or null. */
    private static String command(Path workDir, String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd).directory(workDir.toFile()).redirectErrorStream(false).start();
            p.getOutputStream().close();
            String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            p.getErrorStream().close();
            if (!p.waitFor(10, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return null;
            }
            return p.exitValue() == 0 ? out.trim() : null;
        } catch (IOException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }
}
