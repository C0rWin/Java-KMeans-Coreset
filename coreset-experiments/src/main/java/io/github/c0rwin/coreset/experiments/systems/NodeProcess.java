package io.github.c0rwin.coreset.experiments.systems;

import io.github.c0rwin.coreset.experiments.Jvm;
import io.github.c0rwin.coreset.grpc.cli.NodeCommand;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One {@code coreset-node} JVM on loopback with ephemeral gRPC and metrics ports, started from this JVM's class path
 * (so SPI plug-ins on it, like the lightweight coreset, are available on the node). Output goes to a log file, from
 * which the ports are read once the node reports it is listening.
 */
final class NodeProcess {

    private static final Pattern LISTENING = Pattern.compile(
            "coreset-node listening on \\[.*?:(\\d+)\\], metrics on .*?:(\\d+)");

    final int index;
    final Process process;
    final int port;
    final int metricsPort;
    final Path log;

    private NodeProcess(int index, Process process, int port, int metricsPort, Path log) {
        this.index = index;
        this.process = process;
        this.port = port;
        this.metricsPort = metricsPort;
        this.log = log;
    }

    static NodeProcess spawn(int index, List<String> jvmFlags, int heapMB, Path logDir, Duration timeout)
            throws IOException, InterruptedException {
        List<String> cmd = new ArrayList<>();
        cmd.add(Jvm.java());
        cmd.addAll(jvmFlags);
        cmd.add("-Xmx" + heapMB + "m");
        cmd.add("-cp");
        cmd.add(Jvm.classPath());
        cmd.add(NodeCommand.class.getName());
        cmd.addAll(List.of("--port", "0", "--metrics-port", "0", "--bind", "127.0.0.1", "--metrics-bind",
                "127.0.0.1"));
        Files.createDirectories(logDir);
        Path log = logDir.resolve("node-" + index + "-" + System.nanoTime() + ".log");
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            Matcher m = LISTENING.matcher(Files.readString(log, StandardCharsets.UTF_8));
            if (m.find()) {
                return new NodeProcess(index, p, Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)), log);
            }
            if (!p.isAlive()) {
                throw new IOException("node " + index + " exited with " + p.exitValue() + ", see " + log);
            }
            Thread.sleep(20);
        }
        p.destroyForcibly();
        throw new IOException("node " + index + " did not report its ports within " + timeout + ", see " + log);
    }

    String target() {
        return "127.0.0.1:" + port;
    }

    String metricsUrl() {
        return "http://127.0.0.1:" + metricsPort + "/metrics";
    }

    long pid() {
        return process.pid();
    }

    boolean alive() {
        return process.isAlive();
    }

    /** SIGKILL and wait for the exit, so no later call can still reach the node; sessions vanish with it. */
    void kill() throws InterruptedException {
        if (!process.destroyForcibly().waitFor(10, TimeUnit.SECONDS)) {
            throw new IllegalStateException("node " + index + " survived SIGKILL for 10 s");
        }
    }

    void stop() throws InterruptedException {
        process.destroy();
        if (!process.waitFor(10, TimeUnit.SECONDS)) {
            process.destroyForcibly().waitFor(10, TimeUnit.SECONDS);
        }
    }
}
