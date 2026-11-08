package io.github.c0rwin.coreset.testbed.faults.framework;

import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/** {@code coreset-testbed framework}: simulations testing the predictions of the general loss-invariance framework. */
@Command(name = "framework", mixinStandardHelpOptions = true,
        description = "Simulates the general framework's predictions (A unknown rate and headers, B content-correlated "
                + "loss, C bursts, D multi-level loss, E k-median) and writes summary.md, verdicts.csv, one CSV per "
                + "table and raw per-trial files with their SHA-256 list.")
public final class FrameworkCommand implements Callable<Integer> {

    @Spec
    CommandLine.Model.CommandSpec spec;

    @Option(names = "--out", required = true, description = "Output directory (small files, meant to be committed).")
    Path out;

    @Option(names = "--raw-dir", description = "Raw per-trial output. Default: "
            + "~/.cache/coreset-testbed/framework-sims/<scale>-seed<seed>.")
    Path rawDir;

    @Option(names = "--experiments", split = ",", description = "Subset of A,B,C,D,E. Default: all.")
    List<String> experiments = Lab.EXPERIMENTS;

    @Option(names = "--scale", description = "full (the documented run) or quick (small, for tests).")
    String scale = "full";

    @Option(names = "--seed", description = "Base seed of every data, sampling, packing and channel stream. Default: 42.")
    long seed = 42L;

    @Option(names = "--threads", description = "Worker threads. Default: available processors - 2.")
    int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);

    @Override
    public Integer call() throws Exception {
        PrintWriter err = spec.commandLine().getErr();
        Lab.Scale s = Lab.Scale.valueOf(scale.toUpperCase(Locale.ROOT));
        Path raw = rawDir != null ? rawDir : Path.of(System.getProperty("user.home"), ".cache", "coreset-testbed",
                "framework-sims", s.name().toLowerCase(Locale.ROOT) + "-seed" + seed);
        try (Lab lab = new Lab(out, raw, s, seed, threads, err)) {
            lab.run(experiments);
        }
        return 0;
    }
}
