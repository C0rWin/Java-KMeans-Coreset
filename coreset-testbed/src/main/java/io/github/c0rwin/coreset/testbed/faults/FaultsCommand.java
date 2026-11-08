package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.data.PublicDataset;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/** {@code coreset-testbed faults}: runs the fault-injection experiments F1-F6 of the fault-tolerance study. */
@Command(name = "faults", mixinStandardHelpOptions = true,
        description = "Runs fault-injection experiments (node loss, coreset loss, packet erasure, stragglers, "
                + "merge-reduce depth, tree topology) and writes JSONL, per-cell CSV and summary.md with verdicts.")
public final class FaultsCommand implements Callable<Integer> {

    @Spec
    CommandLine.Model.CommandSpec spec;

    @Option(names = "--out", required = true, description = "Output directory.")
    Path out;

    @Option(names = "--experiments", split = ",", description = "Subset of F1,F2,F3,F4,F5,F6 (default: all), or REAL "
            + "for the real-data pass (cached public datasets, never downloaded).")
    List<String> experiments = FaultLab.EXPERIMENTS;

    @Option(names = "--scale", description = "full (the documented run) or quick (tiny instances, for tests).")
    String scale = "full";

    @Option(names = "--seed", description = "Base seed of every data, sampling and fault stream. Default: 42.")
    long seed = 42L;

    @Option(names = "--threads", description = "Worker threads. Default: available processors - 2.")
    int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 2);

    @Option(names = "--cache-dir", description = "Public dataset cache (never downloaded here).")
    Path cacheDir = PublicDataset.defaultCacheDir();

    @Override
    public Integer call() throws Exception {
        PrintWriter err = spec.commandLine().getErr();
        FaultLab.Scale s = FaultLab.Scale.valueOf(scale.toUpperCase(Locale.ROOT));
        try (FaultLab lab = new FaultLab(out, s, seed, threads, new Datasets(cacheDir, false), err)) {
            lab.run(experiments);
        }
        return 0;
    }
}
