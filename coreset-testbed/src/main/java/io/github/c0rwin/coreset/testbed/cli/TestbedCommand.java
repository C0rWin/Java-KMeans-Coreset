package io.github.c0rwin.coreset.testbed.cli;

import io.github.c0rwin.coreset.node.ServiceLoaderAlgorithmResolver;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.data.PublicDataset;
import io.github.c0rwin.coreset.testbed.faults.FaultsCommand;
import io.github.c0rwin.coreset.testbed.report.Json;
import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import io.github.c0rwin.coreset.testbed.run.OutputDirectory;
import io.github.c0rwin.coreset.testbed.run.Provenance;
import io.github.c0rwin.coreset.testbed.run.Testbed;
import io.github.c0rwin.coreset.testbed.run.TestbedConfig;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.stream.Stream;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

/** {@code coreset-testbed}: runs coreset evaluations and aggregates their validity reports. */
@Command(name = "coreset-testbed", mixinStandardHelpOptions = true,
        description = "Evaluates coreset algorithms and writes one validity report per run.",
        subcommands = {TestbedCommand.Run.class, TestbedCommand.Suite.class, TestbedCommand.Aggregate.class,
                TestbedCommand.DatasetsCommand.class, FaultsCommand.class})
public final class TestbedCommand implements Runnable {

    @Spec
    CommandLine.Model.CommandSpec spec;

    @Override
    public void run() {
        spec.commandLine().usage(spec.commandLine().getOut());
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new TestbedCommand()).execute(args));
    }

    /** Options shared by the commands that execute a config. */
    static class Execution {
        @Option(names = "--out", required = true, description = "Output directory.")
        Path out;

        @Option(names = "--cache-dir", description = "Public dataset cache. Default: ~/.cache/coreset-testbed")
        Path cacheDir = PublicDataset.defaultCacheDir();

        @Option(names = "--allow-download", description = "Download public datasets that are not cached.")
        boolean allowDownload;

        int execute(TestbedConfig config, PrintWriter err) throws Exception {
            Provenance provenance = Provenance.collect(Path.of("").toAbsolutePath());
            long t0 = System.nanoTime();
            try (Testbed testbed = new Testbed(config, new Datasets(cacheDir, allowDownload),
                    ServiceLoaderAlgorithmResolver.load(), provenance, line -> {
                        err.println(line);
                        err.flush();
                    });
                 OutputDirectory out = new OutputDirectory(this.out, config, provenance)) {
                testbed.run(out);
            }
            err.printf(Locale.ROOT, "done in %.1f min; results in %s%n", (System.nanoTime() - t0) / 6e10, out);
            err.flush();
            return 0;
        }
    }

    @Command(name = "run", mixinStandardHelpOptions = true, description = "Runs a YAML or JSON config.")
    static final class Run implements Callable<Integer> {
        @Spec
        CommandLine.Model.CommandSpec spec;

        @Option(names = "--config", required = true, description = "Config file (.yaml, .yml or .json).")
        Path config;

        @CommandLine.Mixin
        Execution execution;

        @Override
        public Integer call() throws Exception {
            return execution.execute(Json.readConfig(config), spec.commandLine().getErr());
        }
    }

    @Command(name = "suite", mixinStandardHelpOptions = true,
            description = "Runs a built-in suite: smoke (about a minute) or standard (see docs/benchmarks).")
    static final class Suite implements Callable<Integer> {
        @Spec
        CommandLine.Model.CommandSpec spec;

        @Parameters(index = "0", description = "smoke or standard")
        String name;

        @Option(names = "--replicates", description = "Override the suite's replicate count.")
        Integer replicates;

        @CommandLine.Mixin
        Execution execution;

        @Override
        public Integer call() throws Exception {
            TestbedConfig cfg = builtin(name);
            if (replicates != null) {
                cfg = new TestbedConfig(cfg.name(), cfg.problem(), cfg.datasets(), cfg.ks(), cfg.sizes(),
                        cfg.algorithms(), cfg.modes(), replicates, cfg.seed(), cfg.evaluation(), cfg.exclude());
            }
            return execution.execute(cfg, spec.commandLine().getErr());
        }
    }

    /** Loads {@code suites/<name>.yaml} from the classpath. */
    public static TestbedConfig builtin(String name) throws IOException {
        String resource = "/suites/" + name.toLowerCase(Locale.ROOT) + ".yaml";
        try (InputStream in = TestbedCommand.class.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalArgumentException("unknown suite '" + name + "', expected smoke or standard");
            }
            return Json.parseConfig(new String(in.readAllBytes(), StandardCharsets.UTF_8), true);
        }
    }

    @Command(name = "aggregate", mixinStandardHelpOptions = true,
            description = "Re-aggregates reports into summary.csv and summary.md.")
    static final class Aggregate implements Callable<Integer> {
        @Option(names = "--in", required = true,
                description = "A results directory, a reports.jsonl.gz file, or a directory of report JSON files.")
        Path in;

        @Option(names = "--out", description = "Where to write the summary. Default: the input directory.")
        Path out;

        @Override
        public Integer call() throws Exception {
            List<ValidityReport> reports = new ArrayList<>();
            Path dir;
            if (Files.isRegularFile(in)) {
                reports.addAll(Json.readJsonLinesGzip(in));
                dir = in.toAbsolutePath().getParent();
            } else if (Files.isRegularFile(in.resolve(OutputDirectory.REPORTS))) {
                reports.addAll(Json.readJsonLinesGzip(in.resolve(OutputDirectory.REPORTS)));
                dir = in;
            } else {
                Path runs = Files.isDirectory(in.resolve("runs")) ? in.resolve("runs") : in;
                try (Stream<Path> files = Files.list(runs)) {
                    for (Path f : files.filter(p -> p.toString().endsWith(".json")).sorted().toList()) {
                        reports.add(Json.readReport(f));
                    }
                }
                dir = in;
            }
            Path target = out == null ? dir : out;
            Files.createDirectories(target);
            List<Testbed.PoolInfo> pools = new ArrayList<>();
            Path shared = dir.resolve("shared.json");
            if (Files.isRegularFile(shared)) {
                pools.addAll(List.of(Json.mapper().readValue(shared.toFile(), Testbed.PoolInfo[].class)));
            }
            OutputDirectory.writeSummary(target, reports, pools);
            return 0;
        }
    }

    @Command(name = "datasets", mixinStandardHelpOptions = true, description = "Lists or fetches public datasets.",
            subcommands = {DatasetsCommand.ListDatasets.class, DatasetsCommand.Fetch.class})
    static final class DatasetsCommand implements Runnable {
        @Spec
        CommandLine.Model.CommandSpec spec;

        @Override
        public void run() {
            spec.commandLine().usage(spec.commandLine().getOut());
        }

        @Command(name = "list", description = "Lists the public dataset descriptors.")
        static final class ListDatasets implements Callable<Integer> {
            @Spec
            CommandLine.Model.CommandSpec spec;

            @Override
            public Integer call() {
                PrintWriter out = spec.commandLine().getOut();
                for (PublicDataset d : new TreeMap<>(PublicDataset.REGISTRY).values()) {
                    out.printf(Locale.ROOT, "%-12s %9d x %-4d %s%n    sha256 %s%n    license: %s%n    %s%n", d.id(),
                            d.n(), d.d(), d.url(), d.sha256(), d.license(), d.citation());
                }
                out.flush();
                return 0;
            }
        }

        @Command(name = "fetch", description = "Downloads a public dataset into the cache and verifies its digest.")
        static final class Fetch implements Callable<Integer> {
            @Spec
            CommandLine.Model.CommandSpec spec;

            @Parameters(index = "0", description = "Dataset name (see 'datasets list').")
            String name;

            @Option(names = "--cache-dir", description = "Cache directory. Default: ~/.cache/coreset-testbed")
            Path cacheDir = PublicDataset.defaultCacheDir();

            @Override
            public Integer call() throws Exception {
                PublicDataset d = PublicDataset.REGISTRY.get(name);
                if (d == null) {
                    throw new IllegalArgumentException("unknown dataset '" + name + "', known: "
                            + new TreeMap<>(PublicDataset.REGISTRY).keySet());
                }
                Path f = d.fetch(cacheDir);
                spec.commandLine().getOut().println(f);
                spec.commandLine().getOut().flush();
                return 0;
            }
        }
    }
}
