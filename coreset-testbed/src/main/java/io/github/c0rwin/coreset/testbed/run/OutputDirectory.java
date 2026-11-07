package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.testbed.report.Aggregator;
import io.github.c0rwin.coreset.testbed.report.Json;
import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Writes a run's results: {@code runs/<run>.json} (one validity report each), {@code reports.jsonl.gz} (all reports,
 * compact), {@code shared.json} (cost of the shared full-data candidates), {@code config.json} and, on close,
 * {@code summary.csv} and {@code summary.md}.
 */
public final class OutputDirectory implements Testbed.Sink, AutoCloseable {

    public static final String REPORTS = "reports.jsonl.gz";

    private final Path dir;
    private final Path runs;
    private final Writer lines;
    private final List<ValidityReport> reports = new ArrayList<>();
    private final List<Testbed.PoolInfo> pools = new ArrayList<>();

    public OutputDirectory(Path dir, TestbedConfig config, Provenance provenance) throws IOException {
        this.dir = dir;
        this.runs = dir.resolve("runs");
        Files.createDirectories(runs);
        Files.writeString(dir.resolve("config.json"), Json.toJson(config) + "\n", StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("environment.json"), Json.toJson(Map.of(
                "gitCommit", String.valueOf(provenance.gitCommit()),
                "gitDirty", String.valueOf(provenance.gitDirty()),
                "jvm", provenance.jvm(), "host", provenance.host())) + "\n", StandardCharsets.UTF_8);
        this.lines = Json.jsonLinesGzip(dir.resolve(REPORTS));
    }

    @Override
    public void report(ValidityReport r) throws IOException {
        ValidityReport.Config c = r.config();
        String name = String.format("%s__%s__k%d__m%d__%s__r%d.json", r.dataset().id(), c.algorithm().label(),
                c.k(), c.targetSize(), c.modeLabel(), c.replicate()).replaceAll("[^A-Za-z0-9._+=-]", "_");
        Json.writeReport(runs.resolve(name), r);
        lines.write(Json.toJsonLine(r));
        lines.write('\n');
        reports.add(r);
    }

    @Override
    public void pool(Testbed.PoolInfo pool) {
        pools.add(pool);
    }

    public List<ValidityReport> reports() {
        return List.copyOf(reports);
    }

    @Override
    public void close() throws IOException {
        lines.close();
        Files.writeString(dir.resolve("shared.json"), Json.toJson(pools) + "\n", StandardCharsets.UTF_8);
        writeSummary(dir, reports, pools);
    }

    /** Writes {@code summary.csv} and {@code summary.md} for the given reports. */
    public static void writeSummary(Path dir, List<ValidityReport> reports, List<Testbed.PoolInfo> pools)
            throws IOException {
        List<Aggregator.Row> rows = Aggregator.aggregate(reports);
        Files.writeString(dir.resolve("summary.csv"), Aggregator.csv(rows), StandardCharsets.UTF_8);
        StringBuilder md = new StringBuilder("## Results\n\n").append(Aggregator.markdown(rows));
        if (!pools.isEmpty()) {
            md.append("### Shared full-data candidates\n\n")
                    .append("| dataset | k | candidates | generation s | full solve s | full-solution cost |\n")
                    .append("|---|---|---|---|---|---|\n");
            for (Testbed.PoolInfo p : pools) {
                md.append(String.format(Locale.ROOT, "| %s | %d | %d | %.1f | %.1f | %.6g |%n", p.dataset(),
                        p.k(), p.candidates(), p.generationMs() / 1e3, p.fullSolveMs() / 1e3, p.fullSolutionCost()));
            }
            md.append('\n');
        }
        Files.writeString(dir.resolve("summary.md"), md.toString(), StandardCharsets.UTF_8);
    }
}
