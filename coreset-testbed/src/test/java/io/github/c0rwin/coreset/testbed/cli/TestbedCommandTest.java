package io.github.c0rwin.coreset.testbed.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.testbed.report.Json;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

class TestbedCommandTest {

    private static int execute(StringWriter out, String... args) {
        CommandLine cli = new CommandLine(new TestbedCommand());
        cli.setOut(new PrintWriter(out, true));
        cli.setErr(new PrintWriter(new StringWriter(), true));
        return cli.execute(args);
    }

    @Test
    void runWritesReportsAndSummariesThatAggregateReproduces(@TempDir Path dir) throws Exception {
        Path config = dir.resolve("tiny.yaml");
        Files.writeString(config, """
                ---
                name: cli
                replicates: 2
                datasets:
                  - {id: esa, type: esa22, params: {k: 3, alpha: 4}}
                ks: [3]
                sizes: [20]
                algorithms: [{id: uniform}, {id: mean}]
                modes: [{type: offline}, {type: stream, blockCount: 4}]
                evaluation: {candidatesPerGenerator: 1, fullRestarts: 1}
                """, StandardCharsets.UTF_8);
        Path out = dir.resolve("out");
        assertEquals(0, execute(new StringWriter(), "run", "--config", config.toString(), "--out", out.toString()));

        try (Stream<Path> runs = Files.list(out.resolve("runs"))) {
            assertEquals(2 * 2 * 2, runs.count());
        }
        assertEquals(8, Json.readJsonLinesGzip(out.resolve("reports.jsonl.gz")).size());
        for (String f : new String[] {"summary.csv", "summary.md", "shared.json", "config.json", "environment.json"}) {
            assertTrue(Files.isRegularFile(out.resolve(f)), f);
        }
        String summary = Files.readString(out.resolve("summary.md"));
        assertTrue(summary.contains("### esa") && summary.contains("δ-sweep"), summary);
        assertTrue(Files.readString(out.resolve("environment.json")).contains("\"cores\""));

        Path again = dir.resolve("again");
        assertEquals(0, execute(new StringWriter(), "aggregate", "--in", out.toString(), "--out", again.toString()));
        assertEquals(Files.readString(out.resolve("summary.csv")), Files.readString(again.resolve("summary.csv")));
        assertEquals(summary, Files.readString(again.resolve("summary.md")));
    }

    @Test
    void datasetsListShowsDescriptorsAndUnknownSuitesFail() {
        StringWriter out = new StringWriter();
        assertEquals(0, execute(out, "datasets", "list"));
        assertTrue(out.toString().contains("covertype") && out.toString().contains("sha256"), out.toString());
        assertTrue(execute(new StringWriter(), "suite", "nope", "--out", "unused") != 0);
    }
}
