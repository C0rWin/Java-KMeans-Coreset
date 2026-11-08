package io.github.c0rwin.coreset.testbed.faults.framework;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.testbed.cli.TestbedCommand;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

class FrameworkCommandTest {

    @Test
    void quickScaleRunsEveryExperimentReproducibly(@TempDir Path dir) throws Exception {
        for (String run : new String[] {"a", "b"}) {
            CommandLine cli = new CommandLine(new TestbedCommand());
            cli.setErr(new PrintWriter(new StringWriter(), true));
            assertEquals(0, cli.execute("framework", "--out", dir.resolve(run).toString(), "--raw-dir",
                    dir.resolve(run + "-raw").toString(), "--scale", "quick", "--threads", run.equals("a") ? "1" : "3"));
        }
        String verdicts = Files.readString(dir.resolve("a/verdicts.csv"), StandardCharsets.UTF_8);
        for (String id : new String[] {"A1 HT floor", "A2 header copies", "A3 header in every packet",
            "B1 pair plateau", "B2 RC limit", "B3 per-class removes bias", "C1 Markov bound", "C2 sorted + block",
            "D1 forwarding accuracy", "D2 resampling accuracy", "E k-median scaling"}) {
            assertTrue(verdicts.contains(id), id);
        }
        // Per-index seed streams: the thread count changes no result, raw files included.
        for (String f : new String[] {"verdicts.csv", "summary.md", "C2-conditional-law.csv", "D2-tree.csv",
            "raw-SHA256SUMS"}) {
            assertEquals(Files.readString(dir.resolve("a").resolve(f), StandardCharsets.UTF_8),
                    Files.readString(dir.resolve("b").resolve(f), StandardCharsets.UTF_8), f);
        }
        String sums = Files.readString(dir.resolve("a/raw-SHA256SUMS"), StandardCharsets.UTF_8);
        assertTrue(sums.contains("C1-delivered-histogram.csv.gz") && sums.contains("E-kmedian.csv.gz"), sums);
        assertEquals(Lab.sha256(dir.resolve("a-raw/E-kmedian.csv.gz")),
                sums.lines().filter(l -> l.endsWith("E-kmedian.csv.gz")).findFirst().orElseThrow().split(" ")[0]);
    }
}
