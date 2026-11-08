package io.github.c0rwin.coreset.testbed.faults;

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

class FaultsCommandTest {

    @Test
    void quickScaleRunsEveryExperimentReproducibly(@TempDir Path dir) throws Exception {
        for (String out : new String[] {"a", "b"}) {
            CommandLine cli = new CommandLine(new TestbedCommand());
            cli.setErr(new PrintWriter(new StringWriter(), true));
            assertEquals(0, cli.execute("faults", "--out", dir.resolve(out).toString(), "--scale", "quick",
                    "--threads", out.equals("a") ? "1" : "3"));
        }
        for (String e : FaultLab.EXPERIMENTS) {
            assertTrue(Files.size(dir.resolve("a").resolve(e + ".jsonl.gz")) > 0, e);
            assertTrue(Files.isRegularFile(dir.resolve("a").resolve(e + "-cells.csv")), e);
        }
        String verdicts = Files.readString(dir.resolve("a").resolve("verdicts.csv"), StandardCharsets.UTF_8);
        assertTrue(verdicts.contains("F3/B1 slope") && verdicts.contains("F5/E1 random order"), verdicts);
        // Seeded streams and index-ordered parallel results: the thread count changes no result (F5 cells carry
        // wall time, so F5 is compared through its verdicts).
        assertEquals(verdicts, Files.readString(dir.resolve("b").resolve("verdicts.csv"), StandardCharsets.UTF_8));
        for (String e : new String[] {"F1", "F2", "F3", "F4", "F6"}) {
            assertEquals(Files.readString(dir.resolve("a").resolve(e + "-cells.csv"), StandardCharsets.UTF_8),
                    Files.readString(dir.resolve("b").resolve(e + "-cells.csv"), StandardCharsets.UTF_8), e);
        }
    }
}
