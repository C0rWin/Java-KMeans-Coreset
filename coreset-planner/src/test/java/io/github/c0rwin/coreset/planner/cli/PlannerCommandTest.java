package io.github.c0rwin.coreset.planner.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.testbed.data.PointFileWriter;
import io.github.c0rwin.coreset.testbed.data.SyntheticData;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

class PlannerCommandTest {

    private static Path writeMixture(Path dir) throws Exception {
        Path f = dir.resolve("mix.cbin");
        try (PointFileWriter w = PointFileWriter.dense(f, 4, 0)) {
            for (WeightedPoint p : SyntheticData.gaussianMixture(6_000, 4, 3, 1.0, 0.3, 10.0, 5L)) {
                w.writeRow(p.vector().toDense());
            }
            w.finish();
        }
        return f;
    }

    private static String run(String... args) {
        StringWriter out = new StringWriter();
        CommandLine cl = new CommandLine(new PlannerCommand());
        cl.setOut(new PrintWriter(out));
        cl.setErr(new PrintWriter(out));
        int code = cl.execute(args);
        assertEquals(0, code, out::toString);
        return out.toString();
    }

    @Test
    void planPrintsTheConfigurationAndWritesJson(@TempDir Path dir) throws Exception {
        Path data = writeMixture(dir);
        Path json = dir.resolve("plan.json");
        String out = run("plan", "--dataset", data.toString(), "--k", "3", "--pilot", "2000", "--rho", "0.1",
                "--bursty", "3", "--nodes", "4", "--fail", "0.05", "--eps", "0.1", "--mode", "calibrated",
                "--json", json.toString());
        assertTrue(out.contains("plan (feasible)"), out);
        JsonNode doc = new ObjectMapper().readTree(Files.readString(json));
        assertTrue(doc.at("/plan/feasible").asBoolean());
        assertTrue(doc.at("/plan/satisfies").toString().contains("CALIBRATED"));
        assertEquals(6_000, doc.at("/workload/n").asLong());
        assertTrue(doc.at("/plan/model").asText().contains("fault-lab-2026-11-07-v1"));
    }

    @Test
    void validateReportsPredictedAndMeasured(@TempDir Path dir) throws Exception {
        Path data = writeMixture(dir);
        String out = run("validate", "--dataset", data.toString(), "--k", "3", "--pilot", "2000", "--rho", "0.2",
                "--nodes", "4", "--eps", "0.1", "--mode", "calibrated", "--families", "sensitivity,uniform",
                "--trials", "5");
        assertTrue(out.contains("measured  (1-delta) quantile"), out);
        assertTrue(out.contains("\"measured\""), out);
    }
}
