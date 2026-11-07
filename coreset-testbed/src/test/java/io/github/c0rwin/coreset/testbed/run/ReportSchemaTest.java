package io.github.c0rwin.coreset.testbed.run;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import io.github.c0rwin.coreset.testbed.report.Aggregator;
import io.github.c0rwin.coreset.testbed.report.Json;
import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ReportSchemaTest {

    private static List<ValidityReport> reports;

    @BeforeAll
    static void run() throws Exception {
        reports = TinyRuns.run(TinyRuns.config(1)).reports;
    }

    private static Set<String> keys(JsonNode node) {
        Set<String> out = new TreeSet<>();
        for (Iterator<String> it = node.fieldNames(); it.hasNext(); ) {
            out.add(it.next());
        }
        return out;
    }

    @Test
    void everyReportRoundTripsThroughJson() throws Exception {
        for (ValidityReport r : reports) {
            assertEquals(r, Json.parseReport(Json.toJson(r)));
            assertEquals(r, Json.parseReport(Json.toJsonLine(r)));
        }
    }

    @Test
    void nonFiniteNumbersSurviveTheRoundTrip() throws Exception {
        ValidityReport r = reports.get(0);
        ValidityReport.DistortionInfo d = r.distortion();
        ValidityReport.Candidate inf = new ValidityReport.Candidate("degenerate", 0.0, 1.0,
                Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, null);
        List<ValidityReport.Candidate> cands = new ArrayList<>(d.candidates());
        cands.add(inf);
        ValidityReport withInf = new ValidityReport(r.schemaVersion(), r.run(), r.config(), r.dataset(),
                r.construction(), r.integrity(), new ValidityReport.DistortionInfo(d.isLowerBound(), cands,
                Double.POSITIVE_INFINITY, Double.NaN, "degenerate", d.ciCoverage(), d.benchmarkDeltaSweep()),
                r.downstream(), r.validation(), r.verdict());
        String json = Json.toJson(withInf);
        assertTrue(json.contains("\"Infinity\""), json);
        assertEquals(withInf, Json.parseReport(json));
    }

    @Test
    void layoutFollowsTheSchemaVersion1Sections() throws Exception {
        JsonNode root = Json.mapper().readTree(Json.toJson(reports.get(0)));
        assertEquals(1, root.get("schemaVersion").asInt());
        assertEquals(new TreeSet<>(List.of("schemaVersion", "run", "config", "dataset", "construction", "integrity",
                "distortion", "downstream", "validation", "verdict")), keys(root));
        assertTrue(keys(root.get("run")).containsAll(List.of("id", "startedAt", "gitCommit", "gitDirty", "jvm",
                "host", "engine")));
        assertTrue(keys(root.get("config")).containsAll(List.of("problem", "z", "k", "algorithm", "targetSize",
                "epsilonTarget", "delta", "seed", "replicate", "mode", "blockCount", "partitions")));
        assertTrue(keys(root.get("dataset")).containsAll(List.of("id", "sha256", "n", "d", "nnz", "weightSum",
                "preprocessing", "license")));
        assertTrue(keys(root.get("construction")).containsAll(List.of("outputSize", "levels", "reduceOps",
                "mergeOps", "additiveOffset", "wallMs", "cpuMs", "allocBytes", "peakHeapBytes", "gcCount", "gcMs",
                "distanceEvals", "pointsRead", "bytesSerialized", "bytesOnWire", "perLevel")));
        assertEquals(new TreeSet<>(List.of("weightRelErr", "firstMomentRelErr", "secondMomentRelErr",
                "negativeWeights", "nonFinite", "oneSidedViolations", "replayHash", "replayDeterministic")),
                keys(root.get("integrity")));
        assertTrue(keys(root.get("distortion")).containsAll(List.of("isLowerBound", "candidates", "max", "mean",
                "argmaxGenerator", "benchmarkDeltaSweep")));
        assertEquals(new TreeSet<>(List.of("generator", "costFull", "costCoreset", "ratio", "distortion", "ci95")),
                keys(root.get("distortion").get("candidates").get(0)));
        assertTrue(keys(root.get("downstream")).containsAll(List.of("solver", "costFullOfCoresetSolution",
                "costFullOfFullSolution", "relativeCost", "solveOnCoresetMs", "solveOnFullMs")));
        assertEquals(new TreeSet<>(List.of("distortionMax", "threshold", "pass", "notes")),
                keys(root.get("verdict")));
    }

    @Test
    void compressedReportFileReadsBackAndAggregatesTheEsa22Way(@TempDir Path dir) throws Exception {
        Path file = dir.resolve(OutputDirectory.REPORTS);
        try (Writer w = Json.jsonLinesGzip(file)) {
            for (ValidityReport r : reports) {
                w.write(Json.toJsonLine(r));
                w.write('\n');
            }
        }
        List<ValidityReport> back = Json.readJsonLinesGzip(file);
        assertEquals(reports, back);

        List<ValidityReport> group = back.stream().filter(r -> r.dataset().id().equals("gauss")
                && r.config().algorithm().label().equals("uniform") && r.config().k() == 3
                && r.config().modeLabel().equals("offline")).toList();
        assertEquals(1, group.size());
        // Two synthetic replicates of the same grid point: mean of maxima and max of maxima.
        ValidityReport a = group.get(0);
        ValidityReport b = withMax(a, a.distortion().max() + 1.0);
        Aggregator.Row row = Aggregator.aggregate(List.of(a, b)).get(0);
        assertEquals(2, row.runs());
        assertEquals(a.distortion().max() + 0.5, row.distortionMeanOfMax(), 1e-12);
        assertEquals(a.distortion().max() + 1.0, row.distortionMax(), 1e-12);
        assertEquals(Math.sqrt(0.5), row.distortionSd(), 1e-12);
        assertTrue(Aggregator.csv(List.of(row)).startsWith(String.join(",", Aggregator.CSV_HEADER) + "\n"));
        assertTrue(Aggregator.markdown(List.of(row)).contains("### gauss"));
    }

    private static ValidityReport withMax(ValidityReport r, double max) {
        ValidityReport.DistortionInfo d = r.distortion();
        return new ValidityReport(r.schemaVersion(), r.run(), r.config(), r.dataset(), r.construction(),
                r.integrity(), new ValidityReport.DistortionInfo(d.isLowerBound(), d.candidates(), max, d.mean(),
                d.argmaxGenerator(), d.ciCoverage(), d.benchmarkDeltaSweep()), r.downstream(), r.validation(),
                r.verdict());
    }

    @Test
    void builtInSuitesParseAndUnknownKeysAreRejected() throws Exception {
        for (String suite : List.of("smoke", "standard")) {
            TestbedConfig cfg = io.github.c0rwin.coreset.testbed.cli.TestbedCommand.builtin(suite);
            assertEquals(suite, cfg.name());
            assertEquals(cfg, Json.parseConfig(Json.toJson(cfg), false));
        }
        Exception unknown = assertThrows(Exception.class, () -> Json.parseConfig("""
                datasets: [{id: a, type: esa22, params: {k: 3, alpha: 3}}]
                ks: [3]
                sizes: [10]
                algorithms: [{id: uniform}]
                replicate: 2
                """, true));
        assertTrue(unknown.getMessage().contains("\"replicate\""), unknown.getMessage());
        assertTrue(Files.notExists(Path.of("unused-cache")), "nothing may be downloaded or cached by tests");
    }
}
