package io.github.c0rwin.coreset.experiments.quality;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.QualityConfig;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import io.github.c0rwin.coreset.experiments.data.PointSources;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class QualityEvaluatorTest {

    @TempDir
    Path dir;

    private PointSource source() throws IOException {
        return PointSources.open(new DatasetConfig("gm", "gaussian-mixture",
                Map.of("n", "3000", "d", "5", "clusters", "4", "sigma", "2", "imbalance", "1", "seed", "4")), dir);
    }

    @Test
    void streamingCostMatchesInMemoryCost() throws IOException {
        PointSource src = source();
        Coreset full = InMemory.load(src, Long.MAX_VALUE, 1);
        List<SparseVector> q = CandidateSolutions.kMeansPlusPlusOnFull(full, 4, 1, new SplittableRandom(2)).get(0)
                .centers();
        FullCost.Result r = FullCost.evaluate(src, List.of(q, q.subList(0, 2)), 3);
        assertEquals(Cost.cost(full, q), r.costs()[0], 1e-9 * r.costs()[0]);
        assertEquals(Cost.cost(full, q.subList(0, 2)), r.costs()[1], 1e-9 * r.costs()[1]);
        assertEquals(3000.0, r.weight());
    }

    @Test
    void sampleKeepsExactlyMaxPointsAndTheTotalWeight() throws IOException {
        Coreset sample = InMemory.load(source(), 700, 3);
        assertEquals(700, sample.size());
        assertEquals(3000.0, sample.totalWeight(), 1e-9);
    }

    @Test
    void distortionAggregationEqualsValidateDistortion() throws IOException {
        Coreset full = InMemory.load(source(), Long.MAX_VALUE, 1);
        Coreset c = CoresetAlgorithms.load().create("uniform", Map.of("m", "150"))
                .compress(full, new SplittableRandom(5));
        List<Candidate> cands = CandidateSolutions.standard(full, c, 4, 5, 2, new SplittableRandom(6));
        Distortion.Report expected = Distortion.evaluate(full, c, cands);
        List<Distortion.Entry> entries = new ArrayList<>();
        for (Candidate q : cands) {
            double cp = Cost.cost(full, q.centers());
            double cc = Cost.cost(c, q.centers());
            double ratio = cc / cp;
            entries.add(new Distortion.Entry(q.generator(), cp, cc, ratio, Math.max(ratio, 1 / ratio)));
        }
        Distortion.Report actual = QualityEvaluator.report(entries);
        assertEquals(expected.max(), actual.max(), 1e-12);
        assertEquals(expected.mean(), actual.mean(), 1e-12);
        assertEquals(expected.argmaxGenerator(), actual.argmaxGenerator());
    }

    @Test
    void referenceIsCachedAndRunsAreScored() throws IOException {
        PointSource src = source();
        QualityConfig q = new QualityConfig(2, 2, 20, 1, 5, null, 2, true);
        QualityEvaluator a = new QualityEvaluator(src, 4, q, dir, 7);
        try (var files = Files.list(dir.resolve("reference"))) {
            assertEquals(1, files.count());
        }
        QualityEvaluator b = new QualityEvaluator(src, 4, q, dir, 7);
        assertEquals(a.reference().cost(), b.reference().cost());
        Map<String, Object> m = a.evaluate(null, a.reference().centers());
        assertEquals(1.0, (Double) m.get("cost_ratio"), 1e-12);
        Coreset full = InMemory.load(src, Long.MAX_VALUE, 1);
        Map<String, Object> withCoreset = a.evaluate(full, a.reference().centers());
        assertEquals(1.0, (Double) withCoreset.get("distortion_max"), 1e-9, "the full data is its own coreset");
        assertTrue((Double) withCoreset.get("weight_error") < 1e-12);
    }
}
