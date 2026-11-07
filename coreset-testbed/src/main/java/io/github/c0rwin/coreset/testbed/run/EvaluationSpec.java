package io.github.c0rwin.coreset.testbed.run;

/**
 * How runs are evaluated. Every field has a default, so any subset may be given.
 *
 * @param candidatesPerGenerator ESA'22 candidates on the full data: this many k-means++ seedings and this many convex
 *                               combinations of bicriteria centers (shared by every run on a dataset and k)
 * @param lloydOnCoreset         extra candidates: single Lloyd runs on the coreset (what a user deploys)
 * @param lloydIterations        Lloyd iteration cap for candidates and downstream solves
 * @param downstreamRestarts     best-of-N restarts of the downstream solve on the coreset
 * @param fullRestarts           best-of-N restarts of the reference solve on the full data
 * @param confidence             level of the per-candidate confidence intervals (sampling coresets, offline)
 * @param distortionThreshold    verdict: pass when the empirical distortion stays at or below this
 * @param deltaSteps             thresholds in the ESA'22 benchmark delta sweep
 * @param verifyReplay           rebuild each coreset with the same seed and compare digests
 * @param evaluationThreads      threads for full-data cost passes; 0 uses every available processor
 */
public record EvaluationSpec(Integer candidatesPerGenerator, Integer lloydOnCoreset, Integer lloydIterations,
                             Integer downstreamRestarts, Integer fullRestarts, Double confidence,
                             Double distortionThreshold, Integer deltaSteps, Boolean verifyReplay,
                             Integer evaluationThreads) {

    public EvaluationSpec {
        candidatesPerGenerator = positive("candidatesPerGenerator", candidatesPerGenerator, 5);
        lloydOnCoreset = nonNegative("lloydOnCoreset", lloydOnCoreset, 3);
        lloydIterations = nonNegative("lloydIterations", lloydIterations, 50);
        downstreamRestarts = positive("downstreamRestarts", downstreamRestarts, 10);
        fullRestarts = positive("fullRestarts", fullRestarts, 3);
        confidence = confidence == null ? 0.95 : confidence;
        if (!(confidence > 0.0 && confidence < 1.0)) {
            throw new IllegalArgumentException("confidence must be in (0, 1): " + confidence);
        }
        distortionThreshold = distortionThreshold == null ? 1.2 : distortionThreshold;
        if (!(distortionThreshold >= 1.0)) {
            throw new IllegalArgumentException("distortionThreshold must be >= 1: " + distortionThreshold);
        }
        deltaSteps = positive("deltaSteps", deltaSteps, 50);
        verifyReplay = verifyReplay != null && verifyReplay;
        evaluationThreads = nonNegative("evaluationThreads", evaluationThreads, 0);
    }

    public static EvaluationSpec defaults() {
        return new EvaluationSpec(null, null, null, null, null, null, null, null, null, null);
    }

    private static int positive(String name, Integer v, int fallback) {
        int r = v == null ? fallback : v;
        if (r < 1) {
            throw new IllegalArgumentException(name + " must be >= 1: " + r);
        }
        return r;
    }

    private static int nonNegative(String name, Integer v, int fallback) {
        int r = v == null ? fallback : v;
        if (r < 0) {
            throw new IllegalArgumentException(name + " must be >= 0: " + r);
        }
        return r;
    }
}
