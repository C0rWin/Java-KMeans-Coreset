package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import io.github.c0rwin.coreset.testbed.eval.CostEvaluator;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Candidate center sets computed on the full data once per (dataset, k) and shared by every run: the ESA'22
 * k-means++ seedings and bicriteria convex combinations, plus the best-of-N Lloyd solution on the full data, which is
 * also the downstream reference. Sharing makes algorithm comparisons paired and pays the full-data passes once.
 */
final class CandidatePool {

    static final String FULL_SOLUTION = "bestOf@P";

    final int k;
    final List<Candidate> candidates;
    final List<Centers> centers;
    final double[] costFull;
    final int fullSolutionIndex;
    final double generationMs;
    final double fullSolveMs;
    final long distanceEvals;
    private final Map<Integer, double[][]> prefixCosts = new ConcurrentHashMap<>();

    private CandidatePool(int k, List<Candidate> candidates, List<Centers> centers, double[] costFull,
                          double generationMs, double fullSolveMs, long distanceEvals) {
        this.k = k;
        this.candidates = List.copyOf(candidates);
        this.centers = List.copyOf(centers);
        this.costFull = costFull;
        this.fullSolutionIndex = candidates.size() - 1;
        this.generationMs = generationMs;
        this.fullSolveMs = fullSolveMs;
        this.distanceEvals = distanceEvals;
    }

    static CandidatePool generate(Dataset ds, int k, EvaluationSpec eval, long seed, CostEvaluator evaluator) {
        long t0 = System.nanoTime();
        long evals0 = evaluator.distanceEvaluations();
        SplittableRandom rng = new SplittableRandom(seed);
        int count = eval.candidatesPerGenerator();
        List<Candidate> cands = new ArrayList<>(2 * count + 1);
        cands.addAll(CandidateSolutions.kMeansPlusPlusOnFull(ds.coreset(), k, count, rng));
        cands.addAll(CandidateSolutions.convexCombinationsOfBicriteria(ds.coreset(), k, count, rng));
        long s0 = System.nanoTime();
        KMeansResult full = WeightedLloyd.bestOf(ds.coreset().pointSet(), k, eval.lloydIterations(), eval.fullRestarts(), rng,
                MetricsSink.NOOP);
        double solveMs = (System.nanoTime() - s0) / 1e6;
        cands.add(new Candidate(FULL_SOLUTION, full.centers()));
        List<Centers> cs = new ArrayList<>(cands.size());
        for (Candidate c : cands) {
            cs.add(Centers.of(c.centers()));
        }
        double[] costs = evaluator.costs(ds.coreset(), cs);
        return new CandidatePool(k, cands, cs, costs, (System.nanoTime() - t0) / 1e6, solveMs,
                evaluator.distanceEvaluations() - evals0);
    }

    double fullSolutionCost() {
        return costFull[fullSolutionIndex];
    }

    /**
     * {@code result[l][j]}: cost of candidate {@code j} on the first {@code 2^l} of {@code blockCount} blocks, for
     * every level {@code l} with {@code 2^l <= blockCount}. Cached per block count.
     */
    double[][] prefixCosts(Dataset ds, int blockCount, CostEvaluator evaluator) {
        return prefixCosts.computeIfAbsent(blockCount, b -> {
            int n = ds.n();
            int levels = 32 - Integer.numberOfLeadingZeros(b);
            int[] ends = new int[levels + 1];
            for (int l = 0; l < levels; l++) {
                ends[l] = Builder.blockEnd(n, b, (1 << l) - 1);
            }
            ends[levels] = n;
            double[][] seg = evaluator.segmentCosts(ds.points(), ends, centers);
            double[][] prefix = new double[levels][centers.size()];
            double[] acc = new double[centers.size()];
            for (int l = 0; l < levels; l++) {
                for (int j = 0; j < acc.length; j++) {
                    acc[j] += seg[l][j];
                }
                prefix[l] = acc.clone();
            }
            return prefix;
        });
    }
}
