package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.random.RandomGenerator;

/**
 * The mean coreset of {@code MeanCoreset(m, lloydIterations)} (same calls, same generator use, so the same means and
 * global offset) together with each cell's clustering-feature cost {@code phi_c = sum_{p in c} w ||p - mean_c||^2},
 * so a message can carry {@code (n_c, LS_c, SS_c)} instead of relying on the single global offset.
 */
final class MeanCells {

    private static final int PLAIN_SEEDING = 1; // as MeanCoreset: one k-means++ local trial

    final PointBlock means;
    final double[] phi;
    final double offset;

    private MeanCells(PointBlock means, double[] phi, double offset) {
        this.means = means;
        this.phi = phi;
        this.offset = offset;
    }

    static MeanCells build(PointSet points, int m, int lloydIterations, RandomGenerator rng) {
        if (points.size() <= m) {
            throw new IllegalArgumentException("mean cells need more than m points: " + points.size());
        }
        KMeansResult part = WeightedLloyd.cluster(points, m, lloydIterations, PLAIN_SEEDING, rng, MetricsSink.NOOP);
        int parts = part.centers().size();
        int[] assignment = part.assignment();
        Coreset mc = WeightedLloyd.meansOf(points, assignment, parts);
        PointBlock means = mc.pointSet().compact();
        int[] rowOfPart = new int[parts];
        double[] partWeight = new double[parts];
        for (int g = 0; g < points.size(); g++) {
            double w = points.weight(g);
            if (w > 0.0) {
                partWeight[assignment[g]] += w;
            }
        }
        for (int j = 0, row = 0; j < parts; j++) {
            rowOfPart[j] = partWeight[j] > 0.0 ? row++ : -1;
        }
        double[] phi = new double[means.size()];
        for (int t = 0, g = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                double w = b.weight(i);
                if (w > 0.0) {
                    int c = rowOfPart[assignment[g]];
                    phi[c] += w * b.squaredDistance(i, means, c);
                }
            }
        }
        return new MeanCells(means, phi, mc.offset());
    }

    int cells() {
        return means.size();
    }

    double count(int c) {
        return means.weight(c);
    }
}
