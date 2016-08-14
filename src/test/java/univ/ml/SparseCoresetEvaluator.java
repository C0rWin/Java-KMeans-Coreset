package univ.ml;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import com.google.common.collect.Iterables;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWSSE;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;
import univ.ml.sparse.algorithm.streaming.StreamingAlgorithm;

public class SparseCoresetEvaluator {

    private SparseWSSE costFunction = new SparseWSSE();

    private SparseWeightedKMeansPlusPlus clusterer;

    public SparseCoresetEvaluator(SparseWeightedKMeansPlusPlus clusterer) {
        this.clusterer = clusterer;
    }

    public double evaluate(final SparseCoresetAlgorithm algorithm, final List<SparseWeightableVector> dataset,
                           final int batchSize) {
        // Safely copy dataset points into the temp list
        final List<SparseWeightableVector> points = new ArrayList<>(dataset.size());
        for (SparseWeightableVector each : dataset) {
            points.add(new SparseWeightableVector(each, each.getWeight()));
        }

        final StreamingAlgorithm streamingAlgorithm = new StreamingAlgorithm(algorithm);
        final Iterable<List<SparseWeightableVector>> partitions = Iterables.partition(points, batchSize);

        for (List<SparseWeightableVector> each : partitions) {
            streamingAlgorithm.addDataset(each);
        }

        final List<SparseWeightableVector> coreset = streamingAlgorithm.getTotalCoreset();

            final List<SparseCentroidCluster> clusters = clusterer.cluster(coreset);

        double energy = 0d;
        for (SparseWeightableVector point : dataset) {
            double d = Double.MAX_VALUE;
            for (SparseCentroidCluster cluster : clusters) {
                final RealVector center = cluster.getCenter().getVector();
                d = FastMath.min(d, FastMath.pow(center.getDistance(point), 2));
            }
            energy += d;
        }

        return energy;
    }
}
