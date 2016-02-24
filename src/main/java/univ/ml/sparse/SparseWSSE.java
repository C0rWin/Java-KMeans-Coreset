package univ.ml.sparse;

import org.apache.commons.math3.util.FastMath;

import java.util.List;
import java.util.stream.Collectors;

public class SparseWSSE {

    public double getCost(final List<SparseCentroidCluster> clusters) {
        return clusters.stream()
                .map(cluster ->
                        cluster.getPoints().stream()
                                .map(point -> {
                                            double d = point.getVector().mapMultiply(-1.0).add(cluster.getCenter().getVector()).getNorm();
                                            return FastMath.pow(d, 2);
                                        }
                                ).collect(Collectors.summingDouble(x -> (double)x))
                ).collect(Collectors.summingDouble(x -> x));
    }

    public double getCost(List<SparseWeightableVector> centers, List<SparseWeightableVector> pointSet) {
        return pointSet.stream()
                .map(point ->
                        centers.stream().map(center -> {
                            final double d = center.getVector().mapMultiply(-1.0).add(point.getVector()).getNorm();
                            return d * d;
                        }).min(Double::compare).get())
                .collect(Collectors.summingDouble(x -> x));
    }

}
