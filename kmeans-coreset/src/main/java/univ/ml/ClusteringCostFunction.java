package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;

import java.util.List;

public interface ClusteringCostFunction {

    double getCost(final List<WeightedDoublePoint> centers, final List<WeightedDoublePoint> pointSet);

    double getCost(final List<CentroidCluster<WeightedDoublePoint>> clusters);
}
