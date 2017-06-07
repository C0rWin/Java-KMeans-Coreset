package univ.ml.distributed.coreset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import univ.ml.sparse.SparseWeightableVector;

public class CoresetUtils {

    private final static Logger log = LoggerFactory.getLogger(CoresetUtils.class);

    public static double getEnergy(final List<SparseWeightableVector> centers, final List<SparseWeightableVector> points) {
        double totalEnergy = 0d;

        for (final SparseWeightableVector point : points) {
            double minDist = Double.MAX_VALUE;
            for (SparseWeightableVector center : centers) {
                double dist = FastMath.sqrt(point.getWeight()) * point.getDistance(center.getVector());
                minDist = FastMath.min(minDist, dist);
            }
            totalEnergy += minDist * minDist;
        }

        return totalEnergy;

    }

    public static double getEnergy(final CoresetPoints centers, final CoresetPoints points) {
        return getEnergy(toSparseWeightableVectors(centers),
                toSparseWeightableVectors(points));
    }

    public static List<SparseWeightableVector> toSparseWeightableVectors(final List<CoresetWeightedPoint> points) {
        return points
                .stream()
                .map(x -> new SparseWeightableVector(x.getPoint().getCoords(), x.getWeight(), x.getPoint().getDim()))
                .collect(Collectors.toList());
    }


    public static List<SparseWeightableVector> toSparseWeightableVectors(final CoresetPoints points) {
        return toSparseWeightableVectors(points.getPoints());
    }

    public static CoresetPoints toCoresetPoints(final List<SparseWeightableVector> points) {
        List<CoresetWeightedPoint> results = new ArrayList<>();

        for (final SparseWeightableVector each : points) {
            Map<Integer, Double> coords = new HashMap<>();

            for (int i = 0; i < each.getDimension(); i++) {
                final double val = each.getEntry(i);
                if (val == 0)
                    continue;

                coords.put(i, val);
            }
            final CoresetPoint point = new CoresetPoint(coords, each.getDimension());
            results.add(new CoresetWeightedPoint(point, each.getWeight()));
        }

        return new CoresetPoints(0, results);
    }
}
