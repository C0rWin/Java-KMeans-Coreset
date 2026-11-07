package io.github.c0rwin.coreset.bench;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import univ.ml.distributed.coreset.CoresetPoint;
import univ.ml.distributed.coreset.CoresetPoints;
import univ.ml.distributed.coreset.CoresetWeightedPoint;

/** Converts between coreset-core points and the legacy Thrift messages (map-of-coordinates encoding). */
public final class LegacyConvert {

    private LegacyConvert() {
    }

    public static List<CoresetWeightedPoint> toThrift(List<WeightedPoint> points) {
        List<CoresetWeightedPoint> out = new ArrayList<>(points.size());
        for (WeightedPoint p : points) {
            SparseVector v = p.vector();
            Map<Integer, Double> coords = new HashMap<>(Math.max(4, v.nnz() * 4 / 3 + 1));
            for (int k = 0, n = v.nnz(); k < n; k++) {
                coords.put(v.indexAt(k), v.valueAt(k));
            }
            out.add(new CoresetWeightedPoint(new CoresetPoint(coords, v.dim()), p.weight()));
        }
        return out;
    }

    /** Legacy coreset as a coreset-core {@link Coreset} (offset 0: the legacy algorithms carry no offset). */
    public static Coreset fromThrift(CoresetPoints points) {
        List<WeightedPoint> out = new ArrayList<>(points.getPointsSize());
        for (CoresetWeightedPoint p : points.getPoints()) {
            Map<Integer, Double> coords = p.getPoint().getCoords();
            int[] idx = new int[coords.size()];
            double[] val = new double[coords.size()];
            int w = 0;
            for (Map.Entry<Integer, Double> e : coords.entrySet()) {
                idx[w] = e.getKey();
                val[w] = e.getValue();
                w++;
            }
            out.add(new WeightedPoint(SparseVector.of(p.getPoint().getDim(), idx, val), p.getWeight()));
        }
        return Coreset.of(out);
    }
}
