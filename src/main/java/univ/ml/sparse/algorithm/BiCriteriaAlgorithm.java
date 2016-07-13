package univ.ml.sparse.algorithm;

import com.google.common.collect.Lists;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.util.FastMath;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

import java.util.Iterator;
import java.util.List;

public class BiCriteriaAlgorithm {


    private static final long serialVersionUID = -8180245061391568851L;

    private JDKRandomGenerator rnd = new JDKRandomGenerator();

    private int k;

    private double percentile;

    private static class PointInfo {

        private int pointIdx;

        private double distance;

        private SparseWeightableVector vector;

        public PointInfo() {
            this(-1, Double.MAX_VALUE, null);
        }

        public PointInfo(int pointIdx, SparseWeightableVector vector) {
            this(pointIdx, Double.MAX_VALUE, vector);
        }

        public PointInfo(int pointIdx, double distance, SparseWeightableVector vector) {
            this.pointIdx = pointIdx;
            this.distance = distance;
            this.vector = vector;
        }

        public int getPointIdx() {
            return pointIdx;
        }

        public void setPointIdx(int pointIdx) {
            this.pointIdx = pointIdx;
        }

        public double getDistance() {
            return distance;
        }

        public void setDistance(double distance) {
            this.distance = distance;
        }

        public SparseWeightableVector getVector() {
            return vector;
        }

        public void setVector(SparseWeightableVector vector) {
            this.vector = vector;
        }
    }

    public BiCriteriaAlgorithm(int k, double percentile) {
        this.k = k;
        this.percentile = percentile;
    }

    public List<SparseCentroidCluster> takeSample(final List<SparseWeightableVector> pointset) {

        final List<SparseCentroidCluster> result = Lists.newArrayList();

        int N = pointset.size();

        // Copy original points
        final List<PointInfo> points = Lists.newArrayList();

        for (int i = 0; i < pointset.size(); i++) {
            final SparseWeightableVector point = pointset.get(i);
            final SparseWeightableVector vector = new SparseWeightableVector(point, point.getWeight());
            points.add(new PointInfo(i, vector));
        }


        // Select first centers of the clusters.
        final int pointIdx = rnd.nextInt(N);
        SparseWeightableVector selectedVector = points.remove(pointIdx).getVector();

        // Sum of squared distances.
        double sumSqDist = 0d;

        // Compute distances of all points to the selected center.
        for (final PointInfo point : points) {
            final SparseWeightableVector vector = point.getVector();
            final double d =  vector.getVector().getDistance(selectedVector) * FastMath.sqrt(vector.getWeight());

            point.setDistance(d);
            sumSqDist += point.getDistance() * point.getDistance();
        }
        result.add(new SparseCentroidCluster(selectedVector));

        while(points.size() > k) {
            for (int i = 0; i < k; i++) {

                double prob = rnd.nextDouble() * sumSqDist;
                double cdf = 0d;

                // Select next point
                for (final Iterator<PointInfo> it = points.iterator(); it.hasNext(); ) {
                    final PointInfo point = it.next();

                    cdf += point.getDistance() * point.getDistance();
                    if (cdf >= prob) {
                        selectedVector = point.getVector();
                        result.add(new SparseCentroidCluster(selectedVector));
                        it.remove();
                        break;
                    }
                }

                // Update distances
                for (PointInfo point : points) {
                    final SparseWeightableVector vector = point.getVector();
                    double d = vector.getDistance(selectedVector) * FastMath.sqrt(vector.getWeight());

                    sumSqDist -= point.getDistance() * point.getDistance();
                    point.setDistance(Math.min(point.getDistance(), d));
                    sumSqDist += point.getDistance() * point.getDistance();
                }
            }

            final Median median = new Median();
            double dist[] = new double[points.size()];
            for (int i = 0; i < points.size(); i++) {
                dist[i] = points.get(i).getDistance();
            }
            final double bound = median.evaluate(dist);

            final Iterator<PointInfo> it = points.iterator();
            while (it.hasNext()) {
                if (it.next().getDistance() <= bound) {
                    it.remove();
                }
            }
        }

        for (PointInfo point : points) {
            result.add(new SparseCentroidCluster(point.getVector()));
        }

        for (SparseWeightableVector vector : pointset) {
            double dist = Double.MAX_VALUE;
            int centerIdx = -1;
            for (int i = 0; i < result.size(); ++i) {
                final SparseCentroidCluster center = result.get(i);
                final double d = FastMath.sqrt(vector.getWeight())*vector.getDistance(center.getCenter().getVector());
                if (d < dist) {
                    dist = d;
                    centerIdx = i;
                }
            }
            result.get(centerIdx).addPoint(vector);
        }

        return result;
    }
}
