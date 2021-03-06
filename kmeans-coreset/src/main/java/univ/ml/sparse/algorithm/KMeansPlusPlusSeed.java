package univ.ml.sparse.algorithm;

import java.io.Serializable;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.util.FastMath;

import com.google.common.collect.Lists;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

public class KMeansPlusPlusSeed implements SparseSeedingAlgorithm, Serializable {

    private static final long serialVersionUID = 5068491411579061009L;

    private final int k;

    private JDKRandomGenerator rnd = new JDKRandomGenerator();

    public KMeansPlusPlusSeed(int k) {
        this.k = k;
//        System.out.println("#####: SEEDING initialized w/ k == " + k);
    }

    @Override
    public List<SparseCentroidCluster> seed(final List<SparseWeightableVector> vectors) {
        List<SparseCentroidCluster> result = Lists.newArrayList();
        final BitSet selected = new BitSet();

        final int N = vectors.size();
        double[] sqDist = new double[N];

        int selectedIdx = rnd.nextInt(N);
        SparseWeightableVector selectedCenter = vectors.get(selectedIdx);
        result.add(new SparseCentroidCluster(selectedCenter));
        selected.set(selectedIdx);

        double sumSqDist = 0d;

        for (int i = 0; i < N; i++) {
            final SparseWeightableVector p = vectors.get(i);
            final double d = p.getVector().getDistance(selectedCenter);
            sqDist[i] = p.getWeight() * d * d;
            sumSqDist += sqDist[i];
        }

        while (result.size() < k) {
            selectedIdx = selectNext(sqDist, sumSqDist);

            if (selected.get(selectedIdx)) {
                continue;
            }

            // Mark point index as already selected to avoid repetitions
            selected.set(selectedIdx);
            selectedCenter = vectors.get(selectedIdx);
            result.add(new SparseCentroidCluster(selectedCenter));

            // Updated distances.
            for (int i = 0; i < N; i++) {
                final SparseWeightableVector p = vectors.get(i);
                final double d = p.getVector().getDistance(selectedCenter);
                sumSqDist -= sqDist[i];
                sqDist[i] = FastMath.min(sqDist[i], d * d * p.getWeight());
                sumSqDist += sqDist[i];
            }
        }

        // Once we here, the only thing remained to complete is to partition dataset into clusters
        // induced by centers selected.
        for (int i = 0; i < N; i++) {
            double minDist = Double.MAX_VALUE;
            int clusterIdx = -1;
            final SparseWeightableVector p = vectors.get(i);
            for (int j = 0; j < result.size(); j++) {
                final SparseCentroidCluster c = result.get(j);
                double d = c.getCenter().getVector().getDistance(p);
                d = d * FastMath.sqrt(p.getWeight());
                if (d < minDist) {
                    minDist = d;
                    clusterIdx = j;
                }
            }
            result.get(clusterIdx).addPoint(p);
        }

        return result;
    }

    private int selectNext(double[] sqDist, double sumSqDist) {
        if (sqDist == null || sqDist.length == 0) {
            throw new IllegalArgumentException("Distances array could not be of zero length or null.");
        }

        final double X = rnd.nextDouble() * sumSqDist;
        double cdf = 0d;
        for (int i = 0; i < sqDist.length; i++) {
            cdf += sqDist[i];
            if (cdf >= X) {
                return i;
            }
        }
        System.out.println("XXXXXX [SIMON SAYS]: " + Arrays.toString(sqDist) + ", total sum is: " + sumSqDist);
        return sqDist.length - 1;
    }
}
