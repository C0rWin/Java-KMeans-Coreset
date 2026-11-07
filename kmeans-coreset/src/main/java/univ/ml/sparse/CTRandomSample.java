package univ.ml.sparse;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class CTRandomSample implements RandomSampleAlgorithm, Serializable {

    private static final int M = 100_000;
    private static final long serialVersionUID = -2531580940763249298L;

    private double[] cumsum;

    private int[] buckets = new int[M];

    public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {

        init(dataset);

        final List<SparseWeightableVector> result = Lists.newArrayListWithExpectedSize(t);

        Map<Integer, Integer> freqMap = Maps.newHashMap();

        final Random rnd = new Random();

        for (int i = 0; i < t; i++) {
            double pr = rnd.nextDouble();
            int index = buckets[Math.min((int)Math.floor(pr * M), M - 1)];
            // Rounding in pr * M can land one bucket late; step back to the first point with cumsum > pr.
            for (; index > 0 && cumsum[index - 1] > pr; --index) {
            }
            for(; index < cumsum.length && cumsum[index] <= pr; ++index) {

            }
            final Integer count = freqMap.get(index);
            if (count == null) {
                freqMap.put(index, 1);
            } else {
                freqMap.put(index, count + 1);
            }
        }

        for (Map.Entry<Integer, Integer> each : freqMap.entrySet()) {
            final SparseWeightableVector vector = dataset.get(each.getKey());
            vector.setProbability(vector.getProbability() * each.getValue());
            vector.setWeight(vector.getWeight() * each.getValue());
            result.add(vector);
        }

        return result;
    }

    private void init(final List<SparseWeightableVector> dataset) {
        final int N = dataset.size();
        cumsum = new double[N];

        cumsum[0] = dataset.get(0).getProbability();
        for (int i = 1; i < N; i++) {
            cumsum[i] = cumsum[i - 1] + dataset.get(i).getProbability();
        }

        for (int i = 0; i < N; i++) {
            cumsum[i] /= cumsum[N-1];
        }

        // Bucket b holds draws in [b/M, (b+1)/M), so it must start at the first point whose CDF exceeds its lower
        // edge b/M; b/M < 1 = cumsum[N-1] keeps s in range.
        int s = 0;
        for (int b = 0; b < M; b++) {
            for (; cumsum[s] <= (1.0 * b) / M; ++s) {
            }
            buckets[b] = s;
        }
    }

}
