package univ.ml.sparse;

import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class CTRandomSample implements RandomSampleAlgorithm {

    private static final int M = 100_000;

    private double[] cumsum;

    private int[] buckets = new int[M];

    private List<SparseWeightableVector> dataset;

    public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {

        init(dataset);

        final List<SparseWeightableVector> result = Lists.newArrayListWithExpectedSize(t);

        Map<Integer, Integer> freqMap = Maps.newHashMap();

        final Random rnd = new Random();

        for (int i = 0; i < t; i++) {
            double pr = rnd.nextDouble();
            int index = buckets[Math.min((int)Math.floor(pr * M), M - 1)];
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
        this.dataset = dataset;
        final int N = dataset.size();
        cumsum = new double[N];

        cumsum[0] = dataset.get(0).getProbability();
        for (int i = 1; i < N; i++) {
            cumsum[i] = cumsum[i - 1] + dataset.get(i).getProbability();
        }

        for (int i = 0; i < N; i++) {
            cumsum[i] /= cumsum[N-1];
        }

        int s = 0;
        for (int i = 1; i <=M ; i++) {
            for(;s < cumsum.length && cumsum[s] <= (1.0*i)/M; ++s) {
            }
            buckets[i - 1] = s < cumsum.length ? s : cumsum.length - 1;
        }
    }

}
