package univ.ml.sparse;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class SparseRandomSample implements RandomSampleAlgorithm, Serializable {

    private static final long serialVersionUID = 7797414433832232341L;

    private final Random rnd;

    public SparseRandomSample() {
        this(new Random());
    }

    public SparseRandomSample(final Random rnd) {
        this.rnd = rnd;
    }

    public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {
        final List<SparseWeightableVector> result = Lists.newArrayListWithExpectedSize(t);

        Map<Integer, Integer> freqMap = Maps.newHashMap();

        final double[] cdf = Cdf.of(dataset);

        for (int j = 0; j < t; j++) {
            final int idx = Cdf.draw(cdf, rnd.nextDouble());
            final Integer count = freqMap.get(idx);
            if (count == null) {
                freqMap.put(idx, 1);
            } else {
                freqMap.put(idx, count + 1);
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

}
