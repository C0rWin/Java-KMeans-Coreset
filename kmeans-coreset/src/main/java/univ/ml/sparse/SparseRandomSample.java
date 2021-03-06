package univ.ml.sparse;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

public class SparseRandomSample implements RandomSampleAlgorithm, Serializable {

    private static final long serialVersionUID = 7797414433832232341L;

    public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {
        final List<SparseWeightableVector> result = Lists.newArrayListWithExpectedSize(t);

        Map<Integer, Integer> freqMap = Maps.newHashMap();

        double total = 0;

        for (SparseWeightableVector vector : dataset) {
            total += vector.getProbability();
        }

        final Random rnd = new Random();

//        while (freqMap.size() < t) {
        for (int j = 0; j < t; j++) {
            double pr = rnd.nextDouble();
            double cdf = total * pr;

            int i = 0;
            for (double sum = 0; sum < cdf; i++) {
                sum += dataset.get(i).getProbability();
            }
            final int idx = Math.max(0, i - 1);
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
