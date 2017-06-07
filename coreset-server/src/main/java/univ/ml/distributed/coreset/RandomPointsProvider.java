package univ.ml.distributed.coreset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomPointsProvider implements CoresetPointsProvider {

    private List<CoresetWeightedPoint> points = new ArrayList<>();

    private int nextPoint = 0;

    public RandomPointsProvider(int N, int d) {
        final Random rnd = new Random(System.nanoTime());
        for (int i = 0; i < N; i++) {
            Map<Integer, Double> coord = new HashMap<>();
            for (int j = 0; j < d; j++) {
                coord.put(j, rnd.nextDouble());
            }
            points.add(new CoresetWeightedPoint(new CoresetPoint(coord, d), 1));
        }
    }

    @Override
    public List<CoresetWeightedPoint> nextBatchOfCoresetPoints(int batchSize) {
        List<CoresetWeightedPoint> result = new ArrayList<>();
        int end = Math.min(nextPoint + batchSize, points.size());
        for (; nextPoint < end; nextPoint++) {
            result.add(points.get(nextPoint));
        }
        return result;
    }

    @Override
    public void reset() {
        nextPoint = 0;
    }
}
