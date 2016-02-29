package univ.ml;

import com.google.common.collect.Lists;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.junit.Assert;
import org.junit.Test;

import java.util.List;

public class NonUniformCorestTest {

    @Test
    public void sampleSizeSmallerThanOverAllPointsShouldReturnAllPoints() {
        final int k = 10;
        final int sampleSize = 100;
        final int dimension = 100;
        final int datasetSize = 1_000;

        final NonUniformCoreset<WeightedDoublePoint> alg = new NonUniformCoreset<>(k, sampleSize);
        final RandomDataGenerator rnd = new RandomDataGenerator();

        List<WeightedDoublePoint> points = Lists.newArrayList();

        for (int j = 0; j < datasetSize; j++) {
            double[] coord = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                coord[i] = rnd.nextGaussian(5, 2);
            }
            points.add(new WeightedDoublePoint(coord, 1, ""));
        }

        final List<WeightedDoublePoint> sample = alg.takeSample(points);

        Assert.assertEquals(sampleSize, sample.size());
    }

}
