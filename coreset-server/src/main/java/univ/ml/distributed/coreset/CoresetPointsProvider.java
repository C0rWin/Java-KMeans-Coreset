package univ.ml.distributed.coreset;

import java.util.List;

/**
 *
 */
public interface CoresetPointsProvider {

    /**
     *
     * @param batchSize
     * @return
     */
    List<CoresetWeightedPoint> nextBatchOfCoresetPoints(final int batchSize);

    /**
     *
     */
    void reset();
}
