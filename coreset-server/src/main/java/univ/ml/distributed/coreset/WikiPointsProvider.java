package univ.ml.distributed.coreset;

import com.google.common.base.Splitter;
import univ.ml.sparse.SparseWeightableVector;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 */
public class WikiPointsProvider implements CoresetPointsProvider {

    private final String fileName;

    private BufferedReader reader;

    public WikiPointsProvider(final String fileName) {
        this.fileName = fileName;
    }

    @Override
    public List<CoresetWeightedPoint> nextBatchOfCoresetPoints(int batchSize) {
        final List<CoresetWeightedPoint> result = new ArrayList<>();
        String line = null;

        try {
            while ((line = reader.readLine()) != null) {
                final List<String> data = Splitter.on(" ").splitToList(line);
                final Map<Integer, Double> coord = new HashMap<>();
                for (int i = 2; i < data.size(); i += 2) {
                    coord.put(Integer.valueOf(data.get(i)), Double.valueOf(data.get(i + 1)));
                }
                result.add(new CoresetWeightedPoint(new CoresetPoint(coord, Integer.valueOf(data.get(1))), 1d));
            }
        } catch (Exception e) {
            throw new RuntimeException("Cannot read next batch of size " + batchSize + " from " + fileName);
        }
        return result;
    }

    @Override
    public void reset() {
        try {
            final Path path = Paths.get(this.fileName);
            reader = Files.newBufferedReader(path);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }
}
