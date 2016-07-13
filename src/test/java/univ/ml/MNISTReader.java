package univ.ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import univ.ml.sparse.SparseWeightableVector;

public class MNISTReader {

    private String fileName;

    public MNISTReader(final String fileName) {
        this.fileName = fileName;
    }

    public List<SparseWeightableVector> readPoints(final int num) throws IOException {
        final List<SparseWeightableVector> pointSet = Lists.newArrayList();

        try (final BufferedReader reader = new BufferedReader(new FileReader(new File(fileName)))) {
            String line = reader.readLine();
            for (int j = 0; j < num; ++j, line = reader.readLine()) {
                List<String> coordinates = Splitter.on(',').splitToList(line);
                Map<Integer, Double> _coords = Maps.newHashMap();
                for (int i = 1; i < coordinates.size(); i++) {
                    _coords.put(i - 1, Double.valueOf(coordinates.get(i)));
                }
                pointSet.add(new SparseWeightableVector(_coords, 1, coordinates.size()));
            }
        }
        return pointSet;
    }
}
