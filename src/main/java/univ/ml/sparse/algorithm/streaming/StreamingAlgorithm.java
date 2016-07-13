package univ.ml.sparse.algorithm.streaming;

import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

public class StreamingAlgorithm {

    private Map<Integer, List<SparseWeightableVector>> coresetTree = Maps.newHashMap();

    private SparseCoresetAlgorithm coresetAlgorithm;

    public StreamingAlgorithm(SparseCoresetAlgorithm coresetAlgorithm) {
        this.coresetAlgorithm = coresetAlgorithm;
    }

    public void addDataset(final List<SparseWeightableVector> dataset) {
        List<SparseWeightableVector> coreset = coresetAlgorithm.takeSample(dataset);
        int treeLevel = 0;
        List<SparseWeightableVector> leaf = coresetTree.get(treeLevel);

        while (leaf != null) {
            coresetTree.remove(treeLevel++);
            coreset.addAll(leaf);
            coreset = coresetAlgorithm.takeSample(coreset);

            leaf = coresetTree.get(treeLevel);
        }

        coresetTree.put(treeLevel, coreset);
    }

    public List<SparseWeightableVector> getTotalCoreset() {
        if (coresetTree.size() == 1) {
            for (List<SparseWeightableVector> coreset : coresetTree.values()) {
                return coreset;
            }
        }
        final List<SparseWeightableVector> treeCoresetView = Lists.newArrayList();
        for (Map.Entry<Integer, List<SparseWeightableVector>> each : coresetTree.entrySet()) {
            treeCoresetView.addAll(each.getValue());
        }

        return coresetAlgorithm.takeSample(treeCoresetView);
    }

}