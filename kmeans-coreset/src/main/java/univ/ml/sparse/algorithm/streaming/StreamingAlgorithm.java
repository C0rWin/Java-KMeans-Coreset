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
        // takeSample may return its input as is, so never merge into or store the list it hands back.
        List<SparseWeightableVector> coreset = Lists.newArrayList(coresetAlgorithm.takeSample(dataset));
        int treeLevel = 0;
        List<SparseWeightableVector> leaf = coresetTree.get(treeLevel);

        while (leaf != null) {
            coresetTree.remove(treeLevel++);
            coreset.addAll(leaf);
            coreset = Lists.newArrayList(coresetAlgorithm.takeSample(coreset));

            leaf = coresetTree.get(treeLevel);
        }

        coresetTree.put(treeLevel, coreset);
    }

    /**
     * Union of the coresets stored on every tree level. It is not compressed again: each extra
     * reduce step would add another level of approximation error on top of the tree's.
     */
    public List<SparseWeightableVector> getTotalCoreset() {
        final List<SparseWeightableVector> treeCoresetView = Lists.newArrayList();
        for (Map.Entry<Integer, List<SparseWeightableVector>> each : coresetTree.entrySet()) {
            treeCoresetView.addAll(each.getValue());
        }

        return treeCoresetView;
    }

}
