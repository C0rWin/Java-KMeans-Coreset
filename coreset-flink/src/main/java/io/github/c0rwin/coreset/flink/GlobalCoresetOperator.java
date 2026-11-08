package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/**
 * Merge stage of {@link FlinkCoresets#stream}, run with parallelism 1. It keeps one {@link MergeReduceTree} per local
 * subtask and inserts that subtask's partials in sequence order, which Flink's per-channel FIFO delivery guarantees.
 * The trees therefore never depend on how partials of different subtasks interleave, and the state after the whole
 * input is a pure function of the per-subtask inputs: the final snapshot is bit-identical across failures, restarts
 * and runs. A single tree fed in arrival order would not be.
 *
 * <p>Tree {@code s} merges with {@code derive(seed, -2 - s, level, partial sequence)}; the compressed snapshot uses
 * {@code derive(seed, -1, FINAL_LEVEL, partials merged)}. A partial whose sequence number is not the next expected one
 * fails the job: under exactly-once delivery that can only mean lost or duplicated state.
 */
final class GlobalCoresetOperator extends AbstractStreamOperator<CoresetSnapshot>
        implements OneInputStreamOperator<PartialCoreset, CoresetSnapshot>, BoundedOneInput {

    private static final long serialVersionUID = 1L;
    static final String STATE_NAME = "coreset-global-trees";
    static final long SNAPSHOT_STREAM = -1;

    private final CoresetConfig config;
    private transient CoresetAlgorithm algorithm;
    private transient ListState<OperatorStates.Merge> state;
    private transient StreamRecord<CoresetSnapshot> reuse;
    private transient int parallelism;
    private transient MergeReduceTree[] trees;
    private transient LevelShadow[] shadows;
    private transient long[] nextSequence;
    private transient long[] records;
    private transient double[] weight;
    private transient long partials;
    private transient long snapshots;
    private transient long sinceSnapshot;
    private transient long partialsAtLastSnapshot;
    private transient long peakStateBytes;
    private transient long lastStateBytes;
    private transient long lastSnapshotPoints;

    GlobalCoresetOperator(CoresetConfig config) {
        this.config = config;
    }

    static long treeStream(int subtask) {
        return -2L - subtask;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        algorithm = config.algorithm();
        trees = new MergeReduceTree[0];
        shadows = new LevelShadow[0];
        nextSequence = new long[0];
        records = new long[0];
        weight = new double[0];
        state = context.getOperatorStateStore().getListState(
                new ListStateDescriptor<>(STATE_NAME, OperatorStates.MergeSerializer.INSTANCE));
        if (!context.isRestored()) {
            return;
        }
        List<OperatorStates.Merge> restored = new ArrayList<>();
        state.get().forEach(restored::add);
        if (restored.isEmpty()) {
            return;
        }
        if (restored.size() > 1) {
            throw new IllegalStateException("merge stage restored " + restored.size() + " states; it must run with"
                    + " parallelism 1");
        }
        OperatorStates.Merge s = restored.get(0);
        if (s.fingerprint() != config.fingerprint()) {
            throw new IllegalStateException("merge-stage state was written with a different algorithm, parameters,"
                    + " seed or chunk size than " + config);
        }
        if (s.parallelism() > 0) {
            init(s.parallelism());
        }
        for (int i = 0; i < s.trees().size(); i++) {
            OperatorStates.SubtaskTree t = s.trees().get(i);
            trees[i] = MergeReduceTree.restore(algorithm, MergeRandomness.derived(config.seed(), treeStream(i)),
                    MetricsSink.NOOP, t.levels(), t.batches());
            shadows[i].restore(t.batches(), t.levels(), t.depths());
            nextSequence[i] = t.nextSequence();
            records[i] = t.records();
            weight[i] = t.weight();
        }
        partials = s.partials();
        snapshots = s.snapshots();
        sinceSnapshot = s.sinceSnapshot();
        partialsAtLastSnapshot = partials;
        peakStateBytes = s.peakStateBytes();
    }

    private void init(int p) {
        parallelism = p;
        trees = new MergeReduceTree[p];
        shadows = new LevelShadow[p];
        nextSequence = new long[p];
        records = new long[p];
        weight = new double[p];
        for (int i = 0; i < p; i++) {
            trees[i] = new MergeReduceTree(algorithm, MergeRandomness.derived(config.seed(), treeStream(i)),
                    MetricsSink.NOOP);
            shadows[i] = new LevelShadow();
        }
    }

    @Override
    public void open() throws Exception {
        super.open();
        reuse = new StreamRecord<>(null);
        MetricGroup g = getMetricGroup().addGroup("coreset");
        g.gauge("recordsIn", this::totalRecords);
        g.gauge("partialsMerged", () -> partials);
        g.gauge("snapshotsEmitted", () -> snapshots);
        g.gauge("snapshotPoints", () -> lastSnapshotPoints);
        g.gauge("treeLevels", this::levels);
        g.gauge("treeDepth", this::depth);
        g.gauge("stateBytes", () -> lastStateBytes);
        g.gauge("peakStateBytes", () -> peakStateBytes);
        long interval = config.snapshotIntervalMillis();
        if (interval > 0) {
            long now = getProcessingTimeService().getCurrentProcessingTime();
            getProcessingTimeService().registerTimer(now + interval, this::onProcessingTime);
        }
    }

    @Override
    public void processElement(StreamRecord<PartialCoreset> element) {
        PartialCoreset p = element.getValue();
        if (parallelism == 0) {
            init(p.parallelism());
        } else if (p.parallelism() != parallelism) {
            throw new IllegalStateException("local stage parallelism changed from " + parallelism + " to "
                    + p.parallelism() + "; rescaling the coreset pipeline is not supported");
        }
        int s = p.subtask();
        if (s < 0 || s >= parallelism) {
            throw new IllegalStateException("partial from subtask " + s + " of a local stage with parallelism "
                    + parallelism);
        }
        if (p.sequence() != nextSequence[s]) {
            throw new IllegalStateException("subtask " + s + " sent partial " + p.sequence() + ", expected "
                    + nextSequence[s] + ": partials were lost or duplicated");
        }
        MergeReduceTree tree = trees[s];
        tree.add(p.coreset());
        Coreset carry = tree.levelsView().get(0);
        shadows[s].onAdd(tree.batchesAdded(), p.depth(), PointCodec.coresetLength(carry));
        nextSequence[s]++;
        records[s] += p.records();
        weight[s] += p.weight();
        partials++;
        peakStateBytes = Math.max(peakStateBytes, stateBytes());
        if (config.snapshotEveryPartials() > 0 && ++sinceSnapshot >= config.snapshotEveryPartials()) {
            sinceSnapshot = 0;
            emitSnapshot(false);
        }
    }

    private void onProcessingTime(long time) {
        if (partials != partialsAtLastSnapshot) {
            emitSnapshot(false);
        }
        getProcessingTimeService().registerTimer(time + config.snapshotIntervalMillis(), this::onProcessingTime);
    }

    @Override
    public void endInput() {
        emitSnapshot(true);
    }

    private void emitSnapshot(boolean last) {
        int size = 0;
        double offset = 0.0;
        for (MergeReduceTree t : trees) {
            for (Coreset c : t.levelsView()) {
                size += c.size();
                offset += c.offset();
            }
        }
        List<WeightedPoint> points = new ArrayList<>(size);
        for (MergeReduceTree t : trees) {
            for (Coreset c : t.levelsView()) {
                points.addAll(c.points());
            }
        }
        Coreset union = new Coreset(points, offset);
        Coreset compressed = null;
        if (config.compressSnapshots() && (!union.isEmpty() || union.offset() != 0.0)) {
            compressed = algorithm.compress(union, new SplittableRandom(MergeRandomness.derive(config.seed(),
                    SNAPSHOT_STREAM, MergeRandomness.FINAL_LEVEL, partials)), MetricsSink.NOOP);
        }
        snapshots++;
        partialsAtLastSnapshot = partials;
        lastSnapshotPoints = union.size();
        double totalWeight = 0.0;
        for (double w : weight) {
            totalWeight += w;
        }
        output.collect(reuse.replace(new CoresetSnapshot(union, compressed, totalRecords(), totalWeight, partials,
                levels(), depth(), last)));
    }

    private long totalRecords() {
        long n = 0;
        for (long r : records) {
            n += r;
        }
        return n;
    }

    private int levels() {
        int n = 0;
        for (MergeReduceTree t : trees) {
            n += t.levels();
        }
        return n;
    }

    private int depth() {
        int d = 0;
        for (int i = 0; i < trees.length; i++) {
            d = Math.max(d, shadows[i].maxDepth(trees[i].batchesAdded()));
        }
        return d;
    }

    /** Exact encoded size of the state {@link #snapshotState} would write now, from cached level sizes. */
    private long stateBytes() {
        long size = OperatorStates.Merge.FIXED_BYTES;
        for (int i = 0; i < trees.length; i++) {
            size += OperatorStates.SubtaskTree.FIXED_BYTES + 4 + shadows[i].totalBytes() + 4L * trees[i].levels();
        }
        return size;
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        List<OperatorStates.SubtaskTree> list = new ArrayList<>(trees.length);
        for (int i = 0; i < trees.length; i++) {
            long b = trees[i].batchesAdded();
            list.add(new OperatorStates.SubtaskTree(nextSequence[i], records[i], weight[i], b, trees[i].levelsView(),
                    shadows[i].depths(b)));
        }
        OperatorStates.Merge s = new OperatorStates.Merge(config.fingerprint(), parallelism, partials, snapshots,
                sinceSnapshot, peakStateBytes, list);
        lastStateBytes = stateBytes();
        state.update(List.of(s));
    }
}
