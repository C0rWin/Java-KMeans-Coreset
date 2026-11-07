package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.List;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/**
 * Local stage of {@link FlinkCoresets#stream}: buffers each subtask's records into chunks, merges the chunks in a
 * {@link ChunkedTree} and emits the tree's summary as a {@link PartialCoreset} on every trigger (count, processing
 * time, event time, end of input).
 *
 * <p>The tree lives in operator fields and is written to Flink operator list state only in
 * {@link #snapshotState}, so records never touch state. Merges are seeded from {@code (seed, subtask, level, chunk
 * index)}; after a restore from a checkpoint and a replay of the same per-subtask input the operator emits exactly the
 * partials of an undisturbed run. Restoring under a different parallelism or configuration fails.
 */
final class LocalCoresetOperator extends AbstractStreamOperator<PartialCoreset>
        implements OneInputStreamOperator<WeightedPoint, PartialCoreset>, BoundedOneInput {

    private static final long serialVersionUID = 1L;
    static final String STATE_NAME = "coreset-local-tree";

    private final CoresetConfig config;
    private transient ChunkedTree tree;
    private transient ListState<OperatorStates.Local> state;
    private transient StreamRecord<PartialCoreset> reuse;
    private transient int subtask;
    private transient int parallelism;
    private transient int chunksInEpoch;
    private transient long partialsEmitted;
    private transient long emittedRecords;
    private transient long nextEventBoundary;
    private transient long lastStateBytes;

    LocalCoresetOperator(CoresetConfig config) {
        this.config = config;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        subtask = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
        parallelism = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        tree = new ChunkedTree(config.algorithm(), config.seed(), subtask, config.chunkSize());
        nextEventBoundary = Long.MIN_VALUE;
        state = context.getOperatorStateStore().getListState(
                new ListStateDescriptor<>(STATE_NAME, OperatorStates.LocalSerializer.INSTANCE));
        if (!context.isRestored()) {
            return;
        }
        List<OperatorStates.Local> restored = new ArrayList<>();
        state.get().forEach(restored::add);
        if (restored.isEmpty()) {
            return;
        }
        OperatorStates.Local s = restored.get(0);
        if (restored.size() > 1 || s.subtask() != subtask || s.parallelism() != parallelism) {
            throw new IllegalStateException("local coreset state written by subtask " + s.subtask() + " of "
                    + s.parallelism() + " restored into subtask " + subtask + " of " + parallelism + " ("
                    + restored.size() + " entries); rescaling the coreset pipeline is not supported");
        }
        if (s.fingerprint() != config.fingerprint()) {
            throw new IllegalStateException("local coreset state was written with a different algorithm, parameters,"
                    + " seed or chunk size than " + config);
        }
        tree.restore(s.builder());
        partialsEmitted = s.partialsEmitted();
        emittedRecords = s.records();
        nextEventBoundary = s.nextEventBoundary();
        tree.restorePeak(s.peakStateBytes() - OperatorStates.Local.FIXED_BYTES);
        chunksInEpoch = (int) s.builder().batches();
    }

    @Override
    public void open() throws Exception {
        super.open();
        reuse = new StreamRecord<>(null);
        MetricGroup g = getMetricGroup().addGroup("coreset");
        g.gauge("recordsIn", () -> emittedRecords + tree.records());
        g.gauge("partialsEmitted", () -> partialsEmitted);
        g.gauge("chunks", () -> tree.chunks());
        g.gauge("treeLevels", () -> tree.levels());
        g.gauge("treeDepth", () -> tree.depth());
        g.gauge("stateBytes", () -> lastStateBytes);
        g.gauge("peakStateBytes", this::peakStateBytes);
        long interval = config.partialIntervalMillis();
        if (interval > 0) {
            long now = getProcessingTimeService().getCurrentProcessingTime();
            getProcessingTimeService().registerTimer(now + interval, this::onProcessingTime);
        }
    }

    @Override
    public void processElement(StreamRecord<WeightedPoint> element) {
        if (tree.add(element.getValue())) {
            tree.flushChunk();
            if (++chunksInEpoch == config.chunksPerPartial()) {
                emitPartial();
            }
        }
    }

    private void onProcessingTime(long time) {
        emitPartial();
        getProcessingTimeService().registerTimer(time + config.partialIntervalMillis(), this::onProcessingTime);
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        long interval = config.partialEventTimeIntervalMillis();
        if (interval > 0 && mark.getTimestamp() != Long.MIN_VALUE) {
            long boundary = Math.floorDiv(mark.getTimestamp(), interval) * interval;
            if (nextEventBoundary == Long.MIN_VALUE) {
                nextEventBoundary = boundary + interval;
            } else if (mark.getTimestamp() >= nextEventBoundary) {
                emitPartial();
                nextEventBoundary = mark.getTimestamp() == Long.MAX_VALUE ? Long.MAX_VALUE : boundary + interval;
            }
        }
        super.processWatermark(mark);
    }

    @Override
    public void endInput() {
        emitPartial();
    }

    /** Largest encoded size of this operator's state so far: the tree's peak plus the fixed header. */
    private long peakStateBytes() {
        return OperatorStates.Local.FIXED_BYTES + tree.peakStateBytes();
    }

    private void emitPartial() {
        chunksInEpoch = 0;
        ChunkedTree.Partial p = tree.takePartial();
        if (p == null) {
            return;
        }
        emittedRecords += p.records();
        output.collect(reuse.replace(new PartialCoreset(subtask, parallelism, partialsEmitted++, p.coreset(),
                p.records(), p.weight(), p.depth())));
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        OperatorStates.Local s = new OperatorStates.Local(config.fingerprint(), subtask, parallelism, partialsEmitted,
                emittedRecords, nextEventBoundary, peakStateBytes(), tree.state());
        lastStateBytes = OperatorStates.Local.FIXED_BYTES + tree.stateBytes(); // cached level sizes, buffer pass
        state.update(List.of(s));
    }
}
