package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
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
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/**
 * Local stage of {@link FlinkCoresets#tumblingWindows}: assigns each record to its tumbling event-time window and feeds
 * that window's {@link ChunkedTree}. When the watermark passes a window's end the window's summary is emitted as a
 * {@link WindowPartial} stamped with the window's max timestamp, before the watermark is forwarded, so the keyed
 * merge stage sees every subtask's partial when its own window fires.
 *
 * <p>Window {@code [start, start + size)} seeds its trees from {@link #windowSeed}; subtask {@code s} merges with
 * {@code derive(windowSeed, s, level, chunk index)}. Records at or behind the watermark are dropped and counted
 * ({@code coreset.lateRecordsDropped}); there is no allowed lateness. Open windows and the watermark live in operator
 * fields and are written to operator state only at checkpoints; restoring the watermark keeps a window that was
 * already emitted from reopening on replay.
 */
final class WindowCoresetOperator extends AbstractStreamOperator<WindowPartial>
        implements OneInputStreamOperator<WeightedPoint, WindowPartial>, BoundedOneInput {

    private static final long serialVersionUID = 1L;
    static final String STATE_NAME = "coreset-window-trees";
    static final long WINDOW_SEED_STREAM = Long.MIN_VALUE;

    private final CoresetConfig config;
    private final long size;
    private transient CoresetAlgorithm algorithm;
    private transient ListState<OperatorStates.Window> state;
    private transient List<Open> open;
    private transient Open last;
    private transient int subtask;
    private transient int parallelism;
    private transient long watermark;
    private transient long records;
    private transient long late;
    private transient long partialsEmitted;
    private transient long peakStateBytes;
    private transient long lastStateBytes;

    private static final class Open {
        final long start;
        final ChunkedTree tree;

        Open(long start, ChunkedTree tree) {
            this.start = start;
            this.tree = tree;
        }
    }

    WindowCoresetOperator(CoresetConfig config, long sizeMillis) {
        this.config = config;
        this.size = sizeMillis;
    }

    /** Seed of every compression of the window starting at {@code start}. */
    static long windowSeed(long seed, long start) {
        return MergeRandomness.derive(seed, WINDOW_SEED_STREAM, 0, start);
    }

    private ChunkedTree newTree(long start) {
        return new ChunkedTree(algorithm, windowSeed(config.seed(), start), subtask, config.chunkSize());
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        algorithm = config.algorithm();
        subtask = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
        parallelism = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
        open = new ArrayList<>();
        watermark = Long.MIN_VALUE;
        state = context.getOperatorStateStore().getListState(
                new ListStateDescriptor<>(STATE_NAME, OperatorStates.WindowSerializer.INSTANCE));
        if (!context.isRestored()) {
            return;
        }
        List<OperatorStates.Window> restored = new ArrayList<>();
        state.get().forEach(restored::add);
        if (restored.isEmpty()) {
            return;
        }
        OperatorStates.Window s = restored.get(0);
        if (restored.size() > 1 || s.subtask() != subtask || s.parallelism() != parallelism) {
            throw new IllegalStateException("window coreset state written by subtask " + s.subtask() + " of "
                    + s.parallelism() + " restored into subtask " + subtask + " of " + parallelism + " ("
                    + restored.size() + " entries); rescaling the coreset pipeline is not supported");
        }
        if (s.fingerprint() != config.fingerprint()) {
            throw new IllegalStateException("window coreset state was written with a different algorithm,"
                    + " parameters, seed or chunk size than " + config);
        }
        for (OperatorStates.OpenWindow w : s.windows()) {
            ChunkedTree t = newTree(w.start());
            t.restore(w.builder());
            open.add(new Open(w.start(), t));
        }
        watermark = s.watermark();
        records = s.records();
        late = s.lateRecords();
        partialsEmitted = s.partialsEmitted();
        peakStateBytes = s.peakStateBytes();
    }

    @Override
    public void open() throws Exception {
        super.open();
        MetricGroup g = getMetricGroup().addGroup("coreset");
        g.gauge("recordsIn", () -> records);
        g.gauge("lateRecordsDropped", () -> late);
        g.gauge("partialsEmitted", () -> partialsEmitted);
        g.gauge("openWindows", () -> open.size());
        g.gauge("stateBytes", () -> lastStateBytes);
        g.gauge("peakStateBytes", () -> peakStateBytes);
    }

    @Override
    public void processElement(StreamRecord<WeightedPoint> element) {
        if (!element.hasTimestamp()) {
            throw new IllegalStateException("tumbling coreset windows need event-time timestamps; assign them with a"
                    + " WatermarkStrategy before the coreset stage");
        }
        long start = TimeWindow.getWindowStartWithOffset(element.getTimestamp(), 0, size);
        if (start + size - 1 <= watermark) {
            late++;
            return;
        }
        Open w = last != null && last.start == start ? last : find(start);
        last = w;
        records++;
        if (w.tree.add(element.getValue())) {
            notePeak();
            w.tree.flushChunk();
        }
    }

    /** Open window starting at {@code start}, created if absent; the list stays sorted by start. */
    private Open find(long start) {
        int i = 0;
        while (i < open.size() && open.get(i).start < start) {
            i++;
        }
        if (i < open.size() && open.get(i).start == start) {
            return open.get(i);
        }
        Open w = new Open(start, newTree(start));
        open.add(i, w);
        return w;
    }

    /** State only grows between flushes, so its size just before each flush is the running maximum. */
    private void notePeak() {
        peakStateBytes = Math.max(peakStateBytes, stateBytes());
    }

    /** Exact encoded size of the state {@link #snapshotState} would write now. */
    private long stateBytes() {
        long bytes = OperatorStates.Window.FIXED_BYTES;
        for (Open w : open) {
            bytes += 8 + w.tree.stateBytes();
        }
        return bytes;
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        watermark = Math.max(watermark, mark.getTimestamp());
        closeUpTo(watermark);
        super.processWatermark(mark);
    }

    @Override
    public void endInput() {
        closeUpTo(Long.MAX_VALUE);
    }

    private void closeUpTo(long time) {
        if (!open.isEmpty() && open.get(0).start + size - 1 <= time) {
            notePeak();
        }
        while (!open.isEmpty() && open.get(0).start + size - 1 <= time) {
            Open w = open.remove(0);
            if (w == last) {
                last = null;
            }
            ChunkedTree.Partial p = w.tree.takePartial();
            if (p != null) {
                partialsEmitted++;
                output.collect(new StreamRecord<>(new WindowPartial(w.start, subtask, p.coreset(), p.records(),
                        p.weight(), p.depth()), w.start + size - 1));
            }
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        List<OperatorStates.OpenWindow> windows = new ArrayList<>(open.size());
        for (Open w : open) {
            windows.add(new OperatorStates.OpenWindow(w.start, w.tree.state()));
        }
        OperatorStates.Window s = new OperatorStates.Window(config.fingerprint(), subtask, parallelism, watermark,
                records, late, partialsEmitted, peakStateBytes, windows);
        lastStateBytes = stateBytes();
        state.update(List.of(s));
    }
}
