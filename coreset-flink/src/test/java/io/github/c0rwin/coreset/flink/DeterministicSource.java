package io.github.c0rwin.coreset.flink;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import org.apache.flink.api.common.eventtime.Watermark;
import org.apache.flink.api.connector.source.ReaderOutput;
import org.apache.flink.api.connector.source.SourceOutput;
import org.apache.flink.api.connector.source.SourceReader;
import org.apache.flink.api.connector.source.SourceReaderContext;
import org.apache.flink.api.connector.source.SplitEnumerator;
import org.apache.flink.api.connector.source.SplitEnumeratorContext;
import org.apache.flink.api.connector.source.lib.NumberSequenceSource;
import org.apache.flink.api.connector.source.lib.util.IteratorSourceReader;
import org.apache.flink.core.io.InputStatus;

/**
 * Bounded sequence {@code 0 .. count - 1}, split into equal contiguous ranges, whose range {@code i} always goes to
 * reader {@code i}: the stock enumerator
 * hands splits to whichever reader asks first, which would make per-subtask inputs differ between runs. Each reader
 * can also pause after {@code pauseAfter} records until a checkpoint taken during the pause completes, so a failure
 * injected later is guaranteed to recover from a checkpoint that holds real state.
 */
final class DeterministicSource extends NumberSequenceSource {

    private static final long serialVersionUID = 1L;

    private final long pauseAfter;

    /** @param pauseAfter records per reader before the pause, negative for no pause */
    DeterministicSource(long count, long pauseAfter) {
        super(0, count - 1);
        this.pauseAfter = pauseAfter;
    }

    @Override
    public SplitEnumerator<NumberSequenceSplit, Collection<NumberSequenceSplit>> createEnumerator(
            SplitEnumeratorContext<NumberSequenceSplit> context) {
        return new StaticEnumerator(context, splitNumberRange(getFrom(), getTo(), context.currentParallelism()));
    }

    @Override
    public SplitEnumerator<NumberSequenceSplit, Collection<NumberSequenceSplit>> restoreEnumerator(
            SplitEnumeratorContext<NumberSequenceSplit> context, Collection<NumberSequenceSplit> checkpoint) {
        return new StaticEnumerator(context, checkpoint);
    }

    @Override
    public SourceReader<Long, NumberSequenceSplit> createReader(SourceReaderContext context) {
        return new GatedReader(new IteratorSourceReader<>(context), pauseAfter);
    }

    private static final class StaticEnumerator
            implements SplitEnumerator<NumberSequenceSplit, Collection<NumberSequenceSplit>> {

        private final SplitEnumeratorContext<NumberSequenceSplit> context;
        private final Map<Integer, NumberSequenceSplit> remaining = new TreeMap<>();

        StaticEnumerator(SplitEnumeratorContext<NumberSequenceSplit> context, Collection<NumberSequenceSplit> splits) {
            this.context = context;
            splits.forEach(s -> remaining.put(owner(s), s));
        }

        private static int owner(NumberSequenceSplit s) {
            return Integer.parseInt(s.splitId()) - 1;
        }

        @Override
        public void start() {
        }

        @Override
        public void handleSplitRequest(int subtask, String host) {
            NumberSequenceSplit split = remaining.remove(subtask);
            if (split != null) {
                context.assignSplit(split, subtask);
            } else {
                context.signalNoMoreSplits(subtask);
            }
        }

        @Override
        public void addSplitsBack(List<NumberSequenceSplit> splits, int subtask) {
            splits.forEach(s -> remaining.put(owner(s), s));
        }

        @Override
        public void addReader(int subtask) {
        }

        @Override
        public Collection<NumberSequenceSplit> snapshotState(long checkpointId) {
            return new ArrayList<>(remaining.values());
        }

        @Override
        public void close() {
        }
    }

    private static final class GatedReader implements SourceReader<Long, NumberSequenceSplit> {

        private final SourceReader<Long, NumberSequenceSplit> delegate;
        private final long pauseAfter;
        private final Counting counting = new Counting();
        private boolean pausedOnce;
        private CompletableFuture<Void> resume;
        private long pauseCheckpoint = -1;

        GatedReader(SourceReader<Long, NumberSequenceSplit> delegate, long pauseAfter) {
            this.delegate = delegate;
            this.pauseAfter = pauseAfter;
        }

        @Override
        public void start() {
            delegate.start();
        }

        @Override
        public InputStatus pollNext(ReaderOutput<Long> output) throws Exception {
            if (resume != null) {
                return InputStatus.NOTHING_AVAILABLE;
            }
            if (pauseAfter >= 0 && !pausedOnce && counting.emitted >= pauseAfter) {
                pausedOnce = true;
                resume = new CompletableFuture<>();
                return InputStatus.NOTHING_AVAILABLE;
            }
            counting.target = output;
            return delegate.pollNext(counting);
        }

        @Override
        public List<NumberSequenceSplit> snapshotState(long checkpointId) {
            if (resume != null && pauseCheckpoint < 0) {
                pauseCheckpoint = checkpointId;
            }
            return delegate.snapshotState(checkpointId);
        }

        @Override
        public void notifyCheckpointComplete(long checkpointId) throws Exception {
            delegate.notifyCheckpointComplete(checkpointId);
            if (resume != null && pauseCheckpoint >= 0 && checkpointId >= pauseCheckpoint) {
                CompletableFuture<Void> r = resume;
                resume = null;
                r.complete(null);
            }
        }

        @Override
        public CompletableFuture<Void> isAvailable() {
            return resume != null ? resume : delegate.isAvailable();
        }

        @Override
        public void addSplits(List<NumberSequenceSplit> splits) {
            delegate.addSplits(splits);
        }

        @Override
        public void notifyNoMoreSplits() {
            delegate.notifyNoMoreSplits();
        }

        @Override
        public void close() throws Exception {
            delegate.close();
        }
    }

    /** Forwards to the task's output and counts records. */
    private static final class Counting implements ReaderOutput<Long> {
        ReaderOutput<Long> target;
        long emitted;

        @Override
        public void collect(Long record) {
            emitted++;
            target.collect(record);
        }

        @Override
        public void collect(Long record, long timestamp) {
            emitted++;
            target.collect(record, timestamp);
        }

        @Override
        public void emitWatermark(Watermark watermark) {
            target.emitWatermark(watermark);
        }

        @Override
        public void markIdle() {
            target.markIdle();
        }

        @Override
        public void markActive() {
            target.markActive();
        }

        @Override
        public SourceOutput<Long> createOutputForSplit(String splitId) {
            return target.createOutputForSplit(splitId);
        }

        @Override
        public void releaseOutputForSplit(String splitId) {
            target.releaseOutputForSplit(splitId);
        }
    }
}
