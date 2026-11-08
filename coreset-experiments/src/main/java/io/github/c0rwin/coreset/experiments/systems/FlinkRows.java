package io.github.c0rwin.coreset.experiments.systems;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.experiments.data.ChunkReader;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.CheckpointListener;
import org.apache.flink.api.connector.source.SplitEnumerator;
import org.apache.flink.api.connector.source.SplitEnumeratorContext;
import org.apache.flink.api.connector.source.lib.NumberSequenceSource;

/**
 * Flink input: row indices from a source whose split {@code i} (shard {@code i}, as for the other systems) always goes to
 * reader {@code i} (the stock enumerator hands splits to whichever reader asks first, so per-subtask inputs and hence
 * coresets would differ between runs), and a chained map that turns indices into rows by reading the shard
 * sequentially in chunks. Optionally one subtask fails once, after a checkpoint completed, to measure recovery.
 */
final class FlinkRows {

    private FlinkRows() {
    }

    /** Failure and recovery timestamps per run, shared within the JVM (MiniCluster tasks run in it). */
    static final Map<String, Map<String, Long>> CLOCK = new ConcurrentHashMap<>();

    /** Row indices {@code 0 .. n-1} in {@code shards} static splits. */
    static final class Indices extends NumberSequenceSource {
        private static final long serialVersionUID = 1L;
        private final int shards;

        Indices(long n, int shards) {
            super(0, n - 1);
            this.shards = shards;
        }

        @Override
        public SplitEnumerator<NumberSequenceSplit, Collection<NumberSequenceSplit>> createEnumerator(
                SplitEnumeratorContext<NumberSequenceSplit> context) {
            long n = getTo() + 1;
            List<NumberSequenceSplit> splits = new ArrayList<>(shards);
            for (int i = 0; i < shards; i++) {
                long from = PointSource.shardStart(n, i, shards);
                long to = PointSource.shardStart(n, i + 1, shards);
                if (to > from) {
                    splits.add(new NumberSequenceSplit(Integer.toString(i), from, to - 1));
                }
            }
            return new StaticEnumerator(context, splits);
        }

        @Override
        public SplitEnumerator<NumberSequenceSplit, Collection<NumberSequenceSplit>> restoreEnumerator(
                SplitEnumeratorContext<NumberSequenceSplit> context, Collection<NumberSequenceSplit> checkpoint) {
            return new StaticEnumerator(context, checkpoint);
        }
    }

    private static final class StaticEnumerator
            implements SplitEnumerator<NumberSequenceSource.NumberSequenceSplit,
            Collection<NumberSequenceSource.NumberSequenceSplit>> {
        private final SplitEnumeratorContext<NumberSequenceSource.NumberSequenceSplit> context;
        private final Map<Integer, NumberSequenceSource.NumberSequenceSplit> remaining = new TreeMap<>();

        StaticEnumerator(SplitEnumeratorContext<NumberSequenceSource.NumberSequenceSplit> context,
                         Collection<NumberSequenceSource.NumberSequenceSplit> splits) {
            this.context = context;
            splits.forEach(s -> remaining.put(Integer.parseInt(s.splitId()), s));
        }

        @Override
        public void start() {
        }

        @Override
        public void handleSplitRequest(int subtask, String host) {
            NumberSequenceSource.NumberSequenceSplit split = remaining.remove(subtask);
            if (split != null) {
                context.assignSplit(split, subtask);
            } else {
                context.signalNoMoreSplits(subtask);
            }
        }

        @Override
        public void addSplitsBack(List<NumberSequenceSource.NumberSequenceSplit> splits, int subtask) {
            splits.forEach(s -> remaining.put(Integer.parseInt(s.splitId()), s));
        }

        @Override
        public void addReader(int subtask) {
        }

        @Override
        public Collection<NumberSequenceSource.NumberSequenceSplit> snapshotState(long checkpointId) {
            return new ArrayList<>(remaining.values());
        }

        @Override
        public void close() {
        }
    }

    /**
     * Index to row, reading ahead one chunk; a non-consecutive index (first record, or replay after a restore)
     * reopens the reader there. With {@code failSubtask >= 0}, that subtask throws in its first attempt once it has
     * mapped {@code failFraction} of its shard and at least one checkpoint has completed.
     */
    static final class Reader extends RichMapFunction<Long, WeightedPoint> implements CheckpointListener {
        private static final long serialVersionUID = 1L;
        private final PointSource src;
        private final int chunkRows;
        private final String run;
        private final int failSubtask;
        private final double failFraction;
        private transient ChunkReader reader;
        private transient PointBlock block;
        private transient long blockStart;
        private transient long seen;
        private transient long failAt;
        private transient int subtask;
        private transient int attempt;
        private transient volatile boolean checkpointed;

        Reader(PointSource src, int chunkRows, String run, int failSubtask, double failFraction) {
            this.src = src;
            this.chunkRows = chunkRows;
            this.run = run;
            this.failSubtask = failSubtask;
            this.failFraction = failFraction;
        }

        @Override
        public void open(OpenContext ctx) {
            subtask = getRuntimeContext().getTaskInfo().getIndexOfThisSubtask();
            attempt = getRuntimeContext().getTaskInfo().getAttemptNumber();
            int shards = getRuntimeContext().getTaskInfo().getNumberOfParallelSubtasks();
            long shard = PointSource.shardStart(src.size(), subtask + 1, shards)
                    - PointSource.shardStart(src.size(), subtask, shards);
            failAt = failSubtask == subtask && attempt == 0 ? Math.max(1, (long) (failFraction * shard)) : -1;
        }

        @Override
        public WeightedPoint map(Long boxed) throws IOException {
            long index = boxed;
            seen++;
            if (failAt > 0 && seen >= failAt && checkpointed) {
                Map<String, Long> c = CLOCK.computeIfAbsent(run, r -> new ConcurrentHashMap<>());
                c.putIfAbsent("failNanos", System.nanoTime());
                c.putIfAbsent("failIndex", index);
                failAt = -1;
                throw new IllegalStateException("injected task failure at row " + index);
            }
            if (attempt > 0 && seen == 1 && subtask == failSubtask) { // the failed subtask's replay start
                Map<String, Long> c = CLOCK.computeIfAbsent(run, r -> new ConcurrentHashMap<>());
                c.putIfAbsent("resumeNanos", System.nanoTime());
                c.putIfAbsent("resumeIndex", index);
            }
            if (block == null || index < blockStart || index >= blockStart + block.size()) {
                if (block == null || index != blockStart + block.size()) {
                    if (reader != null) {
                        reader.close();
                    }
                    reader = src.open(index, src.size(), chunkRows);
                }
                blockStart = index;
                block = reader.next();
            }
            return block.point((int) (index - blockStart));
        }

        @Override
        public void notifyCheckpointComplete(long checkpointId) {
            checkpointed = true;
        }

        @Override
        public void close() throws IOException {
            if (reader != null) {
                reader.close();
            }
        }
    }
}
