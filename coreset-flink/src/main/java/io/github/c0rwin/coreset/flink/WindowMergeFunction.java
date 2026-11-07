package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

/**
 * Merge stage of {@link FlinkCoresets#tumblingWindows}: a window function over the subtask partials of one window.
 * Partials are ordered by subtask, so the result does not depend on arrival order; several partials are united and
 * compressed once with {@code derive(windowSeed, -1, FINAL_LEVEL, 0)}, a single partial is passed through.
 */
final class WindowMergeFunction extends ProcessWindowFunction<WindowPartial, WindowCoreset, Long, TimeWindow> {

    private static final long serialVersionUID = 1L;
    static final long MERGE_STREAM = -1;

    private final CoresetConfig config;
    private transient CoresetAlgorithm algorithm;

    WindowMergeFunction(CoresetConfig config) {
        this.config = config;
    }

    @Override
    public void open(OpenContext openContext) {
        algorithm = config.algorithm();
    }

    @Override
    public void process(Long key, Context context, Iterable<WindowPartial> elements, Collector<WindowCoreset> out) {
        List<WindowPartial> parts = new ArrayList<>();
        elements.forEach(parts::add);
        parts.sort(Comparator.comparingInt(WindowPartial::subtask));
        long records = 0;
        double weight = 0.0;
        int depth = 0;
        int size = 0;
        double offset = 0.0;
        for (WindowPartial p : parts) {
            records += p.records();
            weight += p.weight();
            depth = Math.max(depth, p.depth());
            size += p.coreset().size();
            offset += p.coreset().offset();
        }
        Coreset coreset;
        if (parts.size() == 1) {
            coreset = parts.get(0).coreset();
        } else {
            List<WeightedPoint> points = new ArrayList<>(size);
            for (WindowPartial p : parts) {
                points.addAll(p.coreset().points());
            }
            long seed = WindowCoresetOperator.windowSeed(config.seed(), context.window().getStart());
            coreset = algorithm.compress(new Coreset(points, offset), new SplittableRandom(
                    MergeRandomness.derive(seed, MERGE_STREAM, MergeRandomness.FINAL_LEVEL, 0)), MetricsSink.NOOP);
            depth++;
        }
        TimeWindow w = context.window();
        out.collect(new WindowCoreset(w.getStart(), w.getEnd(), coreset, records, weight, parts.size(), depth));
    }
}
