package io.github.c0rwin.coreset.experiments.quality;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.experiments.data.ChunkReader;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** The rows of a source as one in-memory point set, or a uniform sample of them when there are too many. */
final class InMemory {

    private InMemory() {
    }

    /**
     * All rows when {@code size() <= maxPoints} (blocks kept as read, no copy); otherwise exactly {@code maxPoints}
     * rows chosen uniformly in one sequential pass (selection sampling), each weighted by {@code n / maxPoints}.
     */
    static Coreset load(PointSource src, long maxPoints, long seed) throws IOException {
        long n = src.size();
        List<PointSet> parts = new ArrayList<>();
        try (ChunkReader r = src.open(0, n, FullCost.CHUNK_ROWS)) {
            if (n <= maxPoints) {
                for (PointBlock b = r.next(); b != null; b = r.next()) {
                    parts.add(PointSet.of(b));
                }
                return Coreset.of(PointSet.concat(parts));
            }
            int s = Math.toIntExact(maxPoints);
            double scale = (double) n / s;
            SplittableRandom rnd = new SplittableRandom(seed);
            PointBlock.Builder out = PointBlock.builder(src.dim(), s, (long) s * Math.min(src.dim(), 64));
            long seen = 0;
            int picked = 0;
            int[] idx = new int[src.dim()];
            double[] val = new double[src.dim()];
            for (PointBlock b = r.next(); b != null; b = r.next()) {
                for (int i = 0; i < b.size(); i++, seen++) {
                    if (rnd.nextDouble() * (n - seen) < s - picked) { // Knuth's algorithm S: exactly s rows
                        int nnz = 0;
                        for (int e = b.rowStart(i); e < b.rowEnd(i); e++) {
                            idx[nnz] = b.index(e);
                            val[nnz++] = b.value(e);
                        }
                        out.addRow(idx, val, 0, nnz, b.weight(i) * scale);
                        picked++;
                    }
                }
            }
            return Coreset.of(PointSet.of(out.build()));
        }
    }
}
