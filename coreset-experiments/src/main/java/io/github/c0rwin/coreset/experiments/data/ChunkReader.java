package io.github.c0rwin.coreset.experiments.data;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/** Sequential reader over a row range, handing out rows as CSR blocks (no object per point). */
public interface ChunkReader extends Closeable {

    /** Next chunk of rows as a fresh block (callers may keep it), or null at the end of the range. */
    PointBlock next() throws IOException;

    /** Rows as engine-facing points: one {@link WeightedPoint} view per row, created only when the engine asks. */
    static Iterator<WeightedPoint> points(ChunkReader reader) {
        return new Iterator<>() {
            private PointBlock block;
            private int pos;
            private boolean done;

            @Override
            public boolean hasNext() {
                while (!done && (block == null || pos == block.size())) {
                    try {
                        block = reader.next();
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                    pos = 0;
                    if (block == null) {
                        done = true;
                        try {
                            reader.close();
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    }
                }
                return !done;
            }

            @Override
            public WeightedPoint next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return block.point(pos++);
            }
        };
    }

    /** A block as a list of point views (for APIs taking {@code List<WeightedPoint>}); nothing is copied. */
    static List<WeightedPoint> asList(PointBlock block) {
        return new java.util.AbstractList<>() {
            @Override
            public WeightedPoint get(int i) {
                return block.point(i);
            }

            @Override
            public int size() {
                return block.size();
            }
        };
    }
}
