package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Compact, bit-exact binary form of coresets, used for partial coresets shipped between Spark stages, for Kryo and
 * for streaming checkpoints. Doubles are stored as raw little-endian IEEE-754 bits (so decoding reproduces every
 * value exactly) and indices as delta-encoded unsigned varints, so a point with {@code s} non-zeros costs about
 * {@code 8 + 1 + s * (8 + 1..2)} bytes.
 *
 * <pre>
 * coreset := offset:f64 n:varint [dim:varint point{n}]   (dim present only when n &gt; 0)
 * point   := weight:f64 body
 * body    := nnz:varint delta:varint{nnz} value:f64{nnz} (first delta is the index, then index - previous)
 * vector  := dim:varint body                              (standalone SparseVector)
 * wpoint  := dim:varint weight:f64 body                   (standalone WeightedPoint)
 * </pre>
 *
 * <p>Encoding sizes the output exactly (one read of the index arrays), then writes it in one pass into a single
 * allocation, reading the rows straight from the coreset's segments. Decoding validates lengths against the remaining
 * input before allocating and decodes a coreset in one pass straight into a {@link PointBlock}
 * ({@link PointBlock#wrap}, no further copy), so malformed input fails with {@link IllegalArgumentException}.
 */
public final class CoresetCodec {

    private static final VarHandle LONGS = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    private CoresetCodec() {
    }

    /** Exact size of {@link #encode(Coreset)}'s output, from one read of the index arrays. */
    static long encodedLength(Coreset coreset) {
        PointSet points = coreset.pointSet();
        int n = points.size();
        long size = 8L + varintLength(n);
        if (n > 0) {
            size += varintLength(points.dim());
            for (int t = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++) {
                    size += 8 + rowLength(b, i);
                }
            }
        }
        return size;
    }

    /** Encodes straight from the segments' arrays: no per-point objects. */
    public static byte[] encode(Coreset coreset) {
        PointSet points = coreset.pointSet();
        int n = points.size();
        Writer w = new Writer(encodedLength(coreset));
        w.f64(coreset.offset());
        w.varint(n);
        if (n > 0) {
            w.varint(points.dim());
            for (int t = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++) {
                    w.f64(b.weight(i));
                    w.row(b, i);
                }
            }
        }
        return w.done();
    }

    /** @throws IllegalArgumentException if {@code bytes} is not exactly one encoded coreset */
    public static Coreset decode(byte[] bytes) {
        return decode(bytes, 0, bytes.length);
    }

    /** Decodes {@code bytes[offset, offset + length)}, which must hold exactly one encoded coreset. */
    public static Coreset decode(byte[] bytes, int offset, int length) {
        Objects.checkFromIndexSize(offset, length, bytes.length);
        Reader r = new Reader(bytes, offset, offset + length);
        double off = r.f64();
        int n = r.length(9); // weight + nnz at least
        PointBlock points = r.block(n);
        r.end();
        return new Coreset(points, off);
    }

    public static byte[] encode(SparseVector v) {
        Writer w = new Writer(varintLength(v.dim()) + bodyLength(v));
        w.varint(v.dim());
        w.body(v);
        return w.done();
    }

    public static SparseVector decodeVector(byte[] bytes) {
        Reader r = new Reader(bytes, 0, bytes.length);
        SparseVector v = r.body(r.varint());
        r.end();
        return v;
    }

    public static byte[] encode(WeightedPoint p) {
        Writer w = new Writer(varintLength(p.vector().dim()) + 8L + bodyLength(p.vector()));
        w.varint(p.vector().dim());
        w.f64(p.weight());
        w.body(p.vector());
        return w.done();
    }

    public static WeightedPoint decodePoint(byte[] bytes) {
        Reader r = new Reader(bytes, 0, bytes.length);
        int dim = r.varint();
        double weight = r.f64();
        WeightedPoint p = new WeightedPoint(r.body(dim), weight);
        r.end();
        return p;
    }

    /** Bytes of an unsigned varint; the codec only writes non-negative ints. */
    private static int varintLength(int v) {
        return (32 - Integer.numberOfLeadingZeros(v | 1) + 6) / 7;
    }

    private static long bodyLength(SparseVector v) {
        int nnz = v.nnz();
        long size = varintLength(nnz) + 8L * nnz;
        int prev = 0;
        for (int k = 0; k < nnz; k++) {
            int idx = v.indexAt(k);
            size += varintLength(idx - prev);
            prev = idx;
        }
        return size;
    }

    /** {@link #bodyLength} of row {@code i}. */
    private static long rowLength(PointBlock b, int i) {
        int from = b.rowStart(i);
        int to = b.rowEnd(i);
        long size = varintLength(to - from) + 8L * (to - from);
        int prev = 0;
        for (int e = from; e < to; e++) {
            int idx = b.index(e);
            size += varintLength(idx - prev);
            prev = idx;
        }
        return size;
    }

    private static final class Writer {
        private final byte[] buf;
        private int pos;

        Writer(long size) {
            if (size > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException("encoding of " + size + " bytes exceeds a byte array");
            }
            buf = new byte[(int) size];
        }

        void f64(double v) {
            LONGS.set(buf, pos, Double.doubleToRawLongBits(v));
            pos += 8;
        }

        void varint(int v) {
            while ((v & ~0x7F) != 0) {
                buf[pos++] = (byte) ((v & 0x7F) | 0x80);
                v >>>= 7;
            }
            buf[pos++] = (byte) v;
        }

        void body(SparseVector v) {
            int nnz = v.nnz();
            varint(nnz);
            int prev = 0;
            for (int k = 0; k < nnz; k++) {
                int idx = v.indexAt(k);
                varint(idx - prev);
                prev = idx;
            }
            for (int k = 0; k < nnz; k++) {
                LONGS.set(buf, pos, Double.doubleToRawLongBits(v.valueAt(k)));
                pos += 8;
            }
        }

        void row(PointBlock b, int i) {
            int from = b.rowStart(i);
            int to = b.rowEnd(i);
            varint(to - from);
            int prev = 0;
            for (int e = from; e < to; e++) {
                int idx = b.index(e);
                varint(idx - prev);
                prev = idx;
            }
            for (int e = from; e < to; e++) {
                LONGS.set(buf, pos, Double.doubleToRawLongBits(b.value(e)));
                pos += 8;
            }
        }

        byte[] done() {
            if (pos != buf.length) {
                throw new IllegalStateException("sized " + buf.length + " bytes, wrote " + pos);
            }
            return buf;
        }
    }

    private static final class Reader {
        private final byte[] buf;
        private final int end;
        private int pos;

        Reader(byte[] buf, int from, int end) {
            this.buf = buf;
            this.pos = from;
            this.end = end;
        }

        private void need(long bytes) {
            if (bytes > end - pos) {
                throw new IllegalArgumentException("truncated coreset encoding at byte " + pos);
            }
        }

        double f64() {
            need(8);
            double v = Double.longBitsToDouble((long) LONGS.get(buf, pos));
            pos += 8;
            return v;
        }

        int varint() {
            int v = 0;
            for (int shift = 0; shift < 35; shift += 7) {
                need(1);
                byte b = buf[pos++];
                v |= (b & 0x7F) << shift;
                if (b >= 0) {
                    if (v < 0) {
                        break;
                    }
                    return v;
                }
            }
            throw new IllegalArgumentException("malformed varint before byte " + pos);
        }

        /** A count whose elements each take at least {@code minBytes}, checked before anything is allocated. */
        int length(int minBytes) {
            int n = varint();
            need((long) n * minBytes);
            return n;
        }

        /**
         * The dim header (when {@code n > 0}) and {@code n} points as one block, in one pass: every point takes at
         * least 9 bytes (weight, nnz) and every entry 9 more (index delta, value), so the remaining input bounds the
         * entries and the entry arrays are allocated once at that bound. They are trimmed (one copy) only when the
         * bound overshoots by more than a quarter, which takes index deltas of 3+ bytes on average.
         * {@link PointBlock#wrap} then validates the rows.
         */
        PointBlock block(int n) {
            if (n == 0) {
                return PointBlock.EMPTY;
            }
            int dim = varint();
            int cap = (int) Math.max(0L, ((long) (end - pos) - 9L * n) / 9);
            int[] offsets = new int[n + 1];
            int[] idx = new int[cap];
            double[] val = new double[cap];
            double[] weights = new double[n];
            int e = 0;
            for (int i = 0; i < n; i++) {
                weights[i] = f64();
                int nnz = length(9);
                if (nnz > cap - e) {
                    throw new IllegalArgumentException("truncated coreset encoding at byte " + pos);
                }
                int prev = 0;
                for (int k = 0; k < nnz; k++, e++) {
                    int index = prev + varint();
                    if (k > 0 ? index <= prev : index < 0) {
                        throw new IllegalArgumentException("indices not strictly increasing at byte " + pos);
                    }
                    idx[e] = index;
                    prev = index;
                }
                need(8L * nnz);
                for (int k = e - nnz; k < e; k++) {
                    val[k] = Double.longBitsToDouble((long) LONGS.get(buf, pos));
                    pos += 8;
                }
                offsets[i + 1] = e;
            }
            if (cap - e > e / 4) {
                idx = Arrays.copyOf(idx, e);
                val = Arrays.copyOf(val, e);
            }
            return PointBlock.wrap(dim, offsets, idx, val, weights);
        }

        SparseVector body(int dim) {
            int nnz = length(9);
            int[] idx = new int[nnz];
            int prev = 0;
            for (int k = 0; k < nnz; k++) {
                int i = prev + varint();
                if (k > 0 ? i <= prev : i < 0) {
                    throw new IllegalArgumentException("indices not strictly increasing at byte " + pos);
                }
                idx[k] = i;
                prev = i;
            }
            need(8L * nnz);
            double[] val = new double[nnz];
            for (int k = 0; k < nnz; k++) {
                val[k] = Double.longBitsToDouble((long) LONGS.get(buf, pos));
                pos += 8;
            }
            return SparseVector.ofOwned(dim, idx, val);
        }

        void end() {
            if (pos != end) {
                throw new IllegalArgumentException((end - pos) + " trailing bytes after coreset encoding");
            }
        }
    }
}
