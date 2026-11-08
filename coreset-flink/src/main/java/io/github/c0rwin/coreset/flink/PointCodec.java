package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/**
 * Bit-exact binary layout of points and coresets on Flink's data views, shared by every serializer of this module.
 * Doubles are written as raw little-endian IEEE-754 bits, indices as delta-encoded unsigned varints, so a point with
 * {@code s} non-zeros costs about {@code 8 + 1 + s * (8 + 1..2)} bytes. A coreset encodes to the same bytes as
 * {@code coreset-spark}'s {@code CoresetCodec}, so either engine can read the other's coresets.
 *
 * <pre>
 * body    := nnz:varint delta:varint{nnz} value:f64{nnz}   (first delta is the index, then index - previous)
 * point   := dim:varint weight:f64 body
 * coreset := offset:f64 n:varint [dim:varint (weight:f64 body){n}]   (dim present only when n &gt; 0)
 * </pre>
 *
 * <p>Reads reject indices that are not strictly increasing or are out of range, and rebuild vectors through
 * {@link SparseVector#of}, so corrupt bytes fail with {@link IOException} instead of producing an invalid value.
 */
final class PointCodec {

    /** Upper bound on the list capacity reserved from an untrusted count before its elements are read. */
    private static final int MAX_PRESIZE = 1 << 16;

    private PointCodec() {
    }

    static int varintLength(int v) {
        return (32 - Integer.numberOfLeadingZeros(v | 1) + 6) / 7;
    }

    static long bodyLength(SparseVector v) {
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

    static long pointLength(WeightedPoint p) {
        return varintLength(p.vector().dim()) + 8L + bodyLength(p.vector());
    }

    static long coresetLength(Coreset c) {
        List<WeightedPoint> points = c.points();
        int n = points.size();
        long size = 8L + varintLength(n);
        if (n > 0) {
            size += varintLength(points.get(0).vector().dim());
            for (int i = 0; i < n; i++) {
                size += 8 + bodyLength(points.get(i).vector());
            }
        }
        return size;
    }

    static void writeVarint(int v, DataOutputView out) throws IOException {
        while ((v & ~0x7F) != 0) {
            out.writeByte((v & 0x7F) | 0x80);
            v >>>= 7;
        }
        out.writeByte(v);
    }

    static int readVarint(DataInputView in) throws IOException {
        int v = 0;
        for (int shift = 0; shift < 35; shift += 7) {
            byte b = in.readByte();
            v |= (b & 0x7F) << shift;
            if (b >= 0) {
                if (v < 0) {
                    break;
                }
                return v;
            }
        }
        throw new IOException("malformed varint");
    }

    static void writeF64(double v, DataOutputView out) throws IOException {
        out.writeLong(Long.reverseBytes(Double.doubleToRawLongBits(v)));
    }

    static double readF64(DataInputView in) throws IOException {
        return Double.longBitsToDouble(Long.reverseBytes(in.readLong()));
    }

    static void writeBody(SparseVector v, DataOutputView out) throws IOException {
        int nnz = v.nnz();
        writeVarint(nnz, out);
        int prev = 0;
        for (int k = 0; k < nnz; k++) {
            int idx = v.indexAt(k);
            writeVarint(idx - prev, out);
            prev = idx;
        }
        for (int k = 0; k < nnz; k++) {
            writeF64(v.valueAt(k), out);
        }
    }

    static SparseVector readBody(int dim, DataInputView in) throws IOException {
        int nnz = readVarint(in);
        if (nnz > dim) {
            throw new IOException(nnz + " non-zeros in a vector of dimension " + dim);
        }
        int[] idx = new int[nnz];
        int prev = 0;
        for (int k = 0; k < nnz; k++) {
            int i = prev + readVarint(in);
            if (k > 0 ? i <= prev : i < 0) {
                throw new IOException("indices not strictly increasing");
            }
            if (i >= dim) {
                throw new IOException("index " + i + " out of range [0, " + dim + ")");
            }
            idx[k] = i;
            prev = i;
        }
        double[] val = new double[nnz];
        for (int k = 0; k < nnz; k++) {
            val[k] = readF64(in);
        }
        try {
            return SparseVector.of(dim, idx, val);
        } catch (IllegalArgumentException e) {
            throw new IOException("invalid vector: " + e.getMessage(), e);
        }
    }

    static void writePoint(WeightedPoint p, DataOutputView out) throws IOException {
        writeVarint(p.vector().dim(), out);
        writeF64(p.weight(), out);
        writeBody(p.vector(), out);
    }

    static WeightedPoint readPoint(DataInputView in) throws IOException {
        int dim = readVarint(in);
        double weight = readF64(in);
        return point(readBody(dim, in), weight);
    }

    static void writeCoreset(Coreset c, DataOutputView out) throws IOException {
        List<WeightedPoint> points = c.points();
        int n = points.size();
        writeF64(c.offset(), out);
        writeVarint(n, out);
        if (n > 0) {
            writeVarint(points.get(0).vector().dim(), out);
            for (int i = 0; i < n; i++) {
                WeightedPoint p = points.get(i);
                writeF64(p.weight(), out);
                writeBody(p.vector(), out);
            }
        }
    }

    static Coreset readCoreset(DataInputView in) throws IOException {
        double offset = readF64(in);
        int n = readVarint(in);
        List<WeightedPoint> points = new ArrayList<>(Math.min(n, MAX_PRESIZE));
        if (n > 0) {
            int dim = readVarint(in);
            for (int i = 0; i < n; i++) {
                double weight = readF64(in);
                points.add(point(readBody(dim, in), weight));
            }
        }
        try {
            return new Coreset(points, offset);
        } catch (IllegalArgumentException e) {
            throw new IOException("invalid coreset: " + e.getMessage(), e);
        }
    }

    /** Copies one encoded point without materialising it. */
    static void copyPoint(DataInputView in, DataOutputView out) throws IOException {
        copyVarint(in, out);
        out.writeLong(in.readLong());
        copyBody(in, out);
    }

    /** Copies one encoded coreset without materialising it. */
    static void copyCoreset(DataInputView in, DataOutputView out) throws IOException {
        out.writeLong(in.readLong());
        int n = copyVarint(in, out);
        if (n > 0) {
            copyVarint(in, out);
            for (int i = 0; i < n; i++) {
                out.writeLong(in.readLong());
                copyBody(in, out);
            }
        }
    }

    private static void copyBody(DataInputView in, DataOutputView out) throws IOException {
        int nnz = copyVarint(in, out);
        for (int k = 0; k < nnz; k++) {
            copyVarint(in, out);
        }
        out.write(in, 8 * nnz);
    }

    private static int copyVarint(DataInputView in, DataOutputView out) throws IOException {
        int v = readVarint(in);
        writeVarint(v, out);
        return v;
    }

    /** Point list as stored in operator state: {@code n:int point{n}}. */
    static void writePoints(List<WeightedPoint> points, DataOutputView out) throws IOException {
        out.writeInt(points.size());
        for (WeightedPoint p : points) {
            writePoint(p, out);
        }
    }

    static List<WeightedPoint> readPoints(DataInputView in) throws IOException {
        int n = count(in);
        List<WeightedPoint> points = new ArrayList<>(Math.min(n, MAX_PRESIZE));
        for (int i = 0; i < n; i++) {
            points.add(readPoint(in));
        }
        return List.copyOf(points);
    }

    /** Coreset list as stored in operator state: {@code n:int coreset{n}}. */
    static void writeCoresets(List<Coreset> coresets, DataOutputView out) throws IOException {
        out.writeInt(coresets.size());
        for (Coreset c : coresets) {
            writeCoreset(c, out);
        }
    }

    static List<Coreset> readCoresets(DataInputView in) throws IOException {
        int n = count(in);
        List<Coreset> out = new ArrayList<>(Math.min(n, MAX_PRESIZE));
        for (int i = 0; i < n; i++) {
            out.add(readCoreset(in));
        }
        return List.copyOf(out);
    }

    static void copyPoints(DataInputView in, DataOutputView out) throws IOException {
        int n = in.readInt();
        out.writeInt(n);
        for (int i = 0; i < n; i++) {
            copyPoint(in, out);
        }
    }

    static void copyCoresets(DataInputView in, DataOutputView out) throws IOException {
        int n = in.readInt();
        out.writeInt(n);
        for (int i = 0; i < n; i++) {
            copyCoreset(in, out);
        }
    }

    static long pointsLength(List<WeightedPoint> points) {
        long size = 4;
        for (WeightedPoint p : points) {
            size += pointLength(p);
        }
        return size;
    }

    static long coresetsLength(List<Coreset> coresets) {
        long size = 4;
        for (Coreset c : coresets) {
            size += coresetLength(c);
        }
        return size;
    }

    static int count(DataInputView in) throws IOException {
        int n = in.readInt();
        if (n < 0) {
            throw new IOException("negative element count " + n);
        }
        return n;
    }

    private static WeightedPoint point(SparseVector v, double weight) throws IOException {
        try {
            return new WeightedPoint(v, weight);
        } catch (IllegalArgumentException e) {
            throw new IOException("invalid point: " + e.getMessage(), e);
        }
    }
}
