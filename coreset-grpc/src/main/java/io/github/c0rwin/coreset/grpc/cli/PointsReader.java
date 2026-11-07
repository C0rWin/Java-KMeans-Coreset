package io.github.c0rwin.coreset.grpc.cli;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Streams unit-weight points from text, one per line; blank lines and lines starting with {@code #} are skipped.
 * Dense format: numbers separated by commas and/or whitespace. Sparse format (when a dimension is given):
 * {@code index:value} tokens with 0-based indices; a leading token without {@code :} (e.g. a libsvm label) is
 * ignored.
 */
final class PointsReader {

    private static final Pattern SEPARATORS = Pattern.compile("[,\\s]+");

    private final BufferedReader in;
    private final int sparseDim;
    private int denseDim = -1;
    private long line;

    /** @param sparseDim dimension for the sparse format, or 0 for the dense format */
    PointsReader(BufferedReader in, int sparseDim) {
        this.in = in;
        this.sparseDim = sparseDim;
    }

    /** Next batch of at most {@code max} points; empty at end of input. */
    List<WeightedPoint> next(int max) throws IOException {
        List<WeightedPoint> out = new ArrayList<>(max);
        String s;
        while (out.size() < max && (s = in.readLine()) != null) {
            line++;
            s = s.strip();
            if (s.isEmpty() || s.startsWith("#")) {
                continue;
            }
            try {
                out.add(new WeightedPoint(sparseDim > 0 ? sparse(s) : dense(s), 1.0));
            } catch (IllegalArgumentException e) {
                throw new IOException("line " + line + ": " + e.getMessage(), e);
            }
        }
        return out;
    }

    private SparseVector dense(String s) {
        String[] tokens = SEPARATORS.split(s);
        if (denseDim < 0) {
            denseDim = tokens.length;
        } else if (tokens.length != denseDim) {
            throw new IllegalArgumentException("expected " + denseDim + " values, found " + tokens.length);
        }
        double[] v = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            v[i] = Double.parseDouble(tokens[i]);
        }
        return SparseVector.dense(v);
    }

    private SparseVector sparse(String s) {
        String[] tokens = SEPARATORS.split(s);
        int first = tokens.length > 0 && tokens[0].indexOf(':') < 0 ? 1 : 0;
        int n = tokens.length - first;
        int[] idx = new int[n];
        double[] val = new double[n];
        for (int k = 0; k < n; k++) {
            String t = tokens[first + k];
            int colon = t.indexOf(':');
            if (colon < 0) {
                throw new IllegalArgumentException("expected index:value, found " + t);
            }
            idx[k] = Integer.parseInt(t.substring(0, colon));
            val[k] = Double.parseDouble(t.substring(colon + 1));
        }
        return SparseVector.of(sparseDim, idx, val);
    }
}
