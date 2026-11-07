package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

/**
 * Readers for the supported file formats. Every point gets weight 1. Blank lines and lines starting with {@code #}
 * are skipped; malformed input fails with the 1-based line number.
 *
 * <ul>
 *   <li>{@link Format#DENSE}: one point per line, values separated by commas and/or whitespace; an optional header
 *       line names the columns.</li>
 *   <li>{@link Format#LIBSVM}: {@code [label] index:value ...} with 1-based indices; the label is ignored.</li>
 *   <li>{@link Format#WIKI}: the legacy document-term format {@code id dim index value index value ...} with 0-based
 *       indices (as read by the old {@code WikiPointsProvider}).</li>
 * </ul>
 */
public final class Loaders {

    private static final Pattern SEPARATORS = Pattern.compile("[,\\s]+");

    private Loaders() {
    }

    public enum Format {
        DENSE, LIBSVM, WIKI;

        public static Format of(String id) {
            return valueOf(id.trim().toUpperCase(Locale.ROOT));
        }
    }

    /** Whether a dense file starts with a header row. */
    public enum Header {
        /** A header when any field of the first data line does not parse as a number. */
        AUTO, PRESENT, ABSENT
    }

    /**
     * @param header      header handling for dense files
     * @param dropColumns dense columns to remove, by header name or by 0-based index
     * @param dim         libsvm dimension; 0 infers the largest index seen
     */
    public record Options(Header header, List<String> dropColumns, int dim) {

        public Options {
            dropColumns = List.copyOf(dropColumns);
        }

        public static Options defaults() {
            return new Options(Header.AUTO, List.of(), 0);
        }
    }

    /** Opens {@code file}, transparently gunzipping when it starts with the gzip magic bytes. */
    public static BufferedReader open(Path file) throws IOException {
        InputStream in = new BufferedInputStream(Files.newInputStream(file));
        in.mark(2);
        int b0 = in.read();
        int b1 = in.read();
        in.reset();
        if (b0 == 0x1f && b1 == 0x8b) {
            in = new GZIPInputStream(in, 1 << 16);
        }
        return new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8), 1 << 16);
    }

    public static List<WeightedPoint> load(Path file, Format format, Options options) throws IOException {
        try (BufferedReader r = open(file)) {
            return read(r, format, options);
        }
    }

    public static List<WeightedPoint> read(BufferedReader r, Format format, Options options) throws IOException {
        switch (format) {
            case DENSE:
                return dense(r, options);
            case LIBSVM:
                return libsvm(r, options.dim());
            case WIKI:
                return wiki(r);
            default:
                throw new IllegalArgumentException("unsupported format " + format);
        }
    }

    static List<WeightedPoint> dense(BufferedReader r, Options options) throws IOException {
        List<WeightedPoint> out = new ArrayList<>();
        boolean[] keep = null;
        int width = -1;
        int keptWidth = 0;
        double[] row = null;
        String line;
        int lineNo = 0;
        while ((line = r.readLine()) != null) {
            lineNo++;
            String s = line.trim();
            if (s.isEmpty() || s.startsWith("#")) {
                continue;
            }
            String[] fields = SEPARATORS.split(s);
            if (width < 0) {
                width = fields.length;
                boolean header = options.header() == Header.PRESENT
                        || (options.header() == Header.AUTO && !allNumeric(fields));
                keep = keepMask(fields, header, options.dropColumns(), lineNo);
                for (boolean k : keep) {
                    keptWidth += k ? 1 : 0;
                }
                if (keptWidth == 0) {
                    throw new IllegalArgumentException("line " + lineNo + ": every column is dropped");
                }
                row = new double[keptWidth];
                if (header) {
                    continue;
                }
            }
            if (fields.length != width) {
                throw new IllegalArgumentException("line " + lineNo + ": expected " + width + " columns, got "
                        + fields.length);
            }
            for (int c = 0, w = 0; c < width; c++) {
                if (keep[c]) {
                    row[w++] = parse(fields[c], lineNo);
                }
            }
            out.add(new WeightedPoint(SparseVector.dense(row), 1.0));
        }
        return out;
    }

    private static boolean[] keepMask(String[] firstLine, boolean header, List<String> drop, int lineNo) {
        boolean[] keep = new boolean[firstLine.length];
        Arrays.fill(keep, true);
        Set<String> names = header ? new HashSet<>(Arrays.asList(firstLine)) : Set.of();
        for (String d : drop) {
            int idx = -1;
            if (header && names.contains(d)) {
                idx = Arrays.asList(firstLine).indexOf(d);
            } else {
                try {
                    idx = Integer.parseInt(d.trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("line " + lineNo + ": no column named '" + d + "'"
                            + (header ? "" : " (file has no header)"));
                }
            }
            if (idx < 0 || idx >= keep.length) {
                throw new IllegalArgumentException("line " + lineNo + ": column index " + idx + " out of range");
            }
            keep[idx] = false;
        }
        return keep;
    }

    private static boolean allNumeric(String[] fields) {
        for (String f : fields) {
            try {
                Double.parseDouble(f);
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }

    static List<WeightedPoint> libsvm(BufferedReader r, int dim) throws IOException {
        List<int[]> indices = new ArrayList<>();
        List<double[]> values = new ArrayList<>();
        int maxIndex = 0;
        String line;
        int lineNo = 0;
        while ((line = r.readLine()) != null) {
            lineNo++;
            String s = line.trim();
            if (s.isEmpty() || s.startsWith("#")) {
                continue;
            }
            String[] tokens = s.split("\\s+");
            int first = tokens[0].indexOf(':') < 0 ? 1 : 0; // leading label is optional
            int nnz = tokens.length - first;
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int t = 0; t < nnz; t++) {
                String tok = tokens[first + t];
                int colon = tok.indexOf(':');
                if (colon <= 0) {
                    throw new IllegalArgumentException("line " + lineNo + ": expected index:value, got '" + tok + "'");
                }
                int i = parseIndex(tok.substring(0, colon), lineNo);
                if (i < 1) {
                    throw new IllegalArgumentException("line " + lineNo + ": libsvm indices are 1-based, got " + i);
                }
                idx[t] = i - 1;
                val[t] = parse(tok.substring(colon + 1), lineNo);
                maxIndex = Math.max(maxIndex, i);
            }
            requireDistinct(idx, lineNo);
            indices.add(idx);
            values.add(val);
        }
        if (dim > 0 && maxIndex > dim) {
            throw new IllegalArgumentException("index " + maxIndex + " exceeds dim " + dim);
        }
        int d = dim > 0 ? dim : maxIndex;
        List<WeightedPoint> out = new ArrayList<>(indices.size());
        for (int i = 0; i < indices.size(); i++) {
            out.add(new WeightedPoint(SparseVector.of(d, indices.get(i), values.get(i)), 1.0));
        }
        return out;
    }

    static List<WeightedPoint> wiki(BufferedReader r) throws IOException {
        List<WeightedPoint> out = new ArrayList<>();
        int dim = -1;
        String line;
        int lineNo = 0;
        while ((line = r.readLine()) != null) {
            lineNo++;
            String s = line.trim();
            if (s.isEmpty() || s.startsWith("#")) {
                continue;
            }
            String[] tokens = s.split("\\s+");
            if (tokens.length < 2 || tokens.length % 2 != 0) {
                throw new IllegalArgumentException("line " + lineNo + ": expected 'id dim (index value)*'");
            }
            int d = parseIndex(tokens[1], lineNo);
            if (dim >= 0 && d != dim) {
                throw new IllegalArgumentException("line " + lineNo + ": dim " + d + " differs from " + dim);
            }
            dim = d;
            int nnz = (tokens.length - 2) / 2;
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int t = 0; t < nnz; t++) {
                idx[t] = parseIndex(tokens[2 + 2 * t], lineNo);
                val[t] = parse(tokens[3 + 2 * t], lineNo);
                if (idx[t] < 0 || idx[t] >= d) {
                    throw new IllegalArgumentException("line " + lineNo + ": index " + idx[t] + " outside [0, " + d
                            + ")");
                }
            }
            requireDistinct(idx, lineNo);
            out.add(new WeightedPoint(SparseVector.of(d, idx, val), 1.0));
        }
        return out;
    }

    private static void requireDistinct(int[] idx, int lineNo) {
        int[] sorted = idx.clone();
        Arrays.sort(sorted);
        for (int t = 1; t < sorted.length; t++) {
            if (sorted[t] == sorted[t - 1]) {
                throw new IllegalArgumentException("line " + lineNo + ": duplicate index " + sorted[t]);
            }
        }
    }

    private static double parse(String s, int lineNo) {
        try {
            double v = Double.parseDouble(s);
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("line " + lineNo + ": value is not finite: " + s);
            }
            return v;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("line " + lineNo + ": not a number: '" + s + "'", e);
        }
    }

    private static int parseIndex(String s, int lineNo) {
        try {
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("line " + lineNo + ": not an integer: '" + s + "'", e);
        }
    }
}
