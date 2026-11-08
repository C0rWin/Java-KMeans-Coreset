package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FilterInputStream;
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
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

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
 *   <li>{@link Format#BOW}: the UCI bag-of-words {@code docword} format: the number of documents, words and
 *       entries on the first three lines, then {@code docID wordID count} lines with 1-based ids, grouped by document.
 *       Documents without an entry become empty rows, so row {@code i} is always document {@code i + 1}.</li>
 * </ul>
 *
 * <p>The {@code stream*} methods hand each row to a sink instead of building points, for converters that must not
 * hold a multi-GB input in memory.
 */
public final class Loaders {

    private static final Pattern SEPARATORS = Pattern.compile("[,\\s]+");

    private Loaders() {
    }

    public enum Format {
        DENSE, LIBSVM, WIKI, BOW;

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

    /** Receives dense rows; {@code values} is reused for the next row. */
    @FunctionalInterface
    public interface DenseRowSink {
        void row(double[] values) throws IOException;
    }

    /** Receives sparse rows as entries {@code [from, to)} of reused arrays, indices strictly increasing. */
    @FunctionalInterface
    public interface SparseRowSink {
        void row(int[] indices, double[] values, int from, int to) throws IOException;
    }

    /** Header of a bag-of-words file. */
    public record BagOfWordsShape(int docs, int words, long nnz) {
    }

    /** Opens {@code file}, transparently gunzipping when it starts with the gzip magic bytes. */
    public static BufferedReader open(Path file) throws IOException {
        return open(file, "");
    }

    /**
     * Opens {@code file}, or its zip entry {@code member} when that is not empty, transparently gunzipping when the
     * content starts with the gzip magic bytes.
     */
    public static BufferedReader open(Path file, String member) throws IOException {
        InputStream raw;
        if (member.isEmpty()) {
            raw = Files.newInputStream(file);
        } else {
            ZipFile zip = new ZipFile(file.toFile());
            ZipEntry entry = zip.getEntry(member);
            if (entry == null) {
                zip.close();
                throw new IOException(file + ": no zip entry '" + member + "'");
            }
            raw = new FilterInputStream(zip.getInputStream(entry)) {
                @Override
                public void close() throws IOException {
                    try {
                        super.close();
                    } finally {
                        zip.close();
                    }
                }
            };
        }
        InputStream in = new BufferedInputStream(raw, 1 << 16);
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
            case BOW:
                return bagOfWords(r);
            default:
                throw new IllegalArgumentException("unsupported format " + format);
        }
    }

    static List<WeightedPoint> dense(BufferedReader r, Options options) throws IOException {
        List<WeightedPoint> out = new ArrayList<>();
        streamDense(r, options, row -> out.add(new WeightedPoint(SparseVector.dense(row), 1.0)));
        return out;
    }

    /** Streams the rows of a dense file (see {@link Format#DENSE}) after dropping columns. */
    public static void streamDense(BufferedReader r, Options options, DenseRowSink sink) throws IOException {
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
            sink.row(row);
        }
    }

    static List<WeightedPoint> bagOfWords(BufferedReader r) throws IOException {
        BagOfWordsShape shape = bagOfWordsShape(r);
        List<WeightedPoint> out = new ArrayList<>(shape.docs());
        streamBagOfWords(r, shape, (idx, val, from, to) -> out.add(new WeightedPoint(
                SparseVector.of(shape.words(), Arrays.copyOfRange(idx, from, to), Arrays.copyOfRange(val, from, to)),
                1.0)));
        return out;
    }

    /** Reads the three header lines of a bag-of-words file; call before {@link #streamBagOfWords}. */
    public static BagOfWordsShape bagOfWordsShape(BufferedReader r) throws IOException {
        long[] h = new long[3];
        for (int i = 0; i < 3; i++) {
            String line = r.readLine();
            if (line == null) {
                throw new IllegalArgumentException("line " + (i + 1) + ": bag-of-words header ends early");
            }
            try {
                h[i] = Long.parseLong(line.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("line " + (i + 1) + ": not an integer: '" + line + "'", e);
            }
        }
        if (h[0] < 0 || h[0] > Integer.MAX_VALUE || h[1] < 1 || h[1] > Integer.MAX_VALUE || h[2] < 0) {
            throw new IllegalArgumentException("bag-of-words header out of range: " + Arrays.toString(h));
        }
        return new BagOfWordsShape((int) h[0], (int) h[1], h[2]);
    }

    /**
     * Streams one row per document {@code 1..docs} (word ids made 0-based, counts as values) from the lines after
     * the header. Counts must be positive integers; the entry total must match the header.
     */
    public static void streamBagOfWords(BufferedReader r, BagOfWordsShape shape, SparseRowSink sink)
            throws IOException {
        int[] idx = new int[1024];
        double[] val = new double[1024];
        int[] tok = new int[6];
        int doc = 1; // the document being collected
        int len = 0;
        boolean sorted = true;
        long entries = 0;
        String line;
        int lineNo = 3;
        while ((line = r.readLine()) != null) {
            lineNo++;
            if (tokens(line, tok) == 0) {
                continue;
            }
            int d = parseIndex(line, tok[0], tok[1], lineNo);
            int w = parseIndex(line, tok[2], tok[3], lineNo);
            int c = parseIndex(line, tok[4], tok[5], lineNo);
            if (d < doc || d > shape.docs()) {
                throw new IllegalArgumentException("line " + lineNo + ": document " + d + " out of order or range");
            }
            if (w < 1 || w > shape.words() || c < 1) {
                throw new IllegalArgumentException("line " + lineNo + ": word " + w + " count " + c
                        + " out of range");
            }
            for (; doc < d; doc++, len = 0, sorted = true) {
                emitRow(idx, val, len, sorted, lineNo, sink);
            }
            if (len == idx.length) {
                idx = Arrays.copyOf(idx, 2 * len);
                val = Arrays.copyOf(val, 2 * len);
            }
            sorted &= len == 0 || idx[len - 1] < w - 1;
            idx[len] = w - 1;
            val[len++] = c;
            entries++;
        }
        for (; doc <= shape.docs(); doc++, len = 0, sorted = true) {
            emitRow(idx, val, len, sorted, lineNo, sink);
        }
        if (entries != shape.nnz()) {
            throw new IllegalArgumentException("header says " + shape.nnz() + " entries, read " + entries);
        }
    }

    private static void emitRow(int[] idx, double[] val, int len, boolean sorted, int lineNo, SparseRowSink sink)
            throws IOException {
        if (!sorted) {
            sortByIndex(idx, val, len);
            for (int t = 1; t < len; t++) {
                if (idx[t] == idx[t - 1]) {
                    throw new IllegalArgumentException("line " + lineNo + ": duplicate word " + (idx[t] + 1));
                }
            }
        }
        sink.row(idx, val, 0, len);
    }

    private static void sortByIndex(int[] idx, double[] val, int len) {
        for (int i = 1; i < len; i++) { // insertion sort: unsorted documents are rare and short
            int k = idx[i];
            double v = val[i];
            int j = i - 1;
            for (; j >= 0 && idx[j] > k; j--) {
                idx[j + 1] = idx[j];
                val[j + 1] = val[j];
            }
            idx[j + 1] = k;
            val[j + 1] = v;
        }
    }

    /** Splits {@code s} at whitespace into exactly three tokens (begin, end pairs); 0 for a blank line. */
    private static int tokens(String s, int[] tok) {
        int n = 0;
        int i = 0;
        int len = s.length();
        while (true) {
            while (i < len && s.charAt(i) <= ' ') {
                i++;
            }
            if (i == len) {
                break;
            }
            if (n == 3) {
                throw new IllegalArgumentException("expected 'docID wordID count', got '" + s + "'");
            }
            tok[2 * n] = i;
            while (i < len && s.charAt(i) > ' ') {
                i++;
            }
            tok[2 * n + 1] = i;
            n++;
        }
        if (n != 0 && n != 3) {
            throw new IllegalArgumentException("expected 'docID wordID count', got '" + s + "'");
        }
        return n;
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
        return parseIndex(s, 0, s.length(), lineNo);
    }

    private static int parseIndex(String s, int begin, int end, int lineNo) {
        try {
            return Integer.parseInt(s, begin, end, 10);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("line " + lineNo + ": not an integer: '" + s.substring(begin, end)
                    + "'", e);
        }
    }
}
