package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/** Seeded generators of synthetic inputs; equal arguments always give equal points in equal order. */
public final class SyntheticData {

    private SyntheticData() {
    }

    /**
     * Cluster sizes {@code n * p_i} with {@code p_i ~ (i+1)^-imbalance} (Zipf; 0 is balanced), rounded so they sum to
     * {@code n} and every cluster keeps at least one point.
     */
    static int[] zipfSizes(int n, int clusters, double imbalance) {
        if (clusters < 1 || n < clusters) {
            throw new IllegalArgumentException("need 1 <= clusters <= n: clusters=" + clusters + ", n=" + n);
        }
        double[] p = new double[clusters];
        double total = 0.0;
        for (int i = 0; i < clusters; i++) {
            p[i] = Math.pow(i + 1.0, -imbalance);
            total += p[i];
        }
        int[] sizes = new int[clusters];
        int assigned = 0;
        for (int i = 0; i < clusters; i++) {
            sizes[i] = Math.max(1, (int) Math.floor(n * p[i] / total));
            assigned += sizes[i];
        }
        for (int i = 0; assigned < n; i = (i + 1) % clusters) {
            sizes[i]++;
            assigned++;
        }
        for (int i = clusters - 1; assigned > n; i = (i + clusters - 1) % clusters) {
            if (sizes[i] > 1) {
                sizes[i]--;
                assigned--;
            }
        }
        return sizes;
    }

    /** Labels {@code 0..sizes.length-1} with the given multiplicities, in seeded random order. */
    static int[] shuffledLabels(int[] sizes, SplittableRandom rnd) {
        int n = Arrays.stream(sizes).sum();
        int[] labels = new int[n];
        for (int c = 0, w = 0; c < sizes.length; c++) {
            for (int t = 0; t < sizes[c]; t++) {
                labels[w++] = c;
            }
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = labels[i];
            labels[i] = labels[j];
            labels[j] = tmp;
        }
        return labels;
    }

    static double gaussian(SplittableRandom rnd) {
        double u = 1.0 - rnd.nextDouble();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * rnd.nextDouble());
    }

    /**
     * Isotropic Gaussian mixture: centers uniform in {@code [0, box]^d}, standard deviation {@code sigma}, cluster
     * sizes from {@link #zipfSizes}, points in random order (so stream blocks are mixed).
     */
    public static List<WeightedPoint> gaussianMixture(int n, int d, int clusters, double sigma, double imbalance,
                                                      double box, long seed) {
        if (d < 1 || !(sigma >= 0.0) || !(box >= 0.0) || !(imbalance >= 0.0)) {
            throw new IllegalArgumentException("gaussian-mixture needs d >= 1, sigma >= 0, box >= 0, imbalance >= 0");
        }
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] means = new double[clusters][d];
        for (double[] m : means) {
            for (int t = 0; t < d; t++) {
                m[t] = box * rnd.nextDouble();
            }
        }
        int[] labels = shuffledLabels(zipfSizes(n, clusters, imbalance), rnd);
        List<WeightedPoint> out = new ArrayList<>(n);
        double[] x = new double[d];
        for (int i = 0; i < n; i++) {
            double[] m = means[labels[i]];
            for (int t = 0; t < d; t++) {
                x[t] = m[t] + sigma * gaussian(rnd);
            }
            out.add(new WeightedPoint(SparseVector.dense(x), 1.0));
        }
        return out;
    }

    /**
     * Bag-of-words-like documents over a {@code d}-term vocabulary, L2-normalised term counts. Each of {@code topics}
     * topics owns {@code topicVocabulary} random terms with Zipf(1) frequencies; a document picks a topic (Zipf sizes
     * with exponent {@code imbalance}), a length uniform in {@code [docLength/2, 3 docLength/2]}, and draws each token
     * from its topic with probability {@code topicMix}, otherwise from a Zipf(1.1) background over the vocabulary.
     */
    public static List<WeightedPoint> sparseBagOfWords(int n, int d, int topics, int topicVocabulary, int docLength,
                                                       double topicMix, double imbalance, long seed) {
        if (d < 2 || topicVocabulary < 1 || topicVocabulary > d || docLength < 1 || !(topicMix >= 0.0
                && topicMix <= 1.0)) {
            throw new IllegalArgumentException("sparse-bow needs d >= 2, 1 <= topicVocabulary <= d, docLength >= 1, "
                    + "topicMix in [0, 1]");
        }
        SplittableRandom rnd = new SplittableRandom(seed);
        double[] background = zipfCumulative(d, 1.1);
        double[] topicCdf = zipfCumulative(topicVocabulary, 1.0);
        int[][] vocab = new int[topics][];
        int[] perm = new int[d];
        for (int t = 0; t < d; t++) {
            perm[t] = t;
        }
        for (int topic = 0; topic < topics; topic++) {
            for (int t = 0; t < topicVocabulary; t++) { // partial Fisher-Yates: first topicVocabulary of a shuffle
                int j = t + rnd.nextInt(d - t);
                int tmp = perm[t];
                perm[t] = perm[j];
                perm[j] = tmp;
            }
            vocab[topic] = Arrays.copyOf(perm, topicVocabulary);
        }
        int[] labels = shuffledLabels(zipfSizes(n, topics, imbalance), rnd);
        List<WeightedPoint> out = new ArrayList<>(n);
        int maxLen = docLength + docLength / 2 + 1;
        int[] terms = new int[maxLen];
        for (int i = 0; i < n; i++) {
            int len = Math.max(1, docLength / 2 + rnd.nextInt(docLength + 1));
            int[] own = vocab[labels[i]];
            for (int t = 0; t < len; t++) {
                terms[t] = rnd.nextDouble() < topicMix ? own[draw(topicCdf, rnd)] : draw(background, rnd);
            }
            Arrays.sort(terms, 0, len);
            int distinct = 0;
            for (int t = 0; t < len; t++) {
                if (t == 0 || terms[t] != terms[t - 1]) {
                    distinct++;
                }
            }
            int[] idx = new int[distinct];
            double[] val = new double[distinct];
            double norm = 0.0;
            for (int t = 0, w = -1; t < len; t++) {
                if (t == 0 || terms[t] != terms[t - 1]) {
                    idx[++w] = terms[t];
                }
                val[w] += 1.0;
            }
            for (double v : val) {
                norm += v * v;
            }
            double inv = 1.0 / Math.sqrt(norm);
            for (int t = 0; t < distinct; t++) {
                val[t] *= inv;
            }
            out.add(new WeightedPoint(SparseVector.of(d, idx, val), 1.0));
        }
        return out;
    }

    private static double[] zipfCumulative(int size, double exponent) {
        double[] cdf = new double[size];
        double acc = 0.0;
        for (int r = 0; r < size; r++) {
            acc += Math.pow(r + 1.0, -exponent);
            cdf[r] = acc;
        }
        for (int r = 0; r < size; r++) {
            cdf[r] /= acc;
        }
        return cdf;
    }

    private static int draw(double[] cdf, SplittableRandom rnd) {
        int i = Arrays.binarySearch(cdf, rnd.nextDouble());
        int r = i >= 0 ? i + 1 : -i - 1; // first index with cdf > u
        return Math.min(r, cdf.length - 1);
    }
}
