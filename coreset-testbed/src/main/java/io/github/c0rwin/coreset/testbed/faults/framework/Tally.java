package io.github.c0rwin.coreset.testbed.faults.framework;

import java.util.Arrays;

/**
 * A received message as per-class sent counts, received counts and received scale-free cost sums; RC, HT(p0),
 * known-p HT and per-class RC are closed forms of these. An empty RC output has cost 0.
 */
final class Tally {

    private final CostSpace space;
    private final int queries;
    private final double[] sum;
    private final int[] sent;
    private final int[] got;
    private int m;
    private int received;

    Tally(CostSpace space) {
        this.space = space;
        this.queries = space.queries;
        this.sum = new double[space.classes * queries];
        this.sent = new int[space.classes];
        this.got = new int[space.classes];
    }

    void reset() {
        Arrays.fill(sum, 0.0);
        Arrays.fill(sent, 0);
        Arrays.fill(got, 0);
        m = 0;
        received = 0;
    }

    /** Adds a sent draw of member {@code t}; only a delivered one contributes cost. */
    void add(int t, boolean delivered) {
        int c = space.cls[t];
        sent[c]++;
        m++;
        if (delivered) {
            got[c]++;
            received++;
            int base = c * queries;
            int gb = t * queries;
            for (int j = 0; j < queries; j++) {
                sum[base + j] += space.g[gb + j];
            }
        }
    }

    int sent() {
        return m;
    }

    int received() {
        return received;
    }

    int received(int c) {
        return got[c];
    }

    private double total(int j) {
        double s = 0.0;
        for (int c = 0; c < sent.length; c++) {
            s += sum[c * queries + j];
        }
        return s;
    }

    double rc(int j) {
        return received == 0 ? 0.0 : total(j) / received;
    }

    double ht(int j, double p0) {
        return total(j) / (m * p0);
    }

    double known(int j, double[] pByClass) {
        double s = 0.0;
        for (int c = 0; c < sent.length; c++) {
            s += sum[c * queries + j] / pByClass[c];
        }
        return s / m;
    }

    double perClass(int j) {
        double s = 0.0;
        for (int c = 0; c < sent.length; c++) {
            if (got[c] > 0) {
                s += sum[c * queries + j] * sent[c] / got[c];
            }
        }
        return s / m;
    }

    /** {@code max_j |RC_j / cost_j - 1|}. */
    double rcMaxAbs() {
        double worst = 0.0;
        for (int j = 0; j < queries; j++) {
            worst = Math.max(worst, Math.abs(space.rel(rc(j), j)));
        }
        return worst;
    }
}
