package io.github.c0rwin.coreset.experiments.run;

import java.util.ArrayList;
import java.util.List;

/** Median and range of repetition values; NaN entries (not applicable) are ignored. */
final class Stats {

    private Stats() {
    }

    static List<Double> finite(List<Double> xs) {
        List<Double> out = new ArrayList<>(xs.size());
        for (Double x : xs) {
            if (x != null && !x.isNaN()) {
                out.add(x);
            }
        }
        out.sort(Double::compare);
        return out;
    }

    static double median(List<Double> xs) {
        List<Double> s = finite(xs);
        int n = s.size();
        if (n == 0) {
            return Double.NaN;
        }
        return n % 2 == 1 ? s.get(n / 2) : (s.get(n / 2 - 1) + s.get(n / 2)) / 2.0;
    }

    static double min(List<Double> xs) {
        List<Double> s = finite(xs);
        return s.isEmpty() ? Double.NaN : s.get(0);
    }

    static double max(List<Double> xs) {
        List<Double> s = finite(xs);
        return s.isEmpty() ? Double.NaN : s.get(s.size() - 1);
    }
}
