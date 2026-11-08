package io.github.c0rwin.coreset.core.validate;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import java.util.ArrayList;
import java.util.List;

/**
 * Empirical distortion of a coreset C against the full input P over candidate center sets: per candidate the ratio
 * {@code r = cost(C, Q) / cost(P, Q)} (offsets included) and distortion {@code max(r, 1/r)}. The maximum is a lower
 * bound on the true distortion, which is a supremum over all Q.
 *
 * <p>Costs {@code O(|candidates| * (|P| + |C|) * k)} distance evaluations.
 */
public final class Distortion {

    private Distortion() {
    }

    /**
     * @param ratio      {@code cost(C, Q) / cost(P, Q)}; 1 when both are 0 and +infinity when only {@code cost(P, Q)}
     *                   is 0
     * @param distortion {@code max(ratio, 1 / ratio)}, +infinity for a ratio of 0 or +infinity
     */
    public record Entry(String generator, double costFull, double costCoreset, double ratio, double distortion) {
    }

    /**
     * @param max              largest distortion
     * @param mean             arithmetic mean of the distortions
     * @param argmaxGenerator  generator of the first candidate reaching {@code max}
     */
    public record Report(List<Entry> entries, double max, double mean, String argmaxGenerator) {

        public Report {
            entries = List.copyOf(entries);
        }
    }

    /** @throws IllegalArgumentException if {@code candidates} is empty */
    public static Report evaluate(Coreset full, Coreset coreset, List<Candidate> candidates) {
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException("candidates must not be empty");
        }
        List<Entry> entries = new ArrayList<>(candidates.size());
        double max = 0.0;
        double sum = 0.0;
        String argmax = null;
        for (Candidate c : candidates) {
            double cp = Cost.cost(full, c.centers());
            double cc = Cost.cost(coreset, c.centers());
            double ratio = cp > 0.0 ? cc / cp : (cc == 0.0 ? 1.0 : Double.POSITIVE_INFINITY);
            double distortion = ratio >= 1.0 ? ratio : 1.0 / ratio;
            entries.add(new Entry(c.generator(), cp, cc, ratio, distortion));
            sum += distortion;
            if (argmax == null || distortion > max) {
                max = distortion;
                argmax = c.generator();
            }
        }
        return new Report(entries, max, sum / candidates.size(), argmax);
    }
}
