package io.github.c0rwin.coreset.testbed.faults;

import java.util.Locale;
import java.util.random.RandomGenerator;

/** Node-level loss models and the decoders that rescale the survivors. */
final class NodeFaults {

    private NodeFaults() {
    }

    /** A fixed lost fraction {@code phi} of the nodes, or i.i.d. failures with probability {@code f}. */
    record Loss(String id, double phi, double f) {

        static Loss none() {
            return new Loss("none", 0.0, Double.NaN);
        }

        static Loss fixed(double phi) {
            return new Loss(String.format(Locale.ROOT, "phi=%s", Channel.trim(phi)), phi, Double.NaN);
        }

        static Loss iid(double f) {
            return new Loss(String.format(Locale.ROOT, "f=%s", Channel.trim(f)), Double.NaN, f);
        }

        boolean isFixed() {
            return !Double.isNaN(phi);
        }

        /** The failure probability a decoder knows in advance. */
        double prior() {
            return isFixed() ? phi : f;
        }

        /** Lost flags for {@code n} nodes, chosen independently of any data placement. */
        boolean[] realize(int n, RandomGenerator rng) {
            boolean[] lost = new boolean[n];
            if (isFixed()) {
                int l = (int) Math.round(phi * n);
                int[] p = Partitions.identity(n);
                for (int i = 0; i < l; i++) {
                    int j = i + rng.nextInt(n - i);
                    int t = p[i];
                    p[i] = p[j];
                    p[j] = t;
                    lost[p[i]] = true;
                }
            } else {
                for (int v = 0; v < n; v++) {
                    lost[v] = rng.nextDouble() < f;
                }
            }
            return lost;
        }
    }

    enum Decoder {
        /** Survivors as sent. */
        NONE("none"),
        /** {@code 1 / (1 - p)} with the prior failure rate. */
        PRIOR("ht-prior"),
        /** {@code n / (n - l)} with the realized count of lost nodes. */
        REALIZED("ht-realized"),
        /** {@code N / sum of surviving part sizes}: realized counts weighted by part size. */
        SIZE("ht-size");

        final String id;

        Decoder(String id) {
            this.id = id;
        }

        double scale(Loss loss, int nodes, int lost, double totalSize, double aliveSize) {
            if (lost == nodes || aliveSize == 0.0) {
                return 0.0;
            }
            return switch (this) {
                case NONE -> 1.0;
                case PRIOR -> 1.0 / (1.0 - loss.prior());
                case REALIZED -> (double) nodes / (nodes - lost);
                case SIZE -> totalSize / aliveSize;
            };
        }
    }
}
