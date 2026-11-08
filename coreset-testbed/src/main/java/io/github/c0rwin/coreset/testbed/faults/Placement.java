package io.github.c0rwin.coreset.testbed.faults;

/**
 * Raw data per node for one partition: each node's exact cost on every candidate, its size, its share of the total
 * sensitivity and its count of every reference cluster, all from one pass over the cost table.
 */
final class Placement {

    final int[][] parts;
    final double[][] cost;
    final double[] size;
    final double[] sens;
    final int[][] clusterCount;
    final double totalSize;

    Placement(Instance inst, int[][] parts) {
        this.parts = parts;
        int n = parts.length;
        int labels = inst.clusterSize.length;
        cost = new double[n][inst.candidates()];
        size = new double[n];
        sens = new double[n];
        clusterCount = new int[n][labels];
        double total = 0.0;
        for (int v = 0; v < n; v++) {
            for (int r : parts[v]) {
                double w = inst.points.weight(r);
                inst.table.addRow(r, w, cost[v]);
                size[v] += w;
                sens[v] += inst.sens[r];
                clusterCount[v][inst.cluster[r]]++;
            }
            sens[v] /= inst.sensTotal;
            total += size[v];
        }
        this.totalSize = total;
    }

    int nodes() {
        return parts.length;
    }

    static int count(boolean[] lost) {
        int l = 0;
        for (boolean b : lost) {
            l += b ? 1 : 0;
        }
        return l;
    }

    double[] aliveCost(boolean[] lost) {
        double[] acc = new double[cost[0].length];
        for (int v = 0; v < parts.length; v++) {
            if (!lost[v]) {
                for (int j = 0; j < acc.length; j++) {
                    acc[j] += cost[v][j];
                }
            }
        }
        return acc;
    }

    double aliveSize(boolean[] lost) {
        double s = 0.0;
        for (int v = 0; v < parts.length; v++) {
            s += lost[v] ? 0.0 : size[v];
        }
        return s;
    }

    double lostSens(boolean[] lost) {
        double s = 0.0;
        for (int v = 0; v < parts.length; v++) {
            s += lost[v] ? sens[v] : 0.0;
        }
        return s;
    }

    /** Largest fraction of any reference cluster that sits on lost nodes. */
    double maxClusterLost(boolean[] lost, Instance inst) {
        double worst = 0.0;
        for (int c = 0; c < inst.clusterSize.length; c++) {
            if (inst.clusterSize[c] == 0) {
                continue;
            }
            int l = 0;
            for (int v = 0; v < parts.length; v++) {
                l += lost[v] ? clusterCount[v][c] : 0;
            }
            worst = Math.max(worst, (double) l / inst.clusterSize[c]);
        }
        return worst;
    }

    /** Nodes holding at least {@code share} of some reference cluster. */
    boolean[] holdsWholeCluster(Instance inst, double share) {
        boolean[] out = new boolean[parts.length];
        for (int v = 0; v < parts.length; v++) {
            for (int c = 0; c < inst.clusterSize.length; c++) {
                if (inst.clusterSize[c] > 0 && clusterCount[v][c] >= share * inst.clusterSize[c]) {
                    out[v] = true;
                }
            }
        }
        return out;
    }
}
