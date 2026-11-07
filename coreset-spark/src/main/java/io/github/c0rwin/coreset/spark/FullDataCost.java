package io.github.c0rwin.coreset.spark;

/**
 * One distributed pass over the full data for fixed centers.
 *
 * @param cost   {@code sum_p w_p * min_c ||p - c||^2}, exact (support-merging) distances
 * @param weight total input weight
 * @param count  input points
 */
public record FullDataCost(double cost, double weight, long count) {
}
