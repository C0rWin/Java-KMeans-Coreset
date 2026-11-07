package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;

/**
 * Immutable coreset tagged with what it summarises; the unit stored in CAS tree slots.
 *
 * @param batches     ingested batches folded into {@code coreset}
 * @param inputWeight total input weight of those batches
 */
record Summary(Coreset coreset, long batches, double inputWeight) {
}
