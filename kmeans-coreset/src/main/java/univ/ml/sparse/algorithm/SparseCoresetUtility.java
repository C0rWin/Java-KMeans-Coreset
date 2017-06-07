/*
 * BEGIN_COPYRIGHT
 *
 *   IBM Confidential
 *   OCO Source Materials
 *
 *   5727-I17
 *   (C) Copyright IBM Corp. 2011, 2016 All Rights Reserved.
 *
 *   The source code for this program is not published or otherwise
 *   divested of its trade secrets, irrespective of what has been
 *   deposited with the U.S. Copyright Office.
 *
 *  END_COPYRIGHT
 *
 */

package univ.ml.sparse.algorithm;

import java.util.List;

import org.apache.commons.math3.util.FastMath;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

public class SparseCoresetUtility {

    public static double getEnergy(final List<SparseCentroidCluster> centers, final List<SparseWeightableVector> points) {
        double result = 0d;
        for (final SparseWeightableVector point : points) {
            double minDist = Double.MAX_VALUE;
            for (final SparseCentroidCluster center : centers) {
                double dist = FastMath.pow(point.getDistance(center.getCenter().getVector()), 2);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            result += minDist;
        }
        return result;
    }
}
