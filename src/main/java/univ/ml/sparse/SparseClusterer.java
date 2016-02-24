package univ.ml.sparse;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;

import java.util.Collection;
import java.util.List;

public interface SparseClusterer {


    List<SparseCentroidCluster> cluster(Collection<SparseWeightableVector> points)
            throws MathIllegalArgumentException, ConvergenceException;


}
