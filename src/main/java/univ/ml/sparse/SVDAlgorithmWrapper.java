package univ.ml.sparse;

import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.util.ArrayList;
import java.util.List;

public class SVDAlgorithmWrapper {

    private SingularValueDecomposition svd;

    public SVDAlgorithmWrapper(List<SparseWeightableVector> matrix) {

        int n = matrix.size();
        int d = matrix.get(0).getDimension();

        final OpenMapRealMatrix realMatrix = new OpenMapRealMatrix(n, d);

        int idx = 0;
        for (SparseWeightableVector vector : matrix) {
            realMatrix.setRowVector(idx++, vector);
        }

        this.svd = new SingularValueDecomposition(realMatrix);
    }

    public List<SparseWeightableVector> getVT() {
        return convertMatrixToSparseList(svd.getVT());
    }

    private List<SparseWeightableVector> convertMatrixToSparseList(RealMatrix matrix) {
        List<SparseWeightableVector> result = new ArrayList<>();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            result.add(new SparseWeightableVector(matrix.getRowVector(i)));
        }
        return result;
    }

    public List<SparseWeightableVector> getV() {
        return convertMatrixToSparseList(svd.getV());
    }

    public double[] getSingularValues() {
        return svd.getSingularValues();
    }

    public List<SparseWeightableVector> getS() {
        return convertMatrixToSparseList(svd.getS());
    }

    public List<SparseWeightableVector> getUT() {
        return convertMatrixToSparseList(svd.getUT());
    }

    public List<SparseWeightableVector> getU() {
        return convertMatrixToSparseList(svd.getU());
    }
}
