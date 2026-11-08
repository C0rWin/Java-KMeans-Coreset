package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.Serializable;
import java.util.Objects;
import org.apache.spark.ml.linalg.SQLDataTypes;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.util.ArrayData;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Which columns of a {@code Dataset<Row>} hold the points. Either a sparse triple ({@code array<int>} indices,
 * {@code array<double>} values, integer dimension) or one dense/ML vector column ({@code array<double>} or Spark ML
 * {@code Vector}, dense or sparse), each with an optional numeric weight column (default weight 1).
 *
 * <p>Rows are read in Spark's internal format, so array elements are copied out as primitive arrays without boxing
 * and no external {@code Row} is materialised.
 */
public final class PointColumns implements Serializable {

    private static final long serialVersionUID = 1L;

    private enum Kind { SPARSE, VECTOR }

    private final Kind kind;
    private final String indices;
    private final String values;
    private final String dim;
    private final String vector;
    private final String weight;

    private PointColumns(Kind kind, String indices, String values, String dim, String vector, String weight) {
        this.kind = kind;
        this.indices = indices;
        this.values = values;
        this.dim = dim;
        this.vector = vector;
        this.weight = weight;
    }

    public static PointColumns sparse(String indicesCol, String valuesCol, String dimCol) {
        return new PointColumns(Kind.SPARSE, Objects.requireNonNull(indicesCol, "indicesCol"),
                Objects.requireNonNull(valuesCol, "valuesCol"), Objects.requireNonNull(dimCol, "dimCol"), null, null);
    }

    public static PointColumns vector(String vectorCol) {
        return new PointColumns(Kind.VECTOR, null, null, null, Objects.requireNonNull(vectorCol, "vectorCol"), null);
    }

    public PointColumns withWeight(String weightCol) {
        return new PointColumns(kind, indices, values, dim, vector, Objects.requireNonNull(weightCol, "weightCol"));
    }

    /** Resolves column positions and type-specific readers once on the driver. */
    RowDecoder decoder(StructType schema) {
        int w = -1;
        Numeric wType = null;
        if (weight != null) {
            w = schema.fieldIndex(weight);
            wType = Numeric.of(schema.fields()[w]);
        }
        if (kind == Kind.SPARSE) {
            int i = schema.fieldIndex(indices);
            int v = schema.fieldIndex(values);
            int d = schema.fieldIndex(dim);
            StructField df = schema.fields()[d];
            require(df, df.dataType().equals(DataTypes.IntegerType) || df.dataType().equals(DataTypes.LongType),
                    "int or bigint");
            return new RowDecoder(Layout.SPARSE, i, nullableArrayOf(schema.fields()[i], DataTypes.IntegerType,
                    "array<int>"), v, nullableArrayOf(schema.fields()[v], DataTypes.DoubleType, "array<double>"), d,
                    Numeric.of(df), w, wType);
        }
        int c = schema.fieldIndex(vector);
        StructField f = schema.fields()[c];
        if (SQLDataTypes.VectorType().equals(f.dataType())) {
            return new RowDecoder(Layout.ML_VECTOR, c, false, -1, false, -1, null, w, wType);
        }
        boolean nullable = nullableArrayOf(f, DataTypes.DoubleType, "array<double> or an ML vector");
        return new RowDecoder(Layout.DENSE_ARRAY, c, nullable, -1, false, -1, null, w, wType);
    }

    /** Checks the element type and returns whether the array may hold nulls. */
    private static boolean nullableArrayOf(StructField f, DataType element, String expected) {
        boolean ok = f.dataType() instanceof ArrayType a && a.elementType().equals(element);
        require(f, ok, expected);
        return ((ArrayType) f.dataType()).containsNull();
    }

    private static void require(StructField f, boolean ok, String expected) {
        if (!ok) {
            throw new IllegalArgumentException(
                    "column '" + f.name() + "' has type " + f.dataType().simpleString() + ", expected " + expected);
        }
    }

    @Override
    public String toString() {
        String w = weight == null ? "" : ", weight=" + weight;
        return kind == Kind.SPARSE
                ? "PointColumns[indices=" + indices + ", values=" + values + ", dim=" + dim + w + "]"
                : "PointColumns[vector=" + vector + w + "]";
    }

    private enum Layout { SPARSE, DENSE_ARRAY, ML_VECTOR }

    /** Reader for one numeric column type, resolved on the driver to a switch code. */
    private record Numeric(int code, int precision, int scale) implements Serializable {

        private static final DataType[] CODES = {DataTypes.DoubleType, DataTypes.LongType, DataTypes.IntegerType,
            DataTypes.FloatType, DataTypes.ShortType, DataTypes.ByteType};
        private static final int DECIMAL = CODES.length;

        static Numeric of(StructField f) {
            DataType t = f.dataType();
            if (t instanceof DecimalType d) {
                return new Numeric(DECIMAL, d.precision(), d.scale());
            }
            for (int c = 0; c < CODES.length; c++) {
                if (CODES[c].equals(t)) {
                    return new Numeric(c, 0, 0);
                }
            }
            require(f, false, "a numeric type");
            throw new AssertionError();
        }

        double read(InternalRow row, int ordinal) {
            return switch (code) {
                case 0 -> row.getDouble(ordinal);
                case 1 -> row.getLong(ordinal);
                case 2 -> row.getInt(ordinal);
                case 3 -> row.getFloat(ordinal);
                case 4 -> row.getShort(ordinal);
                case 5 -> row.getByte(ordinal);
                default -> row.getDecimal(ordinal, precision, scale).toDouble();
            };
        }
    }

    /** Internal-row to point conversion with resolved positions; runs on executors. */
    static final class RowDecoder implements Serializable {

        private static final long serialVersionUID = 1L;
        /** Field order of the ML vector UDT's storage struct: type (0 sparse, 1 dense), size, indices, values. */
        private static final int UDT_FIELDS = 4;

        private final Layout layout;
        private final int first;
        private final boolean firstNullable;
        private final int values;
        private final boolean valuesNullable;
        private final int dim;
        private final Numeric dimType;
        private final int weight;
        private final Numeric weightType;

        private RowDecoder(Layout layout, int first, boolean firstNullable, int values, boolean valuesNullable,
                           int dim, Numeric dimType, int weight, Numeric weightType) {
            this.layout = layout;
            this.first = first;
            this.firstNullable = firstNullable;
            this.values = values;
            this.valuesNullable = valuesNullable;
            this.dim = dim;
            this.dimType = dimType;
            this.weight = weight;
            this.weightType = weightType;
        }

        WeightedPoint decode(InternalRow row) {
            double w = 1.0;
            if (weight >= 0) {
                if (row.isNullAt(weight)) {
                    throw new IllegalArgumentException("null weight in column " + weight);
                }
                w = weightType.read(row, weight);
            }
            if (row.isNullAt(first) || (layout == Layout.SPARSE && (row.isNullAt(values) || row.isNullAt(dim)))) {
                throw new IllegalArgumentException("null point column");
            }
            return new WeightedPoint(vector(row), w);
        }

        private SparseVector vector(InternalRow row) {
            switch (layout) {
                case SPARSE:
                    return SparseVector.of(Math.toIntExact((long) dimType.read(row, dim)),
                            ints(row.getArray(first), firstNullable), doubles(row.getArray(values), valuesNullable));
                case DENSE_ARRAY:
                    return SparseVector.dense(doubles(row.getArray(first), firstNullable));
                default:
                    InternalRow v = row.getStruct(first, UDT_FIELDS);
                    if (v.getByte(0) == 0) {
                        return SparseVector.of(v.getInt(1), v.getArray(2).toIntArray(), v.getArray(3).toDoubleArray());
                    }
                    return SparseVector.dense(v.getArray(3).toDoubleArray());
            }
        }

        private static int[] ints(ArrayData a, boolean nullable) {
            if (nullable) {
                rejectNulls(a);
            }
            return a.toIntArray();
        }

        private static double[] doubles(ArrayData a, boolean nullable) {
            if (nullable) {
                rejectNulls(a);
            }
            return a.toDoubleArray();
        }

        private static void rejectNulls(ArrayData a) {
            for (int k = 0, n = a.numElements(); k < n; k++) {
                if (a.isNullAt(k)) {
                    throw new IllegalArgumentException("null array element at position " + k);
                }
            }
        }
    }
}
