namespace java univ.ml.distributed.coreset

enum CoresetAlgorithm {
    UNIFORM,
    NON_UNIFORM,
    KMEANS_PLUS_PLUS
}

struct CoresetPoint {
    1: map<i32, double> coords,
    2: i32 dim
}

struct CoresetWeightedPoint {
    1: CoresetPoint point,
    2: double weight,
}

struct CoresetPoints {
    1: i64 id,
    2: list<CoresetWeightedPoint> points
}

service CoresetService {

    bool initialize(1:i32 k, 2:i32 sampleSize, 3:CoresetAlgorithm algorithm)

    oneway void compressPoints(1:CoresetPoints message)

    CoresetPoints getTotalCoreset()

    double getEnergy(1: CoresetPoints centers, 2: CoresetPoints points)

    bool isDone()
}
