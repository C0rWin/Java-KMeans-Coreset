k-Means for Streaming and Distributed Big Sparse Data
===

Introduction
---

In this repository we provide Matlab implementation of coreset algorithms, used
for evaluation in following [paper](https://arxiv.org/pdf/1511.08990.pdf):

> **k-Means for Streaming and Distributed Big Sparse Data.**
> *Artem Barger, and Dan Feldman.*
> Proceedings of the 2016 SIAM International Conference on Data
> Mining. Society for Industrial and Applied Mathematics, 2016.

Project Structure
---

The project consists of the following modules:

1. **coreset-core** - dependency-free core (`io.github.c0rwin.coreset.core`):
   immutable sparse vectors, weighted coresets, k-means cost, weighted
   k-means++ seeding and Lloyd, a merge-and-reduce streaming tree, and
   lightweight metrics (`MetricsSink`, `RecordingMetrics`, spans).
   - `algo`: `UniformCoreset`, `SensitivityCoreset` (k-means++ or bicriteria
     seeding, with sampling provenance) and `MeanCoreset` (Barger-Feldman,
     fixed or adaptive m, with the additive offset); `Algorithms.create(id,
     params)` resolves them by id (`uniform`, `sensitivity`, `mean`).
   - `sampling`: Vose alias sampler for multinomial draws.
   - `validate`: empirical distortion over candidate center sets, integrity
     checks (weight and moments, one-sided oracle), per-query confidence
     intervals, merge-and-reduce error budgets and a-priori sample sizes.
2. **kmeans-coreset** (legacy) - the original implementation of coresets algorithms:
    2.1. Uniform coreset
    2.2. Non Uniform coreset (Sensitivity)
    2.3. Deterministic coreset construction suggested in the paper above
3. **coreset-communication** (legacy) - definition of the protocol for distributed
   computation of the coreset
4. **coreset-server** (legacy) - implementation of server and client side of the
   distributed protocol which enables coreset computation.
5. **coreset-node** - transport-agnostic ingestion node: sessions of worker
   threads that merge-and-reduce incoming batches, with lock-free routing,
   idempotent (seq-deduplicated) ingest, backpressure and snapshots.
6. **coreset-grpc** - gRPC service, coordinator client library, Prometheus
   metrics and the `coreset-node` / `coreset-coordinator` command-line tools.
7. **coreset-spark** - Spark 4.0 adapter (`provided` Spark dependencies):
   distributed coreset builds over RDDs/Datasets with per-build metrics,
   coreset k-means evaluated on the full data, and a checkpointed Structured
   Streaming sink. See [docs/engines.md](docs/engines.md) for this and the
   other engine mappings.
8. **coreset-testbed** - evaluation testbed: datasets, offline/stream/distributed
   runs, per-run validity reports (distortion, integrity, cost, memory) and
   aggregated tables; see [docs/benchmarks/README.md](docs/benchmarks/README.md).


APIs
---

1. ### Coreset algorithms, interfaces and definitions

   All algorithms works with `SparseWeightableVector` which is an
   implementation of weighted vector in `R^d`, extends and implements following
   interfaces:

   * SparseSample
   ```java
   public interface SparseSample extends SparseWeightable {

       void setProbability(final double prob);

       double getProbability();

   }
   ```

   * SparseWeightable
   ```java
   public interface SparseWeightable {

       void setWeight(final double weight);

       double  getWeight();
   }
   ```

   * SparseClusterable
   ```java
   public interface SparseClusterable {

       RealVector getVector();
   }
   ```

   **SparseCoresetAlgorithm** - is the interface which defines a main API for

   coreset algorithm, as follows:

   ```java
   public interface SparseCoresetAlgorithm extends Serializable {

       List<SparseWeightableVector> takeSample(final List<SparseWeightableVector>  pointset);

   }

   ```

   where `takeSample` is the method which actually represents coreset
   computation, it receives a list of weighted poinst and returns new weighted
   list of points reduced from the original list.

   We provide three different implementations of this interface:

   * Uniform coreset - `SparseUniformCoreset`

   ```java
    /**
    * @param sampleSize - coreset size to sample
    */
   public SparseUniformCoreset(final int t) {
       this.t = t;
   }
   ```

   Constructor which receives parameter of the coreset size to sample.

   * Non uniform coreset - `SparseNonUniformCoreset`

   ```java
    /**
    * @param seedingAlgorithm - algorithm used to seed and compute sensitivity
    * @param sampleSize - coreset size to sample
    */
   public SparseNonUniformCoreset(final SparseSeedingAlgorithm seedingAlgorithm, final int sampleSize) {
       this.sampleSize = sampleSize;
       // Default is the k-means++ algorithm which provides (1, log(n)) approximation for k-means problem.
           this.seedingAlgorithm = seedingAlgorithm;
   }
   ```

    Constructor which receives coreset size to sample and intial seeding
    algorithm (`SparseSeedingAlgorithm`) used for sensitivity computation.

    ```java
    public interface SparseSeedingAlgorithm {

        /**
         * Seed initial cluster centers.
         *
         * @param vectors vectors to select initial centers from.
         Method should not change input parameter.
         * @return list of the cluster centers.
         */
        List<SparseCentroidCluster> seed(final List<SparseWeightableVector> vectors);

    }

    ```

   * Our algorithm - `SparseKmeansCoresetAlgorithm`

    ```java
    /**
    * @param sampleSize - coreset size to sample
    */
    public SparseKmeansCoresetAlgorithm(int sampleSize) {
        this.sampleSize = sampleSize;
    }
    ```

    Constructor which receives as a parameter coreset size to compute ("sample").

2. ### Streaming and merge-and-reduce tree

   Following class defines the streaming algorithm - encapsulated
   abstraction of the *merge-and-reduce* tree which allows us to compute
   coreset in streaming seting.

   ```java
   public class StreamingAlgorithm {

       private Map<Integer, List<SparseWeightableVector>> coresetTree = Maps.newHashMap();

       private SparseCoresetAlgorithm coresetAlgorithm;

       public StreamingAlgorithm(SparseCoresetAlgorithm coresetAlgorithm)
       {
           this.coresetAlgorithm = coresetAlgorithm;
       }

       public void addDataset(final List<SparseWeightableVector> dataset) {
           List<SparseWeightableVector> coreset = coresetAlgorithm.takeSample(dataset);
           int treeLevel = 0;
           List<SparseWeightableVector> leaf = coresetTree.get(treeLevel);

           while (leaf != null) {
               coresetTree.remove(treeLevel++);
               coreset.addAll(leaf);
               coreset = coresetAlgorithm.takeSample(coreset);

               leaf = coresetTree.get(treeLevel);
           }

           coresetTree.put(treeLevel, coreset);
       }

       public List<SparseWeightableVector> getTotalCoreset() {
           if (coresetTree.size() == 1) {
               for (List<SparseWeightableVector> coreset : coresetTree.values()) {
                   return
                       coreset;
               }
           }
           final List<SparseWeightableVector> treeCoresetView = Lists.newArrayList();
           for (Map.Entry<Integer, List<SparseWeightableVector>> each : coresetTree.entrySet()) {
               treeCoresetView.addAll(each.getValue());
           }

           return coresetAlgorithm.takeSample(treeCoresetView);
       }

   }
   ```
    It accepts concrete implementation of the `SparseCoresetAlgorithm` interface and proceeds with
    coreset tree construction with streaming batch of points.

3. ### Distributed protocol definition

    Following defines Apache Thrift structure and services for distributed
    coreset computation:

    ```thrift
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
    ```

4. ### Distributed protocol implementation:

   `CoresetServiceHandler` implements the logic of the `CoresetService` define
   with Apache Thrift above, it's capable to serve incomming batches of
   streamed data points and maintain coreset **merge-and-reduce** tree
   according to the initialization parameters.

   Protocol definition allows to initialize service handler with three
   different implementations of the coreset algorithm, the k of kmeans
   algorithm parameters and the actual coreset size.

   Once initialized it will allow to send weithed points from `R^d` to compress
   them into coreset and construct **merge-and-reduce** tree. After all it also
   could be used to compute energy function in the distributed manner,
   leveraging `getEnergy` method, which receives centers and points and
   computes sum of squared distances to the provided centers.


How to run
---

1. ### Building the project.

  Requirements: JDK 17+ and Maven 3.6+. From the repository root run:
  ```
  mvn clean verify
  ```

  This compiles all modules and runs the unit tests (JUnit 5; legacy JUnit 4
  tests run through the vintage engine). Tests that need external datasets or
  are JMH benchmarks are ignored. Two legacy stress tests (hundreds of thousands
  of points) are excluded by default; run them with `mvn verify -Plegacy-stress`.

  The Thrift-generated Java sources in `coreset-communication/src/main/java` are
  checked in, so no `thrift` binary is needed. After editing
  `coreset-communication/src/main/thrift/coreset.thrift`, regenerate them with
  Thrift 0.13.0: `thrift --gen java -out src/main/java src/main/thrift/coreset.thrift`.

  Packaging produces jar files in the coreset-server/target folder:

  1.1. **client.jar** - the client application to distribute points accross servers
  1.2. **server.jar** - the server application to receive batch of streamed
  points, compute and construct coreset tree.

2. ### Running server

   To execute server code you need to run:

   ```
   java -jar server.jar 9999
   ```

   where 9999 stands for TCP port server will bind to.

   Next to execute client you need to run:

   ```
   java -jar client.jar -k 32 -algorithm KMEANS_PLUS_PLUS \
   -sampleSize 512 -batchSize 1024 \
   -hosts serverIP1:9999,serverIP2:9999,...,serverIPN
   ```

   Where:

   * k - kmeans clustering parameter
   * algorithm - one of the available agorithms options: UNIFORM, NON_UNIFORM, KMEANS_PLUS_PLUS
   * sampleSize - the coreset size to sample
   * batchSize - number of points to send in each batch to the remote server
   * hosts - comma separated list of the remote hosts to distributed points to

3. ### gRPC node and coordinator (replacement for the Thrift server)

   `coreset-node` hosts ingestion sessions (bounded lock-free worker queues,
   idempotent batches, snapshots with completeness counters) and
   `coreset-grpc` serves it over gRPC. Packaging produces
   `coreset-grpc/target/coreset-node.jar` and
   `coreset-grpc/target/coreset-coordinator.jar`. Algorithms are looked up via
   `java.util.ServiceLoader` providers of
   `io.github.c0rwin.coreset.node.AlgorithmResolver` on the classpath; the node
   refuses to start without one.

   ```
   java -jar coreset-node.jar --port 7070
   java -jar coreset-coordinator.jar --nodes localhost:7070 \
     --input points.csv --algorithm <id> --param key=value --k 32 --batch-size 1000
   ```

   The coordinator keeps the cheapest of `--restarts` (default 10) k-means++ and
   Lloyd runs on the gathered coreset, since a single run often ends in a poor
   local optimum.

   Each node serves the standard `grpc.health.v1.Health` service and Prometheus
   metrics at `http://127.0.0.1:9464/metrics`.

   **Security.** A node listens on `127.0.0.1` for both gRPC and metrics unless
   told otherwise (`--bind`, `--metrics-bind`). Without TLS and authentication
   any peer that can reach the port can create, read, poison or close every
   session, so only expose a node on a trusted network, or turn on:

   * TLS: `--tls-cert node.crt --tls-key node.key` on the node (PEM, PKCS#8 key)
     and `--tls-ca ca.crt` on the coordinator;
   * mutual TLS: additionally `--tls-client-ca clients-ca.crt` on the node and
     `--tls-cert`/`--tls-key` on the coordinator;
   * a shared token: `--auth-token-file <file>` on both sides. Every call then
     carries `authorization: Bearer <token>`; send it over TLS.

   The health service stays unauthenticated for probes. `Stats` lists session
   ids only when callers are authenticated (token or client certificates). The
   metrics endpoint has no authentication of its own.

   **Limits.** Every request is checked against node limits before anything is
   allocated: vector dimension (`--max-dim`, 2^24), points per batch
   (`--max-points-per-batch`, 2^18), request size (`--max-message-bytes`,
   16 MiB), open sessions (`--max-sessions`, 64), workers over all sessions
   (`--max-total-workers`, 4096), queue capacity (`--max-queue-capacity`, 4096),
   cost work in points x centers (`--max-cost-work`, 2^31), snapshot size
   (`--max-snapshot-points`, 2^22) and the longest awaited snapshot
   (`--max-await-ms`, 10 min). Session ids are at most 128 characters of
   `[A-Za-z0-9._:-]`; at most 32 algorithm parameters of up to 256 characters.
   Shape violations answer `INVALID_ARGUMENT` (or a FAILED ack), exhausted
   capacity `RESOURCE_EXHAUSTED`. Sessions without ingest progress or reads for
   `--session-idle-timeout-ms` (1 h) are closed.

   **Exactly-once ingest.** Batches are deduplicated per `(producer, seq)`:
   each coordinator ingest run uses a fresh random producer id and numbers its
   batches 0, 1, 2, ... per node, so several runs (or coordinators) can feed one
   kept session. A producer's seqs must stay within `--max-seq-window` (4096) of
   its lowest unacknowledged seq. An ACCEPTED ack means the batch was enqueued;
   closing a session drops batches still queued, and an ingest that overlaps a
   close or a failure is answered FAILED.

   **Reproducibility.** Every batch coreset is a function of the session seed,
   the node's partition, the producer id and the batch seq only, so it is
   identical across workers, retries and arrival order. The coordinator gives
   node `i` partition `i`, so nodes never sample equal seqs with the same
   random stream, and each run gets a random producer id unless
   `--producer-id` (or `openIngest(session, producerId)`) fixes one; with a
   fixed id and the same node list a run is repeatable. The merged tree is
   reproducible only in the
   `single-writer` tree mode with a fixed per-worker arrival order (BUSY
   retries can reorder it). In `cas` mode, which worker performs a merge, and
   with which generator, depends on timing, so snapshots differ between runs.
   A `cas` snapshot briefly holds off new merges until those in flight finish
   (bounded wait) and is flagged inconsistent if that does not happen in time.


Feedback
---

If you'd like to use this implementation, please reference original paper, any
feedback send to Artem Barger (artem@bargr.net)

License
---

The software is released under the MIT License as detailed in kmeans.pyx.

