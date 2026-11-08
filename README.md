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
2. **coreset-node** - transport-agnostic ingestion node: sessions of worker
   threads that merge-and-reduce incoming batches, with lock-free routing,
   idempotent (seq-deduplicated) ingest, backpressure and snapshots.
3. **coreset-grpc** - gRPC service, coordinator client library, Prometheus
   metrics and the `coreset-node` / `coreset-coordinator` command-line tools.
4. **coreset-spark** - Spark 4.0 adapter (`provided` Spark dependencies):
   distributed coreset builds over RDDs/Datasets with per-build metrics,
   coreset k-means evaluated on the full data, and a checkpointed Structured
   Streaming sink. See [docs/engines.md](docs/engines.md) for this and the
   other engine mappings.
5. **coreset-flink** - Flink 2.3 adapter (DataStream API, `provided` Flink
   dependencies): a continuously updated coreset with per-subtask
   merge-and-reduce in checkpointed operator state and a parallelism-1 merge
   stage, bit-identical after failure recovery; per-window coresets for tumbling
   event-time windows; Kryo-free serializers and Flink metrics. See
   [docs/engines.md](docs/engines.md#apache-flink-implemented).
6. **coreset-testbed** - evaluation testbed: datasets, offline/stream/distributed
   runs, per-run validity reports (distortion, integrity, cost, memory) and
   aggregated tables; see [docs/benchmarks/README.md](docs/benchmarks/README.md).


APIs
---

1. ### Coreset algorithms (coreset-core)

   Points are `WeightedPoint`s over immutable `SparseVector`s (`SparseVector.of(dim,
   indices, values)` or `SparseVector.dense(values)`); a `Coreset` is an immutable
   weighted point set. `Algorithms.create(id, params)` builds a
   `CoresetAlgorithm` from string parameters:

   | id | parameters |
   |---|---|
   | `uniform` | `m` |
   | `sensitivity` | `k`, `m`; optional `seeding` (`kmeans++` or `bicriteria`), `lloydIterations`, `bound`, `alpha` |
   | `mean` | `m`, or `k`, `eps`, `maxM` (adaptive); optional `lloydIterations` |

   A `MergeReduceTree` keeps a streaming coreset of compressed batches, and
   `WeightedLloyd.bestOf` clusters the result:

   ```java
   CoresetAlgorithm algorithm = Algorithms.create("sensitivity", Map.of("k", "10", "m", "200"));
   MergeReduceTree tree = new MergeReduceTree(algorithm, new SplittableRandom(7));
   long seq = 0;
   for (List<WeightedPoint> batch : batches) {
       tree.add(algorithm.compress(Coreset.of(batch), new SplittableRandom(seq++)));
   }
   Coreset coreset = tree.snapshot();
   KMeansResult result = WeightedLloyd.bestOf(coreset.points(), 10, 50, 10, new SplittableRandom(1),
           MetricsSink.NOOP);
   ```

2. ### Distributed protocol (coreset-grpc)

   The gRPC service is defined in
   [`coreset-grpc/src/main/proto/coreset.proto`](coreset-grpc/src/main/proto/coreset.proto):
   `CreateSession`, a bidirectional `Ingest` stream of batches answered by
   acks, `Snapshot`, `PartialCost`, `Stats` and `CloseSession`.
   `CoresetCoordinator` is the client library: it creates a session on every
   node, spreads batches over them, gathers the snapshots into one coreset and
   evaluates costs on the nodes.


How to run
---

1. ### Building the project.

  Requirements: JDK 17+ and Maven 3.6+. From the repository root run:
  ```
  mvn clean verify
  ```

  This compiles all modules and runs the unit tests (JUnit 5).

2. ### Running a node and the coordinator

   `coreset-node` hosts ingestion sessions (bounded lock-free worker queues,
   idempotent batches, snapshots with completeness counters) and
   `coreset-grpc` serves it over gRPC. Packaging produces two executable jars,
   `coreset-grpc/target/coreset-node.jar` and
   `coreset-grpc/target/coreset-coordinator.jar`. Algorithms are looked up via
   `java.util.ServiceLoader` providers of
   `io.github.c0rwin.coreset.node.AlgorithmResolver`; the bundled
   `CoreAlgorithmsResolver` serves the coreset-core ids above.

   Start one node per machine (or several on one host with different ports),
   then run the coordinator over a points file:

   ```
   java -jar coreset-node.jar --port 7070
   java -jar coreset-node.jar --port 7071 --metrics-port 9465

   java -jar coreset-coordinator.jar --nodes localhost:7070,localhost:7071 \
     --input points.csv --algorithm sensitivity --param k=20 --param m=1000 \
     --k 20 --workers 4 --batch-size 1000
   ```

   The input has one point per line: dense values separated by commas or
   whitespace, or, with `--sparse-dim D`, 0-based `index:value` tokens (a
   leading libsvm label is ignored). `--param key=value` passes algorithm
   parameters to the nodes; `--final-algorithm`/`--final-param` compress the
   gathered union once more before clustering. The coordinator prints one
   summary line (`# session=... points=... batches=... coreset_points=...
   cost=...`) followed by the k centers as CSV rows. `cost` is the k-means cost
   of those centers evaluated on the nodes' coresets, an estimate of the
   full-data cost. `--help` on either jar lists every option.

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

The software is released under the MIT License as detailed in [LICENSE](LICENSE).

