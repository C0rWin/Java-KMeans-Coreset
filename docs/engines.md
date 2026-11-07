# Running coresets on data engines

A coreset is a weighted point set whose k-means cost approximates the full
data's cost for every set of centers. Every engine integration rests on the same
two operations from `coreset-core`:

- **union** (`Coreset.union`): concatenate points, add offsets. It is exactly
  associative and commutative, up to floating-point summation order.
- **reduce** (`CoresetAlgorithm.compress`): shrink a coreset to about m points.
  `reduce(union(…))` is *not* associative: `R(R(A∪B)∪C) ≠ R(A∪R(B∪C))`. Both
  sides are valid coresets of `A∪B∪C`, but with different points and errors.

`MergeReduceTree` is a binary counter over these operations (Bentley–Saxe; the
merge-and-reduce scheme of Har-Peled & Mazumdar, used by Barger & Feldman,
[SDM'16](https://arxiv.org/abs/1511.08990)). After `b` batches, each batch has
been compressed at most `floor(log2 b)` times, however the batches arrived.

Status: **Spark is implemented** (`coreset-spark`). The other sections are
designs.

## Things every engine has to get right

### Error accumulation with tree depth

A reduce with relative error ε_ℓ applied L times along a point's path compounds
to (1 + ε_ℓ)^L. To reach an end-to-end ε, each level gets
ε_ℓ = (1 + ε)^(1/L) − 1 ≈ ln(1 + ε)/L. Sampling coresets need m ∝ 1/ε², so m
grows by about L². Failure probabilities are union-bounded over the reduce
operations: δ_ℓ = δ / #reduces (`ErrorBudget`). Worst-case bounds like these
only give trends at practical sizes (Bachem, Lucic, Krause,
[arXiv 1703.06476](https://arxiv.org/abs/1703.06476)). So we **measure
distortion** instead, following the ESA'22 protocol (Schwiegelshohn &
Sheikh-Omar, [arXiv 2207.00966](https://arxiv.org/abs/2207.00966)) with
`validate.Distortion` and `CandidateSolutions`.

Measured in `SparkCoresetsTest` (80,000 points from an unbalanced 6-component
Gaussian mixture, d = 8, 8 partitions, sensitivity sampling with k = 6,
m = 400):

| construction | compressions on a point's path | max distortion |
|---|---|---|
| one `compress` of all data on one machine | 1 | 1.064 |
| Spark build, tree depth 2 | 6 | 1.142 |

ESA'22 and Draganov et al. (SIGMOD'24,
[arXiv 2404.01936](https://arxiv.org/abs/2404.01936)) found sampling coresets
holding up under composition. On this instance there is a measurable cost, so
every Spark build reports its depth (`BuildMetrics.compressionDepth`) and can
split an (ε, δ) target over it (`BuildMetrics.errorBudget`). Fewer levels means
less error: use larger chunks per partition, or a smaller `treeDepth`, which
sends more partials to the driver.

### Determinism under retries

Engines rerun work. Spark recomputes lost partitions and runs speculative
copies; Flink replays from a checkpoint. Every compression therefore draws from
a generator that is a pure function of its position:
`MergeRandomness.derive(seed, stream, level, sequence)`, never the attempt
number, thread or clock. `MergeReduceTree` takes a `MergeRandomness`, and a
derived one makes a tree reproducible from its batches alone. A tree can then be
checkpointed as `(levelsView(), batchesAdded())` and rebuilt with
`MergeReduceTree.restore`.

Streams used by `coreset-spark` under one global seed:

| stream | consumer | level, sequence |
|---|---|---|
| p ≥ 0 | partition p's tree | tree level, chunk index (`-1`, #chunks for its final reduce) |
| −1 | aggregation rounds | round, group index |
| −2 | streaming batch seeds | 0, batchId |
| −3 | streaming driver tree | tree level, batch index |
| −4 | k-means restarts | 0, restart index |

Determinism also needs the *input* to be replayed identically: same partition
contents, in the same order. This is the condition Spark already places on
deterministic lineage.

### Serialization size

A sparse point with s non-zeros costs about 8 + 1 + s·(8 + 1–2) bytes in
`CoresetCodec`: raw IEEE-754 doubles, so round trips are bit-exact, and
delta-varint indices. Measured in `CoresetCodecTest` with m = 2,000 points:

| shape | codec | Java serialization | heap (`estimatedBytes`) | dense doubles |
|---|---|---|---|---|
| d = 784, ~94 nnz/point | 1.71 MB (9.1 B/nnz) | 2.34 MB | 2.46 MB | 12.6 MB |
| d = 100,000, ~100 nnz/point | 1.99 MB (10.0 B/nnz) | 2.49 MB | 2.60 MB | 1.6 GB |

Choices made:

- `SparseVector`, `WeightedPoint` and `Coreset` are `Serializable`, so RDDs of
  points work with Spark's default serializer (`parallelize`, `cache` with
  `MEMORY_ONLY_SER`, shuffles) and centers can be broadcast. `SparseVector`
  serializes through a proxy that is rebuilt with `SparseVector.of`. The records
  re-run their canonical constructors on deserialization. A crafted stream
  therefore cannot break invariants such as sorted indices, finite values and
  non-negative weights.
- Partials shipped between stages, and streaming checkpoints, use
  `CoresetCodec` bytes. That makes them about 27% smaller than Java
  serialization, independent of the configured serializer, and exactly
  measurable.
- Kryo's default field serializer cannot rebuild records. With
  `spark.serializer=org.apache.spark.serializer.KryoSerializer`, also set
  `spark.kryo.registrator=io.github.c0rwin.coreset.spark.CoresetKryoRegistrator`.
  That serializer writes codec bytes plus a length prefix (6 extra bytes on a
  60 kB coreset).

## Apache Spark (implemented)

Module `coreset-spark`. It targets Spark 4.0.5 with Scala 2.13 artifacts in
`provided` scope, and exposes a Java 17 API. It depends only on `coreset-core`.

### Batch

```java
JavaRDD<WeightedPoint> points = ...;            // or a Dataset<Row>, see PointColumns
BuildResult r = SparkCoresets.build(points, "sensitivity", Map.of("k", "10", "m", "2000"),
        /* seed */ 42L, /* treeDepth */ 2);
Coreset coreset = r.coreset();
BuildMetrics m = r.metrics();                   // stages, bytes shipped, depth, reduce ops, wall time
```

`PointColumns.sparse("indices", "values", "dim")` reads `array<int>`,
`array<double>` and `int` columns. `PointColumns.vector("features")` reads an
`array<double>` or Spark ML `Vector` column. Either form takes
`.withWeight("w")` (default weight 1). Rows are decoded from Spark's internal
format, so array elements are copied out as primitive arrays with no boxing.

How a build runs:

1. **Partition.** `mapPartitionsWithIndex` streams each partition through a
   `MergeReduceTree` in chunks: `chunkSize`, default 2·m. It then compresses the
   tree's snapshot once, giving one partial per partition.
2. **Rounds.** While more than `fanIn = max(2, ⌈P^(1/treeDepth)⌉)` partials
   remain, partial i goes to group ⌊i / fanIn⌋. `groupByKey` collects each
   group, which is merged in index order: union, then one compression.
3. **Driver.** The remaining ≤ fanIn partials are merged the same way.

This has the shape of `RDD.treeAggregate`, but it is deliberately not built on
it. `treeAggregate` folds values in shuffle arrival order, and because reduce is
not associative, the result would depend on timing. Keying by partition index
and sorting each group keeps builds bit-identical across runs. The same holds
under failures: `SparkCoresetsTest` makes the first attempt of half the
partitions fail, checks through the status tracker that 4 attempts failed, and
compares the coreset and metrics exactly against an undisturbed run.

`BuildMetrics` is carried inside the partials and summed on the driver, so it
is exact under retries, unlike accumulators updated in transformations. It
records, per stage, the number of partials, points, encoded bytes and estimated
heap bytes. It also records the bytes shipped, the partition tree height, the
compression depth, the reduce operations, the input points and weight, and the
wall time. With the 80,000-point workload, `treeDepth` 3 shipped 289 kB over 3
stages, and `treeDepth` 1 shipped 164 kB, all of it to the driver.

### k-means on the coreset

`SparkCoresets.kMeans(points, algorithmId, params, seed, treeDepth, k,
restarts, maxIterations)`:

1. builds a coreset;
2. keeps the best of `restarts` runs of k-means++ followed by weighted Lloyd,
   judged by coreset cost (a single start can stop in a poor local optimum);
3. evaluates the winner in one distributed pass, with the centers broadcast and
   partition sums added in partition order.

`CoresetKMeans.deployedDistortion()` reports the coreset/full cost ratio at the
deployed solution.

Test comparison against Spark ML `KMeans` (k-means||, `weightCol`) on the same
80,000 weighted points: coreset solution cost 8.168e5, MLlib 8.002e5, a ratio
of 1.021 (the test requires ≤ 1.05). Wall times on cached input in local[4],
reported but not asserted, vary between runs: build 24–67 ms, solve 2–4 ms,
evaluation 36–54 ms; MLlib fit 1.7–3.8 s. This is a smoke comparison on one
machine, not a benchmark.

### Structured Streaming

```java
CoresetStreamSink sink = CoresetStreamSink.open(spark, "hdfs:///coresets/sink",
        PointColumns.vector("features").withWeight("weight"),
        "sensitivity", Map.of("k", "10", "m", "2000"), 42L, 2);
StreamingQuery q = input.writeStream()
        .option("checkpointLocation", "hdfs:///coresets/query")
        .foreachBatch(sink)
        .start();
Coreset now = sink.compressedSnapshot();        // any thread, never blocks
```

The sink builds each micro-batch with `SparkCoresets.build` and inserts the
result into a driver-side tree. After every batch, it writes
`state-<batchId>.bin` to the checkpoint directory: magic, version, the
configuration, `lastBatchId`, the batch count, the levels as codec bytes, and a
CRC32. The file is written under a temporary name and renamed into place.

- **Exactly once per batchId.** The state is written before `foreachBatch`
  returns, which is before Spark commits the batch. A batchId at or below the
  last applied one is skipped. If Spark replays a batch after a crash between
  the two commits, the batch is not counted twice.
- **Exact resume.** A restarted query resumes the tree where it stopped, and
  the result is bit-identical to an uninterrupted run. `CoresetStreamSinkTest`
  checks this with a `MemoryStream`: a stop after 3 of 6 batches, then a restart
  on the same Spark checkpoint with a freshly opened sink. It then replays
  batchIds 4 and 5, which are skipped.
- **Changed configuration.** Opening a directory whose state was written with a
  different algorithm, parameters, seed, depth or chunk size fails. So does
  opening a directory whose newest file is corrupt; rolling back silently would
  lose batches Spark has already committed.
- **Failure mid-batch.** If a batch fails after the tree started changing, the
  sink refuses further batches. Reopen it from the checkpoint.
- **Reading while streaming.** Readers see an immutable view published after
  each batch. `snapshot()`, `compressedSnapshot()`, `lastBatchId()` and
  `lastBuildMetrics()` take no lock.

Caveats: rename must be atomic on the checkpoint file system. That holds for
HDFS and local disks; object stores need a committing file system. BatchIds are
only meaningful within one query's Spark checkpoint. The driver tree holds
O(m · log b) points after b batches.

Spark 4.0's `transformWithState` could instead hold one tree per shard key in
Spark-managed state, with timers emitting snapshots. That remains an option if
per-key summaries are needed.

### Running

- Build and test: `mvn -B -pl coreset-spark -am verify`. Tests run in
  `local[4,3]` and take 15–30 s, most of it Spark start-up. The surefire
  `argLine` carries the `--add-opens` flags Spark needs on Java 17.
- Submit: Spark is `provided`, so ship both jars:
  `spark-submit --jars coreset-core-<v>.jar,coreset-spark-<v>.jar …`.
  `spark-submit` sets the module flags itself.

## Apache Flink (design)

Current stable release: Flink 2.3.0.

- **Per shard.** `keyBy(shard)` feeds a `KeyedProcessFunction` with three pieces
  of state: a `ListState<WeightedPoint>` chunk buffer, a `MapState<Integer,
  byte[]>` of tree levels (codec bytes) and a `ValueState<Long>` batch count.
  This is exactly `restore`'s input. Merges use
  `MergeRandomness.derived(seed, shard)`, so after a restore and replay they
  recompute the same coresets. The checkpointed state is O(m log n) per shard;
  use RocksDB for large m.
- **Global.** Periodically emit each shard's compressed snapshot, re-key to a
  single key (or use `windowAll`), and fold with an `AggregateFunction`. The
  accumulator is the level-indexed counter. `merge(a, b)` adds two counters
  level by level and is required for merging (session) windows. Flink "may
  reuse either input accumulator", which is safe because coresets are
  immutable.
- **Exactly once.** Flink checkpoints state and the replay offsets together. As
  long as the generators are derived from the replayed sequence (never held as
  live state), recovery is bit-identical.

## Apache Beam (design)

Latest release: Beam 2.77.0.

- `Combine.globally(new CoresetCombineFn())`, or `Combine.perKey`. The
  accumulator is the level-indexed counter plus a chunk buffer.
  `createAccumulator`, `addInput`, `mergeAccumulators` (level-wise counter
  addition) and `extractOutput` (compressed snapshot) map one to one. A `Coder`
  wraps `CoresetCodec`. With combiner lifting, `addInput` runs map-side, so only
  accumulators cross the network.
- **Associativity caveat.** The `CombineFn` contract requires associativity and
  commutativity, because inputs are "first broken up into subgroups before being
  combined, and their intermediate results further combined, in an arbitrary
  tree structure". Reduce satisfies this only up to the coreset guarantee: any
  merge tree yields a valid coreset, but not the same one. The level-indexed
  accumulator bounds the compressions per point by ⌈log₂(n/m)⌉ for every tree
  the runner picks, which is the property that matters.
- **Determinism.** Bit-identical reruns are not available. The runner chooses
  the merge tree, and it may differ between attempts or runners. Seeding each
  merge from accumulator contents still keeps the output a function of the
  merge tree. Validity is unaffected; exact reproducibility is.

## Kafka Streams (design; API from memory, not verified against current docs)

- `groupByKey().aggregate(initializer, aggregator, Materialized.with(keySerde,
  coresetSerde))`, with the serde built on `CoresetCodec`. Each key holds an
  add-only accumulator: buffer, levels and batch count, in a RocksDB store backed
  by a changelog topic. Seeds are derived from (key, batch count), which lives
  in the aggregate itself.
- There is no two-accumulator merge, except the session-window `Merger`. A
  global coreset needs a re-key to one partition, or interactive queries that
  merge the per-instance snapshots.
- `processing.guarantee=exactly_once_v2` commits state and output atomically,
  so no separate batch-id bookkeeping is needed.
- Suited to coresets embedded in a service. Build it on demand.

## Apache Storm (status: not planned)

Storm 3.1.0 is the current release; 3.x is active and Trident is retained. It
is a niche choice now.

- **Mapping.** Spout → `fieldsGrouping(shard)` → bolt holding the accumulator →
  `globalGrouping` bolt for the final merge.
- **Trident.** `persistentAggregate` with a `CombinerAggregator`: `init` wraps
  one point, `combine` is the counter merge, `zero` is the empty accumulator.
  This gives transactional state. `combine` must be associative, which carries
  the same caveat as Beam.
