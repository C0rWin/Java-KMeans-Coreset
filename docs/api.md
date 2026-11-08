# API overview

Entry points per use case. Every Java block on this page is a region of
[`ApiDocSnippets`](../coreset-examples/src/test/java/io/github/c0rwin/coreset/examples/lightweight/ApiDocSnippets.java),
compiled with the build; `ApiDocTest` fails if a block here and its region
differ, and runs the in-process, streaming, distributed and evaluation blocks.

## Algorithms by id: the SPI

Every layer turns an algorithm id and string parameters into a
`CoresetAlgorithm` with `CoresetAlgorithms` (coreset-core). It finds
`CoresetAlgorithmProvider`s with `java.util.ServiceLoader`: coreset-core
registers the built-ins (`uniform`, `sensitivity`, `mean`, parameters in the
README), and any jar on the class path can add ids. The registry is checked
when it loads: two providers claiming one id, a provider with no or blank ids,
or no provider at all is an `IllegalStateException`. An unknown id is an
`UnknownAlgorithmException` that lists the available ids; bad parameters are
the provider's `IllegalArgumentException`.

`CoresetAlgorithms.of(providers...)` builds a registry without discovery
(embedding, tests). Engines that ship work to other JVMs take a
`CoresetAlgorithmFactory`: `RegisteredAlgorithm(id, params)` resolves the id on
each JVM, and any other serializable implementation supplies the algorithm in
code.

## Embed in-process

Compress a point set once and cluster the coreset:

```java
CoresetAlgorithm algorithm = CoresetAlgorithms.load().create("sensitivity", Map.of("k", "8", "m", "400"));
Coreset coreset = algorithm.compress(Coreset.of(points), new SplittableRandom(7));
KMeansResult solution = WeightedLloyd.bestOf(coreset.pointSet(), 8, 50, 5, new SplittableRandom(8),
        MetricsSink.NOOP);
```

## Stream with `MergeReduceTree`

A merge-and-reduce tree keeps a summary of every batch so far in
O(log batches) coresets. `snapshot()` is the union of the levels;
`compressedSnapshot()` compresses it once more. Use
`MergeRandomness.derived(seed, stream)` instead of one shared generator to
make the tree reproducible and checkpointable (`levelsView`, `restore`).

```java
CoresetAlgorithm algorithm = CoresetAlgorithms.load().create("sensitivity", Map.of("k", "8", "m", "400"));
MergeReduceTree tree = new MergeReduceTree(algorithm, new SplittableRandom(7));
for (List<WeightedPoint> batch : batches) {
    tree.add(Coreset.of(batch));
}
Coreset summary = tree.compressedSnapshot();
```

## Distributed: node and coordinator

A `CoresetNode` (coreset-node) hosts sessions; `CoresetNodeServer`
(coreset-grpc) serves it, and `CoresetCoordinator` creates a session on every
node, spreads batches over them and gathers the union. The CLIs
(`coreset-node.jar`, `coreset-coordinator.jar`) wrap the same classes; see
[distributed-run.md](distributed-run.md). Here one node on localhost runs the
plugged-in `lightweight` algorithm:

```java
try (CoresetNode node = new CoresetNode(CoresetAlgorithms.load());
     CoresetNodeServer server = CoresetNodeServer.start(NodeServerOptions.builder().port(0).build(), node,
             new RecordingMetrics(), MetricsSink.NOOP);
     CoresetCoordinator coordinator = CoresetCoordinator.connect(List.of("127.0.0.1:" + server.port()),
             CoordinatorOptions.defaults(), ClientSecurity.plaintext(), MetricsSink.NOOP)) {
    coordinator.createSession("demo", new SessionSpec("lightweight", Map.of("m", "400"), 7L, 2, 16));
    try (IngestStream ingest = coordinator.openIngest("demo")) {
        for (int from = 0; from < points.size(); from += 1_000) {
            ingest.submit(points.subList(from, Math.min(points.size(), from + 1_000)));
        }
        ingest.flush(Duration.ofMinutes(1));
    }
    GatheredSnapshot gathered = coordinator.gather("demo", Duration.ofMinutes(1));
    return gathered.coreset();
}
```

## Spark

`SparkCoresets.build` (coreset-spark) takes a registered id, validated on the
driver and resolved again on every executor, or a `CoresetAlgorithmFactory`
serialized into the tasks (a factory has no `m` to derive the chunk size from,
so it is passed explicitly). `SparkCoresets.kMeans` and `CoresetStreamSink.open`
take the same two forms; see [engines.md](engines.md).

```java
BuildResult byId = SparkCoresets.build(points, "lightweight", Map.of("m", "400"), 7L, 2);
BuildResult byFactory = SparkCoresets.build(points, new LightweightCoresetFactory(400), 7L, 2, 800);
```

## Flink

`CoresetConfig` (coreset-flink) holds a registered id or a factory; both are
called once when the configuration is built and again in every operator.
`FlinkCoresets.stream` keeps a continuously updated coreset,
`FlinkCoresets.tumblingWindows` one per window.

```java
CoresetConfig byId = CoresetConfig.builder("lightweight", Map.of("m", "400"), 7L).build();
CoresetConfig byFactory = CoresetConfig.builder(new LightweightCoresetFactory(400), 7L).chunkSize(800).build();
DataStream<CoresetSnapshot> snapshots = FlinkCoresets.stream(points, byId);
```

Checkpoints (Flink operator state, the Spark stream sink) record
`CoresetAlgorithmFactory.identity()`: the id and parameters for a registered
algorithm, else the factory's class name and a hash of its serialized form, so
state is never resumed under a different algorithm. Override `identity()` to
keep checkpoints compatible across changes to a factory class.

## Evaluate

`validate` (coreset-core) measures a coreset against the full data:

```java
double distortion = Distortion.evaluate(full, coreset,
        CandidateSolutions.standard(full, coreset, 8, 10, 3, new SplittableRandom(99))).max();
double weightRelErr = IntegrityCheck.check(full, coreset).weightRelErr();
```

The testbed (coreset-testbed) runs a grid of datasets, sizes, algorithms and
modes (offline, stream, distributed over in-process nodes or gRPC) and writes a
validity report per run; any registered id can appear in a config, with
`inject: [m]` (or `[k, m]`) naming the grid parameters it takes. See
[benchmarks/README.md](benchmarks/README.md).

## Implement your own algorithm

1. Implement `CoresetAlgorithm`: stateless, thread-safe, all randomness from
   the given generator; honour interruption with `Guards.checkInterrupted`
   (nodes interrupt workers on close).
2. Implement `CoresetAlgorithmProvider` with a public no-argument constructor:
   `ids()` and `create(id, params)`, rejecting bad parameters with
   `IllegalArgumentException`.
3. Register it in
   `META-INF/services/io.github.c0rwin.coreset.core.CoresetAlgorithmProvider`
   (or `provides ... with ...` in a module descriptor).
4. For Spark or Flink jobs that ship the algorithm in code, also write a
   named, serializable `CoresetAlgorithmFactory`.

[coreset-examples](../coreset-examples) does all four for the lightweight
coreset of Bachem, Lucic and Krause (KDD 2018), and its tests run it unchanged
through the in-process tree, three gRPC nodes and a coordinator, Spark, Flink
and the testbed.

**Class paths.** The provider jar must be on the class path of every JVM that
resolves ids:

- node and coordinator CLIs: the shaded jars ignore `-cp` with `-jar`, so run
  `java -cp coreset-node.jar:my-algorithms.jar io.github.c0rwin.coreset.grpc.cli.NodeCommand ...`
  (the coordinator needs it only for `--final-algorithm`);
- Spark: driver and executors, e.g. `spark-submit --jars my-algorithms.jar`
  or an assembly jar;
- Flink: inside the job jar, or in `lib/` of every task manager;
- testbed: `java -cp coreset-testbed.jar:my-algorithms.jar io.github.c0rwin.coreset.testbed.cli.TestbedCommand ...`.

A code-supplied factory needs no registration, but its class must travel with
the job (application jar).

## Module boundary

coreset-core is a JPMS module, `io.github.c0rwin.coreset.core`. It exports
`core`, `algo`, `kmeans`, `metrics`, `sampling`, `stream` and `validate`, uses
and provides the algorithm SPI, and keeps shared kernels in the unexported
`io.github.c0rwin.coreset.core.internal` package, which may change in any
release. On the class path (Spark, Flink and the other modules) the descriptor
is ignored.
