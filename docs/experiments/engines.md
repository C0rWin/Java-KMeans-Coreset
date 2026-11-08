# Engine comparison: gRPC coreset cluster vs Spark vs Flink

`coreset-experiments` runs the repository's own distributed system (N `coreset-node` processes plus a coordinator) next
to Apache Spark 4.0 and Apache Flink 2.3. All three get the same data, the same coreset algorithms through the SPI,
the same solver and the same quality evaluation. The harness measures performance, solution quality and behaviour
under real injected failures. It runs on one host today. The same configuration also runs on a cluster: targets are
`host:port` lists, the Spark master is a URL and the Flink JobManager an address.

- [What is compared](#what-is-compared)
- [Method](#method)
- [Fairness rules](#fairness-rules)
- [Faults](#faults)
- [How to run](#how-to-run)
- [Output](#output)
- [Preliminary results](#preliminary-results)
- [Limitations](#limitations)

## What is compared

| system (`type/mode`) | processes | coreset construction | final coreset |
|---|---|---|---|
| `grpc/spawn` | N `coreset-node` JVMs plus the coordinator in the cell JVM | each node merge-reduces the batches of its shard (`workersPerNode` workers, single-writer trees) | union of the N node snapshots, gathered by the coordinator |
| `grpc/external` | nodes already running on `cluster.grpcTargets` | same | same |
| `spark/local` | one JVM, `local[N]` | `SparkCoresets.build`: one partition tree per partition, aggregation tree of depth `treeDepth` | compressed on the driver (final merge) |
| `spark/local-cluster` | standalone master and workers in the cell JVM, N single-core executor JVMs (`local-cluster[N,1,MB]`) | same | same |
| `spark/cluster` | `cluster.sparkMaster` | same | same |
| `spark-mllib/local` | one JVM, `local[N]` | none: MLlib `KMeans` (k-means\|\| init, `weightCol`) on all points | none: the native baseline |
| `flink/minicluster` | MiniCluster in the cell JVM, 1 task manager with N slots | `FlinkCoresets.stream`: N local subtasks, a parallelism-1 merge stage | final snapshot (`last = true`): union of the merge stage's levels (`compressSnapshots: true` adds one compression) |
| `flink/remote` | `cluster.flinkJobManager` | same | same |

**No Flink-native k-means baseline.** Flink ML's last release, 2.3.0 (June 2023), is built for Flink 1.15–1.17.
Maven Central has `flink-ml-lib-1.15/1.16/1.17` artifacts and nothing for Flink 2.x. Flink 2 removed the DataSet API,
and Flink ML's iteration runtime depends on Flink 1.x internals. So the native baseline is Spark MLlib only.

Algorithms are SPI ids with parameters: `uniform`, `sensitivity`, `mean` (built in) and `lightweight`, the third-party
plug-in from `coreset-examples`. Every system loads the same `CoresetAlgorithms` registry: the nodes, the Spark
executors and the Flink task managers all have the plug-in jar on their class path.

## Method

**Cells.** A configuration is a grid: datasets × systems × parallelism N × algorithms. Every cell runs once
undisturbed and once per applicable fault. Each cell runs in a fresh JVM (`CellMain`), one cell at a time. The
orchestrator (`ExperimentRunner`) stays idle meanwhile, so systems never share a heap, JIT state or CPU time. A cell:

1. starts the system and times it as **startup** (node JVMs listening, SparkContext with all N executors registered,
   MiniCluster up). Startup is reported separately and is not part of the job time;
2. runs `warmup` repetitions, which are not recorded;
3. runs `repetitions` measured repetitions, with seed `seed + i`, so the quality spread across seeds is measured
   too;
4. for a fault cell: `baselineRepetitions` undisturbed repetitions, then the fault repetitions, all on the same seed.
   The final coresets can then be compared bit for bit.

**Data path.** Every dataset is a testbed `PointFile`: `.cbin` (dense) or `.csr` (sparse), the format of the
converted public datasets (see [../datasets.md](../datasets.md)). Every system reads it through the same
`PointSource` in contiguous shards, using the ranged reader `PointFile.open(file, chunkRows, from, to)`. That reader
positions directly at a shard's first row: dense rows are fixed-width and sparse rows are found through the row
offsets. Shard *i* of N goes to node *i* (one coordinator thread per shard reads it and streams it with its own
`IngestStream`), to Spark partition *i* (read inside the task) and to Flink subtask *i* (a source with a static split
assignment, then a chained reader). Rows come out as CSR `PointBlock` chunks. Engine APIs get per-row views
(`WeightedPoint`) only where they take them. Nothing materialises the dataset on a driver.

Dataset types:

- `public` reads a converted testbed binary in place: `name`, `variant` (`raw`, `zscore` or `tfidf`), and the
  default cache `~/.cache/coreset-testbed/<name>/<name>[.<variant>].cbin|.csr`. The harness never downloads or
  converts; it points you to `coreset-testbed datasets convert`.
- `gaussian-mixture` and `esa22` are the testbed generators. They are written once to
  `<cacheDir>/datasets/<id>-<key>.cbin`.
- `testbed` takes any testbed spec. It is written once as `.cbin`, or as `.csr` when under half of the entries are
  non-zero.
- `file` reads an existing `.cbin` or `.csr`.

The header CRC32C identifies the content in records and in the reference cache. A dataset's `exclude` list skips
systems or algorithms on it. The real grid uses it to skip `mean` on the sparse NYTimes data.

**Timing split** (per repetition, seconds):

| field | grpc | spark | spark-mllib | flink |
|---|---|---|---|---|
| `build_s` | shards read and streamed, all batches acknowledged | `SparkCoresets.build`, including the driver's final merge | – | job submission to the final snapshot collected (read, local trees, merge stage) |
| `gather_s` | snapshots of all live nodes, unioned | inside `build_s` | – | inside `build_s` |
| `solve_s` | best of `solver.restarts` k-means++ plus weighted Lloyd on the coreset, the same code for every coreset system (`WeightedLloyd.bestOf`, seed of the repetition) | same | `KMeans.fit` on all points | same as grpc |
| `total_s` | build + gather + solve | build + solve | fit | build + solve |
| `eval_s` | one exact streaming pass over all rows for the deployed centers (`FullCost`, `evalThreads` threads, in the orchestrator) | same | same | same |

`points_per_s` = n / `build_s` (for MLlib n / fit time).

**Resources.** `ProcessSampler` samples RSS and CPU time of the cell JVM and all its descendants every `sampleMillis`.
These are the node JVMs and the Spark executors. Linux reads `/proc/<pid>/stat`, macOS uses one `ps` call per sample.
Per process it reports the peak RSS seen and the CPU seconds used inside the repetition. `cpu_s` sums all processes.
Each record lists every process with its role: `coordinator`, `node-i`, `driver`, `executor` or `minicluster`. The sampler never fails a cell. Process enumeration is retried with
backoff: on macOS the JDK's native call fails with ENOMEM when the process table grows mid-read. If it still fails,
the sample covers the processes already known, and the record counts the partial samples in `sampler_errors`.

**Bytes shipped.**

| system | `bytes_ingest` (raw points) | `bytes_summary` (coresets / results) | source |
|---|---|---|---|
| grpc | `Ingest` request bytes | `Snapshot` response bytes | per-node Prometheus counters `grpc_server_<Method>_bytes_{in,out}_total` (serialized protobuf messages, no HTTP/2 framing), scraped before and after each repetition |
| spark | 0 (tasks read their shard) | shuffle bytes read (local plus remote) plus task result bytes | `SparkListener` task metrics per job group; `bytes_partials_encoded` is `BuildMetrics.shippedEncodedBytes` |
| flink | 0 | `numBytesOut` of every task | an in-JVM `MetricReporter` (`FlinkBytesReporter`); remote clusters need a Prometheus reporter instead and report `bytes_available: false` |

**Quality.** The orchestrator evaluates each repetition after its cell JVM has exited, so the evaluation competes
with no system.

- **Reference solution.** The cheapest of `referenceRuns` best-of-`referenceRestarts` Lloyd solutions on the full
  data, by exact full cost. It is cached per dataset checksum, k and settings in `<cacheDir>/reference/`.
- **`cost_full`** is the exact cost of the deployed centers over all rows, from one streaming pass.
  **`cost_ratio`** = `cost_full` / reference cost. Values below 1 mean the run beat the reference.
- **`weight_error`** = |coreset weight − input weight| / input weight.
- **Distortion** follows `validate.Distortion`: ratio cost(C,Q)/cost(P,Q) and distortion max(r, 1/r), reported as max
  and mean over a candidate set. The candidates are `candidates` k-means++ seedings and Lloyd solutions on P, the
  reference, `candidates` Lloyd solutions on the coreset, and the deployed centers. Full-data costs of the shared
  candidates are computed once per dataset in one parallel pass. A unit test checks that the aggregation equals
  `Distortion.evaluate`.
- **Large n.** Above `maxInMemoryPoints` rows (default 2M), the candidates and the reference are computed on a
  uniform sample of that many rows. Selection sampling picks exactly that many in one pass, each weighted n/s.
  Every cost is still an exact pass over all rows, so only the candidate generation is sampled. The record sets
  `distortion_sampled_candidates: true` and the reference info sets `onSample: true`.

## Fairness rules

| rule | how the harness enforces it |
|---|---|
| same data | one point file, same contiguous shards for every system, recorded CRC32C |
| same algorithm and parameters | SPI id and parameter map passed unchanged to nodes (`SessionSpec`), `SparkCoresets.build` and `CoresetConfig` |
| same solver | `WeightedLloyd.bestOf` with the same restarts, iterations and seed for every coreset; MLlib uses its own k-means\|\| with `maxIter = solver.iterations` (it has no restarts) |
| same parallelism | N nodes × `workersPerNode` (default 1) workers, N coordinator shard threads; `local[N]`; N single-core executors; Flink parallelism N with N slots |
| same memory budget | `memory.workersMB` is the workers' heap: each grpc node and each local-cluster executor gets workersMB/N; Spark local and the Flink MiniCluster (one JVM) get all of it. Coordinator and driver JVMs of multi-process modes get `memory.driverMB` |
| warm-up excluded | `warmup` repetitions run in the same JVM and are not recorded |
| ≥ 3 repetitions | `repetitions` (default 3); `summary.md` shows median [min–max] |
| record JVM flags and versions | every record has `cell_jvm_args`, `worker_jvm_flags`, heap sizes; `environment.json` has host, CPU, memory, Java, Spark, Flink and gRPC versions, git commit, load averages |
| startup separate | `startup_s`, never part of `total_s` |
| one host | stated per run; see [Limitations](#limitations) |

Known asymmetries, which are properties of the systems and are kept rather than hidden:

- The gRPC design pushes raw points from the coordinator to the nodes, so it ships the input once over loopback
  (`bytes_ingest`). Spark tasks and Flink subtasks read their shard from storage.
- Final coreset sizes differ. Spark compresses on the driver, so it ends near m points. gRPC unions N node snapshots,
  each a union of its tree levels. Flink's final snapshot is the union of the merge stage's levels. `coreset_size` is
  recorded, and `solve_s` depends on it.
- Spark local and Flink MiniCluster are one JVM; gRPC and Spark local-cluster are N+1 JVMs. RSS sums per system
  include each extra JVM's baseline.
- Flink fault cells checkpoint every 200 ms (`checkpointMs` overrides) in all their repetitions, baselines included.
  Undisturbed cells do not checkpoint unless `checkpointMs` is set.

## Faults

All faults are real: processes are killed with SIGKILL (`destroyForcibly`) or tasks throw. Nothing is simulated.
A fault can race past the work on a loaded host. For example, a Spark executor finishes its task between the
listener event and the signal, or a Flink subtask reaches its failure point before any checkpoint completes. The
system then marks the repetition `fault_missed`. The cell reruns it, up to `maxAttempts` times (default 3), and
records `fault_attempts`. Only a missed record has no recovery figures; a record without `fault_missed` claims a
fault that took effect.

- **`kill-nodes` (grpc).** When a fraction `at` of the points has been submitted, `count` node processes are killed
  (the highest indices). The coordinator in `coreset-grpc` fails a whole ingest or gather when one node is gone. The
  harness therefore implements survivor handling itself, with no change to the coordinator. There is one single-node
  coordinator per node, and sessions are still created through one coordinator, so node *i* keeps partition *i*. A
  node is killed and the harness waits for its process to exit before ingest continues, so no later call can reach
  it. A node whose stream fails is declared dead; the detection time is recorded as `detect_s_node<i>`. Its remaining shard
  is dropped (`afterKill: drop`, recorded as `dropped_rows`) or rerouted round-robin to live nodes
  (`afterKill: reroute`). The final coreset is gathered from live nodes only. With `snapshotAt`, every node is
  snapshotted at that fraction first, in continuous mode with no wait, and a dead node's last snapshot is added to
  the union. **Lost weight** = input weight − (applied weight of the survivors + applied weight in the victims'
  snapshots). `reweight: true` scales the union by input weight / realized weight. Killed nodes are respawned before
  the next repetition; the time is recorded as `respawn_s`, not timed. On a cluster, `cluster.grpcKillCommand` (a
  shell template with `{host}` and `{port}`) does the kill.
- **`kill-executor` (spark local-cluster).** The executor running a task of the repetition is killed `at` × the
  mean task time of the last undisturbed repetition after the first task started. So the kill lands inside a task
  even with one partition per executor. The standalone worker starts a replacement and Spark recomputes the lost
  task. Recorded: `executor_removed_s` (driver notices), `replacement_registered_s`, failed task attempts, build time
  against the baseline, and `equal_to_baseline`. The input is a deterministic function of the partition and
  `SparkCoresets` derives all randomness from (seed, partition, level, sequence), so the recovered coreset should be
  bit-identical.
- **`task-failure` (flink MiniCluster).** The last subtask throws once, in its first attempt, after mapping a
  fraction `at` of its shard and after at least one checkpoint completed. The job restarts (fixed-delay, 100 ms)
  from the checkpoint. Recorded: `restore_s` (failure to the failed subtask's first record after the restore),
  `replayed_rows`, build time against the baseline, and `equal_to_baseline`. Count triggers only, so bit-identical is
  expected.
- **Lossy transport hook** (`lossy:`, off by default). `LossyInterceptors` has a gRPC client interceptor slot, used on
  the coordinator's channels, and a server interceptor slot for embedded servers. They drop messages of the selected
  methods (default `Ingest`) according to a `ChannelModel`: `bernoulli` (p), or `gilbert-elliott` (good/bad state
  transition probabilities and per-state loss rates). The stock ingest protocol never recovers a dropped batch: the
  stream waits for its acknowledgement, and the repetition fails at `flushTimeoutSeconds`. This hook is for the
  loss-resilient protocol work and is not used in the presets.

## How to run

### Single host

```
JAVA_HOME=<jdk17> coreset-experiments/run-engines.sh                      # smoke-1m preset
JAVA_HOME=<jdk17> coreset-experiments/run-engines.sh grid                 # full synthetic preset grid (hours)
JAVA_HOME=<jdk17> coreset-experiments/run-engines.sh real                 # real-dataset grid (many hours)
JAVA_HOME=<jdk17> coreset-experiments/run-engines.sh real-smoke           # covertype, N = 2, every system and fault
coreset-experiments/run-engines.sh my-grid.yaml --out results/engines/x --only 'grpc|flink'
coreset-experiments/run-engines.sh smoke-1m --no-build --summary-only --out results/engines/smoke-1m
```

The script builds with `mvn -B -pl coreset-experiments -am package -DskipTests` (never `install`). It then starts
`ExperimentRunner` with `target/classes` plus `target/runtime-classpath.txt`, which the build writes. Cells, nodes and
executors inherit that class path. `--only` filters cells by a regex on the cell id. Results go to
`results/engines/<name>/` (git-ignored). Configs: [configs/preset-grid.yaml](../../coreset-experiments/configs/preset-grid.yaml)
(systems × N ∈ {1, 2, 4, 8} × {uniform, sensitivity, mean, lightweight} × synthetic datasets),
[configs/smoke-1m.yaml](../../coreset-experiments/configs/smoke-1m.yaml),
[configs/real-grid.yaml](../../coreset-experiments/configs/real-grid.yaml) and
[configs/real-smoke.yaml](../../coreset-experiments/configs/real-smoke.yaml).

The real grid uses covertype, census1990, kddcup99, yearmsd, susy and higgs (all 11M rows), z-scored, plus
nytimes with tf-idf (sparse, d = 102,660). It runs grpc, Spark local-cluster and Flink at N ∈ {1, 4, 8} with uniform
and sensitivity (k = 25, m = 4000), mean (m = 500) and lightweight (m = 4000), and the MLlib k-means|| baseline, all
with 3 repetitions. Fault cells are node kills (1 of 4 and 2 of 8, each before any snapshot and after one), a Spark
executor kill and a Flink task failure. Every system handles the sparse rows: MLlib gets sparse vectors, and the
quality pass compares sparse rows with dense centers in O(nnz) per row. Only `mean` is excluded on nytimes, because
its compressions keep m dense centroid sums of 102,660 doubles each. The real grid has not been run at full scale
yet.

`spark/local-cluster` launches executors like a Spark distribution would. The runner gives every cell JVM
`SPARK_HOME=<out>/spark-home` (an empty `jars/`), `SPARK_SCALA_VERSION` and `SPARK_LOCAL_IP=127.0.0.1`.
`spark.executor.extraClassPath` carries the harness class path.

System options (`systems[].options`): grpc `workersPerNode`, `queueCapacity`, `maxInFlight`, `flushTimeoutSeconds`;
spark `treeDepth`, `chunkSize`, `partitionsPerWorker`, `readPauseMicrosPerChunk` (test only: sleeps after every read
chunk so tasks last long enough for a kill to land inside one); spark-mllib `tol`; flink `chunksPerPartial`,
`snapshotEveryPartials`, `compressSnapshots`, `chunkSize`, `checkpointMs`, `restartDelayMs`. Faults can be
restricted with `parallelism: [...]` and `algorithms: [...]`.

### Cluster

The same grid runs against real targets. Set the systems to `grpc/external`, `spark/cluster` and `flink/remote`, and
fill in `cluster:` (an example is commented at the end of the preset):

- **gRPC:** start `java -jar coreset-node.jar --bind 0.0.0.0 --metrics-bind 0.0.0.0 ...` on each host, with the
  plug-in jar on its class path. List them in `grpcTargets` and their `/metrics` URLs in `grpcMetrics`. The
  coordinator runs in the cell JVM on the submitting host. Kill faults need `grpcKillCommand`.
- **Spark:** `sparkMaster: spark://host:7077`, plus `sparkJars` with the coreset jars from
  `target/runtime-classpath.txt` (core, spark, experiments, examples). N single-core executors are requested.
- **Flink:** `flinkJobManager: host:port` (REST) and `flinkJars`. Byte counts need a Prometheus reporter on the
  cluster; task-failure timing needs the in-JVM MiniCluster.
- The dataset file must exist at the same path on every worker (shared file system or a copy).
  Generated data is materialised by `PointSources.open` on the submitting host.

## Output

`<out>/results.jsonl` has one JSON object per repetition (`phase` = `measure`, `baseline`, `fault` or `error`):
identity (`cell`, `dataset`, `system`, `parallelism`, `algorithm`, `fault_label`, `seed`), the timing split,
`points_per_s`, `cpu_s`, `peak_rss_mb_max`, `peak_rss_mb_sum`, `processes[]`, `coreset_size`,
`coreset_fingerprint` (SHA-256 of entries, weights and offset), `metrics{}` (bytes and system counters), `fault{}`,
the quality fields, `centers`, the load average before and after, and the JVM flags. `<out>/summary.md` aggregates it
(median [min–max]). `<out>/environment.json` holds the host and versions. `<out>/cells/<id>/` holds `cell.json`,
`cell.log`, node logs and the raw `records.jsonl`.

## Preliminary results

**Preliminary, incomplete, and taken on a loaded host.** The machine was an M2 Max (12 cores, 96 GB), macOS, Corretto
17.0.5, Spark 4.0.5, Flink 2.3.0, gRPC 1.85.1. Other jobs were running on it at the same time: the 1-minute load
average was 31–37 during the runs below, on 12 cores. The full `smoke-1m` grid was stopped after three cells so the
harness could move to the shared binary point format. The authoritative grid runs later, in a quiet window.

Dataset `gm-1m-d20`: Gaussian mixture, n = 1,000,000, d = 20, 20 clusters with Zipf(1) sizes, σ = 5. k = 20, solver
best of 10 restarts × 50 iterations. Reference cost 5.0025e8: 3 runs × 5 restarts, 45 s, cached. `grpc/spawn` with
N = 1, warm-up excluded, 3 repetitions, values as median [min–max]:

| algorithm (m) | startup s | build s | gather s | solve s | points/s | CPU s | peak RSS max / sum MB | ingest bytes | snapshot bytes | coreset pts | cost ratio | distortion max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform (2000) | 1.78 | 1.03 [0.87–1.61] | 0.031 [0.022–0.033] | 0.139 [0.114–0.150] | 9.7e5 [6.2e5–1.1e6] | 2.5 [2.1–3.3] | 983 / 1790 | 192.7 MiB | 1.4 MiB | 7,227 | 1.0078 [1.0069–1.0084] | 1.033 [1.029–1.063] |
| sensitivity (k=20, 2000) | 1.59 | 1.04 [1.02–1.05] | 0.345 [0.304–0.440] | 0.161 [0.128–0.166] | 9.6e5 | 3.2 [3.2–4.6] | 1104 / 1760 | 192.7–214.8 MiB | 1.3–1.5 MiB | 7,482 | 1.0066 [1.0060–1.0071] | 1.046 [1.043–1.084] |
| mean (2000) | – | 196 | 61 | 0.12 | 5.1e3 | 284 | – | – | – | 10,576 | – | – |

Readings so far:

- On one node, 1M points are ingested at about 1M points/s. The solution is within 0.6–0.8 % of the reference.
  Measured distortion is at most 1.03–1.08.
- The coordinator ships the raw input once: 193 MiB for 160 MB of doubles. The final summary is 1.4 MiB, 0.7 % of
  the input. Sensitivity runs sometimes ship up to 11 % more ingest bytes; these are resent batches (BUSY
  backpressure), which the stock protocol retries.
- `mean` with m = 2000 needs 196 s of ingest plus 61 s until all accepted batches are applied. Each compression is a
  Lloyd run with 2000 centers. That is why `smoke-1m` uses m = 500 for `mean`, and why `mean` is the long pole of the
  preset grid. gRPC acknowledges a batch when it is queued. For gRPC, only `build_s + gather_s` is comparable with
  the other systems.

An end-to-end check on a small instance (n = 200,000, d = 8, N = 2, `EngineGridSmokeTest`) and a 20,000-point grid
on the same host gave these observations. They are functional only; the inputs are too small for meaningful
timings:

- all five systems produce coresets or centers within 1.0–1.04× of the reference cost;
- killing one of two gRPC nodes halfway with `drop` loses 50 % of the input weight. The survivor's coreset still
  gives cost ratio 1.02, because the data is shuffled. Detection takes 10–30 ms on loopback;
- a killed Spark executor (exit 137) is noticed by the driver within about 30 ms. Its task is recomputed on the
  other executor and the final coreset is bit-identical to the undisturbed run;
- a Flink task failure restores from the 200 ms checkpoint in about 0.16 s. It replays the rows after the
  checkpoint, and the final snapshot is bit-identical to the undisturbed run.

Real-data check (`real-smoke`): covertype z-scored (581,012 × 54), N = 2, sensitivity (k = 25, m = 4000), one
repetition after warm-up, on the same loaded host. These numbers are functional, not a benchmark:

| system | build s | gather s | solve s | coreset pts | cost ratio | distortion max | bytes summary |
|---|---|---|---|---|---|---|---|
| grpc/spawn | 0.68 | 0.73 | 1.68 | 22,385 | 1.01 | 1.05 | 10.8 MiB (plus 281.5 MiB raw ingest) |
| spark/local-cluster | 0.93 | – | 0.15 | 1,796 | 1.04 | 1.11 | 2.1 MiB |
| flink/minicluster | 1.27 | – | 1.10 | 10,052 | 1.01 | 1.04 | 19.9 MiB |
| spark-mllib/local | – | – | 10.58 (fit) | – | 1.03 | – | 0.7 MiB |

Faults on the same data:

- killing one of two nodes after a snapshot at 50 % (kill at 75 %, `drop`) lost 31 % of the weight, and the cost
  ratio went from 1.01 to 1.04;
- the Spark executor kill slowed the build 2.3×, the replacement registered 2.0 s after the kill, and the coreset
  was bit-identical;
- the Flink task failure restored in 0.24 s, and the snapshot was bit-identical.

The final coreset sizes show the asymmetry described above. Spark compresses on the driver; gRPC and Flink return
unions of tree levels, which costs them solve time and gives slightly better cost ratios.

Run `coreset-experiments/run-engines.sh smoke-1m` for the full grid.

## Limitations

- **One host.** All "distributed" processes share 12 cores (8 performance and 4 efficiency on an M2 Max), one memory
  bus and loopback networking. Parallel speedup is bounded by the host, network costs are loopback costs, and node or
  executor failures never involve a network partition. macOS has no CPU pinning, so the processes of a system are not
  confined to N cores; N bounds the parallelism the system uses, not the cores it can touch.
- **Loaded host.** The preliminary run shared the machine with other jobs (load averages in the summary). Times are
  indicative only. Quality numbers are not affected.
- **Sampling.** Peak RSS is sampled (200 ms), so it is a lower bound. JVM RSS includes heap that was reserved but not
  needed.
- **Flink metrics and faults** need the in-JVM MiniCluster. Remote Flink runs report no bytes and support no faults.
  Spark executor kills need local-cluster mode. Spark `local[N]` has no separate executor process.
- **gRPC survivor handling lives in the harness**, not in `CoresetCoordinator`. Lost data is not recovered (no
  replication or re-ingest). Loss-resilient ingest (per-draw packetization, realized-count decoding, coreset
  replication) is future work; the lossy-transport hook exists for it.
- **Bytes** are serialized message sizes (gRPC), Spark's shuffle and result accounting, and Flink's network buffer
  bytes. They are comparable in order of magnitude, not byte for byte: framing, compression and RPC overheads
  differ.
- **Small-n faults.** On tiny inputs a fault may fire after the work is done. The kill points are progress-based
  (gRPC, Flink) or task-time-based (Spark) so they land mid-job at realistic sizes. Records show when a fault was
  skipped (`kill_skipped`, `fail_skipped`).
