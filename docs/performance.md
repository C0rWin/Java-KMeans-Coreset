# Performance

All numbers on this page come from one run of `coreset-bench/run-benchmarks.sh` (full mode) on commit 847492f,
on a quiet host. The raw outputs (JMH JSON and text, harness markdown, JFR recordings and `jfr view`
summaries, environment) were kept with the run; the tables below are copied from them. The
[Optimizations](#optimizations) section summarizes the efficiency work that led here, with before/after numbers
from the individual changes. The legacy Thrift vs gRPC comparison is documented in its own section, added together
with the Thrift removal.

## How to run

Everything (build, JMH, ingestion harness, Thrift parity, JFR profiles) is one command. It writes raw JMH JSON, the
harness markdown, JFR recordings with `jfr view` summaries and a `summary.md`, and records host, JVM, git revision and
the load average before and after every phase:

```
JAVA_HOME=<jdk17> JFR_TOOL=<jdk21+>/bin/jfr coreset-bench/run-benchmarks.sh            # full, about 40 min
JAVA_HOME=<jdk17> coreset-bench/run-benchmarks.sh --quick                              # about 10 min
coreset-bench/run-benchmarks.sh --skip-build --no-jfr --jmh 'CodecBenchmark|DistanceBenchmark'
```

Results go to `coreset-bench/results/<UTC timestamp>/` (git-ignored). `JFR_TOOL` is optional; JDK 17's `jfr` has no
`view` command, so without it only `jfr summary` is written. Individual pieces:

```
mvn -B -pl coreset-bench -am package -DskipTests
java -jar coreset-bench/target/benchmarks.jar -prof gc -rf json -rff jmh.json [regex]
java -Xms8g -Xmx8g -XX:+AlwaysPreTouch -cp coreset-bench/target/benchmarks.jar \
     io.github.c0rwin.coreset.bench.perf.NodeIngestHarness [--points N --workers 1,2,4,8 --grpc-nodes 1,2,3,4 ...]
java -Xms8g -Xmx8g -XX:+AlwaysPreTouch -cp coreset-bench/target/benchmarks.jar \
     io.github.c0rwin.coreset.bench.perf.ParityHarness [--points N --repeats R ...]
```

`mvn -B -pl coreset-bench -am verify` runs `BenchSmokeTest`: one JMH benchmark in process and both harnesses on toy
sizes. The harnesses check themselves (every batch applied, gathered snapshot complete), so the smoke test fails on a
broken pipeline, not only on a crash.

## Environment

- date (UTC): 2026-11-07T23:04:24Z
- git: 847492f (integration), clean
- os: Darwin 27.0.0 arm64
- cpu: Apple M2 Max, 12 logical CPUs (8 performance + 4 efficiency), 96 GiB RAM
- java: Amazon Corretto 17.0.5 (17.0.5+8-LTS), G1 (default)
- harness JVM flags: `-Xms8g -Xmx8g -XX:+AlwaysPreTouch`

| phase | load avg before (1/5/15 min) | load avg after | seconds |
|---|---|---|---|
| build | 3.13 5.46 6.38 | 5.87 5.99 6.55 | 13 |
| jmh | 5.87 5.99 6.55 | 11.54 9.54 8.39 | 1786 |
| node-ingest | 11.54 9.54 8.39 | 11.91 9.69 8.45 | 11 |
| parity | 11.91 9.69 8.45 | 11.28 9.59 8.43 | 7 |
| jfr-node | 11.28 9.59 8.43 | 11.82 9.73 8.48 | 2 |
| jfr-grpc | 11.82 9.73 8.48 | 11.82 9.73 8.48 | 3 |
| jfr-kmeans-seed | 11.82 9.73 8.48 | 11.74 9.79 8.52 | 9 |
| jfr-kmeans-lloydIteration | 11.74 9.79 8.52 | 11.17 9.73 8.51 | 8 |

The load average rises during the JMH phase because the benchmarks
themselves are multi-threaded (forked JVMs, JIT and GC threads, the 12-thread metrics cases); it stays below the
core count throughout.

## Methodology

**Machine.** No core pinning or frequency control is available on macOS, and the scheduler may place a fork on
efficiency cores. Compare ratios within this run rather than with numbers from other machines.

**JMH** (`coreset-bench/src/main/java/.../bench/jmh`, JMH 1.37):
- Most benchmarks: `@Fork(2)`, `@Warmup(3 x 1 s)`, `@Measurement(5 x 1 s)`. `SamplingBenchmark`: 1 fork,
  2 x 1 s warmup, 4 x 1 s measurement. `SparseBenchmark` and `LloydBenchmark`: 1 fork, 2 x 2 s warmup,
  4 x 2 s measurement. Average time per operation, 99.9% confidence intervals.
- Each fork runs with a fixed heap (`-Xms` = `-Xmx`, 2 or 4 GiB). Every run uses `-prof gc`; `B/op` is
  `gc.alloc.rate.norm`, allocated bytes per operation, which is exact and insensitive to load.
- Inputs come from `@State` with seeded generators. Results go to `Blackhole` or are returned. Distance and sampler
  kernels walk 256 distinct pairs or 1024 draws per invocation (`@OperationsPerInvocation`), so one pair cannot be
  memorised by caches or branch predictors.
- `CompressBenchmark` runs `mean` with 3 Lloyd iterations instead of the default 20 (its cost is linear in the
  iteration count).
- `KMeansBenchmark.seed` is the default greedy k-means++ (2 + ⌊ln k⌋ candidates per step); `seedPlain` is plain D²
  seeding. Both use the `List` API, so they include one copy of the input into a point block.

**Ingestion harness** (`NodeIngestHarness`, not JMH): producers cycle over 256 distinct pre-generated 1000-point
batches (Gaussian mixture, d = 16, 10 clusters) with unique seqs, so every batch is compressed with its own
seq-derived generator while the resident data stays small. Each configuration runs once as warmup, then 3 times;
tables give the median points/s with the min-max range, and the other columns come from the median run. Throughput
is end to end: first ingest until `awaitApplied` (in process) or until the gathered snapshot is complete (gRPC).
Latency percentiles come from the node's own `RecordingMetrics`. `node.ingest.latency_ns` is the producer-side
`ingest` call, all outcomes included; batch compress is the worker-side `node.batch.compress` span. A BUSY reply is
retried after 50 us, and the BUSY rate is `busy / (busy + batches)`. Allocation is the sum of
`ThreadMXBean.getThreadAllocatedBytes` over all live threads, taken before the workers exit. Peak heap is the sum of
per-pool peaks after `resetPeakUsage`, an upper bound; the baseline in brackets is heap in use after a full GC at the
start. gRPC runs put the client and every node in one JVM over Netty loopback, so they compete for the same cores.

**Profiling.** JFR with `settings=profile` on four runs: in-process ingestion (2M points, W = 4), gRPC ingestion
(1M points, 2 nodes with 1 worker each), and `KMeansBenchmark.seed` and `lloydIteration` at n = 100k, k = 50,
d = 16. Summaries come from `jfr view hot-methods|allocation-by-site|allocation-by-class|gc-pauses`. The recordings
hold 190-600 execution samples each, so shares are good to a few percent.

## JMH results

Highlights:

| what | result |
|---|---|
| union of two 1000-point coresets | 0.018 us, 144 B |
| 8-way union fold | 0.214 us |
| merge-and-reduce tree snapshot | 0.161 us |
| Lloyd, 20 iterations, dense 200k × 10 | 156 ms (k = 10), 501 ms (k = 50) |
| Lloyd, 20 iterations, sparse 100k × 10⁶, k = 10 | 327 ms, 48 MB/op |
| metrics `record`, 1 / 12 threads | 12.5 / 22.6 ns |
| metrics span (default, no probes) | 41 ns, 48 B |

#### AliasSamplerBenchmark

| benchmark | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|
| build | 10000 | 80,369 | 2,621 | ns/op | 120,056 |
| build | 1000000 | 7,965,791 | 34,991 | ns/op | 12,000,060 |
| draw | 10000 | 5.15 | 0.0248 | ns/op | 0 |
| draw | 1000000 | 4.74 | 0.712 | ns/op | 0 |

#### CodecBenchmark

| benchmark | nnz | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|
| protoDecode | 16 | 277 | 2.33 | us/op | 925,520 |
| protoDecode | 128 | 1,405 | 36.9 | us/op | 4,621,693 |
| protoEncode | 16 | 313 | 3.87 | us/op | 742,237 |
| protoEncode | 128 | 2,061 | 19.3 | us/op | 6,233,145 |
| thriftDecode | 16 | 590 | 5.19 | us/op | 2,195,208 |
| thriftDecode | 128 | 6,184 | 25.4 | us/op | 17,121,083 |
| thriftEncode | 16 | 697 | 6.30 | us/op | 1,691,880 |
| thriftEncode | 128 | 4,928 | 200 | us/op | 11,985,755 |

#### CompressBenchmark

| benchmark | algorithm | m | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| compress | uniform | 200 | 10000 | 0.0183 | 0.000403 | ms/op | 45,453 |
| compress | uniform | 200 | 100000 | 0.112 | 0.00164 | ms/op | 45,900 |
| compress | uniform | 1000 | 10000 | 0.0558 | 0.00135 | ms/op | 217,514 |
| compress | uniform | 1000 | 100000 | 0.165 | 0.00361 | ms/op | 227,228 |
| compress | sensitivity | 200 | 10000 | 2.15 | 0.051 | ms/op | 251,425 |
| compress | sensitivity | 200 | 100000 | 21.3 | 0.469 | ms/op | 2,051,883 |
| compress | sensitivity | 1000 | 10000 | 2.22 | 0.0566 | ms/op | 441,438 |
| compress | sensitivity | 1000 | 100000 | 21.6 | 0.157 | ms/op | 2,252,330 |
| compress | mean | 200 | 10000 | 58.2 | 2.92 | ms/op | 1,475,719 |
| compress | mean | 200 | 100000 | 620 | 40.0 | ms/op | 7,861,987 |
| compress | mean | 1000 | 10000 | 261 | 2.48 | ms/op | 10,710,062 |
| compress | mean | 1000 | 100000 | 2,621 | 148 | ms/op | 16,951,859 |

#### CoresetOpsBenchmark

| benchmark | m | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|
| coresetOf | 1000 | 18.9 | 0.253 | us/op | 212,296 |
| cost | 1000 | 136 | 1.01 | us/op | 72 |
| protoDecodeCoreset | 1000 | 60.2 | 3.92 | us/op | 212,390 |
| sparkDecodeCoreset | 1000 | 81.1 | 0.763 | us/op | 212,272 |
| treeSnapshot | 1000 | 0.161 | 0.00123 | us/op | 368 |
| union2 | 1000 | 0.018 | 0.0002 | us/op | 144 |
| unionFold8 | 1000 | 0.214 | 0.00465 | us/op | 1,248 |

#### DistanceBenchmark

| benchmark | dim | nnz | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|
| sparseDense | 4096 | 16 | 20.5 | 0.565 | ns/op | 0 |
| sparseDense | 4096 | 128 | 115 | 1.03 | ns/op | 0 |
| sparseDense | 4096 | 1024 | 934 | 16.4 | ns/op | 0 |
| sparseDense | 1048576 | 16 | 25.6 | 0.581 | ns/op | 0 |
| sparseDense | 1048576 | 128 | 163 | 6.49 | ns/op | 0 |
| sparseDense | 1048576 | 1024 | 3,253 | 117 | ns/op | 0 |
| sparseSparse | 4096 | 16 | 66.2 | 3.29 | ns/op | 0 |
| sparseSparse | 4096 | 128 | 1,135 | 40.8 | ns/op | 0 |
| sparseSparse | 4096 | 1024 | 9,082 | 169 | ns/op | 0 |
| sparseSparse | 1048576 | 16 | 60.3 | 3.23 | ns/op | 0 |
| sparseSparse | 1048576 | 128 | 1,129 | 7.64 | ns/op | 0 |
| sparseSparse | 1048576 | 1024 | 9,730 | 256 | ns/op | 0 |

#### GrpcMetricsBenchmark

| benchmark | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|
| sendAndAck | 17.9 | 0.468 | ns/op | 0 |

#### KMeansBenchmark

| benchmark | d | k | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| lloydIteration | 16 | 10 | 20000 | 2.82 | 0.0278 | ms/op | 647,410 |
| lloydIteration | 16 | 10 | 100000 | 14.7 | 0.457 | ms/op | 3,207,416 |
| lloydIteration | 16 | 50 | 20000 | 8.81 | 0.401 | ms/op | 691,253 |
| lloydIteration | 16 | 50 | 100000 | 42.8 | 1.32 | ms/op | 3,251,271 |
| lloydIteration | 128 | 10 | 20000 | 16.1 | 0.329 | ms/op | 678,777 |
| lloydIteration | 128 | 10 | 100000 | 80.6 | 1.10 | ms/op | 3,238,813 |
| lloydIteration | 128 | 50 | 20000 | 57.4 | 3.46 | ms/op | 848,079 |
| lloydIteration | 128 | 50 | 100000 | 290 | 4.27 | ms/op | 3,408,183 |
| seed | 16 | 10 | 20000 | 13.9 | 0.169 | ms/op | 642,287 |
| seed | 16 | 10 | 100000 | 69.9 | 1.56 | ms/op | 3,202,317 |
| seed | 16 | 50 | 20000 | 71.9 | 0.78 | ms/op | 649,373 |
| seed | 16 | 50 | 100000 | 353 | 5.58 | ms/op | 3,209,517 |
| seed | 128 | 10 | 20000 | 62.6 | 0.591 | ms/op | 642,313 |
| seed | 128 | 10 | 100000 | 320 | 2.59 | ms/op | 3,202,416 |
| seed | 128 | 50 | 20000 | 326 | 2.84 | ms/op | 649,472 |
| seed | 128 | 50 | 100000 | 1,664 | 24.9 | ms/op | 3,209,874 |
| seedPlain | 16 | 10 | 20000 | 3.49 | 0.0871 | ms/op | 560,626 |
| seedPlain | 16 | 10 | 100000 | 17.3 | 0.277 | ms/op | 2,800,633 |
| seedPlain | 16 | 50 | 20000 | 16.5 | 0.124 | ms/op | 562,713 |
| seedPlain | 16 | 50 | 100000 | 79.6 | 0.77 | ms/op | 2,802,746 |
| seedPlain | 128 | 10 | 20000 | 19.3 | 0.23 | ms/op | 560,634 |
| seedPlain | 128 | 10 | 100000 | 94.1 | 1.86 | ms/op | 2,800,673 |
| seedPlain | 128 | 50 | 20000 | 100 | 1.09 | ms/op | 562,755 |
| seedPlain | 128 | 50 | 100000 | 488 | 16.2 | ms/op | 2,802,903 |

#### LloydBenchmark

| benchmark | d | k | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| lloyd20 | 10 | 10 | 200000 | 156 | 6.40 | ms/op | 6,405,770 |
| lloyd20 | 10 | 50 | 200000 | 501 | 36.3 | ms/op | 6,442,968 |

#### MergeReduceTreeBenchmark

| benchmark | algorithm | m | metrics | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| add | uniform | 200 | noop | 11.5 | 0.157 | us/op | 36,259 |
| add | uniform | 200 | recording | 12.1 | 0.157 | us/op | 36,401 |
| add | uniform | 1000 | noop | 61.2 | 0.476 | us/op | 179,224 |
| add | uniform | 1000 | recording | 63.8 | 1.60 | us/op | 179,380 |
| add | sensitivity | 200 | noop | 83.2 | 1.19 | us/op | 45,955 |
| add | sensitivity | 200 | recording | 85.3 | 1.58 | us/op | 46,129 |
| add | sensitivity | 1000 | noop | 431 | 5.87 | us/op | 223,143 |
| add | sensitivity | 1000 | recording | 432 | 10.7 | us/op | 223,322 |

#### MetricsBenchmark

| benchmark | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|
| allocatedBytesProbe | 47.9 | 0.872 | ns/op | 0 |
| count1 | 5.10 | 0.0539 | ns/op | 0 |
| count12 | 7.47 | 0.386 | ns/op | 0 |
| count4 | 5.27 | 0.0397 | ns/op | 0 |
| longAdder1 | 5.65 | 0.353 | ns/op | 0 |
| longAdder12 | 7.06 | 0.462 | ns/op | 0 |
| longAdder4 | 5.25 | 0.0243 | ns/op | 0 |
| record1 | 12.5 | 0.0878 | ns/op | 0 |
| record12 | 22.6 | 1.74 | ns/op | 0 |
| record4 | 12.6 | 0.157 | ns/op | 0 |
| span | 41.3 | 1.13 | ns/op | 48 |
| spanWithProbes | 1,303 | 15.3 | ns/op | 48 |
| threadCpuTimeProbe | 554 | 6.45 | ns/op | 0 |

#### SamplingBenchmark

| benchmark | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|
| bicriteria | 100000 | 255 | 20.8 | ms/op | 2,912,752 |
| bicriteria | 1000000 | 2,877 | 116 | ms/op | 29,016,536 |
| integrity | 100000 | 1.18 | 0.0156 | ms/op | 337 |
| integrity | 1000000 | 12.2 | 0.971 | ms/op | 423 |
| sensitivity | 100000 | 20.9 | 2.51 | ms/op | 2,252,296 |
| sensitivity | 1000000 | 207 | 14.7 | ms/op | 20,253,533 |
| sensitivityBicriteria | 100000 | 255 | 4.72 | ms/op | 3,957,138 |
| sensitivityBicriteria | 1000000 | 2,822 | 130 | ms/op | 37,268,136 |
| sensitivityWeighted | 100000 | 20.3 | 0.695 | ms/op | 2,252,062 |
| sensitivityWeighted | 1000000 | 203 | 14.3 | ms/op | 20,253,505 |
| uniform | 100000 | 0.162 | 0.0446 | ms/op | 227,227 |
| uniform | 1000000 | 1.01 | 0.0555 | ms/op | 228,234 |
| uniformWeighted | 100000 | 0.933 | 0.113 | ms/op | 1,427,009 |
| uniformWeighted | 1000000 | 8.51 | 0.274 | ms/op | 12,228,267 |

#### SparseBenchmark

| benchmark | k | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|
| bicriteria | 10 | 100000 | 265 | 15.0 | ms/op | 10,912,706 |
| cost | 10 | 100000 | 72.5 | 4.51 | ms/op | 91 |
| kmeansppSeed | 10 | 100000 | 75.6 | 3.27 | ms/op | 7,206,444 |
| kmeansppSeedPlain | 10 | 100000 | 18.9 | 2.81 | ms/op | 10,800,646 |
| lloyd20 | 10 | 100000 | 327 | 23.8 | ms/op | 48,305,814 |
| meanCompress | 10 | 100000 | 1,175 | 183 | ms/op | 50,394,138 |
| sensitivityCompress | 10 | 100000 | 23.5 | 3.71 | ms/op | 10,173,433 |
| uniformCompress | 10 | 100000 | 0.15 | 0.0171 | ms/op | 151,437 |

## Ingestion throughput

### In-process CoresetNode

1M points in batches of 1000, sensitivity (k = 10, m = 200), 2 producers, queue 64:

| mode | W | points/s (median) | min-max | ingest p50 / p99 (us) | batch compress p50 / p99 (us) | BUSY rate | alloc MB/s | GC n / ms | peak heap MiB (baseline) | snapshot pts | weight err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single-writer | 1 | 3081185 | 3048461-3137809 | 18.8 / 39.2 | 237 / 264 | 0.8539 | 330 | 0 / 0 | 1589 (81) | 800 | 3.2e-03 |
| single-writer | 2 | 6255717 | 6129446-6335531 | 18.8 / 30.3 | 230 / 264 | 0.7168 | 668 | 0 / 0 | 913 (81) | 1599 | 7.5e-03 |
| single-writer | 4 | 12734142 | 12640940-12838900 | 25.7 / 32.4 | 228 / 251 | 0.4681 | 1352 | 0 / 0 | 577 (81) | 3207 | 9.4e-03 |
| single-writer | 8 | 19912452 | 18386057-20561465 | 27.0 / 40.7 | 230 / 840 | 0.0610 | 2098 | 0 / 0 | 409 (81) | 6789 | 2.7e-03 |
| cas | 1 | 3120235 | 3057429-3136528 | 18.8 / 36.6 | 232 / 272 | 0.8462 | 334 | 0 / 0 | 1517 (81) | 801 | 9.3e-03 |
| cas | 2 | 6286188 | 6274803-6389504 | 20.6 / 37.1 | 230 / 251 | 0.7061 | 672 | 0 / 0 | 885 (81) | 778 | 6.7e-03 |
| cas | 4 | 12286220 | 10594085-12417257 | 25.5 / 36.6 | 232 / 383 | 0.4559 | 1314 | 0 / 0 | 561 (81) | 783 | 3.1e-03 |
| cas | 8 | 19760503 | 19375496-21748527 | 26.8 / 31.9 | 230 / 440 | 0.0610 | 2112 | 0 / 0 | 413 (81) | 773 | 5.2e-04 |

### gRPC over Netty loopback

1M points, 2 workers per node (single-writer), 2 producers, client and nodes in one JVM:

| nodes | points/s (median) | min-max | node ingest p50 / p99 (us) | BUSY resends | alloc MB/s | GC n / ms | peak heap MiB (baseline) | proto bytes out / point | snapshot pts | weight err |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3114481 | 2925932-3313240 | 4.2 / 15.4 | 0 | 2852 | 0 / 0 | 1613 (85) | 166.0 | 1564 | 5.5e-03 |
| 2 | 5224475 | 4799302-5876567 | 4.5 / 15.8 | 0 | 4809 | 0 / 0 | 1642 (86) | 166.0 | 3251 | 1.4e-03 |
| 3 | 5636049 | 5580935-5912693 | 5.1 / 17.8 | 0 | 5292 | 0 / 0 | 1682 (86) | 166.0 | 4008 | 4.2e-03 |
| 4 | 5373758 | 4973934-5913161 | 5.3 / 16.2 | 0 | 5104 | 0 / 0 | 1718 (86) | 166.0 | 6790 | 1.9e-03 |

**Reading the tables.** In process, throughput scales almost linearly with workers: 3.08, 6.26, 12.7 and 19.9 M
points/s at W = 1, 2, 4, 8 (single-writer; CAS is within a few percent). Batch compress stays at about 230 us
(p50) for 1000 points regardless of W, so the gain is parallelism, not contention relief. BUSY is backpressure
working as designed: two producers outpace one or two workers, so most offers bounce until W = 8. No GC ran during
any measured run.

Over gRPC, one node reaches 3.1 M points/s and three nodes 5.6 M points/s; a fourth node adds nothing because the
client, Netty and all nodes share the 12 cores of one JVM. Allocation runs at 2.9-5.3 GB/s (see the JFR section for where it goes). Each point costs 166 bytes on the wire at d = 16 in the sparse encoding. A run with separate node
processes is described in [distributed-run.md](distributed-run.md).

Single-writer snapshots are W times larger than CAS snapshots (W private trees against one shared tree), as
documented in `CoresetNode`. The weight error column is |sum of coreset weights - input weight| / input weight.
Sensitivity sampling is unbiased, not weight-preserving, so errors of 5e-4 to 9e-3 after hundreds of merges are
sampling noise, not loss.

## Hotspots (JFR)

| recording | top CPU methods (share of samples) | top allocation sites | GC pauses |
|---|---|---|---|
| in-process ingest (W = 4) | `Kernels.merge` 46%, `Seeding$D2.add` 23%, `AliasSampler.ofOwned` 4.7%, `PointBlock.of(List)` 4.7%, `Session.ingest` 4.2% | `PointBlock$Builder.<init>` 79%, `SparseVector.dense` 13% | 9, total 45.5 ms |
| gRPC ingest (2 nodes) | `Kernels.merge` 27%, protobuf `DoubleArrayList.addDouble` 9.7%, protobuf `SparseVector$Builder.mergeFrom` 7.7% + 2.3%, `Seeding$D2.add` 5.4%, protobuf `SparseVector.writeTo` 3.1% | `PointBlock$Builder.<init>` 62%, protobuf builders and lists 14% | 9, total 85.4 ms |
| k-means++ seed (greedy, k = 50) | `Kernels.merge` 57%, `Greedy.boundedPotentials` 21%, `Greedy.absorb` 13%, `RowCandidates.squaredDistances` 5.7% | `Greedy.<init>` 30%, `SparseVector.dense` 20%, `KMeansPlusPlus.seed` 18% | 2, total 6.7 ms |
| Lloyd iteration (k = 50) | `Kernels.dense` 78%, `Centroids$Dense.nearest` 14%, `Kernels.addTo` 3.0% | `Centroids.<init>` 39%, `WeightedLloyd.run` 31%, `KMeansResult.<init>` 16% | 5, total 19.1 ms |

What this says:

- Every profile is dominated by the distance kernels `Kernels.merge` (merge of two sorted supports) and
  `Kernels.dense` (point against a dense center), which is where the time should go. Copying and bookkeeping are each below 5%.
- On the ingest paths most allocation is the point blocks themselves (`PointBlock$Builder`), i.e. the data being
  ingested; per-point scratch no longer shows up. On the gRPC path protobuf's boxed list builders are the next
  largest cost.
- Greedy seeding spends a third of its time on candidate scoring and pruning (`boundedPotentials`, `absorb`).
  This is the price of greedy over plain D² (`seed` vs `seedPlain` in `KMeansBenchmark`: 353 ms against 80 ms at
  n = 100k, k = 50, d = 16).

## Optimizations

The efficiency work replaced list-of-vectors storage with CSR point blocks, removed copies and extra passes, and
added greedy k-means++ and bounded Lloyd. Outputs are bit-identical for a given seed except where seeding changed
(sparse seeding differs in the last bits; `WeightedLloyd.cluster`, `bestOf` and `CandidateSolutions` changed with
greedy seeding).

**All before/after numbers in this table were measured on a loaded host:** before and after jars ran alternately
(A, B, A, B) in one session on a machine shared with other builds, with the one-minute load average between about 4
and 51 depending on the session. Differences below about 15% are within noise; allocation per operation is exact.
The final quiet-host numbers for the same benchmarks are in the JMH tables above.

| area | benchmark | before | after |
|---|---|---|---|
| union | union of two 1000-point coresets | 5.57 us, 32 KB | 0.019 us, 144 B |
| union | 8-way fold | 97.8 us | 0.21 us |
| tree | snapshot of 8 levels | 21.5 us | 0.16 us |
| decode | protobuf / Spark coreset | 65 / 94 us | 56 / 79 us |
| seeding | sparse k-means++ (d = 10⁶) | 82.9 ms | 27.0 ms |
| seeding | greedy k-means++, k = 50, d = 16 | 620 ms (no pruning) | 360 ms (triangle pruning) |
| Lloyd | 20 iterations, sparse 100k × 10⁶ | 3169 ms, 256 MB | 364 ms, 48 MB |
| Lloyd | 20 iterations, dense 200k × 10, k = 50 | 1815 ms | 519 ms |
| algorithm | mean coreset, n = 10⁵, m = 1000 | 23.0 s (greedy partitions) / 6.0 s (before greedy) | 6.1 s |
| algorithm | uniform, unit weights, n = 10⁶ | 5.48 ms, 32 B/point | 1.03 ms, 0.23 B/point |
| algorithm | sensitivity, n = 10⁶ | 222 ms, 60 B/point | 217 ms, 20 B/point |
| algorithm | sparse sensitivity compress | 81.4 ms | 25.2 ms |
| metrics | `record` at 12 threads | 125 ns | 19 ns |
| metrics | default span | 1391 ns, 240 B | 39 ns, 0 B |

What changed:

- **Storage.** `Coreset` is a `PointSet` of CSR `PointBlock` segments (indices, values, row offsets, weights, cached
  row norms). Decoders build blocks directly and `SparseVector.ofOwned` adopts arrays instead of cloning them.
  Union, tree snapshot and slicing share segments, so no entry is copied. `Coreset.of(List)` still copies once.
- **Seeding.** On sparse input each new center is scattered into a dense scratch array once, with the exact
  two-part distance rather than the norm expansion. k-means++ is greedy by default (2 + ⌊ln k⌋ candidates per step,
  scored in one fused pass, pruned with the triangle inequality and a rounding margin that keeps results
  bit-identical to the unpruned version). Sensitivity sampling and the mean coreset's partitions keep plain D²:
  the sensitivity bound is proven for plain D², and greedy matters only for final solves.
- **Lloyd.** Hamerly bounds in both layouts, plus center-to-center bounds on dense data. Sparse centers are sorted
  arrays scattered into one scratch array, with no allocation per iteration. A test checks that bounded and full
  scans give identical assignments, centers and cost.
- **Sampling.** Sensitivity scratch went from 60 to 20 B per point; uniform with equal weights allocates nothing per
  point and makes one pass; the alias sampler builds with only its two tables; bicriteria does one bookkeeping pass
  per round. Guard tests pin the number of passes (through metrics counters) and the allocation per point.
- **Metrics.** Histogram buckets are striped per thread. Span CPU and allocation probes are opt-in
  (`new RecordingMetrics(true)`), which is why a default span costs tens of nanoseconds instead of 1.4 us.

Trade-offs that remain on purpose: Lloyd keeps 16 B per point of bounds; sensitivity keeps an 8 B per point
probability table, because q is needed for the drawn rows; the dense Lloyd update sums all members each iteration,
since summing only moved points would change the floating-point order.

Quality effect of greedy seeding: on a real 8-node run, solutions on the gathered coreset went from up to 1.61× the
full-data optimum to at most 1.012× ([distributed-run.md](distributed-run.md)).

## Scaling summary

| axis | observation |
|---|---|
| n (compress) | 10x more points: sensitivity 9.8-9.9x, mean 10-11x time; uniform 2.9-6.2x (the O(m) part dominates at n = 10⁴) |
| k (Lloyd iteration, n = 100k, d = 16) | k 10 to 50: 2.9x time |
| d (Lloyd iteration, n = 100k, k = 10) | d 16 to 128: 5.5x time |
| k (greedy seeding, n = 100k, d = 16) | k 10 to 50: 5.0x time (plain D²: 4.6x) |
| workers (in process, single-writer) | 1, 2, 4, 8 workers: 3.08, 6.26, 12.7, 19.9 M points/s |
| workers (in process, CAS) | 3.12, 6.29, 12.3, 19.8 M points/s |
| nodes (gRPC loopback, same JVM) | 1, 2, 3, 4 nodes: 3.11, 5.22, 5.64, 5.37 M points/s |
| threads (metrics record) | 1, 4, 12 threads: 12.5, 12.6, 22.6 ns/op |
