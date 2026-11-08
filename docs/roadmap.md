# Roadmap

Where the project stands, what was learned along the way, and what is worth doing next. Measurements cited here are in [performance.md](performance.md), [benchmarks/](benchmarks/README.md) and [distributed-run.md](distributed-run.md).

## Where it stands

| Module | What it is |
|---|---|
| `coreset-core` | Dependency-free library. CSR point storage with zero-copy union/slice; uniform, sensitivity and Barger–Feldman mean coresets; greedy k-means++ and Hamerly-bounded Lloyd; merge-and-reduce tree with deterministic randomness; validation tools (distortion, integrity, confidence intervals, error budget); lock-free metrics. |
| `coreset-node` | Transport-agnostic node: sessions, single-writer lock-free ingestion with backpressure, seq/producer idempotency, CAS tree mode, limits. |
| `coreset-grpc` | gRPC service and coordinator, TLS/mTLS/token, Prometheus metrics, CLIs. |
| `coreset-spark` | Distributed builds over RDD/Dataset, checkpointed Structured Streaming sink, exact results under task retries. |
| `coreset-flink` | Streaming and windowed coresets with exact recovery from checkpoints. |
| `coreset-testbed` | Datasets, offline/stream/distributed runs, per-run validity reports, ESA'22-style aggregation. |
| `coreset-bench` | JMH kernels, node and gRPC ingestion harnesses, JFR profiling. |

## What we learned (and what it changed)

- **The additive constant matters.** The deterministic coreset from the paper needs φ = Σ cost(P_i, μ_i). Without it, costs are underestimated without bound. With it, `cost(C,Q) ≥ cost(P,Q)` for every Q, which became a one-sided test oracle.
- **The final solver decided the quality, not the coreset.** On a real run with up to 8 node processes, sampling coresets estimated cost to within 0.5–3.4%, yet solutions were up to 1.61× the optimum. The cause was plain D² seeding, which leaves small clusters uncovered. Greedy k-means++ brought every configuration to ≤1.012×.
- **Theory-valid α is a poor default.** The provable sensitivity bound (α = 16(log₂ k + 2)) dilutes the per-cluster mass term. In a study over 18 settings, merge-and-reduce distortion averaged 1.81 against 1.24 with α = 2, while offline distortion was about equal (1.18 against 1.20). The provable setting stays available as `alpha=kmeans++` for when a guarantee is required.
- **Uniform sampling is a strong baseline** at equal size, except on skewed cluster sizes. The testbed should report a balancedness diagnostic so users can tell when uniform is safe.
- **Merge order matters for reproducibility.** Union-then-compress is not associative, so engine adapters key merges by partition or subtask and derive randomness from (seed, partition, level, sequence), never from the attempt.

## Next steps

### Algorithms
1. **Fast-coresets** (Draganov, Saulpic, Schwiegelshohn, SIGMOD'24): Õ(nd) construction: a JL projection, quadtree-based fast k-means++ for the assignment, then sensitivity sampling. This is the most promising speed upgrade for the build phase.
2. **Sampler ladder** behind the same SPI: lightweight → welterweight(j) → sensitivity → fast-coreset. Group sampling goes in for testbed completeness, not as a default.
3. **Dimension reduction** for the rough solution only (Makarychev–Makarychev–Razenshteyn, STOC'19: O(log(k/ε)/ε²) dimensions), keeping the original sparse points in the coreset.
4. **Balcan et al. two-round distributed protocol.** It is one-shot, with no tree and therefore no compounding error. It is a natural Spark batch mode.
5. **k-median** (z = 1) solvers. The testbed already supports the cost.
6. **Fair/constrained coresets.** Points carry colors, with per-color coresets then a union.
7. **Signed weights** for fully dynamic coresets: deletions, GDPR erasure, changelog streams.

### Engineering
1. **Dense vector encoding** on the wire. Dense points currently pay ~26% for indices (202 B/point against 160 B of raw doubles at d=20).
2. **Apache Beam `CombineFn`** adapter: one implementation for the Flink, Spark and Dataflow runners.
3. **Flink rescaling** via key groups, since parallelism is currently fixed per checkpoint. A distributed merge stage beyond one subtask.
4. **Kafka Streams** processor. A Storm/Trident adapter only if there is demand: the project is alive (3.x) but niche, and Flink covers the use case.
5. **Legacy Thrift modules: removed.** Before removal, on 100k points the gRPC node was 9.7× faster end to end than the Thrift server (sensitivity), sent 26% fewer bytes, and matched its distortion (1.11 against 1.13); see [Thrift parity](performance.md#thrift-parity-before-removal).
6. **CI.** There is none yet. Add GitHub Actions running build + tests, a nightly testbed smoke, and JMH regression gates on the kernels in `docs/performance.md`.
7. Publish to Maven Central. Keep Spark (4.0 today) and Flink (2.3) on current releases.

### Validation
1. Propagate confidence-interval variance through merge-and-reduce levels. Intervals currently cover one sampling step.
2. An adversarial distortion search (local moves on centers maximizing |cost_C/cost_P − 1|) for offline certification.
3. Run the testbed on public datasets (Census1990, Covertype, KDDCup99) and replicate ESA'22 tables.
4. Publish validity reports alongside every release benchmark.

## Open-sourcing

- **Positioning:** provable, mergeable, measurable coresets for the JVM. No maintained JVM library combines modern coreset algorithms, engine adapters and a distortion testbed. The nearest are a Spark-only k-means package, GPL MOA StreamKM++, and inactive C++/Python projects.
- **The flagship is the testbed:** a reproducible validity report per run, so users can trust a coreset before deploying it.
- **Decisions for the owner:**
  - License: MIT today; Apache-2.0 is common for this ecosystem because of its patent grant.
  - Artifact naming and groupId (`io.github.c0rwin.coreset` today).
  - Whether to keep the Barger–Feldman construction as the headline algorithm, or present it as one algorithm in a family.
