# Coreset benchmarks

This directory holds benchmark results produced by `coreset-testbed`, one
dated directory per run. Each run directory contains:

- `summary.md`: the aggregated tables, one per dataset.
- `summary.csv`: the same data with every column (CI coverage, weight error,
  distance evaluations, serialized bytes, load average, ...).
- `reports.jsonl.gz`: every per-run validity report, one JSON object per line
  (`coreset-testbed aggregate --in <file>` rebuilds the summaries from it).
- `shared.json`: what the shared full-data candidates cost per dataset and k.
- `config.json`, `environment.json`, `command.txt`: what ran, on what, and how.

## What is measured

A coreset C of a weighted point set P is good when, for every set of k centers
Q, `cost(C, Q)` is close to `cost(P, Q)`. The *distortion* of C on Q is
`max(r, 1/r)` with `r = cost(C, Q) / cost(P, Q)`. The true distortion is a
supremum over all Q, and deciding it is co-NP hard (Schwiegelshohn and
Sheikh-Omar, ESA 2022), so every number here is a **lower bound** measured on
a finite set of candidate solutions:

- `kmeans++@P#i`, `convex@B#i`: the ESA'22 candidates on the full data, five
  k-means++ seedings and five random convex combinations of bicriteria
  centers.
- `bestOf@P`: the best of 3 k-means++ and Lloyd runs on the full data. It
  doubles as the reference solution for the downstream comparison.
- `lloyd@C#i`, `bestOf@C`: solutions computed on the coreset itself, which are
  the ones a user would deploy. They are evaluated on P as well.

The full-data candidates are generated once per (dataset, k) and shared by all
runs. Algorithms are therefore compared on identical queries, and the
expensive full-data passes are paid once.

Each report also carries:

- **Integrity**: relative error of total weight, first moment and second
  moment against the input. The mean coreset preserves all three exactly, and
  the sampling coresets preserve them in expectation. For the mean coreset,
  `oneSidedViolations` counts candidates whose cost it underestimates. That
  count must be 0 (Barger and Feldman, Lemma 1).
- **Confidence intervals**: for offline sensitivity samples, a 95% CLT interval
  of `cost(P, Q)` computed from the sample alone, for each full-data candidate.
  `ci_coverage` is the fraction of intervals that contain the true cost.
  Candidates optimized on the coreset get no interval, because they are not
  independent of the sample.
- **ESA'22 δ-sweep** (benchmark instance only): the paper's lower bound. For
  each deficiency threshold δ it takes the clustering with the most deficient
  clusters, puts centers only at non-deficient cluster means, and compares the
  exact full cost with the coreset cost.
- **Stream levels**: in stream mode the input arrives in b = 64 blocks. Each
  block is compressed and added to a merge-and-reduce tree. Right after
  `2^l` blocks the tree holds a single summary that has been reduced
  `l + 1` times. Its distortion against exactly those blocks is reported per
  level (`level 0` and `top level` in the tables), so degradation under
  composition is measured, not assumed.
- **Downstream**: `rel. cost = cost(P, bestOf@C) / cost(P, bestOf@P) - 1`,
  where bestOf@C is the best of 10 restarts on the coreset. This is
  optimization quality, not coreset quality. ESA'22 explains why the two
  differ.
- **Construction cost**: wall and CPU time of the constructing thread,
  allocated bytes (`ThreadMXBean`), heap peak above the pre-run baseline (pool
  peaks plus a 1 ms heap sampler, after a GC), GC count and time,
  distance evaluations counted by the algorithms, output size and protobuf
  size.
- **Validation cost**: wall time and distance evaluations spent evaluating the
  run, plus how many full-data passes that took (normally 1). The shared
  candidates are accounted separately in `shared.json`.

Aggregation follows ESA'22. Per (dataset, mode, algorithm, k, m) group the
headline `distortion` is the **mean over the 5 seeds of each run's maximum
distortion**. `sd` and `max` (the worst seed) are reported next to it.
The `pass` column is the share of runs whose distortion stayed at or below 1.2
with clean integrity. The 1.2 threshold is a configuration input, not a
theoretical constant. Distortions above 5 and above 10 are flagged as failure
and catastrophic failure, following Draganov et al. (SIGMOD'24).

## Reproducing

```
mvn -B -pl coreset-testbed -am package -DskipTests
java -Xms16g -Xmx16g -XX:+UseG1GC -XX:+AlwaysPreTouch \
  -jar coreset-testbed/target/coreset-testbed.jar suite standard --out results/
```

All randomness derives from the suite seed (42), so the same commit, flags and
suite produce identical coresets, costs and distortions. Only timings, memory
and environment fields differ. The test suite checks this
(`ReproducibilityTest`). `suite smoke` runs every dataset kind, algorithm and
mode, including distributed over in-process nodes and over gRPC, in about a
minute. To run your own grid:

```
java -jar coreset-testbed/target/coreset-testbed.jar run --config my.yaml --out results/
```

The built-in suites in `coreset-testbed/src/main/resources/suites/` document
the config format. Public datasets (`datasets list`) are never downloaded
implicitly. `datasets fetch <name>` or `--allow-download` stores them under
`~/.cache/coreset-testbed`, and the file's SHA-256 is checked against the
descriptor.

## Caveats that apply to every run

- Timings come from one JVM run per suite with no warmup discarded. Barrett et
  al. (OOPSLA'17) show steady state is not guaranteed, so all iterations are
  reported. For per-kernel performance use JMH.
- Wall time is only meaningful on an idle machine. Each report records the
  one-minute load average. When it exceeds the core count, compare CPU times,
  allocation and distance evaluations instead.
- The first-moment relative error is ill-conditioned when the data mean is
  near the origin. The ESA'22 instance is centered, so its value there is
  meaningless.
