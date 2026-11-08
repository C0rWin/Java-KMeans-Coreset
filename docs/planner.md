# Resilience planner (`coreset-planner`)

The planner chooses how to build and ship a distributed k-means coreset when packets can be lost and nodes can fail.
Given a workload, an environment and an accuracy requirement, it returns the cheapest configuration whose
predicted error meets the target, says which constraint binds, and can run the plan in-process with injected
faults to compare predicted and measured error.

The formulas come from the paper sections in `docs/research/paper/sections` (branch `research/theory`), cited
below by label. The certificate follows `b-planner.tex` (sec:P-formulas, thm:P-sound, prop:P-search) and the
contracts of `calculus.tex`. They sit behind the `ResilienceModel` interface, so they can be replaced without
touching the search.

## Inputs

### Workload profile (`WorkloadEstimator`)

The estimator makes one pass over the input (a binary point file or an in-memory `PointSet`). The pass keeps a
uniform reservoir of `--pilot` rows (default 20 000, Vitter's algorithm R). Everything else is computed on the pilot:

| field | meaning |
|---|---|
| `n`, `d`, `density` | input size, dimension, stored entries / (n d) |
| `kmeansppCost`, `balancedness` | cost of one k-means++ seeding scaled to n; smallest / largest cluster size |
| `candidates` | the finite query family Q: 4 k-means++ seedings, 2 Lloyd solutions, one k/2 seeding |
| `kappaPt(Q)` | full-scale `sum_p cost(p,Q)^2 / cost(P,Q)^2`, the concentration in the kappa_pt law |
| `costShareMax(Q)` | full-scale `max_p cost(p,Q)/cost(P,Q)`, the Bernstein range of node loss |
| `families[f].sTotal` | `max_p s_p / q_p` with s the standard sensitivity bound (for the sensitivity family `6 alpha + 4k`) |
| `families[f].sigma2(Q)` | `sum_p cost(p,Q)^2/q_p / cost(P,Q)^2 - 1`, m times the relative variance of one draw |
| `mean` | mean-coreset bias and cell concentration on a 10k subsample for cell counts k, 2k, ..., 2048 |

Full-scale extrapolation uses `r = n / pilot`. A pilot point stands for r input points, so `kappa_full =
kappa_pilot / r`, per-point maxima shrink by r, and `sigma2` is scale-free.

Cost of the estimate:
- the pass: O(n) row reads;
- memory: O(pilot * nnz);
- the pilot work: O(pilot * k * nnz * 50);
- the mean grid: O(10^4 * sum(cells) * nnz).

On covertype (581 012 x 54) the whole profile takes about 3 s.

### Environment (`EnvironmentProfile`)

- Erasures: `rho`, plus the mean burst length (a Gilbert channel with loss probability 1 in the bad state).
- Nodes: count, and probabilities `failPre` / `failPost` of crashing before or after the local summary is replicated.
- Partition: random or adversarial.
- Stragglers: count and delay.
- Network: RTT, bandwidth, largest datagram.
- Whether the input is already stored with redundancy.

### Requirement (`Requirement`)

- `eps` and `delta`, and `k`.
- Guarantee: `INSTANCE` (the finite candidate family) or `UNIFORM` (every k-center query).
- Objective: minimum bytes, minimum time, or a weighted sum.
- Mode:
  - **CERTIFIED**: the finite-family certificate must meet eps. It uses no empirical constant (thm:K-fixedq(c)). The
    candidate family is built from the pilot, which is independent of the pipeline's randomness, as thm:P-sound
    requires.
  - **CALIBRATED**: the point prediction must meet eps. It uses instance quantities and the calibration file. It is a
    prediction, not a guarantee.

Every plan lists the modes it satisfies.

## Search space

The search enumerates the following components. With the categorical components fixed, it bisects for the least
feasible m (prop:P-search: the predicate is monotone in m).

- Family: uniform, sensitivity, lightweight or mean (`--families` restricts the set).
- Packetization: per-draw, merged, split (per-record count budget 4), or for the mean coreset cells / cells with
  clustering-feature triples.
- Transport:
  - reliable (TCP/gRPC);
  - one-shot datagrams, with or without oversampling;
  - systematic block FEC (16,2), (16,4), (8,4), plus oversampling;
  - RaptorQ-style fountain coding (RFC 6330), with repair symbols sized for the channel quantile;
  - LTP-style early close (prop:a-stream). Early close and RaptorQ are modelled only; no transport implements them
    yet.
- Redundancy:
  - none;
  - post-compute coreset replication, r = 1, 2;
  - fractional repetition, r = 1, 2;
  - independent-seed replicas, mu = 2, 4, with max(1, stragglers) spare members.
- Topology: flat, or a merge-and-reduce tree of arity 2, 4 or 8, each with or without checkpointing internal nodes.
- Straggler policy: wait, or drop (a dropped straggler is reweighted like a lost node).

## The model (`PaperResilienceModel`)

### Budget split

| share of delta | used for |
|---|---|
| delta/2 | the finite-family certificate (delta_F) |
| delta/4 | the channel (eta), split over senders |
| delta/4 | node failures |

### Calibrated prediction, per candidate Q

The prediction is `z_{1-delta_F/(2|Q|)} * sqrt(V)`, where V adds the terms that apply:

| source | term | result |
|---|---|---|
| per-draw records, realized-count decoder | `sigma2 / E[M']` | thm:erasure-invariance |
| merged or split records, HT decoder | `((sigma2+1)/(1-rho) - 1)/m + rho/(1-rho)(B-1)/m` | thm:merged-floor (a) |
| merging floor (merged only) | `rho/(1-rho)(1-1/m) kappa_pt` | thm:merged-floor (a) |
| split records | the exact Poisson excess of the per-record budget (replaces the merging floor) | — |
| node loss, random partition, survivor reweighting | `phi/(1-phi) kappa_pt` | thm:random-partition (i) |
| merge-and-reduce levels | `0.022 S/m sum_l kappa_l` | thm:martingale, F5/E1 coefficient |
| mean coreset | pilot bias plus `rho/(1-rho) kappa_cells` | prop:cf-triples |

The `(B-1)/m` term in the HT row is the delivered-count fluctuation of B records per packet. The realized-count
decoder removes it.

In the node-loss row, phi is the node-budget quantile of the lost fraction:
- f^(r+1)-type message loss for post-compute replication (prop:post-replication);
- loss of a whole group for fractional repetition and replicas (thm:fr-optimal, thm:rateless);
- subtree loss in trees, using the Cantelli quantile of prop:topology (a, b).

### Certified bound

The bound is `eps_F = sqrt(8 V iota) + (4/3) B iota`, with `iota = ln(2|Q|/delta_F)`. V and B are built as follows:

- `V = S kappa_nd / M_eta + phi/(1-phi) kappa_pt + S/m_- sum_l kappa_l`.
- `B = max(S/M_eta, S/m_-, b_phi sbar)`.
- S is the family's sensitivity total.
- M_eta is the eta-quantile of delivered draws: cor:loss-eps (b) to (d), and cor:B-ge-batch with
  `rho~ = min(rho, 1/2)`.
- kappa_nd is the node concentration: `1/U + kappa_pt` under a random partition, 1 otherwise.

HT decoders use the Chebyshev form `sqrt(V |Q| / delta_F)`; per-draw records use the smaller of the two forms.
kappa_pt and sbar are pilot estimates, inflated by
`1 + 3/sqrt(pilot)`.

The mean coreset's bias is measured on the pilot, so mean-coreset plans are never certified.

### Hard exclusions

- Mean cells without clustering-feature triples on lossy datagrams, when the fragility lower bound
  `1-(1-rho)^k` (one record per packet) or `rho` (several) exceeds eta. This is thm:deterministic-fragile.
- An adversarial partition where `P[some share lost] > delta/4`. This is prop:adversarial-impossible.
- Early close with anything but fresh per-draw records.
- Mean coresets in trees, which would need re-clustering.
- Records larger than a datagram.

### Uniform guarantee

The uniform guarantee uses ass:sampling and thm:random-partition (ii). The constant c is unknown, so the bound is
reported in units of sqrt(c) (`samplingLemmaConstant` = 1).

### Costs (`NetworkCostModel`, a calibrated cost model, not a guarantee)

- Datagrams: per-packet IPv4+UDP plus a 16-byte application header. FEC adds parity packets.
- RaptorQ sends the least packet count whose channel quantile reaches K + 2.
- Early close sends `K/(1-rho)` packets, plus one bandwidth-delay product, plus one RTT.
- Reliable streams:
  - per-segment TCP headers and gRPC prefixes;
  - retransmission at rate rho;
  - the Cardwell, Savage and Anderson (INFOCOM 2000) slow-start model with IW10 (RFC 6928);
  - then the Mathis et al. (CCR 1997) square-root throughput, with one congestion event per burst;
  - timeouts are omitted.
- Coordinator ingress is the bottleneck of the flat topology.
- Data placement for fractional repetition and replicas is charged unless `--pre-replicated` is set.

## Calibration file

The bundled `calibration-v1.json` (version `fault-lab-2026-11-07-v1`) records its provenance:
`docs/research/fault-tolerance-experiments.md` on `research/fault-injection` at 82fe39d, run F1-F6, seed 42.

| constant | value | source |
|---|---|---|
| `mergeReduceCoefficient` | 0.022 | F5/E1 |
| `kappaLawPoint` | 1.0 | F1/A1, measured 0.99-1.02 |
| `kappaLawLow` / `kappaLawHigh` | 0.95 / 1.03 | F1/A1, reported as the calibrated band |
| `erasureSlope` | 0.563 | F3/B1; recorded only, the model uses the theoretical 0.5 |
| `samplingLemmaConstant` | 1.0 | normalisation of the uniform mode |

Pass `--calibration file.json` to use another version. The certificate reads none of these constants.

## CLI

```
java -jar coreset-planner/target/coreset-planner.jar plan \
  --dataset covertype --rho 0.1 --bursty 4 --nodes 16 --fail 0.1 --fail-phase both \
  --eps 0.05 --delta 0.05 --objective bytes --mode calibrated [--families sensitivity,uniform] [--json plan.json]
java -jar coreset-planner/target/coreset-planner.jar validate <same options> --trials 30
```

- `--dataset` takes a public id from the testbed cache (`--variant zscore` by default) or a `.cbin`/`.csr` file.
- `plan` prints:
  - the plan;
  - the binding constraint and the exclusion summary;
  - the cheapest alternative per family, packetization and transport;
  - a JSON document with the workload, environment, requirement and plan.
- `validate` runs the plan in-process (`PlanSimulator`). It reports the measured (1 - delta)-quantile of the worst
  candidate error next to the prediction. Each trial injects:
  - a random (or cluster-sorted, for adversarial) partition;
  - per-link Gilbert erasures over randomly ordered packets, with block-FEC or RaptorQ recovery;
  - independent pre- and post-crashes with replica holders;
  - stragglers;
  - internal tree-node crashes.

  Decoders:
  - realized-count for per-draw records;
  - HT with the realized packet-delivery rate for merged and split records;
  - realized-mass rescaling for clustering-feature cells;
  - survivor reweighting by the reliable per-share weights.

## Validation on covertype

Covertype z-score, k = 10, delta = 0.05, eps = 0.05, 16 nodes, RTT 0.5 ms, 1 Gbit/s, 1400-byte datagrams.
Each row has 30 trials (20 for the all-family rows). "Measured" is the empirical 95% quantile of the worst error
over the 7 pilot candidates, evaluated on the full data.

See the handoff in `.claude/notes/Planner.md` for the exact table produced by the last run, and
`PredictedVsMeasuredTest` for the synthetic check in CI.

The test checks that the measured quantile is at most the certified bound and within [0.4, 1.3] times the point
prediction:
- The lower slack is there because a Bonferroni Gaussian quantile over 7 correlated candidates overstates the
  quantile of their maximum.
- The upper slack covers the kappa-law calibration spread (0.95-1.03), the sampling error of a 95% quantile from 40
  trials (about 20%), and the pilot estimate of sigma2.

## Limitations

- Certified plans are expensive. The sensitivity total S = 6 alpha + 4k with alpha = 16 (log2 k + 2) makes the
  finite-family certificate on covertype at eps = 0.05 need m of about 8 x 10^6, more draws than points. Calibrated
  predictions are 50-100x tighter.
- Merged and split records under bursty channels, and the uniform composition with instance kappas, are
  uncertified in the theory (open gaps in `calculus.tex`). Here they use the memoryless formulas.
- HT decoders need the exact erasure rate for the certificate. The simulator's decoder uses the realized rate.
- The mean coreset's bias comes from a 10k-point pilot subsample at per-share cell counts. It is a calibrated
  input, conservative on covertype (measured/predicted about 0.7).
- The TCP model omits RTO timeouts and modern loss recovery (RACK-TLP, QUIC). RaptorQ decode failure (about 1e-6 at
  K + 2 symbols) is folded into eta and not modelled.
- The search space is fixed by the constants in `Planner`. Sensitivity-aware replication (thm:sens-replication) is
  not enumerated, because its gate is not monotone in m.
