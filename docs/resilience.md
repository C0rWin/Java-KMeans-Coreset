# Loss and failure resilience

The distributed path (coreset-node + coreset-grpc) implements the mechanisms whose guarantees the theory paper
proves for lost packets, crashed nodes and stragglers. This page maps each mechanism to its statement, says what is
guaranteed under which assumptions, how to configure it, and what it costs. Theorem labels refer to
`docs/research/paper/sections/{model,erasures,coding,nodefailures}.tex` on the `research/theory` branch.

Everything below is opt-in. The default session (`MessageSpec.SUMMARY`) and the classic `CoresetCoordinator.gather`
(unary snapshots, any node failure fails the call) are unchanged and stay available as the baseline.

## Overview

| mechanism | statement | where | how to turn it on |
|---|---|---|---|
| snapshot as a packet stream with a reliable header | model.tex, message model | `SnapshotStream` RPC, `MessageCodec` | any `gatherResilient` |
| per-draw packetization, seeded random packing | thm:erasure-invariance, cor:loss-eps | `Packetization.PER_DRAW`, node `MessageBuilder` | `MessageSpec.perDraw(m, k, B)` |
| merged packetization (the foil) | prop:merged-floor-lb, thm:merged-floor | `Packetization.MERGED` | `MessageSpec.merged(m, k, B)` |
| realized-count decoder | thm:erasure-invariance (c) | `Decoder.REALIZED_COUNT`, `MessageDecoder` | `ResilienceOptions.withDecoder` |
| Horvitz-Thompson decoder (known loss) | thm:poisson-erasure (a), lem:a-thinning | `Decoder.HORVITZ_THOMPSON` | `withDecoder(...).withKnownLossRate(rho)` |
| per-centroid (CF) triples | prop:cf-triples (b) | `Packetization.MEAN_CF_TRIPLES`, `WeightedLloyd.meanCells` | `MessageSpec.cfTriples(cells, B, lloyd)` |
| mean coreset with a global offset (the foil) | prop:cf-triples (c) | `Packetization.MEAN_GLOBAL_OFFSET` | `MessageSpec.globalOffset(...)` |
| rescale by received mass | prop:cf-triples remark | `Decoder.REALIZED_WEIGHT` | `withDecoder(REALIZED_WEIGHT)` |
| survivor reweighting from headers | prop:ht-unbiased, thm:random-partition, cor:nf-local | `SurvivorReweighting`, `Census` | `withReweighting(REALIZED_WEIGHT, census)` |
| post-compute coreset replication | prop:post-replication | `Replicate` / `StoreReplica` RPCs | `coordinator.replicate(...)`, `withReplication(r)` |
| fractional repetition groups | thm:fr-optimal | `GroupLayout.fractionalRepetition` | layout on create, ingest and gather |
| rateless independent-seed replicas | thm:rateless (a, b, d) | `GroupLayout.rateless` | layout on create, ingest and gather |
| stragglers: proceed with the first responders | def:node-failures (T), rem:nf-adaptive | `ResilienceOptions.minResponders`, `callDeadline` | `withMinResponders(q)` |
| content-oblivious lossy transport for experiments | def:oblivious, lem:a-ge | `ChannelModel`, `LossyTransport` | client interceptor |
| loss band predictions | cor:loss-eps (b, c, d) | `LossBounds` | — |

## Message model on the wire

A node's snapshot becomes a `NodeMessage`: records `(point, weight, tag)` in send order plus a header. On the wire
(`rpc SnapshotStream(SnapshotStreamRequest) returns (stream SnapshotChunk)`) the node sends one `MessageHeader`
and then packets `0 .. P-1`, packet `j` carrying records `[j*B, (j+1)*B)`. The packet layout is a function of the
record count and `B` only, never of the payload.

The header carries what the decoders need and what the model calls reliable: owner, packetization, sampling stream,
draws `m`, records, packets, `B`, the node's input weight and points, the reliable offset, the record weight as sent,
and whether the summary reflected every accepted batch. The lossy transport never drops the header; in a deployment
this is the part a transport must protect (it is a few dozen bytes).

Tags carry per-record metadata of `O(log m)` bits: the index of the distinct draw for per-draw records (so the
decoder merges copies of one draw without hashing coordinates) and the draw count for merged records. CF records
carry their cell offset.

## Packetizations

`MessageSpec(packetization, size, k, recordsPerPacket, lloydIterations)` is part of `SessionSpec`.

- **summary** (default): the summary's points as they are. The non-resilient baseline.
- **merged**: `m` sensitivity draws of the summary (k-means++ seeding with `k` centers, the standard bound), one record
  per distinct point carrying its count.
- **per-draw**: the same `m` draws, one record per draw, each of weight `w(p) / (m q(p))`, shuffled by a generator
  independent of the draws (draw-oblivious packing, def:draw-oblivious). Copies of a heavy point are separate
  records, so no record carries merged mass. The sampler always draws `m` times, also from summaries of at most
  `m` points (`SensitivityCoreset.sample`), because the guarantee is about i.i.d. draws.
- **mean-global** and **mean-cf**: a mean coreset of the summary over at most `size` cells (plain D² seeding plus
  Lloyd, as `MeanCoreset`), records shuffled. mean-global puts the summed cell offsets into the header offset;
  mean-cf sends the triple (mean, weight, cell offset) per record, computed exactly per cell by
  `WeightedLloyd.meanCells`.

Randomness: every message generator derives from the session seed and the node's partition. The seeding (hence `q`)
and the cell partition use a stream shared by all replicas of a partition; the draws and the record order use the
request's sampling `stream`. Equal (summary, stream) give the identical message; distinct streams give independent
samples of the same `q`. The erasure channel's seed is separate from all of these.

## Decoders

`MessageDecoder.decode(sources, decoder, knownLoss, multiplier, maxDim)` turns what arrived into a weighted set in
one exactly sized block (two passes over the delivered packets, no per-point objects):

- **as-sent**: weights unchanged (baseline; biased low by the erased mass).
- **ht**: weights times `1 / (1 - rho)` for a known loss probability.
- **realized**: weights times (draws sent) / (draws received), counted from the header and the delivered records
  (draw counts for merged messages). Pooled over the members of a rateless group: each draw gets
  `w(p) / (M q(p))` with `M` the delivered draws of the whole group.
- **realized-weight**: weights and per-cell offsets times (record weight sent) / (record weight received).

The reliable header offset is added once per message, unscaled by the loss decoder. `multiplier` is the survivor
scale.

## Guarantees

**Per-draw messages under any content-oblivious channel** (thm:erasure-invariance). If the coreset is an `m`-draw
sensitivity sample, records are per draw in a seeded random order, the channel's delivery indicators are independent
of the payloads and of the sender's seed, and `m` arrives reliably, then conditionally on `M' = m'` delivered draws
the realized-count output has exactly the law of a loss-free `m'`-draw sample. The channel only changes the law of
`M'`. Hence (cor:loss-eps) with probability `1 - delta - eta` the output is as accurate as a loss-free sample of
`m_eta` draws, `eps' = eps * sqrt(m / m_eta)`:

- memoryless, `B` records per packet: `m_eta = B * ceil(mu_B - sqrt(2 mu_B ln(1/eta)))`, `mu_B = (1 - rho) m / B`;
- Gilbert channel, one record per packet: `m_eta = ceil((1 - rho) m - sqrt(V (1/eta - 1)))`,
  `V = m rho (1 - rho) (1 + |lambda|) / (1 - |lambda|)`.

`LossBounds` computes both; the coordinator CLI prints the band. The sample here is a coreset of the node's summary,
so by transitivity the decoded message is an `(eps0 + eps' + eps0 eps')`-coreset of the node's data when the
summary is an `eps0`-coreset.

**Merged messages have a floor** (prop:merged-floor-lb). On `P = {0, 1}` a merged record of `1` is erased with
probability `rho` whatever `m`, and no decoder recovers it; per-draw loses `1` only if all of its draws are erased,
`((1 + rho) / 2)^m`.

**CF triples are consistent** (prop:cf-triples). With per-cell triples the decoded output of any received set `R`
of cells is exactly the mean coreset of the summary points in `R`; for every single center its cost equals the cost
of those points. A global offset instead keeps the offsets of the lost cells. Exactness is relative to the node's
summary: with an offset-free session algorithm (`sensitivity`, `uniform`) the summary has no offset of its own; an
offset carried by the summary (e.g. a `mean` session) is reported in the header and added unscaled.

**Survivor reweighting** (prop:ht-unbiased). The coordinator learns the failed set from fetch failures and
deadlines. With a census of node weights taken before failures (`coordinator.census`), `REALIZED_WEIGHT` scales the
survivors by (total weight) / (weight in the surviving headers). For unit weights this is the count-realized
estimator, conditionally unbiased on every failure pattern for random (or equal-size random) partitions; the
node-realized estimator (`NODE_FRACTION`) is unbiased given `F`. A failed group is one whose members delivered no
record (a header alone carries no mass). Composability alone (`NONE`) summarizes the survivors only.

**Replication** (prop:post-replication). `coordinator.replicate(session, r, peerTargets, layout, await)` freezes every
node's message and has node `i` store copies on nodes `i+1 .. i+r` (mod n), node to node. The frozen message is then
served by the owner and the holders alike, so a gather that fails over to a holder decodes the identical coreset.
A message is lost only if all `r + 1` holders fail (`phi^(r+1)` for i.i.d. failures), at a cost of `r` copies of each
message, independent of the data size. Crashes before the replication step are F_pre failures that this does not
address.

**Fractional repetition** (thm:fr-optimal). `GroupLayout.fractionalRepetition(n, s)` makes `floor(n/(s+1))` groups.
Every batch goes to every member of its group under one seq; members share the partition seed and the sampling
stream, so their summaries and messages are identical and any one member suffices against `s` stragglers. Under
packet loss the decoder takes each packet seq from whichever member delivered it.

**Rateless replicas** (thm:rateless). `GroupLayout.rateless(n, g)`: members hold the same summary (same partition,
same batches) and share `q`, but each samples with its own stream; the decoder pools every delivered draw of the
group. Any `|R|` responders give `|R| m1` i.i.d. draws, so `m1 = ceil(m(eps, delta) / mu)` per member tolerates all
but `mu` responders, and fewer responders degrade accuracy as `sqrt(mu / r)` instead of failing. With loss, the
pooled stream is again covered by thm:erasure-invariance (part (d)).

Identical summaries in a group need the members to apply their batches in the same order: one submitting thread and
no BUSY retries that reorder batches (in the tests: one worker, one batch in flight). Members whose headers disagree
are not mixed by the FR decoder.

**Stragglers**. `callDeadline` bounds each fetch; `minResponders = q` stops waiting once every group has `q`
completed messages and cancels the rest. The theory needs the identity of slow nodes to be independent of the data
(rem:nf-adaptive); data-dependent slowness is outside it.

**Lossy transport**. `LossyTransport` erases packets of `SnapshotStream` responses only, per a `ChannelModel`
(`Bernoulli`, `GilbertElliott`, `Perfect`), from a seed derived from the interceptor's seed and the request's
(owner, replica, stream) plus an attempt counter. It is content-oblivious by construction and replaces every dropped
packet by a `request(1)` so stub flow control never stalls. It is the hook the experiment harness plugs in.

## Configuration

Java:

```java
SessionSpec spec = new SessionSpec("sensitivity", Map.of("k", "10", "m", "1000"), seed, workers, 64,
        TreeMode.SINGLE_WRITER, MessageSpec.perDraw(1024, 10, 16));
LossyTransport lossy = new LossyTransport(new ChannelModel.Bernoulli(0.3), lossSeed); // experiments only
try (CoresetCoordinator c = CoresetCoordinator.connect(targets, CoordinatorOptions.defaults(),
        ClientSecurity.plaintext(), MetricsSink.NOOP, List.of(lossy))) {
    GroupLayout layout = GroupLayout.independent(targets.size());
    c.createSession(id, spec, layout);
    try (IngestStream s = c.openIngest(id, "producer", layout)) { /* submit, flush */ }
    Census census = c.census(id, layout, Duration.ofMinutes(1));      // before failures
    c.replicate(id, 1, layout, Duration.ofMinutes(1));                 // optional
    ResilientSnapshot snap = c.gatherResilient(id, ResilienceOptions.baseline(targets.size())
            .withDecoder(Decoder.REALIZED_COUNT)
            .withReweighting(SurvivorReweighting.REALIZED_WEIGHT, census)
            .withReplication(1));
}
```

Coordinator CLI (`coreset-coordinator.jar`), with `--gather resilient`:

| flag | meaning |
|---|---|
| `--packetization`, `--message-size`, `--message-k`, `--records-per-packet`, `--message-lloyd-iterations` | `MessageSpec` |
| `--layout independent\|fr\|rateless`, `--group-size` | `GroupLayout` |
| `--decoder as-sent\|ht\|realized\|realized-weight`, `--known-loss` | decoder |
| `--reweight none\|nodes\|weight` | survivor reweighting (the census is always taken) |
| `--replicate r`, `--peer-targets` | replication before gathering |
| `--loss none\|bernoulli:RHO\|gilbert:RHO,BURST\|ge:...`, `--loss-seed` | lossy transport |
| `--gather-deadline-ms`, `--min-responders` | stragglers |
| `--repeat`, `--distortion-queries`, `--eta` | repeated gathers on fresh streams, distortion against the input, band |
| `--fault-barrier FILE` | prints `FAULT_BARRIER` after ingest, census, replication and a pre-fault gather, then waits |

Node CLI: `--peer-tls-ca`, `--peer-tls-cert`, `--peer-tls-key` secure the node-to-node replica channels; the node's
auth token is presented to peers too.

## Overheads

Measured on Apple M2 Max, Corretto 17.0.5, all processes on 127.0.0.1.

**CPU and allocation** (JMH `ResilienceBenchmark`, 1 fork, 3x1 s warmup, 5x1 s measurement, `-prof gc`). Input: one
node summary of 4,000 dense points in d = 16; message size 1,000 (draws or cells), B = 16, k = 10 for the seeding;
decode on the 70% of packets a Bernoulli(0.3) channel delivered.

| form | build µs/op | build B/op | encode packets µs/op | decode µs/op | decode B/op | records | packets | message bytes |
|---|---|---|---|---|---|---|---|---|
| summary | 0.095 ± 0.025 | 288 | 655 ± 73 | 294 ± 116 | 609,064 | 4,000 | 250 | 664,656 |
| merged | 1,371 ± 256 | 309,074 | 151 ± 4 | 49.7 ± 5.7 | 138,632 | 874 | 55 | 146,211 |
| per-draw | 1,523 ± 417 | 533,316 | 162 ± 4 | 51.5 ± 6.6 | 147,360 | 1,000 | 63 | 168,137 |
| mean-cf | 253,949 ± 51,021 | 10,398,324 | 164 ± 17 | 56.3 ± 12.0 | 158,608 | 1,000 | 63 | 174,347 |

Reading the table:

- **Framing** is cheap. The summary as a packet stream is 664,656 bytes against 664,000 for the unary
  `SnapshotResponse` of the same points: 656 bytes for the header and 250 packet frames (about 2.4 bytes a packet).
- **Per-draw versus merged** costs 15% more bytes (168,137 against 146,211) for the same sample: 1,000 records for
  874 distinct points. Build time is the sensitivity sampling of the summary (k-means++ seeding plus draws), 1.4 to
  1.5 ms; the per-draw expansion and shuffle add about 0.15 ms and 224 KB.
- **Decoding** reads straight from the packets into one block: about 50 µs for 700 records, 0.15 MB.
- **mean-cf** pays for its cell partition (m-means Lloyd with 1,000 cells on 4,000 points), as the `mean` algorithm
  does: 254 ms per message. The triples add 8 bytes per record over mean-global.

**Replication** costs `r` copies of each message, node to node: 1,024 per-draw records per node and copy in the
smoke run below. **Census** is one header per node (tens of bytes).

**Gather latency** in the smoke run below (4 node JVMs and the coordinator on one machine, per-draw m = 1,024,
B = 16, 256 packets, 40 gathers with and without 30% loss, each building fresh messages): median 36.8 ms, min
12.9 ms, max 132.6 ms. Replica failover after a SIGKILL took 135 ms (one refused connection, then the holder).

Reproduce: `java -jar coreset-bench/target/benchmarks.jar ResilienceBenchmark -prof gc`.

## Smoke run with real processes

Four `coreset-node` JVMs and one `coreset-coordinator` JVM (shaded jars) on 127.0.0.1. Data: 200,000 points in
d = 10 from 10 Gaussian clusters with unbalanced sizes, in random order. Nodes run `sensitivity` (k = 10, m = 1,000,
2 workers); batches of 5,000 points go round-robin, so each node holds 50,000 points. The coordinator clusters with
k = 10; distortion is `max over Q of max(r, 1/r) - 1` with `r = cost(C, Q) / cost(P, Q)` on 10 k-means++ center sets
of the full input, and `cost_ratio` is the range of `r`. The script kills a node with `kill -9` at the coordinator's
fault barrier, after the census (and replication) and one pre-fault gather.

Common flags: `--gather resilient --algorithm sensitivity --param k=10 --param m=1000 --k 10 --seed 3 --workers 2
--batch-size 5000 --distortion-queries 10 --producer-id smoke`; per-draw runs add `--packetization per-draw
--message-size 1024 --records-per-packet 16`.

**A. Survivor reweighting after SIGKILL of node 2** (no replicas):

```
# session=A-weight points=200000 batches=40 layout=independent[[0], [1], [2], [3]] packetization=per-draw census_weight=[50000.0, 50000.0, 50000.0, 50000.0]
14:51:04 A-weight: SIGKILL node 2 (pid 50306, port 7273)
--reweight none:   # gather=0 lost_groups=1 survivor_scale=1.000000 packets=192/192 ... weight=147537.754 distortion=0.3474 cost_ratio=[0.7421,0.7577]
--reweight weight: # gather=0 lost_groups=1 survivor_scale=1.333333 packets=192/192 ... weight=196717.005 distortion=0.0106 cost_ratio=[0.9895,1.0103]
#   group=2 delivered=false weight=NaN ... sources=[SourceOutcome[node=2, servedBy=-1, completed=false, error=node 2: UNAVAILABLE io exception, ...]]
```

The census weights (50,000 per node) give the realized surviving weight 150,000 and scale 4/3. The k-means solution
on the reweighted coreset costs 6,201,138 on the full data (6,195,574 without any fault, run C below).

**B. Replica failover** (`--replicate 1`, SIGKILL of node 1 after replication):

```
# replicate node=1 records=1024 stored=1 failed=[]          (likewise nodes 0, 2, 3)
14:51:10 B-replica: SIGKILL node 1 (pid 51382, port 7272)
# gather=0 lost_groups=0 survivor_scale=1.000000 packets=256/256 ... coreset_points=3078 weight=201432.277 distortion=0.0103
#   identical_to_pre_fault=true
#   group=1 delivered=true ... sources=[SourceOutcome[node=1, servedBy=2, completed=true, error=node 1: UNAVAILABLE io exception, packetsSent=64, packetsReceived=64]]
```

**C. Per-draw packets under 30% Bernoulli loss** (20 gathers each, fresh sampling streams; `--loss bernoulli:0.3`):

| run | distortion mean | median | p90 | max |
|---|---|---|---|---|
| no loss, realized | 0.0157 | 0.0150 | 0.0227 | 0.0252 |
| 30% loss, realized | 0.0191 | 0.0185 | 0.0259 | 0.0334 |
| prediction: no-loss value x 1.4856 | 0.0233 | 0.0223 | 0.0336 | 0.0374 |
| 30% loss, as-sent (baseline) | 0.4485 | 0.4286 | 0.5499 | 0.5714 |
| 30% loss, merged + ht (known rho) | 0.0662 | 0.0586 | 0.1031 | 0.1318 |

The coordinator printed `# cor:loss-eps m=1024 rho=0.3000 B=16 eta=0.05 m_eta=464 inflation=1.485563`. Every
statistic of the lossy per-draw runs lies inside the band, and the mean inflation (1.22) is close to
`sqrt(1 / (1 - rho)) = 1.195`, the factor for the expected number of delivered draws. As-sent keeps 70% of the mass
(cost ratio 0.72 to 0.74); merged records with the HT decoder are unbiased on average but carry the floor of
thm:merged-floor, 3.5 times the per-draw error at the same loss.

**D. Mean coreset under 30% loss, one record per cell** (200 cells per node, 20 gathers):

| run | distortion mean | max | cost ratio of gather 0 |
|---|---|---|---|
| no loss, CF triples | 0.0096 | — | [0.9905, 0.9969] |
| CF triples, as-sent | 0.4517 | 0.4864 | [0.6955, 0.7125] |
| global offset, as-sent | 0.3089 | 0.3346 | [0.7685, 0.7881] |
| CF triples, realized-weight | 0.0392 | 0.0812 | [1.0126, 1.0385] |
| global offset, realized-weight | 0.0331 | 0.0626 | [1.0028, 1.0293] |

As-sent CF triples are the exact mean coreset of the delivered cells: their cost ratio against the full data sits in
a narrow band at the delivered share of the mass (0.686 in gather 0), the same for every query, which is what a
summary of 68.6% of the data must show. The global offset adds back the spread of the lost cells, so its output is
no longer a summary of any subset of the data (prop:cf-triples (c)); here it lands closer to the full cost by
accident of the offsets' size. Rescaled by the received mass both are near-unbiased estimates of the full cost, as
the remark after prop:cf-triples predicts: the triples buy consistency, not unbiasedness.

Reproduce with `docs/resilience-smoke.sh` next to the two shaded jars and a `points.csv` (any dense CSV); outputs are
deterministic given the seeds (`--seed 3`, `--loss-seed` 7 by default, 11 in run C).
