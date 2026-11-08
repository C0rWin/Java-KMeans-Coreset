package io.github.c0rwin.coreset.planner.model;

import io.github.c0rwin.coreset.planner.Configuration;
import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.planner.Configuration.Packetization;
import io.github.c0rwin.coreset.planner.Configuration.Redundancy;
import io.github.c0rwin.coreset.planner.Configuration.StragglerPolicy;
import io.github.c0rwin.coreset.planner.Configuration.Transport;
import io.github.c0rwin.coreset.planner.EnvironmentProfile;
import io.github.c0rwin.coreset.planner.EnvironmentProfile.GilbertElliott;
import io.github.c0rwin.coreset.planner.EnvironmentProfile.Partition;
import io.github.c0rwin.coreset.planner.Requirement;
import io.github.c0rwin.coreset.planner.Requirement.Guarantee;
import io.github.c0rwin.coreset.planner.WorkloadProfile;
import io.github.c0rwin.coreset.planner.WorkloadProfile.CandidateStats;
import io.github.c0rwin.coreset.planner.WorkloadProfile.FamilyStats;
import io.github.c0rwin.coreset.planner.WorkloadProfile.MeanStats;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The resilience model assembled from the results proven in the paper sections (labels refer to
 * {@code docs/research/paper/sections}; the certificate follows b-planner.tex sec:P-formulas). Budget split:
 * {@code delta/2} for the finite-family certificate (delta_F), {@code delta/4} channel (eta, split over senders),
 * {@code delta/4} node failures. Per candidate Q, two variance budgets are assembled.
 * <p><b>Calibrated</b> (point prediction {@code z_{1-delta_F/(2|Q|)} sqrt(V)}, not a guarantee): instance quantities
 * from the pilot and the calibration file.
 * <ul>
 *   <li>per draw, realized-count decoder: {@code sigma2(Q) / E[M']} (thm:erasure-invariance);</li>
 *   <li>merged/split, HT decoder: {@code ((sigma2+1)/(1-rho) - 1)/m + rho/(1-rho)(B-1)/m} plus the merging floor
 *       {@code rho/(1-rho)(1-1/m) kappa_pt} (thm:merged-floor (a)) or the exact Poisson excess of split budgets;</li>
 *   <li>node loss, random partition, survivor reweighting: {@code phi/(1-phi) kappa_pt} (thm:random-partition (i))
 *       with phi the node-budget quantile of the lost fraction after replication (prop:post-replication f^(r+1),
 *       thm:fr-optimal and thm:rateless all-of-group loss, prop:topology subtree loss);</li>
 *   <li>merge-and-reduce: {@code 0.022 S/m sum_l kappa_l} (thm:martingale (b) with the F5/E1 coefficient);</li>
 *   <li>mean coreset: pilot-measured bias plus cell thinning {@code rho/(1-rho) kappa_cells} (prop:cf-triples).</li>
 * </ul>
 * <p><b>Certified</b> (finite candidate family, thm:K-fixedq(c)): {@code eps_F = sqrt(8 V iota) + (4/3) B iota},
 * {@code iota = ln(2|Q|/delta_F)}, with {@code V = S kappa_nd / M_eta + phi/(1-phi) kappa_pt + S/m_- sum kappa_l} and
 * {@code B = max(S/M_eta, S/m_-, b_phi sbar)}, where S is the family's sensitivity total, M_eta the eta-quantile of
 * delivered draws (cor:loss-eps (b)-(d), cor:B-ge-batch with rho~ = min(rho, 1/2)) and kappa_nd the node
 * concentration. HT decoders use the Chebyshev form {@code sqrt(V |Q| / delta_F)}, per-draw records the smaller of
 * the two forms. No calibration constant enters;
 * kappa_pt and sbar are pilot estimates inflated by {@code 1 + 3/sqrt(pilot)}. The mean coreset's bias is empirical,
 * so its predictions are never certifiable. Rules that exclude a configuration outright: thm:deterministic-fragile
 * (mean cells without CF on lossy datagrams), prop:adversarial-impossible (adversarial partition without enough
 * redundancy). The UNIFORM guarantee uses ass:sampling with the calibration's constant c and thm:random-partition (ii).
 */
public final class PaperResilienceModel implements ResilienceModel {

    private final Calibration cal;
    private final NetworkCostModel net = new NetworkCostModel();

    public PaperResilienceModel(Calibration calibration) {
        this.cal = calibration;
    }

    @Override
    public String id() {
        return "paper-v1+" + cal.version();
    }

    @Override
    public Prediction predict(WorkloadProfile w, EnvironmentProfile e, Requirement req, Configuration c) {
        List<String> notes = new ArrayList<>();
        final double deltaS = req.delta() / 2.0;
        final double eta = req.delta() / 4.0;
        final double deltaN = req.delta() / 4.0;
        final int n = e.nodes();
        final boolean tree = c.arity() > 0;
        final Family fam = c.family();

        if (!fam.sampling() && tree) {
            return Prediction.excluded("mean coreset is flat only: merging cell summaries needs re-clustering");
        }
        if (!fam.sampling() && req.guarantee() == Guarantee.UNIFORM) {
            return Prediction.excluded("no a-priori uniform bound is implemented for the mean coreset");
        }
        if (!fam.sampling() && w.mean() == null) {
            return Prediction.excluded("workload profile has no mean-coreset measurements");
        }
        if (c.transport() == Transport.EARLY_CLOSE && c.packetization() != Packetization.PER_DRAW) {
            return Prediction.excluded("early close needs fresh per-draw records (prop:a-stream)");
        }

        // groups of nodes sharing one data share
        int group = switch (c.redundancy()) {
            case NONE, POST_REPLICATION -> 1;
            case FRACTIONAL_REPETITION -> c.redundancyLevel() + 1;
            case RATELESS -> c.redundancyLevel() + c.ratelessSpare();
        };
        if (group > n) {
            return Prediction.excluded("redundancy group of " + group + " exceeds " + n + " nodes");
        }
        final int units = n / group;
        if (tree && c.arity() >= units) {
            return Prediction.excluded("tree arity " + c.arity() + " >= " + units + " leaves: use flat");
        }

        int rec = net.recordBytes(w, c);
        int perPacket = net.recordsPerPacket(e, rec);
        if (c.transport().datagram() && perPacket == 0) {
            return Prediction.excluded("a " + rec + "-byte record exceeds the " + e.maxMessageBytes()
                    + "-byte datagram");
        }

        // channel: delivery unit = packet, or FEC block (all-or-nothing upper bound on the residual)
        GilbertElliott ge = e.gilbertElliott();
        double unitLoss;
        double unitLambda;
        int unitRecords;
        switch (c.transport()) {
            case RELIABLE -> {
                unitLoss = 0.0;
                unitLambda = 0.0;
                unitRecords = 1;
            }
            case DATAGRAM, DATAGRAM_OVERSAMPLE -> {
                unitLoss = e.rho();
                unitLambda = e.meanBurst() > 1.0 ? ge.lambda() : 0.0;
                unitRecords = perPacket;
            }
            case DATAGRAM_FEC -> {
                unitLoss = e.rho() == 0.0 ? 0.0 : Probability.blockFailure(c.fecData(), c.fecParity(),
                        ge.pGoodToBad(), ge.pBadToGood());
                unitLambda = e.meanBurst() > 1.0 ? Math.pow(ge.lambda(), c.fecData() + c.fecParity()) : 0.0;
                unitRecords = perPacket * c.fecData();
                notes.add("FEC residual treated as whole-block loss (upper bound on lost records)");
            }
            case DATAGRAM_RAPTORQ, EARLY_CLOSE -> {
                // RaptorQ: repair symbols sized for the channel quantile (decode failure folded into eta);
                // early close: the receiver acknowledges exactly the target count of fresh i.i.d. draws
                unitLoss = 0.0;
                unitLambda = 0.0;
                unitRecords = 1;
            }
            default -> throw new IllegalStateException();
        }
        final double rhoE = unitLoss;

        // mean-coreset fragility (thm:deterministic-fragile)
        if (!fam.sampling() && c.packetization() == Packetization.CELLS && rhoE > 0.0) {
            double fragile = perPacket <= 1 ? 1.0 - Math.pow(1.0 - rhoE, req.k()) : rhoE;
            if (fragile > eta) {
                return Prediction.excluded(String.format(
                        "mean coreset without CF over a lossy channel: failure probability >= %.3g > eta=%.3g "
                                + "(thm:deterministic-fragile)", fragile, eta));
            }
        }
        if (fam.sampling() && req.guarantee() == Guarantee.UNIFORM && rhoE > 0.0
                && NetworkCostModel.merged(c.packetization())) {
            return Prediction.excluded("UNIFORM guarantee under loss is implemented for per-draw records only "
                    + "(thm:merged-floor (b) is vacuous outside mild merging)");
        }

        // tree levels: |V_l| for the reduce levels above the leaves; the root is the reliable coordinator
        List<Integer> levels = new ArrayList<>();
        if (tree) {
            int v = units;
            while (v > 1) {
                v = (v + c.arity() - 1) / c.arity();
                levels.add(v);
            }
        }
        final int senders = units + (tree ? levels.stream().mapToInt(Integer::intValue).sum() - 1 : 0);
        final double etaPer = eta / Math.max(1, senders * (c.redundancy() == Redundancy.RATELESS ? group : 1));

        // draws per sender
        long target = tree ? c.m() : (c.m() + units - 1) / units;
        int mu = c.redundancy() == Redundancy.RATELESS ? c.redundancyLevel() : 1;
        long memberTarget = (target + mu - 1) / mu;
        long memberSend = fam.sampling() && c.oversamples()
                ? oversampled(memberTarget, unitRecords, rhoE, unitLambda, etaPer) : memberTarget;
        if (memberSend < 0) {
            return Prediction.excluded("oversampling cannot reach the target at erasure " + rhoE);
        }
        long memberEta = delivered(memberSend, unitRecords, rhoE, unitLambda, etaPer);

        // node loss
        double fT = e.failTotal();
        double pUnit = switch (c.redundancy()) {
            case NONE -> fT;
            case POST_REPLICATION -> e.failPre() + (1.0 - e.failPre()) * e.failPost() * Math.pow(fT, c.redundancyLevel());
            case FRACTIONAL_REPETITION, RATELESS -> Math.pow(fT, group);
        };
        boolean absorbed = switch (c.redundancy()) {
            case NONE, POST_REPLICATION -> false;
            case FRACTIONAL_REPETITION -> e.stragglers() <= group - 1;
            case RATELESS -> e.stragglers() <= c.ratelessSpare();
        };
        double wait = 0.0;
        int droppedUnits = 0;
        if (e.stragglers() > 0 && !absorbed) {
            if (c.stragglers() == StragglerPolicy.WAIT) {
                wait = e.stragglerDelaySeconds();
            } else if (group > 1) {
                return Prediction.excluded("stragglers exceed the group's spare members; only WAIT applies");
            } else {
                droppedUnits = e.stragglers();
            }
        } else if (c.stragglers() == StragglerPolicy.DROP) {
            return Prediction.excluded("straggler policy irrelevant here; WAIT is the canonical form");
        }
        double pInternal = tree ? (c.checkpoint() ? fT * fT : fT) : 0.0;

        double phi;
        if (e.partition() == Partition.ADVERSARIAL) {
            if (droppedUnits > 0) {
                return Prediction.excluded("adversarial partition: dropping stragglers loses data no reweighting "
                        + "can recover (prop:adversarial-impossible)");
            }
            double pNone = Math.pow(1.0 - pUnit, units);
            for (int i = 0; i < levels.size() - 1; i++) {
                pNone *= Math.pow(1.0 - pInternal, levels.get(i));
            }
            if (1.0 - pNone > deltaN) {
                return Prediction.excluded(String.format("adversarial partition: P[some share lost] = %.3g > node "
                        + "budget %.3g (prop:adversarial-impossible)", 1.0 - pNone, deltaN));
            }
            phi = 0.0;
        } else if (!tree) {
            int lost = Probability.binomialUpperQuantile(units, pUnit, deltaN);
            phi = (double) Math.min(units, lost + droppedUnits) / units;
        } else {
            // prop:topology (a),(b): lambda+ has mean sum p_l and variance sum p_l(1-p_l) beta_l; Cantelli quantile
            double mean = pUnit;
            double var = pUnit * (1.0 - pUnit) / units;
            for (int i = 0; i < levels.size() - 1; i++) {
                mean += pInternal;
                var += pInternal * (1.0 - pInternal) / levels.get(i);
            }
            phi = Math.min(1.0, mean + Math.sqrt(var * (1.0 - deltaN) / deltaN) + (double) droppedUnits / units);
            if (mean + var == 0.0) {
                phi = (double) droppedUnits / units;
            }
        }
        if (phi >= 0.999) {
            return Prediction.excluded(String.format("lost-data quantile %.3f leaves nothing to reweight", phi));
        }
        int aliveUnits = Math.max(1, (int) Math.round(units * (1.0 - phi)));

        long perUnitEta = memberEta;
        double degrade = 1.0;
        int respondersQ = 1;
        if (c.redundancy() == Redundancy.RATELESS) {
            int responders = Probability.binomialLowerQuantile(group, 1.0 - fT, deltaN / units);
            responders = Math.max(1, Math.min(group, responders));
            respondersQ = responders;
            perUnitEta = memberEta * responders;
            if (responders < mu) {
                degrade = Math.sqrt((double) mu / responders);
                notes.add("rateless group degrades by sqrt(mu/r) = " + String.format("%.3f", degrade)
                        + " at the node-budget responder quantile (thm:rateless (b))");
            }
        }
        long receivedDraws = (long) aliveUnits * perUnitEta;
        if (receivedDraws < 1) {
            return Prediction.excluded("no draws survive at the channel quantile");
        }

        final int nq = w.candidateCount();
        double z = Probability.normalQuantile(1.0 - deltaS / (2.0 * nq));
        double iota = Math.log(2.0 * nq / deltaS);
        double klPoint = cal.get("kappaLawPoint");
        double mrCoef = cal.get("mergeReduceCoefficient");
        double pilotInflation = 1.0 + 3.0 / Math.sqrt(w.pilotSize());
        boolean htDecoder = NetworkCostModel.merged(c.packetization());
        // node concentration sum_i (C_i/C)^2: its expectation 1/U + kappa_pt under a random partition, 1 otherwise
        double kappaNodeBase = e.partition() == Partition.RANDOM ? 1.0 / aliveUnits : 1.0;

        double bestPoint = 0.0;
        double bestBound = 0.0;
        String worst = null;
        Map<String, Double> worstTerms = Map.of();
        long sentDraws = (long) aliveUnits * memberSend * (c.redundancy() == Redundancy.RATELESS ? group : 1);
        for (int q = 0; q < nq; q++) {
            CandidateStats cs = w.candidates().get(q);
            double kappa = cs.kappaPt();
            double kappaCert = kappa * pilotInflation;
            Map<String, Double> terms = new LinkedHashMap<>();
            double vPoint = 0.0; // calibrated: instance quantities and calibration constants
            double vCert = 0.0;  // certified: data-independent sensitivity totals, no calibration constants
            double bCert = 0.0;
            double bias = 0.0;
            if (fam.sampling()) {
                FamilyStats fs = w.families().get(fam);
                double s2 = fs.sigma2()[q];
                double sBar = fs.sTotal();
                double kappaNode = Math.min(1.0, kappaNodeBase + kappaCert);
                double unitEta = Math.max(1.0, tree ? memberEta : perUnitEta);
                double leafExpected = (double) aliveUnits * respondersQ * memberSend * (1.0 - rhoE);
                switch (c.packetization()) {
                    case PER_DRAW -> {
                        double v = s2 / leafExpected;
                        terms.put("sampling", v);
                        vPoint += v;
                        vCert += sBar * kappaNode / unitEta;
                        bCert = Math.max(bCert, sBar / unitEta);
                    }
                    case MERGED, SPLIT -> {
                        double ms = Math.max(1.0, tree ? (double) aliveUnits * memberSend : sentDraws);
                        double v = ((s2 + 1.0) / (1.0 - rhoE) - 1.0) / ms;
                        terms.put("sampling+erasure(HT)", v);
                        // B records share a packet: in random order the HT estimate also carries the fluctuation of
                        // the delivered count, rho/(1-rho) (B-1)/m, which the per-draw realized-count decoder removes
                        double pk = rhoE / (1.0 - rhoE) * (unitRecords - 1) / ms;
                        terms.put("packet-count(HT)", pk);
                        double fl = rhoE == 0.0 ? 0.0 : rhoE / (1.0 - rhoE) * mergedExcess(w, fam, c, q,
                                memberSend * mu, units);
                        terms.put(c.packetization() == Packetization.MERGED ? "merged-floor" : "split-excess", fl);
                        vPoint += v + pk + fl;
                        double msUnit = Math.max(1.0, memberSend * mu);
                        vCert += sBar * kappaNode / ((1.0 - rhoE) * msUnit) + pk + fl * pilotInflation;
                    }
                    default -> throw new IllegalStateException();
                }
                if (tree) {
                    double sumKappa = 0.0;
                    for (int v : levels) {
                        sumKappa += Math.max(1.0 / v, kappa);
                    }
                    long internal = memberSendFor(c.m(), unitRecords, rhoE, unitLambda, etaPer, c.oversamples());
                    double mRed = Math.max(1.0, delivered(internal, unitRecords, rhoE, unitLambda, etaPer));
                    double tp = mrCoef * sBar / (internal * (1.0 - rhoE)) * sumKappa;
                    terms.put("merge-reduce", tp);
                    vPoint += tp;
                    double sumKappaCert = 0.0;
                    for (int v : levels) {
                        sumKappaCert += Math.max(1.0 / v, kappaCert);
                    }
                    vCert += sBar / mRed * sumKappaCert;
                    bCert = Math.max(bCert, sBar / mRed);
                }
            } else {
                MeanStats ms = w.mean();
                // each share is clustered into `target` cells on its own; under a random partition a share is a
                // scaled copy of the input, so the bias is the pilot's at that cell count and kappa_cells shrinks by U
                int cellsPerShare = (int) Math.min(Integer.MAX_VALUE, target);
                bias = interpolate(ms.cells(), column(ms.bias(), q), cellsPerShare);
                double kc = interpolate(ms.cells(), column(ms.kappaCells(), q), cellsPerShare) / units;
                if (Double.isNaN(bias)) {
                    return Prediction.excluded(cellsPerShare + " cells per share is below the smallest measured "
                            + "cell count");
                }
                terms.put("mean-bias", bias);
                if (rhoE > 0.0) {
                    double v = rhoE / (1.0 - rhoE) * kc;
                    terms.put("cell-loss", v);
                    vPoint += v;
                    vCert += v * pilotInflation;
                    bCert = Math.max(bCert, Math.max(1.0, rhoE / (1.0 - rhoE)) * Math.sqrt(kc));
                    if (c.packetization() == Packetization.CELLS) {
                        bias += rhoE; // lost cells take their mass with them when the offset travels whole
                    }
                }
            }
            if (phi > 0.0) {
                double v = phi / (1.0 - phi) * kappa;
                terms.put("node-loss", v * klPoint * klPoint);
                vPoint += v * klPoint * klPoint;
                vCert += phi / (1.0 - phi) * kappaCert;
                bCert = Math.max(bCert, Math.max(1.0, phi / (1.0 - phi)) * cs.costShareMax() * pilotInflation);
            }
            double point = bias + z * Math.sqrt(vPoint);
            // thm:K-fixedq(c): eps_F = sqrt(8 V iota) + (4/3) B iota, or the Chebyshev form sqrt(V |Q| / delta_F),
            // which needs only the variance; HT decoders have only the latter
            double chebyshev = Math.sqrt(vCert * nq / deltaS);
            double bound = bias + (htDecoder ? chebyshev
                    : Math.min(chebyshev, Math.sqrt(8.0 * vCert * iota) + 4.0 / 3.0 * bCert * iota));
            if (point > bestPoint || worst == null) {
                bestPoint = point;
                worst = cs.name();
                worstTerms = terms;
            }
            bestBound = Math.max(bestBound, bound);
        }

        if (req.guarantee() == Guarantee.UNIFORM) {
            bestBound = uniformBound(w, e, c, fam, receivedDraws, levels, aliveUnits, memberEta, deltaS, deltaN, phi);
            worst = "all k-center queries";
            notes.add("UNIFORM bound in units of sqrt(c) of ass:sampling (c=" + cal.get("samplingLemmaConstant")
                    + ")");
        }

        // costs
        long unitRecordsSent = recordsOnWire(w, fam, c, memberSend, units);
        double senderWire = net.wireBytes(e, c, rec, unitRecordsSent, etaPer);
        int activeSenders = switch (c.redundancy()) {
            case RATELESS -> units * group;
            default -> units;
        };
        double bytes = senderWire * activeSenders;
        long records = unitRecordsSent * activeSenders;
        double seconds = wait;
        double leafIngress = tree ? senderWire * c.arity() : senderWire * activeSenders;
        seconds += net.hop(e, c, senderWire, leafIngress).seconds();
        if (c.redundancy() == Redundancy.POST_REPLICATION) {
            double rep = senderWire * c.redundancyLevel();
            bytes += rep * units;
            records += unitRecordsSent * c.redundancyLevel() * units;
            seconds += net.hop(e, c, rep, senderWire).seconds();
        }
        if ((c.redundancy() == Redundancy.FRACTIONAL_REPETITION || c.redundancy() == Redundancy.RATELESS)
                && !e.dataPreReplicated()) {
            double share = net.rowBytes(w) * w.n() / units;
            bytes += share * (group - 1) * units;
            seconds += net.hop(e, c.withM(1), share * (group - 1), share).seconds();
            notes.add("placement of " + (group - 1) + " extra input copies is charged (data not pre-replicated)");
        }
        long internalSend = 0;
        if (tree) {
            internalSend = memberSendFor(c.m(), unitRecords, rhoE, unitLambda, etaPer, c.oversamples());
            long internalRecords = recordsOnWire(w, fam, c, internalSend, 1);
            double internalWire = net.wireBytes(e, c, rec, internalRecords, etaPer);
            for (int i = 0; i < levels.size(); i++) {
                int v = levels.get(i);
                boolean root = i == levels.size() - 1;
                if (!root) {
                    bytes += internalWire * v;
                    records += internalRecords * v;
                    seconds += net.hop(e, c, internalWire, internalWire * c.arity()).seconds();
                    if (c.checkpoint()) {
                        bytes += internalWire * v;
                        records += internalRecords * v;
                        seconds += net.hop(e, c, internalWire, internalWire).seconds();
                    }
                }
            }
        }
        if (rhoE > 0.0 && c.transport() == Transport.DATAGRAM && fam.sampling()) {
            notes.add("no oversampling: accuracy degrades to eps*sqrt(m/M_eta) (cor:loss-eps)");
        }
        boolean certifiable = fam.sampling();
        if (!certifiable) {
            notes.add("mean-coreset bias is measured on the pilot: calibrated only, never certified");
        }
        if (phi > 0.0) {
            notes.add(String.format("calibrated band for the node-loss term: x[%.2f, %.2f] (F1/A1)",
                    cal.get("kappaLawLow"), cal.get("kappaLawHigh")));
        }
        if (NetworkCostModel.merged(c.packetization()) && rhoE > 0.0) {
            notes.add("HT decoder: the certificate assumes the configured erasure rate (a misstated rate biases it)");
        }
        return new Prediction(true, null, bestPoint, bestBound, certifiable, worst, worstTerms, rhoE, phi,
                receivedDraws, memberSend, internalSend, records, bytes, seconds, notes);
    }

    private double uniformBound(WorkloadProfile w, EnvironmentProfile e, Configuration c, Family fam, long received,
                                List<Integer> levels, int aliveUnits, long memberEta, double deltaS, double deltaN,
                                double phi) {
        FamilyStats fs = w.families().get(fam);
        double cst = cal.get("samplingLemmaConstant");
        double pdim = (double) w.d() * w.k() * Math.log(Math.max(2, w.k()));
        double s = Math.max(fs.sTotal(), Math.E);
        int reduces = c.arity() > 0 ? 1 + levels.stream().mapToInt(Integer::intValue).sum() : 1;
        double leafDraws = c.arity() > 0 ? (double) aliveUnits * memberEta : received;
        double epsLeaf = Math.sqrt(cst * s * (pdim * Math.log(s) + Math.log(reduces / deltaS)) / leafDraws);
        double total = 1.0 + epsLeaf;
        if (c.arity() > 0) {
            double epsRed = Math.sqrt(cst * s * (pdim * Math.log(s) + Math.log(reduces / deltaS)) / c.m());
            total *= Math.pow(1.0 + epsRed, levels.size());
            if (epsRed > 1.0) {
                return Double.POSITIVE_INFINITY;
            }
        }
        if (epsLeaf > 1.0) {
            return Double.POSITIVE_INFINITY; // outside the eps <= 1 domain of ass:sampling
        }
        if (phi > 0.0) {
            // thm:random-partition (ii), the full-scale s_max of the standard bound
            double sMax = w.families().get(Family.SENSITIVITY).sMax();
            double lam = pdim * Math.log(Math.max(Math.E, w.n() * sMax)) + Math.log(1.0 / deltaN);
            double phi0 = cst * sMax * lam;
            double epsS = Math.sqrt(phi0 / (1.0 - phi));
            double epsC = (Math.sqrt(phi * phi0) + 2.0 * phi0) / (1.0 - phi);
            double epsR = epsS <= 1.0 ? Math.min(epsS, epsC) : epsC;
            total *= 1.0 + epsR;
        }
        return total - 1.0;
    }

    /** Excess of merged/split records over per-draw: sum_p (c_p/C)^2 h_beta(lambda_p)/lambda_p^2 at full scale. */
    private static double mergedExcess(WorkloadProfile w, Family fam, Configuration c, int q, long drawsPerUnit,
                                       int units) {
        double r = w.tables().scale();
        WorkloadProfile.QBins bins = w.tables().bins().get(fam);
        double[] s2 = bins.share2()[q];
        double acc = 0.0;
        boolean merged = c.packetization() == Packetization.MERGED;
        double mergedFactor = 1.0 - 1.0 / Math.max(1, drawsPerUnit);
        for (int b = 0; b < s2.length; b++) {
            if (s2[b] == 0.0) {
                continue;
            }
            double lambda = drawsPerUnit * units * bins.q()[b] / r;
            double h = merged || lambda <= 0.0 ? mergedFactor
                    : Probability.splitExcess(lambda, c.splitBudget()) / (lambda * lambda);
            acc += s2[b] / r * h;
        }
        return acc;
    }

    /** Records one sender puts on the wire for {@code draws} draws (merged/split count distinct records). */
    private long recordsOnWire(WorkloadProfile w, Family fam, Configuration c, long draws, int units) {
        if (!fam.sampling() || c.packetization() == Packetization.PER_DRAW) {
            return draws;
        }
        double r = w.tables().scale();
        WorkloadProfile.QBins bins = w.tables().bins().get(fam);
        double per = r / units;
        double acc = 0.0;
        for (int b = 0; b < bins.q().length; b++) {
            double lambda = draws * units * bins.q()[b] / r;
            double each = c.packetization() == Packetization.MERGED ? -Math.expm1(-lambda)
                    : expectedSplitRecords(lambda, c.splitBudget());
            acc += per * bins.count()[b] * each;
        }
        return Math.max(1L, Math.min(draws, Math.round(acc)));
    }

    private static double expectedSplitRecords(double lambda, int beta) {
        if (lambda < 1e-4) {
            return lambda;
        }
        if (lambda > 500.0) {
            return lambda / beta + 0.5;
        }
        int max = (int) Math.ceil(lambda + 12.0 * Math.sqrt(lambda) + 30.0);
        double p = Math.exp(-lambda);
        double acc = 0.0;
        for (int kk = 1; kk <= max; kk++) {
            p *= lambda / kk;
            acc += p * ((kk + beta - 1) / beta);
        }
        return acc;
    }

    private static long memberSendFor(long target, int unitRecords, double loss, double lambda, double eta,
                                      boolean oversample) {
        if (!oversample) {
            return target;
        }
        long s = oversampled(target, unitRecords, loss, lambda, eta);
        return s < 0 ? target : s;
    }

    /** Draws delivered at the eta-quantile when {@code sent} draws travel in units of {@code unitRecords}. */
    public static long delivered(long sent, int unitRecords, double loss, double lambda, double eta) {
        if (loss <= 0.0) {
            return sent;
        }
        long units = (sent + unitRecords - 1) / unitRecords;
        long got = Probability.deliveredLowerBound(units, loss, lambda, eta);
        return Math.min(sent, got * unitRecords);
    }

    /** Least count whose eta-quantile of delivered draws reaches {@code target}; -1 if none below 64x. */
    public static long oversampled(long target, int unitRecords, double loss, double lambda, double eta) {
        if (loss <= 0.0) {
            return target;
        }
        long lo = target;
        long hi = target;
        while (delivered(hi, unitRecords, loss, lambda, eta) < target) {
            hi *= 2;
            if (hi > 64 * target + 64L * unitRecords) {
                return -1;
            }
        }
        while (lo < hi) {
            long mid = (lo + hi) >>> 1;
            if (delivered(mid, unitRecords, loss, lambda, eta) >= target) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private static double[] column(double[][] a, int q) {
        double[] out = new double[a.length];
        for (int j = 0; j < a.length; j++) {
            out[j] = a[j][q];
        }
        return out;
    }

    /** Log-linear in m between grid points; the last value beyond the grid (bias falls with m); NaN below it. */
    static double interpolate(int[] grid, double[] values, int m) {
        if (grid.length == 0 || m < grid[0]) {
            return Double.NaN;
        }
        for (int j = 1; j < grid.length; j++) {
            if (m <= grid[j]) {
                double t = Math.log((double) m / grid[j - 1]) / Math.log((double) grid[j] / grid[j - 1]);
                return values[j - 1] + t * (values[j] - values[j - 1]);
            }
        }
        return values[grid.length - 1];
    }
}
