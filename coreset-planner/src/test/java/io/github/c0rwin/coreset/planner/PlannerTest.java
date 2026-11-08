package io.github.c0rwin.coreset.planner;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.planner.Configuration.Packetization;
import io.github.c0rwin.coreset.planner.Configuration.Redundancy;
import io.github.c0rwin.coreset.planner.Configuration.StragglerPolicy;
import io.github.c0rwin.coreset.planner.Configuration.Transport;
import io.github.c0rwin.coreset.planner.Requirement.Mode;
import io.github.c0rwin.coreset.planner.Requirement.Objective;
import io.github.c0rwin.coreset.planner.model.ResilienceModel.Prediction;
import java.util.List;
import org.junit.jupiter.api.Test;

class PlannerTest {

    private static final Requirement REQ = Requirement.of(0.1, 0.05, Fixtures.K, Objective.MIN_BYTES);

    private static List<EnvironmentProfile> environments() {
        return List.of(
                Fixtures.env(0.0, 1.0, 8, 0.0),
                Fixtures.env(0.1, 4.0, 8, 0.0),
                Fixtures.env(0.3, 1.0, 16, 0.1),
                new EnvironmentProfile(0.05, 1.0, 12, 0.05, 0.05, EnvironmentProfile.Partition.ADVERSARIAL, 2, 0.5,
                        0.02, 12.5e6, 1400, false));
    }

    @Test
    void returnedPlansMeetTheModelsOwnConstraintsAndMIsMinimal() {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        for (EnvironmentProfile e : environments()) {
            for (Mode mode : Mode.values()) {
                Requirement r = REQ.withMode(mode);
                Plan plan = planner.plan(w, e, r);
                assertTrue(plan.feasible(), () -> "infeasible for " + e + ": " + plan.binding());
                Prediction p = planner.model().predict(w, e, r, plan.config());
                assertTrue(p.admissible(), p::reason);
                assertTrue(p.eps(mode) <= r.eps(), () -> plan.config().describe() + " eps " + p.eps(mode));
                assertTrue(plan.satisfies().contains(mode));
                assertEquals(p.bytes(), plan.prediction().bytes(), 1e-9 * p.bytes());
                if (plan.config().m() > Math.max(r.k(), 1)) {
                    Prediction below = planner.model().predict(w, e, r, plan.config().withM(plan.config().m() - 1));
                    assertFalse(below.admissible() && below.eps(mode) <= r.eps(),
                            () -> "m-1 is also feasible for " + plan.config().describe());
                }
                for (Plan.Alternative a : plan.alternatives()) {
                    assertTrue((mode == Mode.CERTIFIED ? a.epsBound() : a.epsPoint()) <= r.eps(), a::config);
                    assertTrue(a.objectiveValue() >= plan.objectiveValue() * (1 - 1e-12), a::config);
                }
            }
        }
    }

    @Test
    void certifiedBoundIsNeverBelowThePointPrediction() {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        EnvironmentProfile e = Fixtures.env(0.2, 3.0, 8, 0.05);
        for (Configuration s : planner.skeletons(e)) {
            Prediction p = planner.model().predict(w, e, REQ, s.withM(4000));
            if (p.admissible()) {
                assertTrue(p.epsBound() >= p.epsPoint() - 1e-12, s::describe);
            }
        }
    }

    @Test
    void higherErasureRateNeverYieldsACheaperFeasiblePlan() {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        for (Objective o : Objective.values()) {
            Requirement r = new Requirement(0.1, 0.05, Fixtures.K, Requirement.Guarantee.INSTANCE, Mode.CERTIFIED, o,
                    1.0, 1.0);
            for (double burst : new double[] {1.0, 4.0}) {
                double prev = 0.0;
                for (double rho : new double[] {0.0, 0.02, 0.05, 0.1, 0.2, 0.3}) {
                    Plan plan = planner.plan(w, Fixtures.env(rho, burst, 8, 0.05), r);
                    assertTrue(plan.feasible());
                    double cost = plan.objectiveValue();
                    double floor = prev;
                    assertTrue(cost >= floor * (1 - 1e-9), () -> o + " burst " + burst + " rho " + rho + ": " + cost
                            + " < " + floor + " (" + plan.config().describe() + ")");
                    prev = cost;
                }
            }
        }
    }

    @Test
    void meanCoresetIsExcludedOnLossyDatagramsWithoutFecOrCf() {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        // with several records per datagram thm:deterministic-fragile bounds failure below by rho, so rho > eta = 0.0125
        for (double rho : new double[] {0.05, 0.1, 0.3}) {
            EnvironmentProfile e = Fixtures.env(rho, 1.0, 8, 0.0);
            for (Transport t : new Transport[] {Transport.DATAGRAM}) {
                Configuration c = new Configuration(Family.MEAN, Packetization.CELLS, 0, t, 0, 0, Redundancy.NONE, 0, 0,
                        0, false, StragglerPolicy.WAIT, 400);
                Prediction p = planner.model().predict(w, e, REQ, c);
                assertFalse(p.admissible());
                assertTrue(p.reason().contains("thm:deterministic-fragile"), p.reason());
            }
            Plan plan = planner.plan(w, e, REQ);
            Configuration win = plan.config();
            assertFalse(win.family() == Family.MEAN && win.packetization() == Packetization.CELLS
                    && (win.transport() == Transport.DATAGRAM || win.transport() == Transport.DATAGRAM_OVERSAMPLE),
                    win::describe);
            for (Plan.Alternative a : plan.alternatives()) {
                assertFalse(a.config().startsWith("mean/cells/datagram/") , a::config);
            }
        }
    }

    @Test
    void perDrawBeatsMergedWheneverRecordsCanBeLost() {
        Planner planner = Fixtures.planner();
        for (WorkloadProfile w : List.of(Fixtures.mixtureProfile(), Fixtures.heavyProfile())) {
            for (double rho : new double[] {0.05, 0.2}) {
                EnvironmentProfile e = Fixtures.env(rho, 1.0, 8, 0.0);
                for (Family f : new Family[] {Family.UNIFORM, Family.SENSITIVITY, Family.LIGHTWEIGHT}) {
                    for (Transport t : new Transport[] {Transport.DATAGRAM, Transport.DATAGRAM_OVERSAMPLE}) {
                        double perDraw = best(planner, w, e, f, Packetization.PER_DRAW, t);
                        double merged = best(planner, w, e, f, Packetization.MERGED, t);
                        assertTrue(perDraw <= merged, () -> w.datasetId() + " " + f + " " + t + " rho " + rho
                                + ": per-draw " + perDraw + " > merged " + merged);
                    }
                }
                Plan plan = planner.plan(w, e, REQ);
                if (plan.config().family().sampling() && plan.config().transport() != Transport.RELIABLE) {
                    assertNotEquals(Packetization.MERGED, plan.config().packetization(), plan.config().describe());
                }
            }
        }
    }

    private static double best(Planner planner, WorkloadProfile w, EnvironmentProfile e, Family f, Packetization p,
                               Transport t) {
        Configuration s = new Configuration(f, p, 0, t, 0, 0, Redundancy.NONE, 0, 0, 0, false, StragglerPolicy.WAIT, 1);
        Planner.Evaluated x = planner.evaluate(w, e, REQ, s);
        return x.feasible() ? x.prediction().bytes() : Double.POSITIVE_INFINITY;
    }

    @Test
    void adversarialPartitionNeedsRedundancyWhenNodesFail() {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        EnvironmentProfile e = new EnvironmentProfile(0.0, 1.0, 16, 0.0, 0.05,
                EnvironmentProfile.Partition.ADVERSARIAL, 0, 0.0, 5e-4, 125e6, 1400, false);
        Plan plan = planner.plan(w, e, REQ);
        assertTrue(plan.feasible(), () -> String.join("\n", plan.binding()));
        assertNotEquals(Redundancy.NONE, plan.config().redundancy(), plan.config().describe());
    }

    @Test
    void uniformGuaranteeIsReportedAndMuchMoreDemanding() {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        EnvironmentProfile e = Fixtures.env(0.0, 1.0, 8, 0.0);
        Requirement uni = new Requirement(0.5, 0.05, Fixtures.K, Requirement.Guarantee.UNIFORM, Mode.CERTIFIED,
                Objective.MIN_BYTES, 1, 1);
        Plan u = planner.plan(w, e, uni);
        Plan i = planner.plan(w, e, new Requirement(0.5, 0.05, Fixtures.K, Requirement.Guarantee.INSTANCE,
                Mode.CERTIFIED, Objective.MIN_BYTES, 1, 1));
        assertTrue(u.feasible());
        assertTrue(u.config().family().sampling());
        assertEquals("all k-center queries", u.prediction().worstCandidate());
        assertTrue(u.prediction().bytes() > i.prediction().bytes());
    }
}
