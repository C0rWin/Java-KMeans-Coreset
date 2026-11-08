package io.github.c0rwin.coreset.planner;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.planner.Configuration.Packetization;
import io.github.c0rwin.coreset.planner.Configuration.Redundancy;
import io.github.c0rwin.coreset.planner.Configuration.StragglerPolicy;
import io.github.c0rwin.coreset.planner.Configuration.Transport;
import io.github.c0rwin.coreset.planner.Requirement.Mode;
import io.github.c0rwin.coreset.planner.Requirement.Objective;
import io.github.c0rwin.coreset.planner.validate.PlanSimulator;
import java.util.Locale;
import org.junit.jupiter.api.Test;

/**
 * Runs planned configurations in-process with injected faults and compares the measured (1 - delta)-quantile of the
 * worst candidate error with the prediction.
 *
 * <p>Tolerance: the measured quantile must not exceed the certified bound, and must lie within
 * {@code [LOW, HIGH] x} the point prediction. The point prediction is a Gaussian quantile at {@code delta/(2|Q|)}
 * (Bonferroni over |Q| = 7 correlated candidates), which overstates the quantile of the maximum when candidates are
 * correlated, hence LOW well below 1. HIGH allows the kappa-law calibration spread (0.95-1.03, F1/A1), the
 * sampling error of an empirical 95% quantile from {@value #TRIALS} trials (roughly +-20% for a near-Gaussian
 * maximum) and the pilot estimate of sigma2 on 10k points.
 */
class PredictedVsMeasuredTest {

    static final int TRIALS = 40;
    static final double LOW = 0.4;
    static final double HIGH = 1.3;

    private static void check(String label, EnvironmentProfile e, Configuration skeleton) {
        Planner planner = Fixtures.planner();
        WorkloadProfile w = Fixtures.mixtureProfile();
        Requirement r = new Requirement(0.05, 0.05, Fixtures.K, Requirement.Guarantee.INSTANCE, Mode.CALIBRATED,
                Objective.MIN_BYTES, 1, 1);
        Planner.Evaluated x = planner.evaluate(w, e, r, skeleton);
        assertTrue(x.feasible(), x::reason);
        PlanSimulator sim = new PlanSimulator(Fixtures.mixture(), w.tables().centers());
        PlanSimulator.Result res = sim.run(x.config(), x.prediction(), e, r, TRIALS, 99L);
        double ratio = res.quantile() / x.prediction().epsPoint();
        System.out.printf(Locale.ROOT, "%s %s: point %.4f bound %.4f measured q95 %.4f (mean %.4f) ratio %.3f "
                        + "lost %.3f erased %.3f%n", label, x.config().describe(), x.prediction().epsPoint(),
                x.prediction().epsBound(), res.quantile(), res.mean(), ratio, res.meanLostFraction(),
                res.meanErasure());
        assertTrue(res.quantile() <= x.prediction().epsBound(), label + ": measured above the certified bound");
        assertTrue(ratio >= LOW && ratio <= HIGH, label + ": measured/point " + ratio);
    }

    private static Configuration sampling(Packetization p, Transport t, Redundancy red, int level) {
        return new Configuration(Family.SENSITIVITY, p, 0, t, t == Transport.DATAGRAM_FEC ? 16 : 0,
                t == Transport.DATAGRAM_FEC ? 4 : 0, red, level, 0, 0, false, StragglerPolicy.WAIT, 1);
    }

    @Test
    void erasuresWithOversampledPerDrawRecords() {
        check("erasure", Fixtures.env(0.2, 1.0, 8, 0.0),
                sampling(Packetization.PER_DRAW, Transport.DATAGRAM_OVERSAMPLE, Redundancy.NONE, 0));
    }

    @Test
    void burstyErasuresAndNodeLoss() {
        check("bursty+nodes", Fixtures.env(0.1, 4.0, 16, 0.2),
                sampling(Packetization.PER_DRAW, Transport.DATAGRAM_OVERSAMPLE, Redundancy.NONE, 0));
    }

    @Test
    void postReplicationAgainstNodeLoss() {
        check("post-r1", Fixtures.env(0.0, 1.0, 16, 0.3),
                sampling(Packetization.PER_DRAW, Transport.RELIABLE, Redundancy.POST_REPLICATION, 1));
    }
}
