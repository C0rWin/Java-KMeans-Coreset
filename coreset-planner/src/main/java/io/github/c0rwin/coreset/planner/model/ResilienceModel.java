package io.github.c0rwin.coreset.planner.model;

import io.github.c0rwin.coreset.planner.Configuration;
import io.github.c0rwin.coreset.planner.EnvironmentProfile;
import io.github.c0rwin.coreset.planner.Requirement;
import io.github.c0rwin.coreset.planner.WorkloadProfile;
import java.util.List;
import java.util.Map;

/**
 * Predicts accuracy and cost of one configuration. Implementations hold the formulas; the planner only searches.
 * Implementations must be stateless after construction and monotone in the sense the planner relies on: for fixed
 * everything else, a larger {@code m} never raises {@code epsPoint}/{@code epsBound} by more than rounding.
 */
public interface ResilienceModel {

    /** Identifier echoed in plans, including the calibration version. */
    String id();

    Prediction predict(WorkloadProfile workload, EnvironmentProfile env, Requirement req, Configuration config);

    /**
     * @param admissible false when a hard rule excludes the configuration regardless of m (then {@code reason} says
     *     which); eps fields are then {@code +inf}
     * @param reason the excluding rule, or null
     * @param epsPoint central prediction of the (1 - delta)-quantile of the worst relative error over the guarantee's
     *     query set
     * @param epsBound conservative bound on the same quantity
     * @param certifiable whether {@code epsBound} rests only on proven laws plus the union bound over the finite
     *     candidate set (instance constants still come from the pilot); false when an ingredient is purely empirical,
     *     e.g. the mean coreset's pilot-measured bias. Non-certifiable predictions never satisfy CERTIFIED mode.
     * @param worstCandidate candidate attaining the maximum (INSTANCE) or "all k-center queries" (UNIFORM)
     * @param terms relative-variance terms of the worst candidate, by name (point version)
     * @param effectiveErasure per-record erasure probability after the transport (0 for reliable)
     * @param lostFraction conservative (1 - node budget) quantile of the fraction of data whose summary is lost
     * @param receivedDraws conservative count of draws (or cells) the decoder pools
     * @param drawsPerSender draws (or cells) each leaf sender emits, oversampling included; internal tree nodes
     *     emit {@code internalDraws}
     * @param internalDraws draws each internal tree node emits after its reduce (0 for flat)
     * @param recordsSent all records put on the wire by workers, replicas and checkpoints included
     * @param bytes all bytes on the wire, headers, FEC, retransmissions and placement included
     * @param seconds predicted completion time
     * @param notes approximations and rules that applied
     */
    record Prediction(boolean admissible, String reason, double epsPoint, double epsBound, boolean certifiable,
                      String worstCandidate, Map<String, Double> terms, double effectiveErasure, double lostFraction,
                      long receivedDraws, long drawsPerSender, long internalDraws, long recordsSent, double bytes,
                      double seconds, List<String> notes) {

        public static Prediction excluded(String reason) {
            return new Prediction(false, reason, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, false, null,
                    Map.of(), Double.NaN, Double.NaN, 0, 0, 0, 0, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                    List.of());
        }

        /** The prediction a mode compares with eps; {@code +inf} for CERTIFIED when not certifiable. */
        public double eps(Requirement.Mode mode) {
            if (mode == Requirement.Mode.CALIBRATED) {
                return epsPoint;
            }
            return certifiable ? epsBound : Double.POSITIVE_INFINITY;
        }
    }
}
