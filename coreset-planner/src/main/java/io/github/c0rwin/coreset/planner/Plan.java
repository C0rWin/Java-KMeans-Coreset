package io.github.c0rwin.coreset.planner;

import io.github.c0rwin.coreset.planner.model.ResilienceModel.Prediction;
import java.util.List;

/**
 * The planner's answer.
 *
 * @param feasible whether any configuration meets the requirement in its mode
 * @param config the cheapest feasible configuration (or the most accurate one when none is feasible)
 * @param prediction its predicted accuracy and cost
 * @param objectiveValue the minimised scalar
 * @param satisfies the modes whose prediction meets eps: CERTIFIED (finite-candidate union-bound bound) and/or
 *     CALIBRATED (point prediction)
 * @param binding which constraint fixes m and why the alternatives lost
 * @param alternatives cheapest feasible configuration per family, packetization and transport, for comparison
 * @param model resilience-model id with calibration version
 * @param evaluated configuration skeletons evaluated (each at its least feasible m)
 */
public record Plan(boolean feasible, Configuration config, Prediction prediction, double objectiveValue,
                   List<Requirement.Mode> satisfies, List<String> binding, List<Alternative> alternatives,
                   String model, int evaluated) {

    /** A runner-up: its configuration, prediction and objective value. */
    public record Alternative(String config, double epsPoint, double epsBound, double bytes, double seconds,
                              double objectiveValue, String status) {
    }
}
