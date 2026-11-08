package io.github.c0rwin.coreset.core.metrics;

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/** Point-in-time copy of metrics; maps are immutable and iterate in name order. */
public record MetricsSnapshot(Map<String, Long> counters, Map<String, Distribution> distributions) {

    public MetricsSnapshot {
        counters = Collections.unmodifiableSortedMap(new TreeMap<>(counters));
        distributions = Collections.unmodifiableSortedMap(new TreeMap<>(distributions));
    }

    /**
     * Compact JSON: {@code {"counters":{name:long,...},"distributions":{name:{"count":..,"sum":..,"min":..,"max":..,
     * "mean":..,"p50":..,"p95":..,"p99":..},...}}}. Non-finite doubles are written as {@code null}.
     */
    public String toJson() {
        StringBuilder sb = new StringBuilder(64 + 32 * counters.size() + 160 * distributions.size());
        sb.append("{\"counters\":{");
        boolean first = true;
        for (Map.Entry<String, Long> e : counters.entrySet()) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            appendString(sb, e.getKey());
            sb.append(':').append(e.getValue().longValue());
        }
        sb.append("},\"distributions\":{");
        first = true;
        for (Map.Entry<String, Distribution> e : distributions.entrySet()) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            Distribution d = e.getValue();
            appendString(sb, e.getKey());
            sb.append(":{\"count\":").append(d.count());
            appendField(sb, "sum", d.sum());
            appendField(sb, "min", d.min());
            appendField(sb, "max", d.max());
            appendField(sb, "mean", d.mean());
            appendField(sb, "p50", d.p50());
            appendField(sb, "p95", d.p95());
            appendField(sb, "p99", d.p99());
            sb.append('}');
        }
        return sb.append("}}").toString();
    }

    private static void appendField(StringBuilder sb, String key, double value) {
        sb.append(",\"").append(key).append("\":");
        if (Double.isFinite(value)) {
            sb.append(value);
        } else {
            sb.append("null");
        }
    }

    private static void appendString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append('"');
    }
}
