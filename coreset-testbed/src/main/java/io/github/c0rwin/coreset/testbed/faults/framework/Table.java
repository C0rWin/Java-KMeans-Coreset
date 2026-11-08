package io.github.c0rwin.coreset.testbed.faults.framework;

import java.util.ArrayList;
import java.util.List;

/** A result table, rendered as markdown in the report and saved as {@code <name>.csv}. Cells are preformatted. */
final class Table {

    final String name;
    private final String[] columns;
    private final List<String[]> rows = new ArrayList<>();

    Table(String name, String... columns) {
        this.name = name;
        this.columns = columns;
    }

    Table row(Object... cells) {
        if (cells.length != columns.length) {
            throw new IllegalArgumentException(name + ": " + cells.length + " cells for " + columns.length + " columns");
        }
        String[] r = new String[cells.length];
        for (int i = 0; i < cells.length; i++) {
            r[i] = String.valueOf(cells[i]);
        }
        rows.add(r);
        return this;
    }

    String markdown() {
        StringBuilder sb = new StringBuilder("|");
        for (String c : columns) {
            sb.append(' ').append(cell(c)).append(" |");
        }
        sb.append("\n|");
        sb.append("---|".repeat(columns.length));
        for (String[] r : rows) {
            sb.append("\n|");
            for (String c : r) {
                sb.append(' ').append(cell(c)).append(" |");
            }
        }
        return sb.toString();
    }

    String csv() {
        StringBuilder sb = new StringBuilder();
        line(sb, columns);
        for (String[] r : rows) {
            line(sb, r);
        }
        return sb.toString();
    }

    private static void line(StringBuilder sb, String[] cells) {
        for (int i = 0; i < cells.length; i++) {
            sb.append(i == 0 ? "" : ",").append(csv(cells[i]));
        }
        sb.append('\n');
    }

    static String cell(String s) {
        return s.replace("|", "\\|");
    }

    static String csv(String s) {
        return s.contains(",") || s.contains("\"") ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }
}
