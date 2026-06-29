package noromalization;

import utility.data.Tuple;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class Normalizer {

    private final List<Tuple<Integer, Integer>> bounds;

    public Normalizer(List<Tuple<Integer, Integer>> bounds) {
        this.bounds = bounds;
    }

    public double[][] normalize(double[][] denormalized) {
        Objects.requireNonNull(denormalized, "denormalized cannot be null");
        int vars = denormalized.length;
        if (vars != this.bounds.size()) throw new IllegalArgumentException("position rows must match bounds size");

        int cols = denormalized[0].length;
        double[][] norm = new double[vars][cols];

        for (int v = 0; v < vars; v++) {
            double min = bounds.get(v).getFirst();
            double max = bounds.get(v).getSecond();
            double range = Math.max(1.0, max - min);
            for (int f = 0; f < cols; f++) {
                double val = denormalized[v][f];
                double n = (val - min) / range;
                if (Double.isNaN(n) || Double.isInfinite(n)) n = 0.0;
                norm[v][f] = Math.max(0.0, Math.min(1.0, n));
            }
        }
        return norm;
    }


    public double[][] denormalize(double[][] normalized) {
        Objects.requireNonNull(normalized, "normalized cannot be null");
        int vars = normalized.length;
        if (vars != this.bounds.size()) throw new IllegalArgumentException("position rows must match bounds size");

        int cols = normalized[0].length;
        double[][] denorm = new double[vars][cols];

        for (int v = 0; v < vars; v++) {
            double min = bounds.get(v).getFirst();
            double max = bounds.get(v).getSecond();
            double range = Math.max(1.0, max - min);
            for (int f = 0; f < cols; f++) { // fixed: use cols, not vars
                double n = Math.max(0.0, Math.min(1.0, normalized[v][f]));
                denorm[v][f] = min + n * range;
            }
        }
        return denorm;
    }
}