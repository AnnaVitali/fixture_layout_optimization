package utility;

import org.json.JSONObject;

public class SavedSolutionConverter {

    public static double[][] toHillClimbingSeed(JSONObject savedSolution) {
        double[] savedX = savedSolution.getJSONArray("x").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();
        double[] savedY = savedSolution.getJSONArray("y").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();
        double[] angle = savedSolution.getJSONArray("angle").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();
        double[] t = savedSolution.getJSONArray("fixture_type").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();

        double[] x = new double[savedX.length];
        double[] y = new double[savedY.length];

        for (int i = 0; i < savedX.length; i++) {
            int type = (int) Math.floor(t[i]);
            if (type <= 0) {
                x[i] = savedX[i];
                y[i] = savedY[i];
                continue;
            }

            double width = FixtureDimension.getWidth(type);
            double height = FixtureDimension.getHeight(type);
            double theta = Math.toRadians(angle[i]);

            double centerOffsetX = Math.cos(theta) * (width / 2.0) - Math.sin(theta) * (height / 2.0);
            double centerOffsetY = Math.sin(theta) * (width / 2.0) + Math.cos(theta) * (height / 2.0);

            x[i] = savedX[i] + centerOffsetX - (width / 2.0);
            y[i] = savedY[i] + centerOffsetY - (height / 2.0);
        }

        return new double[][]{x, y, angle, t};
    }
}