package workpiece.coffeeTable;

// java
import org.json.JSONArray;
import org.json.JSONObject;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.IntStream;

public class DebugFixtureGeometry {
    public static void main(String[] args) throws Exception {
        Path p = Path.of("resources/rl/rl_coffee_table.json");
        String text = Files.readString(p);
        JSONObject root = new JSONObject(text);

        JSONArray xs = root.getJSONArray("x");
        JSONArray ys = root.getJSONArray("y");
        JSONArray x1s = root.getJSONArray("x1");
        JSONArray y1s = root.getJSONArray("y1");
        JSONArray x2s = root.getJSONArray("x2");
        JSONArray y2s = root.getJSONArray("y2");
        JSONArray x3s = root.getJSONArray("x3");
        JSONArray y3s = root.getJSONArray("y3");
        JSONArray centersX = root.getJSONArray("fixtures_center_x");
        JSONArray types = root.getJSONArray("fixture_type");

        // compute workpiece bbox from all given corner arrays
        double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
        for (JSONArray arr : new JSONArray[]{xs, x1s, x2s, x3s}) {
            for (int i = 0; i < arr.length(); i++) {
                double vx = arr.getDouble(i);
                minX = Math.min(minX, vx);
                maxX = Math.max(maxX, vx);
            }
        }
        for (JSONArray arr : new JSONArray[]{ys, y1s, y2s, y3s}) {
            for (int i = 0; i < arr.length(); i++) {
                double vy = arr.getDouble(i);
                minY = Math.min(minY, vy);
                maxY = Math.max(maxY, vy);
            }
        }
        System.out.printf("Workpiece bbox: minX=%.3f maxX=%.3f minY=%.3f maxY=%.3f%n", minX, maxX, minY, maxY);

        int n = xs.length();
        for (int i = 0; i < n; i++) {
            double[][] expected = {
                    { xs.getDouble(i), ys.getDouble(i) },
                    { x1s.getDouble(i), y1s.getDouble(i) },
                    { x2s.getDouble(i), y2s.getDouble(i) },
                    { x3s.getDouble(i), y3s.getDouble(i) }
            };

            // Replace this block with your actual vertex computation for fixture i.
            // For now we just reuse the JSON values to show comparison framework.
            double[][] computed = {
                    { xs.getDouble(i), ys.getDouble(i) },   // change to your computed x,y
                    { x1s.getDouble(i), y1s.getDouble(i) }, // change to your computed x1,y1
                    { x2s.getDouble(i), y2s.getDouble(i) },
                    { x3s.getDouble(i), y3s.getDouble(i) }
            };

            System.out.printf("Fixture %d type=%d centerX=%.1f%n", i, types.getInt(i), centersX.getDouble(i));
            System.out.println("  expected vertices:");
            for (double[] v : expected) System.out.printf("    (%.3f, %.3f)%n", v[0], v[1]);
            System.out.println("  computed vertices:");
            for (double[] v : computed) System.out.printf("    (%.3f, %.3f)%n", v[0], v[1]);

            // bbox check
            boolean outside = false;
            for (double[] v : computed) {
                if (v[0] < minX - 1e-6 || v[0] > maxX + 1e-6 || v[1] < minY - 1e-6 || v[1] > maxY + 1e-6) {
                    outside = true;
                }
            }
            if (outside) {
                System.out.println("  -> Geometry violation: one or more vertices are outside the workpiece bbox");
            }

            // quick duplicate center/type check
            final int idx = i;
            IntStream.range(0, n).filter(j -> j != idx).forEach(j -> {
                try {
                    if (Math.abs(centersX.getDouble(j) - centersX.getDouble(idx)) < 1e-6 &&
                            types.getInt(j) == types.getInt(idx)) {
                        System.out.printf("  -> Warning: fixture %d and fixture %d have same centerX and same type%n", idx, j);
                    }
                } catch (Exception ignored) {}
            });

            System.out.println();
        }
    }
}

