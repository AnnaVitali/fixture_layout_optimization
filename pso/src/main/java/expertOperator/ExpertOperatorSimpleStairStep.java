package expertOperator;

import org.json.JSONObject;
import polygon.Fixture;
import utility.FixtureUtility;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class ExpertOperatorSimpleStairStep {

    private static final String FILE_PATH = "./resources/expert/expert_operator_simple_stair_step.json";

    public static void main(String[] args) {
        double [] expertX = {20.00, 365.0, 705.0, 2500.0, 2500.0, 2500.0,  2500.0};
        double[] expertY = {80.0, 80.0, 100.0,  1500.0, 1500.0, 1500.0, 1500.0};
        double[] expertAngle = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double[] expertT = {1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        double[][] position = new double[][] { expertX, expertY, expertAngle, expertT };

        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(position);

        int n = fixtures.size();
        double[] x0 = new double[n];
        double[] y0 = new double[n];
        double[] x1 = new double[n];
        double[] y1 = new double[n];
        double[] x2 = new double[n];
        double[] y2 = new double[n];
        double[] x3 = new double[n];
        double[] y3 = new double[n];

        double[] fixtureCenterX = new double[n];
        double[] fixtureCenterY = new double[n];
        double[] barCenter = new double[n];
        double[] fixtureAngle = new double[n];
        double[] fixtureType = new double[n];
        int[] selectedFixture = new int[n];

        // Fill arrays using stable numeric indices (no indexOf)
        for (int i = 0; i < n; i++) {
            Fixture fixture = fixtures.get(i);

            x0[i] = fixture.getVertices().get(0).getFirst();
            y0[i] = fixture.getVertices().get(0).getSecond();

            x1[i] = fixture.getVertices().get(1).getFirst();
            y1[i] = fixture.getVertices().get(1).getSecond();

            x2[i] = fixture.getVertices().get(2).getFirst();
            y2[i] = fixture.getVertices().get(2).getSecond();

            x3[i] = fixture.getVertices().get(3).getFirst();
            y3[i] = fixture.getVertices().get(3).getSecond();

            fixtureCenterX[i] = fixture.getCentroid().getFirst();
            fixtureCenterY[i] = fixture.getCentroid().getSecond();

            // Use base center and angle/type from the fixture object (preserves exact computed values)
            if (fixture.getBaseCenter() != null) {
                barCenter[i] = fixture.getBaseCenter().getFirst();
            } else {
                barCenter[i] = Double.NaN;
            }

            fixtureAngle[i] = fixture.getAngle();
            fixtureType[i] = fixture.getType();
            selectedFixture[i] = fixtureType[i] != 0.0 ? 1 : 0;
        }

        // Build JSON
        JSONObject obj = new JSONObject();
        obj.put("x", x0);
        obj.put("y", y0);
        obj.put("x1", x1);
        obj.put("y1", y1);
        obj.put("x2", x2);
        obj.put("y2", y2);
        obj.put("x3", x3);
        obj.put("y3", y3);
        obj.put("angle", fixtureAngle);
        obj.put("fixtures_center_x", fixtureCenterX);
        obj.put("fixtures_center_y", fixtureCenterY);
        obj.put("bars_center", barCenter);
        obj.put("fixture_type", fixtureType);
        obj.put("selected_fixture", selectedFixture);

        String jsonString = obj.toString(4);
        try {
            Files.write(Paths.get(FILE_PATH), jsonString.getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }

}
