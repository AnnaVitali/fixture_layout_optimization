package utility;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import org.json.JSONObject;
import polygon.Fixture;
import polygon.Rectangle;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class SaveResult {

    public static void saveResult(String filePath, double[][] result, GeometryConstraint geometryConstraint, FixtureConstraints fixtureConstraints, NoOverlapConstraint noOverlapConstraint, List<Rectangle> rectangles){
        double[] tRaw = (result.length > 3 && result[3] != null) ? result[3] : new double[0];
        double[] tClean = new double[tRaw.length];
        for (int i = 0; i < tRaw.length; i++) {
            double v = Math.floor(tRaw[i]);
            if (v < 0) v = 0.0;
            tClean[i] = v;
        }

        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(result);

        //List<Rectangle> rectanglesToSave = FixtureUtility.getRectangleFromFixture(fixtures, tClean);

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
        double[] angle = new double[n];
        double[] fixtureType = new double[n];
        int[] selectedFixture = new int[n];

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

            if (fixture.getBaseCenter() != null) {
                barCenter[i] = fixture.getBaseCenter().getFirst();
            } else {
                barCenter[i] = Double.NaN;
            }

            angle[i] = fixture.getAngle();
            fixtureType[i] = fixture.getType();
            selectedFixture[i] = fixtureType[i] != 0.0 ? 1 : 0;
        }

        JSONObject obj = new JSONObject();
        obj.put("x", x0);
        obj.put("y", y0);
        obj.put("x1", x1);
        obj.put("y1", y1);
        obj.put("x2", x2);
        obj.put("y2", y2);
        obj.put("x3", x3);
        obj.put("y3", y3);
        obj.put("angle", angle);
        obj.put("fixtures_center_x", fixtureCenterX);
        obj.put("fixtures_center_y", fixtureCenterY);
        obj.put("bars_center", barCenter);
        obj.put("fixture_type", fixtureType);
        obj.put("selected_fixture", selectedFixture);

        int violation = 0;
        for(Fixture fixture : fixtures){
            violation += geometryConstraint.computePenaltyGeometryViolation(fixture.getVertices());
        }


        System.out.println("Geometry violation: " + violation);

        violation += geometryConstraint.computePenaltySecurityDistanceViolation(fixtures);
        System.out.println("Security distance violation: " + violation);

        violation += noOverlapConstraint.computePenaltyOverlapBetweenFixtures(rectangles);
        System.out.println("Overlap violation: " + violation);

        violation += noOverlapConstraint.computePenaltyOverlapWithHoles(rectangles);
        System.out.println("Hole overlap violation: " + violation);

        String jsonString = obj.toString(4);
        try {
            Files.write(Paths.get(filePath), jsonString.getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //saveResultForTest(fixtures, rectanglesToSave);
    }

//    private static void saveResultForTest(List<Fixture> fixtures, List<Rectangle> rectangles){
//        JSONObject obj = new JSONObject();
//        obj.put("fixtures", fixtures);
//        obj.put("rectangles", rectangles);
//
//        String jsonString = obj.toString(4);
//        try {
//            Files.write(Paths.get("./resources/test_result.json"), jsonString.getBytes());
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//    }
}
