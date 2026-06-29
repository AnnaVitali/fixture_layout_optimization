package utility;

import polygon.Fixture;
import polygon.Polygon;
import polygon.Rectangle;
import utility.data.Triple;
import utility.data.Tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FixtureUtility {

    public static Tuple<Float, Float> computeCoordinateWithRotation(Float x, Float y, Float angle, Float cx, Float cy) {
        double theta = Math.toRadians(angle);
        double cos = Math.cos(theta);
        double sin = Math.sin(theta);

        double dx = x - cx;
        double dy = y - cy;

        double rx = dx * cos - dy * sin + cx;
        double ry = dx * sin + dy * cos + cy;

        return new Tuple<>((float) rx, (float) ry);
    }


    public static List<Fixture> getFixturesFromPosition(double[][] position) {
        double[] x = position[0];
        double[] y = position[1];
        double[] angle = position[2];
        int[] t = Arrays.stream(position[3]).mapToInt(d -> (int) Math.floor(d)).toArray();
        List<Fixture> fixtures = new java.util.ArrayList<>();

        for (int i = 0; i < x.length; i++) {
            if (t[i] != 0.0) {
                List<Tuple<Float, Float>> vertices;
                Tuple<Float, Float> center = computeFixtureCenterBase(x[i], y[i], t[i]);
                Tuple<Float, Float> pointA = computeCoordinateWithRotation((float) x[i], (float) y[i], (float) angle[i], center.getFirst(), center.getSecond());
                Tuple<Float, Float> pointB = computeCoordinateWithRotation((float) x[i], (float) y[i] + FixtureDimension.getHeight(t[i]), (float) angle[i], center.getFirst(), center.getSecond());
                Tuple<Float, Float> pointC = computeCoordinateWithRotation((float) x[i] + FixtureDimension.getWidth(t[i]), (float) y[i] + FixtureDimension.getHeight(t[i]), (float) angle[i], center.getFirst(), center.getSecond());
                Tuple<Float, Float> pointD = computeCoordinateWithRotation((float) x[i] + FixtureDimension.getWidth(t[i]), (float) y[i], (float) angle[i], center.getFirst(), center.getSecond());

                vertices = List.of(pointA, pointB, pointC, pointD);

                Float area = MomentOfInertia.computePolygonArea(vertices);
                Tuple<Float, Float> centroid = MomentOfInertia.computePolygonCentroid(vertices);
                Triple<Float, Float, Float> absoluteMomentOfInertia = MomentOfInertia.computeAbsoluteMomentOfInertia(vertices);
                Triple<Float, Float, Float> baricentricMomentsOfInertia = MomentOfInertia.computeBaricentricMomentsOfInertia(area, centroid, absoluteMomentOfInertia, (int) angle[i]);
                Fixture polygon = new Fixture(vertices, area, centroid, absoluteMomentOfInertia, baricentricMomentsOfInertia, (float)angle[i], t[i], center);
                fixtures.add(polygon);
            }
        }
        return fixtures;
    }

    public static List<Rectangle> getRectangleFromFixture(List<Fixture> fixtures, double[] types) {
        List<Rectangle> rectangles = new ArrayList<>();

        for (int i = 0; i < fixtures.size(); i++) {
            Polygon f = fixtures.get(i);
            Integer type = (int) types[i];
            Rectangle r = new Rectangle(f.getVertices(), f.getArea(), f.getCentroid(), f.getAbsoluteMomentOfInertia(),
                    f.getBaricentricMomentOfInertia(), f.getAngle(), (float) FixtureDimension.getWidth(type),
                    (float) FixtureDimension.getHeight(type));
            rectangles.add(r);
        }

        return rectangles;
    }


    public static Tuple<Float, Float> computeFixtureCenterBase(double x, double y, int type) {
        return new Tuple<>((float) (x + (double) FixtureDimension.getWidth(type) / 2), (float) (y + (double) FixtureDimension.getHeight(type) / 2));
    }
}
