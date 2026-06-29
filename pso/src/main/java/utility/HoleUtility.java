package utility;

import polygon.Rectangle;
import utility.data.Triple;
import utility.data.Tuple;

import java.util.List;

public class HoleUtility {

    public static Rectangle getPolygonFromPosition(Float x, Float y, Float width, Float height) {
        List<Tuple<Float, Float>> vertices;
        Tuple<Float, Float> pointA = new Tuple<>(x, y);
        Tuple<Float, Float> pointB = new Tuple<>(x, y + height);
        Tuple<Float, Float> pointC = new Tuple<>(x + width, y + height);
        Tuple<Float, Float> pointD = new Tuple<>(x + width, y);

        vertices = List.of(pointA, pointB, pointC, pointD);

        Float area = MomentOfInertia.computePolygonArea(vertices);
        Tuple<Float, Float> centroid = MomentOfInertia.computePolygonCentroid(vertices);
        Triple<Float, Float, Float> absoluteMomentOfInertia = MomentOfInertia.computeAbsoluteMomentOfInertia(vertices);
        Triple<Float, Float, Float> baricentricMomentsOfInertia = MomentOfInertia.computeBaricentricMomentsOfInertia(area, centroid, absoluteMomentOfInertia, 0);

        return new Rectangle(vertices, area, centroid, absoluteMomentOfInertia, baricentricMomentsOfInertia, 0.0f, width, height);
    }

}
