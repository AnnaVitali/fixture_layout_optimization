package utility;

import polygon.Fixture;
import polygon.Polygon;
import utility.data.Triple;
import utility.data.Tuple;

import java.util.List;

public class MomentOfInertia {

    public static float computePolygonArea(List<Tuple<Float, Float>> vertices){
        int n = vertices.size();
        float area = 0.0f;
        for (int i = 0; i < n; i++) {
            float x0 = vertices.get(i).getFirst();
            float y0 = vertices.get(i).getSecond();
            float x1 = vertices.get((i + 1) % n).getFirst();
            float y1 = vertices.get((i + 1) % n).getSecond();
            area += (x0 * y1) - (y0 * x1);
        }
        return 0.5f * Math.abs(area);
    }

    public static Tuple<Float, Float> computePolygonCentroid(List<Tuple<Float, Float>> vertices) {
        int n = vertices.size();
        float area = computePolygonArea(vertices);
        float cx = 0.0f;
        float cy = 0.0f;

        for (int i = 0; i < n; i++) {
            float x0 = vertices.get(i).getFirst();
            float y0 = vertices.get(i).getSecond();
            float x1 = vertices.get((i + 1) % n).getFirst();
            float y1 = vertices.get((i + 1) % n).getSecond();
            float a = (x0 * y1) - (x1 * y0);
            cx += (x0 + x1) * a;
            cy += (y0 + y1) * a;
        }

        cx /= (6.0f * area);
        cy /= (6.0f * area);

        return new Tuple<>(Math.abs(cx), Math.abs(cy));
    }

    public static Triple<Float, Float, Float> computeAbsoluteMomentOfInertia(List<Tuple<Float, Float>> vertices) {
        int n = vertices.size();
        float jx = 0.0f;
        float jy = 0.0f;
        float jxy = 0.0f;

        for (int i = 0; i < n; i++) {
            float x0 = vertices.get(i).getFirst();
            float y0 = vertices.get(i).getSecond();
            float x1 = vertices.get((i + 1) % n).getFirst();
            float y1 = vertices.get((i + 1) % n).getSecond();
            float a = (x0 * y1) - (x1 * y0);

            jx += (float)(Math.pow(y0, 2) + y0 * y1 + Math.pow(y1, 2)) * a;
            jy += (float)(Math.pow(x0, 2) + x0 * x1 + Math.pow(x1, 2)) * a;
            jxy += (x0 * y1 + 2 * x0 * y0 + 2 * x1 * y1 + x1 * y0) * a;
        }

        jx = Math.abs(jx) /  12.0f;
        jy = Math.abs(jy) / 12.0f;
        jxy = Math.abs(jxy) / 24.0f;

        return new Triple<>(jx, jy, jxy);
    }

    public static Triple<Float, Float, Float> computeBaricentricMomentsOfInertia(Float area, Tuple<Float, Float> centroid, Triple<Float, Float, Float> absoluteMomentOfInertia, int angle){
        Float cosAngle = (float)Math.cos(Math.toRadians(angle));
        Float sinAngle = (float)Math.sin(Math.toRadians(angle));
        
        Float x_g = centroid.getFirst();
        Float y_g = centroid.getSecond();
        Float jx = absoluteMomentOfInertia.getFirst();
        Float jy = absoluteMomentOfInertia.getSecond();
        Float jxy = absoluteMomentOfInertia.getThird();

        Float i_prime = jx - y_g * y_g * area;
        Float j_prime = jy - x_g * x_g * area;
        Float ij_prime = jxy - y_g * x_g * area;

        Float i = i_prime * cosAngle * cosAngle + j_prime * sinAngle * sinAngle - ij_prime * 2 * sinAngle * cosAngle;
        Float j = i_prime * sinAngle * sinAngle + j_prime * cosAngle * cosAngle + ij_prime * 2 * sinAngle * cosAngle;
        Float ij = (i_prime - j_prime) * sinAngle * cosAngle + ij_prime * (cosAngle * cosAngle - sinAngle * sinAngle);

        return new Triple<>(i, j, ij);

    }

    public static Tuple<Float, Float> computeOverallCenterOfGravity(List<Fixture> polygons) {
        float totalArea = 0.0f;
        float xSum = 0.0f;
        float ySum = 0.0f;

        for (Polygon polygon : polygons) {
            float area = polygon.getArea();
            Tuple<Float, Float> centroid = polygon.getCentroid();
            totalArea += area;
            xSum += centroid.getFirst() * area;
            ySum += centroid.getSecond() * area;
        }

        if (totalArea == 0) {
            return new Tuple<>(0.0f, 0.0f);
        }

        return new Tuple<>(xSum / totalArea, ySum / totalArea);
    }

    public static Triple<Float, Float, Float> computeCombinedAbsoluteMomentsOfInertia(List<Polygon> polygons){
        float totalJx = 0.0f;
        float totalJy = 0.0f;
        float totalJxy = 0.0f;

        for (Polygon polygon : polygons) {

            Triple<Float, Float, Float> absoluteMomentOfInertia = polygon.getAbsoluteMomentOfInertia();

            float jx = absoluteMomentOfInertia.getFirst();
            float jy = absoluteMomentOfInertia.getSecond();
            float jxy = absoluteMomentOfInertia.getThird();

            totalJx += jx;
            totalJy += jy;
            totalJxy += jxy;
        }

        return new Triple<>(totalJx, totalJy, totalJxy);
    }

    public static Tuple<Float, Float> computeCombinedBaricentricMomentsOfInertia(List<Fixture> polygons, Tuple<Float, Float> overallCentroid){
        float xG = overallCentroid.getFirst();
        float yG = overallCentroid.getSecond();
        float totalI = 0.0f;
        float totalJ = 0.0f;
        float totalIJ = 0.0f;
        float areaTotal = 0.0f;

        float jx;
        float jy;
        float jxy;

        float i;
        float j;


        for (Polygon polygon : polygons) {
            float jxPoly = polygon.getAbsoluteMomentOfInertia().getFirst();
            float jyPoly = polygon.getAbsoluteMomentOfInertia().getSecond();
            float jxyPoly = polygon.getAbsoluteMomentOfInertia().getThird();
            float area = polygon.getArea();

            totalI += jxPoly;
            totalJ += jyPoly;
            totalIJ += jxyPoly;
            areaTotal += area;

        }

        jx = (float) (totalI - Math.pow(yG, 2) * areaTotal);
        jy = (float) (totalJ - Math.pow(xG, 2) * areaTotal);
        jxy = totalIJ - xG * yG * areaTotal;

        double a = Math.pow((jx - jy) / 2, 2) + 4 * Math.pow(jxy, 2);

        i = (float) ((jx + jy) / 2 - Math.sqrt(a));
        j = (float) ((jx + jy) / 2 + Math.sqrt(a));

        return new Tuple<>(i, j);
    }
}
