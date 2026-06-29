//package utility;
//
//public class SeparationAxisTheorem {
//
//    public static boolean checkOverlap(Rectangle rectangle1, Rectangle rectangle2) {
//        float angle1 = rectangle1.getAngle();
//        float angle2 = rectangle2.getAngle();
//
//        float c1x = rectangle1.getCentroid().getFirst();
//        float c1y = rectangle1.getCentroid().getSecond();
//        float c2x = rectangle2.getCentroid().getFirst();
//        float c2y = rectangle2.getCentroid().getSecond();
//
//        float hx1 = rectangle1.getWidth() / 2f;
//        float hy1 = rectangle1.getHeight() / 2f;
//        float hx2 = rectangle2.getWidth() / 2f;
//        float hy2 = rectangle2.getHeight() / 2f;
//
//        float cos1 = (float) Math.cos(Math.toRadians(angle1));
//        float sin1 = (float) Math.sin(Math.toRadians(angle1));
//        float cos2 = (float) Math.cos(Math.toRadians(angle2));
//        float sin2 = (float) Math.sin(Math.toRadians(angle2));
//
//        // rectangle 1 local axes (unit)
//        float ux1 = cos1, uy1 = sin1;         // local x axis (width direction)
//        float vx1 = -sin1, vy1 = cos1;        // local y axis (height direction)
//
//        // rectangle 2 local axes (unit)
//        float ux2 = cos2, uy2 = sin2;
//        float vx2 = -sin2, vy2 = cos2;
//
//        // the four candidate separating axes: ux1, vx1, ux2, vx2
//        float[][] axes = {
//                { ux1, uy1 },
//                { vx1, vy1 },
//                { ux2, uy2 },
//                { vx2, vy2 }
//        };
//
//        for (float[] axis : axes) {
//            float ax = axis[0];
//            float ay = axis[1];
//
//            // projection of centers onto axis
//            float projC1 = dot(ax, ay, c1x, c1y);
//            float projC2 = dot(ax, ay, c2x, c2y);
//            float centerDist = Math.abs(projC2 - projC1);
//
//            // projection radius of rectangle 1 onto axis
//            float r1 = hx1 * Math.abs(dot(ax, ay, ux1, uy1)) + hy1 * Math.abs(dot(ax, ay, vx1, vy1));
//            // projection radius of rectangle 2 onto axis
//            float r2 = hx2 * Math.abs(dot(ax, ay, ux2, uy2)) + hy2 * Math.abs(dot(ax, ay, vx2, vy2));
//
//            if (centerDist > r1 + r2) {
//                return false; // separating axis found -> no overlap
//            }
//        }
//
//        return true; // no separating axis -> overlap
//    }
//
//    private static float dot(float ax, float ay, float bx, float by) {
//        return ax * bx + ay * by;
//    }
//}

// java
package constraint;

import polygon.Rectangle;
import utility.data.Tuple;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.List;

public class SeparationAxisTheorem {

    public static boolean checkOverlap(Rectangle rectangle1, Rectangle rectangle2) {
        if (rectangle1 == null || rectangle2 == null) return false;
        List<Point2D.Float> poly1 = toPointList(rectangle1);
        List<Point2D.Float> poly2 = toPointList(rectangle2);
        if (poly1.isEmpty() || poly2.isEmpty()) return false;
        return polygonsIntersect(poly1, poly2);
    }

    private static List<Point2D.Float> toPointList(Rectangle rect) {
        List<Point2D.Float> out = new ArrayList<>();
        List<Tuple<Float, Float>> verts = rect.getVertices();
        if (verts == null) return out;
        for (Tuple<Float, Float> t : verts) {
            if (t == null) continue;
            Float fx = t.getFirst();
            Float fy = t.getSecond();
            if (fx != null && fy != null) out.add(new Point2D.Float(fx, fy));
        }
        return out;
    }

    private static boolean polygonsIntersect(List<Point2D.Float> a, List<Point2D.Float> b) {
        if (a.size() < 3 || b.size() < 3) return false;
        if (!testAxes(a, b)) return false;
        return testAxes(b, a);
    }

    private static boolean testAxes(List<Point2D.Float> polyA, List<Point2D.Float> polyB) {
        int n = polyA.size();
        for (int i = 0; i < n; i++) {
            Point2D.Float p1 = polyA.get(i);
            Point2D.Float p2 = polyA.get((i + 1) % n);

            float ex = p2.x - p1.x;
            float ey = p2.y - p1.y;

            float ax = -ey;
            float ay = ex;

            double[] projA = projectPolygon(polyA, ax, ay);
            double[] projB = projectPolygon(polyB, ax, ay);

            if (projA[1] < projB[0] || projB[1] < projA[0]) {
                return false;
            }
        }
        return true;
    }

    private static double[] projectPolygon(List<Point2D.Float> poly, float ax, float ay) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (Point2D.Float p : poly) {
            double proj = p.x * ax + p.y * ay;
            if (proj < min) min = proj;
            if (proj > max) max = proj;
        }
        return new double[]{min, max};
    }

    private static float dot(float ax, float ay, float bx, float by) {
        return ax * bx + ay * by;
    }
}
