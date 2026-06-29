package utility;

import utility.data.Triple;
import utility.data.Tuple;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;



public class PolygonGeometry {

    public static class LineSegment {
        private Triple<Integer, Integer, Integer> coeffs;
        private Tuple<Double, Double> p1;
        private Tuple<Double, Double> p2;

        public LineSegment(Triple<Integer, Integer, Integer> coeffs, Tuple<Double, Double> p1, Tuple<Double, Double> p2) {
            this.coeffs = coeffs;
            this.p1 = p1;
            this.p2 = p2;
        }

        public Triple<Integer, Integer, Integer> getCoeffs() {
            return coeffs;
        }

        public Tuple<Double, Double> getP1() {
            return p1;
        }

        public Tuple<Double, Double> getP2() {
            return p2;
        }
    }

    public static LineSegment lineEquationFromPoints(Tuple<Double, Double> p1, Tuple<Double, Double> p2) {
        double x1 = p1.getFirst();
        double y1 = p1.getSecond();
        double x2 = p2.getFirst();
        double y2 = p2.getSecond();

        double a = y1 - y2;
        double b = x2 - x1;
        double c = x1 * y2 - x2 * y1;

        return new LineSegment(new Triple<>(Math.toIntExact(Math.round(a)), Math.toIntExact(Math.round(b)), Math.toIntExact(Math.round(c))), p1, p2);
    }

    public static List<LineSegment> approximateArcWithLines(Tuple<Double, Double> center, double radius,
                                                            Tuple<Double, Double> start, Tuple<Double, Double> end,
                                                            boolean clockwise, double tol) {
        double xc = center.getFirst();
        double yc = center.getSecond();

        double thetaS = Math.atan2(start.getSecond() - yc, start.getFirst() - xc);
        double thetaE = Math.atan2(end.getSecond() - yc, end.getFirst() - xc);

        if (thetaS < 0) thetaS += 2 * Math.PI;
        if (thetaE < 0) thetaE += 2 * Math.PI;

        if (clockwise) {
            if (thetaE > thetaS) thetaE -= 2 * Math.PI;
        } else {
            if (thetaE < thetaS) thetaE += 2 * Math.PI;
        }

        Function<Double, double[]> f = (t) -> new double[] { xc + radius * Math.cos(t), yc + radius * Math.sin(t) };

        return approximateCurveWithLines(f, thetaS, thetaE, tol);
    }

    public static Float computePolygonArea(List<Tuple<Float, Float>> points) {
        int n = points.size();
        double A = 0.0;

        for (int i = 0; i < n; i++) {
            double x0 = points.get(i).getFirst();
            double y0 = points.get(i).getSecond();
            double x1 = points.get((i + 1) % n).getFirst();
            double y1 = points.get((i + 1) % n).getSecond();

            A += x0 * y1 - x1 * y0;
        }

        A *= 0.5;
        return Math.abs((float)A);
    }

    public static Tuple<Float, Float> computePolygonCentroid(List<Tuple<Float, Float>> points) {
        int n = points.size();
        double A = 0.0;
        double Cx = 0.0;
        double Cy = 0.0;

        for (int i = 0; i < n; i++) {
            double x0 = points.get(i).getFirst();
            double y0 = points.get(i).getSecond();
            double x1 = points.get((i + 1) % n).getFirst();
            double y1 = points.get((i + 1) % n).getSecond();

            double cross = x0 * y1 - x1 * y0;
            A += cross;
            Cx += (x0 + x1) * cross;
            Cy += (y0 + y1) * cross;
        }

        A *= 0.5;
        if (Math.abs(A) < 1e-12) {
            throw new IllegalArgumentException("Polygon area is zero or too small to compute centroid.");
        }
        Cx /= (6.0 * A);
        Cy /= (6.0 * A);

        return new Tuple<>(Math.abs((float)Cx), Math.abs((float)Cy));
    }

    public static Triple<Integer, Integer, Integer> computeInequalitiesCoefficients(double centerX, double centerY, LineSegment line) {
        int a = line.getCoeffs().getFirst();
        int b = line.getCoeffs().getSecond();
        int c = line.getCoeffs().getThird();

        double v = a * centerX + b * centerY + c;
        if (v <= 0) {
            return new Triple<>(a, b, c);
        } else {
            return new Triple<>(-a, -b, -c);
        }
    }

    private static List<LineSegment> approximateCurveWithLines(Function<Double, double[]> f, double t0, double t1, double tol) {
        List<double[]> polygon = subdivide(f, t0, t1, tol);
        List<LineSegment> result = new ArrayList<>();

        for (int i = 0; i < polygon.size() - 1; i++) {
            double[] p1 = polygon.get(i);
            double[] p2 = polygon.get(i + 1);

            double a = p1[1] - p2[1];
            double b = p2[0] - p1[0];
            double c = p1[0] * p2[1] - p2[0] * p1[1];

            Triple<Integer, Integer, Integer> coeffs =
                    new Triple<>(Math.toIntExact(Math.round(a)), Math.toIntExact(Math.round(b)), Math.toIntExact(Math.round(c)));

            result.add(new LineSegment(coeffs, new Tuple<>(p1[0], p1[1]), new Tuple<>(p2[0], p2[1])));
        }

        return result;
    }

    private static List<double[]> subdivide(Function<Double, double[]> f, double t0, double t1, double tol) {
        double[] p0 = f.apply(t0);
        double[] p1 = f.apply(t1);
        double tm = 0.5 * (t0 + t1);
        double[] pm = f.apply(tm);

        double[] p1mp0 = new double[] { p1[0] - p0[0], p1[1] - p0[1] };
        double[] pmp0 = new double[] { pm[0] - p0[0], pm[1] - p0[1] };

        double cross = Math.abs(p1mp0[0] * pmp0[1] - p1mp0[1] * pmp0[0]);
        double denom = Math.hypot(p1mp0[0], p1mp0[1]);
        double d = denom == 0.0 ? 0.0 : cross / denom;

        List<double[]> left;
        List<double[]> right;
        if (d > tol) {
            left = subdivide(f, t0, tm, tol);
            right = subdivide(f, tm, t1, tol);
            // merge without duplicating the middle point
            List<double[]> merged = new ArrayList<>(left);
            if (!right.isEmpty()) {
                merged.remove(merged.size() - 1);
                merged.addAll(right);
            }
            return merged;
        } else {
            List<double[]> pair = new ArrayList<>();
            pair.add(p0);
            pair.add(p1);
            return pair;
        }
    }

}


