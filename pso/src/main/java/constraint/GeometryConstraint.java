package constraint;

import polygon.Fixture;
import utility.data.Triple;
import utility.data.Tuple;
import utility.FixtureDimension;
import utility.parameter.MachineParameter;

import java.util.ArrayList;
import java.util.List;

public class GeometryConstraint {

    private final List<Triple<Integer, Integer, Integer>> inequalities;
    private final List<Tuple<Float, Float>> workpieceVertices;
    private final int wMin;
    private final int hMin;

    public GeometryConstraint(List<Triple<Integer, Integer, Integer>> inequalities,
                              List<Tuple<Float, Float>> workpieceVertices, int wMin, int hMin) {
        this.inequalities = inequalities;
        this.workpieceVertices = workpieceVertices;
        this.wMin = wMin;
        this.hMin = hMin;
    }

    public int computePenaltyGeometryViolation(List<Tuple<Float, Float>> vertices) {
        int penalty = 0;
        float x1 = vertices.get(0).getFirst();
        float y1 = vertices.get(0).getSecond();
        float x2 = vertices.get(1).getFirst();
        float y2 = vertices.get(1).getSecond();
        float x3 = vertices.get(2).getFirst();
        float y3 = vertices.get(2).getSecond();
        float x4 = vertices.get(3).getFirst();
        float y4 = vertices.get(3).getSecond();

        for (Triple<Integer, Integer, Integer> line : inequalities) {
            int a = line.getFirst();
            int b = line.getSecond();
            int c = line.getThird();
            float val1 = a * x1 + b * y1 + c;
            float val2 = a * x2 + b * y2 + c;
            float val3 = a * x3 + b * y3 + c;
            float val4 = a * x4 + b * y4 + c;

            if (val1 > 0 || val2 > 0 || val3 > 0 || val4 > 0) {
//                System.out.println("Geometry violation detected for fixture with vertices: " +
//                        "(" + x1 + ", " + y1 + "), " +
//                        "(" + x2 + ", " + y2 + "), " +
//                        "(" + x3 + ", " + y3 + "), " +
//                        "(" + x4 + ", " + y4 + ")");
                penalty += 1;
            }
        }

        return penalty;
    }

    public int computePenaltySecurityDistanceViolation(List<Fixture> fixtures) {
        int penalty = 0;
        for (int i = 0; i < fixtures.size(); i++) {
            for (int j = i + 1; j < fixtures.size(); j++) {
                Fixture fixtureI = fixtures.get(i);
                Fixture fixtureJ = fixtures.get(j);
                int cx1 = fixtureI.getBaseCenter().getFirst().intValue();
                int cx2 = fixtureJ.getBaseCenter().getFirst().intValue();
                double barWidth = MachineParameter.getWBar();

                if (cx1 != cx2 && Math.abs(cx1 - cx2) > 1) {
                    if (Math.abs(cx1 - cx2) < wMin + barWidth) {
                        penalty = penalty + 1;
//                        System.out.println("cx1: " + cx1 + ", cx2: " + cx2);
//                        System.out.println((Math.abs(cx1 - cx2)) + " < " + (wMin + barWidth));
//                        System.out.println("x1: " + fixtureI.getVertices().get(0).getFirst() + ", y1: " + fixtureI.getVertices().get(0).getSecond());
//                        System.out.println("x2: " + fixtureJ.getVertices().get(0).getFirst() + ", y2: " + fixtureJ.getVertices().get(0).getSecond());
//                        System.out.println("Horizontal Security distance violation detected between fixture " +  (i+1) + " and fixture " + (j+1));

                    }
                }
                if (cx1 == cx2 || Math.abs(cx1 - cx2) == 1) {
                    double cy1 = fixtureI.getBaseCenter().getSecond();
                    double cy2 = fixtureJ.getBaseCenter().getSecond();
                    double heightI = FixtureDimension.getHeight(fixtureI.getType());
                    double heightJ = FixtureDimension.getHeight(fixtureJ.getType());
                    double yi = cy1 - heightI / 2;
                    double yj = cy2 - heightJ / 2;

                    if (!(yi + heightI + hMin <= yj || yj + heightJ + hMin <= yi)) {
                        penalty = penalty + 1;
//                        System.out.println("x1: " + fixtureI.getVertices().get(0).getFirst() + ", y1: " + fixtureI.getVertices().get(0).getSecond());
//                        System.out.println("x2: " + fixtureJ.getVertices().get(0).getFirst() + ", y2: " + fixtureJ.getVertices().get(0).getSecond());
//                        System.out.println("Vertical Security distance violation detected between fixture " + (i+1) + " and fixture " + (j+1));
                    }
                }
            }
        }
        return penalty;
    }

    public List<Tuple<Float, Float>> tryAdjustGeometryViolation(List<Tuple<Float, Float>> vertices) {
        List<Tuple<Float, Float>> adjustedVertices = new ArrayList<>();
        Tuple<Float, Float> displacement = null;
        for (Tuple<Float, Float> vertex : vertices) {
            if (!isPointInsideWorkpiece(vertex)) {
                Tuple<Float, Float> closestPoint = closestPointOnWorkpiece(vertex);
                displacement = new Tuple<>(closestPoint.getFirst() - vertex.getFirst(), closestPoint.getSecond() - vertex.getSecond());
                break;
            }
        }
        if (displacement != null) {
            for (Tuple<Float, Float> vertex : vertices) {
                adjustedVertices.add(new Tuple<>(vertex.getFirst() + displacement.getFirst(), vertex.getSecond() + displacement.getSecond()));
            }
        }
        return adjustedVertices;
    }

    private boolean isPointInsideWorkpiece(Tuple<Float, Float> point) {
        float x = point.getFirst();
        float y = point.getSecond();

        for (Triple<Integer, Integer, Integer> line : inequalities) {
            int a = line.getFirst();
            int b = line.getSecond();
            int c = line.getThird();
            float val = a * x + b * y + c;
            if (val > 0) {
                return false;
            }
        }
        return true;
    }

    private Tuple<Float, Float> closestPointOnWorkpiece(Tuple<Float, Float> point) {
        Tuple<Float, Float> closestPoint = null;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < workpieceVertices.size(); i++) {
            Tuple<Float, Float> p1 = workpieceVertices.get(i);
            Tuple<Float, Float> p2 = workpieceVertices.get((i + 1) % workpieceVertices.size());

            Tuple<Float, Float> projectedPoint = projectPointOnLineSegment(point, p1, p2);
            double dist = distance(point, projectedPoint);
            if (dist < minDistance) {
                minDistance = dist;
                closestPoint = projectedPoint;
            }
        }

        return closestPoint;
    }

    private Tuple<Float, Float> projectPointOnLineSegment(Tuple<Float, Float> p, Tuple<Float, Float> a, Tuple<Float, Float> b) {
        float ax = a.getFirst();
        float ay = a.getSecond();
        float bx = b.getFirst();
        float by = b.getSecond();
        float px = p.getFirst();
        float py = p.getSecond();

        float abx = bx - ax;
        float aby = by - ay;
        float apx = px - ax;
        float apy = py - ay;

        float abSquared = abx * abx + aby * aby;
        if (abSquared == 0) {
            return a; // a and b are the same point
        }

        float ap_ab = apx * abx + apy * aby;
        float t = ap_ab / abSquared;

        if (t < 0) {
            return a; // closest to point a
        } else if (t > 1) {
            return b; // closest to point b
        } else {
            return new Tuple<>(ax + t * abx, ay + t * aby); // projection falls on the segment
        }
    }

    private double distance(Tuple<Float, Float> p1, Tuple<Float, Float> p2) {
        return Math.sqrt(Math.pow(p1.getFirst() - p2.getFirst(), 2) + Math.pow(p1.getSecond() - p2.getSecond(), 2));
    }


}
