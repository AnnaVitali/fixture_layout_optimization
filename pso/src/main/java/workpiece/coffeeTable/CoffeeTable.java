package workpiece.coffeeTable;

import workpiece.Workpiece;
import polygon.Rectangle;
import utility.PolygonGeometry;
import utility.data.Triple;
import utility.data.Tuple;

import java.util.*;

public class CoffeeTable implements Workpiece {

    private final Set<Tuple<Float, Float>> vertices;
    private final List<Triple<Integer, Integer, Integer>> inequalities;
    private final List<Rectangle> holes;

    public CoffeeTable() {
        this.vertices = new HashSet<>();
        this.inequalities = new ArrayList<>();
        this.holes = Collections.emptyList();

        this.defineWorkpieceInequalityAndVertices();
    }

    private void defineWorkpieceInequalityAndVertices() {
        List<PolygonGeometry.LineSegment> lineEquations = new ArrayList<>();
        PolygonGeometry.LineSegment line1 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(350.0, 700.0), new Tuple<>(850.0, 700.0));
        lineEquations.add(line1);

        List<PolygonGeometry.LineSegment> curve1 = PolygonGeometry.approximateArcWithLines(new Tuple<>(350.0, 350.0), 350.0, new Tuple<>(350.0, 700.0), new Tuple<>(350.0, 0.0), false, 0.5);
        lineEquations.addAll(curve1);

        List<PolygonGeometry.LineSegment> curve2 = PolygonGeometry.approximateArcWithLines(new Tuple<>(850.0, 350.0), 350.0, new Tuple<>(850.0, 0.0), new Tuple<>(850.0, 700.0), false, 0.5);
        lineEquations.addAll(curve2);

        PolygonGeometry.LineSegment line2 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(350.0, 0.0), new Tuple<>(850.0, 0.0));
        lineEquations.add(line2);

        Tuple<Float, Float> center = PolygonGeometry.computePolygonCentroid(List.of(new Tuple<>(0.0f, 0.0f),
                new Tuple<>(0.0f, 700.0f), new Tuple<>(1200.0f, 700.0f), new Tuple<>(1200.0f, 0.0f)));

        for (PolygonGeometry.LineSegment line : lineEquations) {
            inequalities.add(PolygonGeometry.computeInequalitiesCoefficients(center.getFirst(), center.getSecond(), line));
            vertices.add(new Tuple<>(line.getP1().getFirst().floatValue(), line.getP1().getSecond().floatValue()));
            vertices.add(new Tuple<>(line.getP2().getFirst().floatValue(), line.getP2().getSecond().floatValue()));
        }
    }


    @Override
    public Set<Tuple<Float, Float>> getVertices() {
        return this.vertices;
    }

    @Override
    public List<Triple<Integer, Integer, Integer>> getInequalities() {
        return this.inequalities;
    }

    @Override
    public List<Rectangle> getHoles() {
        return this.holes;
    }

    @Override
    public int getMinFix() {
        return 6;
    }

    @Override
    public int getMaxFix() {
        return 10;
    }
}
