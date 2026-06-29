package workpiece.simpleStairStep;

import workpiece.Workpiece;
import polygon.Rectangle;
import utility.HoleUtility;
import utility.PolygonGeometry;
import utility.data.Triple;
import utility.data.Tuple;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SimpleStairStep implements Workpiece {

    private static List<Tuple<Float, Float>> HOLES = List.of(
            new Tuple<>(799f,  219f),
            new Tuple<>(799f, 19f)
    );
    private static List<Float> HOLESWIDTH = List.of(
            42f, 42f
    );
    private static List<Float> HOLESHEIGHT = List.of(
            42f, 42f
    );

    private Set<Tuple<Float, Float>> vertices;
    private List<Triple<Integer, Integer, Integer>> inequalities;
    private List<Rectangle> holes;

    public SimpleStairStep() {
        this.vertices = new HashSet<>();
        this.inequalities = new ArrayList<>();
        this.holes = new ArrayList<>();

        this.defineWorkpieceInequalityAndVertices();
        this.defineHoles();
    }

    private void defineWorkpieceInequalityAndVertices(){
        List<PolygonGeometry.LineSegment> lineEquations = new ArrayList<>();

        PolygonGeometry.LineSegment line1 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(0.0, 0.0), new Tuple<>(0.0, 280.0));
        lineEquations.add(line1);

        PolygonGeometry.LineSegment line2 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(0.0, 280.0), new Tuple<>(900.0, 280.0));
        lineEquations.add(line2);

        PolygonGeometry.LineSegment line3 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(900.0, 280.0), new Tuple<>(900.0, 0.0));
        lineEquations.add(line3);

        PolygonGeometry.LineSegment line4 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(900.0, 0.0), new Tuple<>(0.0, 0.0));
        lineEquations.add(line4);

        Tuple<Float, Float> center = PolygonGeometry.computePolygonCentroid(List.of(
                new Tuple<>(0.0f, 0.0f),
                new Tuple<>(0.0f, 280.0f),
                new Tuple<>(900.0f, 280.0f),
                new Tuple<>(900.0f, 0.0f)
        ));

        for (PolygonGeometry.LineSegment line : lineEquations) {
            this.inequalities.add(PolygonGeometry.computeInequalitiesCoefficients(center.getFirst(), center.getSecond(), line));
            this.vertices.add(new Tuple<>(line.getP1().getFirst().floatValue(), line.getP1().getSecond().floatValue()));
            this.vertices.add(new Tuple<>(line.getP2().getFirst().floatValue(), line.getP2().getSecond().floatValue()));
        }
    }

    private void defineHoles(){
        for(int j = 0; j < HOLES.size(); j++){
            holes.add(HoleUtility.getPolygonFromPosition(HOLES.get(j).getFirst(), HOLES.get(j).getSecond(), HOLESWIDTH.get(j), HOLESHEIGHT.get(j)));
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
        return 3;
    }

    @Override
    public int getMaxFix() {
        return 6;
    }
}
