package workpiece.spiralStairStep;

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

public class SpiralStairStep implements Workpiece {
    private static List<Tuple<Float, Float>> HOLES = List.of(
            new Tuple<>(31f, 165f),
            new Tuple<>(642f, 342f),
            new Tuple<>(663f, 186f),
            new Tuple<>(642f, 31f)
    );
    private static List<Float> HOLESWIDTH = List.of(
            67f, 24f, 24f, 24f
    );

    private static List<Float> HOLESHEIGHT = List.of(
            67f, 24f, 24f, 24f
    );

    private Set<Tuple<Float, Float>> vertices;
    private List<Triple<Integer, Integer, Integer>> inequalities;
    private List<Rectangle> holes;

    public SpiralStairStep() {
        this.vertices = new HashSet<>();
        this.inequalities = new ArrayList<>();
        this.holes = new ArrayList<>();

        this.defineWorkpieceInequalityAndVertices();
        this.defineHoles();
    }

    private void defineWorkpieceInequalityAndVertices(){
        List<PolygonGeometry.LineSegment> lineEquations = new ArrayList<>();

        PolygonGeometry.LineSegment line1 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(51.40, 135.33), new Tuple<>(665.84, 3.89));
        lineEquations.add(line1);

        List<PolygonGeometry.LineSegment> curve1 = PolygonGeometry.approximateArcWithLines(new Tuple<>(670.08, 23.44), 20.0, new Tuple<>(665.84, 3.89), new Tuple<>(689.28, 17.87), false, 0.5);
        lineEquations.addAll(curve1);

        List<PolygonGeometry.LineSegment> curve2 = PolygonGeometry.approximateArcWithLines(new Tuple<>(65.08, 199.0), 650.0, new Tuple<>(689.28, 17.87), new Tuple<>(689.27, 379.95), false, 0.5);
        lineEquations.addAll(curve2);

        List<PolygonGeometry.LineSegment> curve3 = PolygonGeometry.approximateArcWithLines(new Tuple<>(670.06, 374.38), 20.0, new Tuple<>(689.27, 379.95), new Tuple<>(665.88, 393.94), false, 0.5);
        lineEquations.addAll(curve3);

        PolygonGeometry.LineSegment line2 = PolygonGeometry.lineEquationFromPoints(new Tuple<>(665.88, 393.94), new Tuple<>(51.40, 262.45));
        lineEquations.add(line2);

        List<PolygonGeometry.LineSegment> curve4 = PolygonGeometry.approximateArcWithLines(new Tuple<>(65.0, 198.89), 65.0, new Tuple<>(51.40, 262.45), new Tuple<>(51.40, 135.33), false, 0.5);
        lineEquations.addAll(curve4);

        Tuple<Float, Float> center = PolygonGeometry.computePolygonCentroid(List.of(new Tuple<>(51.40f, 135.33f), new Tuple<>(665.84f, 3.89f), new Tuple<>(689.28f, 17.87f), new Tuple<>(689.27f, 379.95f), new Tuple<>(665.88f, 393.94f), new Tuple<>(51.40f, 262.45f)));

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
