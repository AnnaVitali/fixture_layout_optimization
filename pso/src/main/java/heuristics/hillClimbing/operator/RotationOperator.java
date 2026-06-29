package heuristics.hillClimbing.operator;

import polygon.Fixture;
import utility.FixtureUtility;
import utility.HeuristicsUtility;
import utility.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RotationOperator implements Operator {

    private final double maxDeltaDegrees;
    private final Random rnd = new Random();

    public RotationOperator(double maxDeltaDegrees) {
        this.maxDeltaDegrees = maxDeltaDegrees;
    }

    @Override
    public double[][] propose(double[][] current) {
        double[][] newPosition = HeuristicsUtility.copySolution(current);
        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(current);

        if (fixtures.isEmpty()) {
            return newPosition;
        }

        List<Integer> fixtureIndices = buildFixtureIndices(current);
        int targetPosition = rnd.nextInt(fixtures.size());
        int targetIndex = fixtureIndices.get(targetPosition);

        double oldAngle = newPosition[Variable.ANGLE.getId()][targetIndex];
        double delta = (rnd.nextDouble() * 2.0 - 1.0) * maxDeltaDegrees;
        double newAngle = clampAngle(oldAngle + delta);
        newPosition[Variable.ANGLE.getId()][targetIndex] = newAngle;

        return newPosition;
    }

    private List<Integer> buildFixtureIndices(double[][] current) {
        List<Integer> fixtureIndices = new ArrayList<>();
        double[] t = current[Variable.T.getId()];

        for (int i = 0; i < t.length; i++) {
            int type = (int) Math.floor(t[i]);
            if (type <= 0) {
                continue;
            }

            fixtureIndices.add(i);
        }

        return fixtureIndices;
    }

    private double clampAngle(double a) {
        double res = a % 360.0;
        if (res < 0) res += 360.0;
        return res;
    }
}
