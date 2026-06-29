package heuristics.hillClimbing.operator;

import polygon.Fixture;
import utility.FixtureUtility;
import utility.HeuristicsUtility;
import utility.MomentOfInertia;
import utility.Variable;
import utility.data.Tuple;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MovementOperator implements Operator {

    private static final double EPSILON = 1e-6;
    private static final double BAR_CENTER_TOLERANCE = 2.0;
    private final double stepSize;
    private final Random rnd = new Random();

    public MovementOperator(double stepSize) {
        this.stepSize = stepSize;
    }

    @Override
    public double[][] propose(double[][] current) {
        double[][] newPosition = HeuristicsUtility.copySolution(current);
        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(current);
        List<Integer> fixtureIndices = buildFixtureIndices(current);

        int targetPosition = selectTargetPosition(fixtures, fixtureIndices, MomentOfInertia.computeOverallCenterOfGravity(fixtures));
        Fixture targetFixture = fixtures.get(targetPosition);
        int targetIndex = fixtureIndices.get(targetPosition);
        Tuple<Float, Float> centerOfGravity = MomentOfInertia.computeOverallCenterOfGravity(fixtures);
        boolean horizontalMove = Math.random() < 0.5;

        if (horizontalMove) {
            double deltaX = moveAwayFromCenter(targetFixture.getBaseCenter().getFirst(), centerOfGravity.getFirst());
            moveWholeBar(newPosition, targetFixture.getBaseCenter().getFirst(), deltaX, current);
        } else {
            double deltaY = moveAwayFromCenter(targetFixture.getBaseCenter().getSecond(), centerOfGravity.getSecond());
            double oldY = newPosition[Variable.Y.getId()][targetIndex];
            newPosition[Variable.Y.getId()][targetIndex] += deltaY;
            double newY = newPosition[Variable.Y.getId()][targetIndex];
            //System.out.println("MovementOperator: fixture " + targetIndex + " Y " + oldY + " -> " + newY);
        }

        return newPosition;
    }

    private List<Integer> buildFixtureIndices(double[][] current) {
        List<Integer> fixtureIndices = new ArrayList<>();
        double[] x = current[Variable.X.getId()];
        double[] t = current[Variable.T.getId()];

        for (int i = 0; i < x.length; i++) {
            int type = (int) Math.floor(t[i]);
            if (type <= 0) {
                continue;
            }

            fixtureIndices.add(i);
        }

        return fixtureIndices;
    }

    private int selectTargetPosition(List<Fixture> fixtures, List<Integer> fixtureIndices, Tuple<Float, Float> centerOfGravity) {
        int bestPosition = 0;
        double bestScore = score(fixtures.get(0), centerOfGravity);

        for (int i = 1; i < fixtures.size(); i++) {
            double score = score(fixtures.get(i), centerOfGravity);
            if (score > bestScore) {
                bestScore = score;
                bestPosition = i;
            }
        }

        return bestPosition;
    }

    private double score(Fixture candidate, Tuple<Float, Float> centerOfGravity) {
        double dx = candidate.getBaseCenter().getFirst() - centerOfGravity.getFirst();
        double dy = candidate.getBaseCenter().getSecond() - centerOfGravity.getSecond();
        return Math.hypot(dx, dy);
    }

    private double moveAwayFromCenter(double coordinate, double center) {
        double direction = coordinate >= center ? 1.0 : -1.0;
        if (Math.abs(coordinate - center) < EPSILON) {
            direction = rnd.nextBoolean() ? -1.0 : 1.0;
        }
        return direction * (rnd.nextDouble() * stepSize);
    }

    private void moveWholeBar(double[][] newPosition, double barCenterX, double deltaX, double[][] current) {
        double[] t = current[Variable.T.getId()];
        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(current);

        for (int i = 0; i < fixtures.size(); i++) {
            int type = (int) Math.floor(t[i]);
            if (type <= 0) {
                continue;
            }

            Tuple<Float, Float> barCenter = fixtures.get(i).getBaseCenter();
            if (Math.abs(barCenter.getFirst() - barCenterX) <= BAR_CENTER_TOLERANCE) {
                double oldX = current[Variable.X.getId()][i];
                newPosition[Variable.X.getId()][i] += deltaX;
                double newX = newPosition[Variable.X.getId()][i];
                //System.out.println("MovementOperator: fixture " + i + " X " + oldX + " -> " + newX);
            }
        }
    }
}
