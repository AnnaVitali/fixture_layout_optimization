package workpiece.simpleStairStep;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import heuristics.hillClimbing.HillClimbing;
import heuristics.hillClimbing.operator.ConstraintsChecker;
import heuristics.hillClimbing.operator.MovementOperator;
import heuristics.hillClimbing.operator.RotationOperator;
import org.json.JSONObject;
import polygon.Fixture;
import polygon.Rectangle;
import utility.FixtureUtility;
import utility.SaveResult;
import utility.SavedSolutionConverter;
import utility.Variable;
import utility.parameter.MachineParameter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class HCSimpleStairStep {

    private static final String PSO_FILE_PATH = "./resources/pso/pso_eo_simple_stair_step.json";
    private static final String RESULT_FILE_PATH = "./resources/pso/hc_pso_cp_simple_stair_step.json";

    public static void main(String[] args) {
        String content;
        try {
            content = new String(Files.readAllBytes(Paths.get(PSO_FILE_PATH)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        SimpleStairStep simpleStairStep = new SimpleStairStep();

        NoOverlapConstraint noOverlapConstraint = new NoOverlapConstraint(simpleStairStep.getHoles());
        GeometryConstraint geometryConstraint = new GeometryConstraint(
                simpleStairStep.getInequalities(),
                simpleStairStep.getVertices().stream().toList(),
                MachineParameter.getWMin(),
                MachineParameter.getHmin());
        FixtureConstraints fixtureConstraints = new FixtureConstraints(MachineParameter.getFixtureAvailability());

        JSONObject initialSolution = new JSONObject(content);
    double[][] seed = SavedSolutionConverter.toHillClimbingSeed(initialSolution);

        ConstraintsChecker constraintsChecker = new ConstraintsChecker(geometryConstraint, noOverlapConstraint, fixtureConstraints);
        MovementOperator movementOperator = new MovementOperator(5.0);
        RotationOperator rotationOperator = new RotationOperator(15.0);

        HillClimbing hillClimbing = new HillClimbing(constraintsChecker, movementOperator, rotationOperator, 1000);
        long startTime = System.currentTimeMillis();
        double[][] bestSolution = hillClimbing.optimize(seed);
        long endTime = System.currentTimeMillis();
        System.out.println("Duration Hill Climbing (ms): " + (endTime - startTime));

        List<Fixture> finalFixtures = FixtureUtility.getFixturesFromPosition(bestSolution);
        List<Rectangle> rectangles = FixtureUtility.getRectangleFromFixture(finalFixtures, bestSolution[Variable.T.getId()]);
        SaveResult.saveResult(RESULT_FILE_PATH, bestSolution, geometryConstraint, fixtureConstraints, noOverlapConstraint, rectangles);
    }
}
