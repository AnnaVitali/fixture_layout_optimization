package workpiece.dashboard;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import heuristics.ParticleSwarmOptimization;
import heuristics.TeachingLearningBasedOptimization;
import heuristics.hillClimbing.HillClimbing;
import heuristics.hillClimbing.operator.ConstraintsChecker;
import heuristics.hillClimbing.operator.MovementOperator;
import heuristics.hillClimbing.operator.RotationOperator;

import org.json.JSONObject;
import polygon.Fixture;
import polygon.Rectangle;
import utility.FixtureUtility;
import utility.SaveResult;
import utility.data.Tuple;
import utility.parameter.MachineParameter;
import utility.Variable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class TLBODashboard {

    private static final String EO_FILE_PATH = "./resources/expert/expert_operator_dashboard.json";
    private static final String RESULT_FILE_PATH = "./resources/tlbo/tlbo_eo_dashboard.json";


    public static void main(String[] args) {
        String content;
        try {
            content = new String(Files.readAllBytes(Paths.get(EO_FILE_PATH)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Dashboard dashboard = new Dashboard();

        NoOverlapConstraint noOverlapConstraint = new NoOverlapConstraint(dashboard.getHoles());
        GeometryConstraint geometryConstraint = new GeometryConstraint(dashboard.getInequalities(), dashboard.getVertices().stream().toList(),
                MachineParameter.getWMin(), MachineParameter.getHmin());
        FixtureConstraints fixtureConstraints = new FixtureConstraints(MachineParameter.getFixtureAvailability());

        JSONObject initialSolution = new JSONObject(content);
        double[] x = initialSolution.getJSONArray("x").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();
        double[] y = initialSolution.getJSONArray("y").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();
        double[] angle = initialSolution.getJSONArray("angle").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();
        double[] t = initialSolution.getJSONArray("fixture_type").toList().stream().mapToDouble(o -> ((Number) o).doubleValue()).toArray();

        List<Tuple<Integer, Integer>> bounds = List.of(
                new Tuple<>(0, MachineParameter.getWTab()),
                new Tuple<>(0, MachineParameter.getHTab()),
                new Tuple<>(0, 360),
                new Tuple<>(0, MachineParameter.getNTypes())
        );
        double[][] seed = new double[][] { x, y, angle, t };

        int populationSize = 1245;
        int numberOfIterations = 3000;
        TeachingLearningBasedOptimization tlbo = new TeachingLearningBasedOptimization(
                populationSize,
                numberOfIterations,
                geometryConstraint,
                noOverlapConstraint,
                fixtureConstraints,
                dashboard.getMinFix(),
                dashboard.getMaxFix());

        tlbo.initialize(seed, bounds);
        double[][] bestSolution = tlbo.optimize();

        List<Fixture> finalFixtures = FixtureUtility.getFixturesFromPosition(bestSolution);
        List<Rectangle> rectangles = FixtureUtility.getRectangleFromFixture(finalFixtures, bestSolution[Variable.T.getId()]);
        SaveResult.saveResult(RESULT_FILE_PATH, bestSolution, geometryConstraint, fixtureConstraints, noOverlapConstraint, rectangles);

    }



}
