package workpiece.coffeeTable;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import heuristics.ParticleSwarmOptimization;
import heuristics.SparrowSearchAlgorithm;
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
import java.util.*;

public class SSACoffeeTable {

    private static final String CP_FILE_PATH = "./resources/mip/mip_model_coffee_table_gurobi.json";
    private static final String RESULT_FILE_PATH = "./resources/ssa/ssa_cp_coffee_table.json";

    public static void main(String[] args) {
        String content;
        try {
            content = new String(Files.readAllBytes(Paths.get(CP_FILE_PATH)));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        CoffeeTable coffeeTable = new CoffeeTable();

        NoOverlapConstraint noOverlapConstraint = new NoOverlapConstraint(coffeeTable.getHoles());
        GeometryConstraint geometryConstraint = new GeometryConstraint(coffeeTable.getInequalities(), coffeeTable.getVertices().stream().toList(),
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
        double[][] seed = new double[][]{x, y, angle, t};

        int maxIterations = 3000;
        int numberOfSparrows = 1200;
        int numberOfProducer = 240;

        SparrowSearchAlgorithm ssa = new SparrowSearchAlgorithm(
                maxIterations,
                numberOfSparrows,
                numberOfProducer,
                geometryConstraint,
                noOverlapConstraint,
                fixtureConstraints,
                coffeeTable.getMinFix(),
                coffeeTable.getMaxFix()
        );

        ssa.initialize(seed, bounds);

        double[][] bestSolution = ssa.optimize();

        List<Fixture> finalFixtures = FixtureUtility.getFixturesFromPosition(bestSolution);
        List<Rectangle> rectangles = FixtureUtility.getRectangleFromFixture(finalFixtures, bestSolution[Variable.T.getId()]);
        SaveResult.saveResult(RESULT_FILE_PATH, bestSolution, geometryConstraint, fixtureConstraints, noOverlapConstraint, rectangles);
    }
}
