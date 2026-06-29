package workpiece.coffeeTable;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import heuristics.ParticleSwarmOptimization;
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

public class MIPCoffeeTable {

    private static final String MIP_FILE_PATH = "./resources/mip/mip_model_coffee_table_gurobi.json";
    private static final String RESULT_FILE_PATH = "./resources/pso/pso_mip_coffee_table.json";

    public static void main(String[] args) {
        String content;
        try {
            content = new String(Files.readAllBytes(Paths.get(MIP_FILE_PATH)));
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
        double[][] seed = new double[][] { x, y, angle, t };

        int swarmSize = 1245;
        int maxIterations = 2977;
        float inertiaWeight =  0.4113008130530398f;
        float cognitiveCoefficient = 1.692120419692678f;
        float socialCoefficient = 1.8742489693157494f;

        ParticleSwarmOptimization pso = new ParticleSwarmOptimization(
                swarmSize,
                maxIterations,
                inertiaWeight,
                cognitiveCoefficient,
                socialCoefficient,
                geometryConstraint,
                noOverlapConstraint,
                fixtureConstraints,
                coffeeTable.getMinFix(),
                coffeeTable.getMaxFix()
        );

        pso.initialize(seed, bounds);
        long startTimePSO = System.currentTimeMillis();
        double[][] bestSolutionPSO = pso.optimize();
        long endTimePSO =  System.currentTimeMillis();
        System.out.println("Duration PSO (ms): " + (endTimePSO - startTimePSO));

        ConstraintsChecker constraintsChecker = new ConstraintsChecker(geometryConstraint, noOverlapConstraint, fixtureConstraints);
        MovementOperator movementOperator = new MovementOperator(5.0);
        RotationOperator rotationOperator = new RotationOperator(15.0);

        HillClimbing hillClimbing = new HillClimbing(constraintsChecker, movementOperator, rotationOperator, 1000);
        double[][] bestSolution = hillClimbing.optimize(bestSolutionPSO);

        List<Fixture> finalFixtures = FixtureUtility.getFixturesFromPosition(bestSolution);
        List<Rectangle> rectangles = FixtureUtility.getRectangleFromFixture(finalFixtures, bestSolution[Variable.T.getId()]);
        SaveResult.saveResult(RESULT_FILE_PATH, bestSolution, geometryConstraint, fixtureConstraints, noOverlapConstraint, rectangles);

    }



}
