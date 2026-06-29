package heuristics;


import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import noromalization.Normalizer;
import polygon.Fixture;
import polygon.Rectangle;
import utility.FixtureUtility;
import utility.MomentOfInertia;
import utility.Variable;
import utility.data.Tuple;

import java.util.ArrayList;
import java.util.Random;


import java.util.List;
import java.util.Objects;

import static heuristics.ParticleSwarmOptimization.NUM_VARIABLES;

public class SparrowSearchAlgorithm {

    private int maxIterations;
    private int numberOfSparrow;
    private int numberOfProducers;
    private List<Sparrow> sparrows;
    private final GeometryConstraint geometryConstraint;
    private final NoOverlapConstraint noOverlapConstraint;
    private final FixtureConstraints fixtureConstraints;
    private final int minFix;
    private final int maxFix;
    private Normalizer normalizer;
    private List<Tuple<Integer, Integer>> bounds;
    private Sparrow globalBestSparrow;
    private Random random = new Random();
    private double[][] matrixL;

    public SparrowSearchAlgorithm(int maxIterations, int numberOfSparrow, int numberOfProducers,
                                  GeometryConstraint geometryConstraint, NoOverlapConstraint noOverlapConstraint, FixtureConstraints fixtureConstraints,
                                  int minFix, int maxFix) {
        this.maxIterations = maxIterations;
        this.numberOfSparrow = numberOfSparrow;
        this.numberOfProducers = numberOfProducers;
        this.geometryConstraint = geometryConstraint;
        this.noOverlapConstraint = noOverlapConstraint;
        this.fixtureConstraints = fixtureConstraints;
        this.minFix = minFix;
        this.maxFix = maxFix;
        this.sparrows = new ArrayList<>(numberOfSparrow);

        this.matrixL = new double[1][this.maxFix];
        for (int f = 0; f < this.maxFix; f++) {
            matrixL[0][f] = 1.0;
        }
    }


    static class Sparrow {
        double[][] position;
        EvaluationResult evaluationResult;
    }

    private static class EvaluationResult {
        double objective;
        int violation;
    }

    public void initialize(double[][] initialSolution, List<Tuple<Integer, Integer>> bounds) {
        Objects.requireNonNull(bounds, "bounds must not be null");
        this.normalizer = new Normalizer(bounds);
        this.bounds = bounds;
        double delta = 10.0;

        double[][] initialNorm = null;
        double[][] initialDenorm = null;

        if (initialSolution != null) {
            initialSolution = setupInitialSolution(initialSolution);
            initialNorm = this.normalizer.normalize(initialSolution);
        }

        initialDenorm = this.normalizer.denormalize(initialNorm);
        adjustDenormalizedSolution(initialDenorm);

        EvaluationResult initialEvaluation = evaluateSolution(initialDenorm);

        System.out.println("Initial objective: " + initialEvaluation.objective + ", violation: " + initialEvaluation.violation);

        for (int i = 0; i < this.numberOfSparrow; i++) {
            double pos[][] = new double[NUM_VARIABLES][this.maxFix];

            if (i == 0) {
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        pos[v][f] = initialNorm[v][f];
                    }
                }
            } else {
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        double base = initialNorm[v][f];
                        double offset = Math.random() * (delta / Math.max(1, bounds.get(v).getSecond() - bounds.get(v).getFirst()));
                        pos[v][f] = Math.max(0.0, Math.min(1.0, base + offset));
                    }
                }
            }

            Sparrow sparrow = new Sparrow();
            sparrow.position = pos;
            double[][] pDenorm = this.normalizer.denormalize(sparrow.position);//denormalizeLocal(particle.position, this.bounds);
            adjustDenormalizedSolution(pDenorm);
            EvaluationResult eval = evaluateSolution(pDenorm);
            sparrow.evaluationResult = eval;
            this.sparrows.add(sparrow);
        }
    }

    public double[][] optimize() {
        double[][][] individualBestPosition = new double[this.numberOfSparrow][][];
        double[] individualBestObjective = new double[this.numberOfSparrow];
        int[] individualBestViolation = new int[this.numberOfSparrow];

        initializeIndividualBest(individualBestPosition, individualBestObjective, individualBestViolation);
        this.globalBestSparrow = this.sparrows.get(0);

        for (int t = 0; t < this.maxIterations; t++) {
            // Sort sparrows by fitness (ascending)
            Integer[] sortIndex = sortSparrowsByViolationAndObjective(individualBestObjective);

            // Find worst sparrow
            int worstIdx = sortIndex[this.numberOfSparrow - 1];
            double[][] worse = this.sparrows.get(worstIdx).position;

            double r2 = Math.random();

            // Producer update
            updateProducers(r2, sortIndex, individualBestPosition);

            // Find current best
            Sparrow bestXX = findCurrentBestSparrow();
            // Scrounger update
            updateScroungers(sortIndex, worse, individualBestPosition, bestXX);

            // Danger-aware sparrow update (random 20%)
            updateDangerAwareSparrows(bestXX, individualBestPosition, worse, worstIdx);

            // Update personal best
            updatePersonalBestAndGlobalBest(individualBestPosition, individualBestObjective, individualBestViolation);
        }

        double[][] bestPositionDenorm = this.normalizer.denormalize(this.globalBestSparrow.position);
        adjustDenormalizedSolution(bestPositionDenorm);
        System.out.println("Best Objective: " + this.globalBestSparrow.evaluationResult.objective);
        System.out.println("Final Violation: " + this.globalBestSparrow.evaluationResult.violation);
        return bestPositionDenorm;
    }

    private void updatePersonalBestAndGlobalBest(double[][][] individualBestPosition, double[] individualBestObjective, int[] individualBestViolation) {
        for (int sparrowIndex = 0; sparrowIndex < this.numberOfSparrow; sparrowIndex++) {
            Sparrow currentSparrow = this.sparrows.get(sparrowIndex);

            if (isBetterSolution(
                    currentSparrow.evaluationResult.objective,
                    currentSparrow.evaluationResult.violation,
                    individualBestObjective[sparrowIndex],
                    individualBestViolation[sparrowIndex])) {
                individualBestObjective[sparrowIndex] = currentSparrow.evaluationResult.objective;
                individualBestViolation[sparrowIndex] = currentSparrow.evaluationResult.violation;
                individualBestPosition[sparrowIndex] = deepCopySolution(currentSparrow.position);
            }

            if (isBetterSolution(
                    individualBestObjective[sparrowIndex],
                    individualBestViolation[sparrowIndex],
                    this.globalBestSparrow.evaluationResult.objective,
                    this.globalBestSparrow.evaluationResult.violation)) {
                this.globalBestSparrow = currentSparrow;
            }
        }
    }


    private void updateDangerAwareSparrows(Sparrow bestXX, double[][][] individualBestPosition, double[][] worse, int worstIdx) {
        int numDangerAware = Math.max(1, this.numberOfSparrow / 5);
        int[] dangerIndices = new int[numDangerAware];
        for (int i = 0; i < numDangerAware; i++) {
            dangerIndices[i] = this.random.nextInt(this.numberOfSparrow);
        }

        for (int j = 0; j < numDangerAware; j++) {
            int idx = dangerIndices[j];
            Sparrow sparrow = this.sparrows.get(idx);
            double[][] newPosition = new double[NUM_VARIABLES][this.maxFix];

            if (!isBetterSolution(sparrow.evaluationResult.objective, sparrow.evaluationResult.violation, bestXX.evaluationResult.objective, bestXX.evaluationResult.violation)) {
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        newPosition[v][f] = bestXX.position[v][f] + this.random.nextGaussian() * Math.abs(individualBestPosition[idx][v][f] - bestXX.position[v][f]);
                        newPosition[v][f] = Math.max(0.0, Math.min(1.0, newPosition[v][f]));
                    }
                }
            } else {
                double K = 2.0 * Math.random() - 1.0;
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        newPosition[v][f] = individualBestPosition[idx][v][f] + K * Math.abs(individualBestPosition[idx][v][f] - worse[v][f]) / (sparrow.evaluationResult.objective - this.sparrows.get(worstIdx).evaluationResult.objective + 1e-50);
                        newPosition[v][f] = Math.max(0.0, Math.min(1.0, newPosition[v][f]));
                    }
                }
            }

            double[][] pDenorm = this.normalizer.denormalize(newPosition);
            adjustDenormalizedSolution(pDenorm);
            EvaluationResult newEval = evaluateSolution(pDenorm);

            if (isBetterSolution(newEval.objective, newEval.violation, sparrow.evaluationResult.objective, sparrow.evaluationResult.violation)) {
                sparrow.position = newPosition;
                sparrow.evaluationResult = newEval;
            }
        }
    }


    private void updateScroungers(Integer[] sortIndex, double[][] worse, double[][][] individualBestPosition, Sparrow bestXX) {
        for (int i = this.numberOfProducers; i < this.numberOfSparrow; i++) {
            int idx = sortIndex[i];
            Sparrow sparrow = this.sparrows.get(idx);
            double[][] newPosition = new double[NUM_VARIABLES][this.maxFix];

            double[][] A = new double[1][this.maxFix];
            for (int f = 0; f < this.maxFix; f++) {
                A[0][f] = (Math.random() < 0.5 ? 0 : 1) * 2 - 1;
            }
            double[][] APseudoInverse = computePseudoInverse(A);

            for (int v = 0; v < NUM_VARIABLES; v++) {
                for (int f = 0; f < this.maxFix; f++) {
                    if (i > this.numberOfSparrow / 2) {
                        newPosition[v][f] = this.random.nextGaussian() * Math.exp((worse[v][f] - individualBestPosition[idx][v][f]) / Math.pow(i, 2));
                    } else {
                        newPosition[v][f] = bestXX.position[v][f] + Math.abs(individualBestPosition[idx][v][f] - bestXX.position[v][f]) * APseudoInverse[f][0];
                    }
                    newPosition[v][f] = Math.max(0.0, Math.min(1.0, newPosition[v][f]));
                }
            }

            double[][] pDenorm = this.normalizer.denormalize(newPosition);
            adjustDenormalizedSolution(pDenorm);
            EvaluationResult newEval = evaluateSolution(pDenorm);

            if (isBetterSolution(newEval.objective, newEval.violation, sparrow.evaluationResult.objective, sparrow.evaluationResult.violation)) {
                sparrow.position = newPosition;
                sparrow.evaluationResult = newEval;
            }
        }
    }


    private Sparrow findCurrentBestSparrow() {
        Sparrow currentBestSparrow = this.sparrows.get(0);

        for (int sparrowIndex = 1; sparrowIndex < this.numberOfSparrow; sparrowIndex++) {
            Sparrow candidateSparrow = this.sparrows.get(sparrowIndex);

            if (isBetterSolution(
                    candidateSparrow.evaluationResult.objective,
                    candidateSparrow.evaluationResult.violation,
                    currentBestSparrow.evaluationResult.objective,
                    currentBestSparrow.evaluationResult.violation)) {
                currentBestSparrow = candidateSparrow;
            }
        }

        return currentBestSparrow;
    }

    private void updateProducers(double r2, Integer[] sortIndex, double[][][] individualBestPosition) {
        if (r2 < 0.8) {
            for (int i = 0; i < this.numberOfProducers; i++) {
                int idx = sortIndex[i];
                Sparrow sparrow = this.sparrows.get(idx);
                double[][] newPosition = new double[NUM_VARIABLES][this.maxFix];
                double r1 = 1 - Math.random();

                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        newPosition[v][f] = individualBestPosition[idx][v][f] * Math.exp(-(i + 1) / (r1 * this.maxIterations));
                        newPosition[v][f] = Math.max(0.0, Math.min(1.0, newPosition[v][f]));
                    }
                }

                double[][] pDenorm = this.normalizer.denormalize(newPosition);
                adjustDenormalizedSolution(pDenorm);
                EvaluationResult newEval = evaluateSolution(pDenorm);

                if (isBetterSolution(newEval.objective, newEval.violation, sparrow.evaluationResult.objective, sparrow.evaluationResult.violation)) {
                    sparrow.position = newPosition;
                    sparrow.evaluationResult = newEval;
                }
            }
        } else {
            for (int i = 0; i < this.numberOfProducers; i++) {
                int idx = sortIndex[i];
                Sparrow sparrow = this.sparrows.get(idx);
                double[][] newPosition = new double[NUM_VARIABLES][this.maxFix];

                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        newPosition[v][f] = individualBestPosition[idx][v][f] + this.random.nextGaussian();
                        newPosition[v][f] = Math.max(0.0, Math.min(1.0, newPosition[v][f]));
                    }
                }

                double[][] pDenorm = this.normalizer.denormalize(newPosition);
                adjustDenormalizedSolution(pDenorm);
                EvaluationResult newEval = evaluateSolution(pDenorm);

                if (isBetterSolution(newEval.objective, newEval.violation, sparrow.evaluationResult.objective, sparrow.evaluationResult.violation)) {
                    sparrow.position = newPosition;
                    sparrow.evaluationResult = newEval;
                }
            }
        }
    }


    private Integer[] sortSparrowsByViolationAndObjective(double[] individualBestObjective) {
        Integer[] sortIndex = new Integer[this.numberOfSparrow];
        for (int i = 0; i < this.numberOfSparrow; i++) {
            sortIndex[i] = i;
        }
        java.util.Arrays.sort(sortIndex, (indexA, indexB) -> {
            Sparrow sparrowA = this.sparrows.get(indexA);
            Sparrow sparrowB = this.sparrows.get(indexB);

            if (isBetterSolution(
                    sparrowA.evaluationResult.objective,
                    sparrowA.evaluationResult.violation,
                    sparrowB.evaluationResult.objective,
                    sparrowB.evaluationResult.violation)) {
                return -1;
            } else if (isBetterSolution(
                    sparrowB.evaluationResult.objective,
                    sparrowB.evaluationResult.violation,
                    sparrowA.evaluationResult.objective,
                    sparrowA.evaluationResult.violation)) {
                return 1;
            } else {
                return 0;
            }
        });
        return sortIndex;
    }

    private void initializeIndividualBest(double[][][] individualBestPosition, double[] individualBestObjective, int[] individualBestViolation) {
        for (int i = 0; i < this.numberOfSparrow; i++) {
            individualBestPosition[i] = deepCopySolution(this.sparrows.get(i).position);
            individualBestObjective[i] = this.sparrows.get(i).evaluationResult.objective;
            individualBestViolation[i] = this.sparrows.get(i).evaluationResult.violation;
        }
    }


    private double[][] computePseudoInverse(double[][] A) {
        // A is 1 × d
        // A^T is d × 1
        double[][] AT = transpose(A);

        // AA^T is 1 × 1 (scalar)
        double AAT = 0.0;
        for (int f = 0; f < this.maxFix; f++) {
            AAT += A[0][f] * A[0][f];
        }

        // A+ = A^T / AA^T (element-wise division)
        double[][] APseudoInverse = new double[this.maxFix][1];
        for (int f = 0; f < this.maxFix; f++) {
            APseudoInverse[f][0] = AT[f][0] / AAT;
        }

        return APseudoInverse;
    }

    private double[][] transpose(double[][] matrix) {
        double[][] transposed = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    private boolean isBetterSolution(double obj1, int viol1, double obj2, int viol2) {
        boolean feasible1 = (viol1 == 0);
        boolean feasible2 = (viol2 == 0);

        if (feasible1 && feasible2) {
            return obj1 < obj2;
        } else if (feasible1 && !feasible2) {
            return true;
        } else if (!feasible1 && feasible2) {
            return false;
        } else {
            return viol1 < viol2;
        }
    }

    private EvaluationResult evaluateSolution(double[][] position) {
        EvaluationResult result = new EvaluationResult();

        if (checkEmptySolution(position)) {
            result.objective = Double.POSITIVE_INFINITY;
            result.violation = Integer.MAX_VALUE;
            return result;
        }

        double[] t = position[Variable.T.getId()];
        int placed = 0;
        for (double val : t) {
            if (Math.floor(val) != 0.0) placed++;
        }

        if (placed < this.minFix) {
            result.objective = Double.POSITIVE_INFINITY;
            result.violation = Integer.MAX_VALUE - (this.minFix - placed);
            return result;
        }

        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(position).stream().filter(f -> f.getType() != 0).toList();
        List<Fixture> repairedFixtures;

        int violation = 0;

        for (int fIdx = 0; fIdx < fixtures.size(); fIdx++) {
            Fixture fixture = fixtures.get(fIdx);
            int fixtureViolation = geometryConstraint.computePenaltyGeometryViolation(fixture.getVertices());

            if (fixtureViolation > 0) {
                List<Tuple<Float, Float>> adjustedVertices = geometryConstraint.tryAdjustGeometryViolation(fixture.getVertices());

                if (adjustedVertices != null && !adjustedVertices.isEmpty() && adjustedVertices.get(0) != null) {
                    Tuple<Float, Float> v0 = adjustedVertices.get(0);
                    // write first vertex back to denormalized position (widen Float -> double)
                    position[Variable.X.getId()][fIdx] = v0.getFirst().doubleValue();
                    position[Variable.Y.getId()][fIdx] = v0.getSecond().doubleValue();
                }
            }
            //System.out.println("Geometry violation: " + violation);
        }

        repairedFixtures = FixtureUtility.getFixturesFromPosition(position);
        List<Rectangle> rectangles = FixtureUtility.getRectangleFromFixture(repairedFixtures, t);


        for (Fixture fixture : fixtures) {
            violation += geometryConstraint.computePenaltyGeometryViolation(fixture.getVertices());
        }


        //System.out.println("Geometry violation: " + violation);

        violation += geometryConstraint.computePenaltySecurityDistanceViolation(repairedFixtures);
        //System.out.println("Security distance violation: " + violation);
        violation += fixtureConstraints.computePenaltyFixtureViolation(t);
        //System.out.println("Fixture type violation: " + violation);
        violation += noOverlapConstraint.computePenaltyOverlapBetweenFixtures(rectangles);
        //System.out.println("Overlap violation: " + violation);
        violation += noOverlapConstraint.computePenaltyOverlapWithHoles(rectangles);
        //System.out.println("Hole overlap violation: " + violation);

        result.violation = violation;
        result.objective = computeObjectiveFunction(repairedFixtures);
        return result;
    }

    private double computeObjectiveFunction(List<Fixture> fixtures) {
        List<Fixture> fixtureUsed = fixtures.stream().filter(f -> f.getType() != 0).toList();
        Tuple<Float, Float> cg = MomentOfInertia.computeOverallCenterOfGravity(fixtureUsed);
        Tuple<Float, Float> moi = MomentOfInertia.computeCombinedBaricentricMomentsOfInertia(fixtureUsed, cg);
        double fIner = Math.abs(moi.getFirst()) + Math.abs(moi.getSecond());

        return -(fIner);
    }

    private boolean checkEmptySolution(double[][] position) {
        double[] t = position[Variable.T.getId()];
        for (double type : t) {
            if (type != 0.0) {
                return false;
            }
        }
        return true;
    }

    private double[][] setupInitialSolution(double[][] initialPosition) {
        double[][] initializedInitialSolution = new double[initialPosition.length][initialPosition[0].length + (this.maxFix - initialPosition[0].length)];
        for (int i = 0; i < NUM_VARIABLES; i++) {
            for (int j = 0; j < initialPosition[0].length + (this.maxFix - initialPosition.length); j++) {
                if (j > initialPosition[0].length - 1) {
                    if (j == Variable.X.getId()) {
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.X.getId()).getSecond();
                    } else if (j == Variable.Y.getId()) {
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.Y.getId()).getSecond();
                    } else if (j == Variable.ANGLE.getId()) {
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.ANGLE.getId()).getFirst();
                    } else if (j == Variable.T.getId()) {
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.T.getId()).getFirst();
                    }
                } else {
                    initializedInitialSolution[i][j] = initialPosition[i][j];
                }
            }
        }

        return initializedInitialSolution;
    }

    private void adjustDenormalizedSolution(double[][] position) {
        double[] x = position[Variable.X.getId()];
        double[] y = position[Variable.Y.getId()];
        double[] angle = position[Variable.ANGLE.getId()];
        double[] t = position[Variable.T.getId()];

        for (int i = 0; i < t.length; i++) {
            t[i] = Math.floor(t[i]);
            if (t[i] == this.bounds.get(Variable.T.getId()).getFirst() || x[i] == this.bounds.get(Variable.X.getId()).getSecond()
                    || y[i] == this.bounds.get(Variable.Y.getId()).getSecond()) {
                t[i] = 0.0;
                x[i] = bounds.get(Variable.X.getId()).getSecond();
                y[i] = bounds.get(Variable.Y.getId()).getSecond();
                angle[i] = 0.0;
            }
        }
    }

    private double[][] deepCopySolution(double[][] src) {
        double[][] dst = new double[NUM_VARIABLES][this.maxFix];
        for (int i = 0; i < NUM_VARIABLES; i++) {
            System.arraycopy(src[i], 0, dst[i], 0, this.maxFix);
        }
        return dst;
    }
}
