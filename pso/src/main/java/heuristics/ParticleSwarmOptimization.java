// java
package heuristics;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import utility.MomentOfInertia;
import noromalization.Normalizer;
import polygon.Fixture;
import polygon.Rectangle;
import utility.FixtureUtility;
import utility.data.Tuple;
import utility.Variable;
import utility.parameter.MachineParameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class ParticleSwarmOptimization {

    static final int NUM_VARIABLES = 4;


    private final int swarmSize;
    private final int maxIterations;
    private final float inertiaWeight;
    private final float cognitiveCoefficient;
    private final float socialCoefficient;
    private final List<Particle> swarm;
    private List<Tuple<Integer, Integer>> bounds;
    private double[][] globalBestPosition;
    private double globalBestFitness;
    private int globalPenalty;
    private final GeometryConstraint geometryConstraint;
    private final NoOverlapConstraint noOverlapConstraint;
    private final FixtureConstraints fixtureConstraints;
    private final int minFix;
    private final int maxFix;
    private Normalizer normalizer;

    public ParticleSwarmOptimization(int swarmSize, int maxIterations, float inertiaWeight, float cognitiveCoefficient,
                                     float socialCoefficient, GeometryConstraint geometryConstraint,
                                     NoOverlapConstraint noOverlapConstraint, FixtureConstraints fixtureConstraints, int minFix, int maxFix) {
        this.swarmSize = swarmSize;
        this.maxIterations = maxIterations;
        this.inertiaWeight = inertiaWeight;
        this.cognitiveCoefficient = cognitiveCoefficient;
        this.socialCoefficient = socialCoefficient;
        this.swarm = new ArrayList<>(swarmSize);
        this.minFix = minFix;
        this.maxFix = maxFix;
        this.globalBestPosition = new double[NUM_VARIABLES][this.maxFix];
        this.globalBestFitness = Double.POSITIVE_INFINITY;
        this.geometryConstraint = geometryConstraint;
        this.noOverlapConstraint = noOverlapConstraint;
        this.fixtureConstraints = fixtureConstraints;
        this.globalPenalty = Integer.MAX_VALUE;
    }

    static class Particle {
        double[][] position;
        double[][] velocity;
        double[][] bestPosition;
        double bestFitness = Double.POSITIVE_INFINITY;
        int bestViolation = Integer.MAX_VALUE;

        Particle(double[][] position, double[][] velocity, int maxFix) {
            this.position = new double[NUM_VARIABLES][maxFix];
            this.velocity = new double[NUM_VARIABLES][maxFix];
            this.bestPosition = new double[NUM_VARIABLES][maxFix];

            for (int i = 0; i < NUM_VARIABLES; i++) {
                System.arraycopy(position[i], 0, this.position[i], 0, maxFix);
                System.arraycopy(velocity[i], 0, this.velocity[i], 0, maxFix);
                System.arraycopy(position[i], 0, this.bestPosition[i], 0, maxFix);
            }
        }
    }

    private static class EvaluationResult {
        double objective;
        int violation;
    }


    public void initialize(double[][] initialPositions, List<Tuple<Integer, Integer>> bounds) {
        Objects.requireNonNull(bounds, "bounds must not be null");
        this.bounds = bounds;
        this.normalizer = new Normalizer(bounds);
        this.swarm.clear();
        double delta = 10.0;

        double[][] initialNorm = null;
        if (initialPositions != null) {
            initialPositions = setupInitialSolution(initialPositions);
            initialNorm = this.normalizer.normalize(initialPositions);//normalizeLocal(initialPositions, this.bounds);
        }

        if (initialNorm == null || !isValidInitialPositions(initialNorm)
                || checkEmptySolution(this.normalizer.denormalize(initialNorm))) {
            initialNorm = createRandomSeedNormalized();
        }

        double[][] initialDenorm = this.normalizer.denormalize(initialNorm);//denormalizeLocal(initialNorm, this.bounds);
        adjustDenormalizedSolution(initialDenorm);
        EvaluationResult initEval = evaluateSolution(initialDenorm);

        if (!Double.isInfinite(initEval.objective) && !Double.isNaN(initEval.objective)) {
            this.globalBestPosition = deepCopyPosition(initialNorm);
            this.globalBestFitness = initEval.objective;
            this.globalPenalty = initEval.violation;
        } else {
            this.globalBestPosition = deepCopyPosition(initialNorm);
            this.globalBestFitness = Double.POSITIVE_INFINITY;
            this.globalPenalty = Integer.MAX_VALUE;
        }

        System.out.println("Initial global objective: " + globalBestFitness + ", violation: " + globalPenalty);

        for (int i = 0; i < swarmSize; i++) {
            double[][] pos = new double[NUM_VARIABLES][this.maxFix];
            double[][] vel = new double[NUM_VARIABLES][this.maxFix];

            if (i == 0) {
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        pos[v][f] = initialNorm[v][f];
                        vel[v][f] = Math.random() * 2 - 1;
                    }
                }
            } else {
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        double base = initialNorm[v][f];
                        double offset = Math.random() * (delta / Math.max(1, bounds.get(v).getSecond() - bounds.get(v).getFirst()));
                        pos[v][f] = Math.max(0.0, Math.min(1.0, base + offset));
                        vel[v][f] = Math.random();//Math.random() * 2 - 1;
                    }
                }
            }

            Particle particle = new Particle(pos, vel, this.maxFix);
            double[][] pDenorm = this.normalizer.denormalize(particle.position);//denormalizeLocal(particle.position, this.bounds);
            adjustDenormalizedSolution(pDenorm);
            EvaluationResult eval = evaluateSolution(pDenorm);
            particle.bestFitness = eval.objective;
            particle.bestViolation = eval.violation;
            particle.bestPosition = deepCopyPosition(particle.position);
            swarm.add(particle);
        }
    }

    public double[][] optimize() {
        for (int iter = 0; iter < maxIterations; iter++) {
            for (int pIdx = 0; pIdx < swarm.size(); pIdx++) {
                Particle particle = swarm.get(pIdx);
                updateVelocity(particle);
                updatePosition(particle);

                double[][] denorm = this.normalizer.denormalize(particle.position);//denormalizeLocal(particle.position, this.bounds);
                adjustDenormalizedSolution(denorm);
                EvaluationResult eval = evaluateSolution(denorm);

                if (isBetterSolution(eval.objective, eval.violation, particle.bestFitness, particle.bestViolation)) {
                    particle.bestFitness = eval.objective;
                    particle.bestViolation = eval.violation;
                    particle.bestPosition = deepCopyPosition(particle.position);
                }

                if (isBetterSolution(eval.objective, eval.violation, this.globalBestFitness, this.globalPenalty)) {
                    this.globalBestFitness = eval.objective;
                    this.globalPenalty = eval.violation;
                    this.globalBestPosition = deepCopyPosition(particle.position); // safe copy
                }
            }
        }

        System.out.println("Best Objective: " + globalBestFitness);
        System.out.println("Final Violation: " + globalPenalty);

        double[][] globalBestDenorm = this.normalizer.denormalize(globalBestPosition);

        return deepCopyPosition(globalBestDenorm);
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

    private double[][] setupInitialSolution(double[][] initialPosition) {
        double[][] initializedInitialSolution  = new double[initialPosition.length][initialPosition[0].length + (this.maxFix - initialPosition[0].length)];
        for (int i = 0; i < NUM_VARIABLES; i++) {
               for (int j = 0; j < initialPosition[0].length + (this.maxFix - initialPosition.length); j++) {
                   if(j > initialPosition[0].length -1){
                   if(j == Variable.X.getId()){
                       initializedInitialSolution[i][j] = bounds.get(Variable.X.getId()).getSecond();
                   } else if (j == Variable.Y.getId()){
                       initializedInitialSolution[i][j] = bounds.get(Variable.Y.getId()).getSecond();
                   } else if (j == Variable.ANGLE.getId()){
                       initializedInitialSolution[i][j] = bounds.get(Variable.ANGLE.getId()).getFirst();
                   } else if (j == Variable.T.getId()){
                       initializedInitialSolution[i][j] = bounds.get(Variable.T.getId()).getFirst();
                   }
               }else{
                       initializedInitialSolution[i][j] = initialPosition[i][j];
               }
           }
        }

        return initializedInitialSolution;
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
            //System.out.println("Placed fixtures: " + placed + " < " + MIN_PLACED_FIXTURES);
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


        for(Fixture fixture : fixtures){
            violation += geometryConstraint.computePenaltyGeometryViolation(fixture.getVertices());
        }


//        System.out.println("Geometry violation: " + violation);

        violation += geometryConstraint.computePenaltySecurityDistanceViolation(repairedFixtures);
//        System.out.println("Security distance violation: " + violation);
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



    private boolean checkEmptySolution(double[][] position) {
        double[] t = position[Variable.T.getId()];
        for (double type : t) {
            if (type != 0.0) {
                return false;
            }
        }
        return true;
    }

    private void updateVelocity(Particle particle) {
        for (int i = 0; i < NUM_VARIABLES; i++) {
            for (int j = 0; j < this.maxFix; j++) {
                double r1 = Math.random();
                double r2 = Math.random();
                particle.velocity[i][j] = inertiaWeight * particle.velocity[i][j]
                        + cognitiveCoefficient * r1 * (particle.bestPosition[i][j] - particle.position[i][j])
                        + socialCoefficient * r2 * (globalBestPosition[i][j] - particle.position[i][j]);
            }
        }
    }

    private void updatePosition(Particle particle) {
        for (int i = 0; i < NUM_VARIABLES; i++) {
            for (int j = 0; j < this.maxFix; j++) {
                particle.position[i][j] += particle.velocity[i][j];
                if (particle.position[i][j] < 0.0) {
                    particle.position[i][j] = 0.0;
                } else if (particle.position[i][j] > 1.0) {
                    particle.position[i][j] = 1.0;
                }
            }
        }
    }

    private double computeObjectiveFunction(List<Fixture> fixtures) {
        List<Fixture> fixtureUsed = fixtures.stream().filter(f -> f.getType() != 0).toList();
        Tuple<Float, Float> cg = MomentOfInertia.computeOverallCenterOfGravity(fixtureUsed);
        Tuple<Float, Float> moi = MomentOfInertia.computeCombinedBaricentricMomentsOfInertia(fixtureUsed, cg);
        double fIner = Math.abs(moi.getFirst()) + Math.abs(moi.getSecond());

        return - (fIner);
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

    private double[][] deepCopyPosition(double[][] src) {
        double[][] dst = new double[NUM_VARIABLES][this.maxFix];
        for (int i = 0; i < NUM_VARIABLES; i++) {
            System.arraycopy(src[i], 0, dst[i], 0, this.maxFix);
        }
        return dst;
    }

    private boolean isValidInitialPositions(double[][] initialPositions) {
        if (initialPositions.length != NUM_VARIABLES) return false;
        for (int i = 0; i < NUM_VARIABLES; i++) {
            if (initialPositions[i] == null || initialPositions[i].length != this.maxFix) return false;
        }
        return true;
    }

    private double[][] createRandomSeedNormalized() {
        double[][] seed = new double[NUM_VARIABLES][this.maxFix];
        for (int v = 0; v < NUM_VARIABLES; v++) {
            for (int f = 0; f < this.maxFix; f++) {
                seed[v][f] = Math.random();
            }
        }
        return seed;
    }
}