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
import java.util.List;
import java.util.Objects;

import static heuristics.ParticleSwarmOptimization.NUM_VARIABLES;

public class TeachingLearningBasedOptimization {

    private int populationSize;
    private int numberOfIterations;
    private Student teacher;
    private List<Student> students;
    private final GeometryConstraint geometryConstraint;
    private final NoOverlapConstraint noOverlapConstraint;
    private final FixtureConstraints fixtureConstraints;
    private final int minFix;
    private final int maxFix;
    private Normalizer normalizer;
    private List<Tuple<Integer, Integer>> bounds;

    public TeachingLearningBasedOptimization(int populationSize, int numberOfIterations, GeometryConstraint geometryConstraint,
                                             NoOverlapConstraint noOverlapConstraint, FixtureConstraints fixtureConstraints, int minFix, int maxFix) {
        this.populationSize = populationSize;
        this.numberOfIterations = numberOfIterations;
        this.students = new ArrayList<>(populationSize);
        this.geometryConstraint = geometryConstraint;
        this.noOverlapConstraint = noOverlapConstraint;
        this.fixtureConstraints = fixtureConstraints;
        this.minFix = minFix;
        this.maxFix = maxFix;
    }

    static class Student{
        double[][] solution;
        EvaluationResult solutionEvaluation;
    }

    private static class EvaluationResult {
        double objective;
        int violation;
    }

    public void initialize(double[][] initialSolution, List<Tuple<Integer, Integer>> bounds) {
        Objects.requireNonNull(bounds, "bounds must not be null");
        this.normalizer = new Normalizer(bounds);
        this.bounds = bounds;

        double [][] initialNorm = null;
        double [][] initialDenorm = null;

        if (initialSolution != null) {
            initialSolution = setupInitialSolution(initialSolution);
            initialNorm = this.normalizer.normalize(initialSolution);
        }

        initialDenorm = this.normalizer.denormalize(initialNorm);
        adjustDenormalizedSolution(initialDenorm);

        EvaluationResult initialEvaluation = evaluateSolution(initialDenorm);

        System.out.println("Initial objective: " + initialEvaluation.objective + ", violation: " + initialEvaluation.violation);
        double delta = 10.0;

        for(int i = 0; i < this.populationSize; i++){
            Student student = new Student();
            if (initialNorm != null && i == 0) {
                student.solution = deepCopySolution(initialNorm);
            } else {
                double[][] perturbedSolution = new double[NUM_VARIABLES][this.maxFix];
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        double base = initialNorm[v][f];
                        double offset = Math.random() * (delta / Math.max(1, bounds.get(v).getSecond() - bounds.get(v).getFirst()));
                        perturbedSolution[v][f] = Math.max(0.0, Math.min(1.0, base + offset));
                    }
                }
                student.solution = perturbedSolution;
            }

            double[][] denormalizedSolution = this.normalizer.denormalize(student.solution);
            adjustDenormalizedSolution(denormalizedSolution);
            student.solutionEvaluation = evaluateSolution(denormalizedSolution);
            this.students.add(student);
        }
    }

    public double[][] optimize(){
        int teacherFacotr = 0;
        double[][] difference = new double[NUM_VARIABLES][this.maxFix];

        for (int i = 0; i < this.numberOfIterations; i++) {
            double[][] meanSolution = computeMeanSolution();
            Student bestStudent = null;
            for (Student student : this.students) {
                if (bestStudent == null || isBetterSolution(student.solutionEvaluation.objective, student.solutionEvaluation.violation,
                        bestStudent.solutionEvaluation.objective, bestStudent.solutionEvaluation.violation)) {
                    bestStudent = student;
                }
            }

            this.teacher = bestStudent;
            teacherFacotr = (int) Math.round(1 + Math.random() * (2 - 1));
            for (int v = 0; v < NUM_VARIABLES; v++) {
                for (int f = 0; f < this.maxFix; f++) {
                    difference[v][f] = Math.random() * (this.teacher.solution[v][f] - teacherFacotr * meanSolution[v][f]);
                }
            }

            //teacher phase
            for (Student student : this.students) {
                double[][] newSolution = new double[NUM_VARIABLES][this.maxFix];
                for (int v = 0; v < NUM_VARIABLES; v++) {
                    for (int f = 0; f < this.maxFix; f++) {
                        newSolution[v][f] = student.solution[v][f] + difference[v][f];
                        newSolution[v][f] = Math.max(0.0, Math.min(1.0, newSolution[v][f]));
                    }
                }

                double[][] denormalizedSolution = this.normalizer.denormalize(newSolution);
                adjustDenormalizedSolution(denormalizedSolution);
                EvaluationResult newEvaluation = evaluateSolution(denormalizedSolution);

                if (isBetterSolution(newEvaluation.objective, newEvaluation.violation,
                        student.solutionEvaluation.objective, student.solutionEvaluation.violation)) {
                    student.solution = newSolution;
                    student.solutionEvaluation = newEvaluation;
                }
            }

            //lerner phase
            for(int s = 0; s < this.students.size(); s++) {
                Student studentA = this.students.get(s);
                int studentBidx = Math.floorMod(s + (int) (Math.random() * (this.students.size() - 1)) + 1, this.students.size());
                Student studentB = this.students.get(studentBidx);
                double[][] newSolution = new double[NUM_VARIABLES][this.maxFix];

                if (isBetterSolution(studentA.solutionEvaluation.objective, studentA.solutionEvaluation.violation,
                        studentB.solutionEvaluation.objective, studentB.solutionEvaluation.violation)) {
                    newSolution = computeInfluence(studentB, studentA, studentB);
                }else if(isBetterSolution(studentB.solutionEvaluation.objective, studentB.solutionEvaluation.violation,
                        studentA.solutionEvaluation.objective, studentA.solutionEvaluation.violation)){
                    newSolution = computeInfluence(studentA, studentB, studentA);
                }

                double[][] denormalizedSolution = this.normalizer.denormalize(newSolution);
                adjustDenormalizedSolution(denormalizedSolution);
                EvaluationResult newEvaluation = evaluateSolution(denormalizedSolution);
                if (isBetterSolution(newEvaluation.objective, newEvaluation.violation,
                        studentA.solutionEvaluation.objective, studentA.solutionEvaluation.violation)) {
                    studentA.solution = newSolution;
                    studentA.solutionEvaluation = newEvaluation;
                }
            }
        }

        double[][] bestSolutionDenormalized = this.normalizer.denormalize(this.teacher.solution);
        adjustDenormalizedSolution(bestSolutionDenormalized);

        System.out.println("Best Objective: " + this.teacher.solutionEvaluation.objective);
        System.out.println("Final Violation: " + this.teacher.solutionEvaluation.violation);

        return bestSolutionDenormalized;
    }

    private double[][] computeInfluence(Student influencedStudent, Student betterStudent, Student worstStudent) {
        double[][] newSolution = new double[NUM_VARIABLES][this.maxFix];
        for (int v = 0; v < NUM_VARIABLES; v++) {
            for (int f = 0; f < this.maxFix; f++) {
                newSolution[v][f] = influencedStudent.solution[v][f] + Math.random() * (betterStudent.solution[v][f] - worstStudent.solution[v][f]);
                newSolution[v][f] = Math.max(0.0, Math.min(1.0, newSolution[v][f]));
            }
        }

        return newSolution;
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

    private double[][] computeMeanSolution() {
        double[][] meanSolution = new double[NUM_VARIABLES][this.maxFix];
        for (Student student : this.students) {
            for (int v = 0; v < NUM_VARIABLES; v++) {
                for (int f = 0; f < this.maxFix; f++) {
                    meanSolution[v][f] += student.solution[v][f];
                }
            }
        }
        for (int v = 0; v < NUM_VARIABLES; v++) {
            for (int f = 0; f < this.maxFix; f++) {
                meanSolution[v][f] /= this.students.size();
            }
        }
        return meanSolution;
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


        for(Fixture fixture : fixtures){
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

        return - (fIner);
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
        double[][] initializedInitialSolution  = new double[initialPosition.length][initialPosition[0].length + (this.maxFix - initialPosition[0].length)];
        for (int i = 0; i < NUM_VARIABLES; i++) {
            for (int j = 0; j < initialPosition[0].length + (this.maxFix - initialPosition.length); j++) {
                if(j > initialPosition[0].length -1){
                    if(j == Variable.X.getId()){
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.X.getId()).getSecond();
                    } else if (j == Variable.Y.getId()){
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.Y.getId()).getSecond();
                    } else if (j == Variable.ANGLE.getId()){
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.ANGLE.getId()).getFirst();
                    } else if (j == Variable.T.getId()){
                        initializedInitialSolution[i][j] = this.bounds.get(Variable.T.getId()).getFirst();
                    }
                }else{
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
