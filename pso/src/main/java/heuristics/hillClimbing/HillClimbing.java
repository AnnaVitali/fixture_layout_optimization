package heuristics.hillClimbing;

import heuristics.hillClimbing.operator.ConstraintsChecker;
import heuristics.hillClimbing.operator.MovementOperator;
import heuristics.hillClimbing.operator.RotationOperator;
import polygon.Fixture;
import utility.FixtureUtility;
import utility.HeuristicsUtility;
import utility.MomentOfInertia;
import utility.data.Tuple;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HillClimbing {

    private final ConstraintsChecker constraintsChecker;
    private final MovementOperator movementOperator;
    private final RotationOperator rotationOperator;
    private final int maxIterations;
    private final Random random = new Random();

    public HillClimbing(
            ConstraintsChecker constraintsChecker,
            MovementOperator movementOperator,
            RotationOperator rotationOperator,
            int maxIterations) {

        this.constraintsChecker = constraintsChecker;
        this.movementOperator = movementOperator;
        this.rotationOperator = rotationOperator;
        this.maxIterations = maxIterations;
    }

    public double[][] optimize(double[][] initialPosition) {
        // default neighborhood size (increased for broader stochastic search)
        return optimize(initialPosition, 20);
    }

    public double[][] optimize(double[][] initialPosition, int neighborhoodSize) {
        double[][] currentPosition = HeuristicsUtility.copySolution(initialPosition);
        double currentScore = evaluateObjective(currentPosition);

        //System.out.println("HillClimbing initial score (fIner): " + currentScore + "  (PSO objective: " + (-currentScore) + ")");

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            List<Candidate> improving = new ArrayList<>();
            Candidate bestSampled = null;

            // sample the neighborhood stochastically
            for (int n = 0; n < neighborhoodSize; n++) {
                double[][] proposal = rotationOperator.propose(currentPosition);
                Candidate cand = evaluateCandidate(proposal);

                if (!constraintsChecker.isValidProposal(proposal)) {
                    proposal = movementOperator.propose(currentPosition);
                    cand = evaluateCandidate(proposal);
                }

                if (bestSampled == null || cand.score > bestSampled.score) {
                    bestSampled = cand;
                }

                if (cand.score > currentScore) {
                    improving.add(cand);
                }
            }

            if (improving.isEmpty()) {
                if (bestSampled != null) {
                    //System.out.println("Iteration " + iteration + ": no improving neighbor, best sampled = " + bestSampled.score + " (current=" + currentScore + ")");
                }
                continue;
            }

            // choose randomly among improving neighbors
            Candidate chosen = improving.get(random.nextInt(improving.size()));
            currentPosition = HeuristicsUtility.copySolution(chosen.position);
            double prevScore = currentScore;
            currentScore = chosen.score;
            //System.out.println("Iteration " + iteration + ": accepted improvement from " + prevScore + " -> " + currentScore);
        }

        System.out.println("Final solution after hill climbing: " + currentScore);

        return currentPosition;
    }

    public double evaluateObjective(double[][] solution) {
        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(solution);
        if (fixtures.isEmpty()) {
            return Double.NEGATIVE_INFINITY;
        }

        Tuple<Float, Float> overallCentroid = MomentOfInertia.computeOverallCenterOfGravity(fixtures);
        Tuple<Float, Float> principalMoments = MomentOfInertia.computeCombinedBaricentricMomentsOfInertia(fixtures, overallCentroid);
        return Math.abs(principalMoments.getFirst()) + Math.abs(principalMoments.getSecond());
    }

    private Candidate evaluateCandidate(double[][] solution) {
        if (!constraintsChecker.isValidProposal(solution)) {
            return new Candidate(solution, Double.NEGATIVE_INFINITY);
        }

        return new Candidate(solution, evaluateObjective(solution));
    }

    private Candidate selectBetter(Candidate currentBest, Candidate challenger) {
        return challenger.score > currentBest.score ? challenger : currentBest;
    }

    private Candidate currentCandidateSolution(double[][] solution, double score) {
        return new Candidate(solution, score);
    }

    private static final class Candidate {
        private final double[][] position;
        private final double score;

        private Candidate(double[][] position, double score) {
            this.position = position;
            this.score = score;
        }
    }
}
