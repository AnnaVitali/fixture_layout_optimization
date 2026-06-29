package heuristics.hillClimbing.operator;

import constraint.FixtureConstraints;
import constraint.GeometryConstraint;
import constraint.NoOverlapConstraint;
import polygon.Fixture;
import polygon.Rectangle;
import utility.FixtureUtility;

import java.util.List;

public class ConstraintsChecker {

    private final GeometryConstraint geometryConstraint;
    private final NoOverlapConstraint noOverlapConstraint;
    private final FixtureConstraints fixtureConstraints;

    public ConstraintsChecker(
            GeometryConstraint geometryConstraint,
            NoOverlapConstraint noOverlapConstraint,
            FixtureConstraints fixtureConstraints) {

        this.geometryConstraint = geometryConstraint;
        this.noOverlapConstraint = noOverlapConstraint;
        this.fixtureConstraints = fixtureConstraints;
    }

    public boolean isValidProposal(double[][] proposedSolutionDenorm) {
        List<Fixture> fixtures = FixtureUtility.getFixturesFromPosition(proposedSolutionDenorm);
        double[] t = proposedSolutionDenorm[proposedSolutionDenorm.length - 1];
        List<Rectangle> rects = FixtureUtility.getRectangleFromFixture(fixtures, t);

        for (Fixture f : fixtures) {
            if (geometryConstraint.computePenaltyGeometryViolation(f.getVertices()) > 0) return false;
        }
        if (geometryConstraint.computePenaltySecurityDistanceViolation(fixtures) > 0) return false;

        if (fixtureConstraints.computePenaltyFixtureViolation(t) > 0) return false;

        if (noOverlapConstraint.computePenaltyOverlapBetweenFixtures(rects) > 0) return false;
        return noOverlapConstraint.computePenaltyOverlapWithHoles(rects) <= 0;
    }
}
