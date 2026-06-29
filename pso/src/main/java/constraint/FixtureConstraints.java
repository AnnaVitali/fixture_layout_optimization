package constraint;
import java.util.List;

public class FixtureConstraints {

    private final List<Integer> availability;

    public FixtureConstraints(List<Integer> availability) {
        this.availability = availability;
    }

    public int computePenaltyFixtureViolation(double[] types){
        return computePenaltyAvailability(types);
    }

    private int computePenaltyAvailability(double[] types){
        double[] count = new double[availability.size()];
        int penalty = 0;
        for (double type : types) {
            count[(int) type] += 1;
        }

        for(int i = 0; i < availability.size(); i++) {
            if (count[i] > availability.get(i)) {
                penalty = penalty + 1;
            }
        }

        return penalty;
    }
}
