package utility.parameter;

import java.util.List;

public class MachineParameter {

    private static final int WTAB = 2500;
    private static final int HTAB = 1250;
    private static final int WBAR = 145;
    private static final int WMIN = 200;
    private static final int HMIN = 99;
    private static final int NTYPES = 2;
    private static final int MAX_FIXTURES = 6;
    private static final int MIN_FIXTURES = 3;
    private static final List<Integer> FIXTURE_AVAILABILITY = List.of(36, 24, 12);


    public static int getWTab() {
        return WTAB;
    }

    public static int getHTab() {
        return HTAB;
    }

    public static int getWBar() {
        return WBAR;
    }

    public static int getWMin() {
        return WMIN;
    }

    public static int getHmin() {
        return HMIN;
    }

    public static int getHorizontalSecurityDistance() {
        return WMIN;  // WMIN (200) is the horizontal security buffer between fixture centers
    }

    public static int getNTypes() {
        return NTYPES;
    }

    public static int getMaxFixtures() {
        return MAX_FIXTURES;
    }

    public static int getMinFixtures() {
        return MIN_FIXTURES;
    }


    public static List<Integer> getFixtureAvailability() {
        return FIXTURE_AVAILABILITY;
    }

}
