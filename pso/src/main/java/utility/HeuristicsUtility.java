package utility;

public class HeuristicsUtility {
    
    public static double[][] copySolution(double[][] current) {
        double[][] newPosition = new double[current.length][current[0].length];
        for (int i = 0; i < current.length; i++) {
            System.arraycopy(current[i], 0, newPosition[i], 0, current[i].length);
        }
        return newPosition;
    }

}
