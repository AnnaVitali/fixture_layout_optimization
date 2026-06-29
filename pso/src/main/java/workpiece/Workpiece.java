package workpiece;

import polygon.Rectangle;
import utility.data.Triple;
import utility.data.Tuple;

import java.util.List;
import java.util.Set;

public interface Workpiece {

    Set<Tuple<Float, Float>> getVertices();

    List<Triple<Integer, Integer, Integer>> getInequalities();

    List<Rectangle> getHoles();

    int getMinFix();

    int getMaxFix();

}
