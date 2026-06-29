package utility;

import utility.data.Tuple;

import java.util.HashMap;
import java.util.Map;

public class FixtureDimension {

    private static final Map<Integer, Tuple<Integer, Integer>> fixturesDimension = new HashMap<>(){{
        put(0, new Tuple<>(0, 0));
        put(1, new Tuple<>(145, 145));
        put(2, new Tuple<>(180, 65));
    }};

    public static Integer getWidth(Integer type){
        return fixturesDimension.get(type).getFirst();
    }

    public static Integer getHeight(Integer type){
        return fixturesDimension.get(type).getSecond();
    }
}
