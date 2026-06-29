package constraint;

import polygon.Rectangle;

import java.util.List;

public class NoOverlapConstraint {

    private final List<Rectangle> Holes;

    public NoOverlapConstraint(List<Rectangle> Holes) {
        this.Holes = Holes;
    }

    public int computePenaltyOverlapWithHoles(List<Rectangle> rectangles) {
        int penalty = 0;

        for(Rectangle rect: rectangles){
            for(Rectangle hole: Holes){
                if(SeparationAxisTheorem.checkOverlap(rect, hole)){
                    penalty += 1;
                }
            }
        }

        return penalty;
    }

    public int computePenaltyOverlapBetweenFixtures(List<Rectangle> rectangles) {
        int penalty = 0;

        for(int i = 0; i < rectangles.size(); i++){
            for(int j = i + 1; j < rectangles.size(); j++){
                if(SeparationAxisTheorem.checkOverlap(rectangles.get(i), rectangles.get(j))){
                    penalty += 1;
                }
            }
        }

        return penalty;
    }


}
