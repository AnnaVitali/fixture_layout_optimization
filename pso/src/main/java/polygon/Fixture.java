package polygon;

import utility.data.Triple;
import utility.data.Tuple;

import java.util.List;
import java.util.Objects;

public class Fixture extends Polygon {
    private int type;
    private Tuple<Float, Float> baseCenter;

    public Fixture(){
        super();
    }

    public Fixture(List<Tuple<Float, Float>> vertices, Float area, Tuple<Float, Float> centroid, Triple<Float, Float, Float> absoluteMomentOfInertia, Triple<Float, Float, Float> baricentricMomentOfInertia, Float angle, int type, Tuple<Float, Float> baseCenter) {
        super(vertices, area, centroid, absoluteMomentOfInertia, baricentricMomentOfInertia, angle);
        this.type = type;
        this.baseCenter = baseCenter;
    }

    public int getType() {
        return type;
    }

    public Tuple<Float, Float> getBaseCenter() {
        return baseCenter;
    }

    public Tuple<Float, Float> getProjectA(){
        float edge1x = this.getVertices().get(1).getFirst() - this.getVertices().get(0).getFirst();
        float edge1y = this.getVertices().get(1).getSecond() - this.getVertices().get(0).getSecond();
        return getProjection(edge1x, edge1y);
    }

    public Tuple<Float, Float> getProjectB(){
        float edge2x = this.getVertices().get(3).getFirst() - this.getVertices().get(0).getFirst();
        float edge2y = this.getVertices().get(3).getSecond() - this.getVertices().get(0).getSecond();
        return getProjection(edge2x, edge2y);
    }

    private Tuple<Float, Float> getProjection(float edgeX, float edgeY) {
        float edgeNorm = (float) Math.sqrt(edgeX * edgeX + edgeY * edgeY);
        float ux = edgeX / edgeNorm;
        float uy = edgeY / edgeNorm;

        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;

        for(int i = 0; i< getVertices().size(); i ++){
            float candidate = getVertices().get(i).getFirst() * ux + getVertices().get(i).getSecond() * uy;
            if(candidate < min){
                min = candidate;
            } else if (candidate > max) {
                max = candidate;
            }
        }

        return new Tuple<>(min, max);
    }

    public void setType(int type) {
        this.type = type;
    }

    public void setBaseCenter(Tuple<Float, Float> baseCenter) {
        this.baseCenter = baseCenter;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        Fixture fixture = (Fixture) o;
        return type == fixture.type && Objects.equals(baseCenter, fixture.baseCenter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), type, baseCenter);
    }

    @Override
    public String toString() {
        return "Fixture{" +
                "type=" + type +
                ", baseCenter=" + baseCenter +
                '}';
    }
}
