package polygon;

import utility.data.Triple;
import utility.data.Tuple;

import java.util.List;
import java.util.Objects;

public class Polygon {

    private List<Tuple<Float, Float>> vertices;
    private Float area;
    private Tuple<Float, Float> centroid;
    private Triple<Float, Float, Float> absoluteMomentOfInertia;
    private Triple<Float, Float, Float> baricentricMomentOfInertia;
    private Float angle;

    public Polygon(){

    }

    public Polygon(List<Tuple<Float, Float>> vertices, Float area, Tuple<Float, Float> centroid, Triple<Float, Float, Float> momentOfInertia, Triple<Float, Float, Float> baricentricMomentOfInertia, Float angle) {
        this.vertices = vertices;
        this.area = area;
        this.centroid = centroid;
        this.absoluteMomentOfInertia = momentOfInertia;
        this.baricentricMomentOfInertia = baricentricMomentOfInertia;
        this.angle = angle;
    }

    public List<Tuple<Float, Float>> getVertices() {
        return vertices;
    }

    public Float getArea() {
        return area;
    }

    public Tuple<Float, Float> getCentroid() {
        return centroid;
    }

    public Triple<Float, Float, Float> getAbsoluteMomentOfInertia() {
        return absoluteMomentOfInertia;
    }

    public Triple<Float, Float, Float> getBaricentricMomentOfInertia() {
        return baricentricMomentOfInertia;
    }

    public Float getAngle() {
        return angle;
    }

    public void setVertices(List<Tuple<Float, Float>> vertices) {
        this.vertices = vertices;
    }

    public void setArea(Float area) {
        this.area = area;
    }

    public void setAngle(Float angle) {
        this.angle = angle;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Polygon polygon = (Polygon) o;
        return angle == polygon.angle && Objects.equals(area, polygon.area) && Objects.equals(centroid, polygon.centroid) && Objects.equals(absoluteMomentOfInertia, polygon.absoluteMomentOfInertia);
    }

    @Override
    public int hashCode() {
        return Objects.hash(area, centroid, absoluteMomentOfInertia, angle);
    }

    @Override
    public String toString() {
        return "Polygon{" +
                "vertices=" + vertices +
                ", area=" + area +
                ", centroid=" + centroid +
                ", absoluteMomentOfInertia=" + absoluteMomentOfInertia +
                ", baricentricMomentOfInertia=" + baricentricMomentOfInertia +
                ", angle=" + angle +
                '}';
    }
}
