package polygon;

import utility.data.Triple;
import utility.data.Tuple;

import java.util.List;
import java.util.Objects;

public class Rectangle extends Polygon {

    private Float width;
    private Float height;

    public Rectangle(){
        super();
    }

    public Rectangle(List<Tuple<Float, Float>> vertices, Float area, Tuple<Float, Float> centroid, Triple<Float, Float, Float> absoluteMomentOfInertia, Triple<Float, Float, Float> baricentricMomentOfInertia, Float angle, Float width, Float height) {
        super(vertices, area, centroid, absoluteMomentOfInertia, baricentricMomentOfInertia, angle);
        this.width = width;
        this.height = height;
    }

    public Float getWidth() {
        return width;
    }

    public Float getHeight() {
        return height;
    }

    public void setWidth(Float width) {
        this.width = width;
    }

    public void setHeight(Float height) {
        this.height = height;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        Rectangle rectangle = (Rectangle) o;
        return Objects.equals(width, rectangle.width) && Objects.equals(height, rectangle.height);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), width, height);
    }

    @Override
    public String toString() {
        return "Rectangle{" +
                "width=" + width +
                ", height=" + height +
                ", vertices=" + getVertices() +
                ", area=" + getArea() +
                ", centroid=" + getCentroid() +
                ", absoluteMomentOfInertia=" + getAbsoluteMomentOfInertia() +
                ", baricentricMomentOfInertia=" + getBaricentricMomentOfInertia() +
                ", angle=" + getAngle() +
                '}';
    }
}
