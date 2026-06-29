package utility;

public enum Variable {
    X(0),
    Y(1),
    ANGLE(2),
    T(3);

    private final int id;

    Variable(int id) {
        this.id = id;
    }

    public int getId() {
        return id;
    }
}
