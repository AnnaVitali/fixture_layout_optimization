import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPoint
from shapely import affinity


# ============================================================
# Accessible colour palette
# Okabe-Ito colour-blind-friendly palette
# ============================================================

RECTANGLE_FACE = "#56B4E9"   # Sky blue
RECTANGLE_EDGE = "#000000"   # Black
HULL_EDGE = "#D55E00"        # Vermillion
HULL_FACE = "#D55E00"        # Vermillion


# ============================================================
# Rectangle
# ============================================================

def make_rectangle(center, width, height, angle=0):
    """Create a rectangle centered at 'center' and rotated by angle."""

    cx, cy = center

    rect = Polygon([
        (-width / 2, -height / 2),
        ( width / 2, -height / 2),
        ( width / 2,  height / 2),
        (-width / 2,  height / 2)
    ])

    rect = affinity.rotate(
        rect,
        angle,
        origin=(0, 0)
    )

    rect = affinity.translate(
        rect,
        xoff=cx,
        yoff=cy
    )

    return rect


# ============================================================
# Generate rectangles
# ============================================================

def generate_rectangles(
    n=8,
    radius=6.0,
    width=3.0,
    height=1.0,
    rotations=None
):

    if rotations is None:
        rotations = np.zeros(n)

    rectangles = []

    for i in range(n):

        theta = 2 * np.pi * i / n

        center = (
            radius * np.cos(theta),
            radius * np.sin(theta)
        )

        rect = make_rectangle(
            center,
            width,
            height,
            rotations[i]
        )

        rectangles.append(rect)

    return rectangles


# ============================================================
# Check that rectangles do not overlap
# ============================================================

def check_no_overlap(rectangles):

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):

            intersection = rectangles[i].intersection(
                rectangles[j]
            )

            if intersection.area > 1e-10:
                return False, (i, j)

    return True, None


# ============================================================
# Global enclosing polygon
# ============================================================

def enclosing_polygon(rectangles):
    """
    Construct the smallest convex polygon containing
    all rectangles.

    The convex hull is computed from ALL vertices of
    ALL rectangles simultaneously.
    """

    all_points = []

    for rectangle in rectangles:
        for x, y in rectangle.exterior.coords:
            all_points.append((x, y))

    return MultiPoint(all_points).convex_hull


# ============================================================
# Plot
# ============================================================

def plot_system(ax, rectangles, title):

    # Global enclosing polygon
    hull = enclosing_polygon(rectangles)

    # --------------------------------------------------------
    # Draw rectangles
    # --------------------------------------------------------

    for rectangle in rectangles:

        x, y = rectangle.exterior.xy

        ax.fill(
            x,
            y,
            facecolor=RECTANGLE_FACE,
            edgecolor=RECTANGLE_EDGE,
            linewidth=1.5,
            alpha=0.75,
            zorder=2
        )

    # --------------------------------------------------------
    # Draw the enclosing polygon
    # --------------------------------------------------------

    x, y = hull.exterior.xy

    ax.fill(
        x,
        y,
        facecolor="none",
        edgecolor=HULL_EDGE,
        linewidth=4,
        zorder=10
    )

    ax.plot(
        x,
        y,
        color=HULL_EDGE,
        linewidth=4,
        zorder=11
    )

    # --------------------------------------------------------
    # Clean SAT-style presentation
    # --------------------------------------------------------

    ax.set_aspect("equal")

    ax.set_title(
        title,
        fontsize=15,
        pad=12
    )

    ax.axis("off")


# ============================================================
# CASE 1 — common orientation
# ============================================================

n = 8

rotations_unrotated = np.zeros(n)

rectangles_1 = generate_rectangles(
    n=n,
    radius=6.0,
    width=3.0,
    height=1.0,
    rotations=rotations_unrotated
)

valid, pair = check_no_overlap(rectangles_1)

if not valid:
    raise RuntimeError(
        f"Rectangles {pair} overlap."
    )


# ============================================================
# CASE 2 — independent rotations
# ============================================================

rotations_rotated = np.array([
     0,
    11,
    -7,
    18,
    -13,
     9,
    -16,
     5
])

rectangles_2 = generate_rectangles(
    n=n,
    radius=6.0,
    width=3.0,
    height=1.0,
    rotations=rotations_rotated
)

valid, pair = check_no_overlap(rectangles_2)

if not valid:
    raise RuntimeError(
        f"Rectangles {pair} overlap."
    )


# ============================================================
# Comparison
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

plot_system(
    axes[0],
    rectangles_1,
    "Unrotated rectangles"
)

plot_system(
    axes[1],
    rectangles_2,
    "Rotated rectangles"
)

plt.tight_layout()

plt.show()