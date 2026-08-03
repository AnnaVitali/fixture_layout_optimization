import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


# Wong 8-Color Palette (colorblind-friendly)
RECT1_COLOR = "#56B4E9"    
RECT2_COLOR = "#D55E00"   
OVERLAP_COLOR = "#009E73"  

def rotate_rectangle(center_x, center_y, width, height, angle_degrees):
    """Return the corners of a rotated rectangle."""
    angle_rad = np.radians(angle_degrees)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    half_width = width / 2
    half_height = height / 2
    local_vertices = [
        [-half_width, -half_height], [half_width, -half_height],
        [half_width, half_height], [-half_width, half_height]
    ]
    vertices = []
    for vx, vy in local_vertices:
        rotated_x = vx * cos_val - vy * sin_val
        rotated_y = vx * sin_val + vy * cos_val
        vertices.append([center_x + rotated_x, center_y + rotated_y])
    return np.array(vertices)

def project_onto_axis(vertices, axis):
    """Project rectangle vertices onto a given axis."""
    dots = np.dot(vertices, axis)
    return np.min(dots), np.max(dots)

def generate_figure_2d_1d(rect1_verts, rect2_verts, title_suffix, xlim_fixed=None, ylim_fixed=None):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1, 1]})
    ax_2d, ax_xproj, ax_yproj = axes

    all_x = np.concatenate([rect1_verts[:, 0], rect2_verts[:, 0]])
    all_y = np.concatenate([rect1_verts[:, 1], rect2_verts[:, 1]])
    margin = 1
    x_min = xlim_fixed[0] if xlim_fixed is not None else min(all_x) - margin
    x_max = xlim_fixed[1] if xlim_fixed is not None else max(all_x) + margin
    y_min = ylim_fixed[0] if ylim_fixed is not None else min(all_y) - margin
    y_max = ylim_fixed[1] if ylim_fixed is not None else max(all_y) + margin

    ax_2d.add_patch(Polygon(rect1_verts, closed=True, facecolor=RECT1_COLOR, alpha=0.6, edgecolor='black', linewidth=1.2))
    ax_2d.add_patch(Polygon(rect2_verts, closed=True, facecolor=RECT2_COLOR, alpha=0.6, edgecolor='black', linewidth=1.2))
    ax_2d.set_aspect('equal', adjustable='box')
    ax_2d.set_xlim(x_min, x_max)
    ax_2d.set_ylim(y_min, y_max)
    ax_2d.set_title(f"2D View ({title_suffix})", fontsize=10)
    ax_2d.grid(True, linestyle='--', alpha=0.6)

    # --- X-axis projection ---
    min1x, max1x = project_onto_axis(rect1_verts, np.array([1, 0]))
    min2x, max2x = project_onto_axis(rect2_verts, np.array([1, 0]))
    x_pad = max((x_max - x_min) * 0.02, 0.15)
    proj_height = 0.16
    top_y = 0.24
    bottom_y = -0.40
    overlap_y = -0.08
    ax_xproj.add_patch(Rectangle((min1x - x_pad, top_y), (max1x - min1x) + 2 * x_pad, proj_height, facecolor=RECT1_COLOR, edgecolor='black', alpha=0.9, linewidth=1.0, label='Rect 1'))
    ax_xproj.add_patch(Rectangle((min2x - x_pad, bottom_y), (max2x - min2x) + 2 * x_pad, proj_height, facecolor=RECT2_COLOR, edgecolor='black', alpha=0.9, linewidth=1.0, label='Rect 2'))
    overlap_min_x = max(min1x - x_pad, min2x - x_pad)
    overlap_max_x = min(max1x + x_pad, max2x + x_pad)
    if overlap_min_x < overlap_max_x:
        ax_xproj.add_patch(Rectangle((overlap_min_x, overlap_y), overlap_max_x - overlap_min_x, proj_height, facecolor=OVERLAP_COLOR, edgecolor='black', alpha=0.95, linewidth=1.0, label='Overlap'))
    ax_xproj.set_ylim(-0.72, 0.62)
    ax_xproj.set_yticks([])
    ax_xproj.set_title("X-axis Projections", fontsize=9)
    ax_xproj.set_xlabel("X-axis scalar value", fontsize=8)
    ax_xproj.set_xlim(x_min, x_max)
    ax_xproj.legend(loc='upper right', fontsize=7)

    # --- Y-axis projection ---
    min1y, max1y = project_onto_axis(rect1_verts, np.array([0, 1]))
    min2y, max2y = project_onto_axis(rect2_verts, np.array([0, 1]))
    y_pad = max((y_max - y_min) * 0.02, 0.15)
    ax_yproj.add_patch(Rectangle((min1y - y_pad, top_y), (max1y - min1y) + 2 * y_pad, proj_height, facecolor=RECT1_COLOR, edgecolor='black', alpha=0.9, linewidth=1.0, label='Rect 1'))
    ax_yproj.add_patch(Rectangle((min2y - y_pad, bottom_y), (max2y - min2y) + 2 * y_pad, proj_height, facecolor=RECT2_COLOR, edgecolor='black', alpha=0.9, linewidth=1.0, label='Rect 2'))
    overlap_min_y = max(min1y - y_pad, min2y - y_pad)
    overlap_max_y = min(max1y + y_pad, max2y + y_pad)
    if overlap_min_y < overlap_max_y:
        ax_yproj.add_patch(Rectangle((overlap_min_y, overlap_y), overlap_max_y - overlap_min_y, proj_height, facecolor=OVERLAP_COLOR, edgecolor='black', alpha=0.95, linewidth=1.0, label='Overlap'))
    ax_yproj.set_ylim(-0.72, 0.62)
    ax_yproj.set_yticks([])
    ax_yproj.set_title("Y-axis Projections", fontsize=9)
    ax_yproj.set_xlabel("Y-axis scalar value", fontsize=8)
    ax_yproj.set_xlim(y_min, y_max)
    ax_yproj.legend(loc='upper right', fontsize=7)

    fig.suptitle(f"SAT Example: {title_suffix}", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return (x_min, x_max), (y_min, y_max)


rect_width = 4
rect_height = 2
rect_angle1 = 20
rect_angle2 = -15


if __name__ == "__main__":
    rect1_no_overlap = rotate_rectangle(center_x=3, center_y=5, width=rect_width, height=rect_height, angle_degrees=rect_angle1)
    rect2_no_overlap = rotate_rectangle(center_x=9, center_y=5, width=rect_width, height=rect_height, angle_degrees=rect_angle2)
    xlim1, ylim1 = generate_figure_2d_1d(rect1_no_overlap, rect2_no_overlap, "No Overlap")

    rect1_overlap = rotate_rectangle(center_x=5, center_y=5, width=rect_width, height=rect_height, angle_degrees=rect_angle1)
    rect2_overlap = rotate_rectangle(center_x=6.5, center_y=5, width=rect_width, height=rect_height, angle_degrees=rect_angle2)
    generate_figure_2d_1d(rect1_overlap, rect2_overlap, "Overlap", xlim_fixed=xlim1, ylim_fixed=ylim1)