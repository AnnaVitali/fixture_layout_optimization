import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from moments_of_inertia import InertiaAnalysis
import json
import re
from pathlib import Path

BAR_WIDTH = 145
IMAGE_OUTPUT_DIR = Path("../solutions/images")

class ResultDisplayer:
    """
    This class is responsible for displaying the results of the fixture analysis.
    """

    def __init__(self, workpiece_vertices, holes, rectangles, workpiece_name=None):
        """
        Initializes the ResultDisplayer with the vertices of the workpiece.

        Args:
            workpiece_vertices (list of tuple): A list of tuples representing the vertices of the workpiece.
                                                Each tuple contains the x and y coordinates of a vertex.
            workpiece_name (str): Name of the workpiece for filename generation
        """
        self.workpiece_vertices = workpiece_vertices
        self.holes = holes
        self.rectangles = rectangles
        self.workpiece_name = workpiece_name
    
    def _extract_provider(self, file_path):
        """
        Extract solution provider from the file path.
        
        Returns:
            str: provider name (e.g., 'cp_pso', 'mip_model_gurobi', etc.)
        """
        filename = Path(file_path).name
        stem = Path(file_path).stem
        
        if self.workpiece_name:
            idx = stem.find(self.workpiece_name)
            if idx > 0:
                provider = stem[:idx].rstrip("_")
                if provider:
                    solver_match = re.search(r'(chuffed|cp-sat|gecode|gurobi|pso)', stem)
                    if solver_match:
                        solver = solver_match.group(1)
                        if f"_{solver}" not in provider:
                            provider = f"{provider}_{solver}"
                    return provider
        
        if filename.startswith("LNS_"):
            model_match = re.search(r'LNS_(cp|mip)_model', filename)
            model_type = f"LNS_{model_match.group(1)}" if model_match else "LNS_cp"
        else:
            model_match = re.search(r'^(cp|mip)_model', filename)
            model_type = model_match.group(1) + "_model" if model_match else "cp_model"
        
        solver_match = re.search(r'_(chuffed|cp-sat|gecode|gurobi|pso|expert_operator|pso_eo)', filename)
        solver = solver_match.group(1) if solver_match else ""
        
        if solver:
            return f"{model_type}_{solver}"
        return model_type

    def show_results(self, file_path):
        """
        Displays the results of the fixture analysis by reading data from a JSON file and plotting the results.

        Args:
            file_path (str): The path to the JSON file containing the fixture analysis results.

        Raises:
            ValueError: If the total area of the fixtures is zero, making it impossible to compute the center of gravity.
        """
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        x = data["x"]
        y = data["y"]
        x1 = data["x1"]
        y1 = data["y1"]
        x2 = data["x2"]
        y2 = data["y2"]
        x3 = data["x3"]
        y3 = data["y3"]
        angle = data["angle"]
        selected_fixtures = data["selected_fixture"]
        fixture_types = data["fixture_type"]
        fixtures_center_x = data["fixtures_center_x"]
        fixtures_center_y = data["fixtures_center_y"]
        bars_center = data["bars_center"]

        polygons = []

        for idx in range(len(fixture_types)):
            if selected_fixtures[idx] == 1:
                fixture_coords = [(x[idx], y[idx]), (x1[idx], y1[idx]), (x2[idx], y2[idx]), (x3[idx], y3[idx]), (x[idx], y[idx])]
                jx, jy, jxy = InertiaAnalysis.compute_absolute_moments_of_inertia(fixture_coords)

                fixture = {
                    "area": InertiaAnalysis.compute_polygon_area(fixture_coords),
                    "absolute_moments_of_inertia": [jx, jy, jxy],
                    "barycenter": (fixtures_center_x[idx], fixtures_center_y[idx]),
                    "angle": angle[idx],
                    "idx": idx
                }

                jxg, jyg, jxyg = InertiaAnalysis.compute_baricentric_moments_of_inertia(fixture)

                fixture["baricentric_moments_of_inertia"] = [jxg, jyg, jxyg]
                polygons.append(fixture)
        x_g, y_g = InertiaAnalysis.compute_overall_center_of_gravity(polygons)
        j_xg, j_yg = InertiaAnalysis.compute_combined_baricentric_moments_of_inertia(polygons, x_g, y_g)

        print("\n-------------Principal Moments of Inertia-------------\n")
        print(f"Principal Moment I: {j_xg:.5e}")
        print(f"Principal moment J: {j_yg:.5e}")
        print(f"I + J: {abs(j_xg) + abs(j_yg):.5e}")


        plt.figure(figsize=(8, 6))
        x_workpiece, y_workpiece = zip(*self.workpiece_vertices)
        plt.plot(x_workpiece, y_workpiece, '-o', color='black', label="Polygon", linewidth=2)
        plt.scatter(*zip(*self.workpiece_vertices), color='#7a7a7a', zorder=5, label="Vertices")

        unique_centers = np.unique(list(bars_center), axis=0)

        y_min, y_max = plt.ylim()

        for center in unique_centers:
            bar_coords = [
                (center - BAR_WIDTH / 2, y_min),
                (center - BAR_WIDTH / 2, y_max),
                (center + BAR_WIDTH / 2, y_max),
                (center + BAR_WIDTH / 2, y_min),
            ]
            bar_patch = patches.Polygon(bar_coords, closed=True, edgecolor='none', facecolor='lightgray', zorder=1)
            plt.gca().add_patch(bar_patch)

        def _draw_oblique_hatch(rect_patch, x_ll, y_ll, width, height, zorder=2):
            """Draw diagonal dashed lines clipped to a rectangle patch."""
            hatch_step = max(min(width, height) / 8, 8)
            start_offsets = np.arange(-height, width, hatch_step)
            for offset in start_offsets:
                x_start = x_ll + offset
                y_start = y_ll
                x_end = x_start + height
                y_end = y_ll + height

                hatch_line = plt.Line2D(
                    [x_start, x_end],
                    [y_start, y_end],
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    zorder=zorder,
                    clip_on=True,
                )
                hatch_line.set_clip_path(rect_patch.get_path(), rect_patch.get_transform())
                plt.gca().add_line(hatch_line)

        for x_hole, y_hole, radius in self.holes:
            theta = np.linspace(0, 2 * np.pi, 500)
            circle_x = x_hole + radius * np.cos(theta)
            circle_y = y_hole + radius * np.sin(theta)
            # plt.plot(circle_x, circle_y, color="red")

            # circle_patch = patches.Circle((x_hole, y_hole), radius, color="red", alpha=0.3)
            # plt.gca().add_patch(circle_patch)

            inner_side = 2 * radius
            inner_ll = (x_hole - radius, y_hole - radius)
            inner_square = patches.Rectangle(inner_ll, inner_side, inner_side,
                                             edgecolor='black', facecolor='none',
                                             linestyle='--', linewidth=1.0, zorder=3)
            plt.gca().add_patch(inner_square)
            _draw_oblique_hatch(inner_square, inner_ll[0], inner_ll[1], inner_side, inner_side, zorder=2)


        for x_ll, y_ll, width, height in self.rectangles:
            rect_outline = patches.Rectangle(
                (x_ll, y_ll),
                width,
                height,
                linewidth=1.2,
                edgecolor="black",
                facecolor="none",
                linestyle="--"
            )
            plt.gca().add_patch(rect_outline)
            _draw_oblique_hatch(rect_outline, x_ll, y_ll, width, height, zorder=2)

        for idx, fixture in enumerate(polygons):
            fixture_coords = [(x[idx], y[idx]), (x1[idx], y1[idx]), (x2[idx], y2[idx]), (x3[idx], y3[idx]), (x[idx], y[idx])]
            patch = patches.Polygon(fixture_coords, closed=True, edgecolor='black', facecolor='none', hatch='/')
            plt.gca().add_patch(patch)
            barycenter_x, barycenter_y = fixture["barycenter"]
            plt.scatter(barycenter_x, barycenter_y, color='green', zorder=5)
            plt.text(barycenter_x + 5, barycenter_y + 5, f"c{idx + 1}", color="green", fontsize=10)

        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.grid(True)
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        
        provider = self._extract_provider(file_path)
        
        filename = f"{self.workpiece_name}_{provider}.png"
        output_dir = Path(IMAGE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / filename)