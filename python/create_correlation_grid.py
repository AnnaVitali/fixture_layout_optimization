from pathlib import Path
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREA_DIR = PROJECT_ROOT / "correlation_analysis" / "images" / "fdist_area_inertia_correlation"
DEFAULT_SEMI_PERIMETER_DIR = PROJECT_ROOT / "correlation_analysis" / "images" / "fdist_semi_perimeter_inertia_correlation"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "correlation_analysis" / "images" / "combined_correlation_grid.png"


def create_grid_image(area_dir: Path, semi_perimeter_dir: Path, output_path: Path):
    area_dir = Path(area_dir)
    semi_perimeter_dir = Path(semi_perimeter_dir)
    
    if not area_dir.exists():
        raise FileNotFoundError(f"Area directory not found: {area_dir}")
    if not semi_perimeter_dir.exists():
        raise FileNotFoundError(f"Semi-perimeter directory not found: {semi_perimeter_dir}")
    
    order = ["spiral_stair_step", "simple_stair_step", "dashboard", "speaker", "coffee_table"]
    
    area_images = {f.stem: f for f in area_dir.glob("*.png")}
    semi_perimeter_images = {f.stem: f for f in semi_perimeter_dir.glob("*.png")}
    
    common_names = set(area_images.keys()) & set(semi_perimeter_images.keys())
    
    def sort_key(name):
        for i, key in enumerate(order):
            if key in name:
                return i
        return len(order)
    
    sorted_names = sorted(common_names, key=sort_key)
    
    if not sorted_names:
        raise FileNotFoundError("No matching images found in both directories")
    
    cols = 2
    rows = len(sorted_names)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    
    if rows == 1:
        axes = axes.reshape(1, cols)
    
    for row, name in enumerate(sorted_names):
        ax = axes[row, 0]
        img = mpimg.imread(area_images[name])
        ax.imshow(img)
        ax.axis("off")
    
    for row, name in enumerate(sorted_names):
        ax = axes[row, 1]
        img = mpimg.imread(semi_perimeter_images[name])
        ax.imshow(img)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.05, wspace=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2-row grid image combining area and semi-perimeter correlation plots."
    )
    parser.add_argument(
        "--area-dir",
        default=str(DEFAULT_AREA_DIR),
        help="Directory containing area correlation plot images",
    )
    parser.add_argument(
        "--semi-perimeter-dir",
        default=str(DEFAULT_SEMI_PERIMETER_DIR),
        help="Directory containing semi-perimeter correlation plot images",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the grid image is saved",
    )
    args = parser.parse_args()

    area_dir = Path(args.area_dir)
    semi_perimeter_dir = Path(args.semi_perimeter_dir)
    output_path = Path(args.output)

    grid_path = create_grid_image(area_dir, semi_perimeter_dir, output_path)
    print(f"Grid image created: {grid_path}")


if __name__ == "__main__":
    main()
