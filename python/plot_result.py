import os
import sys
import json
from typing import Tuple, List, Any
from result_displayer import ResultDisplayer


def load_json(path: str) -> dict:
    """
    Load JSON data from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_geometry(workpiece_info_path: str, workpiece_name: str
                     ) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Load geometry from workpiece_information.json
    Returns (vertices, holes, rectangles)
    """
    data = load_json(workpiece_info_path)

    if workpiece_name not in data:
        raise KeyError(
            f"Workpiece '{workpiece_name}' not found in {workpiece_info_path}"
        )

    entry = data[workpiece_name]

    vertices = entry.get("vertices", [])
    holes = entry.get("holes", [])
    rectangles = entry.get("rectangles", [])

    return vertices, holes, rectangles


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python plot_results.py"
              "<workpiece_name> <solution_json_path>")
        sys.exit(1)

    print(os.getcwd())
    workpiece_info_path = "./resources/workpieces_information.json"
    workpiece_name = sys.argv[1]
    solution_file_path = sys.argv[2]

    vertices, holes, rectangles = resolve_geometry(
        workpiece_info_path,
        workpiece_name
    )
    
    solution_displayer = ResultDisplayer(
        workpiece_vertices=vertices,
        holes=holes,
        rectangles=rectangles,
        workpiece_name=workpiece_name
    )

    solution_displayer.show_results(solution_file_path)


if __name__ == "__main__":
    main()