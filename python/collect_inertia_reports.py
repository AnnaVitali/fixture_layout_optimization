import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
PLOT_SCRIPT = PYTHON_DIR / "plot_result.py"
SOLUTIONS_DIR = PROJECT_ROOT / "solutions" / "json_roberto_model"
REPORTS_DIR = PROJECT_ROOT / "solutions" / "reports"
REPORT_FILE = REPORTS_DIR / "inertia_roberto_reports.txt"
WORKPIECES_PATH = PYTHON_DIR / "resources" / "workpieces_information.json"

I_J_PATTERN = re.compile(r"I \+ J:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")
SOLVER_PATTERN = re.compile(r"(chuffed|cp-sat|gecode|gurobi|pso)")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_plot_and_extract_sum(workpiece_name: str, solution_path: Path) -> float:
    cmd = [sys.executable, str(PLOT_SCRIPT), workpiece_name, str(solution_path)]
    result = subprocess.run(
        cmd,
        cwd=str(PYTHON_DIR),
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr if stderr else stdout
        raise RuntimeError(f"plot_result failed: {message}")

    match = I_J_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError("Unable to parse I + J from plot_result output.")

    return float(match.group(1))


def append_to_log(log_file: Path, provider: str, instance: str, objective_value: float):
    file_exists = log_file.exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with log_file.open("a", encoding="utf-8") as fh:
        if not file_exists:
            header = (
                f"{'Timestamp':<19} {'Solution Provider':<20} {'Instance':<20} {'Objective':<15}\n"
            )
            separator = "-" * 76 + "\n"
            fh.write(header)
            fh.write(separator)

        obj = f"{objective_value:.6f}" if isinstance(objective_value, float) else str(objective_value)
        row = f"{timestamp:<19} {provider:<20} {instance:<20} {obj:<15}\n"
        fh.write(row)


def resolve_workpiece_name(filename: str, workpiece_names: list) -> str:
    stem = Path(filename).stem
    for name in sorted(workpiece_names, key=len, reverse=True):
        if name in stem:
            return name
    return ""


def resolve_provider(filename: str, workpiece_name: str) -> str:
    stem = Path(filename).stem
    idx = stem.find(workpiece_name)
    if idx == -1:
        return "unknown"

    provider = stem[:idx].rstrip("_")
    if not provider:
        provider = "unknown"

    solver_match = SOLVER_PATTERN.search(stem)
    if solver_match:
        solver = solver_match.group(1)
        if f"_{solver}" not in provider:
            provider = f"{provider}_{solver}"

    return provider


def main():
    if not WORKPIECES_PATH.exists():
        raise FileNotFoundError(f"Workpieces file not found: {WORKPIECES_PATH}")

    workpieces = load_json(WORKPIECES_PATH)
    workpiece_names = list(workpieces.keys())

    if not SOLUTIONS_DIR.exists():
        raise FileNotFoundError(f"Solutions directory not found: {SOLUTIONS_DIR}")

    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"plot_result.py not found: {PLOT_SCRIPT}")

    solution_files = sorted(SOLUTIONS_DIR.glob("*.json"))
    if not solution_files:
        print(f"No solution files found in {SOLUTIONS_DIR}")
        return

    for solution_path in solution_files:
        workpiece_name = resolve_workpiece_name(solution_path.name, workpiece_names)
        if not workpiece_name:
            print(f"Skipping (unknown workpiece): {solution_path.name}")
            continue

        provider = resolve_provider(solution_path.name, workpiece_name)

        try:
            objective_value = run_plot_and_extract_sum(workpiece_name, solution_path)
        except Exception as exc:
            print(f"Failed to compute inertia for {solution_path.name}: {exc}")
            continue

        append_to_log(REPORT_FILE, provider, workpiece_name, objective_value)


if __name__ == "__main__":
    main()
