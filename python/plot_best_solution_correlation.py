from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "correlation_analysis" / "reports" / "fdist_semi_perimeter_inertia_reports.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "correlation_analysis" / "images" / "best_solution_correlation"


def parse_report_last_solution(report_path: Path):
    """
    Parse report and keep only the last solution for each Instance-Model-Solver combination.
    Returns dict: {instance: [(objective, inertia, model, solver), ...]}
    """
    data_by_key = {}

    with report_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Timestamp") or set(line) == {"-"}:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            model = parts[2]
            instance = parts[3]
            solver = parts[4]

            try:
                objective = float(parts[5])
                inertia = float(parts[6])
            except ValueError:
                continue

            key = (instance, model, solver)
            data_by_key[key] = (objective, inertia, model, solver)

    by_instance = {}
    for (instance, model, solver), (objective, inertia, model_out, solver_out) in data_by_key.items():
        by_instance.setdefault(instance, [])
        by_instance[instance].append((objective, inertia, model, solver))

    return by_instance


def filter_selected_combinations(by_instance):
    """
    Keep only the 5 selected model-solver combinations per instance.
    But if an instance doesn't have all 5, keep whatever it does have.
    Selected: cp_model+gecode, cp_model+chuffed, cp_model+cp-sat, LNS_cp_model+gecode, mip_model+gurobi
    """
    selected = [
        ("cp_model", "gecode"),
        ("cp_model", "chuffed"),
        ("cp_model", "cp-sat"),
        ("LNS_cp_model", "gecode"),
        ("mip_model", "gurobi"),
    ]

    filtered = {}
    for instance, solutions in by_instance.items():
        selected_solutions = []
        for obj, inertia, model, solver in solutions:
            if (model, solver) in selected:
                selected_solutions.append((obj, inertia, model, solver))
        
        if selected_solutions:
            filtered[instance] = selected_solutions

    return filtered


def pearson_correlation(x_values, y_values):
    n = len(x_values)
    if n < 2:
        return float("nan")
    print(f"Computing correlation for {n} data points")
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    
    print(f"Mean x: {mean_x}, Mean y: {mean_y}")

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for x_val, y_val in zip(x_values, y_values):
        dx = x_val - mean_x
        dy = y_val - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    print(f"Covariance: {cov}, Variance x: {var_x}, Variance y: {var_y}")
    if var_x == 0.0 or var_y == 0.0:
        return float("nan")

    return cov / math.sqrt(var_x * var_y)


def save_plot(instance, solutions, output_dir: Path):
    name_mapping = {
        "coffee_table": "Coffee table",
        "dashboard": "Dashboard",
        "simple_stair_step": "Stair step simple staircase",
        "speaker": "Speaker",
        "spiral_stair_step": "Stair step spiral staircase",
    }

    display_name = name_mapping.get(instance, instance)

    # Define colors for each model-solver combination
    color_map = {
        ("cp_model", "gecode"): "#1f77b4",
        ("cp_model", "chuffed"): "#ff7f0e",
        ("cp_model", "cp-sat"): "#2ca02c",
        ("LNS_cp_model", "gecode"): "#d62728",
        ("mip_model", "gurobi"): "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute Pearson correlation from all solutions
    all_objectives = [s[0] for s in solutions]
    all_inertias = [s[1] for s in solutions]
    pearson_r = pearson_correlation(all_objectives, all_inertias)
    
    # Group solutions by model-solver combination
    by_combo = {}
    for obj, inertia, model, solver in solutions:
        combo = (model, solver)
        if combo not in by_combo:
            by_combo[combo] = ([], [])
        by_combo[combo][0].append(obj)
        by_combo[combo][1].append(inertia)
    
    # Plot each combination with a different color
    for (model, solver), (objectives, inertias) in by_combo.items():
        label = f"{model} + {solver}"
        color = color_map.get((model, solver), "#000000")
        ax.scatter(objectives, inertias, alpha=0.75, s=100, label=label, color=color)
    
    ax.set_xlabel("Objective (Area)")
    ax.set_ylabel("Moment of inertia")
    r_text = f"{pearson_r:.6f}" if not math.isnan(pearson_r) else "N/A"
    ax.set_title(f"{display_name} | Pearson r = {r_text}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{instance}_best_solution_correlation.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return image_path, pearson_r


def main():
    parser = argparse.ArgumentParser(
        description="Plot best solution correlation (5 model-solver pairs) and compute Pearson correlation."
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Path to report file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plot images are saved",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    output_dir = Path(args.output_dir)

    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    by_instance = parse_report_last_solution(report_path)
    filtered = filter_selected_combinations(by_instance)

    if not filtered:
        print("No instances found with any of the selected model-solver combinations.")
        return

    print("Instance\tPearson_r\tCombinations\tPlot")
    for instance in sorted(filtered.keys()):
        solutions = filtered[instance]
        image_path, pearson_r = save_plot(instance, solutions, output_dir)

        combo_count = len(solutions)
        r_text = f"{pearson_r:.6f}" if not math.isnan(pearson_r) else "N/A"
        print(f"{instance}\t{r_text}\t{combo_count}\t{image_path}")


if __name__ == "__main__":
    main()
