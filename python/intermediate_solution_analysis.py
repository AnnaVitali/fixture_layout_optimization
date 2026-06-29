from pathlib import Path
import subprocess
import json
import sys
import re
from datetime import datetime
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
PLOT_SCRIPT = PYTHON_DIR / "plot_result.py"

JSON_OUTPUT_DIR = PROJECT_ROOT / "correlation_analysis" / "json"
REPORTS_OUTPUT_DIR = PROJECT_ROOT / "correlation_analysis" / "reports"
RUNTIME_REPORT_FILE = "objective_inertia_reports.txt"

I_J_PATTERN = re.compile(r"I \+ J:\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")


def run_minizinc(model_file, data_file, solver='gecode', timeout=None, all_solutions=True):
    cmd = ['minizinc', '--solver', solver, '--statistics']
    if all_solutions:
        cmd.append('-a')
    cmd.extend([model_file, data_file])

    if timeout:
        cmd.extend(['--time-limit', str(timeout * 1000)])

    print(f"Executing: {' '.join(cmd)}")
    if timeout:
        print(f"MiniZinc timeout: {timeout}s")
    print("Running... (Press Ctrl+C to cancel)")

    process = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    return {
        'returncode': process.returncode,
        'stdout': process.stdout,
        'stderr': process.stderr,
    }


def extract_json_objects(output_text):
    lines = output_text.splitlines()
    json_objects = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith('{'):
            i += 1
            continue

        brace_count = line.count('{') - line.count('}')
        if brace_count == 0:
            try:
                json_objects.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            i += 1
            continue

        json_lines = [line]
        j = i + 1
        while j < len(lines) and brace_count > 0:
            next_line = lines[j]
            json_lines.append(next_line)
            brace_count += next_line.count('{') - next_line.count('}')
            j += 1

        if brace_count == 0:
            json_text = '\n'.join(json_lines)
            try:
                json_objects.append(json.loads(json_text))
            except json.JSONDecodeError:
                pass

        i = j

    return json_objects


def extract_objective(solution_data):
    objective_value = solution_data.get('objective_value', None)
    if objective_value is not None:
        return objective_value

    for key in ['obj', 'cost', 'value', 'total']:
        if key in solution_data:
            return solution_data[key]

    return None


def run_plot_and_extract_inertia(instance_name: str, solution_path: Path) -> float:
    cmd = [sys.executable, str(PLOT_SCRIPT), instance_name, str(solution_path)]
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


def append_to_log(log_file: Path, model_name: str, instance_name: str, solver: str, objective_value, inertia_value):
    file_exists = log_file.exists()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with log_file.open('a', encoding='utf-8') as f:
        if not file_exists:
            header = (
                f"{'Timestamp':<19} {'Model':<20} {'Instance':<20} "
                f"{'Solver':<10} {'Objective':<12} {'Moment of inertia':<20}\n"
            )
            separator = "-" * 105 + "\n"
            f.write(header)
            f.write(separator)

        obj = f"{float(objective_value):.6f}" if isinstance(objective_value, (int, float)) else str(objective_value)
        inertia = f"{float(inertia_value):.6f}" if isinstance(inertia_value, (int, float)) else str(inertia_value)
        row = (
            f"{timestamp:<19} {model_name:<20} {instance_name:<20} "
            f"{solver:<10} {obj:<12} {inertia:<20}\n"
        )
        f.write(row)


def main():
    parser = argparse.ArgumentParser(
        description='Run a MiniZinc model, save every intermediate solution, and log objective + moment of inertia.'
    )
    parser.add_argument('model', help='MiniZinc model file (.mzn)')
    parser.add_argument('data', help='MiniZinc data file (.dzn)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output report file (default: intermediate_solutions/reports/objective_inertia_reports.txt)')
    parser.add_argument('-s', '--solver', default='gecode',
                        help='Solver to use (default: gecode)')
    parser.add_argument('-t', '--timeout', type=int,
                        help='Timeout in seconds')
    parser.add_argument('--model-name',
                        help='Custom model name (default: file name)')
    parser.add_argument('--instance-name',
                        help='Custom instance name (default: file name)')
    parser.add_argument('--no-all-solutions', action='store_true',
                        help='Do not use -a flag (not recommended for intermediate analysis)')

    args = parser.parse_args()

    report_file = Path(args.output) if args.output else REPORTS_OUTPUT_DIR / RUNTIME_REPORT_FILE
    model_name = args.model_name if args.model_name else Path(args.model).stem
    instance_name = args.instance_name if args.instance_name else Path(args.data).stem
    all_solutions = not args.no_all_solutions

    print(f"Model: {model_name}")
    print(f"Instance: {instance_name}")
    print(f"Solver: {args.solver}")
    print('-' * 60)

    result = run_minizinc(args.model, args.data, args.solver, args.timeout, all_solutions)
    json_objects = extract_json_objects(result['stdout'])

    if not json_objects:
        print("Execution failed: no JSON solutions found.", file=sys.stderr)
        if result['stderr'].strip():
            print(result['stderr'].strip(), file=sys.stderr)
        sys.exit(1)

    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    logged_count = 0
    base_name = f"{model_name}_{instance_name}_{args.solver}"

    for index, solution_data in enumerate(json_objects, start=1):
        file_name = f"{base_name}_sol_{index:04d}.json"
        output_file = JSON_OUTPUT_DIR / file_name

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(solution_data, f, indent=4)
        saved_count += 1

        objective_value = extract_objective(solution_data)
        if objective_value is None:
            print(f"Skipping report row for {file_name}: objective not found.", file=sys.stderr)
            continue

        try:
            inertia_value = run_plot_and_extract_inertia(instance_name, output_file)
        except Exception as exc:
            print(f"Skipping report row for {file_name}: {exc}", file=sys.stderr)
            continue

        append_to_log(
            report_file,
            model_name=model_name,
            instance_name=instance_name,
            solver=args.solver,
            objective_value=objective_value,
            inertia_value=inertia_value,
        )
        logged_count += 1

    if result['returncode'] != 0:
        print(f"MiniZinc exited with code {result['returncode']}; saved parseable solutions anyway.", file=sys.stderr)

    print('-' * 60)
    print(f"[OK] Saved intermediate JSON solutions: {saved_count}")
    print(f"[OK] Logged objective + inertia rows: {logged_count}")
    print(f"[OK] JSON directory: {JSON_OUTPUT_DIR}")
    print(f"[OK] Report file: {report_file}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user (Ctrl+C)", file=sys.stderr)
        sys.exit(130)