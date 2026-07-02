from pathlib import Path
import subprocess
import json
import sys
import os
import time
from datetime import datetime
import argparse


JSON_OUTPUT_DIR = "../solutions/json"        
REPORTS_OUTPUT_DIR = "../solutions/reports"  
RUNTIME_REPORT_FILE = "runtime_reports.txt"


def run_minizinc(model_file, data_file, solver='gecode', timeout=None, all_solutions=True):
    """
    Runs a MiniZinc model with the specified data.
    
    Args:
        model_file (str): Path to the model file .mzn
        data_file (str): Path to the data file .dzn
        solver (str): Solver to use (default: gecode)
        timeout (int): Timeout in seconds (optional)
        all_solutions (bool): If True, uses -a flag to get all solutions (default: True)
        
    Returns:
        dict: Parsed output with data, runtime and statistics
    """
    cmd = ['minizinc', '--solver', solver, '--statistics']
    if all_solutions:
        cmd.append('-a')
    cmd.extend([model_file, data_file])
    
    if timeout:
        cmd.extend(['--time-limit', str(timeout * 1000)])  # MiniZinc uses milliseconds
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        if timeout:
            print(f"MiniZinc timeout: {timeout}s")
        print("Running... (Press Ctrl+C to cancel)")
        
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        stdout_lines = []
        stderr_lines = []
        
        try:
            for line in process.stdout:
                stdout_lines.append(line)
            stderr_lines = process.stderr.readlines()

            process.wait()

            elapsed = time.time() - start_time
            stdout_text = ''.join(stdout_lines)
            stderr_text = ''.join(stderr_lines)
            parsed_output = parse_output(stdout_text)
            if parsed_output is not None:
                parsed_output['wall_time'] = elapsed

            if process.returncode != 0:
                if parsed_output is not None:
                    print(f"MiniZinc exited with code {process.returncode}; returning last parsed solution.", file=sys.stderr)
                    return parsed_output
                print(f"Execution error (code {process.returncode}): {stderr_text}", file=sys.stderr)
                return None
                
            return parsed_output

        except KeyboardInterrupt:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                out_rest, err_rest = process.communicate(timeout=5)
            except Exception:
                out_rest, err_rest = ('', '')

            elapsed = time.time() - start_time
            stdout_text = ''.join(stdout_lines) + out_rest
            stderr_text = ''.join(stderr_lines) + err_rest
            parsed_output = parse_output(stdout_text)
            if parsed_output is not None:
                parsed_output['wall_time'] = elapsed

            if parsed_output is not None:
                print("Received interrupt; returning last parsed solution.", file=sys.stderr)
                return parsed_output

            print("Received interrupt; no parseable solution found.", file=sys.stderr)
            return None
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user (Ctrl+C)", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: MiniZinc not found. Make sure it is installed and in the PATH.", file=sys.stderr)
        return None


def parse_output(output_text):
    """
    Parse MiniZinc output to extract JSON data and runtime.
    Handles multiple solutions and checks for optimality (========== separator).
    
    Args:
        output_text (str): Raw output from MiniZinc
        
    Returns:
        dict: Parsed data with 'data', 'runtime', 'initTime', 'solveTime' and 'is_optimal' or None if error
    """
    lines = output_text.strip().split('\n')
    
    is_optimal = any('==========' in line for line in lines)
    
    stats_lines = [line for line in lines if '%%%mzn-stat' in line]
    
    json_objects = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('{'):
            brace_count = line.count('{') - line.count('}')
            json_start = i
            
            if brace_count == 0:
                try:
                    data = json.loads(line)
                    json_objects.append((json_start, i, data))
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
            
            json_text = '\n'.join(json_lines)
            try:
                data = json.loads(json_text)
                json_objects.append((json_start, j - 1, data))
            except json.JSONDecodeError:
                pass
            
            i = j
        else:
            i += 1
    
    if not json_objects:
        print(f"Error: No JSON found in output", file=sys.stderr)
        print(f"Output received (first 500 chars): {output_text[:500]}", file=sys.stderr)
        return None
    
    _, _, data = json_objects[-1]
    
    runtime = None
    init_time = None
    solve_time = None
    
    for line in reversed(stats_lines):
        if '%%%mzn-stat: solveTime=' in line:
            parts = line.split('solveTime=')
            if len(parts) > 1:
                val = parts[1].strip()
                try:
                    solve_time = float(val)
                    runtime = solve_time
                except ValueError:
                    solve_time = None
        # Check for initTime (used by CP solvers like chuffed, gecode, cp-sat)
        if '%%%mzn-stat: initTime=' in line and init_time is None:
            parts = line.split('initTime=')
            if len(parts) > 1:
                val = parts[1].strip()
                try:
                    init_time = float(val)
                except ValueError:
                    init_time = None
        # Check for flatTime (used by MIP solvers like gurobi)
        if '%%%mzn-stat: flatTime=' in line and init_time is None:
            parts = line.split('flatTime=')
            if len(parts) > 1:
                val = parts[1].strip()
                try:
                    init_time = float(val)
                except ValueError:
                    init_time = None
    
    return {'data': data, 'runtime': runtime, 'initTime': init_time, 'solveTime': solve_time, 'is_optimal': is_optimal}



def append_to_log(log_file, model_name, instance_name, objective_value, runtime=None, init_time=None, solve_time=None, solver=None, timestamp=None):
    """
    Append results to log file in fixed-width column format.
    
    Args:
        log_file (str): Path to log file
        model_name (str): Model name
        instance_name (str): Instance name
        objective_value (float): Objective value
        runtime (str): Execution time (optional)
        init_time (float): Initialization time (optional)
        solve_time (float): Solve time (optional)
        solver (str): Solver name used (optional)
        timestamp (str): Timestamp (optional)
    """
    file_exists = os.path.exists(log_file)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_file, 'a', encoding='utf-8') as f:
        if not file_exists:
            header = f"{'Timestamp':<19} {'Model':<20} {'Instance':<20} {'Solver':<10} {'InitTime (s)':<15} {'SolveTime (s)':<15} {'Runtime (s)':<15} {'Objective':<15}\n"
            separator = "-" * 129 + "\n"
            f.write(header)
            f.write(separator)

        rt = "N/A" if runtime is None else f"{runtime:.6f}"
        it = "N/A" if init_time is None else f"{init_time:.6f}"
        st = "N/A" if solve_time is None else f"{solve_time:.6f}"
        obj = f"{objective_value:.6f}" if isinstance(objective_value, float) else str(objective_value)
        row = f"{timestamp:<19} {model_name:<20} {instance_name:<20} {solver:<10} {it:<15} {st:<15} {rt:<15} {obj:<15}\n"
        f.write(row)
    
    print(f"Result saved to {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Runs a MiniZinc model and saves objective results to a log file.'
    )
    parser.add_argument('model', help='MiniZinc model file (.mzn)')
    parser.add_argument('data', help='MiniZinc data file (.dzn)')
    parser.add_argument('-o', '--output', default=None, 
                        help='Output file for results (default: {REPORTS_OUTPUT_DIR}/{RUNTIME_REPORT_FILE})')
    parser.add_argument('-s', '--solver', default='gecode',
                        help='Solver to use (default: gecode)')
    parser.add_argument('-t', '--timeout', type=int,
                        help='Timeout in seconds')
    parser.add_argument('--model-name', 
                        help='Custom model name (default: file name)')
    parser.add_argument('--instance-name',
                        help='Custom instance name (default: file name)')
    parser.add_argument('--no-all-solutions', action='store_true',
                        help='Do not use -a flag (may help with solvers that produce huge outputs)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(REPORTS_OUTPUT_DIR, RUNTIME_REPORT_FILE)
    
    model_name = args.model_name if args.model_name else os.path.splitext(os.path.basename(args.model))[0]
    instance_name = args.instance_name if args.instance_name else os.path.splitext(os.path.basename(args.data))[0]
    
    print(f"Model: {model_name}")
    print(f"Instance: {instance_name}")
    print("-" * 60)
    
    all_solutions = not args.no_all_solutions
    result = run_minizinc(args.model, args.data, args.solver, args.timeout, all_solutions)
    
    # Handle case where no solution was found
    if result is None:
        print("[!] No solution found (UNSATISFIABLE or timeout with no feasible solution)")
        
        # Log to report with "--" for objective
        wall_time = args.timeout if args.timeout else None
        Path(REPORTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        append_to_log(args.output, model_name, instance_name, "--", runtime=wall_time, init_time=None, solve_time=None, solver=args.solver)
        
        print("-" * 60)
        print(f"[!] Objective: --")
        print(f"[!] Init Time (s): N/A")
        print(f"[!] Solve Time (s): N/A")
        print(f"[!] Runtime (s): {wall_time if wall_time else 'N/A'}")
        print(f"[!] Solver: {args.solver}")
        print(f"[!] Solution: NONE (UNSATISFIABLE)")
        print(f"[!] Report saved to: {args.output}")
        print("[!] Continuing with next instance...")
        sys.exit(0)  # Clean exit for no-solution case
    
    file_name = model_name + "_" + instance_name + "_" + args.solver + ".json"
    output_file = Path(JSON_OUTPUT_DIR) / file_name
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result["data"], f, indent=4)
    
    objective_value = result['data'].get('objective_value', None)
    wall_time = result.get('wall_time', None)
    stats_runtime = result.get('runtime', None)
    init_time = result.get('initTime', None)
    solve_time = result.get('solveTime', None)
    is_optimal = result.get('is_optimal', False)

    if is_optimal:
        runtime = wall_time if wall_time is not None else stats_runtime
    else:
        runtime = args.timeout if args.timeout else wall_time if wall_time is not None else stats_runtime
    
    if objective_value is None:
        print("Warning: 'objective_value' not found in output.", file=sys.stderr)
        print(f"Available keys: {list(result['data'].keys())}")
        
        for key in ['obj', 'cost', 'value', 'total']:
            if key in result['data']:
                objective_value = result['data'][key]
                print(f"Using '{key}' as objective value: {objective_value}")
                break
    
    if objective_value is None:
        print("Unable to find objective value.")
        # Log as no-solution instead of exiting
        Path(REPORTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        append_to_log(args.output, model_name, instance_name, "--", runtime=args.timeout, init_time=None, solve_time=None, solver=args.solver)
        print(f"[!] Report saved to: {args.output}")
        sys.exit(0)  # Clean exit for objective parsing failure
    
    Path(REPORTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    append_to_log(args.output, model_name, instance_name, objective_value, runtime=runtime, init_time=init_time, solve_time=solve_time, solver=args.solver)
    
    print("-" * 60)
    print(f"[OK] Objective: {objective_value}")
    if init_time is not None:
        print(f"[OK] Init Time (s): {init_time:.6f}")
    else:
        print(f"[OK] Init Time (s): N/A")
    if solve_time is not None:
        print(f"[OK] Solve Time (s): {solve_time:.6f}")
    else:
        print(f"[OK] Solve Time (s): N/A")
    if runtime is not None:
        print(f"[OK] Runtime (s): {runtime}")
    else:
        print(f"[OK] Runtime (s): N/A")
    print(f"[OK] Solver: {args.solver}")
    if is_optimal:
        print(f"[OK] Solution: OPTIMAL")
    else:
        print(f"[OK] Solution: FEASIBLE (not proven optimal)")
    print(f"[OK] Saved JSON to: {output_file}")
    print(f"[OK] Saved report to: {args.output}")
    sys.exit(0)  # Explicit clean exit


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user (Ctrl+C)", file=sys.stderr)
        sys.exit(0)  # Exit gracefully with code 0 to allow other instances to run
    except Exception as e:
        print(f"\n\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(0)  # Exit gracefully to allow shell script to continue