"""
Unified script to generate all LaTeX tables:
1. Simple Comparison Table (CP, MIP, RL without images)
2. Detailed Comparison Table (with images)
3. Objective Table ($\fdist$ and $\finer$)
4. Runtime Table (solve times)
"""

from pathlib import Path
from collections import defaultdict
import shutil


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_inertia_reports(filepath):
    """Parse inertia_reports.txt to extract objective values."""
    data = defaultdict(lambda: defaultdict(lambda: None))
    provider_map = defaultdict(lambda: defaultdict(lambda: None))  # Track full provider names
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Timestamp') or line.startswith('-'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            provider = parts[2]
            instance = parts[3]
            objective_str = parts[4]
            
            try:
                objective = float(objective_str)
            except ValueError:
                continue
            
            solver_key = None
            if provider == 'cp_model_gecode':
                solver_key = 'gecode'
            elif provider == 'LNS_cp_model_gecode':
                solver_key = 'lns'
            elif provider == 'cp_model_chuffed':
                solver_key = 'chuffed'
            elif provider == 'cp_model_cp-sat':
                solver_key = 'or-tools'
            elif provider == 'mip_model_gurobi':
                solver_key = 'gurobi'
            elif provider == 'rl':
                solver_key = 'rl'
            elif provider == 'expert_operator':
                solver_key = 'expert'
            elif provider.startswith('pso_'):
                # PSO providers: pso_cp_pso, pso_mip_pso, pso_rl_pso, pso_eo_pso
                solver_key = provider
            
            if solver_key:
                data[instance][solver_key] = objective
                provider_map[instance][solver_key] = provider
    
    return data, provider_map


def parse_runtime_reports(filepath):
    """Parse runtime_reports.txt for both objective values and runtime data."""
    fdist_data = defaultdict(lambda: defaultdict(lambda: None))
    timeout_data = defaultdict(lambda: defaultdict(lambda: False))
    solve_time_data = defaultdict(lambda: defaultdict(lambda: None))
    runtime_data = defaultdict(lambda: defaultdict(lambda: None))
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Timestamp') or line.startswith('-'):
                continue
            
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                model = parts[2]
                instance = parts[3]
                solver = parts[4]
                init_time_str = parts[5]
                solve_time_str = parts[6]
                runtime_str = parts[7]
                objective_str = parts[8]
            except IndexError:
                continue
            
            if objective_str == '--' or objective_str == 'N/A':
                continue
            
            # Handle N/A for solve_time
            if solve_time_str == 'N/A':
                solve_time = None
            else:
                try:
                    solve_time = float(solve_time_str)
                except ValueError:
                    solve_time = None
            
            try:
                runtime = float(runtime_str)
                objective = float(objective_str)
            except ValueError:
                continue
            
            solver_key = None
            if model == 'cp_model' and solver == 'gecode':
                solver_key = 'gecode'
            elif model == 'LNS_cp_model' and solver == 'gecode':
                solver_key = 'lns'
            elif model == 'cp_model' and solver == 'chuffed':
                solver_key = 'chuffed'
            elif model == 'cp_model' and solver == 'cp-sat':
                solver_key = 'or-tools'
            elif model == 'mip_model' and solver == 'gurobi':
                solver_key = 'gurobi'
            
            if solver_key:
                if fdist_data[instance][solver_key] is None or objective > fdist_data[instance][solver_key]:
                    fdist_data[instance][solver_key] = objective
                    solve_time_data[instance][solver_key] = solve_time
                    runtime_data[instance][solver_key] = runtime
                    timeout_data[instance][solver_key] = (runtime >= 300.0)
    
    return fdist_data, solve_time_data, timeout_data, runtime_data


# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_objective(value, is_max=False):
    """Format objective value in scientific notation with optional bold."""
    if value is None:
        return "--"
    
    formatted = f"{value:.5e}"
    parts = formatted.split('e')
    mantissa = parts[0]
    exponent = int(parts[1])
    
    result = f"{mantissa}\\times 10^{{{exponent}}}"
    
    if is_max:
        result = f"\\mathbf{{{result}}}"
    
    return result


def format_fdist(value, is_max=False, is_optimal=False):
    """Format fdist value as integer with thousands separators."""
    if value is None:
        return "--"
    
    result = f"{int(round(value)):,}"
    
    if is_optimal:
        result += "{}^*"
    
    if is_max:
        result = f"\\mathbf{{{result}}}"
    
    result = f"${result}$"
    
    return result


def format_finer(value):
    """Format finer value in scientific notation."""
    if value is None:
        return "--"
    
    formatted = f"{value:.5e}"
    parts = formatted.split('e')
    mantissa = parts[0]
    exponent = int(parts[1])
    
    return f"{mantissa}\\times 10^{{{exponent}}}"


def format_solve_time(value):
    """Format solve time with -- for timeouts."""
    if value is None:
        return "--"
    
    return f"{value:.3f}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_max_values(data, workpieces, solvers):
    """Find maximum values for each workpiece."""
    max_vals = {}
    for workpiece in workpieces:
        values = [data[workpiece][solver] for solver in solvers if data[workpiece][solver] is not None]
        if values:
            max_vals[workpiece] = max(values)
        else:
            max_vals[workpiece] = None
    return max_vals


def copy_image_to_paper_dir(workpiece, solver_key):
    """Find and copy image from solutions/images to paper/img."""
    provider_map = {
        'gecode': 'cp_model_gecode',
        'lns': 'LNS_cp_model_gecode',
        'chuffed': 'cp_model_chuffed',
        'or-tools': 'cp_model_cp-sat',
        'gurobi': 'mip_model_gurobi',
        'rl': 'rl',
        'expert': 'expert_operator',
        'pso_cp_pso': 'pso_cp_pso',
        'pso_mip_pso': 'pso_mip_pso',
        'pso_rl_pso': 'pso_rl_pso',
        'pso_eo_pso': 'pso_eo_pso'
    }
    
    provider = provider_map.get(solver_key)
    if not provider:
        return None
    
    solutions_dir = Path(__file__).parent.parent / 'solutions' / 'images'
    image_filename = f"{workpiece}_{provider}.png"
    source_image = solutions_dir / image_filename
    
    if not source_image.exists():
        return None
    
    paper_img_dir = Path(__file__).parent.parent / 'paper' / 'img'
    paper_img_dir.mkdir(parents=True, exist_ok=True)
    
    dest_image = paper_img_dir / image_filename
    shutil.copy2(source_image, dest_image)
    
    return image_filename


# ============================================================================
# TABLE GENERATION FUNCTIONS
# ============================================================================

def generate_simple_comparison_table(data, workpieces, solvers):
    """Generate simple LaTeX table without images."""
    max_vals = find_max_values(data, workpieces, solvers)
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\t\\centering")
    lines.append("\t\\footnotesize")
    lines.append("\t\\caption{Objective values $\\finer$ comparison: CP solvers (best among Gecode, LNS, Chuffed, OR-Tools), MIP solver (Gurobi), and RL approach. For each workpiece, the maximum value is highlighted in bold.}")
    lines.append("\t\\label{tab:comparison_simple}")
    lines.append("\t\\setlength{\\tabcolsep}{5pt}")
    lines.append("\t\\begin{tabular}{cccc}")
    lines.append("\t\t\\toprule")
    lines.append("\t\tWorkpiece & CP & MIP & RL \\\\")
    lines.append("\t\t\\midrule")
    lines.append("")
    
    workpiece_display_names = {
        'spiral_stair_step': 'Stair step\\\\spiral staircase',
        'simple_stair_step': 'Stair step\\\\simple staircase',
        'dashboard': 'Dashboard',
        'speaker': 'Speaker',
        'coffee_table': 'Coffee table',
        'door': 'Door',
        'door_porthole': 'Door\\\\with porthole'
    }
    
    for i, workpiece in enumerate(workpieces):
        display_name = workpiece_display_names.get(workpiece, workpiece)
        
        cp_solvers = ['gecode', 'lns', 'chuffed', 'or-tools']
        cp_values = [data[workpiece][s] for s in cp_solvers if data[workpiece][s] is not None]
        cp_best = max(cp_values) if cp_values else None
        
        mip_best = data[workpiece]['gurobi']
        rl_best = data[workpiece]['rl']
        
        all_values = [v for v in [cp_best, mip_best, rl_best] if v is not None]
        overall_best = max(all_values) if all_values else None
        
        cp_cell = f"${format_objective(cp_best, is_max=(cp_best == overall_best))}$" if cp_best else "--"
        mip_cell = f"${format_objective(mip_best, is_max=(mip_best == overall_best))}$" if mip_best else "--"
        rl_cell = f"${format_objective(rl_best, is_max=(rl_best == overall_best))}$" if rl_best else "--"
        
        row = f"\t\t\\makecell{{{display_name}}} & {cp_cell} & {mip_cell} & {rl_cell} \\\\"
        lines.append(row)
        if i < len(workpieces) - 1:
            lines.append("\t\t\\midrule")
            lines.append("")
    
    lines.append("")
    lines.append("\t\t\\bottomrule")
    lines.append("\t\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_detailed_comparison_table(data, workpieces, solvers):
    """Generate detailed LaTeX table with images."""
    max_vals = find_max_values(data, workpieces, solvers)
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\t\\centering")
    lines.append("\t\\renewcommand{\\arraystretch}{1.3}")
    lines.append("\t\\setlength{\\tabcolsep}{3pt}")
    lines.append("\t\\caption{Objective values $\\finer$ comparison between the best CP model solution, the MIP solution and the Q-learning solution. For each workpiece, the maximum value is highlighted in bold.}")
    lines.append("\t\\label{tab:cp_mip_rl_comparison}")
    lines.append("\t\\begin{tabular}{c c c c}")
    lines.append("\t\t\\hline")
    lines.append("\t\t\\textbf{Workpiece} & \\textbf{CP} & \\textbf{MIP} & \\textbf{Q-learning} \\\\")
    lines.append("\t\t\\hline")
    
    workpiece_display_names = {
        'spiral_stair_step': 'Stair step\\\\spiral staircase',
        'simple_stair_step': 'Stair step\\\\simple staircase',
        'dashboard': 'Dashboard',
        'speaker': 'Speaker',
        'coffee_table': 'Coffee table',
        'door': 'Door',
        'door_porthole': 'Door\\\\with porthole'
    }
    
    for i, workpiece in enumerate(workpieces):
        display_name = workpiece_display_names.get(workpiece, workpiece)
        
        cp_solvers = ['gecode', 'lns', 'chuffed', 'or-tools']
        cp_values = {s: data[workpiece][s] for s in cp_solvers if data[workpiece][s] is not None}
        cp_best = max(cp_values.values()) if cp_values else None
        cp_best_solver = [s for s, v in cp_values.items() if v == cp_best][0] if cp_best else None
        
        mip_best = data[workpiece]['gurobi']
        rl_best = data[workpiece]['rl']
        
        all_values = {
            'cp': cp_best,
            'mip': mip_best,
            'rl': rl_best
        }
        all_values = {k: v for k, v in all_values.items() if v is not None}
        overall_best = max(all_values.values()) if all_values else None
        
        cp_is_best = cp_best == overall_best if cp_best else False
        mip_is_best = mip_best == overall_best if mip_best else False
        rl_is_best = rl_best == overall_best if rl_best else False
        
        # Format CP cell
        if cp_best is not None:
            cp_formatted = format_objective(cp_best, is_max=cp_is_best)
            img_file = copy_image_to_paper_dir(workpiece, cp_best_solver)
            solver_label_map = {
                'gecode': 'Gecode',
                'lns': 'LNS',
                'chuffed': 'Chuffed',
                'or-tools': 'OR-Tools'
            }
            solver_label = solver_label_map.get(cp_best_solver, 'CP')
            if img_file:
                cp_cell = f"\\makecell{{{solver_label}: ${cp_formatted}$ \\\\ \\includegraphics[width=0.20\\linewidth]{{img/{img_file}}}}}"
            else:
                cp_cell = f"\\makecell{{{solver_label}: ${cp_formatted}$}}"
        else:
            cp_cell = "--"
        
        # Format MIP cell
        if mip_best is not None:
            mip_formatted = format_objective(mip_best, is_max=mip_is_best)
            img_file = copy_image_to_paper_dir(workpiece, 'gurobi')
            if img_file:
                mip_cell = f"\\makecell{{${mip_formatted}$ \\\\ \\includegraphics[width=0.20\\linewidth]{{img/{img_file}}}}}"
            else:
                mip_cell = f"\\makecell{{${mip_formatted}$}}"
        else:
            mip_cell = "--"
        
        # Format RL cell
        if rl_best is not None:
            rl_formatted = format_objective(rl_best, is_max=rl_is_best)
            img_file = copy_image_to_paper_dir(workpiece, 'rl')
            if img_file:
                rl_cell = f"\\makecell{{${rl_formatted}$ \\\\ \\includegraphics[width=0.20\\linewidth]{{img/{img_file}}}}}"
            else:
                rl_cell = f"\\makecell{{${rl_formatted}$}}"
        else:
            rl_cell = "--"
        
        workpiece_cell = f"\\multirow{{2}}{{*}}{{\\makecell{{{display_name}}}}}"
        row = f"\t\t{workpiece_cell}"
        row += f"\n\t\t& {cp_cell}"
        row += f"\n\t\t& {mip_cell}"
        row += f"\n\t\t& {rl_cell} \\\\"
        lines.append(row)
        lines.append("\t\t\\hline")
        lines.append("")
    
    lines.append("\t\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_unified_objective_runtime_table(fdist_data, finer_data, solve_time_data, runtime_data, timeout_data, workpieces, solvers):
    """Generate unified LaTeX table for objective values and runtime."""
    max_fdist = find_max_values(fdist_data, workpieces, solvers)
    max_finer = find_max_values(finer_data, workpieces, solvers)
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("    \\centering")
    lines.append("    \\footnotesize")
    lines.append("    \\caption{Objective values for $\\fdist$ and $\\finer$, and solve times. Solutions where the solver reached optimality (runtime $< 300$ s) are marked with $^*$. For each workpiece, the maximum values of $\\fdist$ and $\\finer$ are highlighted in bold, including ties.}")
    lines.append("    \\label{tab:objective_runtime}")
    lines.append("    \\setlength{\\tabcolsep}{2pt}")
    lines.append("    \\begin{tabular}{llccccc}")
    lines.append("        \\toprule")
    
    solver_display_names = {
        'gecode': 'Gecode',
        'lns': '\\lns',
        'chuffed': 'Chuffed',
        'or-tools': 'OR-Tools',
        'gurobi': 'Gurobi'
    }
    
    header = "        Workpiece & Metric & " + " & ".join([solver_display_names.get(s, s) for s in solvers]) + " \\\\"
    lines.append(header)
    lines.append("        \\midrule")
    lines.append("")
    
    workpiece_display_names = {
        'spiral_stair_step': 'Stair step\\\\spiral staircase',
        'simple_stair_step': 'Stair step\\\\simple staircase',
        'dashboard': 'Dashboard',
        'speaker': 'Speaker',
        'coffee_table': 'Coffee table',
        'door': 'Door',
        'door_porthole': 'Door\\\\with porthole'
    }
    
    for i, workpiece in enumerate(workpieces):
        display_name = workpiece_display_names.get(workpiece, workpiece)
        
        workpiece_cell = f"\\multirow{{3}}{{*}}{{\\makecell{{{display_name}}}}}"
        
        # $\fdist$ row
        fdist_values = []
        for solver in solvers:
            value = fdist_data[workpiece][solver]
            is_max = value == max_fdist[workpiece]
            reached_optimality = not timeout_data[workpiece][solver] if solver in timeout_data[workpiece] else True
            formatted = format_fdist(value, is_max=is_max, is_optimal=reached_optimality)
            fdist_values.append(formatted)
        
        fdist_row = f"        {workpiece_cell} & $\\fdist$ & " + " & ".join(fdist_values) + " \\\\"
        lines.append(fdist_row)
        
        # $\finer$ row
        finer_values = []
        for solver in solvers:
            value = finer_data[workpiece][solver]
            is_max = value == max_finer[workpiece]
            formatted_value = format_finer(value)
            
            if formatted_value == "--":
                formatted = f"\\cellcolor{{gray!15}}--"
            elif is_max:
                formatted = f"\\cellcolor{{gray!15}}$\\mathbf{{{formatted_value}}}$"
            else:
                formatted = f"\\cellcolor{{gray!15}}${formatted_value}$"
            
            finer_values.append(formatted)
        
        finer_row = "        & $\\finer$ & " + " & ".join(finer_values) + " \\\\"
        lines.append(finer_row)
        
        # Solve time row
        solve_time_values = []
        for solver in solvers:
            runtime = runtime_data[workpiece][solver]
            # Display solve time only if runtime < 300 (not a timeout), otherwise display "--"
            if runtime is not None and runtime >= 300.0:
                formatted = "--"
            else:
                value = solve_time_data[workpiece][solver]
                formatted = format_solve_time(value)
            solve_time_values.append(formatted)
        
        solve_time_row = "        & $t$ (s) & " + " & ".join(solve_time_values) + " \\\\"
        lines.append(solve_time_row)
        
        if i < len(workpieces) - 1:
            lines.append("        \\midrule")
        lines.append("")
    
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_objective_table(fdist_data, finer_data, timeout_data, workpieces, solvers):
    """Generate LaTeX table for objective values."""
    max_fdist = find_max_values(fdist_data, workpieces, solvers)
    max_finer = find_max_values(finer_data, workpieces, solvers)
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("    \\centering")
    lines.append("    \\footnotesize")
    lines.append("    \\caption{Objective values for $\\fdist$ and $\\finer$. Solutions where the solver reached optimality (runtime $< 300$ s) are marked with $^*$. For each workpiece, the maximum values of $\\fdist$ and $\\finer$ are highlighted in bold, including ties.}")
    lines.append("    \\label{tab:objective}")
    lines.append("    \\setlength{\\tabcolsep}{2pt}")
    lines.append("    \\begin{tabular}{llccccc}")
    lines.append("        \\toprule")
    
    solver_display_names = {
        'gecode': 'Gecode',
        'lns': '\\lns',
        'chuffed': 'Chuffed',
        'or-tools': 'OR-Tools',
        'gurobi': 'Gurobi'
    }
    
    header = "        Workpiece & Obj. & " + " & ".join([solver_display_names.get(s, s) for s in solvers]) + " \\\\"
    lines.append(header)
    lines.append("        \\midrule")
    lines.append("")
    
    workpiece_display_names = {
        'spiral_stair_step': 'Stair step\\\\spiral staircase',
        'simple_stair_step': 'Stair step\\\\simple staircase',
        'dashboard': 'Dashboard',
        'speaker': 'Speaker',
        'coffee_table': 'Coffee table',
        'door': 'Door',
        'door_porthole': 'Door\\\\with porthole'
    }
    
    for i, workpiece in enumerate(workpieces):
        display_name = workpiece_display_names.get(workpiece, workpiece)
        
        workpiece_cell = f"\\multirow{{2}}{{*}}{{\\makecell{{{display_name}}}}}"
        fdist_values = []
        
        for solver in solvers:
            value = fdist_data[workpiece][solver]
            is_max = value == max_fdist[workpiece]
            reached_optimality = not timeout_data[workpiece][solver]
            formatted = format_fdist(value, is_max=is_max, is_optimal=reached_optimality)
            fdist_values.append(formatted)
        
        fdist_row = f"        {workpiece_cell} & $\\fdist$ & " + " & ".join(fdist_values) + " \\\\"
        lines.append(fdist_row)
        
        finer_values = []
        
        for solver in solvers:
            value = finer_data[workpiece][solver]
            is_max = value == max_finer[workpiece]
            formatted_value = format_finer(value)
            
            if formatted_value == "--":
                formatted = f"\\cellcolor{{gray!15}}--"
            elif is_max:
                formatted = f"\\cellcolor{{gray!15}}$\\mathbf{{{formatted_value}}}$"
            else:
                formatted = f"\\cellcolor{{gray!15}}${formatted_value}$"
            
            finer_values.append(formatted)
        
        finer_row = "        & $\\finer$ & " + " & ".join(finer_values) + " \\\\"
        lines.append(finer_row)
        
        if i < len(workpieces) - 1:
            lines.append("        \\midrule")
        lines.append("")
    
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_runtime_table(solve_time_data, workpieces, solvers):
    """Generate LaTeX table for runtime."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("    \\centering")
    lines.append("    \\footnotesize")
    lines.append("    \\caption{Solve time in seconds to reach the best $\\fdist$ solution. Timeout (300~s) is indicated by ``--''.}")
    lines.append("    \\label{tab:runtime}")
    lines.append("    \\rowcolors{2}{gray!15}{white}")
    lines.append("    \\begin{tabular}{lccccc}")
    lines.append("        \\toprule")
    
    solver_display_names = {
        'gecode': 'Gecode',
        'lns': '\\lns',
        'chuffed': 'Chuffed',
        'or-tools': 'OR-Tools',
        'gurobi': 'Gurobi'
    }
    
    header = "        Workpiece & " + " & ".join([solver_display_names.get(s, s) for s in solvers]) + " \\\\"
    lines.append(header)
    lines.append("        \\midrule")
    lines.append("")
    
    workpiece_display_names = {
        'spiral_stair_step': 'Stair step\\\\spiral staircase',
        'simple_stair_step': 'Stair step\\\\simple staircase',
        'dashboard': 'Dashboard',
        'speaker': 'Speaker',
        'coffee_table': 'Coffee table',
        'door': 'Door',
        'door_porthole': 'Door\\\\with porthole'
    }
    
    for i, workpiece in enumerate(workpieces):
        display_name = workpiece_display_names.get(workpiece, workpiece)
        
        row_parts = [f"        \\makecell{{{display_name}}}"]
        
        for solver in solvers:
            value = solve_time_data[workpiece][solver]
            formatted = format_solve_time(value)
            row_parts.append(formatted)
        
        row = " & ".join(row_parts) + " \\\\"
        lines.append(row)
        
        if i < len(workpieces) - 1:
            lines.append("        \\hline")
            lines.append("")
    
    lines.append("")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_best_solutions_table(data, provider_map, workpieces):
    """Generate table with EO, Best CP/MIP, Best RL, and Best PSO."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\t\\centering")
    lines.append("\t\\renewcommand{\\arraystretch}{1.3}")
    lines.append("\t\\setlength{\\tabcolsep}{3pt}")
    lines.append("\t\\caption{Summary of the best solutions, corresponding solution providers, and moment of inertia values for the different workpieces. ")
    lines.append("\t\tThe workpiece perimeter is shown with solid lines. Dashed lines indicate extrusion areas, fixture regions are marked with diagonal hatching, and supporting bars are shown as shaded rectangles.}")
    lines.append("\t\\label{tab:best_solutions}")
    lines.append("\t\\begin{tabular}{c c c c c}")
    lines.append("\t\t\\hline")
    lines.append("\t\t\\textbf{Workpiece} & \\textbf{Expert Operator} & \\textbf{Best CP/MIP} & \\textbf{Best RL} & \\textbf{Best with PSO} \\\\")
    lines.append("\t\t\\hline")
    
    workpiece_display_names = {
        'spiral_stair_step': 'Spiral stair step',
        'simple_stair_step': 'Simple stair step',
        'dashboard': 'Dashboard',
        'speaker': 'Speaker',
        'coffee_table': 'Coffee table',
        'door': 'Door',
        'door_porthole': 'Door porthole'
    }
    
    for workpiece in workpieces:
        display_name = workpiece_display_names.get(workpiece, workpiece)
        
        # Get Expert Operator value
        eo_value = data[workpiece]['expert']
        
        # Get best CP solution
        cp_solvers = ['gecode', 'lns', 'chuffed', 'or-tools']
        cp_values = {s: data[workpiece][s] for s in cp_solvers if data[workpiece][s] is not None}
        cp_best = max(cp_values.values()) if cp_values else None
        cp_best_solver = [s for s, v in cp_values.items() if v == cp_best][0] if cp_best else None
        
        # Get best MIP solution
        mip_value = data[workpiece]['gurobi']
        
        # Determine best CP/MIP and provider label
        cp_mip_values = {}
        if cp_best is not None:
            cp_mip_values['CP'] = (cp_best, cp_best_solver)
        if mip_value is not None:
            cp_mip_values['MIP'] = (mip_value, 'gurobi')
        
        if cp_mip_values:
            best_cp_mip_value = max([v[0] for v in cp_mip_values.values()])
            best_cp_mip_providers = [k for k, v in cp_mip_values.items() if v[0] == best_cp_mip_value]
            provider_label = ", ".join(sorted(best_cp_mip_providers))
            # Use first provider for image
            first_provider = best_cp_mip_providers[0]
            if first_provider == 'CP':
                img_solver = cp_best_solver
            else:
                img_solver = 'gurobi'
        else:
            best_cp_mip_value = None
            provider_label = None
            img_solver = None
        
        # Get best RL solution
        rl_value = data[workpiece]['rl']
        
        # Get best PSO solution
        pso_solvers = ['pso_cp_pso', 'pso_mip_pso', 'pso_rl_pso', 'pso_eo_pso']
        pso_values = {s: data[workpiece][s] for s in pso_solvers if data[workpiece][s] is not None}
        pso_best = max(pso_values.values()) if pso_values else None
        pso_best_key = [s for s, v in pso_values.items() if v == pso_best][0] if pso_best else None
        
        # Map PSO solver key to display name (starting solution)
        pso_display_map = {
            'pso_cp_pso': 'CP',
            'pso_mip_pso': 'MIP',
            'pso_rl_pso': 'RL',
            'pso_eo_pso': 'EO'
        }
        pso_best_provider = pso_display_map.get(pso_best_key, 'Unknown')
        
        # Format EO cell
        eo_formatted = format_objective(eo_value)
        eo_img = copy_image_to_paper_dir(workpiece, 'expert')
        if eo_img:
            eo_cell = f"\\makecell{{${eo_formatted}$ \\\\ \\includegraphics[width=0.18\\linewidth]{{img/{eo_img}}}}}"
        else:
            eo_cell = f"\\makecell{{${eo_formatted}$}}"
        
        # Format Best CP/MIP cell
        if best_cp_mip_value is not None and provider_label is not None:
            cpmip_formatted = format_objective(best_cp_mip_value)
            cpmip_img = copy_image_to_paper_dir(workpiece, img_solver)
            if cpmip_img:
                cpmip_cell = f"\\makecell{{{provider_label}: ${cpmip_formatted}$ \\\\ \\includegraphics[width=0.18\\linewidth]{{img/{cpmip_img}}}}}"
            else:
                cpmip_cell = f"\\makecell{{{provider_label}: ${cpmip_formatted}$}}"
        else:
            cpmip_cell = "--"
        
        # Format Best RL cell
        if rl_value is not None:
            rl_formatted = format_objective(rl_value)
            rl_img = copy_image_to_paper_dir(workpiece, 'rl')
            if rl_img:
                rl_cell = f"\\makecell{{${rl_formatted}$ \\\\ \\includegraphics[width=0.18\\linewidth]{{img/{rl_img}}}}}"
            else:
                rl_cell = f"\\makecell{{${rl_formatted}$}}"
        else:
            rl_cell = "--"
        
        # Format Best PSO cell
        if pso_best is not None:
            pso_formatted = format_objective(pso_best)
            pso_img = copy_image_to_paper_dir(workpiece, pso_best_key)
            if pso_img:
                pso_cell = f"\\makecell{{{pso_best_provider}: ${pso_formatted}$ \\\\ \\includegraphics[width=0.18\\linewidth]{{img/{pso_img}}}}}"
            else:
                pso_cell = f"\\makecell{{{pso_best_provider}: ${pso_formatted}$}}"
        else:
            pso_cell = "--"
        
        # Build row
        row = f"\t\t\\makecell{{{display_name}}} & {eo_cell} & {cpmip_cell} & {rl_cell} & {pso_cell} \\\\"
        lines.append(row)
        lines.append("\t\t\\hline")
    
    lines.append("\t\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)



# ============================================================================
# MAIN
# ============================================================================

def main():
    base_path = Path(__file__).parent.parent / 'solutions' / 'reports'
    inertia_file = base_path / 'inertia_reports.txt'
    runtime_file = base_path / 'runtime_reports.txt'
    
    print("Parsing data files...")
    inertia_data, provider_map = parse_inertia_reports(str(inertia_file))
    fdist_data, solve_time_data, timeout_data, runtime_data = parse_runtime_reports(str(runtime_file))
    
    workpieces = ['spiral_stair_step', 'simple_stair_step', 'dashboard', 'speaker', 'coffee_table', 'door', 'door_porthole']
    solvers_comparison = ['gecode', 'lns', 'chuffed', 'or-tools', 'gurobi', 'rl']
    solvers_objective = ['gecode', 'lns', 'chuffed', 'or-tools', 'gurobi']
    
    print("Generating tables...")
    
    # Generate all tables
    simple_comparison = generate_simple_comparison_table(inertia_data, workpieces, solvers_comparison)
    best_solutions = generate_best_solutions_table(inertia_data, provider_map, workpieces)
    unified_objective_runtime = generate_unified_objective_runtime_table(fdist_data, inertia_data, solve_time_data, runtime_data, timeout_data, workpieces, solvers_objective)
    
    # Write all tables to single output file
    output_file = base_path / 'all_tables.tex'
    with open(output_file, 'w') as f:
        f.write("% ============================================================================\n")
        f.write("% TABLE 1: Simple Comparison (CP, MIP, RL without images)\n")
        f.write("% ============================================================================\n\n")
        f.write(simple_comparison)
        f.write("\n\n")
        
        f.write("% ============================================================================\n")
        f.write("% TABLE 2: Best Solutions (EO, CP/MIP, RL, Best with PSO)\n")
        f.write("% ============================================================================\n\n")
        f.write(best_solutions)
        f.write("\n\n")
        
        f.write("% ============================================================================\n")
        f.write("% TABLE 3: Objective Values and Runtime (unified)\n")
        f.write("% ============================================================================\n\n")
        f.write(unified_objective_runtime)
    
    print(f"\n✓ All tables generated: {output_file}")
    print("\nGenerated tables:")
    print("  1. Simple Comparison Table (CP, MIP, RL without images)")
    print("  2. Best Solutions Table (EO, CP/MIP, RL, Best with PSO)")
    print("  3. Unified Objective and Runtime Table ($\\fdist$, $\\finer$, solve times)")
    
    # Also write individual table files for compatibility
    print("\nAlso generating individual table files...")
    
    (base_path / 'comparison_table_simple.tex').write_text(simple_comparison)
    print(f"  ✓ {base_path / 'comparison_table_simple.tex'}")
    
    (base_path / 'best_solutions.tex').write_text(best_solutions)
    print(f"  ✓ {base_path / 'best_solutions.tex'}")
    
    (base_path / 'objective_runtime_unified.tex').write_text(unified_objective_runtime)
    print(f"  ✓ {base_path / 'objective_runtime_unified.tex'}")


if __name__ == '__main__':
    main()
