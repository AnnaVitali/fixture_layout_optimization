import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import matplotlib as mpl
from matplotlib.lines import Line2D

colors_tab10 = plt.cm.tab10(np.linspace(0, 1, 10))

WORKPIECE_LABEL_FONTSIZE = 11
PERCENTAGE_LABEL_FONTSIZE = 8

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = PROJECT_ROOT / "solutions" / "reports" / "inertia_reports.txt"


def parse_inertia_report(report_path: Path) -> dict:
    """
    Parse inertia_reports.txt and return a dict of best values by provider and workpiece.
    Structure: {workpiece: {provider: value}}
    """
    data = {}
    
    with open(report_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 4:
            continue
        
        provider = parts[2]
        instance = parts[3]
        
        try:
            objective = float(parts[4])
        except (ValueError, IndexError):
            continue
        
        if instance not in data:
            data[instance] = {}
        
        if provider not in data[instance]:
            data[instance][provider] = objective
        else:
            data[instance][provider] = max(data[instance][provider], objective)
    
    return data


def compute_improvement(new_values, baseline_values):
    """Compute percentage improvement relative to baseline. Skips None values."""
    improvements = []
    for new_val, base_val in zip(new_values, baseline_values):
        if new_val is None or base_val is None or base_val == 0:
            improvements.append(None)
        else:
            improvements.append((new_val - base_val) / base_val * 100)
    return improvements


def get_best_value(data: dict, workpiece: str, providers: list) -> float:
    """Get the best (maximum) value from a list of providers for a workpiece."""
    if workpiece not in data:
        return None
    
    values = []
    for provider in providers:
        if provider in data[workpiece]:
            values.append(data[workpiece][provider])
    
    return max(values) if values else None


def main():
    if not REPORT_FILE.exists():
        print(f"Error: Report file not found: {REPORT_FILE}")
        return
    
    data = parse_inertia_report(REPORT_FILE)
    
    workpiece_names_mapped = {
        "spiral_stair_step": "Stair step\nspiral",
        "simple_stair_step": "Stair step\nsimple",
        "speaker": "Speaker",
        "dashboard": "Dashboard",
        "coffee_table": "Coffee table",
        "door": "Door",
        "door_porthole": "Door\nporthole"
    }
    
    workpieces_display = list(workpiece_names_mapped.values())
    workpieces_keys = list(workpiece_names_mapped.keys())
    
    cp_raw = []
    mip_raw = []
    # RL results temporarily excluded from plots.
    # Uncomment these lines (and the corresponding blocks below) to include them again.
    # rl_raw = []
    cp_pso_raw = []
    mip_pso_raw = []
    # rl_pso_raw = []
    eo_pso_raw = []
    expert_raw = []
    
    for wp_key in workpieces_keys:
        cp_providers = [
            p for p in data.get(wp_key, {}).keys()
            if p in ['cp_model_chuffed', 'cp_model_cp-sat', 'cp_model_gecode', 'LNS_cp_model_gecode']
        ]
        cp_val = get_best_value(data, wp_key, cp_providers)
        cp_raw.append(cp_val)
        
        mip_val = data.get(wp_key, {}).get('mip_model_gurobi', None)
        mip_raw.append(mip_val)
        
        # Reinforcement learning retrieval kept for future use:
        # rl_val = data.get(wp_key, {}).get('rl', None)
        # rl_raw.append(rl_val)
        
        cp_pso_val = data.get(wp_key, {}).get('pso_cp_pso', None)
        cp_pso_raw.append(cp_pso_val)
        
        mip_pso_val = data.get(wp_key, {}).get('pso_mip_pso', None)
        mip_pso_raw.append(mip_pso_val)
        
        # RL + PSO retrieval kept for future use:
        # rl_pso_val = data.get(wp_key, {}).get('pso_rl_pso', None)
        # rl_pso_raw.append(rl_pso_val)
        
        eo_pso_val = data.get(wp_key, {}).get('pso_eo_pso', None)
        eo_pso_raw.append(eo_pso_val)
        
        expert_val = data.get(wp_key, {}).get('expert_operator', None)
        expert_raw.append(expert_val)
    
    impr_cp = compute_improvement(cp_raw, expert_raw)
    impr_mip = compute_improvement(mip_raw, expert_raw)
    # RL improvement calculations kept for future use:
    # impr_rl = compute_improvement(rl_raw, expert_raw)
    impr_cp_pso = compute_improvement(cp_pso_raw, expert_raw)
    impr_mip_pso = compute_improvement(mip_pso_raw, expert_raw)
    # impr_rl_pso = compute_improvement(rl_pso_raw, expert_raw)
    impr_eo_pso = compute_improvement(eo_pso_raw, expert_raw)
    
    # Compute average differences between RL and CP/MIP for first plot
    # rl_vs_cp_diffs = [rl - cp if (rl is not None and cp is not None) else None 
                      # for rl, cp in zip(impr_rl, impr_cp)]
    # rl_vs_mip_diffs = [rl - mip if (rl is not None and mip is not None) else None 
                       # for rl, mip in zip(impr_rl, impr_mip)]
    
    # rl_vs_cp_valid = [v for v in rl_vs_cp_diffs if v is not None]
    # rl_vs_mip_valid = [v for v in rl_vs_mip_diffs if v is not None]
    
    # avg_rl_vs_cp = np.mean(rl_vs_cp_valid) if rl_vs_cp_valid else 0
    # avg_rl_vs_mip = np.mean(rl_vs_mip_valid) if rl_vs_mip_valid else 0
    
    # print(f"\n=== Average Differences (First Plot - Overall) ===")
    # print(f"RL vs CP: {avg_rl_vs_cp:+.2f}% (RL is {abs(avg_rl_vs_cp):.2f}% {'better' if avg_rl_vs_cp > 0 else 'worse'})")
    # print(f"RL vs MIP: {avg_rl_vs_mip:+.2f}% (RL is {abs(avg_rl_vs_mip):.2f}% {'better' if avg_rl_vs_mip > 0 else 'worse'})")
    # print()
    
    # RL series excluded from y-axis scaling while they are not plotted.
    # To restore: add `impr_rl` and `impr_rl_pso` back into this expression.
    all_improvements = [v for v in impr_cp + impr_mip + impr_cp_pso + impr_mip_pso + impr_eo_pso if v is not None]
    y_min = min(all_improvements) if all_improvements else 0
    y_max = max(all_improvements) if all_improvements else 0
    
    y_range = y_max - y_min
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    x = np.arange(len(workpieces_display))
    width = 0.25
    
    # Define colors for each solver (Wong 8-Color Palette - colorblind friendly)
    color_cp = '#56B4E9'      
    color_mip = '#E69F00'    
    # color_rl = '#D55E00'  # Keep for future RL re-enabling
    color_eo = '#009E73'
    
    fig = plt.figure(figsize=(22, 6))
    
    plt.subplot(1, 2, 1)
    
    impr_cp_masked = [v if v is not None else np.nan for v in impr_cp]
    impr_mip_masked = [v if v is not None else np.nan for v in impr_mip]
    # impr_rl_masked = [v if v is not None else np.nan for v in impr_rl]
    
    bars1 = plt.bar(
        x - width/2,
        impr_cp_masked,
        width,
        label="CP",
        color=color_cp,
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = plt.bar(
        x + width/2,
        impr_mip_masked,
        width,
        label="MIP",
        color=color_mip,
        edgecolor="black",
        linewidth=0.7,
    )
    # RL bar kept for future use:
    # bars3 = plt.bar(
    #     x + width,
    #     impr_rl_masked,
    #     width,
    #     label="RL",
    #     color=color_rl,
    #     edgecolor="black",
    #     linewidth=0.7,
    # )
    
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            
            if np.isnan(h):
                continue
            
            if h >= 0:
                va = 'bottom'
                offset = 3
            else:
                va = 'top'
                offset = -3
            
            plt.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=PERCENTAGE_LABEL_FONTSIZE
            )
    
    plt.axhline(y=0, color='black', linestyle=':', linewidth=2.0)
    plt.xticks(x, workpieces_display, rotation=0, fontsize=WORKPIECE_LABEL_FONTSIZE)
    plt.ylabel("Improvement (%)")
    plt.title("Without PSO")
    handles, labels = plt.gca().get_legend_handles_labels()
    baseline_line = Line2D([0], [0], color='black', linestyle=':', linewidth=2.0, label='Expert Operator')
    plt.legend(handles + [baseline_line], labels + ['Expert Operator'], loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(y_min, y_max)
    
    plt.subplot(1, 2, 2)
    
    impr_cp_pso_masked = [v if v is not None else np.nan for v in impr_cp_pso]
    impr_mip_pso_masked = [v if v is not None else np.nan for v in impr_mip_pso]
    # impr_rl_pso_masked = [v if v is not None else np.nan for v in impr_rl_pso]
    impr_eo_pso_masked = [v if v is not None else np.nan for v in impr_eo_pso]
    
    bars4 = plt.bar(
        x - width,
        impr_cp_pso_masked,
        width,
        label="CP + PSO",
        color=color_cp,
        edgecolor="black",
        linewidth=0.7,
    )
    bars5 = plt.bar(
        x,
        impr_mip_pso_masked,
        width,
        label="MIP + PSO",
        color=color_mip,
        edgecolor="black",
        linewidth=0.7,
    )
    # RL + PSO bar kept for future use:
    # bars6 = plt.bar(
    #     x + width*0.5,
    #     impr_rl_pso_masked,
    #     width,
    #     label="RL + PSO",
    #     color=color_rl,
    #     edgecolor="black",
    #     linewidth=0.7,
    # )
    bars7 = plt.bar(
        x + width,
        impr_eo_pso_masked,
        width,
        label="EO + PSO",
        color=color_eo,
        edgecolor="black",
        linewidth=0.7,
    )
    
    for bars in [bars4, bars5, bars7]:
        for bar in bars:
            h = bar.get_height()
            
            if np.isnan(h):
                continue
            
            if h >= 0:
                va = 'bottom'
                offset = 3
            else:
                va = 'top'
                offset = -3
            
            plt.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=PERCENTAGE_LABEL_FONTSIZE
            )
    
    plt.axhline(y=0, color='black', linestyle=':', linewidth=2.0)
    plt.xticks(x, workpieces_display, rotation=0, fontsize=WORKPIECE_LABEL_FONTSIZE)
    plt.ylabel("Improvement (%)")
    plt.title("With PSO integration")
    handles, labels = plt.gca().get_legend_handles_labels()
    baseline_line = Line2D([0], [0], color='black', linestyle=':', linewidth=2.0, label='Expert Operator')
    plt.legend(handles + [baseline_line], labels + ['Expert Operator'], loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "solutions" / "reports" / "comparison_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
    