import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


BAR_GRAY_DARK = "#4d4d4d"
BAR_GRAY_MEDIUM = "#8c8c8c"
BAR_GRAY_LIGHT = "#c7c7c7"
ACCENT_GRAY = "#1f1f1f"

WORKPIECE_LABEL_FONTSIZE = 12
PERCENTAGE_LABEL_FONTSIZE = 9


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
        "spiral_stair_step": "Stair step spiral staircase",
        "simple_stair_step": "Stair step simple staircase",
        "speaker": "Speaker",
        "dashboard": "Dashboard",
        "coffee_table": "Coffee Table"
    }
    
    workpieces_display = list(workpiece_names_mapped.values())
    workpieces_keys = list(workpiece_names_mapped.keys())
    
    cp_raw = []
    mip_raw = []
    cp_pso_raw = []
    mip_pso_raw = []
    eo_pso_raw = []
    expert_raw = []
    
    for wp_key in workpieces_keys:
        # Best CP (all solvers, including LNS CP)
        cp_providers = [
            p for p in data.get(wp_key, {}).keys()
            if p.startswith("cp_model") or p.startswith("LNS_cp_model")
        ]
        cp_val = get_best_value(data, wp_key, cp_providers)
        cp_raw.append(cp_val if cp_val else None)
        
        # Best MIP
        mip_val = get_best_value(data, wp_key, [p for p in data.get(wp_key, {}).keys() if p.startswith("mip_model") and "pso" not in p])
        mip_raw.append(mip_val if mip_val else None)
        
        # CP + PSO
        cp_pso_val = data.get(wp_key, {}).get("cp_pso", None)
        cp_pso_raw.append(cp_pso_val)
        
        # MIP + PSO
        mip_pso_val = data.get(wp_key, {}).get("mip_pso", None)
        mip_pso_raw.append(mip_pso_val)
        
        # EO + PSO
        eo_pso_val = data.get(wp_key, {}).get("pso_eo_pso", None)
        eo_pso_raw.append(eo_pso_val)
        
        # Expert Operator
        expert_val = data.get(wp_key, {}).get("expert_operator", None)
        expert_raw.append(expert_val)
    
    impr_cp = compute_improvement(cp_raw, expert_raw)
    impr_mip = compute_improvement(mip_raw, expert_raw)
    impr_cp_pso = compute_improvement(cp_pso_raw, expert_raw)
    impr_mip_pso = compute_improvement(mip_pso_raw, expert_raw)
    impr_eo_pso = compute_improvement(eo_pso_raw, expert_raw)
    
    all_improvements = [v for v in impr_cp + impr_mip + impr_cp_pso + impr_mip_pso + impr_eo_pso if v is not None]
    y_min = min(all_improvements) if all_improvements else 0
    y_max = max(all_improvements) if all_improvements else 0
    
    y_range = y_max - y_min
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    x = np.arange(len(workpieces_display))
    width = 0.25
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    
    impr_cp_masked = [v if v is not None else np.nan for v in impr_cp]
    impr_mip_masked = [v if v is not None else np.nan for v in impr_mip]
    
    bars1 = plt.bar(
        x - width/2,
        impr_cp_masked,
        width,
        label="CP Model",
        color=BAR_GRAY_DARK,
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = plt.bar(
        x + width/2,
        impr_mip_masked,
        width,
        label="MIP Model",
        color=BAR_GRAY_LIGHT,
        edgecolor="black",
        linewidth=0.7,
    )
    
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
    
    plt.axhline(y=0, color=ACCENT_GRAY, linestyle=':', linewidth=2.0)
    plt.text(
        0.02,
        0.04,
        "expert operator solution",
        transform=plt.gca().transAxes,
        color=ACCENT_GRAY,
        fontsize=9,
        fontweight="bold",
        alpha=0.7
    )
    plt.xticks(x, workpieces_display, rotation=20, fontsize=WORKPIECE_LABEL_FONTSIZE)
    plt.ylabel("Improvement (%)")
    plt.title("Without PSO")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(y_min, y_max)
    
    plt.subplot(1, 2, 2)
    
    impr_cp_pso_masked = [v if v is not None else np.nan for v in impr_cp_pso]
    impr_mip_pso_masked = [v if v is not None else np.nan for v in impr_mip_pso]
    impr_eo_pso_masked = [v if v is not None else np.nan for v in impr_eo_pso]
    
    bars3 = plt.bar(
        x - width,
        impr_cp_pso_masked,
        width,
        label="CP + PSO",
        color=BAR_GRAY_DARK,
        edgecolor="black",
        linewidth=0.7,
    )
    bars4 = plt.bar(
        x,
        impr_mip_pso_masked,
        width,
        label="MIP + PSO",
        color=BAR_GRAY_MEDIUM,
        edgecolor="black",
        linewidth=0.7,
    )
    bars5 = plt.bar(
        x + width,
        impr_eo_pso_masked,
        width,
        label="EO + PSO",
        color=BAR_GRAY_LIGHT,
        edgecolor="black",
        linewidth=0.7,
    )
    
    for bars in [bars3, bars4, bars5]:
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
    
    plt.axhline(y=0, color=ACCENT_GRAY, linestyle=':', linewidth=2.0)
    plt.text(
        0.02,
        0.04,
        "expert operator solution",
        transform=plt.gca().transAxes,
        color=ACCENT_GRAY,
        fontsize=9,
        fontweight="bold",
        alpha=0.7
    )
    plt.xticks(x, workpieces_display, rotation=20, fontsize=WORKPIECE_LABEL_FONTSIZE)
    plt.title("With PSO (vs Expert Operator)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    output_path = PROJECT_ROOT / "solutions" / "reports" / "comparison_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
