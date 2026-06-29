#!/usr/bin/env python3
"""
Visualize a solution from the results JSON file
"""

import json
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from machine_parameters import FIXTURE_DIMENSIONS
    workpiece_file = Path(__file__).parent / "resources" / "workpieces_information.json"
except ImportError:
    try:
        from python.machine_parameters import FIXTURE_DIMENSIONS
        workpiece_file = Path(__file__).parent.parent / "python" / "resources" / "workpieces_information.json"
    except ImportError:
        print("Error: Cannot import machine_parameters")
        sys.exit(1)


def load_workpiece_data(workpiece_name):
    """Load workpiece data from JSON"""
    with open(workpiece_file, 'r') as f:
        data = json.load(f)
    
    if workpiece_name not in data:
        print(f"Error: Workpiece '{workpiece_name}' not found in {workpiece_file}")
        sys.exit(1)
    
    return data[workpiece_name]


def visualize_solution(solution_file):
    """Load and visualize a solution JSON file"""
    
    solution_path = Path(solution_file)
    if not solution_path.exists():
        print(f"Error: Solution file not found: {solution_file}")
        return
    
    with open(solution_path, 'r') as f:
        solution = json.load(f)
    
    # Check if old format (x, y, x1, y1) or new format (fixtures array)
    if 'fixtures' in solution:
        # New format
        fixtures = solution.get('fixtures', [])
        total_moi = solution.get('sum_moi', 0)
        workpiece_name = solution.get('workpiece', 'unknown')
    else:
        # Old format (x, y, x1, y1, x2, y2, x3, y3)
        x_coords = solution.get('x', [])
        y_coords = solution.get('y', [])
        x1_coords = solution.get('x1', [])
        y1_coords = solution.get('y1', [])
        x2_coords = solution.get('x2', [])
        y2_coords = solution.get('y2', [])
        x3_coords = solution.get('x3', [])
        y3_coords = solution.get('y3', [])
        total_moi = solution.get('objective_value', 0)
        fixture_types = solution.get('fixture_type', [])
        
        # Convert to fixture format
        fixtures = []
        for i in range(len(x_coords)):
            fixtures.append({
                'type_id': fixture_types[i] if i < len(fixture_types) else 1,
                'x': [x_coords[i], x1_coords[i], x2_coords[i], x3_coords[i]],
                'y': [y_coords[i], y1_coords[i], y2_coords[i], y3_coords[i]],
                'area': 0  # Not stored in old format
            })
        
        # Try to infer workpiece name from file
        workpiece_name = 'unknown'
        file_stem = solution_path.stem
        for wname in ['coffee_table', 'simple_stair_step', 'dashboard', 'speaker', 'spiral_stair_step']:
            if wname in file_stem:
                workpiece_name = wname
                break
    
    print(f"\n{'='*80}")
    print(f"Solution: {solution_path.name}")
    print(f"{'='*80}")
    print(f"Workpiece: {workpiece_name}")
    print(f"Fixtures placed: {len(fixtures)}")
    print(f"Sum of MOI / Objective: {total_moi:.2e}")
    print(f"\nFixtures:")
    for i, fixture in enumerate(fixtures, 1):
        ftype = fixture.get('type_id', 0)
        x_corners = fixture.get('x', [])
        y_corners = fixture.get('y', [])
        area = fixture.get('area', 0)
        print(f"  {i}. Type {ftype} | Corners: {len(x_corners)} | Pos: ({x_corners[0]:.1f}, {y_corners[0]:.1f})")
    
    if len(fixtures) == 0:
        print("\n⚠️  No fixtures in solution!")
        return
    
    # Load workpiece data
    try:
        workpiece_data = load_workpiece_data(workpiece_name)
    except Exception as e:
        print(f"Error loading workpiece: {e}")
        return
    
    # Create visualization
    print(f"\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw workpiece boundary
    vertices = np.array(workpiece_data['vertices'])
    workpiece_poly = patches.Polygon(
        vertices,
        closed=True,
        edgecolor='black',
        facecolor='lightgray',
        linewidth=2,
        alpha=0.3,
        label='Workpiece'
    )
    ax.add_patch(workpiece_poly)
    
    # Draw holes if present
    if 'holes' in workpiece_data:
        for hole in workpiece_data['holes']:
            if len(hole) == 3:  # Circle: [cx, cy, radius]
                circle = patches.Circle(
                    (hole[0], hole[1]),
                    hole[2],
                    edgecolor='red',
                    facecolor='red',
                    alpha=0.2,
                    linewidth=1.5,
                    linestyle='--',
                    label='Hole' if hole == workpiece_data['holes'][0] else ''
                )
                ax.add_patch(circle)
    
    # Draw fixtures
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    for fixture_idx, fixture in enumerate(fixtures):
        ftype = fixture.get('type_id', 0)
        x_corners = fixture.get('x', [])
        y_corners = fixture.get('y', [])
        
        if not x_corners or not y_corners or len(x_corners) < 2:
            continue
        
        # Create polygon from corners (assume order: LL, LR, UR, UL)
        corners = list(zip(x_corners, y_corners))
        color = colors[fixture_idx % len(colors)]
        
        fixture_poly = patches.Polygon(
            corners,
            closed=True,
            edgecolor=color,
            facecolor=color,
            linewidth=2,
            alpha=0.7,
            label=f'Fixture {fixture_idx + 1} (Type {ftype})'
        )
        ax.add_patch(fixture_poly)
        
        # Calculate and draw center
        center_x = np.mean(x_corners)
        center_y = np.mean(y_corners)
        ax.plot(center_x, center_y, 'k+', markersize=10, markeredgewidth=2)
        ax.text(center_x, center_y, str(fixture_idx + 1), 
                fontsize=9, ha='center', va='center', 
                color='white', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
    
    # Set axis properties
    ax.set_xlim(vertices[:, 0].min() - 50, vertices[:, 0].max() + 50)
    ax.set_ylim(vertices[:, 1].min() - 50, vertices[:, 1].max() + 50)
    ax.set_aspect('equal')
    ax.set_title(f'{workpiece_name} - {len(fixtures)} Fixtures (Objective: {total_moi:.2e})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.grid(True, alpha=0.3)
    if len(fixtures) <= 6:
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize PPO training results')
    parser.add_argument('solution_file', help='Path to solution JSON file')
    args = parser.parse_args()
    
    visualize_solution(args.solution_file)
