#!/usr/bin/env python3
"""
Visualize a trained PPO solution from results JSON file
"""

import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from rl_graphical_agent import WorkpieceVisualizer
except ImportError:
    from python.rl_graphical_agent import WorkpieceVisualizer

def visualize_solution(result_file):
    """Load and visualize a solution JSON file"""
    
    result_path = Path(result_file)
    if not result_path.exists():
        print(f"Error: Solution file not found: {result_file}")
        return
    
    with open(result_path, 'r') as f:
        solution = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Visualizing solution from: {result_path.name}")
    print(f"{'='*80}\n")
    
    # Extract info
    workpiece_name = solution.get('workpiece', 'unknown')
    fixtures = solution.get('fixtures', [])
    total_moi = solution.get('sum_moi', 0)
    
    print(f"Workpiece: {workpiece_name}")
    print(f"Fixtures placed: {len(fixtures)}")
    print(f"Sum of MOI: {total_moi:.2e}")
    
    if len(fixtures) == 0:
        print("\n⚠️  No fixtures in solution!")
        return
    
    # Print fixture details
    print(f"\nFixtures:")
    for i, fixture in enumerate(fixtures, 1):
        ftype = fixture.get('type_id', 0)
        x = fixture.get('x', 0)
        y = fixture.get('y', 0)
        angle = fixture.get('angle', 0)
        print(f"  {i}. Type {ftype} at ({x:.1f}, {y:.1f}) angle={angle}°")
    
    # Visualize
    print(f"\nOpening visualization...")
    visualizer = WorkpieceVisualizer(workpiece_name)
    visualizer.render_from_solution(solution)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize PPO training results')
    parser.add_argument('result_file', help='Path to results JSON file')
    args = parser.parse_args()
    
    visualize_solution(args.result_file)
