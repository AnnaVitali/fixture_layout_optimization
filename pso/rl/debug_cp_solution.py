"""
Debug script to understand why agent doesn't find the CP solution with 5 fixtures.
Tests if the CP solution can be placed in the RL environment.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rl_graphical_agent import FixtureLayoutEnv

def find_closest_valid_position(env, target_x, target_y, fixture_type):
    """Find the closest valid position to target coordinates."""
    valid_pairs, _, _ = env._compute_valid_action_positions()
    
    if not valid_pairs:
        return None
    
    # Find closest position
    min_dist = float('inf')
    closest = None
    
    for x_idx, y_idx in valid_pairs:
        x_pos = env.x_positions[x_idx]
        y_pos = env.y_positions[y_idx]
        dist = abs(x_pos - target_x) + abs(y_pos - target_y)
        
        if dist < min_dist:
            min_dist = dist
            closest = (x_idx, y_idx, x_pos, y_pos, dist)
    
    return closest

def analyze_cp_solution():
    """Load and analyze the CP solution."""
    cp_solution = {
        "x": [1, 1, 391, 391, 754],
        "y": [1, 214, 1, 214, 67],
        "fixture_type": [2, 2, 2, 2, 1],  # Type 2, 2, 2, 2, 1
        "angle": [0, 0, 0, 0, 0],
        "objective_value": 72823
    }
    
    print("=" * 80)
    print("CP SOLUTION ANALYSIS")
    print("=" * 80)
    print(f"Number of fixtures: {len(cp_solution['fixture_type'])}")
    print(f"Fixture types: {cp_solution['fixture_type']}")
    print(f"Positions (x, y):")
    for i, (x, y, ft) in enumerate(zip(cp_solution['x'], cp_solution['y'], cp_solution['fixture_type'])):
        print(f"  Fixture {i}: Type {ft} at ({x}, {y})")
    print(f"Objective value (MOI): {cp_solution['objective_value']}")
    print()
    
    # Test if environment allows these placements
    print("=" * 80)
    print("TESTING PLACEMENTS IN RL ENVIRONMENT")
    print("=" * 80)
    
    env = FixtureLayoutEnv(workpiece_name='simple_stair_step')
    obs, info = env.reset()
    
    successful_placements = 0
    
    # Try to place fixtures in sequence
    for i, (target_x, target_y, ft) in enumerate(zip(cp_solution['x'], cp_solution['y'], cp_solution['fixture_type'])):
        print(f"\nFixture {i}: Type {ft} at target ({target_x}, {target_y})")
        
        # Get valid positions
        valid_pairs, valid_x, valid_y = env._compute_valid_action_positions()
        print(f"  Available valid positions: {len(valid_pairs)}")
        
        # Find closest valid position
        closest = find_closest_valid_position(env, target_x, target_y, ft)
        
        if closest:
            x_idx, y_idx, actual_x, actual_y, distance = closest
            print(f"  [OK] Found position at ({actual_x}, {actual_y}) - distance: {distance}mm")
            
            # Place it
            try:
                # Find action index
                action_idx = valid_pairs.index((x_idx, y_idx))
                action = [ft, action_idx]
                
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"    Placed successfully. Reward: {reward}")
                successful_placements += 1
                
                # Show what was actually placed
                if hasattr(env, 'fixtures') and env.fixtures:
                    last_fixture = env.fixtures[-1]
                    print(f"    Actual placement: Type {last_fixture.type_id} at ({last_fixture.x}, {last_fixture.y})")
            except Exception as e:
                print(f"    [FAIL] Failed to place: {e}")
        else:
            print(f"  [FAIL] No valid positions found!")
    
    print("\n" + "=" * 80)
    print(f"RESULT: Successfully placed {successful_placements}/{len(cp_solution['fixture_type'])} fixtures")
    print("=" * 80)
    
    # Compare with agent performance
    print("\nCOMPARISON WITH RL AGENT:")
    print(f"  CP solution: {len(cp_solution['fixture_type'])} fixtures, MOI={cp_solution['objective_value']}")
    print(f"  RL agent (current): 2.08 fixtures on average, MOI≈5.90e9")
    print(f"\n  Gap: {len(cp_solution['fixture_type']) / 2.08:.1f}x more fixtures in CP solution")
    print(f"  Performance gap: {72823 / (5.90e9):.4f}x (CP/Agent MOI ratio)")

if __name__ == '__main__':
    analyze_cp_solution()

