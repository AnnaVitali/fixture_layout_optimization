"""
Debug script to test if continuous action space allows placing CP solution's 5 fixtures.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl_graphical_agent import FixtureLayoutEnv, FIXTURE_DIMENSIONS

def test_continuous_placement():
    """Test if continuous action space allows CP solution placements."""
    
    # CP solution positions (TOP-LEFT corners)
    cp_solution = {
        "x": [1, 1, 391, 391, 754],
        "y": [1, 214, 1, 214, 67],
        "fixture_type": [2, 2, 2, 2, 1],
        "objective_value": 72823
    }
    
    print("=" * 80)
    print("TESTING CONTINUOUS ACTION SPACE WITH CENTER COORDINATES")
    print("=" * 80)
    print(f"\nCP Solution: {len(cp_solution['fixture_type'])} fixtures")
    print(f"Positions (x, y as top-left): {list(zip(cp_solution['x'], cp_solution['y']))}")
    print()
    
    # Create environment with continuous action space
    env = FixtureLayoutEnv(
        workpiece_name='simple_stair_step',
        use_continuous_action_space=True
    )
    
    print(f"Workpiece: {env.workpiece_name}")
    print(f"Workpiece dimensions: {env.workpiece_width} × {env.workpiece_height} mm")
    print(f"Action space: {env.action_space}")
    print()
    
    obs, info = env.reset()
    
    successful_placements = 0
    total_reward = 0.0
    
    # Try to place all 5 fixtures
    for i, (x_tl, y_tl, ft) in enumerate(zip(cp_solution['x'], cp_solution['y'], cp_solution['fixture_type'])):
        # Convert top-left to center coordinates
        width, height = FIXTURE_DIMENSIONS[ft]
        center_x = x_tl + width / 2
        center_y = y_tl + height / 2
        
        print(f"\nFixture {i}: Type {ft}")
        print(f"  Top-left (CP): ({x_tl}, {y_tl})")
        print(f"  Center (RL): ({center_x}, {center_y})")
        
        # Convert center coordinates to normalized [0, 1]
        x_norm = center_x / env.workpiece_width
        y_norm = center_y / env.workpiece_height
        
        # Create continuous action
        action = np.array([float(ft), x_norm, y_norm], dtype=np.float32)
        print(f"  Normalized action: [{float(ft):.2f}, {x_norm:.4f}, {y_norm:.4f}]")
        
        # Execute placement
        obs, reward, done, truncated, info = env.step(action)
        
        if info['placement_success']:
            print(f"  [OK] Placement successful!")
            print(f"    Reward: {reward:.2f}")
            fixture_center = info.get('fixture_center', ('N/A', 'N/A'))
            print(f"    Actual center: ({fixture_center[0]:.1f}, {fixture_center[1]:.1f})")
            successful_placements += 1
            total_reward += reward
        else:
            print(f"  [FAIL] Placement failed: {info['reason']}")
            print(f"    Reward: {reward:.2f}")
    
    print("\n" + "=" * 80)
    print(f"RESULT: Placed {successful_placements}/{len(cp_solution['fixture_type'])} fixtures")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total MOI: {env.cumulative_moment:.2e}")
    print("=" * 80)
    
    env.close()
    
    return successful_placements == len(cp_solution['fixture_type'])

if __name__ == '__main__':
    success = test_continuous_placement()
    exit(0 if success else 1)
