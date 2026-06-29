#!/usr/bin/env python3
"""Test if spiral_stair_step can support multiple fixtures."""

from rl_graphical_agent import FixtureLayoutEnv
from train_qlearning_graphical_agent import get_valid_actions, place_fixture_directly, QLearningAgent

MINIMUM_FIXTURES = {
    "coffee_table": 6,
    "dashboard": 3,
    "simple_stair_step": 3,
    "spiral_stair_step": 3,
    "speaker": 3,
}

env = FixtureLayoutEnv(
    workpiece_name="spiral_stair_step",
    render_mode=None,
    verbose=False,
    use_hybrid_action_space=False,
    use_continuous_action_space=False,
    grid_resolution=10.0,
    max_steps=30
)

print(f"[Test] Workpiece: spiral_stair_step")
print(f"[Test] Safe positions: {len(env.x_positions)} total")
print(f"[Test] Minimum fixtures: {MINIMUM_FIXTURES['spiral_stair_step']}")

obs, info = env.reset()
agent = QLearningAgent()
step = 0

while step < env.max_steps:
    valid_actions = get_valid_actions(env, agent)
    if not valid_actions:
        print(f"[Test] No valid actions at step {step}, num_fixtures={len(env.fixtures)}")
        break
    
    pos_idx, fixture_type = valid_actions[0]
    valid_pairs, _, _ = env._compute_valid_action_positions()
    x_idx, y_idx = valid_pairs[pos_idx]
    
    num_before = len(env.fixtures)
    reward, success = place_fixture_directly(env, x_idx, y_idx, fixture_type, num_before)
    
    if success:
        print(f"[Test] Step {step}: placed type {fixture_type}, "
              f"num_fixtures={len(env.fixtures)}, reward={reward:.2e}")
    else:
        print(f"[Test] Step {step}: placement failed")
        break
    
    step += 1
    if len(env.fixtures) >= 5:
        break

print(f"\n[Test] Final: {len(env.fixtures)} fixtures")
print(f"[Test] Min required: {MINIMUM_FIXTURES['spiral_stair_step']}")
print(f"[Test] Feasible: {len(env.fixtures) >= MINIMUM_FIXTURES['spiral_stair_step']}")
