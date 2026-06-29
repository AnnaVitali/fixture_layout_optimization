#!/usr/bin/env python3
"""Quick test to verify minimum fixtures constraint works."""

from rl_graphical_agent import FixtureLayoutEnv

# Test minimum fixtures enforcement
MINIMUM_FIXTURES = {
    "coffee_table": 6,
    "dashboard": 3,
    "simple_stair_step": 3,
    "spiral_stair_step": 3,
    "speaker": 3,
}

# Create environment
env = FixtureLayoutEnv(
    workpiece_name="simple_stair_step",
    render_mode=None,
    verbose=False,
    use_hybrid_action_space=False,
    use_continuous_action_space=False,
    grid_resolution=10.0,
    max_steps=30
)

print(f"[Test] Environment created")
print(f"[Test] Workpiece: simple_stair_step")
print(f"[Test] Minimum fixtures required: {MINIMUM_FIXTURES['simple_stair_step']}")
print(f"[Test] Max steps per episode: {env.max_steps}")

# Simulate a quick training episode
obs, info = env.reset()
print(f"\n[Test] Episode started")
print(f"[Test] Initial fixtures: 0")

# Check valid positions
from train_qlearning_graphical_agent import get_valid_actions, place_fixture_directly, QLearningAgent

agent = QLearningAgent()
step = 0
while step < env.max_steps:
    valid_actions = get_valid_actions(env, agent)
    if not valid_actions:
        print(f"[Test] No valid actions at step {step}, num_fixtures={len(env.fixtures)}")
        break
    
    # Take first valid action
    pos_idx, fixture_type = valid_actions[0]
    valid_pairs, _, _ = env._compute_valid_action_positions()
    x_idx, y_idx = valid_pairs[pos_idx]
    
    reward, success = place_fixture_directly(env, x_idx, y_idx, fixture_type)
    print(f"[Test] Step {step}: placed fixture type {fixture_type}, "
          f"num_fixtures={len(env.fixtures)}, success={success}")
    
    step += 1
    
    if len(env.fixtures) >= 6:  # Stop after reasonable number
        break

print(f"\n[Test] Final: {len(env.fixtures)} fixtures placed")
print(f"[Test] Min required: {MINIMUM_FIXTURES['simple_stair_step']}")
print(f"[Test] Valid: {len(env.fixtures) >= MINIMUM_FIXTURES['simple_stair_step']}")
