#!/usr/bin/env python
"""
Test an untrained (random policy) agent to establish baseline performance.
This compares against the trained PPO agent to understand learning impact.
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add pso directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.rl_graphical_agent import FixtureLayoutEnv


def test_random_agent(workpiece_name: str, episodes: int = 50, max_steps: int = 100, use_continuous: bool = False, use_hybrid: bool = True):
    """Run random policy evaluation."""
    
    env = FixtureLayoutEnv(
        workpiece_name=workpiece_name,
        render_mode=None,
        verbose=False,
        use_continuous_action_space=use_continuous,
        use_hybrid_action_space=use_hybrid
    )
    
    print(f"\n{'='*80}")
    print(f"RANDOM (UNTRAINED) AGENT BASELINE")
    print(f"{'='*80}")
    print(f"Workpiece: {workpiece_name}")
    action_mode = "Hybrid (discrete grid + continuous fine-tuning)" if use_hybrid else ("Continuous" if use_continuous else "Discrete")
    print(f"Action space: {action_mode}")
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"{'='*80}\n")
    
    fixtures_placed_list = []
    rewards_list = []
    moi_list = []
    steps_taken_list = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        
        for step in range(max_steps):
            # Random action: choose random fixture type and position
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            ep_steps += 1
            
            if terminated or truncated:
                break
        
        fixtures_placed = len(env.fixtures)
        moi = info.get('sum_moments', 0.0)
        
        fixtures_placed_list.append(fixtures_placed)
        rewards_list.append(ep_reward)
        moi_list.append(moi)
        steps_taken_list.append(ep_steps)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d}/{episodes} | Fixtures: {fixtures_placed:2d} | "
                  f"Reward: {ep_reward:8.2f} | MOI: {moi:.2e} | Steps: {ep_steps}")
    
    env.close()
    
    # Statistics
    fixtures_placed_array = np.array(fixtures_placed_list)
    rewards_array = np.array(rewards_list)
    moi_array = np.array(moi_list)
    steps_taken_array = np.array(steps_taken_list)
    
    print(f"\n{'='*80}")
    print(f"STATISTICS (over {episodes} episodes)")
    print(f"{'='*80}\n")
    
    print(f"Fixtures Placed:")
    print(f"  Mean:     {fixtures_placed_array.mean():.2f}")
    print(f"  Std Dev:  {fixtures_placed_array.std():.2f}")
    print(f"  Min:      {int(fixtures_placed_array.min())}")
    print(f"  Max:      {int(fixtures_placed_array.max())}")
    print(f"  Median:   {int(np.median(fixtures_placed_array))}")
    
    print(f"\nEpisode Rewards:")
    print(f"  Mean:     {rewards_array.mean():.2f}")
    print(f"  Std Dev:  {rewards_array.std():.2f}")
    print(f"  Min:      {rewards_array.min():.2f}")
    print(f"  Max:      {rewards_array.max():.2f}")
    
    print(f"\nMOI Values:")
    print(f"  Mean:     {moi_array.mean():.2e}")
    print(f"  Max:      {moi_array.max():.2e}")
    
    print(f"\nEpisode Length (steps):")
    print(f"  Mean:     {steps_taken_array.mean():.1f}")
    print(f"  Max:      {int(steps_taken_array.max())}")
    
    # Distribution
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION OF FIXTURES PLACED")
    print(f"{'='*80}")
    for n_fixtures in sorted(set(int(x) for x in fixtures_placed_list)):
        count = sum(1 for x in fixtures_placed_list if int(x) == n_fixtures)
        percentage = 100.0 * count / episodes
        bar = '#' * int(percentage / 2)
        print(f"  {n_fixtures} fixtures: {count:3d} episodes ({percentage:5.1f}%) {bar}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test untrained (random) agent")
    parser.add_argument("--workpiece", type=str, default="simple_stair_step",
                        help="Workpiece name")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--hybrid", action="store_true", default=True,
                        help="Use hybrid discrete+continuous action space (default)")
    parser.add_argument("--continuous", action="store_false", dest="hybrid",
                        help="Use pure continuous action space")
    parser.add_argument("--discrete", action="store_false", dest="hybrid",
                        help="Use pure discrete action space")
    
    args = parser.parse_args()
    
    test_random_agent(
        workpiece_name=args.workpiece,
        episodes=args.episodes,
        max_steps=args.max_steps,
        use_continuous=(not args.hybrid),
        use_hybrid=args.hybrid
    )
