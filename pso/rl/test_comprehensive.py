#!/usr/bin/env python3
"""
Comprehensive evaluation of trained PPO models with configurable episodes and steps
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_graphical_agent import FixtureLayoutEnv
from stable_baselines3 import PPO


def evaluate_comprehensive(
    model_path,
    workpiece_name,
    num_episodes=50,
    max_steps=100,
    render=False,
    use_continuous=False,
    use_hybrid=True,
):
    """
    Comprehensive evaluation with many episodes and extended steps
    
    Args:
        model_path: Path to trained model
        workpiece_name: Name of the workpiece
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render episodes
        use_continuous: Whether to use pure continuous action space
        use_hybrid: Whether to use hybrid discrete + continuous
    """
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    # Create environment with extended max_steps
    env = FixtureLayoutEnv(workpiece_name=workpiece_name, render_mode="human" if render else None, verbose=False, use_continuous_action_space=use_continuous, use_hybrid_action_space=use_hybrid)
    original_max_steps = env.max_steps
    env.max_steps = max_steps  # Override for this test
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(str(model_path.with_suffix('')), env=env)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION")
    print(f"{'='*80}")
    print(f"Workpiece: {workpiece_name}")
    print(f"Model: {model_path.name}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"\n{'='*80}\n")
    
    # Tracking statistics
    episode_rewards = []
    fixtures_placed_list = []
    moi_values = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        done = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        
        while not (done or truncated) and step_count < max_steps:
            # Use trained policy (stochastic)
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        sum_moments = info.get('sum_moments', 0.0)
        num_fixtures = info.get('fixtures_count', 0)
        
        episode_rewards.append(episode_reward)
        fixtures_placed_list.append(num_fixtures)
        moi_values.append(sum_moments)
        episode_lengths.append(step_count)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}/{num_episodes} | Fixtures: {num_fixtures:2d} | Reward: {episode_reward:7.2f} | MOI: {sum_moments:.2e} | Steps: {step_count:3d}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"STATISTICS (over {num_episodes} episodes)")
    print(f"{'='*80}")
    print(f"\nFixtures Placed:")
    print(f"  Mean:     {np.mean(fixtures_placed_list):.2f}")
    print(f"  Std Dev:  {np.std(fixtures_placed_list):.2f}")
    print(f"  Min:      {int(np.min(fixtures_placed_list))}")
    print(f"  Max:      {int(np.max(fixtures_placed_list))}")
    print(f"  Median:   {int(np.median(fixtures_placed_list))}")
    
    print(f"\nEpisode Rewards:")
    print(f"  Mean:     {np.mean(episode_rewards):.2f}")
    print(f"  Std Dev:  {np.std(episode_rewards):.2f}")
    print(f"  Min:      {np.min(episode_rewards):.2f}")
    print(f"  Max:      {np.max(episode_rewards):.2f}")
    
    print(f"\nMOI Values:")
    print(f"  Mean:     {np.mean(moi_values):.2e}")
    print(f"  Max:      {np.max(moi_values):.2e}")
    
    print(f"\nEpisode Length (steps):")
    print(f"  Mean:     {np.mean(episode_lengths):.1f}")
    print(f"  Max:      {int(np.max(episode_lengths))}")
    
    # Find best episode
    best_idx = np.argmax(moi_values)
    print(f"\n{'='*80}")
    print(f"BEST EPISODE (by MOI)")
    print(f"{'='*80}")
    print(f"Episode:      {best_idx + 1}")
    print(f"Fixtures:     {int(fixtures_placed_list[best_idx])}")
    print(f"Reward:       {episode_rewards[best_idx]:.2f}")
    print(f"MOI:          {moi_values[best_idx]:.2e}")
    print(f"Steps taken:  {int(episode_lengths[best_idx])}")
    
    # Distribution
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION OF FIXTURES PLACED")
    print(f"{'='*80}")
    for n_fixtures in sorted(set(int(x) for x in fixtures_placed_list)):
        count = sum(1 for x in fixtures_placed_list if int(x) == n_fixtures)
        percentage = 100.0 * count / num_episodes
        bar = '#' * int(percentage / 2)
        print(f"  {n_fixtures} fixtures: {count:3d} episodes ({percentage:5.1f}%) {bar}")
    
    env.close()
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('workpiece', help='Workpiece name')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--hybrid', action='store_true', default=True, help='Use hybrid discrete+continuous action space (default)')
    parser.add_argument('--continuous', action='store_false', dest='hybrid', help='Use pure continuous action space')
    parser.add_argument('--discrete', action='store_false', dest='hybrid', help='Use pure discrete action space')
    
    args = parser.parse_args()
    
    evaluate_comprehensive(
        args.model_path,
        args.workpiece,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        use_continuous=(not args.hybrid),
        use_hybrid=args.hybrid
    )
