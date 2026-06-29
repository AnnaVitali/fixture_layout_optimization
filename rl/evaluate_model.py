#!/usr/bin/env python3
"""
Evaluate and visualize trained PPO model solutions
"""

import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from rl_graphical_agent import FixtureLayoutEnv, fixtures_to_solution_json
except ImportError:
    from python.rl_graphical_agent import FixtureLayoutEnv, fixtures_to_solution_json

from stable_baselines3 import PPO


def evaluate_and_visualize(model_path, workpiece_name, num_episodes=3, render=True, use_hybrid=True, max_steps=100):
    """
    Load a trained model and evaluate it
    
    Args:
        model_path: Path to trained model
        workpiece_name: Name of the workpiece
        num_episodes: Number of episodes to evaluate
        render: Whether to render with GUI
        use_hybrid: Whether to use hybrid action space (should match training)
        max_steps: Maximum steps per episode
    """
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    # Create environment with SAME PARAMETERS as training (HYBRID MODE)
    env = FixtureLayoutEnv(
        workpiece_name=workpiece_name,
        render_mode="human" if render else None,
        verbose=True,
        use_hybrid_action_space=use_hybrid,  # EXPLICITLY SET to match training
        max_steps=max_steps  # Set custom max steps
    )
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path, env=env)
    
    print(f"\n{'='*80}")
    print(f"Evaluating trained model on {workpiece_name}")
    print(f"{'='*80}\n")
    
    best_solution = None
    best_moi = 0
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        
        done = False
        total_reward = 0
        
        while not done:
            # Get action from model using STOCHASTIC evaluation (like training)
            # This matches the training behavior better than deterministic=True
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Get final state
        num_fixtures = len(env.fixtures)
        moi = env.cumulative_moment if hasattr(env, 'cumulative_moment') else 0
        
        print(f"  Episode reward: {total_reward:.2f}")
        print(f"  Fixtures placed: {num_fixtures}")
        print(f"  Sum of MOI: {moi:.2e}")
        
        # Track best solution
        if moi > best_moi:
            best_moi = moi
            best_solution = fixtures_to_solution_json(env.fixtures, moi, env.visualizer)
    
    # Save best solution
    if best_solution:
        output_file = Path(__file__).parent / f"rl_{workpiece_name}_ppo_eval.json"
        with open(output_file, 'w') as f:
            json.dump(best_solution, f, indent=2)
        print(f"\n[OK] Best solution saved to: {output_file}")
        print(f"     Best MOI: {best_moi:.2e}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate and visualize trained PPO model')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('workpiece', help='Workpiece name')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode (default: 100)')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--hybrid', action='store_true', default=True, help='Use hybrid action space (default: True)')
    parser.add_argument('--continuous', action='store_true', default=False, help='Use continuous action space')
    parser.add_argument('--discrete', action='store_true', default=False, help='Use discrete action space')
    
    args = parser.parse_args()
    
    # Determine action space mode
    use_hybrid = True
    if args.continuous:
        use_hybrid = False
    elif args.discrete:
        use_hybrid = False
    
    evaluate_and_visualize(
        args.model_path,
        args.workpiece,
        num_episodes=args.episodes,
        render=not args.no_render,
        use_hybrid=use_hybrid,
        max_steps=args.max_steps
    )
