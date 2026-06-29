#!/usr/bin/env python3
"""
Train PPO agent directly on the FixtureLayoutEnv (graphical agent).

This trains the full fixture placement problem without the bar-positioning bottleneck.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    raise SystemExit("stable-baselines3 is required. Install it with: pip install stable-baselines3[extra]")

from rl_graphical_agent import FixtureLayoutEnv, fixtures_to_solution_json

ROOT = Path(__file__).resolve().parent


def evaluate_random_baseline(workpiece_name: str, n_episodes: int = 10):
    """
    Evaluate the environment with random actions to establish baseline.
    """
    print(f"\n{'='*80}")
    print(f"RANDOM BASELINE TEST: {n_episodes} episodes")
    print(f"{'='*80}")
    
    env = FixtureLayoutEnv(workpiece_name=workpiece_name, verbose=False, max_steps=500)
    
    total_reward = 0
    total_fixtures = 0
    total_moi = 0
    best_moi = 0
    worst_moi = float('inf')
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        num_fixtures = len(env.fixtures)
        moi = env.cumulative_moment
        
        total_reward += episode_reward
        total_fixtures += num_fixtures
        total_moi += moi
        best_moi = max(best_moi, moi)
        worst_moi = min(worst_moi, moi)
        
        print(f"  Episode {episode+1:2d}: reward={episode_reward:7.2f}, fixtures={num_fixtures}, MOI={moi:.2e}")
    
    env.close()
    
    mean_reward = total_reward / n_episodes
    mean_fixtures = total_fixtures / n_episodes
    mean_moi = total_moi / n_episodes
    
    print(f"\nRandom baseline summary:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Mean fixtures: {mean_fixtures:.1f}")
    print(f"  Mean MOI: {mean_moi:.2e}")
    print(f"  Best MOI: {best_moi:.2e}")
    print(f"  Worst MOI: {worst_moi:.2e}")
    
    return {
        'mean_reward': mean_reward,
        'mean_fixtures': mean_fixtures,
        'mean_moi': mean_moi,
        'best_moi': best_moi,
    }


def train_ppo_graphical(
    workpiece_name: str = "simple_stair_step",
    total_timesteps: int = 50000,
    learning_rate: float = 0.0003,
    batch_size: int = 64,
    n_steps: int = 2048,
):
    """
    Train PPO agent on FixtureLayoutEnv.
    """
    print(f"\n{'='*80}")
    print(f"TRAINING PPO ON GRAPHICAL ENVIRONMENT")
    print(f"Workpiece: {workpiece_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"{'='*80}\n")
    
    # Create training environment with LONGER max_steps to allow multi-fixture placement
    train_env = FixtureLayoutEnv(
        workpiece_name=workpiece_name,
        verbose=False,
        use_hybrid_action_space=True,  # Use hybrid action space
        max_steps=500,  # Allow up to 500 steps per episode for exploration
    )
    
    # Create evaluation environment
    eval_env = FixtureLayoutEnv(
        workpiece_name=workpiece_name,
        verbose=False,
        use_hybrid_action_space=True,
        max_steps=500,  # Same as training env
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models"),
        log_path=str(ROOT / "logs"),
        eval_freq=n_steps,
        n_eval_episodes=5,
        deterministic=False,  # Use stochastic evaluation during training
        render=False,
    )
    
    # Train the model
    print("[INFO] Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    
    # Save model
    model_path = ROOT / "models" / f"ppo_graphical_{workpiece_name}"
    model.save(str(model_path))
    print(f"\n[OK] Model saved to: {model_path}.zip")
    
    train_env.close()
    eval_env.close()
    
    return str(model_path)


def evaluate_best_model(workpiece_name: str, model_path: str = None):
    """
    Evaluate the best trained model and save solution.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING BEST MODEL")
    print(f"{'='*80}\n")
    
    env = FixtureLayoutEnv(workpiece_name=workpiece_name, verbose=False, max_steps=500)
    
    if model_path is None:
        model_path = ROOT / "models" / f"best_model_{workpiece_name}"
    
    # Load model
    model = PPO.load(str(model_path), env=env)
    
    # Evaluate stochastically (exploration)
    print("[INFO] Evaluating with stochastic policy (exploration)...")
    total_reward = 0
    total_fixtures = 0
    total_moi = 0
    best_fixtures = 0
    best_moi = 0
    best_env_copy = None
    
    for episode in range(5):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        num_fixtures = len(env.fixtures)
        moi = env.cumulative_moment
        
        total_reward += episode_reward
        total_fixtures += num_fixtures
        total_moi += moi
        
        print(f"  Episode {episode+1}: reward={episode_reward:7.2f}, fixtures={num_fixtures}, MOI={moi:.2e}")
        
        if moi > best_moi:
            best_moi = moi
            best_fixtures = num_fixtures
            import copy
            best_env_copy = copy.deepcopy(env)
    
    print(f"\n[SUMMARY] Stochastic evaluation:")
    print(f"  Mean reward: {total_reward/5:.2f}")
    print(f"  Mean fixtures: {total_fixtures/5:.1f}")
    print(f"  Mean MOI: {total_moi/5:.2e}")
    print(f"  Best MOI: {best_moi:.2e} ({best_fixtures} fixtures)")
    
    # Save best solution
    if best_env_copy is not None and len(best_env_copy.fixtures) > 0:
        print(f"\n[OK] Saving best solution...")
        solution = fixtures_to_solution_json(
            best_env_copy.fixtures,
            best_env_copy.cumulative_moment,
            best_env_copy.visualizer,
        )
        
        output_dir = ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"ppo_graphical_{workpiece_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(solution, f, indent=2)
        
        print(f"[OK] Solution saved to: {output_file}")
        print(f"     Fixtures: {len(solution['x'])}")
        print(f"     MOI: {solution['objective_value']:.2e}")
    
    # Evaluate deterministically (exploitation)
    print(f"\n[INFO] Evaluating with deterministic policy (exploitation)...")
    obs, info = env.reset()
    done = False
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
    
    num_fixtures = len(env.fixtures)
    moi = env.cumulative_moment
    
    print(f"  Deterministic result: {num_fixtures} fixtures, MOI={moi:.2e}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Train PPO on FixtureLayoutEnv')
    parser.add_argument('--workpiece', type=str, default='simple_stair_step',
                       help='Workpiece name')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                       help='PPO learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--random-baseline', action='store_true',
                       help='Evaluate random baseline first')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the best model')
    
    args = parser.parse_args()
    
    # If no flags specified, do all steps
    if not (args.random_baseline or args.train or args.evaluate):
        args.random_baseline = True
        args.train = True
        args.evaluate = True
    
    # Evaluate random baseline
    if args.random_baseline:
        evaluate_random_baseline(args.workpiece, n_episodes=10)
    
    # Train model
    model_path = None
    if args.train:
        model_path = train_ppo_graphical(
            args.workpiece,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    
    # Evaluate best model
    if args.evaluate:
        if model_path:
            # Use the model we just trained
            best_model_dir = ROOT / "models" / f"best_model_{args.workpiece}"
            evaluate_best_model(args.workpiece, str(best_model_dir))
        else:
            # Use existing best model
            best_model_dir = ROOT / "models" / f"best_model_{args.workpiece}"
            if best_model_dir.with_suffix('.zip').exists():
                evaluate_best_model(args.workpiece, str(best_model_dir))
            else:
                print(f"[ERROR] No trained model found at {best_model_dir}")


if __name__ == "__main__":
    main()
