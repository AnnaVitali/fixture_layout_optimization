"""
PPO Training Script for Fixture Layout Optimization

Trains a Proximal Policy Optimization (PPO) agent to optimize fixture layouts
using either the placement agent (rl_graphical_agent.py) or rotation agent
(rl_rotation_agent.py).

Requirements:
    pip install stable-baselines3[extra]

Usage:
    # Train placement agent
    python train_ppo.py --agent placement --workpiece simple_stair_step --timesteps 100000 --render
    
    # Train rotation agent
    python train_ppo.py --agent rotation --workpiece simple_stair_step --timesteps 50000
    
    # Load and continue training
    python train_ppo.py --agent placement --workpiece simple_stair_step --load models/ppo_placement_simple_stair_step --timesteps 50000
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import shutil

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError as exc:
    raise SystemExit(
        "stable-baselines3 is required. Install it with: pip install stable-baselines3[extra]"
    ) from exc

# Import environments
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_graphical_agent import FixtureLayoutEnv, fixtures_to_solution_json, save_best_solution
try:
    from rl_rotation_agent import FixtureRotationEnv
except ImportError:
    FixtureRotationEnv = None

ROOT = Path(__file__).resolve().parent


def create_environment(agent_type: str, workpiece_name: str, render: bool = False, verbose: bool = False, use_continuous: bool = False, use_hybrid: bool = True):
    """
    Create the appropriate environment based on agent type.
    
    Args:
        agent_type: "placement" or "rotation"
        workpiece_name: Name of the workpiece
        render: Whether to render the environment
        verbose: Whether to print debug information
        use_continuous: If True, use pure continuous action space (inefficient, legacy)
        use_hybrid: If True, use hybrid discrete + continuous fine-tuning (recommended, default)
        
    Returns:
        Gymnasium environment
    """
    if agent_type == "placement":
        env = FixtureLayoutEnv(
            workpiece_name=workpiece_name,
            render_mode="human" if render else None,
            verbose=verbose,
            use_continuous_action_space=use_continuous,
            use_hybrid_action_space=use_hybrid
        )
    elif agent_type == "rotation":
        if FixtureRotationEnv is None:
            raise ValueError("RotationEnv not available")
        env = FixtureRotationEnv(
            workpiece_name=workpiece_name,
            render_mode="human" if render else None
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return env


def train_ppo(
    agent_type: str,
    workpiece_name: str,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    load_model_path: Optional[str] = None,
    render: bool = False,
    eval_freq: int = 10000,
    use_continuous: bool = False,
    use_hybrid: bool = True,
):
    """
    Train a PPO agent for fixture layout optimization.
    
    Args:
        agent_type: "placement" or "rotation"
        workpiece_name: Name of the workpiece
        total_timesteps: Total number of timesteps to train
        learning_rate: Learning rate for PPO
        batch_size: Batch size for training
        n_steps: Number of steps per rollout
        load_model_path: Path to pre-trained model to continue training
        render: Whether to render during evaluation
        eval_freq: Evaluation frequency (in timesteps)
        use_continuous: If True, use pure continuous action space (inefficient)
        use_hybrid: If True, use hybrid discrete + continuous (recommended, default True)
    """
    print(f"\n{'='*80}")
    print(f"PPO Training for {agent_type.upper()} Agent")
    print(f"{'='*80}")
    print(f"Workpiece: {workpiece_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"N steps: {n_steps}")
    
    # Create environment
    print(f"\nCreating environment...")
    env = create_environment(agent_type, workpiece_name, render=False, verbose=False, use_continuous=use_continuous, use_hybrid=use_hybrid)
    
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Create directories for models and logs
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    model_name = f"ppo_{agent_type}_{workpiece_name}"
    model_path = models_dir / model_name
    
    # Create or load PPO model
    print(f"\nInitializing PPO model...")
    if load_model_path:
        print(f"  Loading pre-trained model from: {load_model_path}")
        model = PPO.load(load_model_path, env=env)
        # Update learning rate if specified
        model.learning_rate = learning_rate
    else:
        print(f"  Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            verbose=1,
            tensorboard_log=None,
        )
    
    # Create callbacks
    print(f"\nSetting up callbacks...")
    
    # Evaluation callback - save only best model
    eval_env = create_environment(agent_type, workpiece_name, render=render, verbose=False, use_continuous=use_continuous, use_hybrid=use_hybrid)
    best_model_path = models_dir / f"best_{agent_type}_{workpiece_name}"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_path),
        log_path=str(logs_dir),
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=False,
        render=render,
    )
    
    # Train
    print(f"\n{'='*80}")
    print(f"Training started...")
    print(f"{'='*80}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback],
            tb_log_name=model_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Load and save best model
    print(f"\n\n{'='*80}")
    print(f"Training completed!")
    print(f"{'='*80}")
    
    # Check if best model exists from EvalCallback
    best_zip_path = best_model_path.parent / f"{best_model_path.name}.zip"
    if best_zip_path.exists():
        print(f"\n[OK] Best model saved to: {best_zip_path}")
        best_model_path_final = best_zip_path
    else:
        # Fallback: save current model
        model.save(str(best_model_path))
        print(f"\n[OK] Model saved to: {best_model_path}.zip")
        best_model_path_final = Path(str(best_model_path) + ".zip")
    
    # Evaluate trained model
    print(f"\nEvaluating best model...\n")
    
    try:
        # Clean up any conflicting directory with the same name (without .zip)
        conflicting_dir = best_model_path
        if conflicting_dir.exists() and conflicting_dir.is_dir():
            shutil.rmtree(conflicting_dir)
            print(f"[OK] Removed conflicting directory: {conflicting_dir}")
        
        best_model = PPO.load(str(best_model_path_final.with_suffix('')), env=env)
        evaluate_model(best_model, agent_type, workpiece_name, num_episodes=5, render=render, use_continuous=use_continuous, use_hybrid=use_hybrid)
    except Exception as e:
        print(f"\n[WARNING] Could not evaluate model: {e}")
        print(f"          Model is saved at: {best_model_path_final}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    print(f"\n{'='*80}")
    print(f"Best model filename: {best_model_path_final.name}")
    print(f"To continue training:")
    print(f"  python train_ppo.py --agent {agent_type} --workpiece {workpiece_name} --load {best_model_path_final.with_suffix('')}")
    print(f"{'='*80}")


def evaluate_model(
    model,
    agent_type: str,
    workpiece_name: str,
    num_episodes: int = 5,
    render: bool = False,
    use_continuous: bool = False,
    use_hybrid: bool = True,
):
    """
    Evaluate a trained PPO model.
    
    Args:
        model: Trained PPO model
        agent_type: "placement" or "rotation"
        workpiece_name: Name of the workpiece
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        use_continuous: If True, use pure continuous action space
        use_hybrid: If True, use hybrid discrete + continuous
    """
    env = create_environment(agent_type, workpiece_name, render=render, use_continuous=use_continuous, use_hybrid=use_hybrid)
    
    best_reward = -float('inf')
    best_solution = None
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        done = False
        truncated = False
        episode_reward = 0.0
        
        while not (done or truncated):
            # Use trained policy (stochastic to match training dynamics)
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        sum_moments = info.get('sum_moments', 0.0)
        num_fixtures = info.get('fixtures_count', 0)
        
        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Sum of MOI: {sum_moments:.2e}")
        print(f"  Fixtures placed: {num_fixtures}")
        
        # Track best solution for placement agent
        if agent_type == "placement" and sum_moments > best_reward and num_fixtures > 0:
            best_reward = sum_moments
            best_solution = (env.fixtures.copy(), sum_moments)
    
    env.close()
    
    # Save best solution from evaluation
    if agent_type == "placement" and best_solution:
        print(f"\n{'='*80}")
        print(f"Saving best evaluation solution...")
        fixtures, objective_value = best_solution
        solution_json = fixtures_to_solution_json(fixtures, objective_value, env.visualizer)
        save_best_solution(solution_json, f"{workpiece_name}_ppo_trained")
        print(f"  Best MOI achieved: {best_reward:.2e}")
    
    print(f"\n{'='*80}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for fixture layout optimization"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["placement", "rotation"],
        default="placement",
        help="Which agent to train (placement or rotation)",
    )
    parser.add_argument(
        "--workpiece",
        type=str,
        default="simple_stair_step",
        help="Workpiece name to train on",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO batch size",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO n_steps (rollout buffer size)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to pre-trained model to continue training from",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render evaluation episodes",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency in timesteps",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Use hybrid discrete+continuous action space (default)",
    )
    parser.add_argument(
        "--continuous",
        action="store_false",
        dest="hybrid",
        help="Use pure continuous action space (legacy, inefficient)",
    )
    parser.add_argument(
        "--discrete",
        action="store_false",
        dest="hybrid",
        help="Use pure discrete action space",
    )
    
    args = parser.parse_args()
    
    try:
        train_ppo(
            agent_type=args.agent,
            workpiece_name=args.workpiece,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            load_model_path=args.load,
            render=args.render,
            eval_freq=args.eval_freq,
            use_continuous=(not args.hybrid),
            use_hybrid=args.hybrid,
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
