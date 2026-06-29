#!/usr/bin/env python3
"""
Training script for Bar Positioning Agent (Stage 1)

Trains an agent to position bars on a workpiece, maximizing:
- Distance between bar centers
- Total area covered

Usage:
    python train_bar_positioning.py --workpiece simple_stair_step --timesteps 50000
"""

import argparse
from pathlib import Path
import sys
import json
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    raise SystemExit(
        "stable-baselines3 is required. Install it with: pip install stable-baselines3[extra]"
    )

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from rl_bar_positioning_agent import BarPositioningEnv, BAR_WIDTH

ROOT = Path(__file__).resolve().parent


def visualize_bar_positions(bar_positions: list, workpiece_name: str, env: BarPositioningEnv):
    """
    Visualize bar positions on the workpiece.
    
    Args:
        bar_positions: List of bar center x-coordinates
        workpiece_name: Name of the workpiece
        env: BarPositioningEnv instance (for workpiece info)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[INFO] Matplotlib not available, skipping visualization")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Draw workpiece boundary
    vertices = env.workpiece_data['vertices']
    xs = [v[0] for v in vertices] + [vertices[0][0]]
    ys = [v[1] for v in vertices] + [vertices[0][1]]
    ax.plot(xs, ys, 'k-', linewidth=2, label='Workpiece boundary')
    ax.fill(xs, ys, alpha=0.05, color='gray')
    
    # Draw holes if they exist
    if 'holes' in env.workpiece_data:
        for hole in env.workpiece_data['holes']:
            if len(hole) == 3:
                circle = plt.Circle((hole[0], hole[1]), hole[2], fill=False, edgecolor='red', linestyle='--', linewidth=1)
                ax.add_patch(circle)
    
    # Draw bars
    colors = plt.cm.tab10(range(len(bar_positions)))
    for i, center_x in enumerate(bar_positions):
        # Bar bounds (in workpiece coordinates)
        left = center_x - BAR_WIDTH / 2
        right = center_x + BAR_WIDTH / 2
        
        # Bar rectangle (from min_y to max_y)
        rect = patches.Rectangle(
            (left, env.workpiece_min_y),
            BAR_WIDTH,
            env.workpiece_height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.3,
            label=f'Bar {i+1} (x={center_x:.1f})'
        )
        ax.add_patch(rect)
        
        # Center line
        ax.plot([center_x, center_x], [env.workpiece_min_y, env.workpiece_max_y], 
                color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Mark center
        ax.plot(center_x, (env.workpiece_min_y + env.workpiece_max_y) / 2, 
                'o', color=colors[i], markersize=8)
    
    # Add spacing indicators
    if len(bar_positions) >= 2:
        sorted_bars = sorted(bar_positions)
        for i in range(len(sorted_bars) - 1):
            dist = sorted_bars[i + 1] - sorted_bars[i]
            mid_x = (sorted_bars[i] + sorted_bars[i + 1]) / 2
            y_text = env.workpiece_max_y + 20
            ax.text(mid_x, y_text, f'{dist:.0f}mm', ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Format plot
    ax.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    ax.set_title(f'Bar Positioning Solution - {workpiece_name}\n({len(bar_positions)} bars)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = Path(__file__).parent / f"bar_positions_visualization_{workpiece_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_file}")
    
    # Show
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def train_bar_positioning(
    workpiece_name: str = "simple_stair_step",
    total_timesteps: int = 20000,  # Reduced for faster iteration
    learning_rate: float = 0.0005,  # INCREASED to learn faster
    batch_size: int = 64,
    n_steps: int = 2048,
    eval_freq: int = 2048,
    render: bool = False,
):
    """
    Train bar positioning agent.
    
    Args:
        workpiece_name: Name of the workpiece
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        batch_size: Batch size
        n_steps: Steps per rollout
        eval_freq: Evaluation frequency
        render: Whether to render during training
    """
    print(f"\n{'='*80}")
    print(f"Training BAR POSITIONING Agent (Stage 1)")
    print(f"{'='*80}")
    print(f"Workpiece: {workpiece_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"N steps: {n_steps}")
    
    # Create environment
    print(f"\nCreating environment...")
    env = BarPositioningEnv(
        workpiece_name=workpiece_name,
        render_mode=None,
        verbose=False
    )
    
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Create directories
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    model_name = f"ppo_bar_positioning_{workpiece_name}"
    best_model_path = models_dir / f"best_bar_positioning_{workpiece_name}"
    
    # Create PPO model
    print(f"\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        verbose=1,
    )
    
    # Setup callbacks
    print(f"\nSetting up callbacks...")
    eval_env = BarPositioningEnv(
        workpiece_name=workpiece_name,
        render_mode=None,
        verbose=False
    )
    
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
    print(f"Training started with reward: placement=500.0, bar_count=150×count, edge_bonus=0-200.0")
    print(f"Expected: 3+ bars at efficient positions")
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
    
    # Save best model
    print(f"\n\n{'='*80}")
    print(f"Training completed!")
    print(f"{'='*80}")
    
    best_zip_path = best_model_path.parent / f"{best_model_path.name}.zip"
    if best_zip_path.exists():
        print(f"\n[OK] Best model saved to: {best_zip_path}")
    else:
        model.save(str(best_model_path))
        print(f"\n[OK] Model saved to: {best_model_path}.zip")
        best_zip_path = Path(str(best_model_path) + ".zip")
    
    # Evaluate best model
    print(f"\nEvaluating best model...\n")
    try:
        # Clean up any conflicting directory with the same name (without .zip)
        conflicting_dir = best_model_path
        if conflicting_dir.exists() and conflicting_dir.is_dir():
            shutil.rmtree(conflicting_dir)
            print(f"[OK] Removed conflicting directory: {conflicting_dir}")
        
        best_model = PPO.load(str(best_model_path), env=eval_env)
        
        total_reward = 0
        best_episode_bars = []
        best_episode_reward = 0
        
        for episode in range(5):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = best_model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            num_bars = len(eval_env.bars)
            bar_positions = eval_env.get_bar_positions()
            total_reward += episode_reward
            
            print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, bars={num_bars}, positions={[f'{x:.1f}' for x in bar_positions]}")
            
            # Track best episode
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_bars = bar_positions
        
        print(f"\n  Mean reward: {total_reward / 5:.2f}")
        print(f"  Best episode bars: {[f'{x:.1f}' for x in best_episode_bars]}")
        
        # SAVE best bar positions to JSON file
        if best_episode_bars:
            bar_positions_file = models_dir / f"best_bar_positions_{workpiece_name}.json"
            bar_data = {
                'workpiece': workpiece_name,
                'bar_positions': best_episode_bars,
                'num_bars': len(best_episode_bars),
                'best_reward': float(best_episode_reward),
            }
            with open(bar_positions_file, 'w') as f:
                json.dump(bar_data, f, indent=2)
            print(f"\n[OK] Best bar positions saved to: {bar_positions_file}")
        
        # Visualize best solution
        if best_episode_bars:
            print(f"\n[INFO] Generating visualization of best bar positions...")
            visualize_bar_positions(best_episode_bars, workpiece_name, eval_env)
        
    except Exception as e:
        print(f"\n[WARNING] Could not evaluate model: {e}")
    
    env.close()
    eval_env.close()
    
    print(f"\n{'='*80}")
    print(f"Best model: {best_zip_path.name}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Train bar positioning agent')
    parser.add_argument('--workpiece', default='simple_stair_step', help='Workpiece name')
    parser.add_argument('--timesteps', type=int, default=20000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--eval-freq', type=int, default=2048, help='Evaluation frequency')
    parser.add_argument('--render', action='store_true', help='Render during training')
    
    args = parser.parse_args()
    
    train_bar_positioning(
        workpiece_name=args.workpiece,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        eval_freq=args.eval_freq,
        render=args.render,
    )


if __name__ == "__main__":
    main()
