#!/usr/bin/env python3
"""
Training script for Fixture Placement on Bars Agent (Stage 2)

Takes bar positions from Stage 1 (bar positioning agent) and trains
an agent to place fixtures on those bars, maximizing MOI while ensuring
every bar has at least 1 fixture.

Usage:
    # First train bar positioning
    python train_bar_positioning.py --workpiece simple_stair_step --timesteps 20000
    
    # Then train fixture placement with those bar positions
    python train_fixture_on_bars.py --workpiece simple_stair_step --timesteps 20000 --use-best-bars
"""

import argparse
from pathlib import Path
import sys
import json
import pickle
import shutil
import copy

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
    from matplotlib.patches import Polygon as MplPolygon
    import numpy as np
except ImportError:
    raise SystemExit(
        "matplotlib and numpy are required. Install them with: pip install matplotlib numpy"
    )

from rl_fixture_on_bars_agent import FixtureOnBarsEnv, FIXTURE_DIMENSIONS, FixtureType
from rl_bar_positioning_agent import BarPositioningEnv

ROOT = Path(__file__).resolve().parent


def save_detailed_solution(
    env: FixtureOnBarsEnv,
    workpiece_name: str,
    objective_value: float,
    principal_moment_I: float,
    principal_moment_J: float,
):
    """
    Save detailed solution with fixture coordinates and MOI values.
    
    Saves to: python/results/rl_discrete_fixture_placement_{workpiece_name}.json
    
    Format includes:
    - Fixture coordinates (x, y, corners x1-x3, y1-y3)
    - Fixture types and selection info
    - Bar positions
    - MOI values (I, J, and sum)
    """
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / f"rl_discrete_fixture_placement_{workpiece_name}.json"
    
    # Extract fixture information
    solution_data = {
        "x": [],        # Top-left x for each fixture
        "y": [],        # Top-left y for each fixture
        "x1": [],       # Top-right x
        "y1": [],       # Top-right y
        "x2": [],       # Bottom-right x
        "y2": [],       # Bottom-right y
        "x3": [],       # Bottom-left x
        "y3": [],       # Bottom-left y
        "angle": [],    # Rotation angle (all 0 for now)
        "fixtures_center_x": [],
        "fixtures_center_y": [],
        "bars_center": [],
        "selected_fixture": [],
        "fixture_type": [],
    }
    
    for fixture_idx, fixture in enumerate(env.fixtures):
        width, height = FIXTURE_DIMENSIONS[fixture.type_id]
        
        # Top-left corner
        x = fixture.x
        y = fixture.y
        
        # Calculate all corners
        x_min, x_max = x, x + width
        y_min, y_max = y, y + height
        
        solution_data["x"].append(x)
        solution_data["y"].append(y)
        solution_data["x1"].append(x_max)  # Top-right
        solution_data["y1"].append(y)
        solution_data["x2"].append(x_max)  # Bottom-right
        solution_data["y2"].append(y_max)
        solution_data["x3"].append(x)      # Bottom-left
        solution_data["y3"].append(y_max)
        solution_data["angle"].append(0)
        
        # Center coordinates
        center_x = x + width / 2
        center_y = y + height / 2
        solution_data["fixtures_center_x"].append(center_x)
        solution_data["fixtures_center_y"].append(center_y)
        
        # Find which bar this fixture is on
        bar_idx = None
        for i, bar_x in enumerate(env.bar_positions):
            if abs(bar_x - center_x) < width / 2 + 1:  # Small tolerance
                bar_idx = i
                break
        
        solution_data["bars_center"].append(env.bar_positions[bar_idx] if bar_idx is not None else 0)
        solution_data["selected_fixture"].append(fixture_idx + 1)
        solution_data["fixture_type"].append(fixture.type_id)
    
    # Add MOI values
    solution_data["objective_value"] = int(objective_value) if objective_value else 0
    solution_data["principal_moment_I"] = int(principal_moment_I) if principal_moment_I else 0
    solution_data["principal_moment_J"] = int(principal_moment_J) if principal_moment_J else 0
    solution_data["moi_note"] = "Values computed using proper moment of inertia calculation (not simplified RL proxy)"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(solution_data, f, indent=2)
    
    print(f"[OK] Detailed solution saved to: {output_file}")


def visualize_fixture_placement(
    model_path: Path,
    bar_positions: list,
    workpiece_name: str,
    output_path: Path = None,
):
    """
    Visualize best fixture placement found by trained agent.
    
    Generates PNG showing:
    - Workpiece boundary and holes
    - Bars with fixtures placed
    - Fixture types (colors) and positions
    - Total MOI achieved
    
    Uses same style as bar visualization for consistency.
    """
    if output_path is None:
        output_path = ROOT / f"fixture_placement_visualization_{workpiece_name}.png"
    
    # Create a FRESH unwrapped environment
    print(f"[INFO] Creating fresh environment for visualization...")
    env = FixtureOnBarsEnv(
        bar_positions=bar_positions,
        workpiece_name=workpiece_name,
        render_mode=None,
        verbose=False
    )
    
    print(f"[INFO] Environment created with {len(bar_positions)} bars at positions: {[f'{x:.1f}' for x in bar_positions]}")
    
    # Load model bound to environment for proper inference
    try:
        conflicting_dir = model_path
        if conflicting_dir.exists() and conflicting_dir.is_dir():
            shutil.rmtree(conflicting_dir)
        
        model = PPO.load(str(model_path), env=env)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        env.close()
        return
    
    # Run inference to get best solution (deterministic for reproducibility)
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    print(f"[INFO] Running inference (deterministic) to visualize best solution...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    num_fixtures = len(env.fixtures)
    bars_filled = sum(1 for c in env.fixtures_per_bar if c > 0)
    print(f"[INFO] Solution: {num_fixtures} fixtures in {bars_filled}/{len(bar_positions)} bars, MOI={env.cumulative_moi:.2e}, Reward={total_reward:.2f}")
    
    # Create figure matching bar visualization style
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Draw workpiece boundary (polygon with light fill)
    workpiece_vertices = env.workpiece_data['vertices']
    workpiece_poly = patches.Polygon(
        workpiece_vertices,
        closed=True,
        edgecolor='black',
        facecolor='lightgray',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(workpiece_poly)
    
    # Draw holes (excluded regions) - circles with [cx, cy, radius] format
    if 'holes' in env.workpiece_data:
        for hole in env.workpiece_data['holes']:
            if isinstance(hole, (list, tuple)) and len(hole) == 3:
                # Circle hole: [cx, cy, radius]
                circle = patches.Circle(
                    (hole[0], hole[1]),
                    hole[2],
                    edgecolor='red',
                    facecolor='red',
                    alpha=0.2,
                    linewidth=1.5,
                    linestyle='--',
                    label='Holes' if hole == env.workpiece_data['holes'][0] else ''
                )
                ax.add_patch(circle)
    
    # Color scheme for bars and fixtures (matching tab10)
    bar_colors = plt.cm.tab10(np.arange(len(bar_positions)))
    fixture_colors = {
        FixtureType.TYPE_1: '#FF6B6B',  # Red
        FixtureType.TYPE_2: '#4ECDC4',  # Teal
    }
    
    # Draw bars with center lines (matching bar visualization style)
    for i, center_x in enumerate(bar_positions):
        left = center_x - 145.0 / 2
        
        # Bar rectangle
        rect = patches.Rectangle(
            (left, env.workpiece_min_y),
            145.0,
            env.workpiece_height,
            linewidth=2,
            edgecolor=bar_colors[i],
            facecolor=bar_colors[i],
            alpha=0.15,  # Light fill to show bar region
            label=f'Bar {i+1} (x={center_x:.1f})'
        )
        ax.add_patch(rect)
        
        # Center line
        ax.plot([center_x, center_x], [env.workpiece_min_y, env.workpiece_max_y], 
                color=bar_colors[i], linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Draw fixtures with darker colors
    fixture_type_counts = {FixtureType.TYPE_1: 0, FixtureType.TYPE_2: 0}
    
    for fixture in env.fixtures:
        width, height = FIXTURE_DIMENSIONS[fixture.type_id]
        color = fixture_colors[fixture.type_id]
        
        fixture_rect = patches.Rectangle(
            (fixture.x, fixture.y),
            width,
            height,
            linewidth=1.5,
            edgecolor='black',
            facecolor=color,
            alpha=0.8
        )
        ax.add_patch(fixture_rect)
        
        # Center mark
        center_x = fixture.x + width / 2
        center_y = fixture.y + height / 2
        ax.plot(center_x, center_y, 'ko', markersize=4)
        
        fixture_type_counts[fixture.type_id] += 1
    
    # Format plot
    ax.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    
    title_text = f"Fixture Placement Solution - {workpiece_name}\n"
    title_text += f"Bars: {len(bar_positions)} | Fixtures: {num_fixtures} (Type1: {fixture_type_counts[FixtureType.TYPE_1]}, Type2: {fixture_type_counts[FixtureType.TYPE_2]}) | "
    title_text += f"MOI: {env.cumulative_moi:.2e} | Reward: {total_reward:.2f}"
    
    ax.set_title(title_text, fontsize=13, fontweight='bold')
    
    # Create legend with all elements
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor=fixture_colors[FixtureType.TYPE_1], edgecolor='black', label=f'Type 1 (145×145mm): {fixture_type_counts[FixtureType.TYPE_1]}'),
        patches.Rectangle((0, 0), 1, 1, facecolor=fixture_colors[FixtureType.TYPE_2], edgecolor='black', label=f'Type 2 (180×65mm): {fixture_type_counts[FixtureType.TYPE_2]}'),
        patches.Circle((0, 0), 1, edgecolor='red', facecolor='red', alpha=0.2, linestyle='--', label='Holes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(env.workpiece_min_x - 50, env.workpiece_max_x + 50)
    ax.set_ylim(env.workpiece_min_y - 50, env.workpiece_max_y + 50)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_path}")
    
    # Show
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    env.close()


def get_best_bar_positions(workpiece_name: str):
    """
    Get best bar positions from saved training results (Stage 1).
    
    First tries to load from automatically saved JSON file.
    Falls back to loading from trained model if JSON not found.
    
    Returns: List of bar center x-coordinates
    """
    # Try to load from automatically saved results
    saved_positions_file = ROOT / "models" / f"best_bar_positions_{workpiece_name}.json"
    
    if saved_positions_file.exists():
        print(f"\n[OK] Loading saved bar positions from: {saved_positions_file.name}")
        with open(saved_positions_file, 'r') as f:
            data = json.load(f)
        bar_positions = data['bar_positions']
        print(f"     Found {len(bar_positions)} bars from training (reward={data['best_reward']:.2f})")
        print(f"     Positions: {[f'{x:.1f}' for x in bar_positions]}")
        return bar_positions
    
    # Fallback: load from model
    model_path = ROOT / "models" / f"best_bar_positioning_{workpiece_name}.zip"
    
    if not model_path.exists():
        print(f"\n[ERROR] Neither saved positions nor model found for {workpiece_name}")
        print(f"        Please train bar positioning first:")
        print(f"        python train_bar_positioning.py --workpiece {workpiece_name}")
        return None
    
    print(f"\n[WARNING] Saved positions not found; loading from model: {model_path.name}")
    
    # Create environment and load model
    env = BarPositioningEnv(workpiece_name=workpiece_name, verbose=False)
    
    # Clean up any conflicting directory before loading
    conflicting_dir = model_path.with_suffix('')
    if conflicting_dir.exists() and conflicting_dir.is_dir():
        shutil.rmtree(conflicting_dir)
    
    model = PPO.load(str(model_path.with_suffix('')), env=env)
    
    # Generate bar positions - run to completion
    obs, info = env.reset()
    done = False
    step_count = 0
    
    print(f"[INFO] Running bar positioning model...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        if step_count % 10 == 0:
            print(f"  Step {step_count}: {len(env.bars)} bars placed")
    
    bar_positions = env.get_bar_positions()
    print(f"[OK] Bar positioning complete: {len(bar_positions)} bars at positions {[f'{x:.1f}' for x in bar_positions]}")
    
    env.close()
    
    return bar_positions


def train_fixture_on_bars(
    workpiece_name: str = "simple_stair_step",
    bar_positions: list = None,
    use_best_bars: bool = False,
    total_timesteps: int = 20000,
    learning_rate: float = 0.0003,
    batch_size: int = 64,
    n_steps: int = 2048,
    eval_freq: int = 2048,
    render: bool = False,
):
    """
    Train fixture placement agent on fixed bar positions.
    
    Args:
        workpiece_name: Name of the workpiece
        bar_positions: List of bar center x-coordinates (or None to use default)
        use_best_bars: If True, load bar positions from trained stage 1 model
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        batch_size: Batch size
        n_steps: Steps per rollout
        eval_freq: Evaluation frequency
        render: Whether to render
    """
    print(f"\n{'='*80}")
    print(f"Training FIXTURE PLACEMENT Agent (Stage 2)")
    print(f"{'='*80}")
    print(f"Workpiece: {workpiece_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    
    # Get bar positions
    if use_best_bars:
        print(f"\n[INFO] Loading best bar positions from Stage 1 model...")
        bar_positions = get_best_bar_positions(workpiece_name)
        if bar_positions is None:
            print("Error: Could not load best bar positions")
            return
        print(f"[DEBUG] Got {len(bar_positions)} bars from Stage 1: {[f'{x:.1f}' for x in bar_positions]}")
    elif bar_positions is None:
        # Default: use evenly spaced bars
        env_temp = BarPositioningEnv(workpiece_name=workpiece_name, verbose=False)
        # Place bars at 1/4, 1/2, 3/4 of workpiece width
        width = env_temp.workpiece_width
        min_x = env_temp.workpiece_min_x
        bar_positions = [
            min_x + width * 0.25,
            min_x + width * 0.75,
        ]
        env_temp.close()
        print(f"[INFO] Using default bar positions: {[f'{x:.1f}' for x in bar_positions]}")
    
    print(f"\n[OK] Bar positions for fixture placement: {[f'{x:.1f}' for x in bar_positions]}")
    print(f"     Total bars: {len(bar_positions)}")
    
    # Create environment with fixed bar positions
    print(f"\nCreating environment with fixed bar positions...")
    env = FixtureOnBarsEnv(
        bar_positions=bar_positions,
        workpiece_name=workpiece_name,
        render_mode=None,
        verbose=False
    )
    
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Number of bars in environment: {env.num_bars}")
    print(f"  Bar positions in environment: {[f'{x:.1f}' for x in env.bar_positions]}")
    
    # Create directories
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    model_name = f"ppo_fixture_on_bars_{workpiece_name}"
    best_model_path = models_dir / f"best_fixture_on_bars_{workpiece_name}"
    
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
    eval_env = FixtureOnBarsEnv(
        bar_positions=bar_positions,
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
    print(f"Training started with reward based on ACTUAL MOMENT OF INERTIA:")
    print(f"  - Each fixture placement: Reward = MOI_increase × 1e-7 + 10.0 (if first in bar)")
    print(f"  - Completion bonus: 100.0 + (Total_MOI / 1e9) × 50.0")
    print(f"  - Objective: Maximize principal moments I and J")
    print(f"Constraints: No hole overlaps, valid y-position, vertical spacing")
    print(f"Note: MOI computed using proper moment of inertia calculation")
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
    
    # Save bar positions config
    bar_config_path = best_model_path.parent / f"{best_model_path.name}_bars.pkl"
    with open(bar_config_path, 'wb') as f:
        pickle.dump(bar_positions, f)
    print(f"\n[OK] Bar positions saved to: {bar_config_path}")
    
    # Evaluate best model
    print(f"\nEvaluating best model to find best solution...\n")
    try:
        # Clean up any conflicting directory with the same name (without .zip)
        conflicting_dir = best_model_path
        if conflicting_dir.exists() and conflicting_dir.is_dir():
            shutil.rmtree(conflicting_dir)
            print(f"[OK] Removed conflicting directory: {conflicting_dir}")
        
        best_model = PPO.load(str(best_model_path), env=eval_env)
        
        # Track best solution during evaluation
        best_episode_reward = float('-inf')
        best_episode_fixtures = 0
        best_episode_moi = 0.0
        best_episode_moi_i = 0.0
        best_episode_moi_j = 0.0
        best_episode_env_copy = None  # Will hold deep copy of best episode
        
        total_reward = 0
        total_fixtures = 0
        total_moi = 0
        
        for episode in range(5):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = best_model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            num_fixtures = len(eval_env.fixtures)
            moi = eval_env.cumulative_moi
            total_reward += episode_reward
            total_fixtures += num_fixtures
            total_moi += moi
            
            bars_filled = sum(1 for count in eval_env.fixtures_per_bar if count > 0)
            print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, fixtures={num_fixtures}, bars_filled={bars_filled}/{len(bar_positions)}, MOI={moi:.2e}")
            
            # Track best episode (by total fixtures first, then by MOI)
            if num_fixtures > best_episode_fixtures or (num_fixtures == best_episode_fixtures and moi > best_episode_moi):
                best_episode_reward = episode_reward
                best_episode_fixtures = num_fixtures
                best_episode_moi = moi
                best_episode_moi_i = eval_env.principal_moment_i
                best_episode_moi_j = eval_env.principal_moment_j
                
                # **SAVE BEST EPISODE STATE IMMEDIATELY** - deep copy before next reset
                best_episode_env_copy = copy.deepcopy(eval_env)
                print(f"    ✓ Saving this as best episode (fixtures={num_fixtures}, MOI={moi:.2e})")
        
        print(f"\n  Mean reward: {total_reward / 5:.2f}")
        print(f"  Mean fixtures: {total_fixtures / 5:.1f}")
        print(f"  Mean MOI: {total_moi / 5:.2e}")
        print(f"\n  Best episode: {best_episode_fixtures} fixtures, MOI={best_episode_moi:.2e}")
        
        # Save best solution config
        best_solution_file = models_dir / f"best_fixture_solution_{workpiece_name}.json"
        solution_data = {
            'workpiece': workpiece_name,
            'bar_positions': bar_positions,
            'num_bars': len(bar_positions),
            'best_fixtures': best_episode_fixtures,
            'best_moi': float(best_episode_moi),
            'best_reward': float(best_episode_reward),
        }
        with open(best_solution_file, 'w') as f:
            json.dump(solution_data, f, indent=2)
        print(f"\n[OK] Best solution config saved to: {best_solution_file}")
        
        # Save detailed solution with fixture coordinates and MOI
        # Use the saved best episode environment, not a fresh deterministic run
        if best_episode_env_copy is not None:
            print(f"\n[OK] Saving detailed solution from best episode...")
            save_detailed_solution(
                env=best_episode_env_copy,
                workpiece_name=workpiece_name,
                objective_value=best_episode_moi,
                principal_moment_I=best_episode_moi_i,
                principal_moment_J=best_episode_moi_j,
            )
        else:
            print(f"\n[WARNING] No best episode environment to save")
        
    except Exception as e:
        print(f"\n[WARNING] Could not evaluate model: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()
    eval_env.close()
    
    # Visualize best solution
    print(f"\nGenerating visualization of best solution...")
    try:
        visualize_fixture_placement(
            model_path=best_model_path,
            bar_positions=bar_positions,
            workpiece_name=workpiece_name,
        )
    except Exception as e:
        print(f"[WARNING] Could not generate visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Best model: {best_zip_path.name}")
    print(f"Bar positions saved: {bar_config_path.name}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Train fixture placement on bars agent')
    parser.add_argument('--workpiece', default='simple_stair_step', help='Workpiece name')
    parser.add_argument('--timesteps', type=int, default=20000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per rollout')
    parser.add_argument('--eval-freq', type=int, default=2048, help='Evaluation frequency')
    parser.add_argument('--use-best-bars', action='store_true', help='Load bar positions from trained stage 1 model')
    parser.add_argument('--render', action='store_true', help='Render during training')
    
    args = parser.parse_args()
    
    train_fixture_on_bars(
        workpiece_name=args.workpiece,
        use_best_bars=args.use_best_bars,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        eval_freq=args.eval_freq,
        render=args.render,
    )


if __name__ == "__main__":
    main()
