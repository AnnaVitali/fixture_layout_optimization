#!/usr/bin/env python3
"""
Debug script to verify bar positioning model output
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from rl_bar_positioning_agent import BarPositioningEnv

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_bar_positioning_simple_stair_step.zip"

print(f"\n{'='*80}")
print(f"Debugging Bar Positioning Model")
print(f"{'='*80}\n")

print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}\n")

if not MODEL_PATH.exists():
    print("[ERROR] Model not found!")
    sys.exit(1)

# Create environment
print("Creating environment...")
env = BarPositioningEnv(workpiece_name="simple_stair_step", verbose=True)

print(f"\nEnvironment created:")
print(f"  Workpiece width: {env.workpiece_width:.1f} mm")
print(f"  Workpiece height: {env.workpiece_height:.1f} mm")
print(f"  Max bars: {env.max_bars}")

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
conflicting_dir = MODEL_PATH.with_suffix('')
if conflicting_dir.exists() and conflicting_dir.is_dir():
    shutil.rmtree(conflicting_dir)

model = PPO.load(str(MODEL_PATH.with_suffix('')), env=env)

# Run inference
print(f"\nRunning inference...")
obs, info = env.reset()
done = False
step = 0
episode_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    done = terminated or truncated
    step += 1
    
    if step % 5 == 0 or done:
        print(f"  Step {step}: {len(env.bars)} bars, reward={reward:.2f}, action={action}")

# Final result
bar_positions = env.get_bar_positions()
print(f"\n{'='*80}")
print(f"FINAL RESULT:")
print(f"{'='*80}")
print(f"Total steps: {step}")
print(f"Episode reward: {episode_reward:.2f}")
print(f"Number of bars: {len(bar_positions)}")
print(f"Bar positions: {[f'{x:.1f}' for x in bar_positions]}")
print(f"Bar spacing distances:")
if len(bar_positions) >= 2:
    sorted_bars = sorted(bar_positions)
    for i in range(len(sorted_bars) - 1):
        dist = sorted_bars[i + 1] - sorted_bars[i]
        print(f"  Bar {i+1} to Bar {i+2}: {dist:.1f} mm")
else:
    print("  (Only 1 bar, no spacing)")

env.close()
print(f"\n{'='*80}\n")
