#!/usr/bin/env python3
"""
Quick debug script to verify hybrid action space and flexibility bonus (5 steps max).
"""
import sys
sys.path.insert(0, '.')

from rl_graphical_agent import FixtureLayoutEnv
from stable_baselines3 import PPO
import numpy as np

def test_hybrid_quick():
    """Test hybrid action space with minimal steps."""
    print("\n" + "="*80)
    print("HYBRID ACTION SPACE DEBUG (5 STEPS)")
    print("="*80)
    
    # Create environment with hybrid action space
    env = FixtureLayoutEnv(
        workpiece_name="simple_stair_step",
        verbose=True,
        use_continuous_action_space=False,
        use_hybrid_action_space=True
    )
    
    print(f"\n[OK] Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Hybrid offset range: +/-{env.hybrid_offset_range}mm")
    
    obs, info = env.reset()
    print(f"\n[OK] Environment reset")
    print(f"  Initial observation shape: {obs.shape}")
    
    # Test 5 random actions
    print(f"\n" + "-"*80)
    print("TESTING 5 RANDOM ACTIONS")
    print("-"*80)
    
    for step in range(5):
        action = env.action_space.sample()
        print(f"\nStep {step+1}/5:")
        print(f"  Action: {action}")
        print(f"    fixture_type_norm: {action[0]:.3f}")
        print(f"    grid_pos_norm: {action[1]:.3f}")
        print(f"    offset_x: {action[2]:.3f}")
        print(f"    offset_y: {action[3]:.3f}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Reward: {reward:.2f}")
        print(f"  Success: {info.get('placement_success', False)}")
        print(f"  Reason: {info.get('reason', 'unknown')}")
        if 'valid_positions_before' in info:
            print(f"  Valid positions: {info['valid_positions_before']} -> {info['valid_positions_after']}")
            flexibility_ratio = info['valid_positions_after'] / max(1, info['valid_positions_before'])
            print(f"  Flexibility ratio: {flexibility_ratio:.2%}")
        print(f"  Fixtures placed: {info['fixtures_count']}")
        
        if terminated or truncated:
            print(f"  Episode ended (terminated={terminated}, truncated={truncated})")
            break
    
    env.close()
    
    print(f"\n" + "="*80)
    print("HYBRID DEBUG COMPLETE - ACTION SPACE WORKING")
    print("="*80 + "\n")

def test_training_quick():
    """Test PPO training for just 5 steps."""
    print("\n" + "="*80)
    print("HYBRID PPO TRAINING DEBUG (5 STEPS)")
    print("="*80)
    
    env = FixtureLayoutEnv(
        workpiece_name="simple_stair_step",
        use_continuous_action_space=False,
        use_hybrid_action_space=True
    )
    
    print(f"\n[OK] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2,  # Minimal rollout (must be > 1)
        batch_size=2,  # Must be > 1
        verbose=1
    )
    
    print(f"\n[OK] Training for 5 steps...")
    model.learn(total_timesteps=5)
    
    env.close()
    
    print(f"\n" + "="*80)
    print("TRAINING DEBUG COMPLETE - PPO WORKING")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_hybrid_quick()
    test_training_quick()
