"""
DQN (Deep Q-Network) Reinforcement Learning Agent for Fixture Layout Optimization.

This implementation uses actual RL learning, training a neural network to learn
good fixture placement policies through trial-and-error.

The agent learns to:
1. Maximize moment of inertia (MOI)
2. Place as many fixtures as possible
3. Maintain flexibility for future placements

Usage:
    python train_dqn_graphical_agent.py --workpiece simple_stair_step --episodes 500 --render
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from rl_graphical_agent import FixtureLayoutEnv


class DQNNetwork(nn.Module):
    """
    Deep Q-Network: maps observation to Q-values for each action.
    
    Architecture:
    - Input: observation vector (normalized state)
    - Hidden: 256 -> 256 neurons with ReLU
    - Output: Q-value for each discrete action
    """
    
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(observation_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ExperienceReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores (state, action, reward, next_state, done) tuples.
    Sampling from buffer breaks correlation between sequential experiences.
    """
    
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = np.array([self.buffer[i][0] for i in indices])
        actions = np.array([self.buffer[i][1] for i in indices])
        rewards = np.array([self.buffer[i][2] for i in indices])
        next_states = np.array([self.buffer[i][3] for i in indices])
        dones = np.array([self.buffer[i][4] for i in indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for fixture layout optimization.
    
    Learns through:
    1. Exploration: epsilon-greedy action selection
    2. Experience replay: break correlation in training
    3. Target network: stable Q-value targets
    4. Experience buffer: store and sample transitions
    """
    
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        hidden_size: int = 256,
        buffer_size: int = 100000,
        batch_size: int = 32
    ):
        """Initialize DQN agent."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Using device: {self.device}")
        
        # Networks
        self.network = DQNNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay
        self.buffer = ExperienceReplayBuffer(buffer_size)
        
        # Tracking
        self.update_count = 0
        self.target_update_frequency = 1000
    
    def select_action(self, state: np.ndarray, action_space, training: bool = True) -> int:
        """
        Select action using epsilon-greedy strategy during training,
        or greedy strategy during evaluation.
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return action_space.sample()
        else:
            # Exploit: use network to select best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action
    
    def update(self) -> Optional[float]:
        """
        Update network using experience replay.
        
        Returns:
            Loss value, or None if buffer is too small
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert actions to flat indices (handle both int and array actions)
        actions_flat = []
        for action in actions:
            if isinstance(action, np.ndarray):
                # MultiDiscrete: convert to flat index
                fixture_type, pair_idx = action[0], action[1]
                flat_idx = int(fixture_type) + int(pair_idx) * 3
                actions_flat.append(flat_idx)
            else:
                # Already a scalar
                actions_flat.append(int(action))
        actions_flat = np.array(actions_flat)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions_flat).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        q_values = self.network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
            # Set Q-values to 0 for terminal states
            target_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save network weights."""
        torch.save(self.network.state_dict(), filepath)
        print(f"[DQN] Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load network weights."""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.network.state_dict())
        print(f"[DQN] Loaded model from {filepath}")


def train_episode(
    agent: DQNAgent,
    env: FixtureLayoutEnv,
    render: bool = False
) -> Tuple[float, int]:
    """
    Run one training episode.
    
    Returns:
        (cumulative_reward, num_fixtures_placed)
    """
    obs, info = env.reset()
    cumulative_reward = 0.0
    done = False
    
    while not done:
        # Select action using epsilon-greedy
        action = agent.select_action(obs, env.action_space, training=True)
        
        # Convert discrete action to MultiDiscrete if needed
        if isinstance(action, int):
            # For MultiDiscrete action space, convert flat index to (fixture_type, pair_idx)
            fixture_type = action % 3
            pair_idx = action // 3
            action = np.array([fixture_type, pair_idx])
        
        # Take step in environment
        next_obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # Store experience
        agent.buffer.add(obs, action, reward, next_obs, done)
        
        # Update network
        loss = agent.update()
        
        cumulative_reward += reward
        obs = next_obs
        
        if render:
            env.render()
    
    num_fixtures = len(env.fixtures)
    return cumulative_reward, num_fixtures


def evaluate_episode(
    agent: DQNAgent,
    env: FixtureLayoutEnv,
    render: bool = False
) -> Tuple[float, int, float]:
    """
    Run one evaluation episode (no learning, greedy policy).
    
    Returns:
        (cumulative_reward, num_fixtures_placed, final_moi)
    """
    obs, info = env.reset()
    cumulative_reward = 0.0
    done = False
    
    while not done:
        # Select action greedily (no exploration)
        action = agent.select_action(obs, env.action_space, training=False)
        
        # Convert discrete action to MultiDiscrete if needed
        if isinstance(action, int):
            fixture_type = action % 3
            pair_idx = action // 3
            action = np.array([fixture_type, pair_idx])
        
        # Take step in environment
        next_obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        cumulative_reward += reward
        obs = next_obs
        
        if render:
            env.render()
    
    num_fixtures = len(env.fixtures)
    final_moi = env.cumulative_moment
    
    return cumulative_reward, num_fixtures, final_moi


def main():
    parser = argparse.ArgumentParser(description="DQN Training for Fixture Layout Optimization")
    parser.add_argument("--workpiece", type=str, default="simple_stair_step",
                       help="Workpiece name")
    parser.add_argument("--episodes", type=int, default=500,
                       help="Number of training episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during training")
    parser.add_argument("--eval-freq", type=int, default=50,
                       help="Evaluate every N episodes")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden layer size in DQN network")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for network updates")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--output-dir", type=str, default="./dqn_models",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create environment
    print(f"[Main] Creating environment for '{args.workpiece}'...")
    env = FixtureLayoutEnv(
        workpiece_name=args.workpiece,
        render_mode="human" if args.render else None,
        verbose=False,
        use_hybrid_action_space=False,
        use_continuous_action_space=False,  # Use discrete action space
        grid_resolution=1.0,  # 1mm grid resolution for finer control
        max_steps=20
    )
    
    # Create DQN agent
    print(f"[Main] Initializing DQN agent...")
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.nvec.prod() if hasattr(env.action_space, 'nvec') else env.action_space.n
    
    agent = DQNAgent(
        observation_size=observation_size,
        action_size=action_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size
    )
    
    # Training loop
    print(f"[Main] Starting training for {args.episodes} episodes...")
    best_moi = 0.0
    best_fixtures = 0
    
    for episode in range(1, args.episodes + 1):
        # Train
        train_reward, train_fixtures = train_episode(agent, env, render=args.render and episode % 10 == 0)
        
        # Evaluate periodically
        if episode % args.eval_freq == 0:
            eval_reward, eval_fixtures, eval_moi = evaluate_episode(agent, env, render=args.render)
            
            # Track best performance
            if eval_moi > best_moi:
                best_moi = eval_moi
                best_fixtures = eval_fixtures
                agent.save(str(output_dir / f"best_model_moi_{eval_moi:.0f}.pt"))
            
            print(f"[Episode {episode:3d}] "
                  f"Train: reward={train_reward:7.2f}, fixtures={train_fixtures} | "
                  f"Eval: reward={eval_reward:7.2f}, fixtures={eval_fixtures}, MOI={eval_moi:.2e} | "
                  f"ε={agent.epsilon:.4f}")
        else:
            print(f"[Episode {episode:3d}] Train: reward={train_reward:7.2f}, fixtures={train_fixtures} | ε={agent.epsilon:.4f}")
    
    # Final save
    agent.save(str(output_dir / "final_model.pt"))
    
    print(f"\n[Main] Training complete!")
    print(f"[Main] Best performance: {best_fixtures} fixtures, MOI={best_moi:.2e}")
    print(f"[Main] Models saved to {output_dir}")
    
    # Visualize best solution
    print(f"\n[Main] Loading best model and visualizing final solution...")
    best_model_files = list(output_dir.glob("best_model_moi_*.pt"))
    if best_model_files:
        best_model_file = sorted(best_model_files, key=lambda x: float(x.stem.split('_')[-1]), reverse=True)[0]
        agent.load(str(best_model_file))
        
        # Run final evaluation with rendering
        env_viz = FixtureLayoutEnv(
            workpiece_name=args.workpiece,
            render_mode="human",
            verbose=False,
            use_hybrid_action_space=False,
            use_continuous_action_space=False,
            grid_resolution=1.0,
            max_steps=20
        )
        
        obs, info = env_viz.reset()
        done = False
        step = 0
        
        print(f"[Visualization] Running best policy (MOI={best_moi:.2e})...")
        while not done:
            action = agent.select_action(obs, env_viz.action_space, training=False)
            if isinstance(action, int):
                fixture_type = action % 3
                pair_idx = action // 3
                action = np.array([fixture_type, pair_idx])
            
            obs, reward, done, truncated, info = env_viz.step(action)
            done = done or truncated
            step += 1
            env_viz.render()
        
        print(f"[Visualization] Final solution: {len(env_viz.fixtures)} fixtures, MOI={env_viz.cumulative_moment:.2e}")
        plt.show()
        env_viz.close()
    
    env.close()


if __name__ == "__main__":
    main()
