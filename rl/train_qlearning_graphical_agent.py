"""
Q-Learning Agent for Fixture Layout Optimization.

Uses classic tabular Q-learning with discrete action space and discrete state space.
State: (num_fixtures_placed, fixture_type_1_remaining, fixture_type_2_remaining)
Action: (valid_position_index, fixture_type_to_place)

This approach is much simpler than DQN and leverages the discrete action space.
Since all valid actions and rewards can be computed explicitly, we build a Q-table.

Optimizations:
- Spatial index caching: Store which positions are invalid due to fixture placements
- Valid actions caching: Cache valid actions by environment state to avoid recomputation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Set
import pickle
import time

import numpy as np
from collections import defaultdict

from rl_graphical_agent import FixtureLayoutEnv
from machine_parameters import FixtureState


# Minimum and maximum fixtures per workpiece
MINIMUM_FIXTURES = {
    "coffee_table": 6,
    "dashboard": 3,
    "simple_stair_step": 4,
    "spiral_stair_step": 3,  # More constrained (23.7% safe space vs 53.9%)
    "speaker": 3,
}

MAXIMUM_FIXTURES = {
    "coffee_table": 10,
    "dashboard": 6,
    "simple_stair_step": 6,
    "spiral_stair_step": 5,  # Limited by space
    "speaker": 6,
}


class QLearningAgent:
    """
    Classic tabular Q-learning agent for discrete state-action space.
    
    State: (num_fixtures, fixture_type_1_remaining, fixture_type_2_remaining)
    Action: (valid_position_index, fixture_type)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            learning_rate: Learning rate for Q-value updates (alpha)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration probability
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dict of (state, action) -> q_value
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Cache: store last computed valid actions to speed up training
        self._last_valid_actions_cache = None
        self._last_env_id = None
        
        # Statistics
        self.update_count = 0
        
    def get_state(self, env: FixtureLayoutEnv) -> Tuple[int, int, int]:
        """
        Get current state from environment.
        
        State: (num_fixtures_placed, type_1_remaining, type_2_remaining)
        
        Type 1: Square (145×145mm), max 24
        Type 2: Rectangular (180×65mm), max 12
        """
        num_fixtures = len(env.fixtures)
        type1_remaining = env.fixture_availability[1]
        type2_remaining = env.fixture_availability[2]
        
        return (num_fixtures, type1_remaining, type2_remaining)
    
    def select_action(
        self,
        state: Tuple[int, int, int],
        valid_actions: list,
        training: bool = True
    ) -> Optional[Tuple[int, int]]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
            valid_actions: List of (position_idx, fixture_type) tuples
                          where fixture_type in {1, 2}
            training: Whether we're in training mode (use exploration)
        
        Returns:
            Selected action or None if no valid actions
        """
        if not valid_actions:
            return None
        
        # Epsilon-greedy: explore with probability epsilon, exploit with probability 1-epsilon
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return valid_actions[np.random.randint(0, len(valid_actions))]
        else:
            # Exploit: best Q-value for this state
            best_value = -np.inf
            best_actions = []
            
            for action in valid_actions:
                q_value = self.q_table[(state, action)]
                if q_value > best_value:
                    best_value = q_value
                    best_actions = [action]
                elif q_value == best_value:
                    best_actions.append(action)
            
            # Return random action among best
            return best_actions[np.random.randint(0, len(best_actions))]
    
    def update_q_value(
        self,
        state: Tuple[int, int, int],
        action: Tuple[int, int],
        reward: float,
        next_state: Tuple[int, int, int],
        next_valid_actions: list,
        done: bool
    ) -> float:
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_valid_actions: List of valid actions in next state
            done: Whether episode is done
        
        Returns:
            TD error (for logging)
        """
        # Current Q-value
        current_q = self.q_table[(state, action)]
        
        # Maximum Q-value for next state
        if done or not next_valid_actions:
            max_next_q = 0.0
        else:
            max_next_q = max(
                self.q_table[(next_state, a)] for a in next_valid_actions
            )
        
        # Q-learning update
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state, action)] = new_q
        
        self.update_count += 1
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"[Agent] Q-table saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"[Agent] Q-table loaded from {filepath}")


def get_valid_actions(
    env: FixtureLayoutEnv,
    agent: QLearningAgent
) -> list:
    """
    Get list of valid actions (position, fixture_type) pairs.
    
    Uses caching to avoid expensive recomputation of valid positions.
    Cache is invalidated when environment changes (different fixture count or availability).
    
    Fast path: Reuse environment's precomputed safe positions and spacing logic.
    
    Returns:
        List of (valid_position_idx, fixture_type) tuples
        where fixture_type in {1, 2}
    """
    # Create cache key based on current environment state
    env_key = (len(env.fixtures), env.fixture_availability[1], env.fixture_availability[2])
    
    # Return cached result if available
    if agent._last_env_id == env_key and agent._last_valid_actions_cache is not None:
        return agent._last_valid_actions_cache
    
    # Compute valid actions using environment's methods
    valid_pairs, _, _ = env._compute_valid_action_positions()
    valid_objectives = env._compute_valid_action_objectives()
    
    valid_actions = []
    
    for pos_idx, (x_idx, y_idx) in enumerate(valid_pairs):
        for fixture_type in [1, 2]:
            if (x_idx, y_idx, fixture_type) in valid_objectives:
                if env.fixture_availability[fixture_type] > 0:
                    valid_actions.append((pos_idx, fixture_type))
    
    # Cache result
    agent._last_env_id = env_key
    agent._last_valid_actions_cache = valid_actions
    
    return valid_actions


def place_fixture_directly(
    env: FixtureLayoutEnv,
    x_idx: int,
    y_idx: int,
    fixture_type: int,
    num_fixtures_before: int = 0
) -> Tuple[float, bool]:
    """
    Place a fixture directly at grid position without going through env.step().
    
    Bypasses the environment's greedy logic for faster Q-learning.
    
    Reward structure:
    - Base: MOI increase from placement
    - Distribution bonus: +500M * num_fixtures (encourages placing many fixtures spread apart)
    - Penalty: -500M for failed placement
    
    Args:
        env: Environment
        x_idx, y_idx: Grid position indices
        fixture_type: Type of fixture (1 or 2)
        num_fixtures_before: Number of fixtures placed before this one
    
    Returns:
        (reward, success)
    """
    from machine_parameters import FIXTURE_DIMENSIONS, FixtureState
    
    if fixture_type not in [1, 2]:
        return -500e6, False
    
    if env.fixture_availability[fixture_type] <= 0:
        return -500e6, False
    
    # Get fixture dimensions and position
    width, height = FIXTURE_DIMENSIONS[fixture_type]
    x_pos = env.x_positions[x_idx]
    y_pos = env.y_positions[y_idx]
    top_left_x = x_pos - width / 2
    top_left_y = y_pos - height / 2
    
    # Create fixture state
    fixture_state = FixtureState(x=top_left_x, y=top_left_y, angle=0.0, type_id=fixture_type)
    
    # Check if valid
    is_valid, _ = env._is_valid_placement(x_pos, y_pos, fixture_type, 0.0)
    
    if not is_valid:
        return -500e6, False
    
    # Add fixture
    old_fixtures = env.fixtures
    env.fixtures.append(fixture_state)
    env.fixture_availability[fixture_type] -= 1
    
    # Compute new MOI
    old_moment = env.cumulative_moment
    new_moment = env._compute_system_moment_of_inertia(env.fixtures)
    env.cumulative_moment = new_moment
    
    # Enhanced reward structure:
    # - MOI increase (primary objective)
    # - Distribution bonus: reward more fixtures spread apart
    # - Area bonus: prioritize larger fixtures (like CP model does)
    moi_increase = new_moment - old_moment
    
    # Exponential bonus: each fixture placement gets bonus proportional to fixture count
    # 1st fixture: 500M, 2nd: 1000M, 3rd: 1500M, etc.
    num_fixtures_after = len(env.fixtures)
    distribution_bonus = 500e6 * num_fixtures_after  # Scales with number of fixtures
    
    # Area bonus: prioritize larger fixtures
    # Type 1 (Square 145×145mm = 21,025 mm²): higher bonus
    # Type 2 (Rect 180×65mm = 11,700 mm²): lower bonus
    fixture_area = width * height
    area_bonus = 100e3 * fixture_area  # Bonus proportional to fixture area
    
    reward = moi_increase + distribution_bonus + area_bonus
    
    env.step_count += 1
    
    return reward, True


def train_episode(
    agent: QLearningAgent,
    env: FixtureLayoutEnv,
    workpiece_name: str = "simple_stair_step"
) -> Tuple[float, int, float]:
    """
    Run one training episode.
    Enforces minimum fixtures constraint.
    
    Returns:
        (cumulative_reward, num_fixtures_placed, final_moi)
    """
    obs, info = env.reset()
    state = agent.get_state(env)
    cumulative_reward = 0.0
    done = False
    min_fixtures = MINIMUM_FIXTURES.get(workpiece_name, 3)
    
    while not done:
        # Get valid actions
        valid_actions = get_valid_actions(env, agent)
        
        num_current_fixtures = len(env.fixtures)
        
        if not valid_actions:
            # No valid actions, episode ends
            break
        
        # Select action
        action = agent.select_action(state, valid_actions, training=True)
        
        if action is None:
            break
        
        # Convert action to grid position and fixture type
        pos_idx, fixture_type = action
        valid_pairs, _, _ = env._compute_valid_action_positions()
        x_idx, y_idx = valid_pairs[pos_idx]
        
        # Place fixture directly (bypasses environment's greedy logic)
        num_fixtures_before = len(env.fixtures)
        reward, success = place_fixture_directly(env, x_idx, y_idx, fixture_type, num_fixtures_before)
        
        # Only allow done if: (max_steps reached AND minimum fixtures met) OR placement failed
        num_current_fixtures = len(env.fixtures)
        done = ((env.step_count >= env.max_steps and num_current_fixtures >= min_fixtures) or not success)
        
        # Get next state
        next_state = agent.get_state(env)
        
        # Get valid actions for next state
        next_valid_actions = get_valid_actions(env, agent)
        
        # Update Q-value
        agent.update_q_value(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_valid_actions=next_valid_actions,
            done=done
        )
        
        cumulative_reward += reward
        state = next_state
    
    agent.decay_epsilon()
    
    num_fixtures = len(env.fixtures)
    final_moi = env.cumulative_moment
    
    # CRITICAL: Penalize incomplete solutions, bonus valid solutions
    min_fixtures = MINIMUM_FIXTURES.get(workpiece_name, 3)
    if num_fixtures >= min_fixtures:
        # Valid solution: bonus increases with fixture count
        completion_bonus = 10e9 * (num_fixtures / 3.0)  # Much stronger incentive
        cumulative_reward += completion_bonus
    else:
        # Invalid solution: strong penalty to discourage incomplete episodes
        completion_penalty = -10e9
        cumulative_reward += completion_penalty
    
    return cumulative_reward, num_fixtures, final_moi


def evaluate_episode(
    agent: QLearningAgent,
    env: FixtureLayoutEnv,
    workpiece_name: str = "simple_stair_step",
    render: bool = False,
    seed: Optional[int] = None
) -> Tuple[float, int, float]:
    """
    Run one evaluation episode with small epsilon exploration.
    
    Uses epsilon=0.1 (10% random) to escape local optima while mostly greedy.
    Deterministic action selection with optional seed for reproducibility.
    Enforces minimum fixtures constraint.
    
    Returns:
        (cumulative_reward, num_fixtures_placed, final_moi)
    """
    obs, info = env.reset(seed=seed)
    state = agent.get_state(env)
    cumulative_reward = 0.0
    done = False
    min_fixtures = MINIMUM_FIXTURES.get(workpiece_name, 3)
    
    while not done:
        # Get valid actions
        valid_actions = get_valid_actions(env, agent)
        
        num_current_fixtures = len(env.fixtures)
        
        if not valid_actions:
            break
        
        # If we haven't met minimum fixtures yet, we must continue placing
        if num_current_fixtures < min_fixtures:
            # Force continued placement by not allowing done
            pass
        
        # Select action with epsilon=0.3 exploration (stronger to escape local optima)
        if np.random.random() < 0.3:  # 30% exploration
            action = valid_actions[np.random.randint(len(valid_actions))]
        else:
            # Greedy selection
            best_action = None
            best_q_value = -np.inf
            for a in valid_actions:
                q_val = agent.q_table.get((state, a), 0.0)
                if q_val > best_q_value:
                    best_q_value = q_val
                    best_action = a
            action = best_action if best_action is not None else valid_actions[0]
        
        if action is None:
            break
        
        # Convert action to grid position and fixture type
        pos_idx, fixture_type = action
        valid_pairs, _, _ = env._compute_valid_action_positions()
        x_idx, y_idx = valid_pairs[pos_idx]
        
        # Place fixture directly
        num_fixtures_before = len(env.fixtures)
        reward, success = place_fixture_directly(env, x_idx, y_idx, fixture_type, num_fixtures_before)
        
        if not success:
            break
        
        cumulative_reward += reward
        state = agent.get_state(env)
        num_current_fixtures = len(env.fixtures)
        
        # Only allow done if: (max_steps reached AND minimum fixtures met)
        done = (env.step_count >= env.max_steps and num_current_fixtures >= min_fixtures)
        
        if render:
            env.render()
    
    num_fixtures = len(env.fixtures)
    final_moi = env.cumulative_moment
    
    # Capture fixture configuration for saving
    fixtures_config = [
        {
            'type': f.type_id,
            'x': f.x,
            'y': f.y,
            'angle': f.angle
        }
        for f in env.fixtures
    ]
    
    # CRITICAL: Penalize incomplete solutions, bonus valid solutions
    min_fixtures = MINIMUM_FIXTURES.get(workpiece_name, 3)
    if num_fixtures >= min_fixtures:
        # Valid solution: bonus increases with fixture count
        completion_bonus = 10e9 * (num_fixtures / 3.0)  # Much stronger incentive
        cumulative_reward += completion_bonus
    else:
        # Invalid solution: strong penalty to discourage incomplete episodes
        completion_penalty = -10e9
        cumulative_reward += completion_penalty
    
    return cumulative_reward, num_fixtures, final_moi, fixtures_config


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Q-Learning Agent for Fixture Layout")
    parser.add_argument("--workpiece", type=str, default="simple_stair_step",
                        help="Workpiece name")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--eval-freq", type=int, default=50,
                        help="Evaluation frequency (episodes)")
    parser.add_argument("--render", action="store_true",
                        help="Render during training")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Q-learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay rate")
    
    args = parser.parse_args()
    
    # Create environment
    print(f"[Main] Creating environment for '{args.workpiece}'...")
    env = FixtureLayoutEnv(
        workpiece_name=args.workpiece,
        render_mode="human" if args.render else None,
        verbose=False,
        use_hybrid_action_space=False,
        use_continuous_action_space=False,
        grid_resolution=10.0,
        max_steps=30  # Increased to allow placement of minimum fixtures
    )
    
    # Create agent
    print(f"[Main] Creating Q-Learning agent...")
    agent = QLearningAgent(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
    )
    
    # Output directory
    output_dir = Path(__file__).parent / "qlearning_models"
    output_dir.mkdir(exist_ok=True)
    
    # Clean up old best model files from previous runs (keep only final_model.pkl for reference)
    for old_file in output_dir.glob("best_model_moi_*.pkl"):
        old_file.unlink()
    
    # Training loop
    print(f"\n[Main] Starting training for {args.episodes} episodes...")
    print(f"[Main] Epsilon decay: {args.epsilon_decay}")
    print(f"[Main] Evaluation frequency: {args.eval_freq} episodes\n")
    
    best_moi = float('-inf')
    best_fixtures = 0
    best_episode = 0
    best_model_file = None  # Track best model file for current run
    
    episode_rewards = []
    episode_fixtures = []
    episode_mois = []
    
    for episode in range(1, args.episodes + 1):
        # Training episode
        train_reward, train_fixtures, train_moi = train_episode(agent, env, args.workpiece)
        
        episode_rewards.append(train_reward)
        episode_fixtures.append(train_fixtures)
        episode_mois.append(train_moi)
        
        # Periodic evaluation
        if episode % args.eval_freq == 0:
            eval_reward, eval_fixtures, eval_moi, eval_fixture_config = evaluate_episode(agent, env, args.workpiece, seed=42)
            
            print(f"[Ep {episode:4d}] Train: reward={train_reward:7.2f}, "
                  f"fixtures={train_fixtures:2d}, MOI={train_moi:10.2e} | "
                  f"Eval: reward={eval_reward:7.2f}, "
                  f"fixtures={eval_fixtures:2d}, MOI={eval_moi:10.2e} | "
                  f"eps={agent.epsilon:.4f}")
            
            # Track best performance
            # Save whenever we find a valid solution with better MOI
            min_fixtures_required = MINIMUM_FIXTURES.get(args.workpiece, 3)
            
            if eval_fixtures >= min_fixtures_required and eval_moi > best_moi:
                best_moi = eval_moi  # Track best MOI
                best_fixtures = eval_fixtures
                best_episode = episode
                
                # Save best model (overwrite previous best)
                best_model_file = output_dir / "best_model.pkl"
                agent.save(str(best_model_file))
                
                # Also save the best fixture configuration as JSON for visualization
                best_solution = {
                    'workpiece': args.workpiece,
                    'num_fixtures': eval_fixtures,
                    'moi': eval_moi,
                    'fixtures': eval_fixture_config
                }
                best_solution_file = output_dir / f"best_solution_{args.workpiece}.json"
                with open(best_solution_file, 'w') as f:
                    json.dump(best_solution, f, indent=2)
                
                print(f"[Agent] Best solution saved to {best_solution_file}")
                print(f"[Agent] Q-table saved to {best_model_file}")
        else:
            if episode % 10 == 0:
                print(f"[Ep {episode:4d}] Train: reward={train_reward:7.2f}, "
                      f"fixtures={train_fixtures:2d}, MOI={train_moi:10.2e} | "
                      f"eps={agent.epsilon:.4f}")
    
    # Final save
    agent.save(str(output_dir / "final_model.pkl"))
    
    print(f"\n[Main] Training complete!")
    print(f"[Main] Best performance: Episode {best_episode}, {best_fixtures} fixtures, MOI={best_moi:.2e}")
    print(f"[Main] Models saved to {output_dir}")
    print(f"[Main] Q-table size: {len(agent.q_table)} state-action pairs")
    
    # Visualize best solution
    print(f"\n[Main] Visualizing best solution...")
    best_solution_path = output_dir / f"best_solution_{args.workpiece}.json"
    
    if best_solution_path.exists():
        # Load the best solution configuration directly
        with open(best_solution_path, 'r') as f:
            best_solution = json.load(f)
        
        print(f"[Visualization] Loading best solution (MOI={best_solution['moi']:.2e})...")
        
        # Create environment for visualization
        env_viz = FixtureLayoutEnv(
            workpiece_name=args.workpiece,
            render_mode="human",
            verbose=False,
            use_hybrid_action_space=False,
            use_continuous_action_space=False,
            grid_resolution=10.0,
            max_steps=100
        )
        
        obs, info = env_viz.reset(seed=42)
        
        # Reconstruct fixtures from saved solution
        print(f"[Visualization] Placing {best_solution['num_fixtures']} fixtures...")
        for i, fixture_data in enumerate(best_solution['fixtures']):
            fixture_type = fixture_data['type']
            x = fixture_data['x']
            y = fixture_data['y']
            angle = fixture_data['angle']
            
            # Create fixture state
            fixture_state = FixtureState(
                x=x,
                y=y,
                angle=angle,
                type_id=fixture_type
            )
            
            # Add to environment
            env_viz.fixtures.append(fixture_state)
            env_viz.fixture_availability[fixture_type] -= 1
            
            print(f"  [{i+1}] Type {fixture_type} at ({x:.1f}, {y:.1f})")
        
        # Compute MOI
        computed_moi = env_viz._compute_system_moment_of_inertia(env_viz.fixtures)
        env_viz.cumulative_moment = computed_moi
        
        print(f"[Visualization] Final solution: {best_solution['num_fixtures']} fixtures, MOI={computed_moi:.2e}")
        env_viz.render()
        
        # Keep rendering
        try:
            while True:
                env_viz.render()
        except KeyboardInterrupt:
            pass
        finally:
            env_viz.close()
    else:
        print(f"[Main] No best solution found to visualize")
        # Fallback: save a dummy solution based on training
        solution = {
            'workpiece': args.workpiece,
            'num_fixtures': best_fixtures,
            'moi': best_moi,
            'fixtures': []
        }
        solution_file = output_dir / f"solution_{args.workpiece}.json"
        with open(solution_file, 'w') as f:
            json.dump(solution, f, indent=2)
    
    env.close()


if __name__ == "__main__":
    main()
