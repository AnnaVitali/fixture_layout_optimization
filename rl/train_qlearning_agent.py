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

from rl_agent import FixtureLayoutEnv, fixtures_to_solution_json
from machine_parameters import FixtureState

# Output directory structure for models and solutions
OUTPUT_RESULTS_DIR = "results"
OUTPUT_MODELS_SUBDIR = "models"
OUTPUT_JSON_SUBDIR = "json"


# Minimum and maximum fixtures per workpiece
MINIMUM_FIXTURES = {
    "coffee_table": 6,
    "dashboard": 3,
    "simple_stair_step": 3,
    "spiral_stair_step": 3,  # More constrained (23.7% safe space vs 53.9%)
    "speaker": 3,
    "door": 10,
    "door_porthole": 10
}

MAXIMUM_FIXTURES = {
    "coffee_table": 10,
    "dashboard": 6,
    "simple_stair_step": 6,
    "spiral_stair_step": 5,  # Limited by space
    "speaker": 6,
    "door": 20,
    "door_porthole": 20
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
        
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        self._last_valid_actions_cache = None
        self._last_env_id = None
        
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
        
        if training and np.random.random() < self.epsilon:
            return valid_actions[np.random.randint(0, len(valid_actions))]
        else:
            best_value = -np.inf
            best_actions = []
            
            for action in valid_actions:
                q_value = self.q_table[(state, action)]
                if q_value > best_value:
                    best_value = q_value
                    best_actions = [action]
                elif q_value == best_value:
                    best_actions.append(action)
            
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
        current_q = self.q_table[(state, action)]
        
        if done or not next_valid_actions:
            max_next_q = 0.0
        else:
            max_next_q = max(
                self.q_table[(next_state, a)] for a in next_valid_actions
            )
        
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
    env_key = (len(env.fixtures), env.fixture_availability[1], env.fixture_availability[2])
    
    if agent._last_env_id == env_key and agent._last_valid_actions_cache is not None:
        return agent._last_valid_actions_cache
    
    valid_pairs, _, _ = env._compute_valid_action_positions()
    valid_objectives = env._compute_valid_action_objectives()
    
    valid_actions = []
    
    for pos_idx, (x_idx, y_idx) in enumerate(valid_pairs):
        for fixture_type in [1, 2]:
            if (x_idx, y_idx, fixture_type) in valid_objectives:
                if env.fixture_availability[fixture_type] > 0:
                    valid_actions.append((pos_idx, fixture_type))
    
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
    
    width, height = FIXTURE_DIMENSIONS[fixture_type]
    x_pos = env.x_positions[x_idx]
    y_pos = env.y_positions[y_idx]
    top_left_x = x_pos - width / 2
    top_left_y = y_pos - height / 2
    
    fixture_state = FixtureState(x=top_left_x, y=top_left_y, angle=0.0, type_id=fixture_type)
    
    is_valid, _ = env._is_valid_placement(x_pos, y_pos, fixture_type, 0.0)
    
    if not is_valid:
        return -500e6, False
    
    old_fixtures = env.fixtures
    env.fixtures.append(fixture_state)
    env.fixture_availability[fixture_type] -= 1
    
    old_moment = env.cumulative_moment
    new_moment = env._compute_system_moment_of_inertia(env.fixtures)
    env.cumulative_moment = new_moment
    
    moi_increase = new_moment - old_moment
    num_fixtures_after = len(env.fixtures)
    
    moi_contribution = max(0, moi_increase) * 0.01
    distribution_bonus = 1.0e9 * (num_fixtures_after ** 2)
    
    distance_bonus = 0.0
    if num_fixtures_before > 0:
        total_distance = 0.0
        for i in range(num_fixtures_before):
            prev_fixture = old_fixtures[i]
            prev_width, prev_height = FIXTURE_DIMENSIONS[prev_fixture.type_id]
            prev_center_x = prev_fixture.x + prev_width / 2
            prev_center_y = prev_fixture.y + prev_height / 2
            
            new_center_x = x_pos
            new_center_y = y_pos
           
            distance = ((new_center_x - prev_center_x)**2 + (new_center_y - prev_center_y)**2) ** 0.5
            total_distance += distance
        
        avg_distance = total_distance / num_fixtures_before
        distance_bonus = 1.0e6 * avg_distance
    
    min_fixtures_bonus = 5.0e9 if num_fixtures_after >= env.min_fixtures else 0.0
    
    reward = moi_contribution + distribution_bonus + distance_bonus + min_fixtures_bonus
    
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
        valid_actions = get_valid_actions(env, agent)
        
        num_current_fixtures = len(env.fixtures)
        
        if not valid_actions:
            break
        
        action = agent.select_action(state, valid_actions, training=True)
        assert action is not None, "Agent must select an action when valid_actions is non-empty"
        
        pos_idx, fixture_type = action
        valid_pairs, _, _ = env._compute_valid_action_positions()
        x_idx, y_idx = valid_pairs[pos_idx]
        
        num_fixtures_before = len(env.fixtures)
        reward, success = place_fixture_directly(env, x_idx, y_idx, fixture_type, num_fixtures_before)
        
        num_current_fixtures = len(env.fixtures)
        done = ((env.step_count >= env.max_steps and num_current_fixtures >= min_fixtures) or not success)
        
        next_state = agent.get_state(env)
        
        next_valid_actions = get_valid_actions(env, agent)
        
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
    
    min_fixtures = MINIMUM_FIXTURES.get(workpiece_name, 3)
    if num_fixtures >= min_fixtures:
        completion_bonus = 10e9 * (num_fixtures / 3.0)  # Much stronger incentive
        cumulative_reward += completion_bonus
    else:
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
        valid_actions = get_valid_actions(env, agent)
        
        num_current_fixtures = len(env.fixtures)
        
        if not valid_actions:
            break
        
        if num_current_fixtures < min_fixtures:
            pass
        
        if np.random.random() < 0.3:  # 30% exploration
            action = valid_actions[np.random.randint(len(valid_actions))]
        else:
            best_action = None
            best_q_value = -np.inf
            for a in valid_actions:
                q_val = agent.q_table.get((state, a), 0.0)
                if q_val > best_q_value:
                    best_q_value = q_val
                    best_action = a
            action = best_action if best_action is not None else valid_actions[0]
        
        assert action is not None, "Agent must select an action when valid_actions is non-empty"
        
        pos_idx, fixture_type = action
        valid_pairs, _, _ = env._compute_valid_action_positions()
        x_idx, y_idx = valid_pairs[pos_idx]
        
        num_fixtures_before = len(env.fixtures)
        reward, success = place_fixture_directly(env, x_idx, y_idx, fixture_type, num_fixtures_before)
        
        if not success:
            break
        
        cumulative_reward += reward
        state = agent.get_state(env)
        num_current_fixtures = len(env.fixtures)
        
        done = (env.step_count >= env.max_steps and num_current_fixtures >= min_fixtures)
        
        if render:
            env.render()
    
    num_fixtures = len(env.fixtures)
    final_moi = env.cumulative_moment
    
    fixtures_config = [
        {
            'type': f.type_id,
            'x': f.x,
            'y': f.y,
            'angle': f.angle
        }
        for f in env.fixtures
    ]
    
    min_fixtures = MINIMUM_FIXTURES.get(workpiece_name, 3)
    if num_fixtures >= min_fixtures:
        completion_bonus = 10e9 * (num_fixtures / 3.0)  # Much stronger incentive
        cumulative_reward += completion_bonus
    else:
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
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="Q-learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Discount factor")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Epsilon decay rate")
    
    args = parser.parse_args()
    
    print(f"[Main] Creating environment for '{args.workpiece}'...")
    env = FixtureLayoutEnv(
        workpiece_name=args.workpiece,
        render_mode="human" if args.render else None,
        verbose=False,
        use_hybrid_action_space=False,
        use_continuous_action_space=False,
        grid_resolution=10.0,
        max_steps=30  
    )
    
    print(f"[Main] Creating Q-Learning agent...")
    agent = QLearningAgent(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
    )
    
    output_dir = Path(__file__).parent / OUTPUT_RESULTS_DIR
    models_dir = output_dir / OUTPUT_MODELS_SUBDIR
    json_dir = output_dir / OUTPUT_JSON_SUBDIR
    
    output_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    
    print(f"\n[Main] Starting training for {args.episodes} episodes...")
    print(f"[Main] Epsilon decay: {args.epsilon_decay}")
    print(f"[Main] Evaluation frequency: {args.eval_freq} episodes\n")
    
    best_moi = float('-inf')
    best_fixtures = 0
    best_episode = 0
    best_model_file = None  
    
    episode_rewards = []
    episode_fixtures = []
    episode_mois = []
    
    for episode in range(1, args.episodes + 1):
        train_reward, train_fixtures, train_moi = train_episode(agent, env, args.workpiece)
        
        episode_rewards.append(train_reward)
        episode_fixtures.append(train_fixtures)
        episode_mois.append(train_moi)
        
        if train_moi > best_moi:
            best_moi = train_moi
            best_fixtures = train_fixtures
            best_episode = episode
            
            best_model_file = models_dir / "best_model.pkl"
            agent.save(str(best_model_file))
            
            best_solution = fixtures_to_solution_json(env.fixtures, train_moi, env.visualizer)
            best_solution_file = json_dir / f"rl_{args.workpiece}.json"
            with open(best_solution_file, 'w') as f:
                json.dump(best_solution, f, indent=2)
            
            min_fixtures_required = MINIMUM_FIXTURES.get(args.workpiece, 3)
            status = "[OK] VALID" if train_fixtures >= min_fixtures_required else "[!] INCOMPLETE"
            print(f"[Agent] {status} Best solution saved (TRAIN): {train_fixtures} fixtures, MOI={train_moi:.2e}")
            print(f"[Agent] Files: {best_model_file} | {best_solution_file}")
        
        if episode % args.eval_freq == 0:
            eval_reward, eval_fixtures, eval_moi, eval_fixture_config = evaluate_episode(agent, env, args.workpiece, seed=42)
            
            print(f"[Ep {episode:4d}] Train: reward={train_reward:7.2f}, "
                  f"fixtures={train_fixtures:2d}, MOI={train_moi:10.2e} | "
                  f"Eval: reward={eval_reward:7.2f}, "
                  f"fixtures={eval_fixtures:2d}, MOI={eval_moi:10.2e} | "
                  f"eps={agent.epsilon:.4f}")
            
            if eval_moi > best_moi:
                best_moi = eval_moi
                best_fixtures = eval_fixtures
                best_episode = episode
                
                best_model_file = models_dir / "best_model.pkl"
                agent.save(str(best_model_file))
                
                best_solution = fixtures_to_solution_json(env.fixtures, eval_moi, env.visualizer)
                best_solution_file = json_dir / f"rl_{args.workpiece}.json"
                with open(best_solution_file, 'w') as f:
                    json.dump(best_solution, f, indent=2)
                
                min_fixtures_required = MINIMUM_FIXTURES.get(args.workpiece, 3)
                status = "[OK] VALID" if eval_fixtures >= min_fixtures_required else "[!] INCOMPLETE"
                print(f"[Agent] {status} Best solution saved (EVAL): {eval_fixtures} fixtures, MOI={eval_moi:.2e}")
                print(f"[Agent] Files: {best_model_file} | {best_solution_file}")
        else:
            if episode % 10 == 0:
                print(f"[Ep {episode:4d}] Train: reward={train_reward:7.2f}, "
                      f"fixtures={train_fixtures:2d}, MOI={train_moi:10.2e} | "
                      f"eps={agent.epsilon:.4f}")
    
    agent.save(str(models_dir / "final_model.pkl"))
    
    print(f"\n[Main] Training complete!")
    print(f"[Main] Best performance: Episode {best_episode}, {best_fixtures} fixtures, MOI={best_moi:.2e}")
    print(f"[Main] Models saved to {models_dir}")
    print(f"[Main] Solutions saved to {json_dir}")
    print(f"[Main] Q-table size: {len(agent.q_table)} state-action pairs")
    
    print(f"\n[Main] Visualizing best solution...")
    best_solution_path = json_dir / f"rl_{args.workpiece}.json"
    
    if best_solution_path.exists():
        with open(best_solution_path, 'r') as f:
            best_solution = json.load(f)
        
        print(f"[Visualization] Loading best solution (Objective={best_solution['objective_value']:.2e})...")
        
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
        
        num_fixtures = len(best_solution['x'])
        print(f"[Visualization] Placing {num_fixtures} fixtures...")
        
        workpiece_height = env_viz.visualizer.max_y - env_viz.visualizer.min_y
        
        for i in range(num_fixtures):
            fixture_type = best_solution['fixture_type'][i]
            center_x_orig = best_solution['fixtures_center_x'][i]
            center_y_orig = best_solution['fixtures_center_y'][i]
            angle = best_solution['angle'][i]
            
            center_x_math = center_x_orig - env_viz.visualizer.min_x
            center_y_math = workpiece_height - (center_y_orig - env_viz.visualizer.min_y)
            
            from machine_parameters import FIXTURE_DIMENSIONS
            width, height = FIXTURE_DIMENSIONS[fixture_type]
            top_left_x = center_x_math - width / 2
            top_left_y = center_y_math - height / 2
            
            fixture_state = FixtureState(
                x=top_left_x,
                y=top_left_y,
                angle=angle,
                type_id=fixture_type
            )
            
            env_viz.fixtures.append(fixture_state)
            env_viz.fixture_availability[fixture_type] -= 1
            
            print(f"  [{i+1}] Type {fixture_type} at center ({center_x_orig:.1f}, {center_y_orig:.1f})")
        
        computed_moi = env_viz._compute_system_moment_of_inertia(env_viz.fixtures)
        env_viz.cumulative_moment = computed_moi
        
        print(f"[Visualization] Final solution: {num_fixtures} fixtures, MOI={computed_moi:.2e}")
        
        env_viz.fixture_availability[1] = 0
        env_viz.fixture_availability[2] = 0
        env_viz.max_fixtures = len(env_viz.fixtures)
        
        print("[Visualization] Displaying best solution (close the window to continue)...")
        env_viz.render()
        
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except KeyboardInterrupt:
            print("[Visualization] Interrupted by user...")
        finally:
            env_viz.close()
    else:
        print(f"[Main] No best solution found at {best_solution_path}")
    
    env.close()


if __name__ == "__main__":
    main()
