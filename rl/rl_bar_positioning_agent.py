"""
Bar Positioning Agent - Stage 1 of Hierarchical Fixture Layout Optimization

Trains an agent to position rectangular "bars" (containers) on a workpiece.
Each bar is 145mm wide (same as Type 1 fixture) and spans the full workpiece height.

Objective: Maximize distance between bar centers + total bar area
Constraints: Bars must respect 345mm horizontal spacing (from MiniZinc constraints)

This is the first stage before fixture placement. The bar positions become
input to the fixture placement agent.
"""

import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from shapely.geometry import Point as ShapelyPoint, Polygon
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent

# Workpiece constraints (from MiniZinc model)
WORKPIECE_CONSTRAINTS = {
    'simple_stair_step': {
        'min_fixtures': 1,
        'max_fixtures': 10,
        'min_bars': 1,
        'max_bars': 6,
    },
    'coffee_table': {
        'min_fixtures': 1,
        'max_fixtures': 20,
        'min_bars': 2,
        'max_bars': 8,
    },
    'dashboard': {
        'min_fixtures': 2,
        'max_fixtures': 15,
        'min_bars': 2,
        'max_bars': 7,
    },
    'speaker': {
        'min_fixtures': 1,
        'max_fixtures': 12,
        'min_bars': 2,
        'max_bars': 6,
    },
    'spiral_stair_step': {
        'min_fixtures': 2,
        'max_fixtures': 18,
        'min_bars': 2,
        'max_bars': 8,
    },
}

# Bar dimensions
BAR_WIDTH = 145.0  # mm (same as Type 1 fixture width)
HORIZONTAL_SPACING = 345.0  # mm (minimum center-to-center spacing from MiniZinc)


@dataclass
class BarPosition:
    """Represents a positioned bar"""
    center_x: float  # Center x-coordinate
    
    def get_bounds(self) -> Tuple[float, float]:
        """Get left and right bounds of the bar"""
        left = self.center_x - BAR_WIDTH / 2
        right = self.center_x + BAR_WIDTH / 2
        return left, right


class BarPositioningEnv(gym.Env):
    """
    Gymnasium environment for positioning bars on a workpiece.
    
    Objective: Position N bars to maximize:
    - Distance between bar centers (spread bars out)
    - Total area covered (num_bars * bar_area)
    
    Action: Continuous x-coordinate [0, 1] normalized to valid workpiece width
    Observation: Current bar positions, available width, bar count
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(
        self,
        workpiece_name: str = "simple_stair_step",
        render_mode: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize bar positioning environment.
        
        Args:
            workpiece_name: Name of the workpiece
            render_mode: "human" for visualization
            verbose: Print debug information
        """
        super().__init__()
        
        self.workpiece_name = workpiece_name
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Get constraints
        if workpiece_name not in WORKPIECE_CONSTRAINTS:
            raise ValueError(f"Workpiece '{workpiece_name}' not found")
        
        self.constraints = WORKPIECE_CONSTRAINTS[workpiece_name]
        self.min_bars = self.constraints['min_bars']
        self.max_bars = self.constraints['max_bars']
        
        # Load workpiece data
        workpiece_file = ROOT / "resources" / "workpieces_information.json"
        with open(workpiece_file, 'r') as f:
            all_workpieces = json.load(f)
        
        if workpiece_name not in all_workpieces:
            raise ValueError(f"Workpiece '{workpiece_name}' not found in workpieces_information.json")
        
        self.workpiece_data = all_workpieces[workpiece_name]
        
        # Get workpiece dimensions
        vertices = self.workpiece_data['vertices']
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        self.workpiece_min_x = min(xs)
        self.workpiece_max_x = max(xs)
        self.workpiece_min_y = min(ys)
        self.workpiece_max_y = max(ys)
        
        self.workpiece_width = self.workpiece_max_x - self.workpiece_min_x
        self.workpiece_height = self.workpiece_max_y - self.workpiece_min_y
        
        # Create workpiece polygon
        self.workpiece_polygon = Polygon(vertices)
        
        if self.verbose:
            print(f"[Bar Positioning] Workpiece: {workpiece_name}")
            print(f"[Bar Positioning] Size: {self.workpiece_width:.1f}mm x {self.workpiece_height:.1f}mm")
            print(f"[Bar Positioning] Bar width: {BAR_WIDTH}mm")
            print(f"[Bar Positioning] Max bars: {self.max_bars}")
        
        # Initialize state FIRST (before generating valid positions)
        self.bars: List[BarPosition] = []
        self.step_count = 0
        self.max_steps = 50
        
        # Generate valid placement positions on a fine grid
        self.valid_bar_positions = self._generate_valid_bar_positions()
        
        if len(self.valid_bar_positions) == 0:
            raise ValueError(f"No valid bar positions found for {workpiece_name}")
        
        if self.verbose:
            print(f"[Bar Positioning] Valid bar positions: {len(self.valid_bar_positions)}")
        
        # Action space: Discrete - choose which valid position to place bar
        # Action 0 = no placement, 1..N = valid position indices
        self.action_space = spaces.Discrete(len(self.valid_bar_positions) + 1)
        
        # Observation space:
        # - Normalized count of bars placed (1 value)
        # - Bar positions: max_bars * 1 value each, normalized to [0, 1]
        # - Available width (1 value, normalized)
        obs_size = 1 + self.max_bars + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.bars = []
        self.step_count = 0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _generate_valid_bar_positions(self) -> List[float]:
        """
        Generate all valid x-positions for bar centers.
        A position is valid if:
        - The bar stays COMPLETELY within workpiece bounds (no overhang)
        - At least ONE rectangular fixture (180×65) can be placed on the bar
          without overlapping holes or going out of bounds
        
        This ensures bars are positioned only where fixtures can actually be placed.
        
        Returns: List of valid center x-coordinates
        """
        valid_positions = []
        
        # Grid resolution for bar position search (10mm)
        grid_step = 10.0
        
        # Constrain bar centers so bars stay completely within workpiece
        # Left edge must be >= workpiece_min_x
        # Right edge must be <= workpiece_max_x
        min_center_x = self.workpiece_min_x + BAR_WIDTH / 2
        max_center_x = self.workpiece_max_x - BAR_WIDTH / 2
        
        if self.verbose:
            print(f"[Bar Positioning] Valid x range for bar centers: {min_center_x:.1f} to {max_center_x:.1f}")
            print(f"[Bar Positioning] Scanning for positions with valid fixture placements...")
        
        # Scan across valid width
        for x in np.arange(min_center_x, max_center_x + grid_step, grid_step):
            # Verify bounds (should always pass given our loop, but double-check)
            left = x - BAR_WIDTH / 2
            right = x + BAR_WIDTH / 2
            
            if left >= self.workpiece_min_x and right <= self.workpiece_max_x:
                # Check if at least one fixture can be placed on this bar
                if self._has_valid_fixture_positions(x):
                    valid_positions.append(float(x))
        
        if self.verbose:
            print(f"[Bar Positioning] Found {len(valid_positions)} valid positions")
            if len(valid_positions) <= 10:
                print(f"  Positions: {[f'{x:.1f}' for x in valid_positions]}")
        
        return valid_positions
    
    def _has_valid_fixture_positions(self, bar_center_x: float) -> bool:
        """
        Check if a bar at given x-position can have at least one valid fixture placement.
        
        Strategy:
        1. First try rectangular fixtures (180×65) - these can cover more area
        2. If rectangular doesn't fit, try square fixtures (145×145) - narrower so more likely to fit
        
        A fixture is valid if:
        - It's completely within workpiece polygon (not outside the shape)
        - It doesn't overlap with any holes
        
        Returns: True if at least one valid placement exists (rect or square), False otherwise
        """
        # Try rectangular fixture first (Type 2: 180×65)
        if self._has_valid_fixture_positions_for_type(bar_center_x, fixture_width=180.0, fixture_height=65.0):
            return True
        
        # If rectangular doesn't work, try square fixture (Type 1: 145×145)
        if self._has_valid_fixture_positions_for_type(bar_center_x, fixture_width=145.0, fixture_height=145.0):
            return True
        
        # Neither fixture type can be placed
        return False
    
    def _has_valid_fixture_positions_for_type(self, bar_center_x: float, fixture_width: float, fixture_height: float) -> bool:
        """
        Check if a specific fixture type can be placed on bar at given x-position.
        
        Args:
            bar_center_x: Bar center x-coordinate
            fixture_width: Fixture width (180 for rect, 145 for square)
            fixture_height: Fixture height (65 for rect, 145 for square)
        
        Returns: True if at least one valid placement exists, False otherwise
        """
        # Fixture is centered on the bar
        fixture_left = bar_center_x - fixture_width / 2
        fixture_right = bar_center_x + fixture_width / 2
        
        # Check horizontal bounds first (quick rejection)
        if fixture_left < self.workpiece_min_x or fixture_right > self.workpiece_max_x:
            return False
        
        # Scan valid y-positions (10mm grid)
        y_grid_step = 10.0
        min_y_center = self.workpiece_min_y + fixture_height / 2
        max_y_center = self.workpiece_max_y - fixture_height / 2
        
        for y_center in np.arange(min_y_center, max_y_center + y_grid_step, y_grid_step):
            fixture_top = y_center - fixture_height / 2
            fixture_bottom = y_center + fixture_height / 2
            
            # Check bounds
            if fixture_top < self.workpiece_min_y or fixture_bottom > self.workpiece_max_y:
                continue
            
            # Create fixture polygon
            fixture_polygon = Polygon([
                (fixture_left, fixture_top),
                (fixture_right, fixture_top),
                (fixture_right, fixture_bottom),
                (fixture_left, fixture_bottom),
            ])
            
            # Check if fixture is completely within workpiece polygon
            # NOT just the bounding box, but the actual workpiece shape
            if not fixture_polygon.within(self.workpiece_polygon):
                # Fixture extends outside workpiece shape - skip this y-position
                continue
            
            # Check if fixture overlaps any holes
            has_hole_overlap = False
            if 'holes' in self.workpiece_data:
                for hole in self.workpiece_data['holes']:
                    if isinstance(hole, (list, tuple)) and len(hole) == 3:
                        cx, cy, radius = hole
                        hole_circle = ShapelyPoint(cx, cy).buffer(radius)
                        if fixture_polygon.intersects(hole_circle):
                            has_hole_overlap = True
                            break
            
            # If this position is within workpiece and hole-free, placement is valid
            if not has_hole_overlap:
                return True
        
        # No valid placement found for this fixture type
        return False
    
    def _compute_bar_coverage(self, center_x: float) -> float:
        """
        Compute useful coverage score for a bar at given x-position.
        Useful area = area inside workpiece AND not under holes.
        
        Returns: Fraction in [0, 1] representing useful coverage percentage
        """
        left = center_x - BAR_WIDTH / 2
        right = center_x + BAR_WIDTH / 2
        
        # Create bar polygon spanning full workpiece height
        bar_polygon = Polygon([
            (left, self.workpiece_min_y),
            (right, self.workpiece_min_y),
            (right, self.workpiece_max_y),
            (left, self.workpiece_max_y),
        ])
        
        # Intersect with workpiece to get area inside workpiece
        workpiece_intersection = bar_polygon.intersection(self.workpiece_polygon)
        area_in_workpiece = workpiece_intersection.area
        
        if area_in_workpiece == 0:
            return 0.0
        
        # Subtract area covered by holes
        useful_area = area_in_workpiece
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if isinstance(hole, (list, tuple)) and len(hole) == 3:
                    cx, cy, radius = hole
                    hole_circle = ShapelyPoint(cx, cy).buffer(radius)  # Create circle
                    hole_intersection = workpiece_intersection.intersection(hole_circle)
                    useful_area -= hole_intersection.area
        
        # Compute coverage as fraction of maximum possible bar area
        max_bar_area = BAR_WIDTH * self.workpiece_height
        coverage = useful_area / max_bar_area if max_bar_area > 0 else 0.0
        
        return max(0.0, coverage)  # Clamp to [0, 1]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Discrete action (0 = no placement, 1..N = valid position index)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Parse discrete action
        action_idx = int(action)
        
        reward = 0.0
        terminated = False
        info = {}
        
        # Action 0 = no placement (skip)
        if action_idx == 0:
            reward = -0.1  # Small penalty for skipping
            info['reason'] = 'no_placement'
        else:
            # Convert action index to bar position
            pos_idx = action_idx - 1  # Adjust for 0-indexing
            
            if pos_idx >= len(self.valid_bar_positions):
                reward = -0.1
                info['reason'] = 'invalid_action'
            elif len(self.bars) >= self.max_bars:
                reward = -0.1
                info['reason'] = 'max_bars_reached'
                terminated = True
            else:
                center_x = self.valid_bar_positions[pos_idx]
                
                # Check if this position violates spacing constraint with existing bars
                is_valid = True
                for bar in self.bars:
                    distance = abs(center_x - bar.center_x)
                    if distance < HORIZONTAL_SPACING:
                        is_valid = False
                        break
                
                if is_valid:
                    # Place bar
                    bar = BarPosition(center_x=center_x)
                    self.bars.append(bar)
                    
                    # Compute reward
                    reward = self._compute_reward_for_bar_placement()
                    info['reason'] = 'bar_placed'
                    info['bar_center_x'] = center_x
                    info['num_bars'] = len(self.bars)
                    
                    if self.verbose:
                        print(f"[Step {self.step_count}] Bar placed at x={center_x:.1f}")
                else:
                    # Invalid placement due to spacing constraint
                    reward = -0.1
                    info['reason'] = 'violates_spacing'
        
        # Check termination conditions
        # Episode terminates when:
        # 1. Max bars reached, OR
        # 2. Max steps reached
        # Do NOT terminate early based on action - let agent keep trying!
        if len(self.bars) >= self.max_bars:
            terminated = True
            # Bonus for reaching max bars
            reward += 500.0
            info['reason'] = 'max_bars_reached'
        
        if self.step_count >= self.max_steps:
            terminated = True
            # Bonus at episode end based on bars placed + average coverage
            bar_completion_bonus = len(self.bars) * 50.0
            
            # Add bonus based on average coverage quality of all bars
            if len(self.bars) > 0:
                total_coverage = sum(self._compute_bar_coverage(bar.center_x) for bar in self.bars)
                avg_coverage = total_coverage / len(self.bars)
                # Extra bonus for good average coverage
                coverage_bonus = avg_coverage * 200.0
                reward += coverage_bonus
            
            reward += bar_completion_bonus
        
        obs = self._get_observation()
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward_for_bar_placement(self) -> float:
        """
        Compute reward for placing a new bar.
        
        Objective: Maximize bar placement while prioritizing positions with high useful coverage
        Strategy:
        1. Base reward for each placement
        2. Coverage bonus: reward bars with high useful area (inside workpiece, not under holes)
        3. Spread bonus: encourage bars spread across workpiece width
        """
        if len(self.bars) == 0:
            return 0.0
        
        last_bar_center_x = self.bars[-1].center_x
        
        # PRIMARY: Compute coverage score for this bar
        coverage = self._compute_bar_coverage(last_bar_center_x)
        
        # Reward based on useful coverage (higher coverage = higher reward)
        # Max 300 points for bar with >90% useful area
        coverage_bonus = coverage * 300.0
        
        # Base reward for placement
        placement_bonus = 100.0
        
        # SECONDARY: Spread bonus - encourage bars distributed across workpiece
        spread_bonus = 0.0
        if len(self.bars) > 1:
            # Compute average distance to other bars
            distances = []
            for other_bar in self.bars[:-1]:  # All except the one we just placed
                distances.append(abs(last_bar_center_x - other_bar.center_x))
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                # Reward if bars are well-spread (far apart)
                # Max spread bonus when bars are 345mm+ apart (HORIZONTAL_SPACING)
                spread_bonus = min(100.0, avg_distance / HORIZONTAL_SPACING * 100.0)
        
        # TERTIARY: Edge positioning bonus (slight encouragement to use full width)
        edge_bonus = 0.0
        distance_to_min = abs(last_bar_center_x - (self.workpiece_min_x + BAR_WIDTH / 2))
        distance_to_max = abs(last_bar_center_x - (self.workpiece_max_x - BAR_WIDTH / 2))
        min_distance_to_edge = min(distance_to_min, distance_to_max)
        
        if min_distance_to_edge < 150.0:
            # Small bonus for positions near edges
            edge_bonus = 50.0 * (1.0 - min_distance_to_edge / 150.0)
        
        # Total reward: coverage-focused with spread and edge bonuses
        reward = placement_bonus + coverage_bonus + spread_bonus + edge_bonus
        
        if self.verbose:
            print(f"   Bar at x={last_bar_center_x:.1f}: coverage={coverage:.1%}, reward={reward:.1f}")
            print(f"     Breakdown: placement={placement_bonus:.0f}, coverage={coverage_bonus:.0f}, spread={spread_bonus:.0f}, edge={edge_bonus:.0f}")
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get observation vector"""
        obs = np.zeros(1 + self.max_bars + 1, dtype=np.float32)
        
        # Number of bars placed (normalized)
        obs[0] = len(self.bars) / self.max_bars
        
        # Bar positions (normalized to [0, 1])
        sorted_bars = sorted(self.bars, key=lambda b: b.center_x)
        for i, bar in enumerate(sorted_bars):
            bar_x_norm = (bar.center_x - self.workpiece_min_x) / self.workpiece_width
            obs[1 + i] = bar_x_norm
        
        # Available width (fraction of workpiece width not yet covered by bars)
        # Simplified: just normalize workpiece width
        obs[1 + self.max_bars] = self.workpiece_width / 1000.0
        
        return obs
    
    def get_bar_positions(self) -> List[float]:
        """Get current bar center x-coordinates (in workpiece coordinates)"""
        return sorted([bar.center_x for bar in self.bars])
    
    def render(self):
        """Render the environment (placeholder for visualization)"""
        if self.render_mode == "human":
            print(f"[Render] Bars at x: {[f'{b.center_x:.1f}' for b in self.bars]}")
    
    def close(self):
        """Close the environment"""
        pass


if __name__ == "__main__":
    # Quick test
    env = BarPositioningEnv(workpiece_name="simple_stair_step", verbose=True)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action[0]:.3f}, reward={reward:.2f}, bars={len(env.bars)}")
        
        if terminated:
            print("Episode terminated")
            break
    
    env.close()
    print(f"Final bar positions: {env.get_bar_positions()}")
