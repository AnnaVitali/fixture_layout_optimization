"""
Fixture Placement on Bars Agent - Stage 2 of Hierarchical Fixture Layout Optimization

Given bar positions from Stage 1, places fixtures within each bar.

Objective: Maximize moment of inertia while:
- Respecting vertical spacing constraints
- Ensuring at least 1 fixture per bar (no empty bars)
- Respecting bar horizontal bounds (from stage 1)

Action: For current bar, choose fixture type and y-position
"""

import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from shapely.geometry import Point as ShapelyPoint, Polygon
import math

# Try to import moments_of_inertia module
try:
    from moments_of_inertia import InertiaAnalysis, AREA, BARYCENTER, ABSOLUTE_MOMENTS_OF_INERTIA, ANGLE
except ImportError:
    # Fallback: set to None, we'll check before using
    InertiaAnalysis = None
    AREA = "area"
    BARYCENTER = "barycenter"
    ABSOLUTE_MOMENTS_OF_INERTIA = "absolute_moments_of_inertia"
    ANGLE = "angle"
from dataclasses import dataclass
from enum import IntEnum

ROOT = Path(__file__).resolve().parent

# Workpiece constraints
WORKPIECE_CONSTRAINTS = {
    'simple_stair_step': {'min_fixtures': 1, 'max_fixtures': 10, 'min_bars': 1, 'max_bars': 6},
    'coffee_table': {'min_fixtures': 1, 'max_fixtures': 20, 'min_bars': 2, 'max_bars': 8},
    'dashboard': {'min_fixtures': 2, 'max_fixtures': 15, 'min_bars': 2, 'max_bars': 7},
    'speaker': {'min_fixtures': 1, 'max_fixtures': 12, 'min_bars': 2, 'max_bars': 6},
    'spiral_stair_step': {'min_fixtures': 2, 'max_fixtures': 18, 'min_bars': 2, 'max_bars': 8},
}

# Fixture types and dimensions
class FixtureType(IntEnum):
    NONE = 0
    TYPE_1 = 1  # 145 x 145 mm (square)
    TYPE_2 = 2  # 180 x 65 mm (rectangular)

FIXTURE_DIMENSIONS = {
    FixtureType.TYPE_1: (145.0, 145.0),  # width, height
    FixtureType.TYPE_2: (180.0, 65.0),
}

FIXTURE_AVAILABILITY = {
    FixtureType.TYPE_1: 24,
    FixtureType.TYPE_2: 12,
}

# Spacing constraints from MiniZinc
VERTICAL_SPACING_CONSTANT = 100.0  # mm
BAR_WIDTH = 145.0  # mm
HORIZONTAL_SPACING = 345.0  # mm (already checked in stage 1)


@dataclass
class FixtureState:
    """Represents a placed fixture"""
    x: float  # Top-left x
    y: float  # Top-left y
    type_id: int  # 1 or 2
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get (x_min, x_max, y_min, y_max)"""
        width, height = FIXTURE_DIMENSIONS[self.type_id]
        return self.x, self.x + width, self.y, self.y + height


class FixtureOnBarsEnv(gym.Env):
    """
    Gymnasium environment for placing fixtures on pre-positioned bars.
    
    Input: Bar positions (from stage 1)
    Action: For current bar, choose fixture type and y-position
    Objective: Maximize MOI while ensuring every bar has ≥1 fixture
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(
        self,
        bar_positions: List[float],
        workpiece_name: str = "simple_stair_step",
        render_mode: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize fixture placement environment.
        
        Args:
            bar_positions: List of bar center x-coordinates (from stage 1)
            workpiece_name: Name of the workpiece
            render_mode: "human" for visualization
            verbose: Print debug information
        """
        super().__init__()
        
        self.bar_positions = sorted(bar_positions)  # Sort for consistent ordering
        self.num_bars = len(self.bar_positions)
        self.workpiece_name = workpiece_name
        self.render_mode = render_mode
        self.verbose = verbose
        
        # MOI tracking
        self.cumulative_moi = 0.0
        self.principal_moment_i = 0.0
        self.principal_moment_j = 0.0
        
        # Get constraints
        if workpiece_name not in WORKPIECE_CONSTRAINTS:
            raise ValueError(f"Workpiece '{workpiece_name}' not found")
        
        self.constraints = WORKPIECE_CONSTRAINTS[workpiece_name]
        
        # Load workpiece data
        workpiece_file = ROOT / "resources" / "workpieces_information.json"
        with open(workpiece_file, 'r') as f:
            all_workpieces = json.load(f)
        
        if workpiece_name not in all_workpieces:
            raise ValueError(f"Workpiece '{workpiece_name}' not found")
        
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
        
        self.workpiece_polygon = Polygon(vertices)
        
        if self.verbose:
            print(f"[Fixture Placement] Workpiece: {workpiece_name}")
            print(f"[Fixture Placement] Size: {self.workpiece_width:.1f}mm x {self.workpiece_height:.1f}mm")
            print(f"[Fixture Placement] Bars: {len(self.bar_positions)} at x={[f'{x:.1f}' for x in self.bar_positions]}")
        
        # **DISCRETE ACTION SPACE**: 
        # Action 0: Skip bar (only if bar has ≥1 fixture)
        # Actions 1+: Place Type1 at valid_y[i], then Type2 at valid_y[i]
        # Max valid positions per bar: (workpiece_height / fixture_height) with discretization
        self.y_discretization = 1.0  # 1mm grid for y-positions (fine granularity)
        self.current_bar_idx = 0  # Initialize here so _precompute_valid_y_positions can use it
        self._precompute_valid_y_positions()
        
        # Action space: Discrete choices per bar
        # For simplicity: 2 * max_valid_positions + 1 (for skip action)
        max_actions = 2 * len(self.valid_y_positions_base) + 1  # Type1 and Type2 choices + skip
        self.action_space = spaces.Discrete(max_actions)
        
        # Observation space: normalized values
        obs_size = 1 + 1 + 1 + 2 + 1 + 1  # +1 for count of remaining valid positions
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # State
        self.fixtures: List[FixtureState] = []
        self.fixture_availability = {
            FixtureType.TYPE_1: FIXTURE_AVAILABILITY[FixtureType.TYPE_1],
            FixtureType.TYPE_2: FIXTURE_AVAILABILITY[FixtureType.TYPE_2],
        }
        self.fixtures_per_bar = [0] * self.num_bars  # Track fixtures in each bar
        self.current_bar_idx = 0  # Which bar we're currently filling
        self.step_count = 0
        self.max_steps = 100  # Increased to allow multiple fixtures per bar
        self.cumulative_moi = 0.0
        self.previous_moi = 0.0  # Track MOI for reward calculation
        
        # Valid y-positions for current bar (updated dynamically)
        self.valid_y_positions: List[float] = []  # Y-centers for current bar
    
    def _precompute_valid_y_positions(self):
        """
        Pre-compute base valid y-positions based on rectangular fixture (180×65mm).
        
        These are the y-centers where fixtures can be placed without:
        - Going out of workpiece bounds
        - Overlapping holes
        
        Discretization: 1mm grid for y-positions (fine granularity)
        Uses rectangular fixture height (65mm) as constraint
        """
        fixture_height = FIXTURE_DIMENSIONS[FixtureType.TYPE_2][1]  # 65mm
        max_y_center = self.workpiece_max_y - fixture_height / 2
        min_y_center = self.workpiece_min_y + fixture_height / 2
        
        # Generate grid of y-centers (10mm spacing)
        valid_y = []
        y_center = min_y_center
        while y_center <= max_y_center:
            # Check if this y-position avoids holes
            if self._check_y_position_for_holes(y_center, FixtureType.TYPE_2):
                valid_y.append(y_center)
            y_center += self.y_discretization
        
        self.valid_y_positions_base = valid_y
        if self.verbose:
            print(f"[Fixture Placement] Pre-computed {len(valid_y)} valid y-positions (10mm grid)")
    
    def _check_y_position_for_holes(self, y_center: float, fixture_type: int) -> bool:
        """
        Check if y-position at bar center avoids all holes.
        Returns True if position is valid (no hole overlap).
        """
        if 'holes' not in self.workpiece_data:
            return True
        
        width, height = FIXTURE_DIMENSIONS[fixture_type]
        bar_center_x = self.bar_positions[self.current_bar_idx]
        placement_x = bar_center_x - width / 2
        placement_y = y_center - height / 2
        
        # Check fixture bounds
        fixture_bounds = (placement_x, placement_y, placement_x + width, placement_y + height)
        fixture_polygon = Polygon([
            (fixture_bounds[0], fixture_bounds[1]),
            (fixture_bounds[2], fixture_bounds[1]),
            (fixture_bounds[2], fixture_bounds[3]),
            (fixture_bounds[0], fixture_bounds[3]),
        ])
        
        for hole in self.workpiece_data['holes']:
            if isinstance(hole, (list, tuple)) and len(hole) == 3:
                cx, cy, radius = hole
                hole_point = ShapelyPoint(cx, cy)
                if fixture_polygon.distance(hole_point) < radius:
                    return False
        
        return True
    
    def _update_valid_y_positions(self):
        """
        Update valid y-positions for current bar considering:
        - Existing fixtures on THIS bar's vertical spacing constraints
        - Hole overlaps (bars may have different holes!)
        """
        fixture_height = FIXTURE_DIMENSIONS[FixtureType.TYPE_2][1]  # 65mm
        max_y_center = self.workpiece_max_y - fixture_height / 2
        min_y_center = self.workpiece_min_y + fixture_height / 2
        
        valid_y = []
        for y_center in self.valid_y_positions_base:
            if y_center < min_y_center or y_center > max_y_center:
                continue
            
            # Check hole overlap for current bar (holes may differ per bar!)
            if not self._check_y_position_for_holes(y_center, FixtureType.TYPE_2):
                continue
            
            # Check spacing constraint only with existing fixtures on THIS bar
            spacing_ok = True
            for fixture in self.fixtures:
                # Skip fixtures not on current bar
                fixture_bar_x = fixture.x + FIXTURE_DIMENSIONS[fixture.type_id][0] / 2
                bar_center_x = self.bar_positions[self.current_bar_idx]
                
                # Check if fixture is on current bar (within bar bounds)
                bar_left = bar_center_x - BAR_WIDTH / 2
                bar_right = bar_center_x + BAR_WIDTH / 2
                if fixture_bar_x < bar_left or fixture_bar_x > bar_right:
                    # Fixture not on current bar, skip it
                    continue
                
                # This fixture is on current bar, check spacing
                if not self._check_vertical_spacing_y(y_center, fixture):
                    spacing_ok = False
                    break
            
            if spacing_ok:
                valid_y.append(y_center)
        
        self.valid_y_positions = valid_y
    
    def _check_vertical_spacing_y(self, y_center: float, other_fixture: FixtureState) -> bool:
        """
        Check if y_center maintains vertical spacing with other_fixture.
        Returns True if spacing constraint is satisfied.
        """
        # Assume we're checking against Type2 (rectangular, 65mm height)
        height1 = FIXTURE_DIMENSIONS[FixtureType.TYPE_2][1]  # 65mm
        width2, height2 = FIXTURE_DIMENSIONS[other_fixture.type_id]
        
        # Check horizontal overlap (both fixtures in same bar, so they overlap)
        # Both fixtures centered in the same bar, so they overlap horizontally
        
        # Compute vertical centers
        y1_center = y_center  # New fixture y-center
        y2_center = other_fixture.y + height2 / 2
        
        # Minimum required distance: height1/2 + height2/2 + 100mm
        min_distance = height1 / 2 + height2 / 2 + VERTICAL_SPACING_CONSTANT
        actual_distance = abs(y1_center - y2_center)
        
        return actual_distance >= min_distance
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.fixtures = []
        self.fixture_availability = {
            FixtureType.TYPE_1: FIXTURE_AVAILABILITY[FixtureType.TYPE_1],
            FixtureType.TYPE_2: FIXTURE_AVAILABILITY[FixtureType.TYPE_2],
        }
        self.fixtures_per_bar = [0] * self.num_bars
        self.current_bar_idx = 0
        self.step_count = 0
        self.cumulative_moi = 0.0
        self.principal_moment_i = 0.0
        self.principal_moment_j = 0.0
        self.previous_moi = 0.0
        
        # Initialize valid y-positions for first bar
        self._update_valid_y_positions()
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step with discrete action.
        
        Action space:
        - 0: Skip to next bar (only if current bar has ≥1 fixture)
        - 1 to len(valid_y): Place Type1 at valid_y[i-1]
        - len(valid_y)+1 to 2*len(valid_y): Place Type2 at valid_y[i-len(valid_y)-1]
        """
        self.step_count += 1
        reward = 0.0
        terminated = False
        info = {'bar_index': self.current_bar_idx, 'reason': '', 'action': action}
        
        action = int(action)
        num_valid_y = len(self.valid_y_positions)
        
        # Action 0: Skip to next bar
        if action == 0:
            if self.current_bar_idx < self.num_bars - 1 and self.fixtures_per_bar[self.current_bar_idx] >= 1:
                self.current_bar_idx += 1
                self._update_valid_y_positions()
                reward = 5.0
                info['reason'] = 'bar_advanced'
            else:
                # Penalty for invalid skip
                # Strong penalty for skipping too early if bar has <2 fixtures AND valid positions exist
                if self.fixtures_per_bar[self.current_bar_idx] < 2 and len(self.valid_y_positions) > 0:
                    reward = -20.0  # Strong penalty: skip = loss of ~2 fixture placements
                    info['reason'] = 'invalid_skip_early_abandon'
                else:
                    reward = -5.0  # Mild penalty for other invalid skips
                    info['reason'] = 'invalid_skip'
        # Actions 1 to num_valid_y: Place Type1
        elif 1 <= action <= num_valid_y:
            y_idx = action - 1
            fixture_type = FixtureType.TYPE_1
            reward = self._place_fixture_at_y(y_idx, fixture_type, info)
            if reward > 0:  # Successful placement
                self._update_valid_y_positions()
        # Actions num_valid_y+1 to 2*num_valid_y: Place Type2
        elif num_valid_y + 1 <= action <= 2 * num_valid_y:
            y_idx = action - num_valid_y - 1
            fixture_type = FixtureType.TYPE_2
            reward = self._place_fixture_at_y(y_idx, fixture_type, info)
            if reward > 0:  # Successful placement
                self._update_valid_y_positions()
        else:
            reward = -1.0
            info['reason'] = 'invalid_action'
        
        # Check termination
        all_bars_filled = all(count >= 1 for count in self.fixtures_per_bar)
        if all_bars_filled:
            # Large bonus for completing all bars
            reward += 100.0
            
            # Compute final proper moment of inertia
            principal_i, principal_j, moi_sum = self._compute_proper_moments_of_inertia()
            if InertiaAnalysis is not None and moi_sum > 0:
                # Bonus scaled to total MOI: normalize by 1e9 for meaningful scale
                # This incentivizes finding configurations with genuinely high MOI
                moi_completion_bonus = (moi_sum / 1e9) * 50.0  # Up to 50 bonus per 1e9 MOI
                reward += moi_completion_bonus
            
            info['principal_i'] = principal_i
            info['principal_j'] = principal_j
            info['moi_sum'] = moi_sum
            
            terminated = True
            info['reason'] = 'all_bars_filled'
        
        if self.step_count >= self.max_steps:
            terminated = True
            info['reason'] = 'max_steps_reached'
        
        obs = self._get_observation()
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def _place_fixture_at_y(self, y_idx: int, fixture_type: int, info: dict) -> float:
        """
        Place fixture of given type at valid y-position index.
        Reward is based on actual moment of inertia (MOI) increase.
        Returns reward (>0 for success, ≤0 for failure).
        """
        if y_idx < 0 or y_idx >= len(self.valid_y_positions):
            info['reason'] = 'invalid_y_index'
            return -1.0
        
        if self.fixture_availability[fixture_type] <= 0:
            info['reason'] = 'no_availability'
            return -1.0
        
        y_center = self.valid_y_positions[y_idx]
        width, height = FIXTURE_DIMENSIONS[fixture_type]
        
        # Compute placement (top-left corner)
        bar_center_x = self.bar_positions[self.current_bar_idx]
        placement_x = bar_center_x - width / 2
        placement_y = y_center - height / 2
        
        # Verify bounds
        if placement_y < self.workpiece_min_y or placement_y + height > self.workpiece_max_y:
            info['reason'] = 'out_of_bounds'
            return -1.0
        
        # Get MOI before placement
        if InertiaAnalysis is not None:
            _, _, moi_before = self._compute_proper_moments_of_inertia()
        else:
            moi_before = 0.0
        
        # Place fixture
        fixture = FixtureState(x=placement_x, y=placement_y, type_id=fixture_type)
        self.fixtures.append(fixture)
        
        is_first_in_bar = self.fixtures_per_bar[self.current_bar_idx] == 0
        self.fixtures_per_bar[self.current_bar_idx] += 1
        self.fixture_availability[fixture_type] -= 1
        
        # Compute MOI after placement
        if InertiaAnalysis is not None:
            principal_i, principal_j, moi_after = self._compute_proper_moments_of_inertia()
            self.principal_moment_i = principal_i
            self.principal_moment_j = principal_j
            moi_delta = moi_after - moi_before
        else:
            # Fallback: use simplified proxy if InertiaAnalysis unavailable
            center_x = placement_x + width / 2
            distance_from_center = abs(center_x - self.workpiece_width / 2)
            fixture_area = width * height
            moi_delta = fixture_area * distance_from_center
            moi_after = moi_before + moi_delta
        
        self.cumulative_moi = moi_after
        
        # **REWARD BASED ON PROPER MOMENT OF INERTIA**
        # More MOI increase = more reward
        # Increased scaling to make MOI differences impactful
        base_moi_reward = moi_delta * 2e-7
        
        # Strong bonus structure to encourage multiple fixtures per bar
        if is_first_in_bar:
            first_bar_bonus = 15.0
        else:
            # Generous bonuses for additional fixtures to overcome exploration cost
            additional_bonus = {
                1: 15.0,  # Second fixture gets LARGE bonus
                2: 10.0,  # Third fixture
            }
            first_bar_bonus = additional_bonus.get(self.fixtures_per_bar[self.current_bar_idx] - 1, 8.0)
        
        reward = base_moi_reward + first_bar_bonus
        
        info['reason'] = 'fixture_placed'
        info['fixture_type'] = fixture_type
        info['moi_increase'] = moi_delta
        info['total_moi'] = moi_after
        
        # **NO AUTO-ADVANCE**: Agent must use action 0 to move to next bar
        # This allows placing multiple fixtures per bar before advancing
        # The agent learns to stay on bar while valid positions exist and MOI rewards are good
        
        return reward
    
    
    def _get_fixture_placement(
        self, bar_idx: int, fixture_type: int, y_norm: float
    ) -> Tuple[bool, float, float]:
        """
        Compute fixture placement for given bar and y-position.
        
        Returns: (is_valid, placement_x, placement_y)
        """
        # Get bar bounds
        bar_center_x = self.bar_positions[bar_idx]
        bar_left = bar_center_x - BAR_WIDTH / 2
        bar_right = bar_center_x + BAR_WIDTH / 2
        
        fixture_width, fixture_height = FIXTURE_DIMENSIONS[fixture_type]
        
        # Compute placement x (center of bar)
        placement_x = bar_center_x - fixture_width / 2  # Top-left x
        
        # Compute placement y from normalized y-position
        placement_y = self.workpiece_min_y + y_norm * (self.workpiece_height - fixture_height)
        
        # Check bounds
        if placement_y < self.workpiece_min_y or placement_y + fixture_height > self.workpiece_max_y:
            return False, None, None
        
        # Check fixture fits horizontally within bar
        if placement_x < bar_left or placement_x + fixture_width > bar_right:
            return False, None, None
        
        # Check vertical spacing with other fixtures
        for other in self.fixtures:
            if not self._check_vertical_spacing(placement_x, placement_y, fixture_type, other):
                return False, None, None
        
        # Check hole overlaps (critical!)
        if not self._check_hole_overlap(placement_x, placement_y, fixture_type):
            return False, None, None
        
        # Check availability
        if self.fixture_availability[fixture_type] <= 0:
            return False, None, None
        
        return True, placement_x, placement_y
    
    def _check_vertical_spacing(
        self, x: float, y: float, fixture_type: int, other: FixtureState
    ) -> bool:
        """Check vertical spacing constraint"""
        width1, height1 = FIXTURE_DIMENSIONS[fixture_type]
        width2, height2 = FIXTURE_DIMENSIONS[other.type_id]
        
        # Check if fixtures overlap horizontally
        x1_min, x1_max = x, x + width1
        x2_min, x2_max = other.x, other.x + width2
        
        if x1_max <= x2_min or x1_min >= x2_max:
            return True  # No horizontal overlap, no vertical constraint needed
        
        # Horizontal overlap: check vertical spacing
        # Constraint: center-to-center distance ≥ height1/2 + height2/2 + 100mm
        y1_center = y + height1 / 2
        y2_center = other.y + height2 / 2
        
        min_distance = height1 / 2 + height2 / 2 + VERTICAL_SPACING_CONSTANT
        actual_distance = abs(y1_center - y2_center)
        
        return actual_distance >= min_distance
    
    def _check_hole_overlap(self, placement_x: float, placement_y: float, fixture_type: int) -> bool:
        """
        Check if fixture overlaps with any hole in the workpiece.
        
        Holes format: [cx, cy, radius] for circular holes
        Returns: True if no overlap (valid), False if overlaps (invalid)
        """
        if 'holes' not in self.workpiece_data:
            return True
        
        fixture_width, fixture_height = FIXTURE_DIMENSIONS[fixture_type]
        fixture_polygon = Polygon([
            (placement_x, placement_y),
            (placement_x + fixture_width, placement_y),
            (placement_x + fixture_width, placement_y + fixture_height),
            (placement_x, placement_y + fixture_height),
        ])
        
        for hole in self.workpiece_data['holes']:
            if isinstance(hole, (list, tuple)) and len(hole) == 3:
                # Circular hole: [cx, cy, radius]
                cx, cy, radius = hole
                hole_point = ShapelyPoint(cx, cy)
                
                # Check if hole center is within fixture bounds
                if fixture_polygon.distance(hole_point) < radius:
                    return False  # Hole overlaps fixture
        
        return True  # No hole overlap
    
    def _compute_proper_moments_of_inertia(self):
        """
        Compute proper principal moments of inertia (I, J) for current fixture configuration.
        Uses the moments_of_inertia module for accurate calculation.
        
        Returns:
            tuple: (principal_I, principal_J, I+J) or (0, 0, 0) if no fixtures or module unavailable
        """
        if len(self.fixtures) == 0 or InertiaAnalysis is None:
            return 0.0, 0.0, 0.0
        
        try:
            # Build polygon data for each fixture
            polygons = []
            for fixture in self.fixtures:
                width, height = FIXTURE_DIMENSIONS[fixture.type_id]
                
                # Rectangle vertices
                vertices = [
                    (fixture.x, fixture.y),
                    (fixture.x + width, fixture.y),
                    (fixture.x + width, fixture.y + height),
                    (fixture.x, fixture.y + height),
                ]
                
                area = InertiaAnalysis.compute_polygon_area(vertices)
                jx, jy, jxy = InertiaAnalysis.compute_absolute_moments_of_inertia(vertices)
                
                # Compute barycenter
                x_g = sum(v[0] for v in vertices) / len(vertices)
                y_g = sum(v[1] for v in vertices) / len(vertices)
                
                poly = {
                    AREA: area,
                    BARYCENTER: (x_g, y_g),
                    ABSOLUTE_MOMENTS_OF_INERTIA: (jx, jy, jxy),
                    ANGLE: 0
                }
                polygons.append(poly)
            
            # Compute overall center of gravity
            x_G, y_G = InertiaAnalysis.compute_overall_center_of_gravity(polygons)
            
            # Compute principal moments of inertia
            principal_i, principal_j = InertiaAnalysis.compute_combined_baricentric_moments_of_inertia(
                polygons, x_G, y_G
            )
            
            return principal_i, principal_j, principal_i + principal_j
        
        except Exception as e:
            # Fallback if calculation fails
            if self.verbose:
                print(f"[MOI Calculation Error] {e}")
            return 0.0, 0.0, 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get observation (7 elements total)"""
        obs = np.zeros(1 + 1 + 1 + 2 + 1 + 1, dtype=np.float32)
        
        idx = 0
        
        # Current bar index (normalized)
        obs[idx] = self.current_bar_idx / max(1, self.num_bars - 1)
        idx += 1
        
        # Fixtures in current bar
        obs[idx] = self.fixtures_per_bar[self.current_bar_idx] / 3.0  # Normalize assuming max 3 per bar
        idx += 1
        
        # Total fixtures placed (normalized by max)
        obs[idx] = len(self.fixtures) / self.constraints['max_fixtures']
        idx += 1
        
        # Fixture availability
        obs[idx] = self.fixture_availability[FixtureType.TYPE_1] / FIXTURE_AVAILABILITY[FixtureType.TYPE_1]
        obs[idx + 1] = self.fixture_availability[FixtureType.TYPE_2] / FIXTURE_AVAILABILITY[FixtureType.TYPE_2]
        idx += 2
        
        # Cumulative MOI (normalized)
        obs[idx] = min(1.0, self.cumulative_moi / 1e10)
        idx += 1
        
        # Remaining valid y-positions count (normalized)
        max_possible_y_positions = int((self.workpiece_max_y - self.workpiece_min_y) / self.y_discretization) + 1
        obs[idx] = len(self.valid_y_positions) / max(1, max_possible_y_positions)
        
        return obs
    
    def render(self):
        """Render placeholder"""
        if self.render_mode == "human":
            print(f"[Render] Bar {self.current_bar_idx}: {self.fixtures_per_bar[self.current_bar_idx]} fixtures")
    
    def close(self):
        """Close environment"""
        pass


if __name__ == "__main__":
    # Quick test with sample bar positions
    bar_positions = [200.0, 600.0]  # Two bars
    
    env = FixtureOnBarsEnv(
        bar_positions=bar_positions,
        workpiece_name="simple_stair_step",
        verbose=True
    )
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action}, reward={reward:.2f}, bar_idx={info['bar_index']}")
        
        if terminated:
            print(f"Episode terminated: {info['reason']}")
            break
    
    env.close()
    print(f"Total fixtures placed: {len(env.fixtures)}")
    print(f"Fixtures per bar: {env.fixtures_per_bar}")
