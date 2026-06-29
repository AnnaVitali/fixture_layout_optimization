"""
Graphical RL Agent for Fixture Layout Optimization using Gymnasium.

A visualization-based RL environment where an agent learns to place fixtures on a workpiece
by optimizing the sum of moments of inertia. The agent positions fixture centers while
respecting spacing constraints.

The objective is to MAXIMIZE the sum of moments of inertia (expressed as minimization
of the negative sum).

Constraints:
- Fixtures must be entirely within the workpiece area
- Horizontal distance between fixture centers: 290 mm (145*2)
- Vertical distance between fixture centers: 172.5 mm (145/2 + 100)
- No overlaps with excluded zones

Usage:
    python python/rl_graphical_agent.py --workpiece simple_stair_step
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc

# Data models
try:
    from python.machine_parameters import FIXTURE_AVAILABILITY, HORIZONTAL_SECUIRITY_DISTANCE, VERTICAL_SECUIRITY_DISTANCE, FixtureState, FIXTURE_DIMENSIONS
except ImportError:
    try:
        from models import FIXTURE_AVAILABILITY, HORIZONTAL_SECUIRITY_DISTANCE, VERTICAL_SECUIRITY_DISTANCE, FixtureState, FIXTURE_DIMENSIONS
    except ImportError:
        from machine_parameters import FIXTURE_AVAILABILITY, HORIZONTAL_SECUIRITY_DISTANCE, VERTICAL_SECUIRITY_DISTANCE, FixtureState, FIXTURE_DIMENSIONS

# Utility functions
try:
    from python.utility import define_fixture_from_state
    from python.moments_of_inertia import InertiaAnalysis
except ImportError:
    from utility import define_fixture_from_state
    from moments_of_inertia import InertiaAnalysis

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise SystemExit(
        "gymnasium is required. Install it with: pip install gymnasium"
    ) from exc

try:
    from shapely.geometry import Polygon, box, Point as ShapelyPoint
except ImportError as exc:
    raise SystemExit(
        "shapely is required. Install it with: pip install shapely"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]


def get_horizontal_spacing(fixture_type: int) -> float:
    """
    Get horizontal spacing required for a fixture type.
    
    Based on MiniZinc constraint: xc(i) + WBar + WMin <= xc(j)
    Where WBar is a standard fixture width and WMin is security distance.
    We use the max fixture width (Type 1 = 145) as WBar to be conservative.
    
    Returns:
        Distance between centers: 145 + 200 = 345 mm (constant for all types)
    """
    # Use maximum fixture width (145 from Type 1) + security distance
    # This matches MiniZinc: WBar (145) + WMin (200) = 345
    return 145 + HORIZONTAL_SECUIRITY_DISTANCE
    

def get_vertical_spacing(fixture_type: int) -> float:
    """
    Get vertical spacing required for a fixture type.
    
    This is only used for informational purposes now.
    Actual vertical spacing is computed dynamically based on both fixture types.
    """
    _ , height = FIXTURE_DIMENSIONS[fixture_type]
    return (height / 2) + VERTICAL_SECUIRITY_DISTANCE

def get_max_horizontal_spacing() -> float:
    """Get maximum horizontal spacing across all fixture types."""
    return max(get_horizontal_spacing(ftype) for ftype in FIXTURE_DIMENSIONS.keys())

def get_max_vertical_spacing() -> float:
    """Get maximum vertical spacing across all fixture types."""
    return max(get_vertical_spacing(ftype) for ftype in FIXTURE_DIMENSIONS.keys())

CENTER_ALIGNMENT_TOLERANCE = 1.0  # mm tolerance for same-column detection

WORKPIECE_CONSTRAINTS = {
    'coffee_table': {'min_fixtures': 6, 'max_fixtures': 10},
    'simple_stair_step': {'min_fixtures': 3, 'max_fixtures': 6},
    'dashboard': {'min_fixtures': 3, 'max_fixtures': 6},
    'speaker': {'min_fixtures': 3, 'max_fixtures': 6},
    'spiral_stair_step': {'min_fixtures': 3, 'max_fixtures': 6},
}


class WorkpieceVisualizer:
    """Handles visualization of the workpiece and fixtures from a top-down view."""
    
    def __init__(self, workpiece_data: Dict, figure_size: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualizer.
        
        Args:
            workpiece_data: Dictionary containing vertices, holes, and rectangles
            figure_size: Size of the figure (width, height) in inches
        """
        self.workpiece_data = workpiece_data
        self.figure_size = figure_size
        
        # Calculate bounds
        vertices = np.array(workpiece_data['vertices'])
        self.min_x = vertices[:, 0].min()
        self.max_x = vertices[:, 0].max()
        self.min_y = vertices[:, 1].min()
        self.max_y = vertices[:, 1].max()
        
        # Add padding
        self.padding = 50
        
    def get_figure(self, fixtures: List[Dict] = None, title: str = "Fixture Layout") -> Figure:
        """
        Generate a matplotlib figure showing the workpiece and fixtures.
        
        Args:
            fixtures: List of fixture dictionaries with state, area, etc.
            title: Title for the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Draw workpiece boundary
        vertices = self.workpiece_data['vertices']
        workpiece_poly = patches.Polygon(
            vertices,
            closed=True,
            edgecolor='black',
            facecolor='lightgray',
            linewidth=2,
            alpha=0.3
        )
        ax.add_patch(workpiece_poly)
        
        # Draw holes (excluded regions)
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if len(hole) == 3:  # Circle: [cx, cy, radius]
                    circle = patches.Circle(
                        (hole[0], hole[1]),
                        hole[2],
                        edgecolor='red',
                        facecolor='red',
                        alpha=0.2,
                        linewidth=1.5,
                        linestyle='--'
                    )
                    ax.add_patch(circle)
        
        # Draw fixtures if provided
        if fixtures:
            for idx, fixture in enumerate(fixtures):
                self._draw_fixture(ax, fixture, idx)
        
        # Set axis properties
        ax.set_xlim(self.min_x - self.padding, self.max_x + self.padding)
        ax.set_ylim(self.min_y - self.padding, self.max_y + self.padding)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Standard computer graphics coordinate system
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _draw_fixture(self, ax, fixture: Dict, fixture_id: int):
        """Draw a single fixture on the axis."""
        state = fixture.get('state')
        if not state:
            return
        
        # Get fixture dimensions
        t = state.type_id
        width, height = FIXTURE_DIMENSIONS.get(t, (0, 0))
        
        # Calculate corners
        corners = [
            (state.x, state.y),
            (state.x + width, state.y),
            (state.x + width, state.y + height),
            (state.x, state.y + height),
        ]
        
        # Rotate corners around center
        center_x = state.x + width / 2
        center_y = state.y + height / 2
        angle_rad = math.radians(state.angle)
        
        rotated_corners = []
        for cx, cy in corners:
            dx = cx - center_x
            dy = cy - center_y
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + center_x
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + center_y
            rotated_corners.append((rx, ry))
        
        # Draw fixture
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        color = colors[fixture_id % len(colors)]
        
        fixture_poly = patches.Polygon(
            rotated_corners,
            closed=True,
            edgecolor=color,
            facecolor=color,
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(fixture_poly)
        
        # Draw center point
        ax.plot(center_x, center_y, 'k+', markersize=8, markeredgewidth=2)
        
        # Add fixture ID label
        ax.text(center_x, center_y, str(fixture_id), 
                fontsize=8, ha='center', va='center', color='white', fontweight='bold')


class FixtureLayoutEnv(gym.Env):
    """
    Gymnasium environment for fixture layout optimization with moment of inertia objective.
    
    The agent positions fixture centers while respecting spacing constraints:
    - Horizontal spacing: 290 mm between centers
    - Vertical spacing: 172.5 mm between centers
    - Fixtures must be entirely within workpiece
    
    Objective: MAXIMIZE sum of moments of inertia (minimize negative sum).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        workpiece_name: str = "simple_stair_step",
        grid_resolution: float = 10.0,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        use_continuous_action_space: bool = False,
        use_hybrid_action_space: bool = True,
        max_steps: int = 20
    ):
        """
        Initialize the fixture layout environment.
        
        Args:
            workpiece_name: Name of the workpiece to use
            grid_resolution: Resolution of placement grid in mm (default 10mm for finer visualization)
            render_mode: "human" for visualization, "rgb_array" for numpy array
            verbose: Whether to print debug information during training
            use_continuous_action_space: If True, use pure continuous normalized [0,1] coordinates (legacy, learning inefficient)
            use_hybrid_action_space: If True, use hybrid discrete grid + continuous fine-tuning (recommended, default True)
            max_steps: Maximum number of steps per episode (default 100)
        """
        super().__init__()
        
        self.workpiece_name = workpiece_name
        self.grid_resolution = grid_resolution
        self.render_mode = render_mode
        self.verbose = verbose
        self.use_continuous_action_space = use_continuous_action_space
        self.use_hybrid_action_space = use_hybrid_action_space
        
        # Hybrid mode: fine-tuning offset range in mm around discrete grid positions
        self.hybrid_offset_range = 1.0  # ±5mm fine-tuning
        
        # Get workpiece constraints
        if workpiece_name not in WORKPIECE_CONSTRAINTS:
            raise ValueError(f"Workpiece '{workpiece_name}' not found in WORKPIECE_CONSTRAINTS")
        
        self.constraints = WORKPIECE_CONSTRAINTS[workpiece_name]
        self.min_fixtures = self.constraints['min_fixtures']
        self.max_fixtures = self.constraints['max_fixtures']
        
        # Load workpiece data
        workpiece_file = ROOT / "python" / "resources" / "workpieces_information.json"
        with open(workpiece_file, 'r') as f:
            all_workpieces = json.load(f)
        
        if workpiece_name not in all_workpieces:
            raise ValueError(f"Workpiece '{workpiece_name}' not found in workpieces_information.json")
        
        self.workpiece_data = all_workpieces[workpiece_name]
        self.visualizer = WorkpieceVisualizer(self.workpiece_data)
        
        # Transform workpiece vertices to mathematical coordinate system (0,0 at lower-left, y increases upward)
        workpiece_height = self.visualizer.max_y - self.visualizer.min_y
        workpiece_width = self.visualizer.max_x - self.visualizer.min_x
        
        # OPTIMIZATION: Adaptive grid resolution based on workpiece size
        # Large workpieces (coffee_table ~1500mm²) use coarser grid to reduce computation
        # Smaller workpieces use finer grid for precision
        workpiece_area = workpiece_width * workpiece_height
        if workpiece_area > 1000000 and self.grid_resolution < 20:  # Large workpiece, fine grid requested
            print(f"[Optimization] Large workpiece detected ({workpiece_area:.0f}mm²). "
                  f"Adjusting grid resolution from {self.grid_resolution}mm to 20.0mm for faster computation.")
            self.grid_resolution = 20.0  # Use coarser grid for large workpieces
        
        transformed_vertices = []
        for vx, vy in self.workpiece_data['vertices']:
            x_transformed = vx - self.visualizer.min_x
            y_transformed = workpiece_height - (vy - self.visualizer.min_y)
            transformed_vertices.append((x_transformed, y_transformed))
        
        # Create workpiece polygon for collision detection (in transformed space)
        self.workpiece_polygon = Polygon(transformed_vertices)
        
        # Create exclusion zones from holes in transformed space (must be before _generate_valid_points)
        self.exclusion_zones = []
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if len(hole) == 3:  # Circle
                    x_hole_transformed = hole[0] - self.visualizer.min_x
                    y_hole_transformed = workpiece_height - (hole[1] - self.visualizer.min_y)
                    zone = ShapelyPoint(x_hole_transformed, y_hole_transformed).buffer(hole[2])
                    self.exclusion_zones.append(zone)
        
        # Generate valid placement points inside the workpiece
        self.valid_points = self._generate_valid_points()
        self.num_placement_points = len(self.valid_points)
        
        if self.num_placement_points == 0:
            raise ValueError(f"No valid placement points found for {workpiece_name}")
        
        # Action space: [fixture_type, x_discrete, y_discrete]
        # Fixtures are placed without rotation (angle = 0)
        
        # Create x and y discrete ranges in mathematical coordinate system (0,0 at lower-left)
        workpiece_width = self.visualizer.max_x - self.visualizer.min_x
        self.x_positions = np.arange(
            0,
            workpiece_width + self.grid_resolution,
            self.grid_resolution
        )
        self.y_positions = np.arange(
            0,
            workpiece_height + self.grid_resolution,
            self.grid_resolution
        )
        
        # Store workpiece dimensions for continuous action space
        self.workpiece_width = workpiece_width
        self.workpiece_height = workpiece_height
        
        # Preprocess grid positions: identify which positions can accommodate at least one fixture type
        # This is done once during init to filter out perimeter positions that can't fit any fixture
        self.safe_grid_positions = self._preprocess_valid_grid_positions()
        
        # Action space: depends on mode
        if self.use_hybrid_action_space:
            # Hybrid: discrete grid + continuous fine-tuning
            # Action: [fixture_type_norm, grid_position_norm, offset_x, offset_y]
            # - fixture_type_norm ∈ [0, 1]: scale to 0-2, round to int
            # - grid_position_norm ∈ [0, 1]: scale to valid grid indices
            # - offset_x ∈ [-1, 1]: fine-tuning in mm (scaled by hybrid_offset_range)
            # - offset_y ∈ [-1, 1]: fine-tuning in mm (scaled by hybrid_offset_range)
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        elif self.use_continuous_action_space:
            # Pure continuous: [fixture_type, x_normalized, y_normalized]
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([2.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Discrete grid: [fixture_type, valid_pair_idx]
            max_possible_pairs = len(self.x_positions) * len(self.y_positions)
            self.action_space = spaces.MultiDiscrete([
                3,  # fixture type: 0=none, 1=square, 2=rectangular
                max_possible_pairs,  # index into valid_pairs list
            ])
        
        # Observation space: 
        # - Fixture availability for type 1 and type 2 (2 values, normalized)
        # - Current placed fixtures info (max_fixtures * 5: x_norm, y_norm, angle_norm, type_norm, is_active)
        # - Sum of moments of inertia (1 value, normalized)
        obs_size = 2 + (self.max_fixtures * 5) + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Initialize state
        self.fixtures: List[FixtureState] = []
        self.fixture_availability = {
            1: FIXTURE_AVAILABILITY[1],  # Square fixtures
            2: FIXTURE_AVAILABILITY[2],  # Rectangular fixtures
        }
        self.step_count = 0
        self.max_steps = max_steps  # Configurable max steps per episode
        self.cumulative_moment = 0.0
        
        # PERFORMANCE: Cache fixture bounds and centers to avoid recomputation
        # Each fixture cached as {center: (cx, cy), bounds: (minx, maxx, miny, maxy)}
        self.fixture_cache = {}  # Maps fixture index to {center, bounds, type_id}
        
        # Visualization
        self.fig = None
        self.ax = None
        self.is_displaying = False
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.fixtures = []
        self.fixture_availability = {
            1: FIXTURE_AVAILABILITY[1],
            2: FIXTURE_AVAILABILITY[2],
        }
        self.fixture_cache = {}  # PERFORMANCE: Clear cache on reset
        self.step_count = 0
        self.cumulative_moment = 0.0
        
        obs = self._get_observation()
        info = {}
        
        if self.render_mode == "human":
            self.render()
        
        return obs, info
    
    def _preprocess_valid_grid_positions(self) -> set:
        """
        Preprocess grid positions to identify which can accommodate at least one fixture type.
        
        This runs once during initialization to filter out perimeter positions where fixtures
        would extend outside the workpiece or overlap with exclusion zones.
        
        OPTIMIZATION: Uses bounding box filtering before expensive polygon intersection tests.
        
        Returns:
            Set of (x_idx, y_idx) tuples representing safe grid positions
        """
        import time
        start_time = time.time()
        
        safe_positions = set()
        
        # Precompute fixture dimensions to avoid repeated lookups
        fixture_dims = {1: FIXTURE_DIMENSIONS[1], 2: FIXTURE_DIMENSIONS[2]}
        max_width = max(w for w, h in fixture_dims.values())
        max_height = max(h for w, h in fixture_dims.values())
        
        # Get workpiece bounds
        minx, miny, maxx, maxy = self.workpiece_polygon.bounds
        
        # Check each grid position with early exit optimizations
        for x_idx, x_pos in enumerate(self.x_positions):
            for y_idx, y_pos in enumerate(self.y_positions):
                # FAST CHECK: Bounding box filter first (O(1) vs O(n) polygon check)
                # Check if position could possibly fit any fixture based on bounds
                half_max_width = max_width / 2
                half_max_height = max_height / 2
                
                if (x_pos - half_max_width < minx or x_pos + half_max_width > maxx or
                    y_pos - half_max_height < miny or y_pos + half_max_height > maxy):
                    # Outside workpiece bounds, skip expensive polygon checks
                    continue
                
                # Check if at least one fixture type can fit at this position
                can_fit = False
                
                # Try Type 1 (larger/square) first as it's more constrained
                for fixture_type in [1, 2]:
                    width, height = fixture_dims[fixture_type]
                    top_left_x = x_pos - width / 2
                    top_left_y = y_pos - height / 2
                    
                    # Bounds check (redundant for first type after above, but faster than polygon check)
                    if (top_left_x < minx or top_left_x + width > maxx or
                        top_left_y < miny or top_left_y + height > maxy):
                        continue
                    
                    # Only now do expensive polygon intersection check
                    fixture_box = box(top_left_x, top_left_y, top_left_x + width, top_left_y + height)
                    
                    # Check if entire fixture would be within workpiece
                    if fixture_box.within(self.workpiece_polygon):
                        # Check no overlap with exclusion zones (fast early exit if no zones)
                        overlaps_hole = False
                        if self.exclusion_zones:  # Only check if zones exist
                            for zone in self.exclusion_zones:
                                if fixture_box.intersects(zone):
                                    overlaps_hole = True
                                    break
                        
                        if not overlaps_hole:
                            can_fit = True
                            break  # At least one fixture type fits, so include this position
                
                if can_fit:
                    safe_positions.add((x_idx, y_idx))
        
        # Debug output
        total_positions = len(self.x_positions) * len(self.y_positions)
        safe_count = len(safe_positions)
        elapsed = time.time() - start_time
        print(f"[Grid Analysis] Grid resolution: {self.grid_resolution}mm")
        print(f"[Grid Analysis] Total grid positions: {total_positions}")
        print(f"[Grid Analysis] Safe positions (fixture fits inside workpiece): {safe_count}")
        print(f"[Grid Analysis] Safe positions coverage: {100*safe_count/total_positions:.1f}%")
        print(f"[Grid Analysis] Preprocessing time: {elapsed:.2f}s")
        
        return safe_positions
    
    def _compute_valid_action_positions(self) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Compute valid (x, y) index pairs for the next fixture based on current placement and spacing.
        
        PERFORMANCE: Uses cached fixture centers/bounds to avoid recomputation. Rectangular bounding box
        checks before expensive distance calculations. Only checks fixtures within max_spacing distance.
        
        A position is valid if AT LEAST ONE fixture type can be placed there without violating spacing.
        
        Returns:
            (valid_pairs, valid_x_indices, valid_y_indices) where:
            - valid_pairs: list of (x_idx, y_idx) tuples that form valid positions
            - valid_x_indices: list of x indices used by valid pairs
            - valid_y_indices: list of y indices used by valid pairs
        """
        valid_pairs = []
        
        # PERFORMANCE: For large workpieces, build a spatial index of existing fixtures
        # to avoid checking all fixtures for every grid position
        max_spacing_dist = 400.0  # Horizontal spacing is 345mm, so ~400mm is max check distance
        
        # Check each safe grid position
        for x_idx, y_idx in self.safe_grid_positions:
            x_pos = self.x_positions[x_idx]
            y_pos = self.y_positions[y_idx]
            
            # Check if this position is valid for AT LEAST ONE fixture type
            is_valid_for_any = False
            
            # Try both fixture types at this position (prefer type 1, then type 2)
            for new_fixture_type in [1, 2]:
                violates_spacing = False
                
                # PERFORMANCE: Use cached fixture bounds - only check fixtures in nearby region
                for fixture_idx, cache in self.fixture_cache.items():
                    # Quick bounding box check before distance computation (O(1) vs O(distance calc))
                    bounds = cache['bounds']  # (minx, maxx, miny, maxy)
                    if (x_pos < bounds[0] - max_spacing_dist or 
                        x_pos > bounds[1] + max_spacing_dist or
                        y_pos < bounds[2] - max_spacing_dist or 
                        y_pos > bounds[3] + max_spacing_dist):
                        continue  # This position is too far from this fixture
                    
                    # Now do the full spacing check only for nearby fixtures
                    fx_center, fy_center = cache['center']
                    dx = abs(x_pos - fx_center)
                    dy = abs(y_pos - fy_center)
                    
                    existing_fixture_type = cache['type_id']
                    if self._violates_spacing(dx, dy, existing_fixture_type, new_fixture_type):
                        violates_spacing = True
                        break
                
                if not violates_spacing:
                    is_valid_for_any = True
                    break
            
            if is_valid_for_any:
                valid_pairs.append((x_idx, y_idx))
        
        # Debug output on first step
        if self.verbose and (self.step_count == 0 or len(self.fixtures) == 0):
            safe_count = len(self.safe_grid_positions)
            valid_count = len(valid_pairs)
            print(f"[Step {self.step_count}] Valid action positions: {valid_count}/{safe_count} grid positions")
        
        # Extract unique x and y indices from valid pairs
        valid_x_set = set(x_idx for x_idx, _ in valid_pairs)
        valid_y_set = set(y_idx for _, y_idx in valid_pairs)
        
        return valid_pairs, sorted(list(valid_x_set)), sorted(list(valid_y_set))
    
    def _select_best_valid_action(self, valid_objectives: Dict[Tuple[int, int, int], float]) -> Optional[Tuple[int, int, int]]:
        """
        Select the best valid action based on objective value and flexibility.
        
        Strategy:
        - FIRST PLACEMENT (no fixtures yet): Maximize distance from workpiece center while preserving flexibility
        - SUBSEQUENT PLACEMENTS: Score 80% MOI + 20% flexibility
        
        Args:
            valid_objectives: Dictionary mapping (x_idx, y_idx, fixture_type) to MOI value
            
        Returns:
            (x_idx, y_idx, fixture_type) of the best action, or None if no valid actions
        """
        if not valid_objectives:
            return None
        
        # FIRST PLACEMENT: Prioritize distance from center
        if len(self.fixtures) == 0:
            workpiece_center_x = self.workpiece_width / 2
            workpiece_center_y = self.workpiece_height / 2
            
            scored_actions = []
            
            for (x_idx, y_idx, fixture_type), moi in valid_objectives.items():
                x_pos = self.x_positions[x_idx]
                y_pos = self.y_positions[y_idx]
                
                # Distance from workpiece center
                distance = math.sqrt((x_pos - workpiece_center_x)**2 + (y_pos - workpiece_center_y)**2)
                
                # Simulate placement to count remaining positions
                width, height = FIXTURE_DIMENSIONS[fixture_type]
                top_left_x = x_pos - width / 2
                top_left_y = y_pos - height / 2
                
                temp_state = FixtureState(x=top_left_x, y=top_left_y, angle=0.0, type_id=fixture_type)
                temp_fixtures = self.fixtures + [temp_state]
                
                original_fixtures = self.fixtures
                self.fixtures = temp_fixtures
                valid_pairs_after, _, _ = self._compute_valid_action_positions()
                self.fixtures = original_fixtures
                
                remaining_positions = len(valid_pairs_after)
                
                # Score: prioritize distance (80%), then flexibility (20%)
                # Normalize distance to [0,1] (max distance is diagonal of workpiece)
                max_distance = math.sqrt(self.workpiece_width**2 + self.workpiece_height**2) / 2
                distance_ratio = distance / max_distance if max_distance > 0 else 0.5
                
                flexibility_ratio = min(1.0, remaining_positions / 50)
                
                score = (0.8 * distance_ratio) + (0.2 * flexibility_ratio)
                
                scored_actions.append({
                    'action': (x_idx, y_idx, fixture_type),
                    'distance': distance,
                    'remaining_positions': remaining_positions,
                    'score': score
                })
            
            # Find maximum score
            max_score = max(a['score'] for a in scored_actions)
            
            # Get all actions within 1% of max score
            best_actions = [a for a in scored_actions if a['score'] >= max_score * 0.99]
            
            # Randomly select among best actions
            random_idx = self.np_random.integers(0, len(best_actions))
            return best_actions[random_idx]['action']
        
        # SUBSEQUENT PLACEMENTS: Score 80% MOI + 20% flexibility
        else:
            max_moi = max(valid_objectives.values())
            
            # Group actions by position
            positions_with_types = {}
            for x_idx, y_idx, fixture_type in valid_objectives.keys():
                key = (x_idx, y_idx)
                if key not in positions_with_types:
                    positions_with_types[key] = []
                positions_with_types[key].append(fixture_type)
            
            # Score each position-fixture_type combination
            scored_actions = []
            
            for (x_idx, y_idx), available_types in positions_with_types.items():
                for fixture_type in available_types:
                    moi = valid_objectives[(x_idx, y_idx, fixture_type)]
                    
                    # Simulate this placement to see flexibility impact
                    width, height = FIXTURE_DIMENSIONS[fixture_type]
                    x_pos = self.x_positions[x_idx]
                    y_pos = self.y_positions[y_idx]
                    top_left_x = x_pos - width / 2
                    top_left_y = y_pos - height / 2
                    
                    temp_state = FixtureState(x=top_left_x, y=top_left_y, angle=0.0, type_id=fixture_type)
                    temp_fixtures = self.fixtures + [temp_state]
                    
                    # Count valid positions after this placement
                    original_fixtures = self.fixtures
                    self.fixtures = temp_fixtures
                    valid_pairs_after, _, _ = self._compute_valid_action_positions()
                    self.fixtures = original_fixtures  # Restore
                    
                    remaining_positions = len(valid_pairs_after)
                    
                    # Score: prioritize (1) MOI, (2) flexibility
                    # MOI_ratio: normalize to [0,1] range relative to max
                    moi_ratio = moi / max_moi if max_moi > 0 else 0.5
                    
                    # Combined score: 80% MOI, 20% flexibility (remaining positions)
                    # This ensures MOI is primary driver but fixture type is considered
                    score = (0.8 * moi_ratio) + (0.2 * min(1.0, remaining_positions / 50))
                    
                    scored_actions.append({
                        'action': (x_idx, y_idx, fixture_type),
                        'moi': moi,
                        'remaining_positions': remaining_positions,
                        'moi_ratio': moi_ratio,
                        'score': score
                    })
            
            if not scored_actions:
                return None
            
            # Find the maximum score
            max_score = max(a['score'] for a in scored_actions)
            
            # Get all actions within 1% of max score
            best_actions = [a for a in scored_actions if a['score'] >= max_score * 0.99]
            
            # Randomly select among best actions
            random_idx = self.np_random.integers(0, len(best_actions))
            return best_actions[random_idx]['action']
    
    def _compute_valid_action_objectives(self) -> Dict[Tuple[int, int, int], float]:
        """
        Compute objective function (MOI) values for each valid action.
        
        For each valid position and fixture type, compute the MOI that would result
        from placing a fixture of that type at that position.
        
        Returns:
            Dictionary mapping (x_idx, y_idx, fixture_type) to MOI value
        """
        valid_pairs, _, _ = self._compute_valid_action_positions()
        objectives = {}
        
        for x_idx, y_idx in valid_pairs:
            x_pos = self.x_positions[x_idx]
            y_pos = self.y_positions[y_idx]
            
            # Try both fixture types at this position
            for fixture_type in [1, 2]:
                # Check if placement would be valid
                is_valid, _ = self._is_valid_placement(x_pos, y_pos, fixture_type, 0.0)
                
                if is_valid and self.fixture_availability[fixture_type] > 0:
                    # Compute MOI if we place this fixture
                    width, height = FIXTURE_DIMENSIONS[fixture_type]
                    top_left_x = x_pos - width / 2
                    top_left_y = y_pos - height / 2
                    
                    # Temporarily add fixture and compute MOI
                    temp_state = FixtureState(x=top_left_x, y=top_left_y, angle=0.0, type_id=fixture_type)
                    temp_fixtures = self.fixtures + [temp_state]
                    moi_value = self._compute_system_moment_of_inertia(temp_fixtures)
                    
                    objectives[(x_idx, y_idx, fixture_type)] = moi_value
        
        return objectives
        """
        Compute objective function (MOI) values for each valid action.
        
        For each valid position and fixture type, compute the MOI that would result
        from placing a fixture of that type at that position.
        
        Returns:
            Dictionary mapping (x_idx, y_idx, fixture_type) to MOI value
        """
        valid_pairs, _, _ = self._compute_valid_action_positions()
        objectives = {}
        
        for x_idx, y_idx in valid_pairs:
            x_pos = self.x_positions[x_idx]
            y_pos = self.y_positions[y_idx]
            
            # Try both fixture types at this position
            for fixture_type in [1, 2]:
                # Check if placement would be valid
                is_valid, _ = self._is_valid_placement(x_pos, y_pos, fixture_type, 0.0)
                
                if is_valid and self.fixture_availability[fixture_type] > 0:
                    # Compute MOI if we place this fixture
                    width, height = FIXTURE_DIMENSIONS[fixture_type]
                    top_left_x = x_pos - width / 2
                    top_left_y = y_pos - height / 2
                    
                    # Temporarily add fixture and compute MOI
                    temp_state = FixtureState(x=top_left_x, y=top_left_y, angle=0.0, type_id=fixture_type)
                    temp_fixtures = self.fixtures + [temp_state]
                    moi_value = self._compute_system_moment_of_inertia(temp_fixtures)
                    
                    objectives[(x_idx, y_idx, fixture_type)] = moi_value
        
        return objectives
    
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Depends on action space mode:
                   - Hybrid: [fixture_type_norm, grid_pos_norm, offset_x, offset_y]
                   - Continuous: [fixture_type_norm, x_norm, y_norm]
                   - Discrete: [fixture_type, valid_pair_idx]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        done = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0
        info = {
            'placement_success': False,
            'reason': 'no_action',
            'fixtures_count': len(self.fixtures),
            'fixtures_available': dict(self.fixture_availability),
            'sum_moments': self.cumulative_moment,
        }
        
        # Parse action based on action space mode
        center_x = None
        center_y = None
        fixture_type = None
        
        if self.use_hybrid_action_space:
            # Hybrid: [fixture_type_norm, grid_position_norm, offset_x, offset_y]
            fixture_type_norm = float(action[0])
            grid_position_norm = float(action[1])
            offset_x = float(action[2])  # ∈ [-1, 1]
            offset_y = float(action[3])  # ∈ [-1, 1]
            
            # Convert to discrete fixture type
            fixture_type = int(round(fixture_type_norm * 2))
            fixture_type = np.clip(fixture_type, 0, 2)
            
            # Get valid positions and select via normalized grid position
            if fixture_type > 0 and fixture_type <= 2:
                valid_pairs, _, _ = self._compute_valid_action_positions()
                
                if len(valid_pairs) > 0:
                    # Map normalized grid position [0,1] to valid pair index
                    grid_position_idx = int(grid_position_norm * len(valid_pairs))
                    grid_position_idx = np.clip(grid_position_idx, 0, len(valid_pairs) - 1)
                    
                    # Get grid center position
                    x_idx, y_idx = valid_pairs[grid_position_idx]
                    grid_center_x = self.x_positions[x_idx]
                    grid_center_y = self.y_positions[y_idx]
                    
                    # Apply continuous fine-tuning offset (±hybrid_offset_range mm)
                    center_x = grid_center_x + offset_x * self.hybrid_offset_range
                    center_y = grid_center_y + offset_y * self.hybrid_offset_range
                    
                    # Clamp to workpiece bounds
                    center_x = np.clip(center_x, 0, self.workpiece_width)
                    center_y = np.clip(center_y, 0, self.workpiece_height)
        
        elif self.use_continuous_action_space:
            # Continuous: [fixture_type_norm, x_norm, y_norm]
            fixture_type_norm = float(action[0])
            x_norm = float(action[1])
            y_norm = float(action[2])
            
            # Convert to discrete fixture type
            fixture_type = int(round(fixture_type_norm))
            fixture_type = np.clip(fixture_type, 0, 2)
            
            # Convert normalized coordinates to world coordinates
            center_x = x_norm * self.workpiece_width
            center_y = y_norm * self.workpiece_height
            center_x = np.clip(center_x, 0, self.workpiece_width)
            center_y = np.clip(center_y, 0, self.workpiece_height)
        
        else:
            # Discrete: [fixture_type, pair_idx]
            fixture_type, pair_idx = action
        
        # If fixture_type is 0 or invalid, do nothing (penalize inaction to encourage exploration)
        if fixture_type is None or fixture_type == 0:
            reward = -0.5
            info['reason'] = 'no_op_action'
        elif fixture_type > 0 and fixture_type <= 2:
            # Check if fixture type is available
            if self.fixture_availability[fixture_type] <= 0:
                reward = -1.0
                info['reason'] = 'no_availability'
            # Check if we can place more fixtures
            elif len(self.fixtures) >= self.max_fixtures:
                reward = -1.0
                info['reason'] = 'max_fixtures_reached'
            else:
                # Compute valid objectives and select best action
                valid_pairs, _, _ = self._compute_valid_action_positions()
                valid_objectives = self._compute_valid_action_objectives()
                
                if len(valid_objectives) == 0:
                    reward = -0.25
                    info['reason'] = 'no_valid_positions'
                else:
                    # Select the best valid action (highest MOI, far from center if first placement)
                    best_action = self._select_best_valid_action(valid_objectives)
                    
                    if best_action is None:
                        reward = -0.25
                        info['reason'] = 'no_valid_positions'
                    else:
                        # Extract best action
                        best_x_idx, best_y_idx, best_fixture_type = best_action
                        
                        # Get world coordinates for best position
                        center_x = self.x_positions[best_x_idx]
                        center_y = self.y_positions[best_y_idx]
                        fixture_type = best_fixture_type
                        
                        # Now attempt placement with best position
                        if center_x is not None and center_y is not None:
                            # Fixtures are placed without rotation
                            angle = 0.0
                            
                            # Check if placement is valid
                            is_valid, reason = self._is_valid_placement(center_x, center_y, fixture_type, angle)
                            
                            if is_valid:
                                # Compute valid positions BEFORE placement
                                num_valid_before = len(valid_pairs)
                                
                                # Get objective value for this specific placement
                                placement_moi = valid_objectives.get(best_action, None)
                                
                                width, height = FIXTURE_DIMENSIONS[fixture_type]
                                top_left_x = center_x - width / 2
                                top_left_y = center_y - height / 2
                                
                                state = FixtureState(x=top_left_x, y=top_left_y, angle=angle, type_id=fixture_type)
                                self.fixtures.append(state)
                                
                                # PERFORMANCE: Update fixture cache for faster validity checks in next step
                                fixture_idx = len(self.fixtures) - 1
                                fx_center = top_left_x + width / 2
                                fy_center = top_left_y + height / 2
                                self.fixture_cache[fixture_idx] = {
                                    'center': (fx_center, fy_center),
                                    'bounds': (top_left_x, top_left_x + width, top_left_y, top_left_y + height),
                                    'type_id': fixture_type
                                }
                                
                                # Compute valid positions AFTER placement
                                valid_pairs_after, _, _ = self._compute_valid_action_positions()
                                num_valid_after = len(valid_pairs_after)
                                
                                self.cumulative_moment = self._compute_system_moment_of_inertia(self.fixtures)
                                reward = self._calculate_reward_for_placement(
                                    remaining_valid_positions=num_valid_after, 
                                    total_valid_positions=num_valid_before,
                                    placement_moi=placement_moi,
                                    valid_objectives=valid_objectives
                                )
                                
                                self.fixture_availability[fixture_type] -= 1
                                
                                info['placement_success'] = True
                                info['reason'] = 'placement_success'
                                info['fixture_center'] = (center_x, center_y)
                                info['valid_positions_before'] = num_valid_before
                                info['valid_positions_after'] = num_valid_after
                                info['placement_moi'] = placement_moi
                            else:
                                # Should not happen since we selected from valid positions
                                reward = -1.0
                                info['reason'] = reason
        else:
            # Invalid fixture type (should not happen given fixture_type validation)
            reward = -1.0
            info['reason'] = 'invalid_fixture_type'
        
        self.step_count += 1
        
        # Update info with current state
        info['fixtures_count'] = len(self.fixtures)
        info['fixtures_available'] = dict(self.fixture_availability)
        info['sum_moments'] = self.cumulative_moment
        
        obs = self._get_observation()
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, done, truncated, info
    
    def _generate_valid_points(self) -> List[Tuple[float, float]]:
        """
        Generate valid grid points that lie inside the workpiece polygon.
        
        Returns:
            List of (x, y) tuples for valid placement points
        """
        valid_points = []
        
        # Create a grid of points
        x = self.visualizer.min_x
        while x <= self.visualizer.max_x:
            y = self.visualizer.min_y
            while y <= self.visualizer.max_y:
                # Check if point is inside workpiece polygon
                point = ShapelyPoint(x, y)
                if self.workpiece_polygon.contains(point):
                    # Check if not in exclusion zone
                    in_exclusion = False
                    for zone in self.exclusion_zones:
                        if zone.contains(point):
                            in_exclusion = True
                            break
                    
                    if not in_exclusion:
                        valid_points.append((x, y))
                
                y += self.grid_resolution
            x += self.grid_resolution
        
        return valid_points
    
    def _is_valid_placement(self, center_x: float, center_y: float, fixture_type: int, angle: float) -> Tuple[bool, str]:
        """
        Check if a fixture can be placed at the given center position.
        
        Returns:
            (is_valid, reason)
        """
        if len(self.fixtures) >= self.max_fixtures:
            return False, 'max_fixtures_reached'
        
        # Get fixture dimensions
        width, height = FIXTURE_DIMENSIONS.get(fixture_type, (0, 0))
        if width == 0 or height == 0:
            return False, 'invalid_fixture_type'
        
        # Calculate bounding box around center
        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2
        fixture_box = box(top_left_x, top_left_y, top_left_x + width, top_left_y + height)
        
        # Check if entire fixture is within workpiece
        if not fixture_box.within(self.workpiece_polygon):
            return False, 'outside_workpiece'
        
        # Check if overlaps with exclusion zones
        for zone in self.exclusion_zones:
            if fixture_box.intersects(zone):
                return False, 'overlaps_hole'
        
        # Check spacing constraints with existing fixtures
        for existing_fixture in self.fixtures:
            existing_width, existing_height = FIXTURE_DIMENSIONS[existing_fixture.type_id]
            existing_center_x = existing_fixture.x + existing_width / 2
            existing_center_y = existing_fixture.y + existing_height / 2
            
            dx = abs(center_x - existing_center_x)
            dy = abs(center_y - existing_center_y)
            
            if self._violates_spacing(dx, dy, existing_fixture.type_id, fixture_type):
                return False, 'violates_spacing_constraint'
        
        # Check for overlaps with other fixtures
        for existing_fixture in self.fixtures:
            existing_width, existing_height = FIXTURE_DIMENSIONS[existing_fixture.type_id]
            existing_box = box(
                existing_fixture.x,
                existing_fixture.y,
                existing_fixture.x + existing_width,
                existing_fixture.y + existing_height
            )
            if fixture_box.intersects(existing_box):
                return False, 'overlaps_fixture'
        
        return True, 'valid'
    
    
    def _violates_spacing(self, dx: float, dy: float, existing_fixture_type: int, new_fixture_type: int) -> bool:
        """
        Check if spacing between two fixtures violates constraints.
        
        Based on MiniZinc constraints:
        - Vertical (same column): (height1/2) + (height2/2) + VERTICAL_SECURITY_DISTANCE (100mm)
        - Horizontal (different columns): 145 + 200 = 345mm (constant, based on max fixture width)
        
        Args:
            dx: Horizontal distance between centers
            dy: Vertical distance between centers
            existing_fixture_type: Type ID of the existing fixture
            new_fixture_type: Type ID of the fixture being placed
            
        Returns:
            True if spacing is violated
        """
        same_column = dx < CENTER_ALIGNMENT_TOLERANCE
        
        if same_column:
            # In same column, check vertical spacing
            # Required spacing: (height1/2) + (height2/2) + VERTICAL_SECURITY_DISTANCE
            _, existing_height = FIXTURE_DIMENSIONS[existing_fixture_type]
            _, new_height = FIXTURE_DIMENSIONS[new_fixture_type]
            
            required_spacing = (existing_height / 2) + (new_height / 2) + VERTICAL_SECUIRITY_DISTANCE
            return dy < required_spacing
        else:
            # Different columns, check horizontal spacing
            # MiniZinc uses: xc(i) + WBar + WMin <= xc(j), where WBar=145, WMin=200, so spacing = 345
            required_spacing = 145 + HORIZONTAL_SECUIRITY_DISTANCE
            return dx < required_spacing
    
    def _compute_system_moment_of_inertia(self, fixtures_to_compute: List[FixtureState]) -> float:
        if not fixtures_to_compute:
            return 0.0
        
        computed_fixtures = define_fixture_from_state(fixtures_to_compute)
        if not computed_fixtures:
            return 0.0
        
        x_g, y_g = InertiaAnalysis.compute_overall_center_of_gravity(computed_fixtures)
        i, j = InertiaAnalysis.compute_combined_baricentric_moments_of_inertia(computed_fixtures, x_g, y_g)
        
        return abs(i) + abs(j)
    
    def _calculate_reward_for_placement(
        self, 
        remaining_valid_positions: int = 0, 
        total_valid_positions: int = 1,
        placement_moi: Optional[float] = None,
        valid_objectives: Optional[Dict] = None
    ) -> float:
        """
        Calculate reward for placing a fixture.
        
        Prioritizes placements that maximize the objective function (MOI).
        Encourages the agent to choose valid actions with higher MOI values.
        
        Args:
            remaining_valid_positions: Number of valid positions still available after placement
            total_valid_positions: Number of valid positions before placement
            placement_moi: MOI value if we place a fixture at the chosen position
            valid_objectives: Dictionary of all valid action objectives {(x_idx, y_idx, fixture_type): moi_value}
        """
        # Base placement bonus - encourages placing fixtures
        placement_bonus = 25.0
        
        # MOI-based reward: prioritize high objective values
        # If we have objective information, reward based on how good this placement is
        objective_reward = 0.0
        if placement_moi is not None and valid_objectives is not None and len(valid_objectives) > 0:
            # Find max MOI among all valid actions
            max_moi = max(valid_objectives.values())
            min_moi = min(valid_objectives.values())
            
            if max_moi > min_moi:
                # Normalize placement MOI to [0, 1] range
                moi_ratio = (placement_moi - min_moi) / (max_moi - min_moi)
            else:
                moi_ratio = 0.5
            
            # Reward for choosing high-MOI position
            # Range: [0, +50] - heavily weighted to encourage max-MOI selection
            # moi_ratio = 0.0 (worst) -> reward = 0
            # moi_ratio = 1.0 (best) -> reward = 50
            objective_reward = moi_ratio * 50.0
        
        # Fallback MOI reward if no objective info available
        if objective_reward == 0.0:
            if len(self.fixtures) == 1:
                moi_reward = self.cumulative_moment / 1e6
            else:
                old_moment = self._compute_system_moment_of_inertia(self.fixtures[:-1])
                new_moment = self.cumulative_moment
                moi_reward = (new_moment - old_moment) / 1e5
            objective_reward = max(0, moi_reward)
        
        # Flexibility bonus: secondary preference to maintain options
        if total_valid_positions > 0:
            flexibility_ratio = remaining_valid_positions / total_valid_positions
            flexibility_bonus = (flexibility_ratio - 0.5) * 0.5
        else:
            flexibility_bonus = 0.0
        
        # Total reward structure (descending priority):
        # 1. Placement bonus (25.0) - ensures fixtures are placed
        # 2. Objective reward (0 to 50.0) - DOMINANT for action selection
        # 3. Flexibility bonus (±0.25) - tie-breaker
        return placement_bonus + objective_reward + flexibility_bonus
    
    def _get_observation(self) -> np.ndarray:
        """Get observation as normalized fixture positions and moment of inertia."""
        obs = np.zeros(2 + (self.max_fixtures * 5) + 1, dtype=np.float32)
        idx = 0
        
        # 1. Fixture availability (normalized)
        obs[idx] = self.fixture_availability[1] / FIXTURE_AVAILABILITY[1]
        idx += 1
        obs[idx] = self.fixture_availability[2] / FIXTURE_AVAILABILITY[2]
        idx += 1
        
        # 2. Current placed fixtures
        for i, fixture in enumerate(self.fixtures):
            if i >= self.max_fixtures:
                break
            
            # Get center position
            width, height = FIXTURE_DIMENSIONS[fixture.type_id]
            center_x = fixture.x + width / 2
            center_y = fixture.y + height / 2
            
            # Normalize coordinates
            norm_x = (center_x - self.visualizer.min_x) / max(1.0, (self.visualizer.max_x - self.visualizer.min_x))
            norm_y = (center_y - self.visualizer.min_y) / max(1.0, (self.visualizer.max_y - self.visualizer.min_y))
            norm_angle = fixture.angle / 360.0
            norm_type = fixture.type_id / 2.0
            
            obs[idx] = np.clip(norm_x, 0, 1)
            idx += 1
            obs[idx] = np.clip(norm_y, 0, 1)
            idx += 1
            obs[idx] = norm_angle
            idx += 1
            obs[idx] = norm_type
            idx += 1
            obs[idx] = 1.0  # Fixture is active
            idx += 1
        
        # 3. Sum of moments of inertia (normalized)
        # Normalize to a reasonable range (assuming max is around 1e10)
        obs[idx] = min(1.0, self.cumulative_moment / 1e10)
        
        return obs
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render to display with persistent figure."""
        # Create figure on first render
        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=self.visualizer.figure_size)
            self.is_displaying = True
        
        # Clear and redraw
        self.ax.clear()
        
        # Convert fixtures to full fixture format for visualization
        display_fixtures = []
        if self.fixtures:
            computed_fixtures = define_fixture_from_state(self.fixtures)
            display_fixtures = computed_fixtures
        
        # Transform workpiece vertices to mathematical coordinate system
        workpiece_height = self.visualizer.max_y - self.visualizer.min_y
        transformed_vertices = []
        for vx, vy in self.workpiece_data['vertices']:
            x_transformed = vx - self.visualizer.min_x
            y_transformed = workpiece_height - (vy - self.visualizer.min_y)
            transformed_vertices.append((x_transformed, y_transformed))
        
        # Draw workpiece boundary
        workpiece_poly = patches.Polygon(
            transformed_vertices,
            closed=True,
            edgecolor='black',
            facecolor='lightgray',
            linewidth=2,
            alpha=0.3
        )
        self.ax.add_patch(workpiece_poly)
        
        # Draw holes (excluded regions) - transform to mathematical space
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if len(hole) == 3:  # Circle: [cx, cy, radius]
                    x_hole_transformed = hole[0] - self.visualizer.min_x
                    y_hole_transformed = workpiece_height - (hole[1] - self.visualizer.min_y)
                    circle = patches.Circle(
                        (x_hole_transformed, y_hole_transformed),
                        hole[2],
                        edgecolor='red',
                        facecolor='red',
                        alpha=0.2,
                        linewidth=1.5,
                        linestyle='--'
                    )
                    self.ax.add_patch(circle)
        
        # Draw valid action positions with color-coded MOI values (no text labels)
        if len(self.fixtures) < self.max_fixtures and (self.fixture_availability[1] > 0 or self.fixture_availability[2] > 0):
            valid_pairs, _, _ = self._compute_valid_action_positions()
            valid_objectives = self._compute_valid_action_objectives()
            
            # Find max MOI for color scaling
            if valid_objectives:
                max_moi = max(valid_objectives.values())
                min_moi = min(valid_objectives.values())
                moi_range = max_moi - min_moi if max_moi > min_moi else 1.0
            else:
                max_moi = min_moi = moi_range = 1.0
            
            # Draw valid positions with color-coded MOI values (red=low, green=high)
            for x_idx, y_idx in valid_pairs:
                x_pos = self.x_positions[x_idx]
                y_pos = self.y_positions[y_idx]
                
                # Find best MOI at this position
                best_moi = None
                for fixture_type in [1, 2]:
                    key = (x_idx, y_idx, fixture_type)
                    if key in valid_objectives:
                        if best_moi is None or valid_objectives[key] > best_moi:
                            best_moi = valid_objectives[key]
                
                # Color code by MOI value: red (low) to green (high)
                if best_moi is not None and moi_range > 0:
                    # Normalize MOI to [0, 1] for color mapping
                    norm_moi = (best_moi - min_moi) / moi_range
                    # Red (low) to Yellow (med) to Green (high)
                    if norm_moi < 0.5:
                        # Red to Yellow
                        r, g, b = 1.0, 2 * norm_moi, 0.0
                    else:
                        # Yellow to Green
                        r, g, b = 2 * (1 - norm_moi), 1.0, 0.0
                else:
                    # Default green if no objective
                    r, g, b = 0.0, 1.0, 0.0
                
                # Draw colored marker for this valid position
                valid_marker = patches.Circle(
                    (x_pos, y_pos),
                    4,  # radius
                    edgecolor=(r, g, b),
                    facecolor=(r, g, b),
                    alpha=0.7,
                    linewidth=0.5,
                    zorder=5
                )
                self.ax.add_patch(valid_marker)
        
        # Draw fixtures
        for idx, fixture in enumerate(display_fixtures):
            self._draw_fixture_on_ax(self.ax, fixture, idx)
        
        # Set axis properties
        x_min = 0 - self.visualizer.padding
        x_max = (self.visualizer.max_x - self.visualizer.min_x) + self.visualizer.padding
        y_min = 0 - self.visualizer.padding
        y_max = workpiece_height + self.visualizer.padding
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
        
        # Title with moment of inertia information
        title = (f"{self.workpiece_name} - Step {self.step_count}/{self.max_steps} - "
                f"Fixtures: {len(self.fixtures)}/{self.max_fixtures} - "
                f"Σ MOI: {self.cumulative_moment:.2e}")
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.grid(True, alpha=0.3)
        
        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _draw_spacing_guides(self):
        """Draw spacing constraint visualization."""
        # Optional: Draw spacing guides for placed fixtures
        for fixture in self.fixtures:
            width, height = FIXTURE_DIMENSIONS[fixture.type_id]
            center_x = fixture.x + width / 2
            center_y = fixture.y + height / 2
            
            # Draw horizontal spacing circle
            circle_h = patches.Circle(
                (center_x, center_y),
                HORIZONTAL_SECUIRITY_DISTANCE / 2,
                edgecolor='cyan',
                facecolor='none',
                linewidth=1,
                linestyle=':',
                alpha=0.5
            )
            self.ax.add_patch(circle_h)
            
            # Draw vertical spacing circle
            circle_v = patches.Circle(
                (center_x, center_y),
                VERTICAL_SECUIRITY_DISTANCE / 2,
                edgecolor='magenta',
                facecolor='none',
                linewidth=1,
                linestyle=':',
                alpha=0.5
            )
            self.ax.add_patch(circle_v)
    
    def _draw_fixture_on_ax(self, ax, fixture: Dict, fixture_id: int):
        """Draw a single fixture on the given axis."""
        state = fixture.get('state')
        if not state:
            return
        
        # Get fixture dimensions
        t = state.type_id
        width, height = FIXTURE_DIMENSIONS.get(t, (0, 0))
        
        # Calculate corners
        corners = [
            (state.x, state.y),
            (state.x + width, state.y),
            (state.x + width, state.y + height),
            (state.x, state.y + height),
        ]
        
        # Rotate corners around center
        center_x = state.x + width / 2
        center_y = state.y + height / 2
        angle_rad = math.radians(state.angle)
        
        rotated_corners = []
        for cx, cy in corners:
            dx = cx - center_x
            dy = cy - center_y
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + center_x
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + center_y
            rotated_corners.append((rx, ry))
        
        # Draw fixture
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        color = colors[fixture_id % len(colors)]
        
        fixture_poly = patches.Polygon(
            rotated_corners,
            closed=True,
            edgecolor=color,
            facecolor=color,
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(fixture_poly)
        
        # Draw center point
        ax.plot(center_x, center_y, 'k+', markersize=8, markeredgewidth=2)
        
        # Add fixture ID label
        ax.text(center_x, center_y, str(fixture_id), 
                fontsize=8, ha='center', va='center', color='white', fontweight='bold')
        
        # Add center coordinates label
        coord_text = f"({center_x:.1f}, {center_y:.1f})"
        ax.text(center_x, center_y + 15, coord_text, 
                fontsize=7, ha='center', va='bottom', color='black', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to numpy array."""
        display_fixtures = []
        if self.fixtures:
            computed_fixtures = define_fixture_from_state(self.fixtures)
            display_fixtures = computed_fixtures
        
        fig = self.visualizer.get_figure(
            fixtures=display_fixtures,
            title=f"{self.workpiece_name} - Step {self.step_count}/{self.max_steps} - Σ MOI: {self.cumulative_moment:.2e}"
        )
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        rgb_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[1], size[0], 3)
        plt.close(fig)
        
        return rgb_array
    
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.is_displaying = False



def fixtures_to_solution_json(fixtures: List[FixtureState], objective_value: float, visualizer: 'WorkpieceVisualizer') -> dict:
    """
    Convert environment fixtures to solution JSON format.
    
    Args:
        fixtures: List of FixtureState objects representing placed fixtures
        objective_value: Sum of moments of inertia (objective value)
        visualizer: WorkpieceVisualizer instance to get workpiece bounds
        
    Returns:
        Dictionary in the requested JSON format with coordinates transformed back to original space
        (same coordinate system as workpieces_information.json)
        
    Note: Coordinates follow the corner convention:
        (x, y) = lower-left corner
        (x1, y1) = lower-right corner
        (x2, y2) = upper-right corner
        (x3, y3) = upper-left corner
    """
    solution = {
        "x": [],      # lower left x
        "y": [],      # lower left y
        "x1": [],     # lower right x
        "y1": [],     # lower right y
        "x2": [],     # upper right x
        "y2": [],     # upper right y
        "x3": [],     # upper left x
        "y3": [],     # upper left y
        "angle": [],
        "fixtures_center_x": [],
        "fixtures_center_y": [],
        "bars_center": [],
        "selected_fixture": [],
        "fixture_type": [],
        "objective_value": int(objective_value)
    }
    
    # Calculate workpiece height to transform from mathematical space back to original space
    workpiece_height = visualizer.max_y - visualizer.min_y
    
    for fixture_idx, fixture in enumerate(fixtures):
        width, height = FIXTURE_DIMENSIONS[fixture.type_id]
        
        # Fixtures are stored in mathematical space (0,0 at lower-left, y increases upward)
        # Transform back to original space (same as workpieces_information.json)
        # Transformation: y_original = workpiece_height - y_math
        
        x_math = fixture.x
        y_math = fixture.y
        
        # Compute all four corners in mathematical space
        ll_x_math = x_math              # lower-left
        ll_y_math = y_math
        
        lr_x_math = x_math + width      # lower-right
        lr_y_math = y_math
        
        ur_x_math = x_math + width      # upper-right
        ur_y_math = y_math + height
        
        ul_x_math = x_math              # upper-left
        ul_y_math = y_math + height
        
        # Transform to original space
        ll_x_orig = float(ll_x_math)
        ll_y_orig = float(workpiece_height - ll_y_math)
        
        lr_x_orig = float(lr_x_math)
        lr_y_orig = float(workpiece_height - lr_y_math)
        
        ur_x_orig = float(ur_x_math)
        ur_y_orig = float(workpiece_height - ur_y_math)
        
        ul_x_orig = float(ul_x_math)
        ul_y_orig = float(workpiece_height - ul_y_math)
        
        # Ensure correct corner order by finding min/max coordinates
        # In original space, "lower" means larger y (more downward)
        # "upper" means smaller y (more upward)
        all_xs = [ll_x_orig, lr_x_orig, ur_x_orig, ul_x_orig]
        all_ys = [ll_y_orig, lr_y_orig, ur_y_orig, ul_y_orig]
        
        min_x = min(all_xs)
        max_x = max(all_xs)
        min_y = min(all_ys)
        max_y = max(all_ys)
        
        # Assign corners in correct order
        x_ll = min_x
        y_ll = min_y
        
        x_lr = max_x
        y_lr = min_y
        
        x_ur = max_x
        y_ur = max_y
        
        x_ul = min_x
        y_ul = max_y
        
        # Calculate center in original space
        center_x_orig = float(x_math + width / 2)
        center_y_orig = float(workpiece_height - (y_math + height / 2))
        
        solution["x"].append(x_ll)
        solution["y"].append(y_ll)
        solution["x1"].append(x_lr)
        solution["y1"].append(y_lr)
        solution["x2"].append(x_ur)
        solution["y2"].append(y_ur)
        solution["x3"].append(x_ul)
        solution["y3"].append(y_ul)
        solution["angle"].append(int(fixture.angle))
        solution["fixtures_center_x"].append(center_x_orig)
        solution["fixtures_center_y"].append(center_y_orig)
        solution["bars_center"].append(int(center_x_orig))  # Same as fixtures_center_x, just as integer
        solution["selected_fixture"].append(1)
        solution["fixture_type"].append(int(fixture.type_id))
    
    return solution


def save_best_solution(solution: dict, workpiece: str):
    """
    Save the best solution to a JSON file.
    
    Args:
        solution: Solution dictionary from fixtures_to_solution_json()
        workpiece: Name of the workpiece
    """
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"rl_{workpiece}.json"
    with open(output_file, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"\n[SAVED] Best solution saved to: {output_file}")
    print(f"  Objective value: {solution['objective_value']}")
    print(f"  Number of fixtures: {len(solution['x'])}")


def main():
    """Main function to test the environment."""
    parser = argparse.ArgumentParser(description='Graphical RL Agent for Fixture Layout Optimization')
    parser.add_argument('--workpiece', type=str, default='simple_stair_step',
                       help='Workpiece name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', default=False,
                       help='Render the environment')
    
    args = parser.parse_args()
    
    # Create environment
    env = FixtureLayoutEnv(
        workpiece_name=args.workpiece,
        render_mode="human" if args.render else None
    )
    
    print(f"Environment initialized for '{args.workpiece}':")
    print(f"  Min fixtures: {env.min_fixtures}, Max fixtures: {env.max_fixtures}")
    print(f"  Valid placement points: {env.num_placement_points}")
    print(f"  Fixture availability: {env.fixture_availability}")
    print(f"  Type 1 Spacing: Horizontal={get_horizontal_spacing(1):.1f}mm, Vertical={get_vertical_spacing(1):.1f}mm")
    print(f"  Type 2 Spacing: Horizontal={get_horizontal_spacing(2):.1f}mm, Vertical={get_vertical_spacing(2):.1f}mm")
    print(f"  Fixture dimensions: Type1={FIXTURE_DIMENSIONS[1]}, Type2={FIXTURE_DIMENSIONS[2]}")
    
    # Track best solution
    best_solution = None
    best_objective = -float('inf')
    
    # Run episodes
    for episode in range(args.episodes):
        obs, info = env.reset()
        print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
        
        done = False
        truncated = False
        episode_reward = 0.0
        total_moment = 0.0
        
        while not (done or truncated):
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            total_moment = info.get('sum_moments', 0.0)
            
            if info.get('placement_success'):
                fixture_center = info.get('fixture_center', (0, 0))
                print(f"  Step {env.step_count}: Fixture {len(env.fixtures)} placed at center ({fixture_center[0]:.1f}, {fixture_center[1]:.1f}), "
                      f"Reward={reward:.4f}, sum_MOI={total_moment:.2e}")
            elif info.get('reason') not in ['no_action']:
                print(f"  Step {env.step_count}: Reward={reward:.2f}, Reason: {info.get('reason', 'unknown')}")
        
        print(f"  Episode Total Reward: {episode_reward:.2f}, Final sum_MOI: {total_moment:.2e}")
        
        # Track best solution
        if total_moment > best_objective and len(env.fixtures) > 0:
            best_objective = total_moment
            best_solution = (env.fixtures.copy(), total_moment)
    
    env.close()
    
    # Save best solution
    if best_solution is not None:
        fixtures, objective_value = best_solution
        print(f"\n[DEBUG] Best solution has {len(fixtures)} fixtures:")
        for i, fixture in enumerate(fixtures):
            print(f"  Fixture {i+1}: x={fixture.x}, y={fixture.y}, type={fixture.type_id}, angle={fixture.angle}")
        
        solution_json = fixtures_to_solution_json(fixtures, objective_value, env.visualizer)
        save_best_solution(solution_json, args.workpiece)
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
