"""Rotation RL Agent for Fixture Layout Optimization using Gymnasium.

A visualization-based RL environment where an agent learns to improve existing fixture layouts
by rotating fixtures to optimize the sum of moments of inertia.
The agent starts with a solution from the placement agent and optimizes it through rotations
in 15-degree steps.

The objective is to MAXIMIZE the sum of moments of inertia.

Rotation Mechanism:
===================
Each fixture can be rotated in 15° increments (0°, 15°, 30°, ..., 345°).
The agent can perform the following actions on each fixture:
- Rotate left by 15°
- No rotation (do nothing)
- Rotate right by 15°

Rotation centers:
- Square fixtures (Type 1): center of the fixture (cx, cy)
- Rectangular fixtures (Type 2): (cx - 17.5, cy - 40)

Constraints checked during rotation:
- Fixtures must remain entirely within the workpiece area
- No overlaps with other fixtures
- No overlaps with excluded zones (holes)

Visualization Features:
- Light green circles at valid rotation angles around each fixture center
- Red stars: rotation centers for each fixture
- Black + : fixture centers with current rotation angle shown

Action space: [fixture_idx, action_type]
- action_type: 0 = rotate left, 1 = no-op, 2 = rotate right

Usage:
    python python/rl_rotation_agent.py --workpiece simple_stair_step --load-from results/rl_simple_stair_step.json
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

# Rotation parameters
ROTATION_STEP = 5.0  # degrees
RECTANGULAR_ROTATION_CENTER_OFFSET_X = 0.0  # mm
RECTANGULAR_ROTATION_CENTER_OFFSET_Y = -40.0  # mm

def get_rotation_center(fixture_type: int, center_x: float, center_y: float) -> Tuple[float, float]:
    """
    Get the rotation center for a fixture based on its type.
    
    Args:
        fixture_type: Type ID of the fixture (1 for square, 2 for rectangular)
        center_x: X coordinate of the fixture center
        center_y: Y coordinate of the fixture center
        
    Returns:
        (rotation_center_x, rotation_center_y)
    """
    if fixture_type == 1:  # Square fixture
        return center_x, center_y
    elif fixture_type == 2:  # Rectangular fixture
        return (
            center_x + RECTANGULAR_ROTATION_CENTER_OFFSET_X,
            center_y + RECTANGULAR_ROTATION_CENTER_OFFSET_Y
        )
    else:
        return center_x, center_y  # Default to center


def rotate_point(x: float, y: float, center_x: float, center_y: float, angle_degrees: float) -> Tuple[float, float]:
    """
    Rotate a point around a center by the given angle.
    
    Args:
        x, y: Point coordinates
        center_x, center_y: Rotation center
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        
    Returns:
        (rotated_x, rotated_y)
    """
    angle_rad = math.radians(angle_degrees)
    dx = x - center_x
    dy = y - center_y
    rotated_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + center_x
    rotated_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + center_y
    return rotated_x, rotated_y


def get_rotated_fixture_bounds(
    fixture_type: int,
    center_x: float,
    center_y: float,
    angle: float
) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of a rotated fixture.
    
    Args:
        fixture_type: Type ID of the fixture
        center_x, center_y: Fixture center
        angle: Rotation angle in degrees
        
    Returns:
        (min_x, min_y, max_x, max_y) - bounding box
    """
    width, height = FIXTURE_DIMENSIONS[fixture_type]
    
    # Get rotation center
    rot_cx, rot_cy = get_rotation_center(fixture_type, center_x, center_y)
    
    # Get unrotated corners relative to fixture center
    half_width = width / 2
    half_height = height / 2
    
    corners = [
        (center_x - half_width, center_y - half_height),
        (center_x + half_width, center_y - half_height),
        (center_x + half_width, center_y + half_height),
        (center_x - half_width, center_y + half_height),
    ]
    
    # Rotate corners around rotation center
    rotated_corners = [rotate_point(cx, cy, rot_cx, rot_cy, angle) for cx, cy in corners]
    
    # Get bounding box
    xs = [x for x, y in rotated_corners]
    ys = [y for x, y in rotated_corners]
    
    return min(xs), min(ys), max(xs), max(ys)


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


class FixtureRotationEnv(gym.Env):
    """
    Gymnasium environment for fixture layout optimization through rotation.
    
    The agent optimizes an existing fixture layout by rotating fixtures in 15-degree steps.
    
    Objective: MAXIMIZE sum of moments of inertia.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        workpiece_name: str = "simple_stair_step",
        initial_solution_file: Optional[str] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the fixture rotation environment.
        
        Args:
            workpiece_name: Name of the workpiece to use
            initial_solution_file: Path to JSON file with initial solution
            render_mode: "human" for visualization, "rgb_array" for numpy array
        """
        super().__init__()
        
        self.workpiece_name = workpiece_name
        self.render_mode = render_mode
        
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
        transformed_vertices = []
        for vx, vy in self.workpiece_data['vertices']:
            x_transformed = vx - self.visualizer.min_x
            y_transformed = workpiece_height - (vy - self.visualizer.min_y)
            transformed_vertices.append((x_transformed, y_transformed))
        
        # Create workpiece polygon for collision detection
        self.workpiece_polygon = Polygon(transformed_vertices)
        
        # Create exclusion zones from holes
        self.exclusion_zones = []
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if len(hole) == 3:  # Circle
                    x_hole_transformed = hole[0] - self.visualizer.min_x
                    y_hole_transformed = workpiece_height - (hole[1] - self.visualizer.min_y)
                    zone = ShapelyPoint(x_hole_transformed, y_hole_transformed).buffer(hole[2])
                    self.exclusion_zones.append(zone)
        
        # Load initial solution
        self.initial_fixtures = self._load_initial_solution(initial_solution_file, workpiece_height)
        self.fixtures = [FixtureState(f.x, f.y, f.angle, f.type_id) for f in self.initial_fixtures]
        
        # Store workpiece height for later use
        self.workpiece_height = workpiece_height
        
        # Action space: [fixture_idx, action_type]
        # action_type: 0 = rotate left (-15°), 1 = no action, 2 = rotate right (+15°)
        self.action_space = spaces.MultiDiscrete([
            len(self.fixtures) if self.fixtures else 1,  # fixture index
            3  # action type: 3 possible actions (rotate left, no-op, rotate right)
        ])
        
        # Observation space
        # - Sum of moments of inertia (1 value, normalized)
        # - Current fixture angles (max fixtures, normalized)
        obs_size = 1 + len(self.fixtures)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Track state
        self.step_count = 0
        self.max_steps = 50
        self.cumulative_moment = self._compute_system_moment_of_inertia()
        self.best_moment = self.cumulative_moment
        
        # Visualization
        self.fig = None
        self.ax = None
        self.is_displaying = False
    
    def _load_initial_solution(
        self,
        solution_file: Optional[str],
        workpiece_height: float
    ) -> List[FixtureState]:
        """
        Load initial fixture configuration from JSON file.
        
        Args:
            solution_file: Path to solution JSON file
            workpiece_height: Height of workpiece for coordinate transformation
            
        Returns:
            List of FixtureState objects
        """
        if solution_file is None:
            # Try to find solution in results directory
            solution_file = ROOT / "results" / f"rl_{self.workpiece_name}.json"
        
        solution_file = Path(solution_file)
        
        if not solution_file.exists():
            raise FileNotFoundError(
                f"Solution file not found: {solution_file}\n"
                f"Please run rl_graphical_agent.py first to generate a solution."
            )
        
        with open(solution_file, 'r') as f:
            solution_json = json.load(f)
        
        fixtures = []
        num_fixtures = len(solution_json.get('fixtures_center_x', []))
        
        if num_fixtures == 0:
            raise ValueError("No fixtures found in solution file")
        
        for i in range(num_fixtures):
            fixture_type = solution_json['fixture_type'][i]
            angle = solution_json['angle'][i]
            
            # Get center coordinates from solution
            center_x = solution_json['fixtures_center_x'][i]
            center_y = solution_json['fixtures_center_y'][i]
            
            # Transform from original space to mathematical space
            center_y_math = workpiece_height - center_y
            
            # Calculate top-left corner
            width, height = FIXTURE_DIMENSIONS[fixture_type]
            x = center_x - width / 2
            y = center_y_math - height / 2
            
            fixtures.append(FixtureState(
                x=x,
                y=y,
                angle=float(angle),
                type_id=int(fixture_type)
            ))
        
        print(f"[LOAD] Loaded {num_fixtures} fixtures from {solution_file}")
        return fixtures
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.fixtures = [FixtureState(f.x, f.y, f.angle, f.type_id) for f in self.initial_fixtures]
        self.step_count = 0
        self.cumulative_moment = self._compute_system_moment_of_inertia()
        self.best_moment = self.cumulative_moment
        
        obs = self._get_observation()
        info = {'sum_moments': self.cumulative_moment}
        
        if self.render_mode == "human":
            self.render()
        
        return obs, info
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: [fixture_idx, action_type]
                   action_type: 0 = rotate left (-15°), 1 = no action, 2 = rotate right (+15°)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        fixture_idx, action_type = action
        
        done = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0
        info = {
            'action_success': False,
            'reason': 'no_action',
            'sum_moments': self.cumulative_moment,
        }
        
        # Clamp fixture index
        fixture_idx = fixture_idx % len(self.fixtures) if self.fixtures else 0
        fixture = self.fixtures[fixture_idx]
        
        # Store old moment
        old_moment = self.cumulative_moment
        
        # Determine action type (0-2): rotate left, no-op, rotate right
        action_type = action_type % 3
        
        if action_type == 0:  # Rotate -15° (rotate left)
            new_angle = fixture.angle - ROTATION_STEP
            if self._is_valid_rotation(fixture_idx, new_angle):
                fixture.angle = new_angle % 360.0
                self.cumulative_moment = self._compute_system_moment_of_inertia()
                moment_delta = self.cumulative_moment - old_moment
                reward = moment_delta / 1e7
                info['action_success'] = True
                info['reason'] = 'rotate_left_success'
                info['new_angle'] = float(fixture.angle)
            else:
                reward = -0.5
                info['reason'] = 'rotate_left_invalid'
                
        elif action_type == 1:  # No action
            reward = 0.0
            info['reason'] = 'no_action'
            
        elif action_type == 2:  # Rotate +15° (rotate right)
            new_angle = fixture.angle + ROTATION_STEP
            if self._is_valid_rotation(fixture_idx, new_angle):
                fixture.angle = new_angle % 360.0
                self.cumulative_moment = self._compute_system_moment_of_inertia()
                moment_delta = self.cumulative_moment - old_moment
                reward = moment_delta / 1e7
                info['action_success'] = True
                info['reason'] = 'rotate_right_success'
                info['new_angle'] = float(fixture.angle)
            else:
                reward = -0.5
                info['reason'] = 'rotate_right_invalid'
        
        # Track best solution
        if self.cumulative_moment > self.best_moment:
            self.best_moment = self.cumulative_moment
            info['is_best'] = True
        
        self.step_count += 1
        info['sum_moments'] = self.cumulative_moment
        info['best_moments'] = self.best_moment
        
        obs = self._get_observation()
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, done, truncated, info
    
    def _is_valid_rotation(self, fixture_idx: int, new_angle: float) -> bool:
        """
        Check if a fixture can be rotated to the given angle.
        
        Args:
            fixture_idx: Index of fixture to rotate
            new_angle: Target rotation angle in degrees
            
        Returns:
            True if rotation is valid
        """
        fixture = self.fixtures[fixture_idx]
        width, height = FIXTURE_DIMENSIONS[fixture.type_id]
        
        # Get fixture center
        center_x = fixture.x + width / 2
        center_y = fixture.y + height / 2
        
        # Get bounding box of rotated fixture
        min_x, min_y, max_x, max_y = get_rotated_fixture_bounds(
            fixture.type_id,
            center_x,
            center_y,
            new_angle
        )
        
        rotated_box = box(min_x, min_y, max_x, max_y)
        
        # Check if entire rotated fixture is within workpiece
        if not rotated_box.within(self.workpiece_polygon):
            return False
        
        # Check if overlaps with exclusion zones
        for zone in self.exclusion_zones:
            if rotated_box.intersects(zone):
                return False
        
        # Check for overlaps with other fixtures
        for idx, other_fixture in enumerate(self.fixtures):
            if idx == fixture_idx:
                continue
            
            other_width, other_height = FIXTURE_DIMENSIONS[other_fixture.type_id]
            other_center_x = other_fixture.x + other_width / 2
            other_center_y = other_fixture.y + other_height / 2
            
            other_min_x, other_min_y, other_max_x, other_max_y = get_rotated_fixture_bounds(
                other_fixture.type_id,
                other_center_x,
                other_center_y,
                other_fixture.angle
            )
            
            other_box = box(other_min_x, other_min_y, other_max_x, other_max_y)
            
            if rotated_box.intersects(other_box):
                return False
        
        return True
    
    def _compute_system_moment_of_inertia(self) -> float:
        """Compute the sum of moments of inertia for all fixtures."""
        if not self.fixtures:
            return 0.0
        
        computed_fixtures = define_fixture_from_state(self.fixtures)
        if not computed_fixtures:
            return 0.0
        
        x_g, y_g = InertiaAnalysis.compute_overall_center_of_gravity(computed_fixtures)
        i, j = InertiaAnalysis.compute_combined_baricentric_moments_of_inertia(computed_fixtures, x_g, y_g)
        
        return abs(i) + abs(j)
    
    def _get_observation(self) -> np.ndarray:
        """Get observation as normalized fixture angles and moment of inertia."""
        obs = np.zeros(1 + len(self.fixtures), dtype=np.float32)
        
        # Sum of moments of inertia (normalized)
        obs[0] = min(1.0, self.cumulative_moment / 1e10)
        
        # Current fixture angles (normalized)
        for i, fixture in enumerate(self.fixtures):
            obs[i + 1] = fixture.angle / 360.0
        
        return obs
    

    

    

    
    def get_valid_rotation_options(self, fixture_idx: int) -> List[float]:
        """
        Get list of valid rotation angles for a fixture.
        
        Args:
            fixture_idx: Index of fixture
            
        Returns:
            List of valid rotation angles (in degrees)
        """
        fixture = self.fixtures[fixture_idx]
        valid_angles = []
        
        # Try all 15-degree rotations from 0 to 360
        for angle in np.arange(0, 360, ROTATION_STEP):
            if self._is_valid_rotation(fixture_idx, angle):
                valid_angles.append(angle)
        
        return valid_angles
    
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
        
        # Draw holes (excluded regions)
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if len(hole) == 3:  # Circle
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
        
        # Draw possible rotation angles for all fixtures
        if self.fixtures and len(self.fixtures) > 0:
            for fixture_idx in range(len(self.fixtures)):
                fixture = self.fixtures[fixture_idx]
                width, height = FIXTURE_DIMENSIONS[fixture.type_id]
                center_x = fixture.x + width / 2
                center_y = fixture.y + height / 2
                
                # Get rotation center for this fixture type
                rot_cx, rot_cy = get_rotation_center(fixture.type_id, center_x, center_y)
                
                # Check all possible rotation angles (0°, 15°, 30°, ..., 345°)
                radius = 40  # radius for visualization circles showing possible angles
                for angle_deg in range(0, 360, 15):
                    # Check if this rotation angle is valid
                    is_valid = self._is_valid_rotation(fixture_idx, angle_deg)
                    
                    if is_valid:
                        # Calculate position on circle for this angle
                        angle_rad = math.radians(angle_deg)
                        marker_x = rot_cx + radius * math.cos(angle_rad)
                        marker_y = rot_cy + radius * math.sin(angle_rad)
                        
                        # Current rotation angle gets highlighted
                        if abs(angle_deg - fixture.angle) < 1.0 or abs((angle_deg - fixture.angle) % 360) < 1.0:
                            # Current angle: bright green
                            color = 'green'
                            alpha = 1.0
                            size = 6
                            linewidth = 2
                        else:
                            # Valid alternative angle: light green
                            color = 'green'
                            alpha = 0.5
                            size = 4
                            linewidth = 0.5
                        
                        # Draw marker for this rotation angle
                        angle_marker = patches.Circle(
                            (marker_x, marker_y),
                            size,
                            edgecolor=color,
                            facecolor=color,
                            alpha=alpha,
                            linewidth=linewidth
                        )
                        self.ax.add_patch(angle_marker)
                        
                        # Add angle label for current rotation
                        if abs(angle_deg - fixture.angle) < 1.0 or abs((angle_deg - fixture.angle) % 360) < 1.0:
                            self.ax.text(marker_x, marker_y - 8, f"{int(angle_deg)}°", 
                                       fontsize=7, ha='center', va='top', color='green', fontweight='bold')
        
        # Draw fixtures
        display_fixtures = define_fixture_from_state(self.fixtures) if self.fixtures else []
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
        title = (f"{self.workpiece_name} (Rotation) - Step {self.step_count}/{self.max_steps} - "
                f"Σ MOI: {self.cumulative_moment:.2e} | Best: {self.best_moment:.2e}")
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.grid(True, alpha=0.3)
        
        # Add legend for rotation indicators
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Current rotation angle'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=4, alpha=0.5, label='Valid alternative angles'),
            Line2D([0], [0], marker='*', color='r', markersize=15, label='Rotation center', linestyle='None'),
            Line2D([0], [0], marker='+', color='k', markersize=8, linewidth=2, label='Fixture center', linestyle='None'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _draw_fixture_on_ax(self, ax, fixture: Dict, fixture_id: int):
        """Draw a single fixture on the given axis."""
        state = fixture.get('state')
        if not state:
            return
        
        # Get fixture dimensions
        t = state.type_id
        width, height = FIXTURE_DIMENSIONS.get(t, (0, 0))
        
        # Get rotation center
        center_x = state.x + width / 2
        center_y = state.y + height / 2
        rot_cx, rot_cy = get_rotation_center(t, center_x, center_y)
        
        # Calculate corners
        half_width = width / 2
        half_height = height / 2
        
        corners = [
            (center_x - half_width, center_y - half_height),
            (center_x + half_width, center_y - half_height),
            (center_x + half_width, center_y + half_height),
            (center_x - half_width, center_y + half_height),
        ]
        
        # Rotate corners around rotation center
        angle_rad = math.radians(state.angle)
        rotated_corners = []
        for cx, cy in corners:
            dx = cx - rot_cx
            dy = cy - rot_cy
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + rot_cx
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + rot_cy
            rotated_corners.append((rx, ry))
        
        # Draw fixture
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
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
        
        # Draw rotation center
        ax.plot(rot_cx, rot_cy, 'r*', markersize=12, markeredgewidth=1, label='Rotation center' if fixture_id == 0 else '')
        
        # Draw fixture center
        ax.plot(center_x, center_y, 'k+', markersize=8, markeredgewidth=2)
        
        # Add fixture ID and angle label
        ax.text(center_x, center_y, f"{fixture_id}\n{state.angle:.0f}°", 
                fontsize=8, ha='center', va='center', color='white', fontweight='bold')
        
        # Add center coordinates label
        coord_text = f"({center_x:.1f}, {center_y:.1f})"
        ax.text(center_x, center_y + 20, coord_text, 
                fontsize=7, ha='center', va='bottom', color='black', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render to numpy array."""
        fig, ax = plt.subplots(figsize=self.visualizer.figure_size)
        
        # Transform workpiece vertices to mathematical coordinate system
        workpiece_height = self.visualizer.max_y - self.visualizer.min_y
        transformed_vertices = []
        for vx, vy in self.workpiece_data['vertices']:
            x_transformed = vx - self.visualizer.min_x
            y_transformed = workpiece_height - (vy - self.visualizer.min_y)
            transformed_vertices.append((x_transformed, y_transformed))
        
        # Draw workpiece
        workpiece_poly = patches.Polygon(
            transformed_vertices,
            closed=True,
            edgecolor='black',
            facecolor='lightgray',
            linewidth=2,
            alpha=0.3
        )
        ax.add_patch(workpiece_poly)
        
        # Draw holes
        if 'holes' in self.workpiece_data:
            for hole in self.workpiece_data['holes']:
                if len(hole) == 3:
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
                    ax.add_patch(circle)
        
        # Draw fixtures
        display_fixtures = define_fixture_from_state(self.fixtures) if self.fixtures else []
        for idx, fixture in enumerate(display_fixtures):
            self._draw_fixture_on_ax(ax, fixture, idx)
        
        # Set axis properties
        x_min = 0 - self.visualizer.padding
        x_max = (self.visualizer.max_x - self.visualizer.min_x) + self.visualizer.padding
        y_min = 0 - self.visualizer.padding
        y_max = workpiece_height + self.visualizer.padding
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        
        title = (f"{self.workpiece_name} (Rotation) - Step {self.step_count}/{self.max_steps} - "
                f"Σ MOI: {self.cumulative_moment:.2e}")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        
        # Render to array
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
        Dictionary in the requested JSON format
    """
    solution = {
        "x": [],
        "y": [],
        "x1": [],
        "y1": [],
        "x2": [],
        "y2": [],
        "x3": [],
        "y3": [],
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
    
    for fixture in fixtures:
        width, height = FIXTURE_DIMENSIONS[fixture.type_id]
        
        # Transform back to original space
        x_math = fixture.x
        y_math = fixture.y
        
        x_ll_orig = float(x_math)
        y_ll_orig = float(workpiece_height - y_math)
        
        x_lr_orig = float(x_math + width)
        y_lr_orig = float(workpiece_height - y_math)
        
        x_ur_orig = float(x_math + width)
        y_ur_orig = float(workpiece_height - (y_math + height))
        
        x_ul_orig = float(x_math)
        y_ul_orig = float(workpiece_height - (y_math + height))
        
        center_x_orig = float(x_math + width / 2)
        center_y_orig = float(workpiece_height - (y_math + height / 2))
        
        solution["x"].append(x_ll_orig)
        solution["y"].append(y_ll_orig)
        solution["x1"].append(x_lr_orig)
        solution["y1"].append(y_lr_orig)
        solution["x2"].append(x_ur_orig)
        solution["y2"].append(y_ur_orig)
        solution["x3"].append(x_ul_orig)
        solution["y3"].append(y_ul_orig)
        solution["angle"].append(int(fixture.angle))
        solution["fixtures_center_x"].append(center_x_orig)
        solution["fixtures_center_y"].append(center_y_orig)
        solution["bars_center"].append(int(round(center_x_orig / 20) * 20))
        solution["selected_fixture"].append(1)
        solution["fixture_type"].append(int(fixture.type_id))
    
    return solution


def save_best_solution(solution: dict, workpiece: str, suffix: str = "rotation"):
    """
    Save the best solution to a JSON file.
    
    Args:
        solution: Solution dictionary
        workpiece: Name of the workpiece
        suffix: Suffix to add to filename
    """
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"rl_{workpiece}_{suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(solution, f, indent=2)
    
    print(f"\n[SAVED] Best solution saved to: {output_file}")
    print(f"  Objective value: {solution['objective_value']}")
    print(f"  Number of fixtures: {len(solution['x'])}")


def main():
    """Main function to test the environment."""
    parser = argparse.ArgumentParser(description='Rotation & Movement RL Agent for Fixture Layout Optimization')
    parser.add_argument('--workpiece', type=str, default='simple_stair_step',
                       help='Workpiece name')
    parser.add_argument('--load-from', type=str, default=None,
                       help='Path to JSON file with initial solution (default: results/rl_<workpiece>.json)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Render the environment')
    
    args = parser.parse_args()
    
    # Create environment
    try:
        env = FixtureRotationEnv(
            workpiece_name=args.workpiece,
            initial_solution_file=args.load_from,
            render_mode="human" if args.render else None
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease generate an initial solution first:")
        print(f"  python python/rl_graphical_agent.py --workpiece {args.workpiece} --render")
        return
    
    print(f"Environment initialized for '{args.workpiece}' (Rotation Optimization):")
    print(f"  Number of fixtures: {len(env.fixtures)}")
    print(f"  Rotation step: {ROTATION_STEP}°")
    print(f"  Initial sum of MOI: {env.cumulative_moment:.2e}")
    print(f"\n  Action space: 0=rotate left, 1=no-op, 2=rotate right")
    
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
        actions_applied = 0
        
        while not (done or truncated):
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            total_moment = info.get('sum_moments', 0.0)
            
            if info.get('action_success'):
                actions_applied += 1
                fixture_idx = action[0]
                reason = info.get('reason', 'unknown')
                
                # Format action description
                if 'move_down' in reason:
                    action_desc = "move down"
                elif 'move_up' in reason:
                    action_desc = "move up"
                elif 'rotate_left' in reason:
                    action_desc = f"rotate left {info.get('rotation_delta', 0):.0f}° to {info.get('new_angle', 0):.0f}°"
                elif 'rotate_right' in reason:
                    action_desc = f"rotate right {info.get('rotation_delta', 0):.0f}° to {info.get('new_angle', 0):.0f}°"
                else:
                    action_desc = reason
                
                print(f"  Step {env.step_count}: Fixture {fixture_idx} {action_desc}, "
                      f"Reward={reward:.4f}, sum_MOI={total_moment:.2e}")
                if info.get('is_best'):
                    print(f"    *** NEW BEST: {total_moment:.2e} ***")
        
        print(f"  Episode Total Reward: {episode_reward:.2f}, Final sum_MOI: {total_moment:.2e}, Actions: {actions_applied}")
        
        # Track best solution
        if total_moment > best_objective and len(env.fixtures) > 0:
            best_objective = total_moment
            best_solution = (env.fixtures.copy(), total_moment)
    
    env.close()
    
    # Save best solution
    if best_solution is not None:
        fixtures, objective_value = best_solution
        solution_json = fixtures_to_solution_json(fixtures, objective_value, env.visualizer)
        save_best_solution(solution_json, args.workpiece, suffix="rotation")
    
    print("\nRotation optimization completed!")


if __name__ == "__main__":
    main()
