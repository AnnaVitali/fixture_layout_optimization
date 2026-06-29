#!/usr/bin/env python3
"""Debug valid Y-positions after placing first fixture"""

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_fixture_on_bars_agent import FixtureOnBarsEnv, FixtureType, FixtureState, FIXTURE_DIMENSIONS

# Create environment
env = FixtureOnBarsEnv(
    bar_positions=[92.5, 602.5],
    workpiece_name='simple_stair_step',
    verbose=False
)

print("=" * 80)
print("INITIAL STATE")
print("=" * 80)
print(f"Valid Y-positions (base): {[f'{y:.1f}' for y in env.valid_y_positions_base]}")

# Simulate placing first fixture at y=32.5 on Bar 1
print("\n" + "=" * 80)
print("PLACING FIRST FIXTURE AT Y=32.5")
print("=" * 80)
env.current_bar_idx = 0
env._update_valid_y_positions()
print(f"Before placement - Valid Y-positions on Bar 1: {[f'{y:.1f}' for y in env.valid_y_positions]}")

# Place a rectangular fixture at y=32.5 (NOT 33.5)
fixture_type = FixtureType.TYPE_2  # Rectangular
width, height = FIXTURE_DIMENSIONS[fixture_type]
y_center = 32.5  # Use grid-aligned position
placement_x = env.bar_positions[0] - width / 2
placement_y = y_center - height / 2

fixture1 = FixtureState(x=placement_x, y=placement_y, type_id=fixture_type)
env.fixtures.append(fixture1)
env.fixtures_per_bar[0] += 1

print(f"Placed fixture 1: type={fixture_type}, x={placement_x:.1f}, y={placement_y:.1f}, center_y={y_center:.1f}")

# Now update valid Y-positions with this fixture in place
env._update_valid_y_positions()
print(f"\nAfter placing fixture 1 - Valid Y-positions on Bar 1: {[f'{y:.1f}' for y in env.valid_y_positions]}")

# Check if y=242.5 is in the list (correct grid position)
if 242.5 in env.valid_y_positions:
    print(f"✓ Y=242.5 IS valid for second fixture!")
    
    # Place second fixture at 242.5
    y_center2 = 242.5
    placement_x2 = env.bar_positions[0] - width / 2
    placement_y2 = y_center2 - height / 2
    
    fixture2 = FixtureState(x=placement_x2, y=placement_y2, type_id=fixture_type)
    env.fixtures.append(fixture2)
    env.fixtures_per_bar[0] += 1
    
    print(f"✓ Placed fixture 2 at Y={y_center2:.1f}")
    
    # Verify spacing
    y1 = y_center
    y2 = y_center2
    distance = abs(y2 - y1)
    required = height/2 + height/2 + 100.0
    print(f"\n✓ SUCCESS: Both fixtures placed!")
    print(f"  Fixture 1: Y={y1:.1f}")
    print(f"  Fixture 2: Y={y2:.1f}")
    print(f"  Distance: {distance}mm (required: {required}mm) - {'✓ OK' if distance >= required else '✗ FAIL'}")
else:
    # Find closest valid position
    valid_above_200 = [y for y in env.valid_y_positions if y > 200]
    print(f"✗ Y=242.5 is NOT valid for second fixture")
    if valid_above_200:
        print(f"  Valid positions > 200mm: {[f'{y:.1f}' for y in valid_above_200]}")
    else:
        print(f"  No valid positions remaining above 200mm")
