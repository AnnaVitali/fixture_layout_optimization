#!/usr/bin/env python3
"""Test Y=33.5 and Y=246.5 placement with 1mm grid"""

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
print("TESTING Y=33.5 AND Y=246.5 ON 1MM GRID")
print("=" * 80)

# Check if these positions are in the valid base list
if 33.5 in env.valid_y_positions_base:
    print(f"✓ Y=33.5 is in valid_y_positions_base")
else:
    print(f"✗ Y=33.5 is NOT in valid_y_positions_base")

if 246.5 in env.valid_y_positions_base:
    print(f"✓ Y=246.5 is in valid_y_positions_base")
else:
    print(f"✗ Y=246.5 is NOT in valid_y_positions_base")

# Try placing both on Bar 1
print("\n" + "=" * 80)
print("PLACING ON BAR 1")
print("=" * 80)

env.current_bar_idx = 0
env._update_valid_y_positions()

fixture_type = FixtureType.TYPE_2
width, height = FIXTURE_DIMENSIONS[fixture_type]

# Place first fixture at Y=33.5
y1 = 33.5
fixture1 = FixtureState(
    x=env.bar_positions[0] - width / 2,
    y=y1 - height / 2,
    type_id=fixture_type
)
env.fixtures.append(fixture1)
env.fixtures_per_bar[0] += 1
print(f"✓ Placed fixture 1 at Y={y1}")

# Update valid positions
env._update_valid_y_positions()

# Check if Y=246.5 is still valid
if 246.5 in env.valid_y_positions:
    print(f"✓ Y=246.5 IS valid after placing fixture 1!")
    
    # Place second fixture
    y2 = 246.5
    fixture2 = FixtureState(
        x=env.bar_positions[0] - width / 2,
        y=y2 - height / 2,
        type_id=fixture_type
    )
    env.fixtures.append(fixture2)
    env.fixtures_per_bar[0] += 1
    print(f"✓ Placed fixture 2 at Y={y2}")
    
    distance = abs(y2 - y1)
    required = height/2 + height/2 + 100.0
    print(f"\n✓ SUCCESS: 2 rectangular fixtures on Bar 1!")
    print(f"  Y₁={y1}, Y₂={y2}")
    print(f"  Distance: {distance}mm (need {required}mm) - {'✓ OK' if distance >= required else '✗ FAIL'}")
else:
    print(f"✗ Y=246.5 is NOT valid after placing fixture 1")

# Now try on Bar 2
print("\n" + "=" * 80)
print("PLACING ON BAR 2")
print("=" * 80)

env.current_bar_idx = 1
env._update_valid_y_positions()

# Place fixture on bar 2 at Y=33.5
fixture3 = FixtureState(
    x=env.bar_positions[1] - width / 2,
    y=y1 - height / 2,
    type_id=fixture_type
)
env.fixtures.append(fixture3)
env.fixtures_per_bar[1] += 1
print(f"✓ Placed fixture 3 on Bar 2 at Y={y1}")

env._update_valid_y_positions()

if 246.5 in env.valid_y_positions:
    print(f"✓ Y=246.5 IS valid for Bar 2!")
    
    # Place on bar 2 at Y=246.5
    fixture4 = FixtureState(
        x=env.bar_positions[1] - width / 2,
        y=y2 - height / 2,
        type_id=fixture_type
    )
    env.fixtures.append(fixture4)
    env.fixtures_per_bar[1] += 1
    print(f"✓ Placed fixture 4 on Bar 2 at Y={y2}")
    
    print(f"\n✓✓✓ SUCCESS: 4 rectangular fixtures total (2 per bar)!")
    print(f"    Bar 1: Fixtures at Y={y1}, Y={y2}")
    print(f"    Bar 2: Fixtures at Y={y1}, Y={y2}")
else:
    print(f"✗ Y=246.5 is NOT valid for Bar 2")
