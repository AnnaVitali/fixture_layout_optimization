#!/usr/bin/env python3
"""
Debug: Check if 3 bars can fit with 345mm spacing constraint
Corrected calculation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_bar_positioning_agent import HORIZONTAL_SPACING, BAR_WIDTH

WORKPIECE_WIDTH = 900.0  # simple_stair_step width
WORKPIECE_MIN_X = 0.0
WORKPIECE_MAX_X = 900.0

print(f"\n{'='*80}")
print(f"Spacing Constraint Analysis (CORRECTED)")
print(f"{'='*80}")
print(f"\nWorkpiece dimensions:")
print(f"  Width: {WORKPIECE_WIDTH} mm")
print(f"  Range: {WORKPIECE_MIN_X} to {WORKPIECE_MAX_X} mm")
print(f"\nBar dimensions:")
print(f"  Width: {BAR_WIDTH} mm")
print(f"  Half-width: {BAR_WIDTH/2} mm")
print(f"\nSpacing constraint:")
print(f"  Min center-to-center distance: {HORIZONTAL_SPACING} mm")

# Check if 3 bars can fit - CORRECTED
print(f"\n{'='*80}")
print(f"Can 3 bars fit?")
print(f"{'='*80}")

# For bars at x1, x2, x3:
# - Bar 1: left edge = x1 - 72.5, right edge = x1 + 72.5
# - Bar 2: left edge = x2 - 72.5, right edge = x2 + 72.5
# - Bar 3: left edge = x3 - 72.5, right edge = x3 + 72.5
# 
# Constraints:
# - x1 - 72.5 >= 0  →  x1 >= 72.5
# - x2 >= x1 + 345
# - x3 >= x2 + 345
# - x3 + 72.5 <= 900  →  x3 <= 827.5

print(f"\nConstraints:")
print(f"  x1 >= {BAR_WIDTH/2} (left edge inside workpiece)")
print(f"  x2 >= x1 + {HORIZONTAL_SPACING}")
print(f"  x3 >= x2 + {HORIZONTAL_SPACING}")
print(f"  x3 <= {WORKPIECE_MAX_X - BAR_WIDTH/2} (right edge inside workpiece)")

x1_min = BAR_WIDTH/2
x2_min = x1_min + HORIZONTAL_SPACING
x3_min = x2_min + HORIZONTAL_SPACING
x3_max = WORKPIECE_MAX_X - BAR_WIDTH/2

print(f"\nDerived minimum positions:")
print(f"  x1_min = {x1_min}")
print(f"  x2_min (x1 + spacing) = {x2_min}")
print(f"  x3_min (x2 + spacing) = {x3_min}")
print(f"  x3_max = {x3_max}")

can_fit = x3_min <= x3_max
print(f"\nCan 3 bars fit? {x3_min} <= {x3_max} = {can_fit}")

if can_fit:
    print(f"\n✅ YES! 3 bars CAN fit with 345mm spacing")
    print(f"\nExample from CP solver:")
    print(f"  Bar 1: x=91mm")
    print(f"  Bar 2: x=481mm (distance: {481-91} = 390mm >= 345mm) ✅")
    print(f"  Bar 3: x=826mm (distance: {826-481} = 345mm >= 345mm) ✅")
    
    # Verify these positions
    print(f"\nVerification:")
    for bar_x in [91, 481, 826]:
        left = bar_x - BAR_WIDTH/2
        right = bar_x + BAR_WIDTH/2
        print(f"  Bar at {bar_x}: edges [{left:.1f}, {right:.1f}] - fits: {left >= 0 and right <= 900}")
    
    print(f"\nRange for x3: [{x3_min:.1f}, {x3_max:.1f}]")
    print(f"CP solution x3=826: valid = {x3_min <= 826 <= x3_max}")
else:
    print(f"\n⚠️  3 bars CANNOT fit")
    print(f"  Need x3 >= {x3_min} but max x3 = {x3_max}")
    print(f"  Shortfall: {x3_min - x3_max:.1f}mm")

print(f"\n{'='*80}\n")

