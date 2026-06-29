#!/usr/bin/env python3
"""Test if rectangular fixtures at Y-centers 33.5 and 246.5 are valid on both bars"""

import json
from pathlib import Path
from shapely.geometry import Point as ShapelyPoint, Polygon

# Load workpiece data
workpiece_file = Path("resources/workpieces_information.json")
with open(workpiece_file, 'r') as f:
    all_workpieces = json.load(f)

workpiece = all_workpieces['simple_stair_step']
vertices = workpiece['vertices']
holes = workpiece.get('holes', [])

print("=" * 80)
print("SIMPLE STAIR STEP WORKPIECE")
print("=" * 80)
print(f"Vertices: {vertices}")
print(f"Holes: {holes}")

# Workpiece bounds
xs = [v[0] for v in vertices]
ys = [v[1] for v in vertices]
min_y, max_y = min(ys), max(ys)
print(f"\nY-bounds: {min_y} to {max_y}")

# Test parameters
bar_positions = [92.5, 602.5]
y_centers = [33.5, 246.5]
fixture_width = 180.0
fixture_height = 65.0
vertical_spacing = 100.0

print(f"\nBars at X: {bar_positions}")
print(f"Test Y-centers: {y_centers}")
print(f"Rectangular fixture: {fixture_width}×{fixture_height}mm")

workpiece_polygon = Polygon(vertices)

# Check each bar
for bar_idx, bar_x in enumerate(bar_positions):
    print(f"\n{'='*80}")
    print(f"BAR {bar_idx + 1} (x={bar_x})")
    print(f"{'='*80}")
    
    # Check each Y position
    for y_idx, y_center in enumerate(y_centers):
        print(f"\n  Y-center: {y_center}")
        
        # Compute fixture bounds
        placement_x = bar_x - fixture_width / 2
        placement_y = y_center - fixture_height / 2
        
        print(f"    Top-left: ({placement_x:.1f}, {placement_y:.1f})")
        print(f"    Bottom-right: ({placement_x + fixture_width:.1f}, {placement_y + fixture_height:.1f})")
        
        # Check Y bounds
        if placement_y < min_y or placement_y + fixture_height > max_y:
            print(f"    ❌ Out of Y-bounds (need {min_y} to {max_y - fixture_height})")
            continue
        
        # Check polygon containment
        fixture_polygon = Polygon([
            (placement_x, placement_y),
            (placement_x + fixture_width, placement_y),
            (placement_x + fixture_width, placement_y + fixture_height),
            (placement_x, placement_y + fixture_height),
        ])
        
        if not fixture_polygon.within(workpiece_polygon):
            print(f"    ❌ Not completely within workpiece polygon")
            continue
        
        # Check hole overlaps
        hole_overlap = False
        for hole in holes:
            if isinstance(hole, (list, tuple)) and len(hole) == 3:
                cx, cy, radius = hole
                hole_point = ShapelyPoint(cx, cy)
                if fixture_polygon.distance(hole_point) < radius:
                    print(f"    ❌ Overlaps hole at ({cx}, {cy}, r={radius})")
                    hole_overlap = True
                    break
        
        if hole_overlap:
            continue
        
        print(f"    ✓ Position is valid (bounds OK, no hole overlap)")
        
        # Check spacing between the two fixtures on this bar
        if y_idx == 1:  # Second fixture
            y1 = y_centers[0]
            y2 = y_centers[1]
            min_distance = fixture_height / 2 + fixture_height / 2 + vertical_spacing
            actual_distance = abs(y2 - y1)
            
            print(f"    Spacing check with first fixture:")
            print(f"      Y1={y1}, Y2={y2}")
            print(f"      Required distance: {min_distance}mm (height/2 + height/2 + {vertical_spacing})")
            print(f"      Actual distance: {actual_distance}mm")
            
            if actual_distance >= min_distance:
                print(f"    ✓ Spacing constraint satisfied")
            else:
                print(f"    ❌ Spacing constraint FAILED (gap only {actual_distance - min_distance:.1f}mm short)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Checking if two rectangular fixtures (180×65) can be placed per bar:")
print(f"  Bar 1 (x={bar_positions[0]}): Y-centers {y_centers[0]} and {y_centers[1]}")
print(f"  Bar 2 (x={bar_positions[1]}): Y-centers {y_centers[0]} and {y_centers[1]}")
