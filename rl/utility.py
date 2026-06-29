import json
from math import cos, sin
from typing import List

from moments_of_inertia import InertiaAnalysis
try:
    from python.machine_parameters import FixtureState, FIXTURE_DIMENSIONS
except ImportError:
    try:
        from models import FixtureState, FIXTURE_DIMENSIONS
    except ImportError:
        from machine_parameters import FixtureState, FIXTURE_DIMENSIONS

def define_fixtures_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    x = data["x"]
    y = data["y"]
    x1 = data["x1"]
    y1 = data["y1"]
    x2 = data["x2"]
    y2 = data["y2"]
    x3 = data["x3"]
    y3 = data["y3"]
    angle = data["angle"]
    selected_fixtures = data["selected_fixture"]
    fixture_types = data["fixture_type"]
    fixtures_center_x = data["fixtures_center_x"]
    fixtures_center_y = data["fixtures_center_y"]

    fixtures = []

    for idx in range(len(fixture_types)):
        if selected_fixtures[idx] == 1:
            fixture_coords = [(x[idx], y[idx]), (x1[idx], y1[idx]), (x2[idx], y2[idx]), (x3[idx], y3[idx]), (x[idx], y[idx])]
            jx, jy, jxy = InertiaAnalysis.compute_absolute_moments_of_inertia(fixture_coords)

            fixture_state = FixtureState(x=x[idx], y=y[idx], angle=angle[idx], type_id=fixture_types[idx])

            fixture = {
                "area": InertiaAnalysis.compute_polygon_area(fixture_coords),
                "absolute_moments_of_inertia": [jx, jy, jxy],
                "barycenter": (fixtures_center_x[idx], fixtures_center_y[idx]),
                "angle": angle[idx],
                "state": fixture_state,  # Include the FixtureState
                "center": (fixtures_center_x[idx], fixtures_center_y[idx]),  # Include center for security distance checks
            }

            jxg, jyg, jxyg = InertiaAnalysis.compute_baricentric_moments_of_inertia(fixture)

            fixture["baricentric_moments_of_inertia"] = [jxg, jyg, jxyg]
            fixtures.append(fixture)
    
    return fixtures
            
def compute_coordinate_with_rotation(x, y, cx, cy, angle):
    dx = x - cx
    dy = y - cy

    rx = dx * cos(angle) - dy * sin(angle) + cx
    ry = dx * sin(angle) + dy * cos(angle) + cy
    
    return (rx, ry)

def compute_fixture_center_base(x, y , t):
    cx = x + FIXTURE_DIMENSIONS[t][0] / 2
    cy = y + FIXTURE_DIMENSIONS[t][1] / 2
    
    return (cx, cy)
    
            
def define_fixture_from_state(state: List[FixtureState]):
    fixtures = []
    
    for fixture_state in state:
        t = fixture_state.type_id
        cxb, cyb = compute_fixture_center_base(fixture_state.x, fixture_state.y, t)
        x, y = compute_coordinate_with_rotation(fixture_state.x, fixture_state.y, cxb, cyb, fixture_state.angle)
        x1, y1 = compute_coordinate_with_rotation(fixture_state.x, fixture_state.y + FIXTURE_DIMENSIONS[t][1], cxb, cyb, fixture_state.angle)
        x2, y2 = compute_coordinate_with_rotation(fixture_state.x + FIXTURE_DIMENSIONS[t][0], fixture_state.y + FIXTURE_DIMENSIONS[t][1], cxb, cyb, fixture_state.angle)
        x3, y3 = compute_coordinate_with_rotation(fixture_state.x + FIXTURE_DIMENSIONS[t][0], fixture_state.y, cxb, cyb, fixture_state.angle)
        
        fixture_coords = [(x, y), (x1, y1), (x2, y2), (x3, y3), (x, y)]
        jx, jy, jxy = InertiaAnalysis.compute_absolute_moments_of_inertia(fixture_coords)
        
        fixture = {
                "area": InertiaAnalysis.compute_polygon_area(fixture_coords),
                "absolute_moments_of_inertia": [jx, jy, jxy],
                "barycenter": (cxb, cyb),
                "angle": fixture_state.angle,
                "state": fixture_state,  # Include the original FixtureState
                "center": (cxb, cyb),  # Include center for security distance checks
            }

        jxg, jyg, jxyg = InertiaAnalysis.compute_baricentric_moments_of_inertia(fixture)

        fixture["baricentric_moments_of_inertia"] = [jxg, jyg, jxyg]
        fixtures.append(fixture)
    
    return fixtures
        
        
        
    