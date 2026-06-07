#!/usr/bin/env python3
"""
Interactive Web Tool for S-Entropy Based Positioning
Demos: 1) Shortest Path, 2) Reachable Region, 3) Satellite Visibility
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import math
from datetime import datetime, timedelta
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

EARTH_RADIUS_KM = 6371.0
SATELLITE_ALTITUDE_KM = 20000  # Typical MEO altitude (Galileo/GLONASS)

# Simulated satellite constellation (example: Galileo-like)
SATELLITES = [
    {"id": 1, "lat": 56.0, "lon": 0, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 2, "lat": 56.0, "lon": 120, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 3, "lat": 56.0, "lon": 240, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 4, "lat": -56.0, "lon": 60, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 5, "lat": -56.0, "lon": 180, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 6, "lat": -56.0, "lon": 300, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 7, "lat": 0, "lon": 30, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 8, "lat": 0, "lon": 150, "alt": SATELLITE_ALTITUDE_KM},
    {"id": 9, "lat": 0, "lon": 270, "alt": SATELLITE_ALTITUDE_KM},
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def elevation_angle(observer_lat, observer_lon, observer_alt_m,
                   satellite_lat, satellite_lon, satellite_alt_m):
    """
    Calculate elevation angle of satellite from observer position.
    Returns angle in degrees.
    Positive = above horizon, negative = below horizon.
    """
    observer_alt_km = observer_alt_m / 1000.0
    satellite_alt_km = satellite_alt_m / 1000.0

    # Great-circle distance
    gc_distance = haversine_distance(observer_lat, observer_lon,
                                     satellite_lat, satellite_lon)

    # Slant distance (3D Euclidean approximation)
    slant_distance = math.sqrt(
        gc_distance**2 +
        (satellite_alt_km - observer_alt_km)**2
    )

    if slant_distance == 0:
        return 90.0

    # Angle from observer to satellite
    dalt = satellite_alt_km - observer_alt_km
    elevation = rad2deg(math.atan2(dalt, gc_distance))

    return elevation


def s_entropy_distance(lat1, lon1, lat2, lon2):
    """
    Compute S-entropy-based distance metric.
    Incorporates:
    - Geographic distance (Haversine)
    - Terrain roughness approximation (latitude-dependent)
    - Infrastructure density approximation (longitude-dependent)

    Returns distance in "S-entropy units" (0-1 scale).
    """
    # Normalized geographic distance
    gc_dist = haversine_distance(lat1, lon1, lat2, lon2)
    max_dist = haversine_distance(-90, 0, 90, 180)  # Max possible distance
    geo_component = gc_dist / max_dist

    # Terrain component: rougher at poles, smoother at equator
    terrain1 = abs(lat1) / 90.0  # 0 at equator, 1 at poles
    terrain2 = abs(lat2) / 90.0
    terrain_component = abs(terrain1 - terrain2)

    # Infrastructure component: varies by longitude
    infra1 = (lon1 % 180) / 180.0
    infra2 = (lon2 % 180) / 180.0
    infra_component = abs(infra1 - infra2)

    # Combine components
    s_entropy_dist = math.sqrt(
        0.5 * geo_component**2 +
        0.3 * terrain_component**2 +
        0.2 * infra_component**2
    )

    return min(s_entropy_dist, 1.0)


def dijkstra_shortest_path(lat1, lon1, lat2, lon2, grid_size=10):
    """
    Simple A*-like shortest path using S-entropy metric on a grid.
    Returns list of waypoints (lat, lon) from start to goal.
    """
    from heapq import heappush, heappop

    # Grid of waypoints
    lat_range = np.linspace(min(lat1, lat2) - 5, max(lat1, lat2) + 5, grid_size)
    lon_range = np.linspace(min(lon1, lon2) - 5, max(lon1, lon2) + 5, grid_size)

    grid_points = {}
    for i, lat in enumerate(lat_range):
        for j, lon in enumerate(lon_range):
            grid_points[(i, j)] = (lat, lon)

    # Find start and goal in grid
    start_idx = (0, 0)
    goal_idx = (grid_size-1, grid_size-1)

    # Dijkstra
    open_set = [(0, start_idx)]
    came_from = {}
    g_score = {start_idx: 0}
    closed_set = set()

    while open_set:
        current_cost, current = heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)

        if current == goal_idx:
            # Reconstruct path
            path = []
            node = goal_idx
            while node in came_from:
                path.append(grid_points[node])
                node = came_from[node]
            path.append(grid_points[start_idx])
            path.reverse()
            return path

        # Explore neighbors
        i, j = current
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (i + di, j + dj)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                if neighbor in closed_set:
                    continue

                lat_curr, lon_curr = grid_points[current]
                lat_next, lon_next = grid_points[neighbor]

                # Cost: S-entropy distance
                cost = s_entropy_distance(lat_curr, lon_curr, lat_next, lon_next)

                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # Heuristic: straight-line distance to goal
                    goal_lat, goal_lon = grid_points[goal_idx]
                    h = haversine_distance(lat_next, lon_next, goal_lat, goal_lon)
                    f_score = tentative_g + h

                    heappush(open_set, (f_score, neighbor))

    # No path found
    return [(lat1, lon1), (lat2, lon2)]


def reachable_region(center_lat, center_lon, walking_time_hours, speed_kmh=5.0):
    """
    Compute reachable region from a point in given time.
    Returns list of (lat, lon) points forming a circle.
    """
    max_distance_km = walking_time_hours * speed_kmh

    # Sample circle around the center
    n_points = 360
    points = []

    for angle in np.linspace(0, 2*np.pi, n_points):
        # Compute point at angle and max_distance
        lat_rad = deg2rad(center_lat)
        lon_rad = deg2rad(center_lon)
        d_rad = max_distance_km / EARTH_RADIUS_KM

        lat_new_rad = math.asin(
            math.sin(lat_rad) * math.cos(d_rad) +
            math.cos(lat_rad) * math.sin(d_rad) * math.cos(angle)
        )
        lon_new_rad = lon_rad + math.atan2(
            math.sin(angle) * math.sin(d_rad) * math.cos(lat_rad),
            math.cos(d_rad) - math.sin(lat_rad) * math.sin(lat_new_rad)
        )

        lat_new = rad2deg(lat_new_rad)
        lon_new = rad2deg(lon_new_rad)
        points.append([lat_new, lon_new])

    return points


def visible_satellites(observer_lat, observer_lon, observer_alt_m=0, min_elevation=10):
    """
    Determine which satellites are visible from observer position.
    Returns list of visible satellites with elevation angles.
    """
    visible = []

    for sat in SATELLITES:
        elev = elevation_angle(
            observer_lat, observer_lon, observer_alt_m,
            sat["lat"], sat["lon"], sat["alt"] * 1000
        )

        if elev >= min_elevation:
            visible.append({
                "id": sat["id"],
                "lat": sat["lat"],
                "lon": sat["lon"],
                "elevation": round(elev, 2),
                "distance": haversine_distance(observer_lat, observer_lon,
                                              sat["lat"], sat["lon"])
            })

    return sorted(visible, key=lambda x: x["elevation"], reverse=True)


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/demo/shortest-path')
def shortest_path_demo():
    """Shortest path demo page."""
    return render_template('shortest_path.html')


@app.route('/demo/reachable-region')
def reachable_region_demo():
    """Reachable region demo page."""
    return render_template('reachable_region.html')


@app.route('/demo/satellite-visibility')
def satellite_visibility_demo():
    """Satellite visibility demo page."""
    return render_template('satellite_visibility.html')


@app.route('/api/shortest-path', methods=['POST'])
def api_shortest_path():
    """Compute shortest path between two points."""
    data = request.json

    lat1 = float(data.get('lat1', 40.7128))
    lon1 = float(data.get('lon1', -74.0060))
    lat2 = float(data.get('lat2', 48.8566))
    lon2 = float(data.get('lon2', 2.3522))

    # Compute path
    path = dijkstra_shortest_path(lat1, lon1, lat2, lon2, grid_size=8)

    # Compute total distance
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += haversine_distance(
            path[i][0], path[i][1], path[i+1][0], path[i+1][1]
        )

    return jsonify({
        "success": True,
        "path": path,
        "total_distance_km": round(total_distance, 2),
        "waypoints": len(path)
    })


@app.route('/api/reachable-region', methods=['POST'])
def api_reachable_region():
    """Compute reachable region from a point."""
    data = request.json

    lat = float(data.get('lat', 40.7128))
    lon = float(data.get('lon', -74.0060))
    time_hours = float(data.get('time_hours', 2.0))
    speed_kmh = float(data.get('speed_kmh', 5.0))

    # Compute region
    region = reachable_region(lat, lon, time_hours, speed_kmh)
    max_dist = time_hours * speed_kmh

    return jsonify({
        "success": True,
        "center": [lat, lon],
        "boundary": region,
        "max_distance_km": round(max_dist, 2),
        "time_hours": time_hours,
        "speed_kmh": speed_kmh
    })


@app.route('/api/visible-satellites', methods=['POST'])
def api_visible_satellites():
    """Determine visible satellites from a location."""
    data = request.json

    lat = float(data.get('lat', 40.7128))
    lon = float(data.get('lon', -74.0060))
    alt_m = float(data.get('alt_m', 10))
    min_elev = float(data.get('min_elevation', 10))

    # Get visible satellites
    visible = visible_satellites(lat, lon, alt_m, min_elev)

    return jsonify({
        "success": True,
        "observer": {"lat": lat, "lon": lon, "alt_m": alt_m},
        "visible_satellites": visible,
        "count": len(visible),
        "total_satellites": len(SATELLITES),
        "min_elevation": min_elev
    })


@app.route('/api/all-satellites')
def api_all_satellites():
    """Get all satellites in constellation."""
    return jsonify({
        "success": True,
        "satellites": SATELLITES,
        "count": len(SATELLITES)
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("S-ENTROPY POSITIONING WEB TOOL")
    print("="*70)
    print("\nAvailable demos:")
    print("  1. Shortest Path: http://localhost:5000/demo/shortest-path")
    print("  2. Reachable Region: http://localhost:5000/demo/reachable-region")
    print("  3. Satellite Visibility: http://localhost:5000/demo/satellite-visibility")
    print("\nStarting Flask server on http://localhost:5000")
    print("="*70 + "\n")

    app.run(debug=True, host='localhost', port=5000)
