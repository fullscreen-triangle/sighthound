"""
Generate publication-quality figures for the papers

Creates 4-panel charts with minimal text and 3D visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from pathlib import Path

# Style configuration
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300


def load_validation_results():
    """Load validation results from JSON"""
    results_path = Path("c:/Users/kundai/Documents/geosciences/sighthound/validation/results/complete_circular_validation.json")

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data['results']


def create_cynegeticus_panel_1(results, output_dir):
    """
    Panel 1: GPS Trajectory and S-Entropy Spatial Distribution

    [GPS Track] [S-entropy Heatmap] [3D Trajectory] [Closure Visualization]
    """
    fig = plt.figure(figsize=(20, 5))

    # Extract data
    watch1 = results['original_gps']['watch1']
    watch2 = results['original_gps']['watch2']

    lats1 = [p['lat'] for p in watch1]
    lons1 = [p['lon'] for p in watch1]
    lats2 = [p['lat'] for p in watch2]
    lons2 = [p['lon'] for p in watch2]

    states1 = results['atmospheric_from_gps']['watch1']
    states2 = results['atmospheric_from_gps']['watch2']

    S_k1 = [s['S_k'] for s in states1]
    S_e1 = [s['S_e'] for s in states1]

    # Panel 1: GPS trajectory
    ax1 = fig.add_subplot(141)
    ax1.plot(lons1, lats1, 'b-', linewidth=2, alpha=0.7, label='Watch 1')
    ax1.plot(lons2, lats2, 'r-', linewidth=2, alpha=0.7, label='Watch 2')
    ax1.scatter(lons1[0], lats1[0], c='green', s=100, marker='o', zorder=5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: S-entropy heatmap
    ax2 = fig.add_subplot(142)
    scatter = ax2.scatter(lons1, lats1, c=S_k1, s=50, cmap='viridis', vmin=0, vmax=1)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('S_k Distribution')
    plt.colorbar(scatter, ax=ax2, label='S_k')
    ax2.set_aspect('equal')

    # Panel 3: 3D trajectory in (lon, lat, S_k) space
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.plot(lons1, lats1, S_k1, 'b-', linewidth=2, alpha=0.7)
    ax3.scatter(lons1, lats1, S_k1, c=S_k1, s=20, cmap='viridis', vmin=0, vmax=1)
    ax3.set_xlabel('Lon')
    ax3.set_ylabel('Lat')
    ax3.set_zlabel('S_k')
    ax3.set_title('3D Trajectory')

    # Panel 4: Circular closure
    ax4 = fig.add_subplot(144)

    # Get derived GPS
    derived = results['derived_gps']['trajectories']
    derived_lats = []
    derived_lons = []
    for traj in derived:
        for pos in traj['positions']:
            derived_lats.append(pos['lat'])
            derived_lons.append(pos['lon'])

    all_orig_lats = lats1 + lats2
    all_orig_lons = lons1 + lons2

    n = min(len(all_orig_lats), len(derived_lats))

    # Plot original vs reconstructed
    ax4.plot(all_orig_lons[:n], all_orig_lats[:n], 'b-', alpha=0.5, linewidth=2, label='Original')
    ax4.plot(derived_lons[:n], derived_lats[:n], 'r--', alpha=0.5, linewidth=2, label='Reconstructed')

    # Draw error vectors
    for i in range(0, n, 5):  # Every 5th point to avoid clutter
        ax4.arrow(all_orig_lons[i], all_orig_lats[i],
                 derived_lons[i] - all_orig_lons[i],
                 derived_lats[i] - all_orig_lats[i],
                 head_width=0.00005, head_length=0.00005, fc='gray', ec='gray', alpha=0.3)

    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_title('Circular Closure')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'cynegeticus_panel_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Cynegeticus Panel 1 saved")


def create_cynegeticus_panel_2(results, output_dir):
    """
    Panel 2: S-Entropy Coordinate Analysis

    [S_k vs Point] [S_t vs Point] [S_e vs Point] [3D S-Entropy Space]
    """
    fig = plt.figure(figsize=(20, 5))

    states1 = results['atmospheric_from_gps']['watch1']
    states2 = results['atmospheric_from_gps']['watch2']

    indices1 = [s['point_index'] for s in states1]
    S_k1 = [s['S_k'] for s in states1]
    S_t1 = [s['S_t'] for s in states1]
    S_e1 = [s['S_e'] for s in states1]

    indices2 = [s['point_index'] for s in states2]
    S_k2 = [s['S_k'] for s in states2]
    S_t2 = [s['S_t'] for s in states2]
    S_e2 = [s['S_e'] for s in states2]

    # Panel 1: S_k evolution
    ax1 = fig.add_subplot(141)
    ax1.plot(indices1, S_k1, 'b-', linewidth=2, alpha=0.7, label='Watch 1')
    ax1.plot(indices2, S_k2, 'r-', linewidth=2, alpha=0.7, label='Watch 2')
    ax1.set_xlabel('Point Index')
    ax1.set_ylabel('S_k')
    ax1.set_title('Compositional Entropy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: S_t evolution
    ax2 = fig.add_subplot(142)
    ax2.plot(indices1, S_t1, 'b-', linewidth=2, alpha=0.7, label='Watch 1')
    ax2.plot(indices2, S_t2, 'r-', linewidth=2, alpha=0.7, label='Watch 2')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('S_t')
    ax2.set_title('Temporal Entropy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: S_e evolution
    ax3 = fig.add_subplot(143)
    ax3.plot(indices1, S_e1, 'b-', linewidth=2, alpha=0.7, label='Watch 1')
    ax3.plot(indices2, S_e2, 'r-', linewidth=2, alpha=0.7, label='Watch 2')
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('S_e')
    ax3.set_title('Energy Entropy')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: 3D S-entropy space
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(S_k1, S_t1, S_e1, c=indices1, s=50, cmap='viridis', alpha=0.7, label='Watch 1')
    ax4.scatter(S_k2, S_t2, S_e2, c=indices2, s=50, cmap='plasma', alpha=0.7, marker='^', label='Watch 2')
    ax4.set_xlabel('S_k')
    ax4.set_ylabel('S_t')
    ax4.set_zlabel('S_e')
    ax4.set_title('Partition Space')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'cynegeticus_panel_2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Cynegeticus Panel 2 saved")


def create_cynegeticus_panel_3(results, output_dir):
    """
    Panel 3: Position Reconstruction Validation

    [Lat Comparison] [Lon Comparison] [3D Position Space] [Error Distribution]
    """
    fig = plt.figure(figsize=(20, 5))

    # Extract original and derived positions
    watch1 = results['original_gps']['watch1']
    watch2 = results['original_gps']['watch2']

    orig_lats = [p['lat'] for p in watch1] + [p['lat'] for p in watch2]
    orig_lons = [p['lon'] for p in watch1] + [p['lon'] for p in watch2]

    derived = results['derived_gps']['trajectories']
    derived_lats = []
    derived_lons = []
    for traj in derived:
        for pos in traj['positions']:
            derived_lats.append(pos['lat'])
            derived_lons.append(pos['lon'])

    n = min(len(orig_lats), len(derived_lats))
    orig_lats = orig_lats[:n]
    orig_lons = orig_lons[:n]
    derived_lats = derived_lats[:n]
    derived_lons = derived_lons[:n]

    # Panel 1: Latitude comparison
    ax1 = fig.add_subplot(141)
    ax1.scatter(orig_lats, derived_lats, alpha=0.5, s=20)
    ax1.plot([min(orig_lats), max(orig_lats)], [min(orig_lats), max(orig_lats)], 'r--', linewidth=2)
    ax1.set_xlabel('Original Lat')
    ax1.set_ylabel('Reconstructed Lat')
    ax1.set_title('Latitude Validation')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: Longitude comparison
    ax2 = fig.add_subplot(142)
    ax2.scatter(orig_lons, derived_lons, alpha=0.5, s=20)
    ax2.plot([min(orig_lons), max(orig_lons)], [min(orig_lons), max(orig_lons)], 'r--', linewidth=2)
    ax2.set_xlabel('Original Lon')
    ax2.set_ylabel('Reconstructed Lon')
    ax2.set_title('Longitude Validation')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Panel 3: 3D position space comparison
    ax3 = fig.add_subplot(143, projection='3d')
    indices = np.arange(n)
    ax3.scatter(orig_lons, orig_lats, indices, c='blue', s=20, alpha=0.6, label='Original')
    ax3.scatter(derived_lons, derived_lats, indices, c='red', s=20, alpha=0.6, marker='^', label='Reconstructed')
    ax3.set_xlabel('Lon')
    ax3.set_ylabel('Lat')
    ax3.set_zlabel('Point Index')
    ax3.set_title('3D Trajectory Comparison')
    ax3.legend()

    # Panel 4: Error distribution
    ax4 = fig.add_subplot(144)

    # Compute errors in meters
    lat_errors = np.array([(o - d) * 111000 for o, d in zip(orig_lats, derived_lats)])
    lon_errors = np.array([(o - d) * 111000 * np.cos(np.radians(np.mean(orig_lats))) for o, d in zip(orig_lons, derived_lons)])

    horizontal_errors = np.sqrt(lat_errors**2 + lon_errors**2)

    ax4.hist(horizontal_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(horizontal_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(horizontal_errors):.2f} m')
    ax4.axvline(np.median(horizontal_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(horizontal_errors):.2f} m')
    ax4.set_xlabel('Error (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Position Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cynegeticus_panel_3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Cynegeticus Panel 3 saved")


def create_cynegeticus_panel_4(results, output_dir):
    """
    Panel 4: Virtual Satellite Concept

    [Earth View] [Satellite Constellation] [3D Measurement Geometry] [Uncertainty Ellipses]
    """
    fig = plt.figure(figsize=(20, 5))

    # Panel 1: Earth with measurement location
    ax1 = fig.add_subplot(141)

    # Draw Earth circle
    theta = np.linspace(0, 2*np.pi, 100)
    earth_r = 6371  # km
    ax1.plot(earth_r * np.cos(theta), earth_r * np.sin(theta), 'b-', linewidth=2)
    ax1.fill(earth_r * np.cos(theta), earth_r * np.sin(theta), color='lightblue', alpha=0.3)

    # Munich location (approximate)
    munich_angle = np.radians(48.183)  # latitude
    munich_x = earth_r * np.cos(munich_angle)
    munich_y = earth_r * np.sin(munich_angle)
    ax1.plot(munich_x, munich_y, 'ro', markersize=10, label='Munich')

    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('y (km)')
    ax1.set_title('Measurement Location')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: Virtual satellite constellation
    ax2 = fig.add_subplot(142)

    # GPS orbit radius
    r_gps = 26560  # km

    # Draw Earth
    ax2.plot(earth_r * np.cos(theta), earth_r * np.sin(theta), 'b-', linewidth=2)
    ax2.fill(earth_r * np.cos(theta), earth_r * np.sin(theta), color='lightblue', alpha=0.3)

    # Draw GPS orbit
    ax2.plot(r_gps * np.cos(theta), r_gps * np.sin(theta), 'g--', linewidth=1, alpha=0.5)

    # Virtual satellites (8 satellites evenly spaced)
    n_sats = 8
    sat_angles = np.linspace(0, 2*np.pi, n_sats, endpoint=False)
    sat_x = r_gps * np.cos(sat_angles)
    sat_y = r_gps * np.sin(sat_angles)
    ax2.scatter(sat_x, sat_y, c='red', s=100, marker='^', label='Virtual Satellites', zorder=5)

    # Draw lines from satellites to Munich
    for sx, sy in zip(sat_x[:4], sat_y[:4]):  # Show 4 satellites for clarity
        ax2.plot([munich_x, sx], [munich_y, sy], 'r--', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('x (km)')
    ax2.set_ylabel('y (km)')
    ax2.set_title('Virtual Constellation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Panel 3: 3D measurement geometry
    ax3 = fig.add_subplot(143, projection='3d')

    # 3D Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = earth_r * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_r * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax3.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

    # Virtual satellites in 3D
    phi = np.linspace(0, 2*np.pi, n_sats, endpoint=False)
    sat_x_3d = r_gps * np.cos(phi)
    sat_y_3d = r_gps * np.sin(phi)
    sat_z_3d = np.zeros(n_sats)
    ax3.scatter(sat_x_3d, sat_y_3d, sat_z_3d, c='red', s=100, marker='^', label='Satellites')

    # Munich in 3D
    ax3.scatter([munich_x], [munich_y], [0], c='yellow', s=200, marker='*', label='Munich')

    ax3.set_xlabel('x (km)')
    ax3.set_ylabel('y (km)')
    ax3.set_zlabel('z (km)')
    ax3.set_title('3D Geometry')
    ax3.legend()

    # Panel 4: Position uncertainty ellipses
    ax4 = fig.add_subplot(144)

    watch1 = results['original_gps']['watch1']
    lats = [p['lat'] for p in watch1[:20]]  # First 20 points
    lons = [p['lon'] for p in watch1[:20]]

    # Plot trajectory
    ax4.plot(lons, lats, 'b-', linewidth=1, alpha=0.5)

    # Draw uncertainty ellipses
    for i in range(0, len(lats), 3):  # Every 3rd point
        # Uncertainty from closure results (0.22 m)
        uncertainty_m = 0.22
        uncertainty_lat = uncertainty_m / 111000
        uncertainty_lon = uncertainty_m / (111000 * np.cos(np.radians(lats[i])))

        ellipse = patches.Ellipse((lons[i], lats[i]),
                                 width=2*uncertainty_lon,
                                 height=2*uncertainty_lat,
                                 angle=0,
                                 facecolor='red',
                                 edgecolor='red',
                                 alpha=0.3)
        ax4.add_patch(ellipse)

    ax4.scatter(lons, lats, c='blue', s=30, zorder=5)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_title('Position Uncertainty')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'cynegeticus_panel_4.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Cynegeticus Panel 4 saved")


def create_ober_panel_1(results, output_dir):
    """
    Panel 1: Weather Forecast Evolution

    [Temperature] [Pressure] [Humidity] [3D Weather State]
    """
    fig = plt.figure(figsize=(20, 5))

    forecast = results['weather_forecast']
    actual = results['actual_weather']

    days_f = [f['day'] for f in forecast]
    temp_f = [f['temperature_C'] for f in forecast]
    pres_f = [f['pressure_hPa'] for f in forecast]
    humid_f = [f['humidity_percent'] for f in forecast]

    days_a = [a['day'] for a in actual]
    temp_a = [a.get('temperature_C', 15) for a in actual]
    pres_a = [a.get('pressure_hPa', 1013) for a in actual]
    humid_a = [a.get('humidity_percent', 50) for a in actual]

    # Panel 1: Temperature
    ax1 = fig.add_subplot(141)
    ax1.plot(days_f, temp_f, 'b-', linewidth=2, marker='o', label='Forecast')
    ax1.plot(days_a, temp_a, 'r--', linewidth=2, marker='s', label='Actual')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Pressure
    ax2 = fig.add_subplot(142)
    ax2.plot(days_f, pres_f, 'b-', linewidth=2, marker='o', label='Forecast')
    ax2.plot(days_a, pres_a, 'r--', linewidth=2, marker='s', label='Actual')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title('Pressure Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Humidity
    ax3 = fig.add_subplot(143)
    ax3.plot(days_f, humid_f, 'b-', linewidth=2, marker='o', label='Forecast')
    ax3.plot(days_a, humid_a, 'r--', linewidth=2, marker='s', label='Actual')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Humidity (%)')
    ax3.set_title('Humidity Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: 3D weather state trajectory
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.plot(temp_f, pres_f, humid_f, 'b-', linewidth=2, label='Forecast')
    ax4.scatter(temp_f, pres_f, humid_f, c=days_f, s=50, cmap='viridis')
    ax4.plot(temp_a, pres_a, humid_a, 'r--', linewidth=2, label='Actual', alpha=0.7)
    ax4.set_xlabel('T (°C)')
    ax4.set_ylabel('P (hPa)')
    ax4.set_zlabel('H (%)')
    ax4.set_title('Weather State Trajectory')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'ober_panel_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Ober Panel 1 saved")


def create_ober_panel_2(results, output_dir):
    """
    Panel 2: Partition Dynamics Evolution

    [S_k Evolution] [S_t Evolution] [S_e Evolution] [3D Partition Trajectory]
    """
    fig = plt.figure(figsize=(20, 5))

    forecast = results['weather_forecast']

    days = [f['day'] for f in forecast]
    S_k = [f['S_k'] for f in forecast]
    S_t = [f['S_t'] for f in forecast]
    S_e = [f['S_e'] for f in forecast]

    # Panel 1: S_k evolution
    ax1 = fig.add_subplot(141)
    ax1.plot(days, S_k, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('S_k')
    ax1.set_title('Compositional Partition')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(days, 0, S_k, alpha=0.3)

    # Panel 2: S_t evolution
    ax2 = fig.add_subplot(142)
    ax2.plot(days, S_t, 'g-', linewidth=2, marker='o')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('S_t')
    ax2.set_title('Temporal Partition')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(days, 0, S_t, alpha=0.3, color='green')

    # Panel 3: S_e evolution
    ax3 = fig.add_subplot(143)
    ax3.plot(days, S_e, 'r-', linewidth=2, marker='o')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('S_e')
    ax3.set_title('Energy Partition')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(days, 0, S_e, alpha=0.3, color='red')

    # Panel 4: 3D partition trajectory
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.plot(S_k, S_t, S_e, 'b-', linewidth=3)
    scatter = ax4.scatter(S_k, S_t, S_e, c=days, s=100, cmap='viridis', edgecolors='black', linewidth=0.5)
    ax4.scatter([S_k[0]], [S_t[0]], [S_e[0]], c='green', s=200, marker='o', edgecolors='black', linewidth=2, label='Start')
    ax4.scatter([S_k[-1]], [S_t[-1]], [S_e[-1]], c='red', s=200, marker='s', edgecolors='black', linewidth=2, label='End')
    ax4.set_xlabel('S_k')
    ax4.set_ylabel('S_t')
    ax4.set_zlabel('S_e')
    ax4.set_title('Partition Dynamics')
    ax4.legend()
    plt.colorbar(scatter, ax=ax4, label='Day', shrink=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'ober_panel_2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Ober Panel 2 saved")


def create_ober_panel_3(results, output_dir):
    """
    Panel 3: Forecast Validation

    [Temp Scatter] [Pressure Scatter] [Wind Scatter] [3D Error Distribution]
    """
    fig = plt.figure(figsize=(20, 5))

    forecast = results['weather_forecast']
    actual = results['actual_weather']

    n = min(len(forecast), len(actual))

    temp_f = [forecast[i]['temperature_C'] for i in range(n)]
    pres_f = [forecast[i]['pressure_hPa'] for i in range(n)]
    wind_f = [forecast[i]['wind_speed_ms'] for i in range(n)]

    temp_a = [actual[i].get('temperature_C', 15) for i in range(n)]
    pres_a = [actual[i].get('pressure_hPa', 1013) for i in range(n)]
    wind_a = [actual[i].get('wind_speed_ms', 3) for i in range(n)]

    # Panel 1: Temperature scatter
    ax1 = fig.add_subplot(141)
    ax1.scatter(temp_a, temp_f, s=100, alpha=0.6)
    min_t = min(min(temp_a), min(temp_f))
    max_t = max(max(temp_a), max(temp_f))
    ax1.plot([min_t, max_t], [min_t, max_t], 'r--', linewidth=2)
    ax1.set_xlabel('Actual T (°C)')
    ax1.set_ylabel('Forecast T (°C)')
    ax1.set_title('Temperature')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: Pressure scatter
    ax2 = fig.add_subplot(142)
    ax2.scatter(pres_a, pres_f, s=100, alpha=0.6, c='green')
    min_p = min(min(pres_a), min(pres_f))
    max_p = max(max(pres_a), max(pres_f))
    ax2.plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2)
    ax2.set_xlabel('Actual P (hPa)')
    ax2.set_ylabel('Forecast P (hPa)')
    ax2.set_title('Pressure')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Panel 3: Wind scatter
    ax3 = fig.add_subplot(143)
    ax3.scatter(wind_a, wind_f, s=100, alpha=0.6, c='orange')
    min_w = min(min(wind_a), min(wind_f))
    max_w = max(max(wind_a), max(wind_f))
    ax3.plot([min_w, max_w], [min_w, max_w], 'r--', linewidth=2)
    ax3.set_xlabel('Actual Wind (m/s)')
    ax3.set_ylabel('Forecast Wind (m/s)')
    ax3.set_title('Wind Speed')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Panel 4: 3D error distribution
    ax4 = fig.add_subplot(144, projection='3d')

    temp_err = [f - a for f, a in zip(temp_f, temp_a)]
    pres_err = [f - a for f, a in zip(pres_f, pres_a)]
    wind_err = [f - a for f, a in zip(wind_f, wind_a)]

    days = list(range(n))

    ax4.scatter(temp_err, pres_err, wind_err, c=days, s=100, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Temp Error (°C)')
    ax4.set_ylabel('Pressure Error (hPa)')
    ax4.set_zlabel('Wind Error (m/s)')
    ax4.set_title('3D Error Space')

    # Add zero planes
    ax4.plot([0, 0], ax4.get_ylim(), [0, 0], 'k--', alpha=0.3, linewidth=0.5)
    ax4.plot(ax4.get_xlim(), [0, 0], [0, 0], 'k--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'ober_panel_3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Ober Panel 3 saved")


def create_ober_panel_4(results, output_dir):
    """
    Panel 4: Skill Metrics and Performance

    [RMSE Evolution] [Correlation] [Skill Score] [3D Performance Space]
    """
    fig = plt.figure(figsize=(20, 5))

    forecast = results['weather_forecast']
    actual = results['actual_weather']

    n = min(len(forecast), len(actual))
    days = list(range(n))

    # Compute cumulative errors
    cumulative_temp_rmse = []
    cumulative_pres_rmse = []
    cumulative_humid_rmse = []

    for i in range(1, n+1):
        temp_errors = [(forecast[j]['temperature_C'] - actual[j].get('temperature_C', 15))**2 for j in range(i)]
        pres_errors = [(forecast[j]['pressure_hPa'] - actual[j].get('pressure_hPa', 1013))**2 for j in range(i)]
        humid_errors = [(forecast[j]['humidity_percent'] - actual[j].get('humidity_percent', 50))**2 for j in range(i)]

        cumulative_temp_rmse.append(np.sqrt(np.mean(temp_errors)))
        cumulative_pres_rmse.append(np.sqrt(np.mean(pres_errors)))
        cumulative_humid_rmse.append(np.sqrt(np.mean(humid_errors)))

    # Panel 1: RMSE evolution
    ax1 = fig.add_subplot(141)
    ax1.plot(days, cumulative_temp_rmse, 'b-', linewidth=2, marker='o', label='Temperature')
    ax1.plot(days, [r/10 for r in cumulative_pres_rmse], 'g-', linewidth=2, marker='s', label='Pressure/10')
    ax1.plot(days, cumulative_humid_rmse, 'r-', linewidth=2, marker='^', label='Humidity')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Cumulative RMSE')
    ax1.set_title('Error Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Forecast vs actual correlation
    ax2 = fig.add_subplot(142)

    temp_f = [forecast[i]['temperature_C'] for i in range(n)]
    temp_a = [actual[i].get('temperature_C', 15) for i in range(n)]

    correlation = np.corrcoef(temp_f, temp_a)[0, 1]

    ax2.scatter(temp_a, temp_f, s=100, alpha=0.6, c=days, cmap='viridis')
    min_t = min(min(temp_a), min(temp_f))
    max_t = max(max(temp_a), max(temp_f))
    ax2.plot([min_t, max_t], [min_t, max_t], 'r--', linewidth=2)
    ax2.set_xlabel('Actual T (°C)')
    ax2.set_ylabel('Forecast T (°C)')
    ax2.set_title(f'Correlation: {correlation:.3f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Panel 3: Skill score evolution
    ax3 = fig.add_subplot(143)

    # Skill score: 1 - (RMSE / climatology_std)
    climatology_std = np.std(temp_a)
    skill_scores = [1 - (rmse / climatology_std) for rmse in cumulative_temp_rmse]

    ax3.plot(days, skill_scores, 'b-', linewidth=2, marker='o')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Skill Score')
    ax3.set_title('Forecast Skill')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(days, 0, skill_scores, where=[s > 0 for s in skill_scores], alpha=0.3, color='green')
    ax3.fill_between(days, 0, skill_scores, where=[s <= 0 for s in skill_scores], alpha=0.3, color='red')

    # Panel 4: 3D performance space
    ax4 = fig.add_subplot(144, projection='3d')

    pres_f = [forecast[i]['pressure_hPa'] for i in range(n)]
    humid_f = [forecast[i]['humidity_percent'] for i in range(n)]

    pres_a = [actual[i].get('pressure_hPa', 1013) for i in range(n)]
    humid_a = [actual[i].get('humidity_percent', 50) for i in range(n)]

    ax4.scatter(temp_a, pres_a, humid_a, c='blue', s=100, alpha=0.6, marker='o', label='Actual')
    ax4.scatter(temp_f, pres_f, humid_f, c='red', s=100, alpha=0.6, marker='^', label='Forecast')

    # Connect corresponding points
    for i in range(0, n, 2):  # Every 2nd point for clarity
        ax4.plot([temp_a[i], temp_f[i]],
                [pres_a[i], pres_f[i]],
                [humid_a[i], humid_f[i]],
                'gray', alpha=0.3, linewidth=0.5)

    ax4.set_xlabel('T (°C)')
    ax4.set_ylabel('P (hPa)')
    ax4.set_zlabel('H (%)')
    ax4.set_title('Forecast vs Actual')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'ober_panel_4.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Ober Panel 4 saved")


def main():
    """Generate all figure panels"""
    print("="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)

    # Load validation results
    print("\nLoading validation results...")
    results = load_validation_results()

    # Output directory
    output_dir = Path("c:/Users/kundai/Documents/geosciences/sighthound/validation/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Cynegeticus figures...")
    create_cynegeticus_panel_1(results, output_dir)
    create_cynegeticus_panel_2(results, output_dir)
    create_cynegeticus_panel_3(results, output_dir)
    create_cynegeticus_panel_4(results, output_dir)

    print("\nGenerating Ober figures...")
    create_ober_panel_1(results, output_dir)
    create_ober_panel_2(results, output_dir)
    create_ober_panel_3(results, output_dir)
    create_ober_panel_4(results, output_dir)

    print("\n" + "="*80)
    print("[OK] ALL FIGURES GENERATED")
    print(f"[OK] Saved to: {output_dir}")
    print("="*80)

    print("\nFigures created:")
    print("\nCynegeticus (GPS Positioning):")
    print("  - Panel 1: GPS trajectories and S-entropy distribution")
    print("  - Panel 2: S-entropy coordinate analysis")
    print("  - Panel 3: Position reconstruction validation")
    print("  - Panel 4: Virtual satellite concept")
    print("\nOber (Weather Prediction):")
    print("  - Panel 1: Weather forecast evolution")
    print("  - Panel 2: Partition dynamics evolution")
    print("  - Panel 3: Forecast validation")
    print("  - Panel 4: Skill metrics and performance")


if __name__ == "__main__":
    main()
