"""
Generate 4 visualization panels for celestial-topological positioning experiments.
Each panel: white background, 4 charts in a row (1+ 3D), 300 DPI PNG.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Load results
with open(r"c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\celestial_positioning_results.json") as f:
    results = json.load(f)


def panel_1_weather_latitude():
    """Panel 1: Weather-Based Latitude Determination (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_1_weather_latitude"]
    latitudes = []
    mean_errors = []
    std_errors = []
    position_errors_km = []

    for lat_str in sorted([k for k in data.keys() if k != "0"], key=lambda x: float(x)):
        lat = float(lat_str)
        latitudes.append(lat)
        mean_errors.append(data[lat_str]["mean_latitude_error_deg"])
        std_errors.append(data[lat_str]["std_latitude_error_deg"])
        position_errors_km.append(data[lat_str]["position_error_km"])

    latitudes = np.array(latitudes)
    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)
    position_errors_km = np.array(position_errors_km)

    # Chart 1: Latitude Error vs True Latitude
    ax1 = fig.add_subplot(141)
    ax1.errorbar(latitudes, mean_errors, yerr=std_errors, fmt='o', markersize=8, capsize=5,
                 color='#1f77b4', ecolor='#ff7f0e', linewidth=2, label='Mean ± Std')
    ax1.plot(latitudes, mean_errors, 'b--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('True Latitude (degrees)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Latitude Error (degrees)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-90, 90)

    # Chart 2: Position Error vs Latitude (bar chart)
    ax2 = fig.add_subplot(142)
    colors = ['#2ca02c' if abs(lat) < 45 else '#d62728' for lat in latitudes]
    bars = ax2.bar(range(len(latitudes)), position_errors_km, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(latitudes)))
    ax2.set_xticklabels([f'{int(lat)}°' for lat in latitudes], rotation=45)
    ax2.set_ylabel('Position Error (km)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Latitude', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Chart 3: 3D scatter (latitude, error, position_error)
    ax3 = fig.add_subplot(143, projection='3d')
    scatter = ax3.scatter(latitudes, mean_errors, position_errors_km, c=position_errors_km,
                         cmap='RdYlGn_r', s=100, alpha=0.8, edgecolors='black', linewidth=1)
    ax3.set_xlabel('Latitude (deg)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Latitude Error (deg)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Position Error (km)', fontsize=10, fontweight='bold')
    ax3.view_init(elev=20, azim=45)
    fig.colorbar(scatter, ax=ax3, label='Position Error (km)', shrink=0.6)

    # Chart 4: Coriolis Frequency vs Latitude
    ax4 = fig.add_subplot(144)
    EARTH_OMEGA = 7.2921e-5  # rad/s
    coriolis_freqs = 2.0 * EARTH_OMEGA * np.sin(np.radians(latitudes)) * 86400  # Convert to Hz, then oscillations/day
    ax4.plot(latitudes, coriolis_freqs * 1e5, 'o-', markersize=9, linewidth=2.5,
             color='#9467bd', markerfacecolor='#bcbd22', markeredgecolor='black', markeredgewidth=1.5)
    ax4.fill_between(latitudes, coriolis_freqs * 1e5, alpha=0.2, color='#9467bd')
    ax4.set_xlabel('Latitude (degrees)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Coriolis Frequency (×10⁻⁵ rad/s)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Experiment 1: Weather-Based Latitude Determination', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_1_weather_latitude.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 1 saved: panel_1_weather_latitude.png")
    plt.close()


def panel_2_infrastructure_triangulation():
    """Panel 2: Infrastructure Rank Triangulation (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_2_infrastructure_triangulation"]

    configs = sorted(data.keys())
    n_sensors = [data[c]["n_sensors"] for c in configs]
    k_freqs = [data[c]["k_frequencies"] for c in configs]
    errors = [data[c]["position_error_m"] for c in configs]
    expected_ranks = [data[c]["expected_rank"] for c in configs]

    # Chart 1: Position Error vs Configuration
    ax1 = fig.add_subplot(141)
    x_pos = np.arange(len(configs))
    colors_error = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    ax1.bar(x_pos, errors, color=colors_error, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'N={n}\nK={k}' for n, k in zip(n_sensors, k_freqs)], fontsize=10)
    ax1.set_ylabel('Position Error (m)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(errors) * 1.2)

    # Chart 2: Sensor Count vs Frequency Count (scatter)
    ax2 = fig.add_subplot(142)
    scatter2 = ax2.scatter(n_sensors, k_freqs, s=np.array(errors)*5, c=errors, cmap='plasma',
                          alpha=0.7, edgecolors='black', linewidth=2)
    ax2.set_xlabel('Number of Sensors (N)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Frequencies (K)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Error (m)')
    # Add diagonal line (N = K)
    max_val = max(max(n_sensors), max(k_freqs))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2)
    ax2.text(max_val*0.7, max_val*0.8, 'N=K', color='red', fontsize=10, fontweight='bold')

    # Chart 3: 3D surface (N, K, error)
    ax3 = fig.add_subplot(143, projection='3d')
    n_array = np.array(n_sensors)
    k_array = np.array(k_freqs)
    e_array = np.array(errors)
    ax3.scatter(n_array, k_array, e_array, c=e_array, cmap='coolwarm', s=150, alpha=0.8,
               edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Sensors (N)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequencies (K)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Position Error (m)', fontsize=10, fontweight='bold')
    ax3.view_init(elev=25, azim=45)

    # Chart 4: Expected vs Actual Rank
    ax4 = fig.add_subplot(144)
    x_pos = np.arange(len(configs))
    width = 0.35
    ax4.bar(x_pos - width/2, expected_ranks, width, label='Expected (min(N,K))',
           color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
    actual_ranks = [data[c]["true_rank"] for c in configs]
    ax4.bar(x_pos + width/2, actual_ranks, width, label='Actual (measured)',
           color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{c}' for c in configs], fontsize=9)
    ax4.set_ylabel('Transfer-Matrix Rank', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Experiment 2: Infrastructure Rank Triangulation', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_2_infrastructure_rank.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 2 saved: panel_2_infrastructure_rank.png")
    plt.close()


def panel_3_celestial_positioning():
    """Panel 3: Celestial Harmonic Positioning (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_3_celestial_positioning"]

    n_sources_list = sorted([int(k) for k in data.keys()])
    means = [data[str(n)]["mean_position_error_m"] for n in n_sources_list]
    stds = [data[str(n)]["std_position_error_m"] for n in n_sources_list]
    mins = [data[str(n)]["min_error_m"] for n in n_sources_list]
    maxs = [data[str(n)]["max_error_m"] for n in n_sources_list]

    # Convert to km for readability
    means_km = np.array(means) / 1000
    stds_km = np.array(stds) / 1000
    mins_km = np.array(mins) / 1000
    maxs_km = np.array(maxs) / 1000

    # Chart 1: Mean Error vs Number of Sources
    ax1 = fig.add_subplot(141)
    ax1.errorbar(n_sources_list, means_km, yerr=stds_km, fmt='o-', markersize=10, capsize=7,
                 color='#d62728', ecolor='#2ca02c', linewidth=2.5, label='Mean ± Std')
    ax1.set_xlabel('Number of Celestial Sources', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Position Error (km)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_sources_list)

    # Chart 2: Error Distribution (box plot)
    ax2 = fig.add_subplot(142)
    error_distributions = []
    for n in n_sources_list:
        errors = np.array(data[str(n)]["all_errors_m"]) / 1000  # Convert to km
        error_distributions.append(errors)

    bp = ax2.boxplot(error_distributions, labels=n_sources_list, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(n_sources_list)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, color='black')
    ax2.set_xlabel('Number of Sources', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Position Error (km)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Chart 3: 3D scatter (sources, mean, std, max)
    ax3 = fig.add_subplot(143, projection='3d')
    colors_3d = plt.cm.plasma(np.linspace(0, 1, len(n_sources_list)))
    ax3.scatter(n_sources_list, means_km, stds_km, c=range(len(n_sources_list)), cmap='plasma',
               s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Num Sources', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Mean Error (km)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Std Error (km)', fontsize=10, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # Chart 4: Min/Max Error Range
    ax4 = fig.add_subplot(144)
    ax4.fill_between(n_sources_list, mins_km, maxs_km, alpha=0.3, color='#1f77b4', label='Min-Max Range')
    ax4.plot(n_sources_list, means_km, 'o-', markersize=10, linewidth=2.5,
            color='#ff7f0e', markerfacecolor='#ffbb78', markeredgecolor='black', markeredgewidth=1.5,
            label='Mean')
    ax4.set_xlabel('Number of Celestial Sources', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Position Error (km)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.set_xticks(n_sources_list)

    plt.suptitle('Experiment 3: Celestial Harmonic Positioning', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_3_celestial_positioning.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 3 saved: panel_3_celestial_positioning.png")
    plt.close()


def panel_4_multi_regime_fusion():
    """Panel 4: Multi-Regime Fusion Accuracy Improvement (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_4_multi_regime_fusion"]
    dist_data = data["distribution_data"]

    weather_errors = np.array(dist_data["weather_errors"]) / 1000  # Convert to km
    infra_errors = np.array(dist_data["infrastructure_errors"]) / 1000
    celestial_errors = np.array(dist_data["celestial_errors"]) / 1000
    fused_errors = np.array(dist_data["fused_errors"]) / 1000

    # Chart 1: Mean Error Comparison (bar chart)
    ax1 = fig.add_subplot(141)
    regimes = ['Weather', 'Infrastructure', 'Celestial', 'Fused']
    mean_errors = [
        np.mean(weather_errors),
        np.mean(infra_errors),
        np.mean(celestial_errors),
        np.mean(fused_errors)
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax1.bar(regimes, mean_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for i, (bar, val) in enumerate(zip(bars, mean_errors)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Mean Position Error (km)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(mean_errors) * 1.3)

    # Chart 2: Distribution Comparison (violin plot)
    ax2 = fig.add_subplot(142)
    parts = ax2.violinplot([weather_errors, infra_errors, celestial_errors, fused_errors],
                           positions=[1, 2, 3, 4], widths=0.7, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#8dd3c7')
        pc.set_alpha(0.7)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(regimes, fontsize=10)
    ax2.set_ylabel('Position Error (km)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Chart 3: 3D scatter (regime index, trial number, error)
    ax3 = fig.add_subplot(143, projection='3d')
    n_trials = len(weather_errors)
    trials = np.arange(n_trials)

    # Create data for 3D scatter
    regime_idx_all = np.concatenate([np.ones(n_trials)*1, np.ones(n_trials)*2,
                                      np.ones(n_trials)*3, np.ones(n_trials)*4])
    trials_all = np.concatenate([trials, trials, trials, trials])
    errors_all = np.concatenate([weather_errors, infra_errors, celestial_errors, fused_errors])

    scatter = ax3.scatter(regime_idx_all, trials_all, errors_all, c=errors_all, cmap='RdYlGn_r',
                         s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Regime', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Trial', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Error (km)', fontsize=10, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # Chart 4: Cumulative Distribution Function
    ax4 = fig.add_subplot(144)
    for errors, regime, color in zip([weather_errors, infra_errors, celestial_errors, fused_errors],
                                      regimes, colors):
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
        ax4.plot(sorted_errors, cdf, linewidth=2.5, label=regime, color=color, marker='o',
                markersize=4, markeredgecolor='black', markeredgewidth=0.5)

    ax4.set_xlabel('Position Error (km)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.set_xlim(0, np.max(weather_errors) * 1.1)

    plt.suptitle('Experiment 4: Multi-Regime Fusion Accuracy', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_4_fusion_accuracy.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 4 saved: panel_4_fusion_accuracy.png")
    plt.close()


def main():
    """Generate all four panels."""
    print("Generating celestial positioning visualization panels...")
    panel_1_weather_latitude()
    panel_2_infrastructure_triangulation()
    panel_3_celestial_positioning()
    panel_4_multi_regime_fusion()
    print("\n[COMPLETE] All 4 panels generated successfully")
    print("Output directory: c:\\Users\\kunda\\Documents\\physics\\sighthound\\moriarty\\validation\\")


if __name__ == "__main__":
    main()
