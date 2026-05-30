"""
Generate publication-quality panels for Phase-Locked Messaging validation results.

Produces 4 300-DPI PNG panels, each with 4 complementary charts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 100

def load_results(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def create_panel_1(results: dict, output_path: str):
    """Panel 1: Phase Lock Convergence"""
    print("Generating Panel 1: Phase Lock Convergence...")

    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    data = results['phase_lock_convergence']

    # Chart A: Mean margin over depth
    ax_a = fig.add_subplot(gs[0, 0])
    depths = data['depths']
    all_margins = np.array([trial['margins'] for trial in data['trials']])
    mean_margins = np.mean(all_margins, axis=0)
    std_margins = np.std(all_margins, axis=0)

    ax_a.fill_between(depths, mean_margins - std_margins, mean_margins + std_margins,
                      alpha=0.3, color='#c85a54')
    ax_a.plot(depths, mean_margins, color='#c85a54', linewidth=2.5, label='Mean margin')
    ax_a.set_xlabel('Composition Depth (n)', fontsize=10, fontweight='bold')
    ax_a.set_ylabel('Posterior Margin', fontsize=10, fontweight='bold')
    ax_a.set_title('(A) Convergence Trajectory', fontsize=11, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend()

    # Chart B: Log-scale growth
    ax_b = fig.add_subplot(gs[0, 1])
    for i, trial in enumerate(data['trials'][:5]):
        ax_b.semilogy(depths, trial['margins'], alpha=0.5, linewidth=1)
    ax_b.semilogy(depths, mean_margins, color='#c85a54', linewidth=3, label='Mean')
    ax_b.set_xlabel('Composition Depth (n)', fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Posterior Margin (log scale)', fontsize=10, fontweight='bold')
    ax_b.set_title('(B) Exponential Growth (log scale)', fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    ax_b.legend()

    # Chart C: 3D surface (margin vs depth vs trial)
    ax_c = fig.add_subplot(gs[1, 0], projection='3d')
    X = np.arange(len(data['trials'][:10]))
    depth_indices = np.arange(0, len(depths), 5)
    X_mesh, Y_mesh = np.meshgrid(X, depth_indices)
    Z_mesh = np.array([[all_margins[i, j] for i in range(10)] for j in depth_indices])

    surf = ax_c.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='RdYlGn', alpha=0.8)
    ax_c.set_xlabel('Trial', fontsize=9)
    ax_c.set_ylabel('Depth (×5)', fontsize=9)
    ax_c.set_zlabel('Margin', fontsize=9)
    ax_c.set_title('(C) 3D Phase Lock Surface', fontsize=11, fontweight='bold')

    # Chart D: Distribution of margins at key depths
    ax_d = fig.add_subplot(gs[1, 1])
    depths_to_show = [9, 19, 29, 49]
    margin_data = [all_margins[:, d] for d in depths_to_show]
    bp = ax_d.boxplot(margin_data, labels=[f'n={d+1}' for d in depths_to_show],
                      patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#c85a54')
        patch.set_alpha(0.7)
    ax_d.set_ylabel('Posterior Margin', fontsize=10, fontweight='bold')
    ax_d.set_title('(D) Margin Distribution by Depth', fontsize=11, fontweight='bold')
    ax_d.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Panel 1: Phase-Lock Convergence: Posterior margin increases monotonically with evidence depth',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('white')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Panel 1 saved to {output_path}")


def create_panel_2(results: dict, output_path: str):
    """Panel 2: State Change Detection"""
    print("Generating Panel 2: State Change Detection...")

    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    data = results['state_change_detection']

    # Chart A: Scatter - Detection Ratio vs Margin Before
    ax_a = fig.add_subplot(gs[0, 0])
    margins_before = [t['margin_before'] for t in data['trials']]
    detection_ratios = [t['detection_ratio'] for t in data['trials']]
    colors = ['#c85a54' if t['detected'] else '#999' for t in data['trials']]

    ax_a.scatter(margins_before, detection_ratios, c=colors, alpha=0.6, s=60)
    ax_a.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Detection threshold')
    ax_a.set_xlabel('Posterior Margin Before Change', fontsize=10, fontweight='bold')
    ax_a.set_ylabel('State Change Detection Ratio', fontsize=10, fontweight='bold')
    ax_a.set_title('(A) State Change Detectability', fontsize=11, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend()

    # Chart B: Detection rate bar chart
    ax_b = fig.add_subplot(gs[0, 1])
    detection_count = sum([1 for t in data['trials'] if t['detected']])
    detection_rate = detection_count / len(data['trials']) * 100

    bars = ax_b.bar(['Detected', 'Undetected'],
                    [detection_rate, 100 - detection_rate],
                    color=['#2d5016', '#ddd'], width=0.6)
    ax_b.set_ylabel('Percentage [%]', fontsize=10, fontweight='bold')
    ax_b.set_ylim([0, 105])
    ax_b.set_title('(B) State Change Detection Rate', fontsize=11, fontweight='bold')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Chart C: 3D scatter - Margin Before vs After vs Detection Ratio
    ax_c = fig.add_subplot(gs[1, 0], projection='3d')
    margins_after = [t['margin_after'] for t in data['trials']]

    scatter = ax_c.scatter(margins_before, margins_after, detection_ratios,
                          c=detection_ratios, cmap='RdYlGn', s=60, alpha=0.7)
    ax_c.set_xlabel('Margin Before', fontsize=9)
    ax_c.set_ylabel('Margin After', fontsize=9)
    ax_c.set_zlabel('Detection Ratio', fontsize=9)
    ax_c.set_title('(C) 3D State Change Landscape', fontsize=11, fontweight='bold')

    # Chart D: Histogram of detection ratios
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.hist(detection_ratios, bins=20, color='#c85a54', alpha=0.7, edgecolor='black')
    ax_d.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Detection threshold')
    ax_d.set_xlabel('Detection Ratio', fontsize=10, fontweight='bold')
    ax_d.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_d.set_title('(D) Detection Ratio Distribution', fontsize=11, fontweight='bold')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Panel 2: State Change Detection: Observable state transitions enable message transmission at locked positions',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('white')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Panel 2 saved to {output_path}")


def create_panel_3(results: dict, output_path: str):
    """Panel 3: Replay Attack Detection"""
    print("Generating Panel 3: Replay Attack Detection...")

    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    data = results['replay_attack_detection']

    # Chart A: Detection rate vs epoch delta
    ax_a = fig.add_subplot(gs[0, 0])
    deltas = list(range(1, 21))
    detection_rates = [data['summary'][f'delta_{d}']['empirical_detection_rate'] for d in deltas]
    theoretical_rates = [data['summary'][f'delta_{d}']['theoretical_detection_rate'] for d in deltas]

    ax_a.plot(deltas, detection_rates, 'o-', color='#c85a54', linewidth=2.5, label='Empirical', markersize=6)
    ax_a.plot(deltas, theoretical_rates, '--', color='#2d5016', linewidth=2, label='Theoretical (1-exp(-0.6δ))')
    ax_a.fill_between(deltas, detection_rates, theoretical_rates, alpha=0.2, color='#d4a574')
    ax_a.set_xlabel('Epoch Delta (δ)', fontsize=10, fontweight='bold')
    ax_a.set_ylabel('Detection Rate', fontsize=10, fontweight='bold')
    ax_a.set_ylim([0, 1.05])
    ax_a.set_title('(A) Replay Detection vs Epoch Offset', fontsize=11, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend()

    # Chart B: Bar chart - Color coded by detection success
    ax_b = fig.add_subplot(gs[0, 1])
    colors_bars = ['#2d5016' if r > 0.8 else '#d4a574' if r > 0.5 else '#c85a54' for r in detection_rates]
    ax_b.bar(deltas, [r * 100 for r in detection_rates], color=colors_bars, width=0.8)
    ax_b.set_xlabel('Epoch Delta (δ)', fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Detection Rate [%]', fontsize=10, fontweight='bold')
    ax_b.set_ylim([0, 105])
    ax_b.set_title('(B) Detection Success by Delta', fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y')

    # Chart C: 3D surface - Delta vs Evasion Probability
    ax_c = fig.add_subplot(gs[1, 0], projection='3d')
    delta_range = np.arange(1, 21)
    evasion_probs = [1.0 - (1.0 - np.exp(-0.6 * d)) for d in delta_range]

    trials_subset = [d-1 for d in [1, 5, 10, 15, 20]]
    X = np.arange(len(trials_subset))
    Y = delta_range
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z_mesh = np.array([[evasion_probs[d-1] for _ in X] for d in Y])

    surf = ax_c.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='RdYlGn_r', alpha=0.8)
    ax_c.set_xlabel('Trial', fontsize=9)
    ax_c.set_ylabel('Delta', fontsize=9)
    ax_c.set_zlabel('Evasion Prob', fontsize=9)
    ax_c.set_title('(C) 3D Evasion Probability', fontsize=11, fontweight='bold')

    # Chart D: Histogram of evasion probabilities
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.hist(evasion_probs, bins=15, color='#c85a54', alpha=0.7, edgecolor='black')
    ax_d.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax_d.set_xlabel('Evasion Probability', fontsize=10, fontweight='bold')
    ax_d.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_d.set_title('(D) Evasion Probability Distribution', fontsize=11, fontweight='bold')
    ax_d.legend()
    ax_d.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Panel 3: Replay Attack Detection: Monotone epoch counters detect replayed evidence with >99% certainty at δ≥15',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('white')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Panel 3 saved to {output_path}")


def create_panel_4(results: dict, output_path: str):
    """Panel 4: Multi-Channel Advantage"""
    print("Generating Panel 4: Multi-Channel Advantage...")

    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    data = results['multichannel_advantage']
    channels = data['channel_counts']

    # Chart A: Scatter - Margin vs Channels
    ax_a = fig.add_subplot(gs[0, 0])
    all_trials = data['trials']
    for trial in all_trials[:10]:
        margins = [t['margin'] for t in trial]
        ax_a.plot(channels, margins, alpha=0.3, color='#c85a54')

    mean_margins = [np.mean([trial[i]['margin'] for trial in all_trials]) for i in range(len(channels))]
    ax_a.plot(channels, mean_margins, 'o-', color='#2d5016', linewidth=2.5, markersize=8, label='Mean')
    ax_a.set_xlabel('Number of Channels (d)', fontsize=10, fontweight='bold')
    ax_a.set_ylabel('Posterior Margin', fontsize=10, fontweight='bold')
    ax_a.set_title('(A) Margin Improvement with Channels', fontsize=11, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.legend()

    # Chart B: Bar chart - Mean margin by channel count
    ax_b = fig.add_subplot(gs[0, 1])
    std_margins = [np.std([trial[i]['margin'] for trial in all_trials]) for i in range(len(channels))]
    bars = ax_b.bar(channels, mean_margins, yerr=std_margins, capsize=5, color='#c85a54', alpha=0.7)
    ax_b.set_xlabel('Number of Channels (d)', fontsize=10, fontweight='bold')
    ax_b.set_ylabel('Mean Posterior Margin', fontsize=10, fontweight='bold')
    ax_b.set_title('(B) Average Margin by Channel Count', fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y')

    for i, (bar, val) in enumerate(zip(bars, mean_margins)):
        ax_b.text(bar.get_x() + bar.get_width()/2., val + std_margins[i] + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Chart C: 3D surface - Channels vs Trial vs Margin
    ax_c = fig.add_subplot(gs[1, 0], projection='3d')
    X = np.arange(len(channels))
    Y = np.arange(min(10, len(all_trials)))
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    Z_mesh = np.array([[all_trials[j][i]['margin'] for i in range(len(channels))]
                       for j in range(min(10, len(all_trials)))])

    surf = ax_c.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis', alpha=0.8)
    ax_c.set_xlabel('Channels', fontsize=9)
    ax_c.set_ylabel('Trial', fontsize=9)
    ax_c.set_zlabel('Margin', fontsize=9)
    ax_c.set_title('(C) 3D Multi-Channel Surface', fontsize=11, fontweight='bold')
    plt.colorbar(surf, ax=ax_c, shrink=0.5)

    # Chart D: Composition floor comparison
    ax_d = fig.add_subplot(gs[1, 1])
    floors = [np.mean([trial[i]['composition_floor'] for trial in all_trials]) for i in range(len(channels))]
    ax_d.semilogy(channels, mean_margins, 'o-', color='#c85a54', linewidth=2.5, label='Empirical margin', markersize=8)
    ax_d.semilogy(channels, floors, 's--', color='#2d5016', linewidth=2, label='Composition floor 1/Γ(n,d)', markersize=6)
    ax_d.set_xlabel('Number of Channels (d)', fontsize=10, fontweight='bold')
    ax_d.set_ylabel('Value (log scale)', fontsize=10, fontweight='bold')
    ax_d.set_title('(D) Margin vs Composition Floor', fontsize=11, fontweight='bold')
    ax_d.grid(True, alpha=0.3)
    ax_d.legend()

    fig.suptitle('Panel 4: Multi-Channel Advantage: Independent measurement channels exponentially sharpen position certainty',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('white')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Panel 4 saved to {output_path}")


def generate_all_panels(results: dict, output_dir: str = '.'):
    """Generate all 4 panels."""
    print("\n" + "="*60)
    print("GENERATING PUBLICATION-QUALITY PANELS")
    print("="*60 + "\n")

    create_panel_1(results, f'{output_dir}/panel_1_phase_lock_convergence.png')
    create_panel_2(results, f'{output_dir}/panel_2_state_change_detection.png')
    create_panel_3(results, f'{output_dir}/panel_3_replay_attack_detection.png')
    create_panel_4(results, f'{output_dir}/panel_4_multichannel_advantage.png')

    print("\n" + "="*60)
    print("ALL PANELS GENERATED SUCCESSFULLY")
    print("="*60 + "\n")


if __name__ == "__main__":
    results = load_results('phase_locked_messaging_results.json')
    generate_all_panels(results, '.')
