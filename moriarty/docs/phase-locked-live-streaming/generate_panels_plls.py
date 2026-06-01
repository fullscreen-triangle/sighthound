"""
Panel generation for Phase-Locked Live Streaming validation results.

Creates 4 publication-quality panels (300 DPI, white background, 4 charts per panel):
- Panel 1: Instantaneous synchronization and lossless delivery
- Panel 2: Bandwidth scaling efficiency
- Panel 3: Variable frame rate support (3D visualization)
- Panel 4: Topology comparison (star vs chain)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import os

# Publication-quality settings
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def load_results(filepath):
    """Load validation results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_panel_1_sync_and_delivery(results, output_path):
    """Panel 1: Instantaneous synchronization and lossless frame delivery."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), facecolor='white')
    fig.suptitle('Phase-Locked Live Streaming: Synchronization & Delivery', fontsize=11, fontweight='bold')

    # Chart 1: Synchronization delay by receiver count
    sync_data = results['instantaneous_synchronization']
    sync_trials = sync_data['trials']

    receiver_counts = [t['num_receivers'] for t in sync_trials]
    max_delays = [t['max_observer_delay_ms'] for t in sync_trials]

    sorted_indices = np.argsort(receiver_counts)
    receiver_counts_sorted = [receiver_counts[i] for i in sorted_indices]
    max_delays_sorted = [max_delays[i] for i in sorted_indices]

    axes[0].scatter(receiver_counts_sorted, max_delays_sorted, alpha=0.6, s=30, color='#2E86AB', edgecolors='#06A77D', linewidth=0.5)
    axes[0].axhline(y=0, color='#D62828', linestyle='--', linewidth=1.5, label='PLF Target')
    axes[0].set_xlabel('Number of Receivers')
    axes[0].set_ylabel('Max Delay (ms)')
    axes[0].set_ylim([-0.01, 0.01])
    axes[0].set_title('Instantaneous Synchronization', fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, linestyle=':')

    # Chart 2: Speedup vs traditional streaming
    speedups = [t['speedup_vs_traditional'] for t in sync_trials]
    axes[1].hist(speedups, bins=15, color='#F18F01', alpha=0.7, edgecolor='#D62828', linewidth=1)
    axes[1].axvline(x=np.mean(speedups), color='#06A77D', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(speedups):.0f}×')
    axes[1].set_xlabel('Speedup Factor (×)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('PLF vs Traditional Latency', fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle=':')

    # Chart 3: Lossless delivery verification
    delivery_data = results['lossless_frame_delivery']
    delivery_trials = delivery_data['trials']

    chain_lengths = [t['chain_length'] for t in delivery_trials]
    plls_loss_rates = [t['plls_loss_rate'] for t in delivery_trials]
    traditional_loss_rates = [t['traditional_loss_rate'] for t in delivery_trials]

    sorted_indices_chain = np.argsort(chain_lengths)
    chain_lengths_sorted = [chain_lengths[i] for i in sorted_indices_chain]
    plls_loss_sorted = [plls_loss_rates[i] for i in sorted_indices_chain]
    trad_loss_sorted = [traditional_loss_rates[i] for i in sorted_indices_chain]

    axes[2].plot(chain_lengths_sorted, plls_loss_sorted, marker='o', markersize=5, linewidth=1.5,
                color='#06A77D', label='PLF')
    axes[2].plot(chain_lengths_sorted, trad_loss_sorted, marker='s', markersize=5, linewidth=1.5,
                color='#D62828', label='Traditional')
    axes[2].set_xlabel('Chain Length (hops)')
    axes[2].set_ylabel('Frame Loss Rate')
    axes[2].set_title('Lossless Delivery Across Topology', fontsize=9)
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3, linestyle=':')
    axes[2].set_yscale('log')

    # Chart 4: Reliability gain factor
    reliability_gains = [t['reliability_gain'] for t in delivery_trials]
    mean_gain = np.mean(reliability_gains)
    axes[3].barh(['Reliability\nGain'], [mean_gain], color='#A23B72', alpha=0.7, edgecolor='#A23B72', linewidth=1)
    axes[3].set_xlabel('Gain Factor (×)')
    axes[3].set_title('PLF Reliability Improvement', fontsize=9)
    axes[3].grid(True, alpha=0.3, axis='x', linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Panel 1 saved: {output_path}")

def create_panel_2_bandwidth_scaling(results, output_path):
    """Panel 2: Bandwidth scaling efficiency."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), facecolor='white')
    fig.suptitle('Phase-Locked Live Streaming: Bandwidth Efficiency', fontsize=11, fontweight='bold')

    scale_data = results['bandwidth_independent_scaling']
    scale_trials = scale_data['trials']

    # Chart 1: Bandwidth vs receiver count
    num_receivers = [t['num_receivers'] for t in scale_trials]
    plls_bandwidth = [t['plls_bandwidth_mbps'] for t in scale_trials]
    trad_bandwidth = [t['traditional_bandwidth_mbps'] for t in scale_trials]

    sorted_indices = np.argsort(num_receivers)
    receivers_sorted = [num_receivers[i] for i in sorted_indices]
    plls_sorted = [plls_bandwidth[i] for i in sorted_indices]
    trad_sorted = [trad_bandwidth[i] for i in sorted_indices]

    axes[0].plot(receivers_sorted, plls_sorted, marker='o', markersize=6, linewidth=1.5,
                color='#06A77D', label='PLF (Independent)', markerfacecolor='#06A77D')
    axes[0].plot(receivers_sorted, trad_sorted, marker='s', markersize=6, linewidth=1.5,
                color='#D62828', label='Traditional (Scaled)', markerfacecolor='#D62828')
    axes[0].set_xlabel('Number of Receivers')
    axes[0].set_ylabel('Bandwidth (Mbps)')
    axes[0].set_yscale('log')
    axes[0].set_title('Bandwidth Scaling', fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, linestyle=':', which='both')

    # Chart 2: Bandwidth savings distribution
    savings = [t['bandwidth_savings'] for t in scale_trials]
    axes[1].hist(savings, bins=15, color='#2E86AB', alpha=0.7, edgecolor='#2E86AB', linewidth=1)
    axes[1].axvline(x=np.mean(savings), color='#F18F01', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(savings):.1f}×')
    axes[1].set_xlabel('Bandwidth Savings (×)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cost Reduction Distribution', fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle=':')

    # Chart 3: Frame resolutions impact
    resolutions = [t['frame_resolution'] for t in scale_trials]
    unique_resolutions = list(set(resolutions))
    res_order = ['1080p', '2160p', '4320p']
    unique_resolutions_ordered = [r for r in res_order if r in unique_resolutions]

    bandwidth_by_res_plls = []
    bandwidth_by_res_trad = []
    for res in unique_resolutions_ordered:
        mask = [r == res for r in resolutions]
        if any(mask):
            bandwidth_by_res_plls.append(np.mean([plls_bandwidth[i] for i in range(len(scale_trials)) if mask[i]]))
            bandwidth_by_res_trad.append(np.mean([trad_bandwidth[i] for i in range(len(scale_trials)) if mask[i]]))
        else:
            bandwidth_by_res_plls.append(0)
            bandwidth_by_res_trad.append(0)

    x_pos = np.arange(len(unique_resolutions_ordered))
    width = 0.35
    axes[2].bar(x_pos - width/2, bandwidth_by_res_plls, width, label='PLF', color='#06A77D', alpha=0.7, edgecolor='#06A77D', linewidth=1)
    axes[2].bar(x_pos + width/2, bandwidth_by_res_trad, width, label='Traditional', color='#D62828', alpha=0.7, edgecolor='#D62828', linewidth=1)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(unique_resolutions_ordered)
    axes[2].set_ylabel('Bandwidth (Mbps)')
    axes[2].set_title('Resolution Impact', fontsize=9)
    axes[2].set_yscale('log')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3, axis='y', linestyle=':', which='both')

    # Chart 4: Independence verification
    independent_trials = [t['bandwidth_independent'] for t in scale_trials]
    independent_rate = np.mean(independent_trials) * 100
    axes[3].pie([independent_rate, 100 - independent_rate],
                labels=['Independent', 'Not Independent'],
                colors=['#06A77D', '#D62828'], autopct='%1.1f%%',
                textprops={'fontsize': 8})
    axes[3].set_title('Bandwidth Independence', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Panel 2 saved: {output_path}")

def create_panel_3_frame_rate_3d(results, output_path):
    """Panel 3: Variable frame rate support with 3D visualization."""
    fig = plt.figure(figsize=(16, 3.5), facecolor='white')

    fr_data = results['variable_frame_rate_support']
    fr_trials = fr_data['trials']

    # Chart 1: Frame rate support distribution
    ax1 = plt.subplot(1, 4, 1)
    fps_values = [t['requested_fps'] for t in fr_trials]
    plls_support = [t['plls_supported'] for t in fr_trials]
    trad_support = [t['traditional_supported'] for t in fr_trials]

    fps_unique = sorted(set(fps_values))
    plls_support_rate = []
    trad_support_rate = []

    for fps in fps_unique:
        mask = [f == fps for f in fps_values]
        plls_support_rate.append(np.mean([s for s, m in zip(plls_support, mask) if m]) * 100 if any(mask) else 0)
        trad_support_rate.append(np.mean([s for s, m in zip(trad_support, mask) if m]) * 100 if any(mask) else 0)

    x_pos = np.arange(len(fps_unique))
    width = 0.35
    ax1.bar(x_pos - width/2, plls_support_rate, width, label='PLF', color='#06A77D', alpha=0.7, edgecolor='#06A77D', linewidth=1)
    ax1.bar(x_pos + width/2, trad_support_rate, width, label='Traditional', color='#D62828', alpha=0.7, edgecolor='#D62828', linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(f) for f in fps_unique], rotation=45, fontsize=7)
    ax1.set_ylabel('Support Rate (%)')
    ax1.set_ylim([0, 110])
    ax1.set_title('Frame Rate Support', fontsize=9)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Chart 2: Protocol overhead
    ax2 = plt.subplot(1, 4, 2)
    plls_overhead = [t['plls_overhead'] for t in fr_trials]
    trad_overhead = [t['traditional_overhead_percent'] for t in fr_trials]

    overhead_data = {
        'PLF': np.mean(plls_overhead),
        'Traditional': np.mean(trad_overhead)
    }
    colors = ['#06A77D', '#D62828']
    ax2.bar(overhead_data.keys(), overhead_data.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Protocol Overhead (%)')
    ax2.set_title('Frame Rate Conversion Overhead', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Chart 3: 3D - Frame rate vs latency vs support
    ax3 = plt.subplot(1, 4, 3, projection='3d')

    frame_rates = np.array(fps_values[:50])  # Use first 50 trials
    requested_overhead = np.array([t['traditional_overhead_percent'] for t in fr_trials[:50]])
    supported_mask = np.array([int(t['plls_supported']) for t in fr_trials[:50]])

    ax3.scatter(frame_rates, requested_overhead, supported_mask, c=supported_mask, cmap='RdYlGn',
               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Frame Rate (FPS)', fontsize=8)
    ax3.set_ylabel('Overhead (%)', fontsize=8)
    ax3.set_zlabel('PLF Support', fontsize=8)
    ax3.set_title('Frame Rate Support Space', fontsize=9)

    # Chart 4: Variable rate capability
    ax4 = plt.subplot(1, 4, 4)
    variable_capable = [t['variable_rate_capable'] for t in fr_trials]
    variable_rate = np.mean(variable_capable) * 100

    ax4.barh(['Variable Rate\nCapability'], [variable_rate], color='#F18F01', alpha=0.7, edgecolor='#F18F01', linewidth=1)
    ax4.set_xlim([0, 110])
    ax4.set_xlabel('Capability (%)')
    ax4.set_title('Arbitrary Frame Rate Support', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x', linestyle=':')

    fig.suptitle('Phase-Locked Live Streaming: Variable Frame Rate Support', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Panel 3 saved: {output_path}")

def create_panel_4_topology_comparison(results, output_path):
    """Panel 4: Star vs chain topology comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), facecolor='white')
    fig.suptitle('Phase-Locked Live Streaming: Topology Performance', fontsize=11, fontweight='bold')

    # Chart 1: Star topology receiver coherence
    star_data = results['star_topology_coherence']
    star_trials = star_data['trials']

    receiver_counts_star = [t['num_receivers'] for t in star_trials]
    coherence_strengths = [t['mean_coherence_strength'] for t in star_trials]
    all_coherent = [t['all_receivers_coherent'] for t in star_trials]

    sorted_indices = np.argsort(receiver_counts_star)
    receiver_counts_sorted = [receiver_counts_star[i] for i in sorted_indices]
    coherence_sorted = [coherence_strengths[i] for i in sorted_indices]

    axes[0].scatter(receiver_counts_sorted, coherence_sorted, alpha=0.6, s=40, color='#06A77D', edgecolors='#2E86AB', linewidth=0.5)
    axes[0].axhline(y=1.0, color='#F18F01', linestyle='--', linewidth=1.5, label='Perfect Coherence')
    axes[0].set_ylim([0.9, 1.01])
    axes[0].set_xlabel('Number of Receivers')
    axes[0].set_ylabel('Mean Coherence Strength')
    axes[0].set_title('Star Topology Coherence', fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, linestyle=':')

    # Chart 2: Star topology scalability
    axes[1].hist([c for c in receiver_counts_star], bins=12, color='#2E86AB', alpha=0.7, edgecolor='#2E86AB', linewidth=1)
    axes[1].set_xlabel('Receivers per Trial')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Star Topology Scalability', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle=':')

    # Chart 3: Chain topology propagation delay
    chain_data = results['chain_topology_propagation']
    chain_trials = chain_data['trials']

    chain_lengths = [t['chain_length'] for t in chain_trials]
    prop_delays = [t['total_propagation_delay_ms'] for t in chain_trials]
    synchronized = [t['frames_synchronized'] for t in chain_trials]

    sorted_chain_indices = np.argsort(chain_lengths)
    chain_lengths_sorted = [chain_lengths[i] for i in sorted_chain_indices]
    prop_delays_sorted = [prop_delays[i] for i in sorted_chain_indices]

    axes[2].plot(chain_lengths_sorted, prop_delays_sorted, marker='o', markersize=6, linewidth=1.5,
                color='#A23B72', markerfacecolor='#A23B72', markeredgecolor='#D62828', markeredgewidth=0.5)
    axes[2].fill_between(range(len(prop_delays_sorted)), prop_delays_sorted, alpha=0.3, color='#A23B72')
    axes[2].set_xlabel('Chain Length (hops)')
    axes[2].set_ylabel('Propagation Delay (ms)')
    axes[2].set_title('Chain Topology Delay', fontsize=9)
    axes[2].grid(True, alpha=0.3, linestyle=':')

    # Chart 4: Topology comparison summary
    axes[3].text(0.5, 0.85, 'Star Topology', ha='center', fontsize=9, fontweight='bold', transform=axes[3].transAxes)
    axes[3].text(0.5, 0.70, f'Receivers: {int(np.mean(receiver_counts_star))}', ha='center', fontsize=8, transform=axes[3].transAxes)
    axes[3].text(0.5, 0.60, f'Coherence: {np.mean(all_coherent)*100:.0f}%', ha='center', fontsize=8, transform=axes[3].transAxes)

    axes[3].text(0.5, 0.45, 'Chain Topology', ha='center', fontsize=9, fontweight='bold', transform=axes[3].transAxes)
    axes[3].text(0.5, 0.30, f'Hops: {int(np.mean(chain_lengths))}', ha='center', fontsize=8, transform=axes[3].transAxes)
    axes[3].text(0.5, 0.20, f'Sync Rate: {np.mean(synchronized)*100:.0f}%', ha='center', fontsize=8, transform=axes[3].transAxes)

    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Panel 4 saved: {output_path}")

def main():
    # Load results
    results_path = 'plls_validation_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found!")
        return

    results = load_results(results_path)

    # Create output directory for panels
    panels_dir = 'figures'
    os.makedirs(panels_dir, exist_ok=True)

    # Generate all panels
    create_panel_1_sync_and_delivery(results, os.path.join(panels_dir, 'plls_panel_1_sync_delivery.png'))
    create_panel_2_bandwidth_scaling(results, os.path.join(panels_dir, 'plls_panel_2_bandwidth.png'))
    create_panel_3_frame_rate_3d(results, os.path.join(panels_dir, 'plls_panel_3_frame_rate_3d.png'))
    create_panel_4_topology_comparison(results, os.path.join(panels_dir, 'plls_panel_4_topology.png'))

    print("\n" + "="*70)
    print("Phase-Locked Live Streaming: All panels generated successfully")
    print("="*70)

if __name__ == "__main__":
    main()
