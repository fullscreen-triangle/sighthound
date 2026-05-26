"""
Generate 4 visualization panels for validation experiments.
Each panel: white background, 4 charts in a row (1+ are 3D), limited text, actual data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Load results
with open(r"c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\validation_results.json") as f:
    results = json.load(f)

# Set style
plt.style.use('default')


def panel_1_harmonic_graphs():
    """Panel 1: Harmonic-Graph Channel Capacity (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_1_harmonic_graphs"]
    n_modes = [int(k) for k in sorted(data.keys())]
    medians_C = [data[str(n)]["median_C"] for n in n_modes]
    means_C = [data[str(n)]["mean_C"] for n in n_modes]
    stds_C = [data[str(n)]["std_C"] for n in n_modes]
    medians_Nmax = [data[str(n)]["median_Nmax"] for n in n_modes]

    # Chart 1: Cycle Rank vs N_modes (scatter with error bars)
    ax1 = fig.add_subplot(141)
    ax1.errorbar(n_modes, means_C, yerr=stds_C, fmt='o', markersize=8, capsize=5,
                 color='#2E86AB', ecolor='#A23B72', linewidth=2, label='Mean ± Std')
    ax1.scatter(n_modes, medians_C, s=120, color='#F18F01', marker='s', zorder=5,
                label='Median', edgecolors='black', linewidth=1.5)
    ax1.set_xlabel('Number of Modes', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cycle Rank (C)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(3, 13)

    # Chart 2: Capacity vs N_modes (bar chart)
    ax2 = fig.add_subplot(142)
    bars = ax2.bar(n_modes, medians_Nmax, color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Number of Modes', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Channel Capacity (bits)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (nm, cap) in enumerate(zip(n_modes, medians_Nmax)):
        ax2.text(nm, cap + 200, f'{int(cap)}', ha='center', fontsize=9, fontweight='bold')

    # Chart 3: 3D scatter (N_modes, C, N_max)
    ax3 = fig.add_subplot(143, projection='3d')
    all_ranks = []
    all_capacities = []
    all_modes = []
    for n in n_modes:
        ranks = data[str(n)]["all_cycle_ranks"][:15]  # Subsample for clarity
        caps = data[str(n)]["all_capacities"][:15]
        all_ranks.extend(ranks)
        all_capacities.extend(caps)
        all_modes.extend([n] * len(ranks))

    scatter = ax3.scatter(all_modes, all_ranks, all_capacities, c=all_capacities,
                         cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('N_modes', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Cycle Rank', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Capacity', fontsize=10, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # Chart 4: Distribution of cycle ranks (violin plot)
    ax4 = fig.add_subplot(144)
    all_distributions = [data[str(n)]["all_cycle_ranks"] for n in n_modes]
    parts = ax4.violinplot(all_distributions, positions=n_modes, widths=0.7,
                           showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#E63946')
        pc.set_alpha(0.7)
    ax4.set_xlabel('Number of Modes', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cycle Rank Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(3, 13)

    plt.suptitle('Experiment 1: Harmonic-Graph Channel Capacity', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_1_harmonic.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 1 saved: panel_1_harmonic.png")
    plt.close()


def panel_2_partition_hierarchy():
    """Panel 2: Partition-Hierarchy Distinguishability (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_2_partition_hierarchy"]
    depths = [int(k) for k in sorted(data.keys())]
    leaf_cells = [data[str(n)]["leaf_cells_3^n"] for n in depths]
    shell_capacities = [data[str(n)]["shell_capacity_2n2"] for n in depths]
    ratios = [data[str(n)]["ratio_3^n_over_2n2"] for n in depths]

    # Chart 1: Leaf cells (log scale)
    ax1 = fig.add_subplot(141)
    ax1.semilogy(depths, leaf_cells, 'o-', markersize=10, linewidth=2.5,
                 color='#264653', markerfacecolor='#2A9D8F', markeredgecolor='black', markeredgewidth=1.5)
    ax1.fill_between(depths, leaf_cells, alpha=0.2, color='#2A9D8F')
    ax1.set_xlabel('Partition Depth (n)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Leaf Cells ($3^n$)', fontsize=11, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.3)

    # Chart 2: Shell capacity (log scale)
    ax2 = fig.add_subplot(142)
    ax2.semilogy(depths, shell_capacities, 's-', markersize=10, linewidth=2.5,
                 color='#E76F51', markerfacecolor='#F4A261', markeredgecolor='black', markeredgewidth=1.5)
    ax2.fill_between(depths, shell_capacities, alpha=0.2, color='#F4A261')
    ax2.set_xlabel('Partition Depth (n)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Shell Capacity ($2n^2$)', fontsize=11, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)

    # Chart 3: 3D surface (depth, leaf_cells, shell_capacity)
    ax3 = fig.add_subplot(143, projection='3d')
    depths_expanded = np.linspace(4, 20, 30)
    leaf_expanded = 3 ** depths_expanded
    shell_expanded = 2 * depths_expanded ** 2

    ax3.plot(depths, leaf_cells, shell_capacities, 'o-', markersize=8, linewidth=2.5,
             color='#E63946', markerfacecolor='#F1FAEE', markeredgecolor='black', markeredgewidth=1.5)
    ax3.set_xlabel('Depth', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Leaf Cells', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Shell Capacity', fontsize=10, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    ax3.set_yscale('log')

    # Chart 4: Ratio (3^n / 2n^2) - exponential growth
    ax4 = fig.add_subplot(144)
    ax4.loglog(depths, ratios, 'D-', markersize=10, linewidth=2.5,
               color='#1D3557', markerfacecolor='#A8DADC', markeredgecolor='black', markeredgewidth=1.5)
    ax4.fill_between(depths, ratios, alpha=0.2, color='#A8DADC')
    ax4.set_xlabel('Partition Depth (n)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Ratio ($3^n / C(n)$)', fontsize=11, fontweight='bold')
    ax4.grid(True, which='both', alpha=0.3)

    plt.suptitle('Experiment 2: Partition-Hierarchy Distinguishability', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_2_partition.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 2 saved: panel_2_partition.png")
    plt.close()


def panel_3_optical_stacks():
    """Panel 3: Transfer-Matrix Rank in Optical Stacks (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_3_optical_stacks"]
    n_layers = [int(k) for k in sorted(data.keys())]
    k_wavelengths = [4, 8, 16, 32]

    # Build rank matrix
    rank_matrix = np.zeros((len(n_layers), len(k_wavelengths)))
    for i, n in enumerate(n_layers):
        for j, k in enumerate(k_wavelengths):
            rank_matrix[i, j] = data[str(n)][str(k)]

    # Chart 1: Heatmap of rank(N, K)
    ax1 = fig.add_subplot(141)
    im = ax1.imshow(rank_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax1.set_xticks(range(len(k_wavelengths)))
    ax1.set_yticks(range(len(n_layers)))
    ax1.set_xticklabels(k_wavelengths)
    ax1.set_yticklabels(n_layers)
    ax1.set_xlabel('Number of Wavelengths (K)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Layers (N)', fontsize=11, fontweight='bold')
    for i in range(len(n_layers)):
        for j in range(len(k_wavelengths)):
            ax1.text(j, i, f'{int(rank_matrix[i, j])}', ha='center', va='center',
                    color='black', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax1, label='Rank')

    # Chart 2: Rank vs K for each N (line plot)
    ax2 = fig.add_subplot(142)
    colors_n = plt.cm.cool(np.linspace(0, 1, len(n_layers)))
    for i, n in enumerate(n_layers):
        ranks_for_n = rank_matrix[i, :]
        ax2.plot(k_wavelengths, ranks_for_n, 'o-', label=f'N={n}', linewidth=2.5,
                markersize=8, color=colors_n[i], markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Number of Wavelengths (K)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Transfer-Matrix Rank', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Chart 3: 3D surface (N, K, rank)
    ax3 = fig.add_subplot(143, projection='3d')
    N_mesh, K_mesh = np.meshgrid(n_layers, k_wavelengths, indexing='ij')
    surf = ax3.plot_surface(N_mesh, K_mesh, rank_matrix, cmap='plasma', alpha=0.8,
                           edgecolor='black', linewidth=0.5)
    ax3.scatter(N_mesh.flatten(), K_mesh.flatten(), rank_matrix.flatten(),
               c='red', s=80, marker='o', edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Layers (N)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Wavelengths (K)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Rank', fontsize=10, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax3, label='Rank', shrink=0.5)

    # Chart 4: min(N, K) theoretical vs actual
    ax4 = fig.add_subplot(144)
    theoretical_rank = np.minimum(N_mesh, K_mesh)
    diff = theoretical_rank - rank_matrix
    im2 = ax4.imshow(diff, cmap='RdBu', aspect='auto', origin='lower', vmin=-0.5, vmax=0.5)
    ax4.set_xticks(range(len(k_wavelengths)))
    ax4.set_yticks(range(len(n_layers)))
    ax4.set_xticklabels(k_wavelengths)
    ax4.set_yticklabels(n_layers)
    ax4.set_xlabel('Number of Wavelengths (K)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Number of Layers (N)', fontsize=11, fontweight='bold')
    ax4.set_title('Error: min(N,K) - Actual', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax4, label='Difference')

    plt.suptitle('Experiment 3: Transfer-Matrix Rank in Optical Stacks', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_3_optical.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 3 saved: panel_3_optical.png")
    plt.close()


def panel_4_observer_invisibility():
    """Panel 4: External-Observer Invisibility (4 charts)."""
    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')

    data = results["experiment_4_observer_invisibility"]
    snr_levels = [1, 10, 100]

    # Collect MI values
    mi_by_snr = {}
    for snr in snr_levels:
        mi_by_snr[snr] = data[str(snr)]["all_mi_values"]

    # Chart 1: Box plot of MI at different SNR
    ax1 = fig.add_subplot(141)
    bp = ax1.boxplot([mi_by_snr[snr] for snr in snr_levels],
                      labels=[f'SNR={s}' for s in snr_levels],
                      patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, color='black')
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)
    ax1.set_ylabel('Mutual Information (bits)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-0.001, 0.016)

    # Chart 2: Scatter plot with all trials
    ax2 = fig.add_subplot(142)
    colors = {'1': '#FF6B6B', '10': '#4ECDC4', '100': '#45B7D1'}
    for snr in snr_levels:
        mi_vals = mi_by_snr[snr]
        x_jitter = np.random.normal(snr, 1.5, len(mi_vals))  # Add jitter
        ax2.scatter(x_jitter, mi_vals, alpha=0.5, s=50, color=colors[str(snr)],
                   edgecolors='black', linewidth=0.5)
    ax2.set_xticks(snr_levels)
    ax2.set_xlabel('SNR', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mutual Information (bits)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(-0.001, 0.016)

    # Chart 3: 3D scatter (SNR, trial, MI)
    ax3 = fig.add_subplot(143, projection='3d')
    for snr_idx, snr in enumerate(snr_levels):
        mi_vals = mi_by_snr[snr]
        trials = np.arange(len(mi_vals))
        snr_array = np.full_like(trials, snr)
        ax3.scatter(snr_array, trials, mi_vals, s=30, alpha=0.6,
                   color=colors[str(snr)], edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('SNR', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Trial Number', fontsize=10, fontweight='bold')
    ax3.set_zlabel('MI (bits)', fontsize=10, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # Chart 4: Histograms of MI distribution
    ax4 = fig.add_subplot(144)
    for snr in snr_levels:
        mi_vals = mi_by_snr[snr]
        ax4.hist(mi_vals, bins=15, alpha=0.6, label=f'SNR={snr}',
                color=colors[str(snr)], edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Mutual Information (bits)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axvline(0.01, color='red', linestyle='--', linewidth=2, label='0.01 bits threshold')

    plt.suptitle('Experiment 4: External-Observer Invisibility', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(r'c:\Users\kunda\Documents\physics\sighthound\moriarty\validation\panel_4_invisibility.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("[OK] Panel 4 saved: panel_4_invisibility.png")
    plt.close()


def main():
    """Generate all four panels."""
    print("Generating visualization panels...")
    panel_1_harmonic_graphs()
    panel_2_partition_hierarchy()
    panel_3_optical_stacks()
    panel_4_observer_invisibility()
    print("\n[COMPLETE] All 4 panels generated successfully")
    print("Output directory: c:\\Users\\kunda\\Documents\\physics\\sighthound\\moriarty\\validation\\")


if __name__ == "__main__":
    main()
