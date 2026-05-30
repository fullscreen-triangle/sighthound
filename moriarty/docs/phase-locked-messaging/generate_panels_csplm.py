"""
Generate publication-quality panels for CSPLM validation results.
4 panels, 300 DPI, white background, 4 charts per panel.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# Configuration for 300 DPI publication quality
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8

def load_results(filepath: str) -> dict:
    """Load validation results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_panel_1_complement_operations(results: dict, output_path: str):
    """
    Panel 1: Complement Operations
    (A) Involution verification rate across trials
    (B) Complement consistency rates (τ valid, τ̄ valid, both valid)
    (C) 3D surface of evidence vs complement likelihood
    (D) Reconstruction accuracy distribution
    """
    fig = plt.figure(figsize=(12, 3.5), facecolor='white')

    # (A) Involution verification
    ax1 = plt.subplot(1, 4, 1)
    reconstruction = results['complement_reconstruction']['trials']
    trials = list(range(len(reconstruction)))
    involution_rates = [t['involution_holds'] for t in reconstruction]

    ax1.scatter(trials, involution_rates, alpha=0.6, s=20, color='steelblue')
    ax1.axhline(y=np.mean(involution_rates), color='red', linestyle='--', linewidth=1, label=f"Mean: {np.mean(involution_rates):.3f}")
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Involution Holds (τ̄̄ = τ)')
    ax1.set_title('(A) Involution Verification')
    ax1.set_ylim([-0.1, 1.1])
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # (B) Complement consistency rates
    ax2 = plt.subplot(1, 4, 2)
    consistency = results['complement_consistency']['summary']
    categories = ['τ Valid', 'τ̄ Valid', 'Both Valid', 'Involution']
    rates = [
        consistency['tau_validity_rate'],
        consistency['complement_validity_rate'],
        consistency['both_valid_rate'],
        consistency['involution_rate']
    ]
    colors = ['steelblue', 'darkgreen', 'darkorange', 'crimson']
    bars = ax2.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax2.text(bar.get_x() + bar.get_width()/2, rate + 0.02, f'{rate:.3f}',
                ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('Rate')
    ax2.set_title('(B) Consistency Rates')
    ax2.set_ylim([0, 1.15])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # (C) 3D surface: Evidence vs Complement likelihood
    ax3 = plt.subplot(1, 4, 3, projection='3d')
    consistency_trials = results['complement_consistency']['trials']

    # Create 2D grid for visualization
    num_points = min(50, len(consistency_trials))
    tau_likelihoods = np.array([np.log10(t['tau_likelihood'] + 1e-20) for t in consistency_trials[:num_points]])
    comp_likelihoods = np.array([np.log10(t['complement_likelihood'] + 1e-20) for t in consistency_trials[:num_points]])
    indices = np.arange(num_points)

    # Create mesh
    indices_mesh, tau_mesh = np.meshgrid(indices[::5], np.linspace(tau_likelihoods.min(), tau_likelihoods.max(), 10))
    comp_mesh = np.zeros_like(indices_mesh)

    for i, idx in enumerate(indices[::5]):
        for j in range(comp_mesh.shape[1]):
            comp_mesh[i, j] = comp_likelihoods[int(idx)]

    surf = ax3.plot_surface(indices_mesh, tau_mesh, comp_mesh, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('Trial', fontsize=8)
    ax3.set_ylabel('log(τ Likelihood)', fontsize=8)
    ax3.set_zlabel('log(τ̄ Likelihood)', fontsize=8)
    ax3.set_title('(C) Likelihood Surface', fontsize=10)
    ax3.view_init(elev=20, azim=45)

    # (D) Reconstruction accuracy distribution
    ax4 = plt.subplot(1, 4, 4)
    reconstruction_trials = results['complement_reconstruction']['trials']
    reconstruction_accuracy = [t['reconstruction_exact'] for t in reconstruction_trials]

    exact_count = sum(reconstruction_accuracy)
    counts = [exact_count, len(reconstruction_accuracy) - exact_count]
    labels = [f'Exact\n({exact_count})', f'Inexact\n({len(reconstruction_accuracy) - exact_count})']
    colors_pie = ['steelblue', 'lightcoral']

    wedges, texts, autotexts = ax4.pie(counts, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)
        autotext.set_weight('bold')

    ax4.set_title('(D) Reconstruction Accuracy')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Panel 1 saved to {output_path}")
    plt.close()

def generate_panel_2_forgery_amplification(results: dict, output_path: str):
    """
    Panel 2: Forgery Amplification
    (A) PLM vs CSPLM detection ratio scatter
    (B) Mean detection ratios bar chart
    (C) 3D surface of amplification factor
    (D) Amplification factor distribution
    """
    fig = plt.figure(figsize=(12, 3.5), facecolor='white')

    # (A) PLM vs CSPLM detection scatter
    ax1 = plt.subplot(1, 4, 1)
    forgery = results['forgery_amplification']['trials']
    plm_ratios = np.array([abs(t['plm_detection_ratio']) for t in forgery])
    csplm_ratios = np.array([abs(t['csplm_detection_ratio']) for t in forgery])

    ax1.scatter(plm_ratios, csplm_ratios, alpha=0.6, s=25, color='steelblue', edgecolors='black', linewidth=0.5)
    # Plot diagonal (equal line)
    max_val = max(plm_ratios.max(), csplm_ratios.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Equal (PLM=CSPLM)')
    ax1.set_xlabel('PLM Detection Ratio')
    ax1.set_ylabel('CSPLM Detection Ratio')
    ax1.set_title('(A) PLM vs CSPLM Detection')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # (B) Mean detection ratios
    ax2 = plt.subplot(1, 4, 2)
    summary = results['forgery_amplification']['summary']
    metrics = ['PLM Mean', 'CSPLM Mean']
    values = [summary['mean_plm_detection_ratio'], summary['mean_csplm_detection_ratio']]
    colors = ['steelblue', 'darkgreen']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('Detection Ratio (log scale)')
    ax2.set_title('(B) Mean Detection Ratios')
    ax2.grid(True, alpha=0.3, axis='y')

    # (C) 3D surface: Amplification factor
    ax3 = plt.subplot(1, 4, 3, projection='3d')
    amplification_factors = np.array([t['amplification_factor'] for t in forgery[:50]])
    plm_subset = plm_ratios[:50]
    csplm_subset = csplm_ratios[:50]

    indices = np.arange(len(amplification_factors))
    indices_mesh, plm_mesh = np.meshgrid(indices[::5], np.linspace(plm_subset.min(), plm_subset.max(), 10))
    amp_mesh = np.zeros_like(indices_mesh)

    for i, idx in enumerate(indices[::5]):
        for j in range(amp_mesh.shape[1]):
            amp_mesh[i, j] = amplification_factors[int(idx)]

    surf = ax3.plot_surface(indices_mesh, plm_mesh, amp_mesh, cmap='plasma', alpha=0.8)
    ax3.set_xlabel('Trial Index', fontsize=8)
    ax3.set_ylabel('PLM Detection', fontsize=8)
    ax3.set_zlabel('Amplification Factor', fontsize=8)
    ax3.set_title('(C) Amplification Surface', fontsize=10)
    ax3.view_init(elev=20, azim=45)

    # (D) Amplification distribution
    ax4 = plt.subplot(1, 4, 4)
    ax4.hist(amplification_factors, bins=25, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax4.axvline(x=np.mean(amplification_factors), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(amplification_factors):.2f}x')
    ax4.set_xlabel('Amplification Factor')
    ax4.set_ylabel('Frequency')
    ax4.set_title('(D) Amplification Distribution')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Panel 2 saved to {output_path}")
    plt.close()

def generate_panel_3_noise_detection(results: dict, output_path: str):
    """
    Panel 3: Complement Noise Detection
    (A) Detection rate vs corruption level
    (B) Mean detection across corruption rates
    (C) 3D surface of detection vs corruption
    (D) Likelihood degradation histogram
    """
    fig = plt.figure(figsize=(12, 3.5), facecolor='white')

    # (A) Detection rate vs corruption level
    ax1 = plt.subplot(1, 4, 1)
    noise_trials = results['complement_noise_detection']['trials']
    corruption_rates = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Compute detection rates for each corruption level
    detection_by_corruption = {rate: [] for rate in corruption_rates}
    for trial in noise_trials:
        for item in trial:
            if item['corruption_rate'] in detection_by_corruption:
                detection_by_corruption[item['corruption_rate']].append(item['detected'])

    mean_detections = [np.mean(detection_by_corruption[rate]) for rate in corruption_rates]

    ax1.plot(corruption_rates * 100, mean_detections, 'o-', linewidth=2, markersize=8, color='steelblue', markerfacecolor='steelblue')
    ax1.fill_between(corruption_rates * 100, 0, mean_detections, alpha=0.3, color='steelblue')
    ax1.set_xlabel('Corruption Rate (%)')
    ax1.set_ylabel('Detection Rate')
    ax1.set_title('(A) Detection vs Corruption')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3)

    # (B) Mean detection at each corruption level
    ax2 = plt.subplot(1, 4, 2)
    labels = [f'{int(r*100)}%' for r in corruption_rates]
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(corruption_rates)))
    bars = ax2.bar(labels, mean_detections, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)

    for bar, rate in zip(bars, mean_detections):
        ax2.text(bar.get_x() + bar.get_width()/2, rate + 0.03, f'{rate:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Corruption Rate')
    ax2.set_ylabel('Mean Detection Rate')
    ax2.set_title('(B) Detection by Corruption')
    ax2.set_ylim([0, 1.15])
    ax2.grid(True, alpha=0.3, axis='y')

    # (C) 3D surface
    ax3 = plt.subplot(1, 4, 3, projection='3d')
    trial_indices = np.arange(min(30, len(noise_trials)))

    # Create data for surface
    Z = np.zeros((len(corruption_rates), len(trial_indices)))
    for i, rate in enumerate(corruption_rates):
        for j, trial_idx in enumerate(trial_indices):
            if trial_idx < len(noise_trials) and i * len(trial_indices) + j < len(noise_trials[trial_idx]):
                Z[i, j] = mean_detections[i]

    X, Y = np.meshgrid(trial_indices, corruption_rates * 100)
    surf = ax3.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8)
    ax3.set_xlabel('Trial Index', fontsize=8)
    ax3.set_ylabel('Corruption (%)', fontsize=8)
    ax3.set_zlabel('Detection Rate', fontsize=8)
    ax3.set_title('(C) Detection Surface', fontsize=10)
    ax3.view_init(elev=20, azim=45)

    # (D) Likelihood degradation
    ax4 = plt.subplot(1, 4, 4)
    likelihood_degradations = []
    for trial in noise_trials:
        for item in trial:
            if item['corruption_rate'] > 0.0:
                degradation = item['true_likelihood'] - item['consistency_likelihood']
                likelihood_degradations.append(degradation)

    ax4.hist(likelihood_degradations, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax4.axvline(x=np.mean(likelihood_degradations), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(likelihood_degradations):.4f}')
    ax4.set_xlabel('Likelihood Degradation')
    ax4.set_ylabel('Frequency')
    ax4.set_title('(D) Degradation Distribution')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Panel 3 saved to {output_path}")
    plt.close()

def generate_panel_4_csplm_vs_plm(results: dict, output_path: str):
    """
    Panel 4: CSPLM vs PLM Comparison
    (A) Detection margin comparison scatter
    (B) Mean margins bar chart
    (C) 3D surface of security advantage
    (D) Advantage factor distribution
    """
    fig = plt.figure(figsize=(12, 3.5), facecolor='white')

    # (A) Detection margin comparison scatter
    ax1 = plt.subplot(1, 4, 1)
    comparison = results['csplm_vs_plm']['trials']
    plm_margins = np.array([t['plm_detection_margin'] for t in comparison])
    csplm_margins = np.array([t['csplm_detection_margin'] for t in comparison])

    ax1.scatter(plm_margins, csplm_margins, alpha=0.6, s=25, color='steelblue', edgecolors='black', linewidth=0.5)
    max_margin = max(plm_margins.max(), csplm_margins.max())
    ax1.plot([0, max_margin], [0, max_margin], 'r--', linewidth=1.5, label='Equal (PLM=CSPLM)')
    ax1.set_xlabel('PLM Detection Margin')
    ax1.set_ylabel('CSPLM Detection Margin')
    ax1.set_title('(A) Margin Comparison')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # (B) Mean margins bar chart
    ax2 = plt.subplot(1, 4, 2)
    summary = results['csplm_vs_plm']['summary']
    systems = ['PLM', 'CSPLM']
    margins = [summary['mean_plm_margin'], summary['mean_csplm_margin']]
    colors = ['steelblue', 'darkgreen']
    bars = ax2.bar(systems, margins, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    for bar, margin in zip(bars, margins):
        ax2.text(bar.get_x() + bar.get_width()/2, margin + 0.2, f'{margin:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('Mean Detection Margin')
    ax2.set_title('(B) Mean Margins')
    ax2.grid(True, alpha=0.3, axis='y')

    # (C) 3D surface: Advantage factor
    ax3 = plt.subplot(1, 4, 3, projection='3d')
    advantage_factors = np.array([t['csplm_advantage'] for t in comparison[:50]])
    plm_subset = plm_margins[:50]
    csplm_subset = csplm_margins[:50]

    indices = np.arange(len(advantage_factors))
    indices_mesh, plm_mesh = np.meshgrid(indices[::5], np.linspace(plm_subset.min(), plm_subset.max(), 10))
    adv_mesh = np.zeros_like(indices_mesh)

    for i, idx in enumerate(indices[::5]):
        for j in range(adv_mesh.shape[1]):
            adv_mesh[i, j] = advantage_factors[int(idx)]

    surf = ax3.plot_surface(indices_mesh, plm_mesh, adv_mesh, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('Trial Index', fontsize=8)
    ax3.set_ylabel('PLM Margin', fontsize=8)
    ax3.set_zlabel('Advantage Factor', fontsize=8)
    ax3.set_title('(C) Advantage Surface', fontsize=10)
    ax3.view_init(elev=20, azim=45)

    # (D) Advantage factor distribution
    ax4 = plt.subplot(1, 4, 4)
    ax4.hist(advantage_factors, bins=25, color='darkgreen', alpha=0.7, edgecolor='black', linewidth=1)
    ax4.axvline(x=np.mean(advantage_factors), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(advantage_factors):.2f}x')
    ax4.set_xlabel('Security Advantage Factor')
    ax4.set_ylabel('Frequency')
    ax4.set_title('(D) Advantage Distribution')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"Panel 4 saved to {output_path}")
    plt.close()

def main():
    """Generate all CSPLM validation panels."""
    print("Loading CSPLM validation results...")
    results = load_results('csplm_validation_results.json')

    print("Generating panels...")
    generate_panel_1_complement_operations(results, 'csplm_panel_1_complement_operations.png')
    generate_panel_2_forgery_amplification(results, 'csplm_panel_2_forgery_amplification.png')
    generate_panel_3_noise_detection(results, 'csplm_panel_3_noise_detection.png')
    generate_panel_4_csplm_vs_plm(results, 'csplm_panel_4_csplm_vs_plm.png')

    print("\nAll panels generated successfully!")
    print("Generated files:")
    print("  - csplm_panel_1_complement_operations.png")
    print("  - csplm_panel_2_forgery_amplification.png")
    print("  - csplm_panel_3_noise_detection.png")
    print("  - csplm_panel_4_csplm_vs_plm.png")

if __name__ == "__main__":
    main()
