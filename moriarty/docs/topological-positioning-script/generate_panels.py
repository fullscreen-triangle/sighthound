#!/usr/bin/env python3
"""
Generate 4 publication-quality panels for PoSL validation results.

Each panel has 4 complementary charts (2D scatter, bar, 3D scatter, histogram/boxplot).
300 DPI, white background, minimal text.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Load results
with open("posl_validation_results.json", "r") as f:
    results = json.load(f)

# =============================================================================
# Panel 1: Composition Inflation Theorem Verification
# =============================================================================

def generate_panel_1():
    print("Generating Panel 1: Composition Inflation...")

    fig = plt.figure(figsize=(16, 4), dpi=300)
    fig.patch.set_facecolor('white')

    ci_data = results["experiments"]["composition_inflation"]

    # Extract n and d values
    n_vals = sorted(set(r["n"] for r in ci_data))
    d_vals = sorted(set(r["d"] for r in ci_data))

    # (A) Line plot: Γ(n,d) vs n for different d
    ax1 = fig.add_subplot(141)
    colors = cm.Set1(np.linspace(0, 1, len(d_vals)))

    for d_idx, d in enumerate(d_vals):
        data = [r for r in ci_data if r["d"] == d]
        data.sort(key=lambda x: x["n"])
        n_vals_d = [r["n"] for r in data]
        gamma_vals = [r["formula_gamma"] for r in data]
        ax1.semilogy(n_vals_d, gamma_vals, 'o-', color=colors[d_idx],
                    label=f'd={d}', linewidth=2, markersize=4)

    ax1.set_xlabel('Composition Depth (n)', fontsize=9)
    ax1.set_ylabel(r'$\Gamma(n,d)$', fontsize=9)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(A) Composition Inflation Growth', fontsize=9, fontweight='bold')

    # (B) Bar plot: Formula vs Enumeration agreement
    ax2 = fig.add_subplot(142)
    test_cases = [(2, 3), (3, 3), (4, 4), (5, 3), (6, 3)]
    formula_vals = []
    enum_vals = []

    for n, d in test_cases:
        match_data = [r for r in ci_data if r["n"] == n and r["d"] == d]
        if match_data:
            formula_vals.append(match_data[0]["formula_gamma"])
            enum_vals.append(match_data[0]["enumerated_count"])

    x_pos = np.arange(len(test_cases))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, formula_vals, width, label='Formula',
                    color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, enum_vals, width, label='Enumerated',
                    color='coral', alpha=0.8)

    ax2.set_ylabel('Count', fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'({n},{d})' for n, d in test_cases], fontsize=8)
    ax2.legend(fontsize=8)
    ax2.set_title('(B) Formula vs Enumeration', fontsize=9, fontweight='bold')
    ax2.set_yscale('log')

    # (C) 3D scatter: n, d, Γ(n,d)
    ax3 = fig.add_subplot(143, projection='3d')

    n_scatter = np.array([r["n"] for r in ci_data])
    d_scatter = np.array([r["d"] for r in ci_data])
    gamma_scatter = np.array([r["formula_gamma"] for r in ci_data])

    colors_scatter = cm.viridis(np.log10(gamma_scatter + 1) / np.log10(np.max(gamma_scatter) + 1))
    ax3.scatter(n_scatter, d_scatter, gamma_scatter, c=gamma_scatter,
               cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel('n', fontsize=8)
    ax3.set_ylabel('d', fontsize=8)
    ax3.set_zlabel(r'$\Gamma(n,d)$', fontsize=8)
    ax3.set_title('(C) State Space Inflation', fontsize=9, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # (D) Histogram: Distribution of formula values
    ax4 = fig.add_subplot(144)
    gamma_all = [r["formula_gamma"] for r in ci_data]
    ax4.hist(gamma_all, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel(r'$\Gamma(n,d)$ value', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.set_title('(D) Γ Distribution', fontsize=9, fontweight='bold')
    ax4.set_xscale('log')

    plt.tight_layout()
    plt.savefig('panel_1_composition_inflation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] panel_1_composition_inflation.png")


# =============================================================================
# Panel 2: Confidence Scaling with Composition Depth
# =============================================================================

def generate_panel_2():
    print("Generating Panel 2: Confidence Scaling...")

    fig = plt.figure(figsize=(16, 4), dpi=300)
    fig.patch.set_facecolor('white')

    cs_data = results["experiments"]["confidence_scaling"]

    # Organize by depth
    by_depth = {}
    for r in cs_data:
        depth = r["depth"]
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(r)

    depths = sorted(by_depth.keys())

    # (A) Scatter: Depth vs Confidence with color gradient
    ax1 = fig.add_subplot(141)

    depth_scatter = []
    conf_scatter = []
    colors_scatter = []

    for depth in depths:
        for r in by_depth[depth]:
            depth_scatter.append(depth)
            conf_scatter.append(r["confidence_margin"])
            colors_scatter.append(depth)

    scatter = ax1.scatter(depth_scatter, conf_scatter, c=colors_scatter,
                         cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax1.set_xlabel('Composition Depth (n)', fontsize=9)
    ax1.set_ylabel('Posterior Margin', fontsize=9)
    ax1.set_title('(A) Confidence vs Depth', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Depth')

    # (B) Bar plot: Mean confidence by depth
    ax2 = fig.add_subplot(142)

    mean_confs = [np.mean([r["confidence_margin"] for r in by_depth[d]]) for d in depths]
    std_confs = [np.std([r["confidence_margin"] for r in by_depth[d]]) for d in depths]

    ax2.bar(depths, mean_confs, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.errorbar(depths, mean_confs, yerr=std_confs, fmt='none',
                color='black', capsize=5, linewidth=1)

    ax2.set_xlabel('Composition Depth (n)', fontsize=9)
    ax2.set_ylabel('Mean Posterior Margin', fontsize=9)
    ax2.set_title('(B) Mean Confidence', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # (C) 3D scatter: depth, top_posterior, second_best
    ax3 = fig.add_subplot(143, projection='3d')

    depth_3d = np.array([r["depth"] for r in cs_data])
    top_3d = np.array([r["top_posterior"] for r in cs_data])
    second_3d = np.array([r["second_best"] for r in cs_data])

    ax3.scatter(depth_3d, top_3d, second_3d, c=depth_3d, cmap='plasma',
               s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel('Depth', fontsize=8)
    ax3.set_ylabel('Top Posterior', fontsize=8)
    ax3.set_zlabel('2nd Best', fontsize=8)
    ax3.set_title('(C) Posterior Evolution', fontsize=9, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # (D) Box plot: Distribution by depth
    ax4 = fig.add_subplot(144)

    conf_by_depth = [np.array([r["confidence_margin"] for r in by_depth[d]]) for d in depths]
    bp = ax4.boxplot(conf_by_depth, labels=depths, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)

    ax4.set_xlabel('Composition Depth (n)', fontsize=9)
    ax4.set_ylabel('Posterior Margin', fontsize=9)
    ax4.set_title('(D) Confidence Distribution', fontsize=9, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('panel_2_confidence_scaling.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] panel_2_confidence_scaling.png")


# =============================================================================
# Panel 3: Multi-Modal Positioning Accuracy
# =============================================================================

def generate_panel_3():
    print("Generating Panel 3: Multi-Modal Positioning...")

    fig = plt.figure(figsize=(16, 4), dpi=300)
    fig.patch.set_facecolor('white')

    mm_data = results["experiments"]["multimodal_positioning"]

    # Organize by num_modalities
    by_modality = {}
    for r in mm_data:
        mod = r["num_modalities"]
        if mod not in by_modality:
            by_modality[mod] = []
        by_modality[mod].append(r)

    modalities = sorted(by_modality.keys())

    # (A) Error vs Modality scatter
    ax1 = fig.add_subplot(141)

    mod_scatter = []
    err_scatter = []

    for mod in modalities:
        for r in by_modality[mod]:
            mod_scatter.append(mod + np.random.normal(0, 0.05))
            err_scatter.append(r["error"])

    ax1.scatter(mod_scatter, err_scatter, alpha=0.5, s=20, color='coral', edgecolors='black', linewidth=0.5)

    ax1.set_xlabel('Number of Modalities', fontsize=9)
    ax1.set_ylabel('Position Error (regions)', fontsize=9)
    ax1.set_xticks(modalities)
    ax1.set_title('(A) Positioning Error', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # (B) Bar plot: Accuracy by modality
    ax2 = fig.add_subplot(142)

    accuracy_by_mod = []
    conf_by_mod = []

    for mod in modalities:
        correct_count = sum(1 for r in by_modality[mod] if r["correct"])
        accuracy = 100.0 * correct_count / len(by_modality[mod])
        accuracy_by_mod.append(accuracy)

        mean_conf = np.mean([r["confidence"] for r in by_modality[mod]])
        conf_by_mod.append(mean_conf)

    ax2.bar(modalities, accuracy_by_mod, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Modalities', fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=9)
    ax2.set_xticks(modalities)
    ax2.set_ylim([0, 105])
    ax2.set_title('(B) Classification Accuracy', fontsize=9, fontweight='bold')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # (C) 3D scatter: modality, error, confidence
    ax3 = fig.add_subplot(143, projection='3d')

    mod_3d = np.array([r["num_modalities"] for r in mm_data])
    err_3d = np.array([r["error"] for r in mm_data])
    conf_3d = np.array([r["confidence"] for r in mm_data])

    ax3.scatter(mod_3d, err_3d, conf_3d, c=mod_3d, cmap='plasma',
               s=20, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel('Modalities', fontsize=8)
    ax3.set_ylabel('Error', fontsize=8)
    ax3.set_zlabel('Confidence', fontsize=8)
    ax3.set_title('(C) Error-Confidence Trade-off', fontsize=9, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # (D) Confidence histogram by modality
    ax4 = fig.add_subplot(144)

    for mod, color in zip(modalities, cm.Set2(np.linspace(0, 1, len(modalities)))):
        confs = [r["confidence"] for r in by_modality[mod]]
        ax4.hist(confs, bins=15, alpha=0.5, label=f'{mod} mod', color=color, edgecolor='black')

    ax4.set_xlabel('Confidence', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.set_title('(D) Confidence Distribution', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('panel_3_multimodal_positioning.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] panel_3_multimodal_positioning.png")


# =============================================================================
# Panel 4: Replay Attack Detection via Monotone Epochs
# =============================================================================

def generate_panel_4():
    print("Generating Panel 4: Replay Attack Detection...")

    fig = plt.figure(figsize=(16, 4), dpi=300)
    fig.patch.set_facecolor('white')

    ra_data = results["experiments"]["replay_attacks"]

    # Organize by epoch_delta
    by_delta = {}
    for r in ra_data:
        delta = r["epoch_delta"]
        if delta not in by_delta:
            by_delta[delta] = []
        by_delta[delta].append(r)

    deltas = sorted(by_delta.keys())

    # (A) Detection rate vs epoch delta
    ax1 = fig.add_subplot(141)

    detection_rates = []
    for delta in deltas:
        detected_count = sum(1 for r in by_delta[delta] if r["detected"])
        rate = detected_count / len(by_delta[delta])
        detection_rates.append(rate)

    ax1.plot(deltas, detection_rates, 'o-', color='darkred', linewidth=2, markersize=6)
    ax1.fill_between(deltas, detection_rates, alpha=0.3, color='red')

    ax1.set_xlabel('Epoch Delta', fontsize=9)
    ax1.set_ylabel('Detection Rate', fontsize=9)
    ax1.set_ylim([0, 1.05])
    ax1.set_title('(A) Replay Detection Rate', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # (B) Bar plot: Detection success rate
    ax2 = fig.add_subplot(142)

    colors_bar = ['red' if rate < 0.5 else 'orange' if rate < 0.8 else 'green' for rate in detection_rates]
    ax2.bar(deltas, detection_rates, color=colors_bar, alpha=0.7, edgecolor='black', width=0.8)

    ax2.set_xlabel('Epoch Delta', fontsize=9)
    ax2.set_ylabel('Detection Rate', fontsize=9)
    ax2.set_ylim([0, 1.05])
    ax2.set_title('(B) Success Rates', fontsize=9, fontweight='bold')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # (C) 3D scatter: epoch_delta, timing_shift, probability_undetected
    ax3 = fig.add_subplot(143, projection='3d')

    delta_3d = np.array([r["epoch_delta"] for r in ra_data])
    shift_3d = np.array([r["timing_deviation_shift"] for r in ra_data])
    prob_3d = np.array([r["probability_undetected"] for r in ra_data])

    ax3.scatter(delta_3d, shift_3d, prob_3d, c=delta_3d, cmap='RdYlGn_r',
               s=20, alpha=0.4, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel(r'Epoch $\Delta$', fontsize=8)
    ax3.set_ylabel('Timing Shift', fontsize=8)
    ax3.set_zlabel('P(Undetected)', fontsize=8)
    ax3.set_title('(C) Attack Metrics', fontsize=9, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # (D) Histogram: Distribution of probability_undetected
    ax4 = fig.add_subplot(144)

    probs = [r["probability_undetected"] for r in ra_data]
    ax4.hist(probs, bins=20, color='steelblue', edgecolor='black', alpha=0.7)

    ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Detection Threshold')
    ax4.set_xlabel('P(Undetected)', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.set_title('(D) Evasion Probability', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('panel_4_replay_attacks.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] panel_4_replay_attacks.png")


# =============================================================================
# Bonus Panel 5: Sports Video Positioning Accuracy
# =============================================================================

def generate_panel_5():
    print("Generating Panel 5: Sports Video Positioning...")

    fig = plt.figure(figsize=(16, 4), dpi=300)
    fig.patch.set_facecolor('white')

    sv_data = results["experiments"]["sports_video_positioning"]

    errors = np.array([r["position_error_m"] for r in sv_data])
    confs = np.array([r["mean_camera_confidence"] for r in sv_data])
    frame_idxs = np.array([r["frame_idx"] for r in sv_data])

    # (A) Error over time (frames)
    ax1 = fig.add_subplot(141)

    ax1.plot(frame_idxs, errors, linewidth=1, color='steelblue', alpha=0.7)
    ax1.fill_between(frame_idxs, errors, alpha=0.3, color='steelblue')

    ax1.axhline(y=np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}m')
    ax1.set_xlabel('Frame Index', fontsize=9)
    ax1.set_ylabel('Position Error (m)', fontsize=9)
    ax1.set_title('(A) Error Timeline', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (B) Bar plot: Accuracy binned by error threshold
    ax2 = fig.add_subplot(142)

    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    accuracies = [100.0 * np.mean(errors <= t) for t in thresholds]

    ax2.bar(range(len(thresholds)), accuracies, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(thresholds)))
    ax2.set_xticklabels([f'{t}m' for t in thresholds], fontsize=8)
    ax2.set_ylabel('Accuracy (%)', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.set_title('(B) Accuracy Thresholds', fontsize=9, fontweight='bold')

    # (C) 3D scatter: frame_idx, error, confidence
    ax3 = fig.add_subplot(143, projection='3d')

    ax3.scatter(frame_idxs, errors, confs, c=errors, cmap='RdYlGn_r',
               s=20, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax3.set_xlabel('Frame', fontsize=8)
    ax3.set_ylabel('Error (m)', fontsize=8)
    ax3.set_zlabel('Confidence', fontsize=8)
    ax3.set_title('(C) Error-Confidence Relation', fontsize=9, fontweight='bold')
    ax3.view_init(elev=20, azim=45)

    # (D) Histogram: Error distribution
    ax4 = fig.add_subplot(144)

    ax4.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)

    ax4.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}m')
    ax4.axvline(x=np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}m')

    ax4.set_xlabel('Position Error (m)', fontsize=9)
    ax4.set_ylabel('Frequency', fontsize=9)
    ax4.set_title('(D) Error Distribution', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('panel_5_sports_video.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] panel_5_sports_video.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PoSL PANEL GENERATION")
    print("="*70 + "\n")

    generate_panel_1()
    generate_panel_2()
    generate_panel_3()
    generate_panel_4()
    generate_panel_5()

    print("\n" + "="*70)
    print("PANELS GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nAll panels saved at 300 DPI with white backgrounds.")
    print("Ready for publication.\n")
