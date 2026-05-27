#!/usr/bin/env python3
"""
Generate visualization panels for coordinate field validation results.
4 panels, one for each experiment, with 4 charts per panel.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import os

# High-quality output
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8

def load_results(filename):
    """Load validation results from JSON."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_panel_1(results, output_path):
    """
    Panel 1: Synthetic Sports Fields
    Shows error distribution across field types and camera angles.
    """
    exp1 = results['experiment_1_synthetic_fields']
    fields = exp1['fields']

    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Chart A: Bar chart of mean errors per field type
    ax_a = fig.add_subplot(gs[0, 0])
    field_names = [fields[k]['field_name'] for k in fields.keys()]
    mean_errors = [fields[k]['mean_error'] for k in fields.keys()]
    std_errors = [fields[k]['std_error'] for k in fields.keys()]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(field_names)))
    bars = ax_a.bar(range(len(field_names)), mean_errors, yerr=std_errors,
                     color=colors, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
    ax_a.set_ylabel('Position Error (%)')
    ax_a.set_title('(A) Position Error by Field Type')
    ax_a.set_xticks(range(len(field_names)))
    ax_a.set_xticklabels([n.replace(' ', '\n') for n in field_names], fontsize=7)
    ax_a.set_ylim(0, 20)
    ax_a.grid(axis='y', alpha=0.3)

    # Chart B: Scatter of error vs camera height
    ax_b = fig.add_subplot(gs[0, 1])
    for field_key in fields.keys():
        field_data = fields[field_key]
        heights = np.array(field_data['camera_heights'])
        errors = np.array(field_data['errors_percent'])
        ax_b.scatter(heights, errors, alpha=0.5, s=30, label=field_data['field_name'])
    ax_b.set_xlabel('Camera Height (m)')
    ax_b.set_ylabel('Position Error (%)')
    ax_b.set_title('(B) Error vs Camera Height')
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=6, loc='upper left')

    # Chart C: 3D scatter of field type, viewing angle, error
    ax_c = fig.add_subplot(gs[0, 2], projection='3d')
    field_indices = []
    angles = []
    errors_3d = []
    colors_3d = []

    color_map = {k: i for i, k in enumerate(fields.keys())}
    for field_key, field_data in fields.items():
        for i, angle in enumerate(field_data['viewing_angles']):
            field_indices.append(color_map[field_key])
            angles.append(angle)
            errors_3d.append(field_data['errors_percent'][i])
            colors_3d.append(color_map[field_key])

    scatter = ax_c.scatter(field_indices, angles, errors_3d, c=colors_3d,
                          cmap='viridis', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax_c.set_xlabel('Field Type', fontsize=8)
    ax_c.set_ylabel('Viewing Angle (°)', fontsize=8)
    ax_c.set_zlabel('Error (%)', fontsize=8)
    ax_c.set_title('(C) Multi-Dimensional Error')
    ax_c.set_xticks(list(color_map.values()))
    ax_c.set_xticklabels([fields[k]['field_name'][:4] for k in fields.keys()], fontsize=7)

    # Chart D: Distribution boxplot
    ax_d = fig.add_subplot(gs[0, 3])
    error_lists = [fields[k]['errors_percent'] for k in fields.keys()]
    bp = ax_d.boxplot(error_lists, labels=[fields[k]['field_name'][:4] for k in fields.keys()],
                      patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_d.set_ylabel('Position Error (%)')
    ax_d.set_title('(D) Error Distribution')
    ax_d.grid(axis='y', alpha=0.3)

    plt.suptitle('Panel 1: Synthetic Sports Field Metric Reconstruction',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Panel 1 saved: {output_path}")

def create_panel_2(results, output_path):
    """
    Panel 2: Temporal Consistency
    Shows scale stability across frames and sequences.
    """
    exp2 = results['experiment_2_temporal_consistency']
    sequences = exp2['sequences']

    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Chart A: Temporal stability per sequence
    ax_a = fig.add_subplot(gs[0, 0])
    stabilities = [s['statistics']['temporal_stability'] for s in sequences]
    seq_indices = range(len(sequences))
    colors_seq = plt.cm.RdYlGn(np.array(stabilities))
    ax_a.bar(seq_indices, stabilities, color=colors_seq, alpha=0.8, edgecolor='black', linewidth=1)
    ax_a.set_ylabel('Temporal Stability (0-1)')
    ax_a.set_xlabel('Sequence Index')
    ax_a.set_title('(A) Temporal Stability per Sequence')
    ax_a.set_ylim(0.7, 1.0)
    ax_a.grid(axis='y', alpha=0.3)

    # Chart B: Scale coefficient of variation
    ax_b = fig.add_subplot(gs[0, 1])
    cvs = [s['statistics']['scale_cv'] for s in sequences]
    ax_b.bar(seq_indices, cvs, color='coral', alpha=0.8, edgecolor='black', linewidth=1)
    ax_b.set_ylabel('Coefficient of Variation')
    ax_b.set_xlabel('Sequence Index')
    ax_b.set_title('(B) Scale Variability (CV)')
    ax_b.grid(axis='y', alpha=0.3)
    ax_b.axhline(y=exp2['aggregate']['mean_scale_cv'], color='red', linestyle='--',
                 label=f'Mean: {exp2["aggregate"]["mean_scale_cv"]:.3f}')
    ax_b.legend(fontsize=8)

    # Chart C: 3D scatter of stability, CV, and variance
    ax_c = fig.add_subplot(gs[0, 2], projection='3d')
    stabilities_arr = np.array(stabilities)
    cvs_arr = np.array(cvs)
    variances = [s['statistics']['scale_variance'] for s in sequences]
    variances_arr = np.array(variances)

    scatter = ax_c.scatter(stabilities_arr, cvs_arr, variances_arr, c=seq_indices,
                          cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax_c.set_xlabel('Stability', fontsize=8)
    ax_c.set_ylabel('CV', fontsize=8)
    ax_c.set_zlabel('Variance', fontsize=8)
    ax_c.set_title('(C) Scale Statistics')
    plt.colorbar(scatter, ax=ax_c, label='Seq#', shrink=0.8)

    # Chart D: Scale consistency histogram
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.hist(stabilities, bins=8, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax_d.axvline(exp2['aggregate']['mean_temporal_stability'], color='red', linestyle='--',
                 linewidth=2, label=f'Mean: {exp2["aggregate"]["mean_temporal_stability"]:.3f}')
    ax_d.set_xlabel('Temporal Stability')
    ax_d.set_ylabel('Frequency')
    ax_d.set_title('(D) Stability Distribution')
    ax_d.legend(fontsize=8)
    ax_d.grid(axis='y', alpha=0.3)

    plt.suptitle('Panel 2: Temporal Consistency in Video Sequences',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Panel 2 saved: {output_path}")

def create_panel_3(results, output_path):
    """
    Panel 3: Multi-View Consistency
    Shows agreement between different camera viewpoints.
    """
    exp3 = results['experiment_3_multiview_consistency']
    scenes = exp3['scenes']

    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Chart A: Cross-view disagreement per scene
    ax_a = fig.add_subplot(gs[0, 0])
    disagreements = [s['agreement_metrics']['mean_disagreement'] for s in scenes]
    scene_indices = range(len(scenes))
    colors_scenes = plt.cm.plasma(np.linspace(0.2, 0.9, len(scenes)))
    bars = ax_a.bar(scene_indices, disagreements, color=colors_scenes, alpha=0.8,
                   edgecolor='black', linewidth=1)
    ax_a.set_ylabel('Mean Disagreement')
    ax_a.set_xlabel('Scene Index')
    ax_a.set_title('(A) Cross-View Disagreement')
    ax_a.grid(axis='y', alpha=0.3)
    ax_a.axhline(y=exp3['aggregate']['mean_cross_view_disagreement'], color='red',
                 linestyle='--', linewidth=2)

    # Chart B: Scatter of disagreement vs percent low error
    ax_b = fig.add_subplot(gs[0, 1])
    for scene in scenes:
        metrics = scene['agreement_metrics']
        ax_b.scatter(metrics['mean_disagreement'], metrics.get('percent_low_error', 80),
                    s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax_b.set_xlabel('Mean Disagreement')
    ax_b.set_ylabel('Percent Low Error (<10%)')
    ax_b.set_title('(B) Error vs Accuracy Rate')
    ax_b.grid(alpha=0.3)

    # Chart C: 3D scatter of disagreement, max, and std
    ax_c = fig.add_subplot(gs[0, 2], projection='3d')
    disagreements_arr = np.array(disagreements)
    max_disagreements = np.array([s['agreement_metrics']['max_disagreement'] for s in scenes])
    std_disagreements = np.array([s['agreement_metrics']['std_disagreement'] for s in scenes])

    scatter = ax_c.scatter(disagreements_arr, max_disagreements, std_disagreements,
                          c=scene_indices, cmap='coolwarm', s=50, alpha=0.7,
                          edgecolors='black', linewidth=0.5)
    ax_c.set_xlabel('Mean', fontsize=8)
    ax_c.set_ylabel('Max', fontsize=8)
    ax_c.set_zlabel('Std Dev', fontsize=8)
    ax_c.set_title('(C) Error Statistics')
    plt.colorbar(scatter, ax=ax_c, label='Scene#', shrink=0.8)

    # Chart D: Consistency score distribution
    ax_d = fig.add_subplot(gs[0, 3])
    consistency_score = 1.0 - np.array(disagreements)
    ax_d.hist(consistency_score, bins=8, color='mediumseagreen', alpha=0.7,
             edgecolor='black', linewidth=1)
    ax_d.axvline(exp3['aggregate']['consistency_score'], color='red', linestyle='--',
                linewidth=2, label=f'Mean: {exp3["aggregate"]["consistency_score"]:.3f}')
    ax_d.set_xlabel('Consistency Score (0-1)')
    ax_d.set_ylabel('Frequency')
    ax_d.set_title('(D) Consistency Distribution')
    ax_d.legend(fontsize=8)
    ax_d.grid(axis='y', alpha=0.3)

    plt.suptitle('Panel 3: Multi-View Coordinate Field Consistency',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Panel 3 saved: {output_path}")

def create_panel_4(results, output_path):
    """
    Panel 4: Terrain Scale Reconstruction
    Shows metric scale recovery accuracy on different terrain types.
    """
    exp4 = results['experiment_4_terrain_scale']
    locations = exp4['locations']

    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Chart A: Bar chart of mean errors per terrain type
    ax_a = fig.add_subplot(gs[0, 0])
    terrain_types = ['rolling_hills', 'mountain', 'plateau']
    terrain_display = ['Rolling Hills', 'Mountain', 'Plateau']
    terrain_errors = [exp4.get(f'{t}_error', {}).get('mean', 0) for t in terrain_types]
    terrain_stds = [exp4.get(f'{t}_error', {}).get('std', 0) for t in terrain_types]

    colors_terrain = plt.cm.Set2(np.linspace(0, 1, len(terrain_types)))
    bars = ax_a.bar(range(len(terrain_display)), terrain_errors, yerr=terrain_stds,
                   color=colors_terrain, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
    ax_a.set_ylabel('Scale Error (%)')
    ax_a.set_title('(A) Scale Error by Terrain Type')
    ax_a.set_xticks(range(len(terrain_display)))
    ax_a.set_xticklabels(terrain_display)
    ax_a.set_ylim(0, 60)
    ax_a.grid(axis='y', alpha=0.3)

    # Chart B: Scatter of error vs terrain relief
    ax_b = fig.add_subplot(gs[0, 1])
    for terrain_type in terrain_types:
        locs = [l for l in locations if l['terrain_type'] == terrain_type]
        if locs:
            reliefs = [l['dem_relief'] for l in locs]
            errors = [l['error_percent'] for l in locs]
            ax_b.scatter(reliefs, errors, alpha=0.6, s=50,
                        label=terrain_type.replace('_', ' ').title())
    ax_b.set_xlabel('DEM Relief (m)')
    ax_b.set_ylabel('Scale Error (%)')
    ax_b.set_title('(B) Error vs Terrain Relief')
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8)

    # Chart C: 3D scatter of terrain type, relief, error
    ax_c = fig.add_subplot(gs[0, 2], projection='3d')
    terrain_indices = [terrain_types.index(l['terrain_type']) for l in locations]
    reliefs_arr = np.array([l['dem_relief'] for l in locations])
    errors_arr = np.array([l['error_percent'] for l in locations])

    scatter = ax_c.scatter(terrain_indices, reliefs_arr, errors_arr, c=terrain_indices,
                          cmap='Set2', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax_c.set_xlabel('Terrain Type', fontsize=8)
    ax_c.set_ylabel('Relief (m)', fontsize=8)
    ax_c.set_zlabel('Error (%)', fontsize=8)
    ax_c.set_title('(C) Multi-Dimensional Scale Error')
    ax_c.set_xticks(range(len(terrain_types)))
    ax_c.set_xticklabels([t[:4] for t in terrain_display], fontsize=7)

    # Chart D: Error distribution histogram
    ax_d = fig.add_subplot(gs[0, 3])
    errors_all = [l['error_percent'] for l in locations]
    ax_d.hist(errors_all, bins=10, color='teal', alpha=0.7, edgecolor='black', linewidth=1)
    ax_d.axvline(exp4['aggregate']['mean_scale_error'], color='red', linestyle='--',
                linewidth=2, label=f'Mean: {exp4["aggregate"]["mean_scale_error"]:.1f}%')
    ax_d.axvline(exp4['aggregate']['median_scale_error'], color='orange', linestyle='--',
                linewidth=2, label=f'Median: {exp4["aggregate"]["median_scale_error"]:.1f}%')
    ax_d.set_xlabel('Scale Error (%)')
    ax_d.set_ylabel('Frequency')
    ax_d.set_title('(D) Error Distribution')
    ax_d.legend(fontsize=8)
    ax_d.grid(axis='y', alpha=0.3)

    plt.suptitle('Panel 4: Terrain Scale Reconstruction Accuracy',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Panel 4 saved: {output_path}")

def main():
    """Generate all panels."""
    results_file = 'coordinate_field_validation_results.json'

    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return

    print("=" * 70)
    print("GENERATING PANELS")
    print("=" * 70)

    results = load_results(results_file)

    print("\nGenerating panels...")
    create_panel_1(results, 'figures/panel_1_synthetic_fields.png')
    create_panel_2(results, 'figures/panel_2_temporal_consistency.png')
    create_panel_3(results, 'figures/panel_3_multiview_consistency.png')
    create_panel_4(results, 'figures/panel_4_terrain_scale.png')

    print("\n" + "=" * 70)
    print("All panels generated successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
