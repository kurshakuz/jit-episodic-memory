#!/usr/bin/env python3
"""
Generate Pareto Efficiency Plot: Storage vs Localization Accuracy

This script creates a scatter plot showing the efficiency-accuracy trade-off
between VLMaps-LSeg, DenseMap-CLIP, and JIT Cascade methods.

Updated to include depth-enabled ablations from full_scale_eval.
"""

import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# POSE-ONLY METHODS (no depth projection)
# Data from full 36-scene evaluation (all methods use the same TEST_QUERIES)
# =============================================================================
methods_pose_only = {
    'JIT Cascade (pose)': {
        'storage_mb': 0.3,
        'loc_1m': 9.4,
        'color': '#2ecc71',  # Green
        'marker': 'o',
        'size': 350
    },
    'VLMaps-LSeg': {
        'storage_mb': 490.2,
        'loc_1m': 4.3,
        'color': '#3498db',  # Blue
        'marker': 's',
        'size': 300
    },
    'DenseMap-CLIP': {
        'storage_mb': 1154.1,
        'loc_1m': 2.2,
        'color': '#e74c3c',  # Red
        'marker': '^',
        'size': 300
    }
}

# =============================================================================
# DEPTH-ENABLED METHODS (with depth projection)
# Data from outputs/full_scale_eval/{extended_baselines_val.json, jit_cascade_depth_val.json}
# These methods use RGB-D and have much higher localization accuracy
# Storage includes depth images (~50MB/scene average) + FAISS index (0.3MB)
# =============================================================================
methods_depth = {
    'JIT + Depth (DBSCAN)': {
        'storage_mb': 50.3,  # Index (0.3MB) + depth images (~50MB)
        'loc_1m': 70.4,      # From jit_cascade_depth_val.json
        'det_5m': 83.7,      # Detection recall
        'color': '#27ae60',  # Dark green
        'marker': '*',
        'size': 500
    },
    'L1 + OWL + Depth': {
        'storage_mb': 50.3,
        'loc_1m': 49.4,     # From extended_baselines_val.json
        'det_5m': 60.3,
        'color': '#f39c12',  # Orange
        'marker': 'D',
        'size': 300
    },
    'Mean Projection': {
        'storage_mb': 50.3,
        'loc_1m': 28.2,     # From extended_baselines_val.json
        'det_5m': 60.3,
        'color': '#9b59b6',  # Purple
        'marker': 'p',
        'size': 300
    },
    'Brute Force + Depth': {
        'storage_mb': 50.3,
        'loc_1m': 65.4,     # From extended_baselines_val.json
        'det_5m': 83.1,
        'color': '#1abc9c',  # Teal
        'marker': 'h',
        'size': 300
    }
}

# Combined for backward compatibility
methods = methods_pose_only.copy()

def create_pareto_plot(output_dir: Path):
    """Create the Pareto efficiency plot."""
    
    # Set up the figure with a clean style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot each method
    for name, data in methods.items():
        ax.scatter(
            data['storage_mb'], 
            data['loc_1m'],
            c=data['color'],
            marker=data['marker'],
            s=data['size'],
            label=name,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        
        # Add annotation with offset
        if 'JIT' in name:
            offset = (15, 10)
            ha = 'left'
        elif name == 'VLMaps-LSeg':
            offset = (10, -20)
            ha = 'left'
        else:  # DenseMap-CLIP
            offset = (-10, 15)
            ha = 'right'
            
        ax.annotate(
            f'{name}\n({data["storage_mb"]:.1f} MB, {data["loc_1m"]:.1f}%)',
            xy=(data['storage_mb'], data['loc_1m']),
            xytext=offset,
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5)
        )
    
    # Draw Pareto frontier arrow from JIT to indicate dominance
    ax.annotate(
        '',
        xy=(0.1, 11),  # Arrow end (ideal corner)
        xytext=(0.3, 9.4),  # Arrow start (JIT position)
        arrowprops=dict(
            arrowstyle='->',
            color='#2ecc71',
            lw=2,
            ls='--'
        )
    )
    
    # Add "Pareto Optimal" region shading
    ax.fill_between(
        [0.01, 1],  # x range for optimal region
        [0, 0],     # y lower bound
        [15, 15],   # y upper bound
        alpha=0.1,
        color='#2ecc71',
        label='_nolegend_'
    )
    
    # Add text for optimal region
    ax.text(
        0.15, 12.5,
        '← Pareto\n   Optimal',
        fontsize=11,
        fontweight='bold',
        color='#27ae60',
        ha='center',
        va='center'
    )
    
    # Configure axes
    ax.set_xscale('log')
    ax.set_xlabel('Storage (MB) — Log Scale', fontsize=13, fontweight='bold')
    ax.set_ylabel('Localization Accuracy (Loc@1m, %)', fontsize=13, fontweight='bold')
    ax.set_title('Efficiency-Accuracy Trade-off: Storage vs Localization', 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Set axis limits
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(0, 15)
    
    # Customize grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add horizontal reference lines
    ax.axhline(y=5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add vertical reference lines for storage thresholds
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=1000, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add legend
    legend = ax.legend(
        loc='lower right',
        fontsize=11,
        framealpha=0.95,
        edgecolor='gray'
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Add efficiency arrows
    ax.annotate(
        'Better ->',
        xy=(0.12, 0.5),
        fontsize=9,
        color='gray',
        ha='left',
        rotation=90
    )
    ax.annotate(
        '← Better',
        xy=(2000, 0.5),
        fontsize=9,
        color='gray',
        ha='right'
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'pareto_efficiency_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_dir / 'pareto_efficiency_plot.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()
    
    return output_path


def create_pareto_plot_with_improvements(output_dir: Path):
    """Create an enhanced version showing improvement factors."""
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Plot each method
    for name, data in methods.items():
        ax.scatter(
            data['storage_mb'], 
            data['loc_1m'],
            c=data['color'],
            marker=data['marker'],
            s=data['size'],
            label=name,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
    
    # Add improvement arrows from baselines to JIT
    jit = methods['JIT Cascade (pose)']
    
    for baseline_name in ['VLMaps-LSeg', 'DenseMap-CLIP']:
        baseline = methods[baseline_name]
        
        # Draw arrow from baseline to JIT
        ax.annotate(
            '',
            xy=(jit['storage_mb'], jit['loc_1m']),
            xytext=(baseline['storage_mb'], baseline['loc_1m']),
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                lw=1.5,
                ls='-',
                connectionstyle='arc3,rad=0.2',
                alpha=0.6
            )
        )
    
    # Add improvement labels
    # Storage improvement vs DenseMap
    storage_improvement = methods['DenseMap-CLIP']['storage_mb'] / jit['storage_mb']
    ax.annotate(
        f'{storage_improvement:.0f}× less\nstorage',
        xy=(10, 5.5),
        fontsize=10,
        fontweight='bold',
        color='#2ecc71',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#2ecc71')
    )
    
    # Accuracy improvement vs DenseMap
    acc_improvement = jit['loc_1m'] / methods['DenseMap-CLIP']['loc_1m']
    ax.annotate(
        f'{acc_improvement:.1f}× better\naccuracy',
        xy=(3, 7),
        fontsize=10,
        fontweight='bold',
        color='#2ecc71',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#2ecc71')
    )
    
    # Method labels
    for name, data in methods.items():
        if 'JIT' in name:
            offset = (20, 15)
        elif name == 'VLMaps-LSeg':
            offset = (15, -25)
        else:
            offset = (-15, 20)
            
        ax.annotate(
            name,
            xy=(data['storage_mb'], data['loc_1m']),
            xytext=offset,
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=data['color'], alpha=0.2, edgecolor=data['color']),
        )
    
    # Shade the Pareto optimal region
    ax.fill_betweenx(
        [7, 15],
        [0.01, 0.01],
        [2, 2],
        alpha=0.15,
        color='#2ecc71',
        label='_nolegend_'
    )
    ax.text(0.2, 13, 'Pareto\nOptimal\nRegion', fontsize=10, fontweight='bold', 
            color='#27ae60', ha='center', va='center')
    
    # Configure axes
    ax.set_xscale('log')
    ax.set_xlabel('Storage (MB) — Log Scale', fontsize=14, fontweight='bold')
    ax.set_ylabel('Localization Accuracy (Loc@1m, %)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 5: Efficiency-Accuracy Trade-off\nJIT Cascade dominates the Pareto frontier', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(0, 15)
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'pareto_efficiency_plot_enhanced.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()
    
    return output_path


def create_depth_ablation_pareto_plot(output_dir: Path):
    """
    Create a Pareto plot showing both pose-only and depth-enabled methods.
    
    This visualizes the ablation study showing how depth projection
    dramatically improves localization accuracy.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # --- Plot POSE-ONLY methods (lower accuracy) ---
    for name, data in methods_pose_only.items():
        ax.scatter(
            data['storage_mb'], 
            data['loc_1m'],
            c=data['color'],
            marker=data['marker'],
            s=data['size'],
            label=name,
            edgecolors='white',
            linewidths=2,
            zorder=5,
            alpha=0.7
        )
    
    # --- Plot DEPTH-ENABLED methods (higher accuracy) ---
    for name, data in methods_depth.items():
        ax.scatter(
            data['storage_mb'], 
            data['loc_1m'],
            c=data['color'],
            marker=data['marker'],
            s=data['size'],
            label=name,
            edgecolors='white',
            linewidths=2,
            zorder=6
        )
    
    # --- Add arrow showing depth improvement for JIT ---
    jit_pose = methods_pose_only['JIT Cascade (pose)']
    jit_depth = methods_depth['JIT + Depth (DBSCAN)']
    
    ax.annotate(
        '',
        xy=(jit_depth['storage_mb'], jit_depth['loc_1m']),
        xytext=(jit_pose['storage_mb'], jit_pose['loc_1m']),
        arrowprops=dict(
            arrowstyle='-|>',
            color='#2ecc71',
            lw=3,
            mutation_scale=20
        ),
        zorder=4
    )
    
    # Add improvement label
    improvement = jit_depth['loc_1m'] / jit_pose['loc_1m']
    ax.annotate(
        f'+{jit_depth["loc_1m"] - jit_pose["loc_1m"]:.0f}pp\n({improvement:.1f}×)',
        xy=(0.35, 40),
        fontsize=12,
        fontweight='bold',
        color='#27ae60',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='#27ae60', lw=2)
    )
    
    # --- Add method labels ---
    label_positions = {
        'JIT Cascade (pose)': (15, -20),
        'VLMaps-LSeg': (20, -15),
        'DenseMap-CLIP': (-20, 15),
        'JIT + Depth (DBSCAN)': (25, 10),
        'L1 + OWL + Depth': (25, -15),
        'Mean Projection': (25, 10),
        'Brute Force + Depth': (-25, 15),
    }
    
    all_methods = {**methods_pose_only, **methods_depth}
    for name, data in all_methods.items():
        offset = label_positions.get(name, (15, 10))
        ha = 'left' if offset[0] > 0 else 'right'
        
        # Shorter labels for depth methods
        display_name = name.replace(' + Depth', '+D').replace('(DBSCAN)', '')
        
        ax.annotate(
            f'{display_name}\n{data["loc_1m"]:.1f}%',
            xy=(data['storage_mb'], data['loc_1m']),
            xytext=offset,
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=data['color']),
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.5)
        )
    
    # --- Add region labels ---
    ax.axhspan(0, 15, xmin=0, xmax=1, alpha=0.05, color='gray')
    ax.axhspan(15, 80, xmin=0, xmax=1, alpha=0.05, color='#27ae60')
    
    ax.text(600, 7.5, 'POSE-ONLY\n(RGB only)', fontsize=12, fontweight='bold',
            color='gray', ha='center', va='center', alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.text(600, 55, 'DEPTH-ENABLED\n(RGB-D)', fontsize=12, fontweight='bold',
            color='#27ae60', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # --- Add horizontal separator ---
    ax.axhline(y=15, color='gray', linestyle='--', alpha=0.5, lw=2)
    
    # --- Configure axes ---
    ax.set_xscale('log')
    ax.set_xlabel('Storage (MB) — Log Scale', fontsize=14, fontweight='bold')
    ax.set_ylabel('Localization Accuracy (Loc@1m, %)', fontsize=14, fontweight='bold')
    ax.set_title('Depth Ablation: Pose-Only vs Depth-Enabled Methods\n'
                 'DBSCAN clustering with depth projection yields +61pp improvement',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(0, 80)
    
    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # --- Create custom legend with two groups ---
    from matplotlib.lines import Line2D
    
    legend_pose = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=12, label='JIT Cascade (pose)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=10, label='VLMaps-LSeg'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c', markersize=10, label='DenseMap-CLIP'),
    ]
    
    legend_depth = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#27ae60', markersize=16, label='JIT + Depth (DBSCAN)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#f39c12', markersize=10, label='L1 + OWL + Depth'),
        Line2D([0], [0], marker='p', color='w', markerfacecolor='#9b59b6', markersize=11, label='Mean Projection'),
        Line2D([0], [0], marker='h', color='w', markerfacecolor='#1abc9c', markersize=11, label='Brute Force + Depth'),
    ]
    
    leg1 = ax.legend(handles=legend_pose, loc='lower right', title='Pose-Only', 
                      fontsize=10, title_fontsize=11, framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=legend_depth, loc='center right', title='Depth-Enabled',
                      fontsize=10, title_fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'pareto_depth_ablation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    pdf_path = output_dir / 'pareto_depth_ablation.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()
    
    return output_path


def create_combined_pareto_plot(output_dir: Path):
    """
    Create a single comprehensive Pareto plot with ALL methods.
    
    Uses storage (MB) as the x-axis for consistency with other Pareto plots.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # All methods with storage data
    # Note: Depth-enabled methods require storing depth images (~50MB/scene avg)
    # JIT methods use FAISS index only (0.3MB), but depth methods need RGB-D frames
    all_methods = {
        # Pose-only (no depth storage needed)
        'JIT Cascade (pose)': {'storage_mb': 0.3, 'loc_1m': 9.4, 'color': '#2ecc71', 'marker': 'o', 'size': 300},
        'VLMaps-LSeg': {'storage_mb': 490.2, 'loc_1m': 4.3, 'color': '#3498db', 'marker': 's', 'size': 250},
        'DenseMap-CLIP': {'storage_mb': 1154.1, 'loc_1m': 2.2, 'color': '#e74c3c', 'marker': '^', 'size': 250},
        # Depth-enabled (require depth image storage ~50MB + index)
        'JIT + Depth': {'storage_mb': 50.3, 'loc_1m': 70.4, 'color': '#27ae60', 'marker': '*', 'size': 500},
        'L1 + OWL + Depth': {'storage_mb': 50.3, 'loc_1m': 49.4, 'color': '#f39c12', 'marker': 'D', 'size': 300},
        'Mean Projection': {'storage_mb': 50.3, 'loc_1m': 28.2, 'color': '#9b59b6', 'marker': 'p', 'size': 300},
        'Brute Force + Depth': {'storage_mb': 50.3, 'loc_1m': 65.4, 'color': '#1abc9c', 'marker': 'h', 'size': 300},
    }
    
    # Plot all methods
    for name, data in all_methods.items():
        ax.scatter(
            data['storage_mb'],
            data['loc_1m'],
            c=data['color'],
            marker=data['marker'],
            s=data['size'],
            label=name,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        
        # Add labels
        if 'Depth' in name:
            offset = (15, 10) if 'JIT' in name else (15, -10)
        else:
            offset = (15, -15) if name != 'DenseMap-CLIP' else (-15, 15)
        ha = 'left' if offset[0] > 0 else 'right'
        ax.annotate(
            f'{name}\n{data["loc_1m"]:.1f}%',
            xy=(data['storage_mb'], data['loc_1m']),
            xytext=offset,
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=data['color']),
        )
    
    # Configure axes
    ax.set_xscale('log')
    ax.set_xlabel('Storage (MB) — Log Scale', fontsize=14, fontweight='bold')
    ax.set_ylabel('Localization Accuracy (Loc@1m, %)', fontsize=14, fontweight='bold')
    ax.set_title('Storage-Accuracy Trade-off: All Methods Compared', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    ax.legend(loc='center right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    output_path = output_dir / 'pareto_all_methods.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()
    return output_path


if __name__ == '__main__':
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating Pareto Efficiency Plots...")
    print("=" * 50)
    
    # Generate all versions
    create_pareto_plot(output_dir)
    create_pareto_plot_with_improvements(output_dir)
    create_depth_ablation_pareto_plot(output_dir)
    create_combined_pareto_plot(output_dir)
    
    print("=" * 50)
    print("Done! Figures saved to:", output_dir)
