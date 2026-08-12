#!/usr/bin/env python3
"""
Generate Latency vs Accuracy Plot with Corrected Data.

Creates a scatter plot showing the trade-off between query latency and 
localization accuracy for all three methods.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Corrected data from filtered evaluation (all methods on same 278 TEST_QUERIES)
methods = {
    'JIT Cascade': {
        'latency_ms': 916.2,
        'loc_05m': 4.7,
        'loc_1m': 9.4,
        'loc_2m': 15.1,
        'loc_3m': 24.5,
        'color': '#2ecc71',  # Green
        'marker': 'o',
    },
    'VLMaps-LSeg': {
        'latency_ms': 61.6,
        'loc_05m': 2.2,
        'loc_1m': 4.3,
        'loc_2m': 10.4,
        'loc_3m': 18.7,
        'color': '#3498db',  # Blue
        'marker': 's',
    },
    'DenseMap-CLIP': {
        'latency_ms': 190.9,
        'loc_05m': 0.7,
        'loc_1m': 2.2,
        'loc_2m': 6.5,
        'loc_3m': 10.8,
        'color': '#e74c3c',  # Red
        'marker': '^',
    }
}


def create_latency_vs_accuracy_plot(output_dir: Path):
    """Create latency vs accuracy scatter plot."""
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot each method
    for name, data in methods.items():
        ax.scatter(
            data['latency_ms'], 
            data['loc_1m'],
            c=data['color'],
            marker=data['marker'],
            s=400,
            label=name,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        
        # Add annotation
        if name == 'JIT Cascade':
            offset = (-80, 15)
            ha = 'right'
        elif name == 'VLMaps-LSeg':
            offset = (15, -20)
            ha = 'left'
        else:  # DenseMap-CLIP
            offset = (15, 10)
            ha = 'left'
            
        ax.annotate(
            f'{name}\n({data["latency_ms"]:.0f}ms, {data["loc_1m"]:.1f}%)',
            xy=(data['latency_ms'], data['loc_1m']),
            xytext=offset,
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor=data['color'], alpha=0.9)
        )
    
    # Labels and title
    ax.set_xlabel('Query Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Localization Accuracy @ 1m (%)', fontsize=12, fontweight='bold')
    ax.set_title('Query Latency vs Localization Accuracy\n(36 HM3D Scenes, 278 Queries)', 
                 fontsize=14, fontweight='bold')
    
    # Add ideal region annotation
    ax.annotate(
        '← Ideal\n(low latency,\nhigh accuracy)',
        xy=(100, 9),
        fontsize=10,
        color='gray',
        fontstyle='italic',
        ha='center'
    )
    
    # Set axis limits
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 12)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = output_dir / 'latency_vs_accuracy.png'
    pdf_path = output_dir / 'latency_vs_accuracy.pdf'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    
    plt.close()


def create_multi_threshold_plot(output_dir: Path):
    """Create plot showing accuracy at multiple distance thresholds."""
    
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = ['@0.5m', '@1m', '@2m', '@3m']
    x = np.arange(len(thresholds))
    width = 0.25
    
    # Data for each method
    jit_data = [methods['JIT Cascade']['loc_05m'], methods['JIT Cascade']['loc_1m'],
                methods['JIT Cascade']['loc_2m'], methods['JIT Cascade']['loc_3m']]
    vlmap_data = [methods['VLMaps-LSeg']['loc_05m'], methods['VLMaps-LSeg']['loc_1m'],
                  methods['VLMaps-LSeg']['loc_2m'], methods['VLMaps-LSeg']['loc_3m']]
    dense_data = [methods['DenseMap-CLIP']['loc_05m'], methods['DenseMap-CLIP']['loc_1m'],
                  methods['DenseMap-CLIP']['loc_2m'], methods['DenseMap-CLIP']['loc_3m']]
    
    # Create bars
    bars1 = ax.bar(x - width, vlmap_data, width, label='VLMaps-LSeg', 
                   color=methods['VLMaps-LSeg']['color'], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, dense_data, width, label='DenseMap-CLIP', 
                   color=methods['DenseMap-CLIP']['color'], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, jit_data, width, label='JIT Cascade', 
                   color=methods['JIT Cascade']['color'], edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Labels and title
    ax.set_xlabel('Distance Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Localization Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Localization Accuracy at Multiple Thresholds\n(36 HM3D Scenes, 278 Queries per Method)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, fontsize=11)
    ax.set_ylim(0, 30)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save
    png_path = output_dir / 'accuracy_by_threshold.png'
    pdf_path = output_dir / 'accuracy_by_threshold.pdf'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    
    plt.close()


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "figures"
    
    print("Generating Latency vs Accuracy Plots (Corrected Data)...")
    print("=" * 50)
    
    create_latency_vs_accuracy_plot(output_dir)
    create_multi_threshold_plot(output_dir)
    
    print("=" * 50)
    print(f"Done! Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
