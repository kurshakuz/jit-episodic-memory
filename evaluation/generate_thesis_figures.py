#!/usr/bin/env python3
"""
Generate thesis figures for JIT Episodic Memory.

Figure 1: Spatial Denoiser - DBSCAN as geometric super-resolution
Figure 2: L2 Paradox - Two-path architecture diagram
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
from pathlib import Path

# Set up matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def generate_spatial_denoiser_figure(output_dir: Path):
    """
    Generate Figure 4: Spatial Denoiser / Geometric Super-Resolution
    
    Shows:
    - Panel A: Raw noisy depth projections (scattered points)
    - Panel B: DBSCAN clusters with clean centroids
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simulate a room layout (top-down view)
    room_width = 8  # meters
    room_height = 6  # meters
    
    # True object location (e.g., a couch)
    true_object = np.array([5.0, 3.5])
    object_size = (1.5, 0.8)  # couch dimensions
    
    # Wall positions
    walls = [
        [(0, 0), (room_width, 0)],      # bottom wall
        [(room_width, 0), (room_width, room_height)],  # right wall
        [(room_width, room_height), (0, room_height)], # top wall
        [(0, room_height), (0, 0)],     # left wall
    ]
    
    # Generate simulated depth projections
    np.random.seed(42)
    n_frames = 50
    
    # Good projections (near true object) - represents majority
    good_projections = true_object + np.random.randn(n_frames, 2) * 0.4
    
    # Noisy projections (depth errors, through walls, etc.)
    n_noise = 20
    noise_projections = np.array([
        # Through-wall projections (depth penetrated geometry)
        [room_width + 0.5 + np.random.rand() * 1.5, 3 + np.random.randn() * 0.5],
        [room_width + 0.3 + np.random.rand() * 1.0, 4 + np.random.randn() * 0.3],
        [room_width + 0.8, 3.5],
        [room_width + 1.2, 3.2],
        [room_width + 0.6, 4.1],
        # Far outliers (depth sensor errors)
        [2.0, 1.5],
        [1.5, 4.5],
        [3.0, 5.5],
        [7.0, 1.0],
        [6.5, 5.0],
        # Random scatter
        [4.0, 2.0],
        [6.5, 2.5],
        [3.5, 4.0],
        [5.5, 1.5],
        [4.5, 5.0],
        [2.5, 3.0],
        [7.0, 4.0],
        [3.0, 2.5],
        [6.0, 4.5],
        [4.0, 4.5],
    ])
    
    all_projections = np.vstack([good_projections, noise_projections])
    
    # ============ Panel A: Raw Projections ============
    ax1 = axes[0]
    ax1.set_xlim(-0.5, room_width + 2.5)
    ax1.set_ylim(-0.5, room_height + 0.5)
    ax1.set_aspect('equal')
    ax1.set_title('(A) Raw Single-View Depth Projections', fontweight='bold', fontsize=14)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    
    # Draw room walls
    for wall in walls:
        ax1.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 
                'k-', linewidth=3, solid_capstyle='round')
    
    # Draw true object (couch) as rectangle
    couch_rect = plt.Rectangle(
        (true_object[0] - object_size[0]/2, true_object[1] - object_size[1]/2),
        object_size[0], object_size[1],
        facecolor='lightblue', edgecolor='navy', linewidth=2,
        label='True Object (Couch)'
    )
    ax1.add_patch(couch_rect)
    
    # Draw ground truth center
    ax1.scatter([true_object[0]], [true_object[1]], 
               s=200, c='green', marker='*', zorder=10,
               edgecolors='darkgreen', linewidths=1.5,
               label='Ground Truth Center')
    
    # Draw ALL projections as scattered red dots
    ax1.scatter(all_projections[:, 0], all_projections[:, 1],
               s=40, c='red', alpha=0.6, edgecolors='darkred', linewidths=0.5,
               label=f'Depth Projections (n={len(all_projections)})')
    
    # Highlight through-wall errors with circles
    for i in range(5):  # First 5 noise points are through-wall
        ax1.scatter([noise_projections[i, 0]], [noise_projections[i, 1]],
                   s=100, facecolors='none', edgecolors='orange', linewidths=2)
    
    # Add annotation for through-wall errors
    ax1.annotate('Through-wall\nerrors', 
                xy=(room_width + 0.8, 3.5), xytext=(room_width + 1.8, 4.8),
                fontsize=10, color='darkorange',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add annotation for scatter
    ax1.annotate('Depth noise\nscatter', 
                xy=(2.0, 1.5), xytext=(0.5, 0.5),
                fontsize=10, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ============ Panel B: DBSCAN Clustering ============
    ax2 = axes[1]
    ax2.set_xlim(-0.5, room_width + 2.5)
    ax2.set_ylim(-0.5, room_height + 0.5)
    ax2.set_aspect('equal')
    ax2.set_title('(B) After DBSCAN Clustering (ε=1.0m)', fontweight='bold', fontsize=14)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    
    # Draw room walls
    for wall in walls:
        ax2.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 
                'k-', linewidth=3, solid_capstyle='round')
    
    # Draw true object
    couch_rect2 = plt.Rectangle(
        (true_object[0] - object_size[0]/2, true_object[1] - object_size[1]/2),
        object_size[0], object_size[1],
        facecolor='lightblue', edgecolor='navy', linewidth=2
    )
    ax2.add_patch(couch_rect2)
    
    # Draw ground truth
    ax2.scatter([true_object[0]], [true_object[1]], 
               s=200, c='green', marker='*', zorder=10,
               edgecolors='darkgreen', linewidths=1.5,
               label='Ground Truth')
    
    # Simulate DBSCAN results
    # Main cluster (inliers around true object)
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=1.0, min_samples=3).fit(all_projections)
    labels = clustering.labels_
    
    # Find the main cluster (largest one near object)
    unique_labels = set(labels)
    main_cluster_label = -1
    main_cluster_size = 0
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        cluster_points = all_projections[mask]
        cluster_center = cluster_points.mean(axis=0)
        dist_to_object = np.linalg.norm(cluster_center - true_object)
        if dist_to_object < 2.0 and np.sum(mask) > main_cluster_size:
            main_cluster_label = label
            main_cluster_size = np.sum(mask)
    
    # Draw noise points (rejected) as faded
    noise_mask = labels == -1
    ax2.scatter(all_projections[noise_mask, 0], all_projections[noise_mask, 1],
               s=30, c='gray', alpha=0.3, marker='x',
               label=f'Rejected Outliers (n={np.sum(noise_mask)})')
    
    # Draw main cluster
    if main_cluster_label >= 0:
        main_mask = labels == main_cluster_label
        main_points = all_projections[main_mask]
        centroid = main_points.mean(axis=0)
        
        # Draw cluster points
        ax2.scatter(main_points[:, 0], main_points[:, 1],
                   s=50, c='blue', alpha=0.5, edgecolors='darkblue', linewidths=0.5,
                   label=f'Main Cluster (n={len(main_points)})')
        
        # Draw cluster boundary (ellipse)
        from matplotlib.patches import Ellipse
        cov = np.cov(main_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2-sigma ellipse
        ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle,
                         facecolor='blue', alpha=0.1, edgecolor='blue', linewidth=2,
                         linestyle='--')
        ax2.add_patch(ellipse)
        
        # Draw centroid (the output location)
        ax2.scatter([centroid[0]], [centroid[1]], 
                   s=300, c='blue', marker='o', zorder=10,
                   edgecolors='darkblue', linewidths=2,
                   label=f'Cluster Centroid')
        
        # Calculate and show error reduction
        single_view_error = np.mean([np.linalg.norm(p - true_object) for p in all_projections])
        cluster_error = np.linalg.norm(centroid - true_object)
        
        ax2.annotate(f'Centroid Error: {cluster_error:.2f}m\n(vs {single_view_error:.2f}m avg)',
                    xy=centroid, xytext=(centroid[0]-2.5, centroid[1]+1.5),
                    fontsize=10, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue'))
    
    # Draw other small clusters (noise clusters)
    for label in unique_labels:
        if label == -1 or label == main_cluster_label:
            continue
        mask = labels == label
        cluster_points = all_projections[mask]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   s=30, c='orange', alpha=0.5, marker='s')
    
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Figure 4: Geometric Super-Resolution via Multi-View Clustering',
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'fig4_spatial_denoiser.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_spatial_denoiser.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: fig4_spatial_denoiser.png/pdf")


def generate_l2_paradox_figure(output_dir: Path):
    """
    Generate Figure: L2 Paradox - Two-Path Architecture
    
    Shows the trade-off between:
    - Path 1 (Pose-First): Fast/Navigational
    - Path 2 (Depth-First): Precise/Manipulation
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    color_l1 = '#3498db'  # Blue
    color_l2_pose = '#2ecc71'  # Green
    color_l2_depth = '#e74c3c'  # Red
    color_l3 = '#9b59b6'  # Purple
    color_output = '#f39c12'  # Orange
    
    # ===== Title =====
    ax.text(7, 7.5, 'JIT Retrieval Cascade: Two Operating Modes',
           fontsize=16, fontweight='bold', ha='center', va='center')
    
    # ===== Shared Input =====
    # Query input
    query_box = FancyBboxPatch((0.5, 4.2), 2, 1.2, boxstyle='round,pad=0.1',
                               facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(query_box)
    ax.text(1.5, 4.8, 'Query:\n"Find the couch"', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Memory Bank
    memory_box = FancyBboxPatch((0.5, 2.2), 2, 1.2, boxstyle='round,pad=0.1',
                                facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(memory_box)
    ax.text(1.5, 2.8, 'Memory Bank\n(CLIP Embeddings)', fontsize=9, ha='center', va='center')
    
    # ===== L1: Semantic Filter (Shared) =====
    l1_box = FancyBboxPatch((3.5, 3.2), 2, 1.2, boxstyle='round,pad=0.1',
                            facecolor=color_l1, edgecolor='#2980b9', linewidth=2, alpha=0.8)
    ax.add_patch(l1_box)
    ax.text(4.5, 3.8, 'L1: CLIP + FAISS\nRetrieval', fontsize=10, ha='center', va='center',
           fontweight='bold', color='white')
    ax.text(4.5, 3.1, '~5ms', fontsize=8, ha='center', va='center', color='white', style='italic')
    
    # Arrows to L1
    ax.annotate('', xy=(3.5, 3.8), xytext=(2.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.annotate('', xy=(3.5, 3.8), xytext=(2.5, 3.0),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # ===== SPLIT POINT =====
    ax.plot([5.5, 6.5], [3.8, 3.8], 'k-', lw=2)
    ax.scatter([6.0], [3.8], s=100, c='black', zorder=10)
    ax.text(6.0, 3.3, 'BRANCH', fontsize=9, ha='center', fontweight='bold')
    
    # ===== PATH 1: Pose-First (Top) =====
    # Box for path label
    path1_label = FancyBboxPatch((6.8, 5.8), 6.5, 1.8, boxstyle='round,pad=0.1',
                                 facecolor='#d5f5e3', edgecolor=color_l2_pose, linewidth=3, alpha=0.3)
    ax.add_patch(path1_label)
    ax.text(10.0, 7.2, 'PATH 1: Pose-First (Current Implementation)',
           fontsize=12, fontweight='bold', ha='center', color='#1e8449')
    ax.text(10.0, 6.8, 'Fast / Navigational', fontsize=10, ha='center', 
           color='#27ae60', style='italic')
    
    # L2 Pose Clustering
    l2_pose = FancyBboxPatch((7, 5.2), 2, 1.0, boxstyle='round,pad=0.1',
                             facecolor=color_l2_pose, edgecolor='#27ae60', linewidth=2, alpha=0.8)
    ax.add_patch(l2_pose)
    ax.text(8, 5.7, 'L2: DBSCAN\n(Camera Poses)', fontsize=9, ha='center', va='center',
           fontweight='bold', color='white')
    ax.text(8, 5.0, '~15ms', fontsize=8, ha='center', va='center', color='white', style='italic')
    
    # L3 Detection
    l3_pose = FancyBboxPatch((9.5, 5.2), 2, 1.0, boxstyle='round,pad=0.1',
                             facecolor=color_l3, edgecolor='#8e44ad', linewidth=2, alpha=0.8)
    ax.add_patch(l3_pose)
    ax.text(10.5, 5.7, 'L3: OWL-ViT\nVerification', fontsize=9, ha='center', va='center',
           fontweight='bold', color='white')
    ax.text(10.5, 5.0, '~400ms', fontsize=8, ha='center', va='center', color='white', style='italic')
    
    # Output 1
    out1 = FancyBboxPatch((12, 5.2), 1.5, 1.0, boxstyle='round,pad=0.1',
                          facecolor=color_output, edgecolor='#d68910', linewidth=2, alpha=0.8)
    ax.add_patch(out1)
    ax.text(12.75, 5.7, 'Object\nLocation', fontsize=9, ha='center', va='center',
           fontweight='bold', color='white')
    
    # Path 1 arrows
    ax.annotate('', xy=(7, 5.7), xytext=(6.0, 4.2),
               arrowprops=dict(arrowstyle='->', color=color_l2_pose, lw=2.5))
    ax.annotate('', xy=(9.5, 5.7), xytext=(9, 5.7),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.annotate('', xy=(12, 5.7), xytext=(11.5, 5.7),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # ===== PATH 2: Depth-First (Bottom) =====
    # Box for path label
    path2_label = FancyBboxPatch((6.8, 0.8), 6.5, 2.2, boxstyle='round,pad=0.1',
                                 facecolor='#fadbd8', edgecolor=color_l2_depth, linewidth=3, alpha=0.3)
    ax.add_patch(path2_label)
    ax.text(10.0, 2.7, 'PATH 2: Depth-First (Future Work)',
           fontsize=12, fontweight='bold', ha='center', color='#922b21')
    ax.text(10.0, 2.3, 'Precise / Manipulation', fontsize=10, ha='center',
           color='#c0392b', style='italic')
    
    # L3 Detection (first in this path)
    l3_depth = FancyBboxPatch((7, 1.4), 1.8, 1.0, boxstyle='round,pad=0.1',
                              facecolor=color_l3, edgecolor='#8e44ad', linewidth=2, alpha=0.8)
    ax.add_patch(l3_depth)
    ax.text(7.9, 1.9, 'L3: OWL-ViT\nDetection', fontsize=8, ha='center', va='center',
           fontweight='bold', color='white')
    ax.text(7.9, 1.2, '~2000ms', fontsize=7, ha='center', va='center', color='white', style='italic')
    
    # Depth Projection
    proj_depth = FancyBboxPatch((9.0, 1.4), 1.8, 1.0, boxstyle='round,pad=0.1',
                                facecolor='#85c1e9', edgecolor='#2980b9', linewidth=2, alpha=0.8)
    ax.add_patch(proj_depth)
    ax.text(9.9, 1.9, 'Depth\nProjection', fontsize=8, ha='center', va='center',
           fontweight='bold', color='#1a5276')
    ax.text(9.9, 1.2, '~100ms', fontsize=7, ha='center', va='center', color='#1a5276', style='italic')
    
    # L2 Object Clustering
    l2_depth = FancyBboxPatch((11.0, 1.4), 1.8, 1.0, boxstyle='round,pad=0.1',
                              facecolor=color_l2_depth, edgecolor='#a93226', linewidth=2, alpha=0.8)
    ax.add_patch(l2_depth)
    ax.text(11.9, 1.9, 'DBSCAN\n(Object Poses)', fontsize=8, ha='center', va='center',
           fontweight='bold', color='white')
    ax.text(11.9, 1.2, '~20ms', fontsize=7, ha='center', va='center', color='white', style='italic')
    
    # Path 2 arrows
    ax.annotate('', xy=(7, 1.9), xytext=(6.0, 3.4),
               arrowprops=dict(arrowstyle='->', color=color_l2_depth, lw=2.5))
    ax.annotate('', xy=(9.0, 1.9), xytext=(8.8, 1.9),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.annotate('', xy=(11.0, 1.9), xytext=(10.8, 1.9),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Output 2 (reuses same box style)
    out2 = FancyBboxPatch((12.0, 0.3), 1.5, 0.8, boxstyle='round,pad=0.1',
                          facecolor=color_output, edgecolor='#d68910', linewidth=2, alpha=0.8)
    ax.add_patch(out2)
    ax.text(12.75, 0.7, 'Precise\nLocation', fontsize=8, ha='center', va='center',
           fontweight='bold', color='white')
    ax.annotate('', xy=(12.7, 0.8), xytext=(12.3, 1.4),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # ===== Comparison Box =====
    ax.text(7, 0.15, 'Latency Trade-off:', fontsize=10, fontweight='bold')
    
    # ===== Legend =====
    legend_elements = [
        mpatches.Patch(facecolor=color_l1, edgecolor='#2980b9', label='L1: Semantic Retrieval'),
        mpatches.Patch(facecolor=color_l2_pose, edgecolor='#27ae60', label='L2: Geometric Clustering'),
        mpatches.Patch(facecolor=color_l3, edgecolor='#8e44ad', label='L3: Visual Verification'),
        mpatches.Patch(facecolor=color_output, edgecolor='#d68910', label='Output Location'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
             framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'fig_l2_paradox_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_l2_paradox_architecture.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: fig_l2_paradox_architecture.png/pdf")


def generate_jit_architecture_diagram(output_dir: Path):
    """
    Generate the main JIT Architecture diagram with the two-path visualization.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#ecf0f1',
        'l1': '#3498db',
        'l2': '#2ecc71',
        'l3': '#9b59b6',
        'output': '#f39c12',
        'memory': '#95a5a6',
    }
    
    # ===== Title =====
    ax.text(8, 9.5, 'JIT Retrieval Cascade Architecture',
           fontsize=18, fontweight='bold', ha='center')
    
    # ===== Memory Bank (Left side) =====
    memory_box = FancyBboxPatch((0.5, 3), 2.5, 4, boxstyle='round,pad=0.2',
                                facecolor=colors['memory'], edgecolor='#7f8c8d', 
                                linewidth=2, alpha=0.6)
    ax.add_patch(memory_box)
    ax.text(1.75, 6.5, 'Episodic Memory', fontsize=12, ha='center', fontweight='bold')
    ax.text(1.75, 5.8, '────────────', fontsize=10, ha='center', color='#7f8c8d')
    ax.text(1.75, 5.3, 'RGB Images', fontsize=10, ha='center')
    ax.text(1.75, 4.7, 'Depth Maps', fontsize=10, ha='center')
    ax.text(1.75, 4.1, 'Camera Poses', fontsize=10, ha='center')
    ax.text(1.75, 3.5, 'CLIP Embeddings', fontsize=10, ha='center')
    
    # ===== Query Input (Top) =====
    query_box = FancyBboxPatch((6.5, 8), 3, 1, boxstyle='round,pad=0.1',
                               facecolor=colors['input'], edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(query_box)
    ax.text(8, 8.5, 'Natural Language Query', fontsize=11, ha='center', fontweight='bold')
    ax.text(8, 8.1, '"Where is the red couch?"', fontsize=9, ha='center', style='italic')
    
    # ===== L1: Semantic Filter =====
    l1_box = FancyBboxPatch((4, 5.5), 3, 1.5, boxstyle='round,pad=0.1',
                            facecolor=colors['l1'], edgecolor='#2980b9', linewidth=2, alpha=0.9)
    ax.add_patch(l1_box)
    ax.text(5.5, 6.5, 'Level 1: Semantic Filter', fontsize=11, ha='center', 
           fontweight='bold', color='white')
    ax.text(5.5, 5.9, 'CLIP Text Encoding', fontsize=9, ha='center', color='white')
    ax.text(5.5, 5.5, 'FAISS Similarity Search', fontsize=9, ha='center', color='white')
    ax.text(5.5, 5.1, 'k=100 candidates | ~5ms', fontsize=8, ha='center', 
           color='white', style='italic')
    
    # Arrow from query to L1
    ax.annotate('', xy=(5.5, 7), xytext=(7.5, 8),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    # Arrow from memory to L1
    ax.annotate('', xy=(4, 6), xytext=(3, 5.5),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    
    # ===== L2: Geometric Clustering =====
    l2_box = FancyBboxPatch((8.5, 5.5), 3, 1.5, boxstyle='round,pad=0.1',
                            facecolor=colors['l2'], edgecolor='#27ae60', linewidth=2, alpha=0.9)
    ax.add_patch(l2_box)
    ax.text(10, 6.5, 'Level 2: Geometric Cluster', fontsize=11, ha='center',
           fontweight='bold', color='white')
    ax.text(10, 5.9, 'Depth Projection to 3D', fontsize=9, ha='center', color='white')
    ax.text(10, 5.5, 'DBSCAN Clustering (ε=1m)', fontsize=9, ha='center', color='white')
    ax.text(10, 5.1, '~10 clusters | ~15ms', fontsize=8, ha='center',
           color='white', style='italic')
    
    # Arrow L1 to L2
    ax.annotate('', xy=(8.5, 6.25), xytext=(7, 6.25),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(7.75, 6.6, '100\ncandidates', fontsize=8, ha='center')
    
    # ===== L3: Visual Verification =====
    l3_box = FancyBboxPatch((13, 5.5), 2.5, 1.5, boxstyle='round,pad=0.1',
                            facecolor=colors['l3'], edgecolor='#8e44ad', linewidth=2, alpha=0.9)
    ax.add_patch(l3_box)
    ax.text(14.25, 6.5, 'Level 3: Verify', fontsize=11, ha='center',
           fontweight='bold', color='white')
    ax.text(14.25, 5.9, 'OWL-ViT Detection', fontsize=9, ha='center', color='white')
    ax.text(14.25, 5.5, 'Zero-shot Matching', fontsize=9, ha='center', color='white')
    ax.text(14.25, 5.1, '5 clusters | ~400ms', fontsize=8, ha='center',
           color='white', style='italic')
    
    # Arrow L2 to L3
    ax.annotate('', xy=(13, 6.25), xytext=(11.5, 6.25),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text(12.25, 6.6, '10\nclusters', fontsize=8, ha='center')
    
    # ===== Output =====
    out_box = FancyBboxPatch((12, 3), 3.5, 1.5, boxstyle='round,pad=0.1',
                             facecolor=colors['output'], edgecolor='#d68910', linewidth=2, alpha=0.9)
    ax.add_patch(out_box)
    ax.text(13.75, 4, 'Verified 3D Location', fontsize=11, ha='center',
           fontweight='bold', color='white')
    ax.text(13.75, 3.5, '(x, y, z) + confidence', fontsize=9, ha='center', color='white')
    
    # Arrow L3 to output
    ax.annotate('', xy=(13.75, 4.5), xytext=(14.25, 5.5),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    
    # ===== Key Innovation Box =====
    innovation_box = FancyBboxPatch((4, 1), 8, 1.8, boxstyle='round,pad=0.2',
                                    facecolor='#fef9e7', edgecolor='#f4d03f', linewidth=2)
    ax.add_patch(innovation_box)
    ax.text(8, 2.4, ' Key Innovation: Just-in-Time Processing', fontsize=11, 
           ha='center', fontweight='bold')
    ax.text(8, 1.8, 'No preprocessing during exploration -> Lazy evaluation at query time',
           fontsize=10, ha='center')
    ax.text(8, 1.3, 'L2 clustering acts as spatial denoiser: 21% localization improvement',
           fontsize=10, ha='center', style='italic')
    
    # ===== Latency Summary =====
    latency_text = 'Total Latency: ~420ms (16× faster than brute force)'
    ax.text(8, 0.5, latency_text, fontsize=11, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'fig_jit_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_jit_architecture.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: fig_jit_architecture.png/pdf")


def main():
    output_dir = Path(__file__).parent.parent / 'outputs' / 'thesis_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating thesis figures...")
    print("=" * 50)
    
    # Figure 4: Spatial Denoiser
    print("\n1. Generating Spatial Denoiser figure...")
    generate_spatial_denoiser_figure(output_dir)
    
    # Figure: L2 Paradox
    print("\n2. Generating L2 Paradox (Two-Path) figure...")
    generate_l2_paradox_figure(output_dir)
    
    # Figure: Main Architecture
    print("\n3. Generating JIT Architecture figure...")
    generate_jit_architecture_diagram(output_dir)
    
    print("\n" + "=" * 50)
    print(f"All figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
